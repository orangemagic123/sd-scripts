"""
Apply post-hoc Exponential Moving Average (EMA) to a sequence of LoRA checkpoints.

Given a folder (or explicit list) of LoRA checkpoints saved at different epochs/steps
of the same training run, this script computes a running EMA over them:

    ema = checkpoints[0]
    for w in checkpoints[1:]:
        ema = decay * ema + (1 - decay) * w

The result is mathematically equivalent to tracking an EMA during training, but
sampled only at checkpoint boundaries (so a larger `ema_decay` is typically
desirable than for per-step EMA).

Usage examples:

    # Apply EMA to every .safetensors file inside a folder
    python networks/ema_lora.py \
        --model_dir /path/to/A_lora_epochs \
        --ema_decay 0.9 \
        --save_to /path/to/A_lora_ema.safetensors

    # Apply EMA to an explicit, ordered list of checkpoints
    python networks/ema_lora.py \
        --models epoch1.safetensors epoch2.safetensors epoch3.safetensors \
        --ema_decay 0.9 \
        --save_to A_lora_ema.safetensors

    # Also save the EMA snapshot after each checkpoint (useful for sweeping)
    python networks/ema_lora.py \
        --model_dir /path/to/A_lora_epochs \
        --ema_decay 0.9 \
        --save_to /path/to/A_lora_ema.safetensors \
        --save_every_step

    # Only apply EMA from epoch 50 onward (earlier epochs are skipped;
    # the final 'A.safetensors' is always kept)
    python networks/ema_lora.py \
        --model_dir /path/to/A_lora_epochs \
        --start_epoch 50 \
        --ema_decay 0.9 \
        --save_to /path/to/A_lora_ema.safetensors
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# keys whose values should NOT be averaged (they are training-time constants
# such as the LoRA alpha scale). They are copied from the final checkpoint.
NON_EMA_KEY_SUBSTRINGS = (".alpha",)


# sd-scripts writes intermediate checkpoints using these patterns (see
# library/train_util.py):
#   EPOCH_FILE_NAME: "{output_name}-{epoch:06d}"      -> '...-000042'
#   STEP_FILE_NAME:  "{output_name}-step{step:08d}"   -> '...-step00001000'
# We only treat a trailing group that matches these exact shapes as the epoch
# suffix. Requiring the zero-padded digit counts (6 for epochs, 8 for steps)
# makes this robust against an output_name that itself ends in '-<digits>'
# (for example 'v1-2.safetensors') and, importantly, against output_names
# that merely contain digits somewhere in the middle (for example
# 'hazano163A.safetensors', where an earlier implementation would otherwise
# have read '163' as an epoch number).
_EPOCH_SUFFIX_RE = re.compile(r"-(?:step(\d{8,})|(\d{6,}))$")


def _split_epoch_suffix(stem: str):
    """Return (base_name, epoch_or_None) for a checkpoint filename stem.

    ``None`` is returned for stems that do not end with an sd-scripts style
    epoch/step marker, which is how we recognise the final ``{output_name}``
    checkpoint that is written without any numeric suffix.
    """
    m = _EPOCH_SUFFIX_RE.search(stem)
    if not m:
        return stem, None
    epoch = int(m.group(1) if m.group(1) is not None else m.group(2))
    return stem[: m.start()], epoch


def _str_to_dtype(p):
    if p == "float":
        return torch.float
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None


def _natural_sort_key(path: str):
    """Sort file paths naturally (epoch2 < epoch10).

    Only the trailing ``-<epoch>`` / ``-step<step>`` suffix produced by
    sd-scripts is treated as the numeric "epoch key"; digits elsewhere in the
    base name are sorted as part of the base name itself. Files without such a
    trailing suffix (typically the final ``{output_name}.safetensors`` that
    sd-scripts writes at the end of training) are pushed to the end of their
    natural group so that the EMA runs from oldest to newest and the final
    checkpoint dominates the average.
    """
    name = os.path.basename(path).lower()
    stem, ext = os.path.splitext(name)
    base, epoch = _split_epoch_suffix(stem)
    # Natural-sort the base name so e.g. 'runA' < 'runB' and 'v2' < 'v10'
    # within the base (independent of epoch suffix handling).
    base_parts = re.split(r"(\d+)", base)
    base_tokens = tuple(int(p) if p.isdigit() else p for p in base_parts)
    has_epoch = epoch is not None
    # Secondary key: 0 for numbered epoch files, 1 for unnumbered files,
    # so the final unnumbered checkpoint sorts AFTER its numbered siblings.
    return (base_tokens, 0 if has_epoch else 1, epoch if has_epoch else 0, ext)


def _is_non_ema_key(key: str) -> bool:
    return any(s in key for s in NON_EMA_KEY_SUBSTRINGS)


def _extract_epoch(path: str):
    """Return the trailing epoch number in a checkpoint filename, or None.

    sd-scripts saves intermediate LoRA checkpoints with the pattern
    ``{output_name}-{epoch:06d}.safetensors`` (and sometimes step-based
    ``-step{n}``). Only a suffix that matches one of those exact shapes is
    recognised, so digits embedded in ``output_name`` itself (e.g.
    ``hazano163A.safetensors``) are not mistaken for an epoch number.
    Files whose stem has no such trailing suffix — typically the final
    ``{output_name}.safetensors`` — return ``None`` and are treated as
    "after all numbered epochs" by callers.
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    _, epoch = _split_epoch_suffix(stem)
    return epoch


def _detect_source_dtype(sd: dict):
    """Return the dtype of a representative weight tensor in the state dict.

    Prefers a non-scalar tensor (so we don't pick the `.alpha` scalars which
    may be stored as float32 even when the main weights are fp16/bf16).
    """
    fallback = None
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            if v.numel() > 1:
                return v.dtype
            if fallback is None:
                fallback = v.dtype
    return fallback


def _load_state_dict(
    file_name: str, dtype, device: "torch.device" = torch.device("cpu")
) -> Tuple[dict, dict, "torch.dtype"]:
    """Load a checkpoint directly onto ``device`` in the requested ``dtype``.

    For safetensors files we stream each tensor straight to ``device`` via
    ``safe_open``; this avoids a full CPU copy of the file and lets the dtype
    upcast (e.g. bf16 -> float32) run on the GPU instead of saturating the
    CPU, which was the dominant cost on network-mounted checkpoint stores.
    """
    if os.path.splitext(file_name)[1] == ".safetensors":
        # safe_open understands 'cpu', 'cuda', 'cuda:0' etc. Passing the device
        # here causes get_tensor() to materialize straight into that device's
        # memory — no intermediate CPU tensor, no per-tensor .to(device) hop.
        device_str = str(device)
        sd: dict = {}
        source_dtype: Optional["torch.dtype"] = None
        fallback_dtype: Optional["torch.dtype"] = None
        with safe_open(file_name, framework="pt", device=device_str) as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                t = f.get_tensor(key)
                # Track the source dtype from a representative (non-scalar)
                # tensor; .alpha scalars are often fp32 even when weights are
                # bf16/fp16, and we want to preserve the weight dtype on save.
                if source_dtype is None:
                    if t.numel() > 1:
                        source_dtype = t.dtype
                    elif fallback_dtype is None:
                        fallback_dtype = t.dtype
                if dtype is not None and t.dtype != dtype:
                    t = t.to(dtype)
                sd[key] = t
        if source_dtype is None:
            source_dtype = fallback_dtype
        metadata = dict(metadata)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}
        source_dtype = _detect_source_dtype(sd)
        for key in list(sd.keys()):
            if isinstance(sd[key], torch.Tensor):
                t = sd[key]
                # Combine dtype cast + device move into a single op so the
                # runtime can fuse them (e.g. bf16 CPU -> fp32 CUDA in one hop).
                if dtype is not None:
                    t = t.to(device=device, dtype=dtype)
                elif t.device != device:
                    t = t.to(device)
                sd[key] = t

    return sd, metadata, source_dtype


def _save_state_dict(file_name: str, state_dict: dict, dtype, metadata: dict):
    # safetensors.save_file requires CPU tensors; torch.save accepts any device
    # but we normalize to CPU anyway so the saved file is portable.
    for key in list(state_dict.keys()):
        if isinstance(state_dict[key], torch.Tensor):
            t = state_dict[key]
            if dtype is not None:
                t = t.to(dtype)
            if t.device.type != "cpu":
                t = t.detach().cpu()
            state_dict[key] = t

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata if metadata else None)
    else:
        torch.save(state_dict, file_name)


def _collect_models(args) -> List[str]:
    if args.models:
        models = list(args.models)
    else:
        if args.model_dir is None:
            raise ValueError("Either --models or --model_dir must be specified")
        if not os.path.isdir(args.model_dir):
            raise ValueError(f"--model_dir is not a directory: {args.model_dir}")

        exts = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
        entries = []
        for fname in os.listdir(args.model_dir):
            full = os.path.join(args.model_dir, fname)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            if args.pattern is not None and not re.search(args.pattern, fname):
                continue
            entries.append(full)
        models = sorted(entries, key=_natural_sort_key)

    if args.start_epoch is not None:
        filtered = []
        skipped = []
        for path in models:
            epoch = _extract_epoch(path)
            # keep unnumbered files (e.g. the final "A.safetensors") and any
            # file whose trailing epoch number is >= start_epoch
            if epoch is None or epoch >= args.start_epoch:
                filtered.append(path)
            else:
                skipped.append(path)
        if skipped:
            logger.info(
                f"--start_epoch={args.start_epoch}: skipping {len(skipped)} checkpoint(s) below threshold"
            )
            for p in skipped:
                logger.info(f"  skip: {os.path.basename(p)} (epoch={_extract_epoch(p)})")
        models = filtered

    if len(models) == 0:
        raise ValueError("No checkpoints found to apply EMA on")
    return models


def apply_ema(
    models: List[str],
    ema_decay: float,
    compute_dtype,
    save_dtype,
    save_to: str,
    save_every_step: bool = False,
    keep_last_metadata: bool = True,
    device: "torch.device" = torch.device("cpu"),
):
    assert 0.0 <= ema_decay < 1.0, f"ema_decay must be in [0, 1), got {ema_decay}"
    assert len(models) >= 1, "need at least one checkpoint"

    logger.info(f"applying EMA to {len(models)} checkpoints with decay={ema_decay} on device={device}")
    for i, m in enumerate(models):
        logger.info(f"  [{i}] {m}")

    # Single-worker prefetch: while we're updating the running EMA with
    # checkpoint N, the worker is already pulling checkpoint N+1 off disk
    # (and onto `device`). For runs where the per-file load dominates — which
    # is the usual case on network-mounted checkpoint dirs — this hides almost
    # all of the compute behind the I/O wait.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ema-prefetch")
    try:
        # initialize EMA from the first checkpoint
        logger.info(f"loading initial checkpoint: {models[0]}")
        t_init = time.perf_counter()
        ema_sd, first_metadata, source_dtype = _load_state_dict(models[0], compute_dtype, device)
        logger.info(f"  loaded initial checkpoint in {time.perf_counter() - t_init:.2f}s")
        last_metadata = first_metadata

        # if the caller did not request a specific save dtype, preserve the source
        # dtype so the output file size matches the input files (e.g. fp16 in -> fp16 out)
        if save_dtype is None:
            save_dtype = source_dtype
            logger.info(f"save_dtype not specified; using source dtype: {save_dtype}")

        if save_every_step:
            step_path = _build_step_path(save_to, 0, models[0])
            logger.info(f"saving EMA snapshot after step 0 to: {step_path}")
            snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ema_sd.items()}
            _save_state_dict(step_path, snapshot, save_dtype, dict(first_metadata) if first_metadata else {})

        # Kick off the first prefetch so the worker overlaps with setup above
        # (and with any subsequent compute).
        pending: Optional[Future] = None
        if len(models) > 1:
            pending = executor.submit(_load_state_dict, models[1], compute_dtype, device)

        for step, model_path in enumerate(models[1:], start=1):
            logger.info(f"loading checkpoint [{step}]: {model_path}")
            t_load = time.perf_counter()
            assert pending is not None
            cur_sd, cur_metadata, _ = pending.result()
            load_wait = time.perf_counter() - t_load
            # Queue the next file before we start computing on the current one
            # so its disk read runs in parallel with our EMA update.
            if step + 1 < len(models):
                pending = executor.submit(_load_state_dict, models[step + 1], compute_dtype, device)
            else:
                pending = None

            last_metadata = cur_metadata if keep_last_metadata else last_metadata

            ema_keys = set(ema_sd.keys())
            cur_keys = set(cur_sd.keys())
            if ema_keys != cur_keys:
                only_ema = ema_keys - cur_keys
                only_cur = cur_keys - ema_keys
                if only_ema:
                    logger.warning(
                        f"keys present in previous EMA but missing in {os.path.basename(model_path)}: "
                        f"{sorted(list(only_ema))[:5]}{'...' if len(only_ema) > 5 else ''}"
                    )
                if only_cur:
                    logger.warning(
                        f"new keys in {os.path.basename(model_path)} not present in previous EMA: "
                        f"{sorted(list(only_cur))[:5]}{'...' if len(only_cur) > 5 else ''}"
                    )

            t_compute = time.perf_counter()
            for key in cur_keys:
                cur_val = cur_sd[key]
                if not isinstance(cur_val, torch.Tensor):
                    ema_sd[key] = cur_val
                    continue

                if _is_non_ema_key(key):
                    # alpha etc. — just copy the latest value
                    ema_sd[key] = cur_val
                    continue

                if key not in ema_sd:
                    # new tensor appearing mid-sequence; initialize from this checkpoint
                    ema_sd[key] = cur_val.clone()
                    continue

                prev = ema_sd[key]
                if prev.shape != cur_val.shape:
                    logger.warning(
                        f"shape mismatch for key {key}: {tuple(prev.shape)} vs {tuple(cur_val.shape)}, "
                        f"replacing with current checkpoint"
                    )
                    ema_sd[key] = cur_val.clone()
                    continue

                # In-place EMA update: `prev.mul_(d).add_(cur, alpha=1-d)` avoids
                # allocating a fresh output tensor for every key, which for LoRA
                # checkpoints with thousands of small tensors otherwise means
                # thousands of extra allocator/kernel-launch round trips.
                prev.mul_(ema_decay).add_(cur_val, alpha=1.0 - ema_decay)

            compute_elapsed = time.perf_counter() - t_compute
            # Release references to the consumed checkpoint's tensors before
            # the next iteration so peak memory stays at roughly 2x a single
            # checkpoint (ema_sd + one prefetched sd) rather than 3x.
            del cur_sd
            logger.info(
                f"  step {step}: load_wait={load_wait:.2f}s compute={compute_elapsed:.2f}s"
            )

            if save_every_step:
                step_path = _build_step_path(save_to, step, model_path)
                logger.info(f"saving EMA snapshot after step {step} to: {step_path}")
                snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ema_sd.items()}
                _save_state_dict(step_path, snapshot, save_dtype, dict(last_metadata) if last_metadata else {})
    finally:
        executor.shutdown(wait=True)

    metadata = dict(last_metadata) if last_metadata else {}
    metadata["ss_ema_decay"] = str(ema_decay)
    metadata["ss_ema_source_count"] = str(len(models))
    metadata["ss_ema_sources"] = ",".join(os.path.basename(m) for m in models)

    logger.info(f"saving final EMA model to: {save_to}")
    _save_state_dict(save_to, ema_sd, save_dtype, metadata)


def _build_step_path(save_to: str, step: int, source_model: str) -> str:
    base, ext = os.path.splitext(save_to)
    source_stem = os.path.splitext(os.path.basename(source_model))[0]
    return f"{base}-ema-step{step:04d}-{source_stem}{ext}"


def _resolve_device(requested: str) -> "torch.device":
    """Turn a user-supplied device string into a torch.device, falling back
    gracefully to CPU when CUDA was requested but is unavailable."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.info("CUDA not available; using CPU for EMA computation")
        return torch.device("cpu")

    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning(f"requested device '{requested}' but CUDA is not available; falling back to CPU")
        return torch.device("cpu")
    return dev


def run(args):
    compute_dtype = _str_to_dtype(args.precision)
    # NOTE: intentionally do NOT fall back to compute_dtype here. A None
    # save_dtype is interpreted by apply_ema() as "preserve the source
    # checkpoint's dtype" so the output file size matches the input.
    save_dtype = _str_to_dtype(args.save_precision)

    if args.save_to is None:
        raise ValueError("--save_to must be specified")

    save_dir = os.path.dirname(os.path.abspath(args.save_to))
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    models = _collect_models(args)

    device = _resolve_device(args.device)

    t0 = time.time()
    apply_ema(
        models=models,
        ema_decay=args.ema_decay,
        compute_dtype=compute_dtype,
        save_dtype=save_dtype,
        save_to=args.save_to,
        save_every_step=args.save_every_step,
        device=device,
    )
    logger.info(f"done in {time.time() - t0:.1f}s")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Exponential Moving Average (EMA) to a sequence of saved LoRA checkpoints."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="folder containing LoRA epoch checkpoints to be EMA-averaged (sorted by natural order) / "
        "EMA를 적용할 LoRA 에폭 체크포인트가 들어있는 폴더 (파일명 기준 자연 정렬)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="explicit, ordered list of LoRA checkpoints (alternative to --model_dir) / "
        "명시적으로 순서를 지정한 LoRA 체크포인트 목록 (--model_dir 대신 사용)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="optional regex to filter filenames when using --model_dir / "
        "--model_dir 사용 시 파일명 필터로 적용할 정규식 (선택)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="only apply EMA from this epoch onward (inclusive). A checkpoint's "
        "epoch is read from the last digit group in its filename stem (e.g. "
        "'A-000050.safetensors' -> 50). Files with no digits in the stem, such "
        "as the final 'A.safetensors', are always kept. / "
        "지정한 에폭 이상부터 EMA를 적용합니다 (경계값 포함). 파일명 stem의 마지막 "
        "숫자 그룹을 에폭으로 해석합니다 (예: 'A-000050.safetensors' -> 50). 숫자가 "
        "없는 파일(예: 최종 'A.safetensors')은 항상 포함됩니다.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        required=True,
        help="EMA decay in [0, 1). Larger values place more weight on older checkpoints. / "
        "EMA 감쇠 계수 (0 이상 1 미만). 값이 클수록 이전 체크포인트 비중이 커집니다.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="output file (ckpt or safetensors) for the final EMA model / "
        "EMA 적용 결과를 저장할 파일 경로 (ckpt 또는 safetensors)",
    )
    parser.add_argument(
        "--save_every_step",
        action="store_true",
        help="also save the EMA snapshot after each checkpoint step / "
        "각 체크포인트 단계마다 중간 EMA 스냅샷도 저장",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision used during EMA computation (float is recommended) / "
        "EMA 계산 시 정밀도 (float 권장)",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision when saving; if omitted, the dtype of the source "
        "checkpoints is preserved so the output file size matches the input / "
        "저장 시 정밀도. 생략 시 원본 체크포인트의 dtype을 그대로 사용하여 "
        "출력 파일 크기가 원본과 동일하게 유지됩니다",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device used for EMA computation: 'auto' (cuda if available, else cpu), "
        "'cuda', 'cuda:0', 'cpu', etc. Defaults to 'auto'. / "
        "EMA 계산에 사용할 장치: 'auto'(가능하면 cuda, 아니면 cpu), 'cuda', 'cuda:0', "
        "'cpu' 등. 기본값은 'auto'.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    run(args)
