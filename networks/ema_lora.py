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
from typing import List, Tuple

import torch
from safetensors.torch import load_file, save_file

from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# keys whose values should NOT be averaged (they are training-time constants
# such as the LoRA alpha scale). They are copied from the final checkpoint.
NON_EMA_KEY_SUBSTRINGS = (".alpha",)


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

    Files whose stem contains no digit group are pushed to the end, so that a
    final checkpoint like ``A.safetensors`` sorts after all per-epoch files like
    ``A-000002.safetensors`` or ``A_000002.safetensors`` regardless of the
    separator used. sd-scripts writes the final LoRA without an epoch suffix,
    and this matches that convention.
    """
    name = os.path.basename(path).lower()
    stem, ext = os.path.splitext(name)
    parts = re.split(r"(\d+)", stem)
    has_number = any(p.isdigit() for p in parts)
    tokens = [int(p) if p.isdigit() else p for p in parts]
    # primary key: 0 for numbered files, 1 for unnumbered (unnumbered last)
    return (0 if has_number else 1, tokens, ext)


def _is_non_ema_key(key: str) -> bool:
    return any(s in key for s in NON_EMA_KEY_SUBSTRINGS)


def _extract_epoch(path: str):
    """Return the trailing epoch number in a checkpoint filename, or None.

    sd-scripts saves intermediate LoRA checkpoints with the pattern
    ``{output_name}-{epoch:06d}.safetensors`` (and sometimes step-based
    ``-step{n}``). We use the last digit group in the stem as the epoch
    number. Files whose stem contains no digits — for example the final
    ``A.safetensors`` — return ``None`` and are treated as "after all
    numbered epochs" by callers.
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def _load_state_dict(file_name: str, dtype) -> Tuple[dict, dict]:
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    if dtype is not None:
        for key in list(sd.keys()):
            if isinstance(sd[key], torch.Tensor):
                sd[key] = sd[key].to(dtype)

    return sd, metadata


def _save_state_dict(file_name: str, state_dict: dict, dtype, metadata: dict):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(dtype)

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
):
    assert 0.0 <= ema_decay < 1.0, f"ema_decay must be in [0, 1), got {ema_decay}"
    assert len(models) >= 1, "need at least one checkpoint"

    logger.info(f"applying EMA to {len(models)} checkpoints with decay={ema_decay}")
    for i, m in enumerate(models):
        logger.info(f"  [{i}] {m}")

    # initialize EMA from the first checkpoint
    logger.info(f"loading initial checkpoint: {models[0]}")
    ema_sd, first_metadata = _load_state_dict(models[0], compute_dtype)
    last_metadata = first_metadata

    if save_every_step:
        step_path = _build_step_path(save_to, 0, models[0])
        logger.info(f"saving EMA snapshot after step 0 to: {step_path}")
        snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ema_sd.items()}
        _save_state_dict(step_path, snapshot, save_dtype, dict(first_metadata) if first_metadata else {})

    for step, model_path in enumerate(models[1:], start=1):
        logger.info(f"loading checkpoint [{step}]: {model_path}")
        cur_sd, cur_metadata = _load_state_dict(model_path, compute_dtype)
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

            ema_sd[key] = prev * ema_decay + cur_val * (1.0 - ema_decay)

        if save_every_step:
            step_path = _build_step_path(save_to, step, model_path)
            logger.info(f"saving EMA snapshot after step {step} to: {step_path}")
            snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ema_sd.items()}
            _save_state_dict(step_path, snapshot, save_dtype, dict(last_metadata) if last_metadata else {})

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


def run(args):
    compute_dtype = _str_to_dtype(args.precision)
    save_dtype = _str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = compute_dtype

    if args.save_to is None:
        raise ValueError("--save_to must be specified")

    save_dir = os.path.dirname(os.path.abspath(args.save_to))
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    models = _collect_models(args)

    t0 = time.time()
    apply_ema(
        models=models,
        ema_decay=args.ema_decay,
        compute_dtype=compute_dtype,
        save_dtype=save_dtype,
        save_to=args.save_to,
        save_every_step=args.save_every_step,
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
        help="precision when saving (defaults to --precision) / "
        "저장 시 정밀도 (미지정 시 --precision과 동일)",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    run(args)
