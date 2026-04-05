"""Convert T-LoRA (orthogonal) checkpoints trained with networks/lora_tlora_anima.py
into a ComfyUI-compatible LoRA safetensors file.

Orthogonal T-LoRA stores per-module tensors:
    <lora_name>.q_layer.weight          shape [r, in]     (for conv: in = in_ch*kh*kw)
    <lora_name>.p_layer.weight          shape [out, r]
    <lora_name>.lambda_layer            shape [1, r]
    <lora_name>.base_q.weight           shape [r, in]     (frozen at init)
    <lora_name>.base_p.weight           shape [out, r]    (frozen at init)
    <lora_name>.base_lambda_buf         shape [1, r]      (frozen at init)
    <lora_name>.alpha                   scalar

The effective delta weight applied at inference is:
    delta_W = scale * ( P @ diag(lam) @ Q  -  Pb @ diag(lamb) @ Qb )
where scale = alpha / r.

This is a rank-<=2r update, which we encode losslessly as a standard LoRA of rank 2r:
    lora_down.weight = concat([Q, Qb], dim=0)              shape [2r, in]
    lora_up.weight   = concat([P*lam, -Pb*lamb], dim=1)    shape [out, 2r]
    alpha_new        = 2 * alpha                           (keeps scale = alpha/r)

For conv modules q_layer/p_layer are Linear but q was reshaped to conv kernel at runtime.
Shapes from lora_tlora_anima.py's forward:
    q_weight -> [r, in_channels, kh, kw]
    p_weight -> [out_channels, r, 1, 1]
We do the same reshape so the output uses conv lora_down/lora_up tensors.

After converting to standard (sd-scripts) LoRA keys, we run the same key-renaming
as networks/convert_anima_lora_to_comfy.py to produce ComfyUI keys.
"""

import argparse
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


COMFYUI_DIT_PREFIX = "diffusion_model."
COMFYUI_QWEN3_PREFIX = "text_encoders.qwen3_06b.transformer.model."


def _is_conv_module(q_weight: torch.Tensor, p_weight: torch.Tensor, orig_out_feat_hint=None):
    # Heuristic: for Linear q is [r, in_features] (no kernel dims); for conv original
    # module the stored q is still [r, in_ch*kh*kw]. We can't tell from shape alone
    # whether this was a conv. Default to treating it as Linear. If the user knows
    # a specific module is conv, they can pass --conv-names. For Anima DiT there are
    # no conv modules inside the targeted replace list, so Linear is correct.
    return False


def _tlora_module_to_standard(q, p, lam, bq, bp, blam, dtype):
    """Convert orthogonal T-LoRA tensors to standard lora_down/lora_up of rank 2r."""
    # shapes: q, bq: [r, in]; p, bp: [out, r]; lam, blam: [1, r]
    r = q.shape[0]
    assert p.shape[1] == r and lam.shape[1] == r
    assert bq.shape == q.shape and bp.shape == p.shape and blam.shape == lam.shape

    # up_pos = P * lam  (broadcast [1,r] over rows) -> [out, r]
    up_pos = p * lam
    up_neg = -(bp * blam)

    lora_down = torch.cat([q, bq], dim=0).contiguous()          # [2r, in]
    lora_up = torch.cat([up_pos, up_neg], dim=1).contiguous()   # [out, 2r]

    return lora_down.to(dtype), lora_up.to(dtype)


def _rename_to_comfy(sd_scripts_key: str) -> str:
    """Port of the forward-direction logic in convert_anima_lora_to_comfy.py."""
    k = sd_scripts_key
    is_dit_lora = k.startswith("lora_unet_")
    # strip "lora_unet_" or "lora_te_"
    module_and_weight_name = "_".join(k.split("_")[2:])
    module_name, weight_name = module_and_weight_name.split(".", 1)

    if weight_name.startswith("lora_up"):
        weight_name = weight_name.replace("lora_up", "lora_B")
    elif weight_name.startswith("lora_down"):
        weight_name = weight_name.replace("lora_down", "lora_A")
    # else: alpha, keep as-is

    original_module_name = module_name.replace("_", ".")

    # Restore compound names (identical list to convert_anima_lora_to_comfy.py)
    replacements_dit = [
        ("llm.adapter", "llm_adapter"),
        (".linear.", ".linear_"),
        ("t.embedding.norm", "t_embedding_norm"),
        ("x.embedder", "x_embedder"),
        ("adaln.modulation.cross_attn", "adaln_modulation_cross_attn"),
        ("adaln.modulation.mlp", "adaln_modulation_mlp"),
        ("cross.attn", "cross_attn"),
        ("k.proj", "k_proj"),
        ("k.norm", "k_norm"),
        ("q.proj", "q_proj"),
        ("q.norm", "q_norm"),
        ("v.proj", "v_proj"),
        ("o.proj", "o_proj"),
        ("output.proj", "output_proj"),
        ("self.attn", "self_attn"),
        ("final.layer", "final_layer"),
        ("adaln.modulation", "adaln_modulation"),
        ("norm.cross.attn", "norm_cross_attn"),
        ("norm.mlp", "norm_mlp"),
        ("norm.self.attn", "norm_self_attn"),
        ("out.proj", "out_proj"),
        # Qwen3
        ("embed.tokens", "embed_tokens"),
        ("input.layernorm", "input_layernorm"),
        ("down.proj", "down_proj"),
        ("gate.proj", "gate_proj"),
        ("up.proj", "up_proj"),
        ("post.attention.layernorm", "post_attention_layernorm"),
    ]
    for a, b in replacements_dit:
        original_module_name = original_module_name.replace(a, b)

    new_prefix = COMFYUI_DIT_PREFIX if is_dit_lora else COMFYUI_QWEN3_PREFIX
    return f"{new_prefix}{original_module_name}.{weight_name}"


def main(args):
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    # Group tensors by lora_name (the part before the first ".")
    groups = defaultdict(dict)
    for k, v in state_dict.items():
        if "." not in k:
            continue
        lora_name, suffix = k.split(".", 1)
        groups[lora_name][suffix] = v

    out_sd = {}
    converted_ortho = 0
    passthrough_standard = 0
    skipped = 0

    # Pick output dtype
    if args.dtype == "keep":
        target_dtype = None
    else:
        target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    for lora_name, tensors in groups.items():
        keys_set = set(tensors.keys())
        is_orthogonal = {
            "q_layer.weight", "p_layer.weight", "lambda_layer",
            "base_q.weight", "base_p.weight", "base_lambda_buf",
        }.issubset(keys_set)

        if is_orthogonal:
            q = tensors["q_layer.weight"].float()
            p = tensors["p_layer.weight"].float()
            lam = tensors["lambda_layer"].float()
            bq = tensors["base_q.weight"].float()
            bp = tensors["base_p.weight"].float()
            blam = tensors["base_lambda_buf"].float()

            alpha = tensors.get("alpha")
            if alpha is None:
                alpha_val = float(q.shape[0])
            else:
                alpha_val = float(alpha.item())

            r = q.shape[0]
            new_alpha_val = 2.0 * alpha_val  # keep effective scale = alpha_orig / r

            dt = target_dtype if target_dtype is not None else q.dtype if tensors["q_layer.weight"].dtype.is_floating_point else torch.float32
            # Prefer the original saved dtype when keeping
            if target_dtype is None:
                dt = tensors["q_layer.weight"].dtype

            lora_down, lora_up = _tlora_module_to_standard(q, p, lam, bq, bp, blam, dt)

            out_sd[f"{lora_name}.lora_down.weight"] = lora_down
            out_sd[f"{lora_name}.lora_up.weight"] = lora_up
            out_sd[f"{lora_name}.alpha"] = torch.tensor(new_alpha_val, dtype=torch.float32)
            converted_ortho += 1

            # Warn about any unexpected leftover keys in this group
            extra = keys_set - {
                "q_layer.weight", "p_layer.weight", "lambda_layer",
                "base_q.weight", "base_p.weight", "base_lambda_buf",
                "alpha", "base_lambda",
            }
            if extra:
                logger.warning(f"{lora_name}: ignoring unexpected keys {sorted(extra)}")
        elif "lora_down.weight" in keys_set and "lora_up.weight" in keys_set:
            # Standard (kaiming) T-LoRA module – already uses lora_down/lora_up.
            for suffix, v in tensors.items():
                if target_dtype is not None and v.dtype.is_floating_point:
                    v = v.to(target_dtype)
                out_sd[f"{lora_name}.{suffix}"] = v
            passthrough_standard += 1
        else:
            logger.warning(f"Skipping {lora_name}: unrecognized tensor set {sorted(keys_set)}")
            skipped += 1

    logger.info(
        f"Modules: {converted_ortho} orthogonal T-LoRA -> standard (rank doubled), "
        f"{passthrough_standard} already standard, {skipped} skipped"
    )

    # Rename keys to ComfyUI convention (unless --sd-scripts-only)
    if not args.sd_scripts_only:
        renamed = {}
        for k, v in out_sd.items():
            renamed[_rename_to_comfy(k)] = v
        out_sd = renamed

    # Clean metadata
    if metadata is None:
        metadata = {}
    # Remove T-LoRA specific metadata that could confuse ComfyUI loaders; keep others
    metadata.pop("ss_tlora_init", None)
    metadata.pop("ss_tlora_sig_type", None)
    metadata.pop("ss_tlora_min_rank", None)
    metadata.pop("ss_tlora_alpha_rank_scale", None)
    metadata.pop("ss_tlora_max_timestep", None)

    logger.info("Calculating hashes and creating metadata...")
    try:
        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(out_sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
    except Exception as e:
        logger.warning(f"Hash calculation failed: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst_path)) or ".", exist_ok=True)
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(out_sd, args.dst_path, metadata=metadata if metadata else None)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an orthogonal T-LoRA (lora_tlora_anima.py) checkpoint "
                    "to a ComfyUI-compatible standard LoRA safetensors file."
    )
    parser.add_argument("src_path", type=str, help="source T-LoRA safetensors path")
    parser.add_argument("dst_path", type=str, help="destination safetensors path")
    parser.add_argument(
        "--dtype", type=str, default="keep",
        choices=["keep", "fp16", "bf16", "fp32"],
        help="output tensor dtype (default: keep)",
    )
    parser.add_argument(
        "--sd-scripts-only", action="store_true",
        help="only convert T-LoRA tensors to standard lora_down/lora_up; keep sd-scripts "
             "(lora_unet_*) key names instead of ComfyUI naming.",
    )
    args = parser.parse_args()
    main(args)
