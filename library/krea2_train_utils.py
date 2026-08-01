"""Training arguments, schedules, VAE loading, and sampling for Krea 2."""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import time
from typing import Optional

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from library import sampling
from library.device_utils import clean_memory_on_device, synchronize_device
from library.krea2_models import SingleStreamDiT, prepare, roundup, timesteps

logger = logging.getLogger(__name__)


def add_krea2_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B-Instruct safetensors file (official or ComfyUI key layout)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Hugging Face repository or local directory containing the Qwen3-VL tokenizer",
    )
    parser.add_argument("--text_encoder_max_length", type=int, default=512)
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "krea2_shift"],
        default="krea2_shift",
        help="Flow timestep distribution. krea2_shift follows Krea 2's resolution-aware inference schedule.",
    )
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=2.5,
        help="Fixed shift used by --timestep_sampling shift (ignored by krea2_shift)",
    )
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],
        default=None,
        help="Attention backend override. torch/sdpa uses PyTorch scaled dot-product attention.",
    )
    parser.add_argument("--split_attn", action="store_true", help="Process variable-length attention one sample at a time")
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="Dynamically quantize main-block Linear weights; use together with --fp8_base",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument(
        "--qwen_image_vae_2d",
        action="store_true",
        help="Use the image-only 2D Qwen-Image VAE implementation",
    )

    # Per-block torch.compile. This is separate from Accelerate's legacy
    # --torch_compile option and mirrors other DiT integrations in sd-scripts.
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--compile_dynamic", choices=["true", "false", "auto"], default=None)
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--compile_cache_size_limit", type=int, default=None)
    parser.add_argument("--cuda_allow_tf32", action="store_true")
    parser.add_argument("--cuda_cudnn_benchmark", action="store_true")


def load_qwen_image_vae(args, device: str | torch.device = "cpu", disable_mmap: bool = True):
    if args.qwen_image_vae_2d:
        from library import qwen_image_autoencoder_kl_2d

        return qwen_image_autoencoder_kl_2d.load_vae(
            args.vae,
            device=device,
            disable_mmap=disable_mmap,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )

    from library import qwen_image_autoencoder_kl

    return qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device=device,
        disable_mmap=disable_mmap,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )


def get_krea2_mu(
    latent_height: int,
    latent_width: int,
    patch_size: int = 2,
    min_resolution: int = 256,
    max_resolution: int = 1280,
    vae_scale_factor: int = 8,
    y1: float = 0.5,
    y2: float = 1.15,
) -> float:
    token_count = (latent_height // patch_size) * (latent_width // patch_size)
    x1 = (min_resolution // (vae_scale_factor * patch_size)) ** 2
    x2 = (max_resolution // (vae_scale_factor * patch_size)) ** 2
    slope = (y2 - y1) / (x2 - x1)
    return slope * token_count + (y1 - slope * x1)


def get_krea2_shift(latent_height: int, latent_width: int, patch_size: int = 2) -> float:
    return math.exp(get_krea2_mu(latent_height, latent_width, patch_size))


def _should_sample(args: argparse.Namespace, epoch, steps: int) -> bool:
    if args.sample_prompts is None:
        return False
    if steps == 0:
        return bool(args.sample_at_first)
    if args.sample_every_n_epochs is not None:
        return epoch is not None and epoch % args.sample_every_n_epochs == 0
    if args.sample_every_n_steps is not None:
        return epoch is None and steps % args.sample_every_n_steps == 0
    return False


@torch.no_grad()
def do_sample(
    model: SingleStreamDiT,
    vae,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embed: Optional[torch.Tensor],
    negative_mask: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    explicit_shift: Optional[float] = None,
) -> Image.Image:
    patch = model.config.patch
    scale_factor = 8
    alignment = scale_factor * patch
    width = roundup(width, alignment, "width")
    height = roundup(height, alignment, "height")
    latent_height, latent_width = height // scale_factor, width // scale_factor

    prompt_embed = prompt_embed.unsqueeze(0).to(device=device, dtype=dtype)
    prompt_mask = prompt_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
    use_cfg = cfg_scale > 1.0 and negative_embed is not None
    if use_cfg:
        negative_embed = negative_embed.unsqueeze(0).to(device=device, dtype=dtype)
        negative_mask = negative_mask.unsqueeze(0).to(device=device, dtype=torch.bool)

    noise = torch.randn(
        (1, model.config.channels, latent_height, latent_width),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=dtype)
    img, pos, mask = prepare(noise, prompt_embed.shape[1], patch, prompt_mask)
    if use_cfg:
        _, negative_pos, negative_full_mask = prepare(noise, negative_embed.shape[1], patch, negative_mask)

    x1 = (256 // alignment) ** 2
    x2 = (1280 // alignment) ** 2
    mu = math.log(explicit_shift) if explicit_shift is not None else None
    schedule = timesteps(img.shape[1], steps, x1, x2, mu=mu)

    model.switch_block_swap_for_inference()
    model.prepare_block_swap_before_forward()
    try:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type != "cpu"):
            for current, previous in tqdm(
                zip(schedule[:-1], schedule[1:]), total=len(schedule) - 1, desc="Krea 2 sampling"
            ):
                t = torch.full((1,), current, dtype=img.dtype, device=device)
                cond = model(img=img, context=prompt_embed, t=t, pos=pos, mask=mask)
                if use_cfg:
                    uncond = model(
                        img=img,
                        context=negative_embed,
                        t=t,
                        pos=negative_pos,
                        mask=negative_full_mask,
                    )
                    velocity = uncond + cfg_scale * (cond - uncond)
                else:
                    velocity = cond
                img = img + (previous - current) * velocity
    finally:
        model.switch_block_swap_for_training()

    latent = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=latent_height // patch,
        w=latent_width // patch,
    )
    original_vae_device = vae.device
    vae.to(device)
    pixels = vae.decode_to_pixels(latent.to(vae.dtype))
    vae.to(original_vae_device)
    clean_memory_on_device(device)

    pixels = ((pixels.float().clamp(-1, 1) + 1) / 2)[0]
    pixels = (255.0 * rearrange(pixels, "c h w -> h w c")).byte().cpu().numpy()
    return Image.fromarray(pixels)


def sample_images(
    accelerator,
    args: argparse.Namespace,
    epoch,
    steps: int,
    model,
    vae,
    sample_prompt_outputs: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]],
):
    if not _should_sample(args, epoch, steps):
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not sample_prompt_outputs:
            logger.warning("Krea 2 sample prompt outputs were not cached; skipping samples")
        else:
            model = accelerator.unwrap_model(model)
            prompts = sampling.load_prompts(args.sample_prompts)
            save_dir = os.path.join(args.output_dir, "sample")
            os.makedirs(save_dir, exist_ok=True)
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            try:
                for prompt_dict in prompts:
                    prompt = prompt_dict.get("prompt", "")
                    negative_prompt = prompt_dict.get("negative_prompt", "")
                    width = prompt_dict.get("width", 1024)
                    height = prompt_dict.get("height", 1024)
                    sample_steps = prompt_dict.get("sample_steps", 28)
                    cfg_scale = prompt_dict.get("scale", 5.5)
                    seed = prompt_dict.get("seed", args.seed)
                    explicit_shift = prompt_dict.get("flow_shift")
                    explicit_shift = float(explicit_shift) if explicit_shift is not None else None

                    positive = sample_prompt_outputs[prompt]
                    negative = sample_prompt_outputs.get(negative_prompt) if cfg_scale > 1 else None
                    image = do_sample(
                        model,
                        vae,
                        positive[0],
                        positive[1],
                        None if negative is None else negative[0],
                        None if negative is None else negative[1],
                        device=accelerator.device,
                        dtype=torch.bfloat16,
                        width=width,
                        height=height,
                        steps=sample_steps,
                        cfg_scale=cfg_scale,
                        seed=seed,
                        explicit_shift=explicit_shift,
                    )
                    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                    enum = prompt_dict.get("enum", 0)
                    prefix = "" if args.output_name is None else args.output_name + "_"
                    image.save(os.path.join(save_dir, f"{prefix}{suffix}_{enum:02d}_{timestamp}_{seed}.png"))
            finally:
                torch.set_rng_state(rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state)
                gc.collect()
                synchronize_device(accelerator.device)
                clean_memory_on_device(accelerator.device)
    accelerator.wait_for_everyone()
