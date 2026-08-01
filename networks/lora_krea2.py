"""LoRA network for Krea 2's single-stream MMDiT.

Krea 2 keeps its normalization and modulation weights as raw parameters.  The
trainable affine projections are ordinary ``nn.Linear`` modules, so targeting
the root ``SingleStreamDiT`` module gives a stable, checkpoint-compatible LoRA
name for every projection without wrapping the sensitive raw parameters.
"""

import ast
import os
from typing import Optional

import torch

from library.utils import setup_logging
from networks import lora_flux
from networks.network_base import AdditionalNetwork, ArchConfig, _parse_kv_pairs

setup_logging()

KREA2_ARCH_CONFIG = ArchConfig(
    unet_target_modules=["SingleStreamDiT"],
    te_target_modules=[],
    unet_prefix="lora_unet",
    te_prefixes=["lora_te"],
)


def _parse_patterns(value) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, list) else [parsed]


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _set_loraplus_ratios(network: AdditionalNetwork, kwargs) -> None:
    ratios = [
        kwargs.get("loraplus_lr_ratio"),
        kwargs.get("loraplus_unet_lr_ratio"),
        kwargs.get("loraplus_text_encoder_lr_ratio"),
    ]
    ratios = [float(value) if value is not None else None for value in ratios]
    if any(value is not None for value in ratios):
        network.set_loraplus_lr_ratio(*ratios)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a training-time Krea 2 LoRA network."""

    network_dim = 4 if network_dim is None else network_dim
    network_alpha = 1.0 if network_alpha is None else network_alpha

    rank_dropout = kwargs.get("rank_dropout")
    module_dropout = kwargs.get("module_dropout")
    rank_dropout = float(rank_dropout) if rank_dropout is not None else None
    module_dropout = float(module_dropout) if module_dropout is not None else None

    reg_dims = kwargs.get("network_reg_dims")
    reg_lrs = kwargs.get("network_reg_lrs")
    reg_dims = _parse_kv_pairs(reg_dims, is_int=True) if reg_dims is not None else None
    reg_lrs = _parse_kv_pairs(reg_lrs, is_int=False) if reg_lrs is not None else None

    network = AdditionalNetwork(
        text_encoders=[],  # Qwen3-VL outputs are cached and its weights are frozen.
        unet=unet,
        arch_config=KREA2_ARCH_CONFIG,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=float(network_alpha),
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=lora_flux.LoRAModule,
        exclude_patterns=_parse_patterns(kwargs.get("exclude_patterns")),
        include_patterns=_parse_patterns(kwargs.get("include_patterns")),
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=_parse_bool(kwargs.get("verbose")),
    )
    _set_loraplus_ratios(network, kwargs)
    return network


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    """Recreate a Krea 2 LoRA network from a saved state dict.

    Weight loading is intentionally left to the caller, matching the other
    sd-scripts network modules and allowing merge-only inference workflows.
    """

    if weights_sd is None:
        if os.path.splitext(file)[1].lower() == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu", weights_only=False)

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".", 1)[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif key.endswith(".lora_down.weight"):
            modules_dim[lora_name] = value.shape[0]

    for lora_name, dim in modules_dim.items():
        modules_alpha.setdefault(lora_name, dim)

    module_class = lora_flux.LoRAInfModule if for_inference else lora_flux.LoRAModule
    network = AdditionalNetwork(
        text_encoders=[],
        unet=unet,
        arch_config=KREA2_ARCH_CONFIG,
        multiplier=multiplier,
        lora_dim=4,
        alpha=1,
        module_class=module_class,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        exclude_patterns=_parse_patterns(kwargs.get("exclude_patterns")),
        include_patterns=_parse_patterns(kwargs.get("include_patterns")),
        verbose=_parse_bool(kwargs.get("verbose")),
    )
    return network, weights_sd
