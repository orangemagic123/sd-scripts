# Copyright 2026 kohya-ss and Krea AI
# Licensed under the Apache License, Version 2.0.

"""Krea 2 checkpoint and Qwen3-VL conditioner loading helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from accelerate import init_empty_weights
from torch import Tensor

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.krea2_models import SingleMMDiTConfig, SingleStreamDiT, single_mmdit_large_wide
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library.safetensors_utils import load_safetensors, load_split_weights

logger = logging.getLogger(__name__)


QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Vendored from Qwen/Qwen3-VL-4B-Instruct so a single official/ComfyUI
# safetensors file can be loaded without a second model directory.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_path: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _get_qwen3_vl_classes():
    try:
        from transformers import AutoTokenizer, Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Krea 2 requires transformers>=4.57 with Qwen3-VL support. "
            "Install the repository requirements before loading the text encoder."
        ) from exc
    return AutoTokenizer, Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration


def _convert_comfyui_qwen3vl_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert ComfyUI Qwen3-VL keys to Transformers' model layout."""

    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = True,
) -> Any:
    _, _, Qwen3VLConfig, Qwen3VLForConditionalGeneration = _get_qwen3_vl_classes()
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 Qwen3-VL weights from {model_path}")
    state_dict = load_split_weights(model_path, device=device, disable_mmap=disable_mmap, dtype=dtype)
    state_dict = _convert_comfyui_qwen3vl_state_dict(state_dict)
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    model.tie_weights()

    missing = [key for key in info.missing_keys if key != "lm_head.weight"]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            "Qwen3-VL checkpoint mismatch: "
            f"missing={missing[:10]}, unexpected={list(info.unexpected_keys)[:10]}"
        )
    model.to(device=device, dtype=dtype)
    return model.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    """Text-only Qwen3-VL conditioner used by Krea 2."""

    def __init__(
        self,
        qwen: torch.nn.Module,
        tokenizer,
        suffix_tokenizer,
        max_length: int = TextEncoderConfig.max_length,
        select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.suffix_tokenizer = suffix_tokenizer
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prefix_start_index = 34
        self.suffix_start_index = 5

    @property
    def device(self) -> torch.device:
        return self.qwen.device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        text = [self.prompt_prefix + item for item in text]
        suffix_inputs = self.suffix_tokenizer(
            text=[self.prompt_suffix] * len(text), return_tensors="pt"
        ).to(self.qwen.device, non_blocking=True)
        inputs = self.tokenizer(
            text,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length + self.prefix_start_index - self.suffix_start_index,
            return_tensors="pt",
        ).to(self.qwen.device, non_blocking=True)
        input_ids = torch.cat((inputs["input_ids"], suffix_inputs["input_ids"]), dim=1)
        mask = torch.cat((inputs["attention_mask"].bool(), suffix_inputs["attention_mask"].bool()), dim=1)

        with torch.no_grad():
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            hiddens = torch.stack([states.hidden_states[index] for index in self.select_layers], dim=2)
        return hiddens[:, self.prefix_start_index :], mask[:, self.prefix_start_index :]


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_path: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> Qwen3VLConditioner:
    AutoTokenizer, Qwen2TokenizerFast, _, _ = _get_qwen3_vl_classes()
    qwen = _load_qwen3_vl_model(path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_length)
    suffix_tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, suffix_tokenizer, max_length, select_layers)
    return conditioner.eval().requires_grad_(False)


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[Tensor, Tensor]:
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)


KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    lora_weights: Optional[list[dict[str, Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    disable_mmap: bool = True,
) -> SingleStreamDiT:
    """Construct the fixed Krea 2 DiT on meta and assign checkpoint tensors."""

    calc_device = torch.device(device)
    loading_device = calc_device if loading_device is None else torch.device(loading_device)
    has_lora = bool(lora_weights)
    logger.info(
        f"Loading Krea 2 DiT from {dit_path}"
        + (" with scaled fp8" if fp8_scaled else "")
        + (f" and {len(lora_weights)} merged LoRA(s)" if has_lora else "")
    )
    with init_empty_weights():
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    if fp8_scaled or has_lora:
        state_dict = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled,
            calc_device=calc_device,
            move_to_device=loading_device == calc_device,
            dit_weight_dtype=None if fp8_scaled else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
            disable_numpy_memmap=disable_mmap,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(dit, state_dict, use_scaled_mm=False)
        if loading_device.type != "cpu":
            state_dict = {key: value.to(loading_device) for key, value in state_dict.items()}
    else:
        state_dict = load_safetensors(
            dit_path, device=loading_device, disable_mmap=disable_mmap, dtype=dtype
        )

    missing, unexpected = dit.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Krea 2 DiT checkpoint mismatch: missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    return dit
