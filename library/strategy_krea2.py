"""Dataset strategies for Krea 2 LoRA training."""

from __future__ import annotations

import hashlib
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch

from library.krea2_models import gather_valid_text
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.strategy_base import TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy


class Krea2TokenizeStrategy(TokenizeStrategy):
    """Placeholder token strategy.

    Krea 2 requires pre-cached Qwen3-VL outputs. Tokenization therefore happens
    inside the conditioner during the cache pass, not in dataset workers. A
    deterministic dummy token keeps dataset debugging and the shared interface
    operational without serializing a Transformers tokenizer into workers.
    """

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        batch_size = 1 if isinstance(text, str) else len(text)
        return [torch.zeros((batch_size, 1), dtype=torch.long)]


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        raise RuntimeError(
            "Krea 2 does not encode text in the training step. Enable --cache_text_encoder_outputs."
        )


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Cache compacted selected Qwen3-VL hidden states.

    bfloat16 values are persisted as their uint16 bit representation because
    NumPy/NPZ has no portable bfloat16 dtype. ``load_outputs_npz`` restores a
    bfloat16 torch Tensor before dataset collation, preserving both disk size and
    the exact encoder values.
    """

    KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_krea2_te.npz"
    CACHE_VERSION = 2
    EXPECTED_TEXT_LAYERS = 12
    EXPECTED_TEXT_DIM = 2560
    requires_processed_captions = True

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: Optional[int],
        skip_disk_cache_validity_check: bool,
        text_encoder_max_length: int = 512,
        tokenizer_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        num_variants: int = 0,
    ) -> None:
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            is_partial=False,
            num_variants=num_variants,
        )
        self.cache_signature = self._build_cache_signature(
            text_encoder_max_length,
            tokenizer_path,
            text_encoder_path,
        )

    @staticmethod
    def _source_signature(source: Optional[str]) -> str:
        if source is None:
            return ""
        source = os.path.expanduser(str(source))
        if not os.path.exists(source):
            return f"id:{source}"
        absolute_source = os.path.abspath(source)
        if os.path.isdir(source):
            entries = []
            for root, dirnames, filenames in os.walk(source):
                dirnames.sort()
                for filename in sorted(filenames):
                    path = os.path.join(root, filename)
                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue
                    relative_path = os.path.relpath(path, source).replace(os.sep, "/")
                    entries.append(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}")
            contents = hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()
            return f"dir:{absolute_source}:{contents}"
        stat = os.stat(source)
        return f"file:{absolute_source}:{stat.st_size}:{stat.st_mtime_ns}"

    @classmethod
    def _build_cache_signature(
        cls,
        text_encoder_max_length: int,
        tokenizer_path: Optional[str],
        text_encoder_path: Optional[str],
    ) -> str:
        description = "|".join(
            (
                f"max_length={text_encoder_max_length}",
                f"tokenizer={cls._source_signature(tokenizer_path)}",
                f"text_encoder={cls._source_signature(text_encoder_path)}",
            )
        )
        return hashlib.sha256(description.encode("utf-8")).hexdigest()

    @staticmethod
    def _caption_hash(caption: str) -> str:
        return hashlib.sha256(caption.encode("utf-8")).hexdigest()

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def get_variant_outputs_npz_path(self, image_abs_path: str, variant_idx: int) -> str:
        return os.path.splitext(image_abs_path)[0] + f"_krea2_te_v{variant_idx}.npz"

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with np.load(npz_path) as data:
                if not (
                    "vl_embed" in data
                    and "attention_mask" in data
                    and "vl_embed_dtype" in data
                    and "cache_version" in data
                    and "cache_signature" in data
                    and "caption_sha256" in data
                    and int(data["cache_version"]) == self.CACHE_VERSION
                ):
                    return False
                embed = data["vl_embed"]
                attention_mask = data["attention_mask"]
                return (
                    embed.dtype == np.uint16
                    and embed.ndim == 3
                    and embed.shape[1:] == (self.EXPECTED_TEXT_LAYERS, self.EXPECTED_TEXT_DIM)
                    and attention_mask.dtype == np.bool_
                    and attention_mask.ndim == 1
                    and attention_mask.shape[0] == embed.shape[0]
                    and str(data["vl_embed_dtype"].item()) == "bfloat16"
                    and str(data["cache_signature"].item()) == self.cache_signature
                )
        except Exception:
            return False

    def is_disk_cached_outputs_expected_for_caption(self, npz_path: str, caption: str) -> bool:
        if not self.is_disk_cached_outputs_expected(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with np.load(npz_path) as data:
                return str(data["caption_sha256"].item()) == self._caption_hash(caption)
        except Exception:
            return False

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
        tensor = tensor.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy(), "bfloat16"
        return tensor.numpy(), str(tensor.dtype).removeprefix("torch.")

    @staticmethod
    def _numpy_to_tensor(array: np.ndarray, dtype_name: str) -> torch.Tensor:
        # np.load arrays may be read-only; copy before exposing them to PyTorch.
        tensor = torch.from_numpy(np.array(array, copy=True))
        if dtype_name == "bfloat16":
            return tensor.view(torch.bfloat16)
        dtype = getattr(torch, dtype_name, None)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def load_outputs_npz(self, npz_path: str) -> List[torch.Tensor]:
        with np.load(npz_path) as data:
            dtype_name = str(data["vl_embed_dtype"].item())
            embed = self._numpy_to_tensor(data["vl_embed"], dtype_name)
            attention_mask = torch.from_numpy(np.array(data["attention_mask"], copy=True)).bool()
        return [embed, attention_mask]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List[Any],
        captions: Optional[List[str]] = None,
        variant_idx: Optional[int] = None,
    ):
        if variant_idx is not None and not self.cache_to_disk:
            raise ValueError("Krea 2 caption variants require disk caching")
        if not models or models[0] is None:
            raise ValueError("The Qwen3-VL text encoder is required to build Krea 2 caches")
        if captions is None:
            captions = [info.caption for info in infos]

        with torch.no_grad():
            hidden_states, attention_mask = models[0](captions)
            hidden_states, attention_mask = gather_valid_text(hidden_states, attention_mask)

        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"Krea 2 text caches require bfloat16 encoder outputs, got {hidden_states.dtype}")
        if hidden_states.shape[2:] != (self.EXPECTED_TEXT_LAYERS, self.EXPECTED_TEXT_DIM):
            raise ValueError(
                "Unexpected Krea 2 text embedding shape: "
                f"{tuple(hidden_states.shape)}; expected (..., {self.EXPECTED_TEXT_LAYERS}, {self.EXPECTED_TEXT_DIM})"
            )

        for index, info in enumerate(infos):
            valid_length = int(attention_mask[index].sum().item())
            embed = hidden_states[index, :valid_length].contiguous()
            mask = attention_mask[index, :valid_length].contiguous()
            if self.cache_to_disk:
                npz_path = (
                    self.get_variant_outputs_npz_path(info.absolute_path, variant_idx)
                    if variant_idx is not None
                    else info.text_encoder_outputs_npz
                )
                array, dtype_name = self._tensor_to_numpy(embed)
                np.savez(
                    npz_path,
                    vl_embed=array,
                    vl_embed_dtype=np.asarray(dtype_name),
                    attention_mask=mask.cpu().numpy(),
                    cache_version=np.asarray(self.CACHE_VERSION, dtype=np.int32),
                    cache_signature=np.asarray(self.cache_signature),
                    caption_sha256=np.asarray(self._caption_hash(captions[index])),
                )
            else:
                info.text_encoder_outputs = (embed.cpu(), mask.cpu())


class Krea2LatentsCachingStrategy(AnimaLatentsCachingStrategy):
    """Qwen-Image VAE cache with a Krea-specific suffix."""

    KREA2_LATENTS_NPZ_SUFFIX = "_krea2.npz"

    @property
    def cache_suffix(self) -> str:
        return self.KREA2_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.cache_suffix
