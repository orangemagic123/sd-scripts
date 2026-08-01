# Copyright 2026 kohya-ss and Krea AI
#
# Licensed under the Apache License, Version 2.0. This file is adapted from
# krea-ai/krea-2 and kohya-ss/musubi-tuner's Krea 2 integration.

"""Krea 2 single-stream MMDiT and sampling tensor helpers.

The checkpoint architecture is a 28-block, image-first single-stream MMDiT.
Qwen3-VL supplies a stack of selected hidden states; the trainable text-fusion
transformer inside the DiT reduces that stack before joining it with image
tokens.  Image-first ordering is intentional: valid tokens then form a prefix,
which is required by the shared variable-length attention implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from torch import Tensor

from library.attention import AttentionParams, attention as common_attention
from library.custom_offloading_utils import ModelOffloader


def roundup(value: int, multiple: int, name: str = "value") -> int:
    """Round ``value`` up to ``multiple`` (Krea 2 needs 16-pixel alignment)."""

    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(f"[Krea 2] {name}={value} is not a multiple of {multiple}; using {aligned}")
    return aligned


def gather_valid_text(txt: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact valid Qwen3-VL tokens to a prefix and right-pad the batch.

    Qwen3-VL produces ``[valid prompt, padding, valid assistant suffix]``.
    Keeping the original mask would make prefix-based variable-length attention
    discard the suffix.  Compaction is lossless because text RoPE positions are
    all zero and masked tokens do not participate in attention.
    """

    if txt.ndim != 4 or mask.ndim != 2:
        raise ValueError(f"Expected txt [B,S,L,D] and mask [B,S], got {txt.shape} and {mask.shape}")
    if txt.shape[:2] != mask.shape:
        raise ValueError(f"Text/mask shape mismatch: {txt.shape[:2]} vs {mask.shape}")

    mask = mask.to(dtype=torch.bool)
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]
    max_len = max((item.shape[0] for item in valid), default=0)
    out = txt.new_zeros((txt.shape[0], max_len, txt.shape[2], txt.shape[3]))
    new_mask = torch.zeros((txt.shape[0], max_len), device=txt.device, dtype=torch.bool)
    for i, item in enumerate(valid):
        out[i, : item.shape[0]] = item
        new_mask[i, : item.shape[0]] = True
    return out, new_mask


def prepare(img: Tensor, txt_len: int, patch: int, txt_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Patchify latents and build image-first position and validity tensors."""

    b, _, h, w = img.shape
    if h % patch or w % patch:
        raise ValueError(f"Latent size {(h, w)} must be divisible by patch size {patch}")
    h_tokens, w_tokens = h // patch, w // patch
    img_ids = torch.zeros((h_tokens, w_tokens, 3), device=img.device)
    img_ids[..., 1] = torch.arange(h_tokens, device=img.device)[:, None]
    img_ids[..., 2] = torch.arange(w_tokens, device=img.device)[None, :]
    img_pos = repeat(img_ids, "h w axes -> b (h w) axes", b=b)
    img_mask = torch.ones((b, h_tokens * w_tokens), device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txt_pos = torch.zeros((b, txt_len, 3), device=img.device)
    return img, torch.cat((img_pos, txt_pos), dim=1), torch.cat((img_mask, txt_mask), dim=1)


def timesteps(
    seq_len: int,
    steps: int,
    x1: int,
    x2: int,
    y1: float = 0.5,
    y2: float = 1.15,
    sigma: float = 1.0,
    mu: Optional[float] = None,
) -> list[float]:
    """Return Krea 2's shifted flow schedule from one to zero."""

    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    exp_mu = math.exp(mu)
    ts = exp_mu / (exp_mu + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack((torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)), dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_float = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_float = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_float = freqs[..., 0] * xq_float[..., 0] + freqs[..., 1] * xq_float[..., 1]
    xk_float = freqs[..., 0] * xk_float[..., 0] + freqs[..., 1] * xk_float[..., 1]
    return xq_float.reshape_as(xq).to(xq.dtype), xk_float.reshape_as(xk).to(xk.dtype)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = (t.float() * tfactor)[:, None, None] * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: Optional[int] = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        return (vec + rearrange(self.lin, "two d -> 1 two d")).chunk(2, dim=1)


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return (vec + self.lin).chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat([rope(pos[..., i], dim, self.theta, self.ntk) for i, dim in enumerate(self.axdims)], dim=-3)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = F.rms_norm(x.float(), (self.features,), eps=self.eps, weight=self.scale.float() + 1.0)
        return x.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlp_dim = int(2 * features / 3) * multiplier
        mlp_dim = multiple * ((mlp_dim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlp_dim, bias=bias)
        self.up = nn.Linear(features, mlp_dim, bias=bias)
        self.down = nn.Linear(mlp_dim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = heads if kvheads is None else kvheads
        if self.heads % self.kvheads:
            raise ValueError(f"Query heads ({heads}) must be divisible by KV heads ({self.kvheads})")
        self.head_dim = dim // heads
        self.wq = nn.Linear(dim, self.head_dim * heads, bias=bias)
        self.wk = nn.Linear(dim, self.head_dim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.head_dim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.head_dim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Optional[Tensor] = None, attn_params: Optional[AttentionParams] = None) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.kvheads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.kvheads)
        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mode = "torch" if attn_params is None or attn_params.attn_mode is None else attn_params.attn_mode
        # PyTorch 2.4 SDPA and xformers do not have a reliable common GQA path.
        # Repeating K/V is numerically equivalent; FlashAttention and SageAttention
        # accept grouped heads natively and avoid the extra activation memory.
        if self.heads != self.kvheads and mode in ("torch", "xformers"):
            repeats = self.heads // self.kvheads
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        x = common_attention([q, k, v], attn_params=attn_params)
        return self.wo(x * torch.sigmoid(gate))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: Optional[int] = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads, bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, attn_params: Optional[AttentionParams] = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), attn_params=attn_params)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: Optional[int] = None,
    ):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(
        self,
        x: Tensor,
        attn_params_nomask: Optional[AttentionParams] = None,
        attn_params: Optional[AttentionParams] = None,
    ) -> Tensor:
        batch, seq, layers, dim = x.shape
        x = x.reshape(batch * seq, layers, dim)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), attn_params_nomask)
        x = rearrange(x, "(b s) layers d -> b s d layers", b=batch, s=seq)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, attn_params)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: Optional[int] = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads, bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, attn_params: Optional[AttentionParams] = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, attn_params)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        self.config = config
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        head_dim = config.features // config.heads
        axes = [head_dim - 12 * (head_dim // 16), 6 * (head_dim // 16), 6 * (head_dim // 16)]
        if sum(axes) != head_dim or any(axis % 2 for axis in axes):
            raise ValueError(f"Invalid Krea 2 RoPE axes {axes} for head dimension {head_dim}")

        self.posemb = PositionalEncoding(axes, theta=config.theta)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads) for _ in range(config.layers)]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))

        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader: Optional[ModelOffloader] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        if cpu_offload:
            raise ValueError("Krea 2 does not support --cpu_offload_checkpointing; use --blocks_to_swap instead")
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        if num_blocks > len(self.blocks) - 2:
            raise ValueError(f"Cannot swap more than {len(self.blocks) - 2} Krea 2 blocks; requested {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader(self.blocks, num_blocks, device)

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            saved_blocks = self.blocks
            self.blocks = None
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def forward(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        img = self.first(img)
        time_embed = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        time_vec = self.tproj(time_embed)

        img_len = img.shape[1]
        if mask is None:
            txt_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        else:
            txt_mask = mask[:, img_len:].to(dtype=torch.bool)

        no_mask_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, None)
        txt_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, txt_mask)
        context = self.txtfusion(context, no_mask_params, txt_params)
        context = self.txtmlp(context)
        combined = torch.cat((img, context), dim=1)

        full_len = combined.shape[1]
        pad_len = (-full_len) % 256
        if pad_len:
            combined = F.pad(combined, (0, 0, 0, pad_len))
            pos = F.pad(pos, (0, 0, 0, pad_len))
            txt_mask = F.pad(txt_mask, (0, pad_len), value=False)

        attn_params = AttentionParams.create_attention_params_from_mask(
            self.attn_mode, self.split_attn, img_len, txt_mask
        )
        freqs = self.posemb(pos)
        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if self.gradient_checkpointing and self.training:
                combined = torch.utils.checkpoint.checkpoint(
                    block, combined, time_vec, freqs, attn_params, use_reentrant=False
                )
            else:
                combined = block(combined, time_vec, freqs, attn_params)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, index)

        return self.last(combined, time_embed)[:, :img_len]
