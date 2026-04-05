# T-LoRA (Timestep-dependent LoRA) network module
# Based on: https://github.com/ControlGenAI/T-LoRA
# Paper: "T-LoRA: Single Image Diffusion Model Customization Without Overfitting" (AAAI 2026)

import copy
import gc
import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import re

from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel

setup_logging()
import logging

logger = logging.getLogger(__name__)

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


class TLoRAModule(torch.nn.Module):
    """
    T-LoRA module: standard LoRA with timestep-dependent rank masking.
    The sigma_mask is computed by the parent TLoRANetwork based on the current timestep
    and applied between lora_down and lora_up projections.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        network=None,
        is_unet=True,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.network = network
        self.is_unet = is_unet

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"

        if self.is_conv2d:
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # T-LoRA: apply timestep-dependent sigma mask (only for UNet modules)
        sigma_mask = None
        if self.is_unet and self.network is not None:
            sigma_mask = self.network.current_sigma_mask

        if sigma_mask is not None:
            sm = sigma_mask
            if self.lora_dim != sm.shape[-1]:
                r = self.network.current_sigma_r if self.network.current_sigma_r is not None else self.lora_dim
                r = min(self.lora_dim, r)
                sm = torch.ones((1, self.lora_dim), device=lx.device)
                sm[:, r:] = 0.0

            if len(lx.size()) == 3:
                sm = sm.unsqueeze(1)
            elif len(lx.size()) == 4:
                sm = sm.unsqueeze(-1).unsqueeze(-1)
            lx = lx * sm

        # rank dropout (applied additionally if specified, after sigma mask)
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale


class OrthogonalTLoRAModule(torch.nn.Module):
    """
    T-LoRA module with orthogonal SVD-based initialization.
    Uses Q (down), Lambda (diagonal), P (up) decomposition with frozen base copies
    to ensure zero initialization: output = P(Q(x)*lambda*mask) - base_P(base_Q(x)*base_lambda*mask)
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        network=None,
        is_unet=True,
        sig_type="last",
        use_original_weight=False,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.network = network
        self.is_unet = is_unet

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Orthogonal LoRA only supports Linear layers
        if self.is_conv2d:
            # For Conv2d, reshape weight to 2D for SVD, then use Linear layers
            weight_2d = org_module.weight.data.reshape(out_dim, -1)
            effective_in_dim = weight_2d.shape[1]
            self.conv_kernel_size = org_module.kernel_size
            self.conv_stride = org_module.stride
            self.conv_padding = org_module.padding
            self.conv_in_channels = org_module.in_channels
        else:
            effective_in_dim = in_dim

        self.q_layer = torch.nn.Linear(effective_in_dim, lora_dim, bias=False)
        self.p_layer = torch.nn.Linear(lora_dim, out_dim, bias=False)
        self.lambda_layer = torch.nn.Parameter(torch.ones(1, lora_dim))

        if use_original_weight:
            # SVD on original pretrained weight - run on GPU if available for speed
            svd_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if self.is_conv2d:
                base_m = weight_2d.float().to(svd_device)
            else:
                base_m = org_module.weight.data.float().to(svd_device)
            u, s, v = torch.linalg.svd(base_m, full_matrices=False)
            u = u.cpu(); s = s.cpu(); v = v.cpu()

            if sig_type == "principal":
                self.q_layer.weight.data = v[:lora_dim].clone()
                self.p_layer.weight.data = u[:, :lora_dim].clone()
                self.lambda_layer.data = s[None, :lora_dim].clone()
            elif sig_type == "last":
                self.q_layer.weight.data = v[-lora_dim:].clone()
                self.p_layer.weight.data = u[:, -lora_dim:].clone()
                self.lambda_layer.data = s[None, -lora_dim:].clone()
            elif sig_type == "middle":
                start_v = math.ceil((v.shape[0] - lora_dim) / 2)
                start_u = math.ceil((u.shape[1] - lora_dim) / 2)
                start_s = math.ceil((s.shape[0] - lora_dim) / 2)
                self.q_layer.weight.data = v[start_v:start_v + lora_dim].clone()
                self.p_layer.weight.data = u[:, start_u:start_u + lora_dim].clone()
                self.lambda_layer.data = s[None, start_s:start_s + lora_dim].clone()

            del u, s, v, base_m
            gc.collect()
        else:
            # Random orthogonal init - skip SVD entirely (orders of magnitude faster)
            torch.nn.init.orthogonal_(self.q_layer.weight)
            torch.nn.init.orthogonal_(self.p_layer.weight)
            self.lambda_layer.data = torch.abs(torch.randn(1, lora_dim)) / math.sqrt(lora_dim)

        # Frozen base copies for zero-init residual
        self.base_q = copy.deepcopy(self.q_layer)
        self.base_p = copy.deepcopy(self.p_layer)
        self.base_lambda = self.lambda_layer.data.clone()
        self.register_buffer("base_lambda_buf", self.base_lambda)

        for param in self.base_q.parameters():
            param.requires_grad_(False)
        for param in self.base_p.parameters():
            param.requires_grad_(False)

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _apply_conv_reshape(self, x):
        """For Conv2d modules, unfold input to 2D for linear operations."""
        if not self.is_conv2d:
            return x, None
        # Use unfold to match conv2d behavior
        batch_size = x.shape[0]
        # Use conv2d with the identity-like approach: apply q_layer as 1x1 after a grouped conv
        return x, batch_size

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        orig_dtype = x.dtype
        dtype = self.q_layer.weight.dtype

        # Get effective rank from network (set by hook); default to full rank
        r = self.lora_dim
        if self.is_unet and self.network is not None and self.network.current_sigma_r is not None:
            r = min(self.network.current_sigma_r, self.lora_dim)

        if self.is_conv2d:
            q_weight = self.q_layer.weight[:r].reshape(r, self.conv_in_channels, *self.conv_kernel_size)
            base_q_weight = self.base_q.weight[:r].reshape(r, self.conv_in_channels, *self.conv_kernel_size)
            p_weight = self.p_layer.weight[:, :r].reshape(self.p_layer.weight.shape[0], r, 1, 1)
            base_p_weight = self.base_p.weight[:, :r].reshape(self.base_p.weight.shape[0], r, 1, 1)

            x_dt = x.to(dtype)
            lx = torch.nn.functional.conv2d(x_dt, q_weight, stride=self.conv_stride, padding=self.conv_padding)
            lx = lx * self.lambda_layer[:, :r].unsqueeze(-1).unsqueeze(-1)
            lx = torch.nn.functional.conv2d(lx, p_weight)

            base_lx = torch.nn.functional.conv2d(x_dt, base_q_weight, stride=self.conv_stride, padding=self.conv_padding)
            base_lx = base_lx * self.base_lambda_buf[:, :r].unsqueeze(-1).unsqueeze(-1)
            base_lx = torch.nn.functional.conv2d(base_lx, base_p_weight)
        else:
            x_dt = x.to(dtype)
            q_w = self.q_layer.weight[:r]
            p_w = self.p_layer.weight[:, :r]
            lam = self.lambda_layer[:, :r]
            lx = torch.nn.functional.linear(x_dt, q_w) * lam
            lx = torch.nn.functional.linear(lx, p_w)

            base_q_w = self.base_q.weight[:r]
            base_p_w = self.base_p.weight[:, :r]
            base_lam = self.base_lambda_buf[:, :r]
            base_lx = torch.nn.functional.linear(x_dt, base_q_w) * base_lam
            base_lx = torch.nn.functional.linear(base_lx, base_p_w)

        # normal dropout
        result = lx - base_lx
        if self.dropout is not None and self.training:
            result = torch.nn.functional.dropout(result, p=self.dropout)

        return org_forwarded + result.to(orig_dtype) * self.multiplier * self.scale



# Import LoRA utilities from existing lora.py
from networks.lora import (
    get_block_dims_and_alphas,
    get_block_lr_weight,
    remove_block_dims_and_alphas,
    parse_block_lr_kwargs,
    get_block_index,
    convert_diffusers_to_sai_if_needed,
    LoRAInfModule,
)
from diffusers import AutoencoderKL
from transformers import CLIPTextModel


def get_timestep_sigma_mask(timestep, max_timestep, max_rank, min_rank=1, alpha=1.0):
    """
    Compute the T-LoRA sigma mask based on the current timestep.
    Lower timesteps (less noise) -> higher effective rank.
    Higher timesteps (more noise) -> lower effective rank.
    """
    t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    r = int(((max_timestep - t) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    r = max(min_rank, min(r, max_rank))  # clamp
    sigma_mask = torch.zeros((1, max_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask


class TLoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12
    NUM_OF_MID_BLOCKS = 1
    SDXL_NUM_OF_BLOCKS = 9
    SDXL_NUM_OF_MID_BLOCKS = 3

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        module_class: Type[object] = None,
        varbose: Optional[bool] = False,
        is_sdxl: Optional[bool] = False,
        # T-LoRA specific
        tlora_min_rank: int = 1,
        tlora_alpha_rank_scale: float = 1.0,
        tlora_max_timestep: int = 1000,
        tlora_init: str = "kaiming",
        tlora_sig_type: str = "last",
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # T-LoRA params
        self.tlora_min_rank = tlora_min_rank
        self.tlora_alpha_rank_scale = tlora_alpha_rank_scale
        self.tlora_max_timestep = tlora_max_timestep
        self.tlora_init = tlora_init
        self.tlora_sig_type = tlora_sig_type
        self.current_sigma_mask = None
        self.current_sigma_r = None

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if module_class is None:
            if tlora_init in ("orthogonal", "layer_orthogonal"):
                module_class = OrthogonalTLoRAModule
            else:
                module_class = TLoRAModule

        self._module_class = module_class

        logger.info(f"create T-LoRA network. dim: {lora_dim}, alpha: {alpha}, init: {tlora_init}")
        logger.info(f"T-LoRA params: min_rank={tlora_min_rank}, alpha_rank_scale={tlora_alpha_rank_scale}, max_timestep={tlora_max_timestep}")
        if tlora_init != "kaiming":
            logger.info(f"T-LoRA orthogonal init: sig_type={tlora_sig_type}, use_original_weight={tlora_init == 'layer_orthogonal'}")
        logger.info(
            f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
        )
        if self.conv_lora_dim is not None:
            logger.info(f"apply T-LoRA to Conv2d with kernel size (3,3). dim: {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[torch.nn.Module]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                block_idx = get_block_index(lora_name, is_sdxl)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha_val = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha_val = conv_block_alphas[block_idx]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha_val = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha_val = self.conv_alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            kwargs = dict(
                                lora_name=lora_name,
                                org_module=child_module,
                                multiplier=self.multiplier,
                                lora_dim=dim,
                                alpha=alpha_val,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                network=self,
                                is_unet=is_unet,
                            )
                            if module_class == OrthogonalTLoRAModule:
                                kwargs["sig_type"] = self.tlora_sig_type
                                kwargs["use_original_weight"] = (self.tlora_init == "layer_orthogonal")

                            lora = module_class(**kwargs)
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        self.text_encoder_loras = []
        skipped_te = []
        for i, te in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
                logger.info(f"create T-LoRA for Text Encoder {index}:")
            else:
                index = None
                logger.info(f"create T-LoRA for Text Encoder:")
            text_encoder_loras, skipped = create_modules(False, index, te, TLoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        logger.info(f"create T-LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        target_modules = TLoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += TLoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        logger.info(f"create T-LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            logger.warning(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} T-LoRA modules are skipped:"
            )
            for name in skipped:
                logger.info(f"	{name}")

        self.block_lr_weight = None
        self.block_lr = False

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def _unet_forward_pre_hook(self, module, args, kwargs=None):
        """Hook to capture timestep from UNet forward call and compute sigma_mask."""
        # SdxlUNet2DConditionModel.forward(self, x, timesteps=None, context=None, y=None, **kwargs)
        # Standard UNet: forward(sample, timestep, encoder_hidden_states, ...)
        timestep = None
        if len(args) >= 2:
            timestep = args[1]
        elif kwargs is not None and "timesteps" in kwargs:
            timestep = kwargs["timesteps"]
        elif kwargs is not None and "timestep" in kwargs:
            timestep = kwargs["timestep"]

        if timestep is not None:
            target_device = None
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                target_device = args[0].device

            if isinstance(timestep, torch.Tensor):
                t = timestep.flatten()[0] if timestep.dim() > 0 else timestep
                t = t.item()
            else:
                t = timestep
            r = int(((self.tlora_max_timestep - t) / self.tlora_max_timestep) ** self.tlora_alpha_rank_scale
                    * (self.lora_dim - self.tlora_min_rank)) + self.tlora_min_rank
            r = max(self.tlora_min_rank, min(r, self.lora_dim))
            mask = torch.zeros((1, self.lora_dim), device=target_device)
            mask[:, :r] = 1.0
            self.current_sigma_mask = mask
            self.current_sigma_r = r

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable T-LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable T-LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # Register forward pre-hook on UNet to capture timesteps
        if apply_unet and unet is not None:
            self._hook_handle = unet.register_forward_pre_hook(self._unet_forward_pre_hook, with_kwargs=True)
            logger.info("registered T-LoRA timestep hook on UNet")

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(TLoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(TLoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable T-LoRA for text encoder")
        else:
            self.text_encoder_loras = []
        if apply_unet:
            logger.info("enable T-LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1:]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)
        logger.info(f"weights are merged")

    def set_block_lr_weight(self, block_lr_weight: Optional[List[float]]):
        self.block_lr = True
        self.block_lr_weight = block_lr_weight

    def get_lr_weight(self, block_idx: int) -> float:
        if not self.block_lr or self.block_lr_weight is None:
            return 1.0
        return self.block_lr_weight[block_idx]

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio
        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if ratio is not None and ("lora_up" in name or "p_layer" in name):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param
            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            params, descriptions = assemble_params(
                self.text_encoder_loras,
                text_encoder_lr if text_encoder_lr is not None else default_lr,
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["textencoder" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            if self.block_lr:
                is_sdxl = False
                for lora in self.unet_loras:
                    if "input_blocks" in lora.lora_name or "output_blocks" in lora.lora_name:
                        is_sdxl = True
                        break
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = get_block_index(lora.lora_name, is_sdxl)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)
                for idx, block_loras in block_idx_to_lora.items():
                    params, descriptions = assemble_params(
                        block_loras,
                        (unet_lr if unet_lr is not None else default_lr) * self.get_lr_weight(idx),
                        self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
                    )
                    all_params.extend(params)
                    lr_descriptions.extend([f"unet_block{idx}" + (" " + d if d else "") for d in descriptions])
            else:
                params, descriptions = assemble_params(
                    self.unet_loras,
                    unet_lr if unet_lr is not None else default_lr,
                    self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
                )
                all_params.extend(params)
                lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            # Store T-LoRA specific metadata
            metadata["ss_tlora_min_rank"] = str(self.tlora_min_rank)
            metadata["ss_tlora_alpha_rank_scale"] = str(self.tlora_alpha_rank_scale)
            metadata["ss_tlora_max_timestep"] = str(self.tlora_max_timestep)
            metadata["ss_tlora_init"] = self.tlora_init
            metadata["ss_tlora_sig_type"] = self.tlora_sig_type

            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_region(self, sub_prompt_index, is_last_network, mask):
        if mask.max() == 0:
            mask = torch.ones_like(mask)
        self.mask = mask
        self.sub_prompt_index = sub_prompt_index
        self.is_last_network = is_last_network
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_network(self)

    def set_current_generation(self, batch_size, num_sub_prompts, width, height, shared, ds_ratio=None):
        pass



def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    # T-LoRA specific args
    tlora_min_rank = int(kwargs.get("tlora_min_rank", 1))
    tlora_alpha_rank_scale = float(kwargs.get("tlora_alpha_rank_scale", 1.0))
    tlora_max_timestep = int(kwargs.get("tlora_max_timestep", 1000))
    tlora_init = kwargs.get("tlora_init", "kaiming")  # kaiming, orthogonal, layer_orthogonal
    tlora_sig_type = kwargs.get("tlora_sig_type", "last")  # principal, last, middle

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    block_lr_weight = parse_block_lr_kwargs(is_sdxl, kwargs)

    if block_dims is not None or block_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
        )

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, conv_block_dims, conv_block_alphas, block_lr_weight
        )
    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    network = TLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,
        is_sdxl=is_sdxl,
        tlora_min_rank=tlora_min_rank,
        tlora_alpha_rank_scale=tlora_alpha_rank_scale,
        tlora_max_timestep=tlora_max_timestep,
        tlora_init=tlora_init,
        tlora_sig_type=tlora_sig_type,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    if is_sdxl:
        convert_diffusers_to_sai_if_needed(weights_sd)

    # Detect if this is an orthogonal T-LoRA (has q_layer/p_layer keys) or standard
    has_orthogonal = any("q_layer" in k for k in weights_sd.keys())

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key or "q_layer" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    # Read T-LoRA metadata if available
    tlora_min_rank = int(kwargs.get("tlora_min_rank", 1))
    tlora_alpha_rank_scale = float(kwargs.get("tlora_alpha_rank_scale", 1.0))
    tlora_max_timestep = int(kwargs.get("tlora_max_timestep", 1000))
    tlora_init = "orthogonal" if has_orthogonal else "kaiming"
    tlora_sig_type = kwargs.get("tlora_sig_type", "last")

    if has_orthogonal:
        module_class = OrthogonalTLoRAModule
    elif for_inference:
        module_class = TLoRAModule  # Could use LoRAInfModule variant if needed
    else:
        module_class = TLoRAModule

    network = TLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        is_sdxl=is_sdxl,
        tlora_min_rank=tlora_min_rank,
        tlora_alpha_rank_scale=tlora_alpha_rank_scale,
        tlora_max_timestep=tlora_max_timestep,
        tlora_init=tlora_init,
        tlora_sig_type=tlora_sig_type,
    )

    block_lr_weight = parse_block_lr_kwargs(is_sdxl, kwargs)
    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network, weights_sd
