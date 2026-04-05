# T-LoRA (Timestep-dependent LoRA) network module for Anima
# Based on: https://github.com/ControlGenAI/T-LoRA
# Adapted from networks/lora_anima.py for Anima DiT architecture

import ast
import copy
import gc
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def get_timestep_sigma_mask(timestep, max_timestep, max_rank, min_rank=1, alpha=1.0):
    """
    Compute the T-LoRA sigma mask based on the current timestep.
    Lower timesteps (less noise) -> higher effective rank.
    Higher timesteps (more noise) -> lower effective rank.

    For Anima, timesteps are in [0, 1] range (Flow Matching).
    """
    t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    r = int(((max_timestep - t) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    r = max(min_rank, min(r, max_rank))
    sigma_mask = torch.zeros((1, max_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask


class TLoRAModule(torch.nn.Module):
    """
    T-LoRA module for Anima: standard LoRA with timestep-dependent rank masking.
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

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

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

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # T-LoRA: apply timestep-dependent sigma mask (only for DiT modules, not text encoder)
        sigma_mask = None
        if self.is_unet and self.network is not None:
            sigma_mask = self.network.current_sigma_mask

        if sigma_mask is not None:
            sm = sigma_mask
            if self.lora_dim != sm.shape[-1]:
                sm = torch.ones((1, self.lora_dim), device=lx.device)
                r = min(self.lora_dim, (sigma_mask > 0).sum().item())
                sm[:, r:] = 0.0
            else:
                sm = sm.to(lx.device)

            if isinstance(self.lora_down, torch.nn.Conv2d):
                sm = sm.unsqueeze(-1).unsqueeze(-1)
            else:
                for _ in range(len(lx.size()) - 2):
                    sm = sm.unsqueeze(1)
            lx = lx * sm

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if isinstance(self.lora_down, torch.nn.Conv2d):
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            else:
                for _ in range(len(lx.size()) - 2):
                    mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class OrthogonalTLoRAModule(torch.nn.Module):
    """
    T-LoRA module with orthogonal SVD-based initialization for Anima.
    Uses Q/Lambda/P decomposition with frozen base copies for zero init.
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

        if self.is_conv2d:
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

        # SVD initialization - run on GPU if available for speed
        svd_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if use_original_weight:
            if self.is_conv2d:
                base_m = weight_2d.float().to(svd_device)
            else:
                base_m = org_module.weight.data.float().to(svd_device)
            u, s, v = torch.linalg.svd(base_m, full_matrices=False)
        else:
            base_m = torch.normal(mean=0, std=1.0 / lora_dim, size=(effective_in_dim, out_dim), device=svd_device)
            u, s, v = torch.linalg.svd(base_m, full_matrices=False)

        # Move SVD results to CPU for storage
        u = u.cpu()
        s = s.cpu()
        v = v.cpu()

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

        # Frozen base copies
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

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        orig_dtype = x.dtype
        dtype = self.q_layer.weight.dtype

        sigma_mask = None
        if self.is_unet and self.network is not None:
            sigma_mask = self.network.current_sigma_mask

        if sigma_mask is None:
            mask = torch.ones((1, self.lora_dim), device=x.device)
        else:
            mask = sigma_mask.to(x.device)
            if self.lora_dim != mask.shape[-1]:
                mask = torch.ones((1, self.lora_dim), device=x.device)
                r = min(self.lora_dim, (sigma_mask > 0).sum().item())
                mask[:, r:] = 0.0

        if self.is_conv2d:
            q_weight = self.q_layer.weight.data.reshape(
                self.lora_dim, self.conv_in_channels, *self.conv_kernel_size
            )
            base_q_weight = self.base_q.weight.data.reshape(
                self.lora_dim, self.conv_in_channels, *self.conv_kernel_size
            )
            p_weight = self.p_layer.weight.data.reshape(self.p_layer.weight.shape[0], self.lora_dim, 1, 1)
            base_p_weight = self.base_p.weight.data.reshape(self.base_p.weight.shape[0], self.lora_dim, 1, 1)

            lx = torch.nn.functional.conv2d(x.to(dtype), q_weight, stride=self.conv_stride, padding=self.conv_padding)
            lx = lx * self.lambda_layer.unsqueeze(-1).unsqueeze(-1) * mask.unsqueeze(-1).unsqueeze(-1)
            lx = torch.nn.functional.conv2d(lx, p_weight)

            base_lx = torch.nn.functional.conv2d(x.to(dtype), base_q_weight, stride=self.conv_stride, padding=self.conv_padding)
            base_lx = base_lx * self.base_lambda_buf.unsqueeze(-1).unsqueeze(-1) * mask.unsqueeze(-1).unsqueeze(-1)
            base_lx = torch.nn.functional.conv2d(base_lx, base_p_weight)
        else:
            lx = self.q_layer(x.to(dtype)) * self.lambda_layer * mask
            lx = self.p_layer(lx)

            base_lx = self.base_q(x.to(dtype)) * self.base_lambda_buf * mask
            base_lx = self.base_p(base_lx)

        result = lx - base_lx
        if self.dropout is not None and self.training:
            result = torch.nn.functional.dropout(result, p=self.dropout)

        return org_forwarded + result.to(orig_dtype) * self.multiplier * self.scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype



class TLoRANetwork(torch.nn.Module):
    """T-LoRA network for Anima DiT models with timestep-dependent rank masking."""

    ANIMA_TARGET_REPLACE_MODULE = ["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"]
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"]

    LORA_PREFIX_ANIMA = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoders: list,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_llm_adapter: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
        # T-LoRA specific
        tlora_min_rank: int = 1,
        tlora_alpha_rank_scale: float = 1.0,
        tlora_max_timestep: float = 1.0,
        tlora_init: str = "kaiming",
        tlora_sig_type: str = "last",
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs

        # T-LoRA params
        self.tlora_min_rank = tlora_min_rank
        self.tlora_alpha_rank_scale = tlora_alpha_rank_scale
        self.tlora_max_timestep = tlora_max_timestep
        self.tlora_init = tlora_init
        self.tlora_sig_type = tlora_sig_type
        self.current_sigma_mask = None

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if module_class is None:
            if tlora_init in ("orthogonal", "layer_orthogonal"):
                module_class = OrthogonalTLoRAModule
            else:
                module_class = TLoRAModule
        self._module_class = module_class

        if modules_dim is not None:
            logger.info("create T-LoRA network from weights")
        else:
            logger.info(f"create T-LoRA network for Anima. dim: {lora_dim}, alpha: {alpha}, init: {tlora_init}")
            logger.info(f"T-LoRA params: min_rank={tlora_min_rank}, alpha_rank_scale={tlora_alpha_rank_scale}, max_timestep={tlora_max_timestep}")
            if tlora_init != "kaiming":
                logger.info(f"T-LoRA orthogonal init: sig_type={tlora_sig_type}, use_original_weight={tlora_init == 'layer_orthogonal'}")
            logger.info(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")

        # compile regex patterns
        def str_to_re_patterns(patterns):
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_patterns.append(re.compile(pattern))
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

        def create_modules(is_unet, text_encoder_idx, root_module, target_replace_modules, default_dim=None):
            prefix = self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            excluded = any(p.fullmatch(original_name) for p in exclude_re_patterns)
                            included = any(p.fullmatch(original_name) for p in include_re_patterns)
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if self.reg_dims is not None:
                                    for reg, d in self.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = self.alpha
                                            logger.info(f"Module {original_name} matched regex '{reg}' -> dim: {dim}")
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = default_dim if default_dim is not None else self.lora_dim
                                        alpha_val = self.alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
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
                            lora.original_name = original_name
                            loras.append(lora)

                    if target_replace_modules is None:
                        break
            return loras, skipped

        # Create for text encoders
        self.text_encoder_loras = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create T-LoRA for Text Encoder {i+1}:")
                te_loras, te_skipped = create_modules(False, i, text_encoder, TLoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                logger.info(f"create T-LoRA for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create for DiT
        target_modules = list(TLoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(TLoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        logger.info(f"create T-LoRA for Anima DiT: {len(self.unet_loras)} modules.")

        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} T-LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def _unet_forward_pre_hook(self, module, args, kwargs=None):
        """Hook to capture timestep from Anima DiT forward and compute sigma_mask.
        Anima forward: forward(self, x, timesteps, context, ...) where timesteps is in [0, 1].
        """
        timestep = None
        if len(args) >= 2:
            timestep = args[1]
        elif kwargs is not None and "timesteps" in kwargs:
            timestep = kwargs["timesteps"]

        if timestep is not None:
            if isinstance(timestep, torch.Tensor):
                t = timestep.flatten()[0] if timestep.dim() > 0 else timestep
            else:
                t = timestep
            self.current_sigma_mask = get_timestep_sigma_mask(
                t,
                self.tlora_max_timestep,
                self.lora_dim,
                self.tlora_min_rank,
                self.tlora_alpha_rank_scale,
            )

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

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable T-LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable T-LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # Register hook on DiT to capture timesteps
        if apply_unet and unet is not None:
            self._hook_handle = unet.register_forward_pre_hook(self._unet_forward_pre_hook, with_kwargs=True)
            logger.info("registered T-LoRA timestep hook on Anima DiT")

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(TLoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(TLoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable T-LoRA for text encoder")
        else:
            self.text_encoder_loras = []
        if apply_unet:
            logger.info("enable T-LoRA for DiT")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1:]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)
        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio
        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]

        self.requires_grad_(True)
        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        break

                for name, param in lora.named_parameters():
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "lr": reg_lr}
                        if loraplus_ratio is not None and ("lora_up" in name or "p_layer" in name):
                            reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                        else:
                            reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                        continue

                    if loraplus_ratio is not None and ("lora_up" in name or "p_layer" in name):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = reg_lr * loraplus_ratio if loraplus_ratio is not None else reg_lr
                    else:
                        param_data["lr"] = reg_lr
                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    param_data["lr"] = lr * loraplus_ratio if key == "plus" else lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            te_loras = [l for l in self.text_encoder_loras if l.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)]
            if len(te_loras) > 0:
                params, descriptions = assemble_params(te_loras, text_encoder_lr[0], loraplus_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
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
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    # T-LoRA specific args
    tlora_min_rank = int(kwargs.get("tlora_min_rank", 1))
    tlora_alpha_rank_scale = float(kwargs.get("tlora_alpha_rank_scale", 1.0))
    # Anima uses [0, 1] scaled timesteps (Flow Matching)
    tlora_max_timestep = float(kwargs.get("tlora_max_timestep", 1.0))
    tlora_init = kwargs.get("tlora_init", "kaiming")
    tlora_sig_type = kwargs.get("tlora_sig_type", "last")

    # LLM adapter
    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if train_llm_adapter is not None:
        train_llm_adapter = True if train_llm_adapter.lower() == "true" else False

    # exclude/include patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]
    exclude_patterns.append(r".*(_modulation|_norm|_embedder|final_layer).*")

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if verbose.lower() == "true" else False

    # regex-specific dims/lrs
    def parse_kv_pairs(kv_pair_str, is_int):
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            try:
                pairs[key.strip()] = int(value.strip()) if is_int else float(value.strip())
            except ValueError:
                pass
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False) if network_reg_lrs is not None else None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    reg_dims = parse_kv_pairs(network_reg_dims, is_int=True) if network_reg_dims is not None else None

    network = TLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
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

    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    has_orthogonal = any("q_layer" in k for k in weights_sd.keys())

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key or "q_layer" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    tlora_min_rank = int(kwargs.get("tlora_min_rank", 1))
    tlora_alpha_rank_scale = float(kwargs.get("tlora_alpha_rank_scale", 1.0))
    tlora_max_timestep = float(kwargs.get("tlora_max_timestep", 1.0))
    tlora_init = "orthogonal" if has_orthogonal else "kaiming"
    tlora_sig_type = kwargs.get("tlora_sig_type", "last")

    if has_orthogonal:
        module_class = OrthogonalTLoRAModule
    else:
        module_class = TLoRAModule

    network = TLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        tlora_min_rank=tlora_min_rank,
        tlora_alpha_rank_scale=tlora_alpha_rank_scale,
        tlora_max_timestep=tlora_max_timestep,
        tlora_init=tlora_init,
        tlora_sig_type=tlora_sig_type,
    )
    return network, weights_sd
