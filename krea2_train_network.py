"""LoRA training entry point for Krea 2 RAW."""

import argparse
import gc
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from library import (
    flux_train_utils,
    krea2_models,
    krea2_train_utils,
    krea2_utils,
    sampling,
    sd3_train_utils,
    strategy_krea2,
)
import library.args as args_util
import library.compile_utils as compile_utils
import library.model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
import train_network
from library.utils import setup_logging

setup_logging()

import logging

logger = logging.getLogger(__name__)


class Krea2NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks = False
        self._using_fp8_scaled = False
        self._requested_fp8_base = False
        self._requested_fp8_base_unet = False

    @staticmethod
    def _assert_deterministic_text_cache(dataset_group, label: str):
        datasets = getattr(dataset_group, "datasets", [dataset_group])
        for dataset in datasets:
            for subset in getattr(dataset, "subsets", []):
                if getattr(subset, "enable_wildcard", False):
                    raise ValueError(f"Krea 2 {label} text caching does not support enable_wildcard")
                if getattr(subset, "caption_dropout_every_n_epochs", 0) > 0:
                    raise ValueError(
                        f"Krea 2 {label} text caching does not support caption_dropout_every_n_epochs"
                    )
            if any(isinstance(value, list) for value in getattr(dataset, "replacements", {}).values()):
                raise ValueError(f"Krea 2 {label} text caching does not support random caption replacements")

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        if not args.pretrained_model_name_or_path:
            raise ValueError("--pretrained_model_name_or_path must point to the Krea 2 RAW DiT checkpoint")
        if not args.vae:
            raise ValueError("--vae must point to a Qwen-Image VAE checkpoint")
        if not args.text_encoder:
            raise ValueError("--text_encoder must point to a Qwen3-VL-4B-Instruct checkpoint")
        if args.mixed_precision != "bf16":
            raise ValueError("Krea 2 training requires --mixed_precision=bf16")

        if args.network_module is None:
            args.network_module = "networks.lora_krea2"
        if args.network_module != "networks.lora_krea2":
            raise ValueError("Krea 2 currently supports --network_module=networks.lora_krea2")
        if args.network_train_text_encoder_only:
            raise ValueError("Krea 2 does not support training Qwen3-VL")
        args.network_train_unet_only = True

        if args.cache_text_encoder_outputs_num_variants:
            raise ValueError("Krea 2 text caches do not support caption variants")
        if not args.cache_text_encoder_outputs:
            logger.warning("Krea 2 requires cached Qwen3-VL outputs; enabling --cache_text_encoder_outputs")
            args.cache_text_encoder_outputs = True
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
        if not train_dataset_group.is_text_encoder_output_cacheable():
            raise ValueError(
                "Krea 2 text caching cannot be combined with stochastic or step-dependent caption processing"
            )
        self._assert_deterministic_text_cache(train_dataset_group, "training")
        if val_dataset_group is not None and not val_dataset_group.is_text_encoder_output_cacheable():
            raise ValueError("The Krea 2 validation dataset must also be text-encoder-output cacheable")
        if val_dataset_group is not None:
            self._assert_deterministic_text_cache(val_dataset_group, "validation")
        if args.weighted_captions:
            raise ValueError("Krea 2 does not support --weighted_captions")

        requested_base_fp8 = bool(args.fp8_base or args.fp8_base_unet)
        if requested_base_fp8 != bool(args.fp8_scaled):
            raise ValueError("Krea 2 scaled fp8 must be enabled with both --fp8_base and --fp8_scaled")
        self._using_fp8_scaled = bool(args.fp8_scaled)
        self._requested_fp8_base = bool(args.fp8_base)
        self._requested_fp8_base_unet = bool(args.fp8_base_unet)
        if self._using_fp8_scaled:
            if args.base_weights:
                raise ValueError(
                    "--base_weights cannot be merged after scaled-fp8 quantization; merge them into a bf16 RAW checkpoint first"
                )
            # The Krea loader already quantizes only safe Linear weights. Prevent
            # NetworkTrainer's generic path from recasting the whole model.
            args.fp8_base = False
            args.fp8_base_unet = False

        if args.cpu_offload_checkpointing:
            raise ValueError("Krea 2 does not support --cpu_offload_checkpointing; use --blocks_to_swap")
        if args.blocks_to_swap is not None and not 0 <= args.blocks_to_swap <= 26:
            raise ValueError("--blocks_to_swap must be between 0 and 26 for Krea 2")
        if args.compile:
            if args.torch_compile:
                raise ValueError("--compile and --torch_compile cannot be enabled together")
            if args.compile_fullgraph and args.split_attn:
                raise ValueError("--compile_fullgraph cannot be combined with --split_attn")

        effective_attn_mode = "xformers" if args.xformers else "torch"
        if args.sdpa:
            effective_attn_mode = "torch"
        if args.attn_mode is not None:
            effective_attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode
        if effective_attn_mode == "xformers" and not args.split_attn:
            raise ValueError("Krea 2 grouped-query attention requires --split_attn with xformers")
        if effective_attn_mode == "sageattn":
            raise ValueError("SageAttention does not currently support training; use --sdpa, xformers, or flash")

        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)
        flux_train_utils.log_timestep_sampling_info(args)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = bool(args.blocks_to_swap and args.blocks_to_swap > 0)

        logger.info("Loading Qwen3-VL conditioner for Krea 2...")
        text_encoder = krea2_utils.load_krea2_text_encoder(
            args.text_encoder,
            dtype=torch.bfloat16,
            device="cpu",
            max_length=args.text_encoder_max_length,
            tokenizer_path=args.tokenizer_path,
            disable_mmap=args.disable_mmap_load_safetensors,
        )

        logger.info("Loading Qwen-Image VAE for Krea 2...")
        vae = krea2_train_utils.load_qwen_image_vae(
            args, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        vae.to(dtype=weight_dtype).eval().requires_grad_(False)
        return "krea2", [text_encoder], vae, None

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        # All prompt embeddings have now been cached. Release the 4B encoder
        # storage before allocating the much larger Krea 2 DiT.
        if args.cache_text_encoder_outputs and text_encoders:
            logger.info("Moving cached-only Qwen3-VL weights to meta to release host memory")
            text_encoders[0].to("meta")
            gc.collect()

        attn_mode = "xformers" if args.xformers else "torch"
        if args.sdpa:
            attn_mode = "torch"
        if args.attn_mode is not None:
            attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode

        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device
        model = krea2_utils.load_krea2_dit(
            args.pretrained_model_name_or_path,
            device=accelerator.device,
            dtype=weight_dtype,
            fp8_scaled=self._using_fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
            disable_mmap=args.disable_mmap_load_safetensors,
        )
        if self.is_swapping_blocks:
            logger.info(f"Enabling Krea 2 block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        return model, text_encoders

    def get_tokenize_strategy(self, args):
        return strategy_krea2.Krea2TokenizeStrategy()

    def get_tokenizers(self, tokenize_strategy):
        return []

    def get_latents_caching_strategy(self, args):
        return strategy_krea2.Krea2LatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_krea2.Krea2TextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        return strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            text_encoder_max_length=args.text_encoder_max_length,
            tokenizer_path=args.tokenizer_path,
            text_encoder_path=args.text_encoder,
        )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return None

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False] * len(text_encoders)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        original_vae_device = vae.device
        if not args.lowram:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

        encoder = text_encoders[0]
        logger.info("Moving Qwen3-VL to the accelerator for the Krea 2 text-cache pass")
        encoder.to(accelerator.device)
        dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

        if args.sample_prompts is not None and self.sample_prompts_te_outputs is None:
            logger.info(f"Caching Krea 2 sample-prompt embeddings: {args.sample_prompts}")
            prompt_cache = {}
            with torch.no_grad():
                for prompt_dict in sampling.load_prompts(args.sample_prompts):
                    for prompt in (prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")):
                        if prompt in prompt_cache:
                            continue
                        hidden, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [prompt])
                        hidden, mask = krea2_models.gather_valid_text(hidden, mask)
                        valid_length = int(mask[0].sum().item())
                        prompt_cache[prompt] = (
                            hidden[0, :valid_length].contiguous().cpu(),
                            mask[0, :valid_length].contiguous().cpu(),
                        )
            self.sample_prompts_te_outputs = prompt_cache

        accelerator.wait_for_everyone()
        encoder.to("cpu")
        if not args.lowram:
            vae.to(original_vae_device)
        clean_memory_on_device(accelerator.device)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        krea2_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            self.sample_prompts_te_outputs,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: krea2_models.SingleStreamDiT,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        if latents.ndim != 4:
            raise ValueError(f"Krea 2 expects 4D image latents, got {latents.shape}")

        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args,
            noise_scheduler,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
            return_timesteps_float32=True,
        )
        # Keep the architecture's 0..1000 index in float32. In particular,
        # scheduler-based `sigma` sampling has more precision than its bf16
        # noise-interpolation tensor.

        prompt_embeds, prompt_mask = text_encoder_conds[:2]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        prompt_mask = prompt_mask.to(accelerator.device, dtype=torch.bool)
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        dit = accelerator.unwrap_model(unet)
        image_tokens, positions, full_mask = krea2_models.prepare(
            noisy_model_input,
            prompt_embeds.shape[1],
            dit.config.patch,
            prompt_mask,
        )
        # Musubi offsets continuous custom sampling to the architecture's
        # 1..1000 training index, but scheduler-based `sigma` timesteps are
        # already in that range and must not receive a second offset.
        continuous_sampling = {"uniform", "sigmoid", "shift", "flux_shift", "krea2_shift"}
        if args.timestep_sampling in continuous_sampling:
            timesteps = timesteps + 1.0
        model_timesteps = timesteps / 1000.0
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(
                img=image_tokens,
                context=prompt_embeds,
                t=model_timesteps,
                pos=positions,
                mask=full_mask,
            )

        patch = dit.config.patch
        latent_height, latent_width = latents.shape[-2:]
        model_pred = rearrange(
            model_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_height // patch,
            w=latent_width // patch,
            ph=patch,
            pw=patch,
            c=dit.config.channels,
        )
        target = noise - latents
        weighting = flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return model_io.get_sai_model_spec_dataclass(
            None, args, False, True, False, krea2="raw"
        ).to_metadata_dict()

    def update_metadata(self, metadata, args):
        # assert_extra_args clears these generic switches so NetworkTrainer does
        # not cast the complete DiT. Preserve what the user actually requested.
        metadata["ss_fp8_base"] = self._requested_fp8_base
        metadata["ss_fp8_base_unet"] = self._requested_fp8_base_unet
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_fp8_scaled"] = self._using_fp8_scaled
        metadata["ss_krea2_text_encoder_max_length"] = args.text_encoder_max_length

    def is_text_encoder_not_needed_for_training(self, args):
        return True

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass

    def cast_text_encoder(self, args):
        return False

    def cast_unet(self, args):
        return not self._using_fp8_scaled

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if self.is_swapping_blocks:
            model = accelerator.prepare(unet, device_placement=[False])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()
        else:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)

        compile_utils.apply_cuda_optimizations(args)
        if args.compile:
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=self.is_swapping_blocks)
        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    krea2_train_utils.add_krea2_training_arguments(parser)
    parser.set_defaults(
        network_module="networks.lora_krea2",
        network_train_unet_only=True,
        weighting_scheme="none",
        model_prediction_type="raw",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.show_timesteps:
        flux_train_utils.show_timesteps(args)
    else:
        Krea2NetworkTrainer().train(args)
