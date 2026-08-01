Status: experimental

# Krea 2 LoRA and LyCORIS training with `krea2_train_network.py`

`krea2_train_network.py` trains LoRA and LyCORIS adapters for Krea 2 (K2) with the sd-scripts dataset and training pipeline. Krea 2 support is experimental and currently covers text-to-image training only; image editing, control inputs, and video training are not supported.

The recommended workflow is:

1. Train the LoRA on **Krea 2 RAW**.
2. Apply the resulting LoRA to **Krea 2 Turbo** for inference.

RAW is the undistilled, full-step training model. Turbo is the distilled, few-step inference model. This guide does not cover inference itself.

For options shared with other network trainers, see [LoRA training](./train_network.md) and [advanced network training](./train_network_advanced.md).

## Required models

Prepare local model files before starting:

| Component | Training option | Suggested source |
| --- | --- | --- |
| Krea 2 RAW DiT | `--pretrained_model_name_or_path` | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (`raw.safetensors`) |
| Qwen-Image VAE | `--vae` | [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) (`qwen_image_vae.safetensors`) |
| Qwen3-VL-4B-Instruct | `--text_encoder` | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL) (`qwen3vl_4b_bf16.safetensors`) |
| Krea 2 Turbo, optional for later inference | not used by this training script | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (`turbo.safetensors`) |

The DiT, VAE, and text encoder options expect compatible local checkpoints. Use the RAW DiT for training, not Turbo.

`--tokenizer_path` defaults to [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), which downloads and caches the tokenizer when needed. For offline use, set it to a complete local tokenizer directory instead. The tokenizer directory is separate from the single-file checkpoint passed through `--text_encoder`.

## Dataset requirements

Use the normal sd-scripts TOML dataset format described in the [dataset configuration guide](./config_README-en.md). Krea 2 uses a Qwen-Image VAE with 8x spatial compression followed by 2x2 DiT patches, so every training width and height must be divisible by **16**.

A minimal image dataset configuration looks like this:

```toml
[general]

[[datasets]]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_reso_steps = 16

[[datasets.subsets]]
image_dir = "/path/to/training_images"
caption_extension = ".txt"
num_repeats = 1
```

Only target images and captions are used. There are no control or reference-image inputs in the current Krea 2 integration.

## Integrated caching

Separate Krea 2 cache scripts are not used in sd-scripts. `krea2_train_network.py` performs the cache passes before training when these options are present:

- `--cache_latents`: encode images with the Qwen-Image VAE before training.
- `--cache_text_encoder_outputs`: encode captions with Qwen3-VL before training.
- `--cache_text_encoder_outputs_to_disk`: store the text cache on disk instead of keeping the complete cache in memory.

The text cache contains the selected Qwen3-VL hidden-state layers for valid, non-padding tokens. The trainable text-fusion layers remain inside the DiT; the Qwen3-VL model itself is not trained. Deterministic dataset processing such as caption prefix/suffix, separators, and fixed replacements is applied before encoding. Stochastic or epoch-dependent caption processing (wildcards, dropout, shuffling, token warmup, mixed captions, and random replacement choices) is rejected because one fixed embedding cannot represent it.

Disk caches record the processed-caption hash and the encoder, tokenizer, and maximum-length configuration. A changed caption or text-encoder configuration therefore rebuilds the affected cache unless `--skip_cache_check` is explicitly used.

## Basic training example

```bash
accelerate launch --num_cpu_threads_per_process 1 krea2_train_network.py \
  --pretrained_model_name_or_path="/models/krea2/raw.safetensors" \
  --vae="/models/qwen/qwen_image_vae.safetensors" \
  --text_encoder="/models/qwen/qwen3vl_4b_bf16.safetensors" \
  --tokenizer_path="/models/qwen/Qwen3-VL-4B-Instruct" \
  --dataset_config="/data/krea2_dataset.toml" \
  --output_dir="/output/krea2" \
  --output_name="krea2_style_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_krea2 \
  --network_dim=32 \
  --network_alpha=32 \
  --learning_rate=1e-4 \
  --optimizer_type=AdamW8bit \
  --lr_scheduler=constant \
  --mixed_precision=bf16 \
  --timestep_sampling=krea2_shift \
  --weighting_scheme=none \
  --sdpa \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --max_train_epochs=16 \
  --save_every_n_epochs=1 \
  --seed=42
```

The required Krea 2 settings in this example are:

- RAW checkpoint through `--pretrained_model_name_or_path`.
- Qwen-Image VAE and Qwen3-VL through `--vae` and `--text_encoder`.
- Optional local Qwen3-VL tokenizer directory through `--tokenizer_path`; omit it to use the default Hugging Face model ID.
- `--network_module=networks.lora_krea2` for the built-in LoRA, or `--network_module=lycoris.kohya` for LyCORIS.
- bf16 mixed precision.
- LoRA rank and alpha of 32 as the recommended starting point.
- `--timestep_sampling=krea2_shift --weighting_scheme=none`.

The default Krea 2 network targets all DiT `Linear` layers, including attention, MLP, projection, and text-fusion layers. Training settings are not yet considered final, so treat the command as a starting point and validate outputs for your dataset.

### LyCORIS

Install the pinned LyCORIS dependency through `requirements.txt`, then select its regular kohya module name. Krea 2 automatically routes it through an architecture adapter, so no custom LyCORIS preset is needed:

```toml
network_module = "lycoris.kohya"
network_dim = 32
network_alpha = 32
network_args = ["algo=lokr", "factor=8"]
```

The Krea 2 preset targets the same 264 DiT `Linear` layers as `networks.lora_krea2` and keeps standard `lora_unet_*` weight names. LoRA/LoCon, LoHa, and LoKr are covered by the Krea 2 tests. Other LyCORIS algorithms should be treated as experimental. `train_norm=True` is rejected because Krea 2 uses a custom RMSNorm, and `algo=tlora` is rejected because timestep-mask integration is not implemented. Qwen3-VL remains frozen for every network type.

If `network_args` contains a `preset=...` entry copied from another architecture, Krea 2 ignores it and uses its built-in target preset. Other algorithm-specific arguments, such as `factor`, `full_matrix`, `dora_wd`, dropout settings, and LoRA+ ratios, are passed through to LyCORIS.

## Timestep sampling and loss

Krea 2 is trained with flow matching. The noisy input and target are conceptually:

```text
x_t = (1 - t) * clean_latent + t * noise
target = noise - clean_latent
```

`krea2_shift` adjusts the timestep distribution for each sample according to its image-token count. Its resolution range matches Krea 2's RAW sampling schedule, from 256 px to 1280 px. This makes it preferable to a single fixed shift for multi-resolution buckets. `--weighting_scheme=none` applies ordinary mean-squared error without additional timestep weighting.

For a fixed 1024x1024 training resolution, `--timestep_sampling=shift --discrete_flow_shift=2.5` is a reasonable alternative, but `krea2_shift` is the recommended default in this guide.

## Attention

Krea 2 uses grouped-query attention with 48 query heads and 12 key/value heads. The Krea 2 implementation handles this head ratio for its supported attention backends.

- `--sdpa` is the recommended dependency-free setting. `--attn_mode=torch` selects the same PyTorch SDPA path when the explicit attention-mode option is used.
- xFormers and FlashAttention require their corresponding optional packages.
- Use `--split_attn` where required by the selected backend; it is especially important for xFormers with grouped-query attention.
- SageAttention is not currently available for training because its backward path is unsupported.

Do not replace the Krea-specific attention call with a wrapper that assumes the same number of query and key/value heads.

## Memory options

Krea 2 RAW is large. The following options can be combined according to available GPU and system memory:

- `--gradient_checkpointing`: recompute activations during backward to reduce VRAM use.
- `--blocks_to_swap=N`: offload main DiT blocks between CPU and GPU. `N` may be at most **26** because two of the 28 main blocks must remain resident.
- `--fp8_base --fp8_scaled`: enable Krea 2's scaled-fp8 base-weight path.

For Krea 2, plain `--fp8_base` without `--fp8_scaled` is invalid because normalization and modulation parameters must not be cast to unscaled fp8. Scaled fp8 is applied to the heavy main-block linear weights while sensitive components remain in bf16.

Block swapping saves VRAM but needs additional host memory and reduces throughput. Start with a modest value and increase it only when necessary.

## Sample images during training

The common RAW-model sample workflow is supported. Add a prompt file and sampling interval to the training command:

```text
--sample_prompts="/data/krea2_samples.txt" --sample_every_n_epochs=1
```

An example line in the prompt file is:

```text
A fox walking through fresh snow. --n low quality, blurry --w 1024 --h 1024 --s 28 --l 5.5 --d 0
```

Samples are generated with the same RAW checkpoint supplied through `--pretrained_model_name_or_path`. RAW normally needs classifier-free guidance, so provide a negative prompt and a guidance scale greater than 1.

Musubi Tuner's `--turbo_dit` weight-swapping feature for generating Turbo samples during RAW training is **not supported by the current sd-scripts integration**. Do not add `--turbo_dit` to this training command.

## Using the LoRA

Krea recommends training on RAW and applying the saved LoRA to Turbo for inference. Turbo is normally used with about eight steps, classifier-free guidance disabled, and its Turbo-specific timestep schedule. Consult the inference tool you use for its exact Krea 2 and LoRA options.

## License notice

The Krea 2 code is published under Apache-2.0, but the Krea 2 model weights and derived models are governed by the [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE). That license explicitly treats fine-tuned and merged models as derivatives, which includes LoRA training and distribution.

Review the current license before downloading weights, using a trained LoRA commercially, or distributing it. The Community License includes commercial-use thresholds, attribution and redistribution requirements, naming requirements for distributed derivatives, and deployment safety obligations. A separate Krea enterprise license may be required for uses outside those terms.
