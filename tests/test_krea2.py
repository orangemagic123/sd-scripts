import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from library.flux_train_utils import get_noisy_model_input_and_timesteps
from library.krea2_models import SingleMMDiTConfig, SingleStreamDiT, gather_valid_text
from library.dataset import BaseDataset
from library.krea2_utils import _convert_comfyui_qwen3vl_state_dict, load_krea2_dit
from library.sai_model_spec import build_metadata_dataclass
from library.strategy_base import TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy
from library.strategy_krea2 import (
    Krea2TextEncoderOutputsCachingStrategy,
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
)
from networks import lora_krea2, lycoris_krea2


def _make_text_tokens(sequence_length: int, *, layers: int = 2, dim: int = 4, dtype=torch.float32):
    """Make tokens whose value identifies their original sequence position."""
    positions = torch.arange(1, sequence_length + 1, dtype=dtype).view(sequence_length, 1, 1)
    return positions * torch.ones(sequence_length, layers, dim, dtype=dtype)


def _tiny_config(*, layers: int = 1) -> SingleMMDiTConfig:
    # A 16-wide attention head is the smallest useful Krea2 RoPE configuration:
    # it splits into the model's three even-sized axes as [4, 6, 6].
    return SingleMMDiTConfig(
        features=32,
        tdim=8,
        txtdim=32,
        heads=2,
        kvheads=1,
        multiplier=1,
        layers=layers,
        patch=2,
        channels=2,
        txtlayers=2,
        txtheads=2,
        txtkvheads=1,
    )


def test_gather_valid_text_compacts_interior_padding():
    tokens = _make_text_tokens(5).unsqueeze(0)
    mask = torch.tensor([[True, True, False, False, True]])

    gathered, gathered_mask = gather_valid_text(tokens, mask)

    assert torch.equal(gathered_mask, torch.tensor([[True, True, True]]))
    assert torch.equal(gathered, tokens[:, [0, 1, 4]])


def test_gather_valid_text_trims_an_already_compact_prefix():
    tokens = _make_text_tokens(4).unsqueeze(0)
    mask = torch.tensor([[True, True, False, False]])

    gathered, gathered_mask = gather_valid_text(tokens, mask)

    assert gathered.shape == (1, 2, 2, 4)
    assert torch.equal(gathered_mask, torch.tensor([[True, True]]))
    assert torch.equal(gathered, tokens[:, :2])


def test_gather_valid_text_pads_mixed_length_batches_after_compaction():
    tokens = torch.stack([_make_text_tokens(5), _make_text_tokens(5) + 10])
    mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, False, True, False, True],
        ]
    )

    gathered, gathered_mask = gather_valid_text(tokens, mask)

    assert gathered.shape == (2, 3, 2, 4)
    assert torch.equal(
        gathered_mask,
        torch.tensor(
            [
                [True, False, False],
                [True, True, True],
            ]
        ),
    )
    assert torch.equal(gathered[0, 0], tokens[0, 0])
    assert torch.count_nonzero(gathered[0, 1:]) == 0
    assert torch.equal(gathered[1], tokens[1, [0, 2, 4]])


def test_gather_valid_text_preserves_dtype_device_and_bool_mask():
    tokens = _make_text_tokens(3, dtype=torch.bfloat16).unsqueeze(0)
    mask = torch.tensor([[True, False, True]], dtype=torch.bool, device=tokens.device)

    gathered, gathered_mask = gather_valid_text(tokens, mask)

    assert gathered.dtype == torch.bfloat16
    assert gathered.device == tokens.device
    assert gathered_mask.dtype == torch.bool
    assert gathered_mask.device == mask.device
    assert torch.equal(gathered, tokens[:, [0, 2]])


class _NoiseScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)


@pytest.mark.parametrize(
    ("image_resolution", "expected_mu"),
    [
        (256, 0.5),
        (1024, 0.90625),
        (1280, 1.15),
    ],
)
def test_krea2_shift_matches_the_resolution_aware_schedule(image_resolution, expected_mu):
    args = SimpleNamespace(
        timestep_sampling="krea2_shift",
        sigmoid_scale=1.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        min_timestep=None,
        max_timestep=None,
    )
    latent_side = image_resolution // 8
    latents = torch.zeros(1, 4, latent_side, latent_side)
    noise = torch.ones_like(latents)

    # A zero normal sample becomes q=0.5 after sigmoid. This isolates the
    # resolution-dependent shift from the random timestep draw.
    with patch("library.flux_train_utils.torch.randn", return_value=torch.zeros(1)):
        noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            args,
            _NoiseScheduler(),
            latents,
            noise,
            device="cpu",
            dtype=torch.float32,
        )

    expected_shift = math.exp(expected_mu)
    expected_sigma = expected_shift / (1.0 + expected_shift)
    assert sigmas.shape == (1, 1, 1, 1)
    assert sigmas.item() == pytest.approx(expected_sigma, rel=1e-6)
    assert noisy_input[0, 0, 0, 0].item() == pytest.approx(expected_sigma, rel=1e-6)
    assert timesteps.item() == pytest.approx(1000 * expected_sigma, rel=1e-6)

    if image_resolution == 1024:
        # The reported regression value is about 2.47487; exact interpolation
        # of mu=0.90625 yields 2.47502.
        assert expected_shift == pytest.approx(2.47487, abs=2e-4)


def test_krea2_shift_honors_a_fixed_timestep_range():
    args = SimpleNamespace(
        timestep_sampling="krea2_shift",
        sigmoid_scale=1.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        min_timestep=500,
        max_timestep=500,
    )
    latents = torch.zeros(1, 4, 16, 16)
    noise = torch.ones_like(latents)

    with patch("library.flux_train_utils.torch.randn", return_value=torch.tensor([4.0])):
        noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            args,
            _NoiseScheduler(),
            latents,
            noise,
            device="cpu",
            dtype=torch.float32,
        )

    assert timesteps.item() == pytest.approx(500.0)
    assert sigmas.item() == pytest.approx(0.5)
    assert noisy_input[0, 0, 0, 0].item() == pytest.approx(0.5)


def test_tiny_single_stream_dit_gqa_forward_and_backward_on_cpu():
    torch.manual_seed(0)
    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch").train()

    image_tokens = torch.randn(1, 4, config.channels * config.patch**2, requires_grad=True)
    text_tokens = torch.randn(1, 3, config.txtlayers, config.txtdim, requires_grad=True)
    timesteps = torch.tensor([0.5])
    positions = torch.zeros(1, 7, 3)
    positions[0, :4, 1:] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Image tokens are always valid. Text padding is a tail so the combined
    # image-first sequence remains a contiguous valid prefix.
    mask = torch.tensor([[True, True, True, True, True, True, False]])

    output = model(image_tokens, text_tokens, timesteps, positions, mask)

    assert output.shape == image_tokens.shape
    assert torch.isfinite(output).all()

    main_attention = model.blocks[0].attn
    text_attention = model.txtfusion.layerwise_blocks[0].attn
    for attention in (main_attention, text_attention):
        assert attention.heads == 2
        assert attention.kvheads == 1
        assert attention.wq.weight.shape == (32, 32)
        assert attention.wk.weight.shape == (16, 32)
        assert attention.wv.weight.shape == (16, 32)

    output.square().mean().backward()
    assert image_tokens.grad is not None
    assert text_tokens.grad is not None
    assert torch.isfinite(image_tokens.grad).all()
    assert torch.isfinite(text_tokens.grad).all()


def test_krea2_lora_wraps_every_linear_but_no_raw_norm_or_modulation_parameter():
    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch")
    linear_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}

    # Krea2 has 40 fixed Linears (text fusion and projections) plus eight
    # attention/MLP Linears per main block: 40 + 8*28 == 264 in the full model.
    assert len(linear_names) == 40 + 8 * config.layers
    assert 40 + 8 * 28 == 264

    network = lora_krea2.create_network(1.0, 4, 4, None, [], model)
    expected_lora_names = {"lora_unet_" + name.replace(".", "_") for name in linear_names}
    actual_lora_names = {module.lora_name for module in network.unet_loras}

    assert actual_lora_names == expected_lora_names
    assert all(isinstance(module.org_module, nn.Linear) for module in network.unet_loras)

    raw_parameter_names = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(".scale") or name.endswith(".mod.lin") or name.endswith(".modulation.lin")
    }
    assert raw_parameter_names
    assert not {
        "lora_unet_" + name.replace(".", "_") for name in raw_parameter_names
    } & actual_lora_names


@pytest.mark.parametrize(
    ("algo", "module_class_name"),
    [
        ("lora", "LoConModule"),
        ("loha", "LohaModule"),
        ("lokr", "LokrModule"),
    ],
)
def test_krea2_lycoris_wraps_every_linear_with_checkpoint_compatible_names(algo, module_class_name):
    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch")
    linear_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}

    # The cached Qwen3-VL object can already be on meta here; the Krea 2
    # wrapper must not traverse or adapt it.
    network = lycoris_krea2.create_network(1.0, 4, 4, None, [object()], model, algo=algo)
    expected_names = {"lora_unet_" + name.replace(".", "_") for name in linear_names}
    actual_names = {module.lora_name for module in network.unet_loras}

    assert actual_names == expected_names
    assert not network.text_encoder_loras
    assert {type(module).__name__ for module in network.unet_loras} == {module_class_name}
    assert not any(name.startswith("lora_unet__") for name in actual_names)


@pytest.mark.parametrize("algo", ["lora", "loha", "lokr"])
def test_krea2_lycoris_forward_backward_and_weight_restore(algo):
    torch.manual_seed(0)
    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch").train()
    network = lycoris_krea2.create_network(1.0, 2, 2, None, [], model, algo=algo)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)

    image_tokens = torch.randn(1, 4, config.channels * config.patch**2)
    text_tokens = torch.randn(1, 3, config.txtlayers, config.txtdim)
    positions = torch.zeros(1, 7, 3)
    mask = torch.tensor([[True, True, True, True, True, True, False]])
    output = model(image_tokens, text_tokens, torch.tensor([0.5]), positions, mask)
    output.square().mean().backward()

    adapter_grads = [parameter.grad for parameter in network.parameters() if parameter.requires_grad]
    assert adapter_grads
    assert any(grad is not None and torch.isfinite(grad).all() for grad in adapter_grads)

    weights_sd = {key: value.detach().clone() for key, value in network.state_dict().items()}
    restored_model = SingleStreamDiT(config, attn_mode="torch")
    restored, returned_weights = lycoris_krea2.create_network_from_weights(
        1.0,
        None,
        None,
        [],
        restored_model,
        weights_sd=weights_sd,
    )

    expected_names = {module.lora_name for module in network.unet_loras}
    assert {module.lora_name for module in restored.unet_loras} == expected_names
    assert returned_weights is weights_sd


@pytest.mark.parametrize("algo", ["lora", "loha", "lokr"])
def test_krea2_checkpoint_loader_merges_lycoris_weights(tmp_path, algo):
    torch.manual_seed(0)
    config = _tiny_config()
    base_model = SingleStreamDiT(config, attn_mode="torch")
    base_state = {key: value.detach().contiguous().clone() for key, value in base_model.state_dict().items()}
    checkpoint = tmp_path / f"tiny_krea2_{algo}.safetensors"
    save_file(base_state, checkpoint)

    network = lycoris_krea2.create_network(1.0, 2, 2, None, [], base_model, algo=algo)
    network.apply_to([], base_model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for parameter in network.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.05)
    weights_sd = {key: value.detach().contiguous().clone() for key, value in network.state_dict().items()}

    loaded = load_krea2_dit(
        str(checkpoint),
        device="cpu",
        dtype=torch.float32,
        config=config,
        lora_weights=[weights_sd],
        lora_multipliers=[1.0],
        disable_mmap=True,
    )

    linear_weight_keys = [f"{name}.weight" for name, module in loaded.named_modules() if isinstance(module, nn.Linear)]
    assert any(not torch.equal(loaded.state_dict()[key], base_state[key]) for key in linear_weight_keys)
    assert torch.equal(loaded.state_dict()["blocks.0.prenorm.scale"], base_state["blocks.0.prenorm.scale"])


def test_krea2_resolves_the_public_lycoris_module_name_to_its_architecture_adapter():
    from krea2_train_network import Krea2NetworkTrainer, resolve_krea2_network_module

    assert resolve_krea2_network_module(None) == "networks.lora_krea2"
    assert resolve_krea2_network_module("networks.lora_krea2") == "networks.lora_krea2"
    assert resolve_krea2_network_module("lycoris.kohya") == "lycoris.kohya"
    assert resolve_krea2_network_module("networks.lycoris_krea2") == "networks.lycoris_krea2"
    trainer = Krea2NetworkTrainer()
    assert trainer.get_network_module_name(SimpleNamespace(network_module="lycoris.kohya")) == "networks.lycoris_krea2"
    with pytest.raises(ValueError, match="lycoris.kohya"):
        resolve_krea2_network_module("networks.lora")


def test_qwen3vl_comfyui_and_hf_state_dict_key_conversion():
    state_dict = {
        "model.language_model.layers.0.weight": torch.tensor([0.0]),
        "model.visual.patch_embed.weight": torch.tensor([1.0]),
        "visual.blocks.0.weight": torch.tensor([2.0]),
        "language_model.layers.1.weight": torch.tensor([3.0]),
        "model.layers.2.weight": torch.tensor([4.0]),
        "lm_head.weight": torch.tensor([5.0]),
    }

    converted = _convert_comfyui_qwen3vl_state_dict(state_dict)

    expected_keys = {
        "model.language_model.layers.0.weight",
        "model.visual.patch_embed.weight",
        "model.visual.blocks.0.weight",
        "model.language_model.layers.1.weight",
        "model.language_model.layers.2.weight",
        "lm_head.weight",
    }
    assert set(converted) == expected_keys
    assert converted["model.language_model.layers.0.weight"] is state_dict["model.language_model.layers.0.weight"]
    assert converted["model.visual.patch_embed.weight"] is state_dict["model.visual.patch_embed.weight"]
    assert converted["model.visual.blocks.0.weight"] is state_dict["visual.blocks.0.weight"]
    assert converted["model.language_model.layers.1.weight"] is state_dict["language_model.layers.1.weight"]
    assert converted["model.language_model.layers.2.weight"] is state_dict["model.layers.2.weight"]
    assert converted["lm_head.weight"] is state_dict["lm_head.weight"]


def test_krea2_text_cache_round_trips_bfloat16_and_compacts_padding(tmp_path):
    cache_path = tmp_path / "sample_krea2_te.npz"
    strategy = Krea2TextEncoderOutputsCachingStrategy(True, 1, False)

    class Encoder:
        def __call__(self, captions):
            tokens = _make_text_tokens(5, layers=12, dim=2560, dtype=torch.bfloat16).unsqueeze(0)
            return tokens, torch.tensor([[True, False, True, False, True]])

    info = SimpleNamespace(caption="a prompt", text_encoder_outputs_npz=str(cache_path))
    strategy.cache_batch_outputs(None, [Encoder()], None, [info])

    assert strategy.is_disk_cached_outputs_expected(str(cache_path))
    assert strategy.is_disk_cached_outputs_expected_for_caption(str(cache_path), "a prompt")
    assert not strategy.is_disk_cached_outputs_expected_for_caption(str(cache_path), "changed prompt")
    embed, mask = strategy.load_outputs_npz(str(cache_path))
    assert embed.dtype == torch.bfloat16
    assert torch.equal(embed, _make_text_tokens(5, layers=12, dim=2560, dtype=torch.bfloat16)[[0, 2, 4]])
    assert torch.equal(mask, torch.ones(3, dtype=torch.bool))

    with np.load(cache_path) as cached:
        assert cached["vl_embed"].dtype == np.uint16


def test_krea2_cache_pass_encodes_the_processed_caption():
    strategy = Krea2TextEncoderOutputsCachingStrategy(False, 1, False)
    captured = {}

    class Encoder:
        def __call__(self, captions):
            captured["captions"] = captions
            batch_size = len(captions)
            hidden = torch.zeros(batch_size, 2, 12, 2560, dtype=torch.bfloat16)
            return hidden, torch.ones(batch_size, 2, dtype=torch.bool)

    subset = SimpleNamespace(
        caption_prefix="prefix",
        caption_suffix="suffix",
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        enable_wildcard=False,
        shuffle_caption=False,
        token_warmup_step=0,
        token_warmup_min=1,
        caption_tag_dropout_rate=0.0,
        special_caption_tag_dropout_rate=0.0,
        keep_tokens_separator=None,
        keep_tokens=0,
        caption_separator=",",
        caption_mode="tags",
        secondary_separator=";",
        protected_tags_file=None,
    )
    info = SimpleNamespace(
        image_key="image",
        absolute_path="image.png",
        caption="red; green",
        caption_nl=None,
        text_encoder_outputs=None,
    )
    dataset = object.__new__(BaseDataset)
    dataset.batch_size = 1
    dataset.image_data = {info.image_key: info}
    dataset.image_to_subset = {info.image_key: subset}
    dataset.replacements = {"red": "blue"}
    dataset.current_epoch = 1
    dataset.current_step = 0
    dataset.max_train_steps = 100
    dataset.protected_tags_cache = {}

    accelerator = SimpleNamespace(num_processes=1, process_index=0)
    with (
        patch.object(TextEncoderOutputsCachingStrategy, "_strategy", strategy),
        patch.object(TokenizeStrategy, "_strategy", Krea2TokenizeStrategy()),
        patch.object(TextEncodingStrategy, "_strategy", Krea2TextEncodingStrategy()),
    ):
        dataset.new_cache_text_encoder_outputs([Encoder()], accelerator)

    assert captured["captions"] == ["prefix blue, green suffix"]
    assert info.caption == "red; green"
    assert info.text_encoder_outputs[0].dtype == torch.bfloat16


def test_krea2_text_cache_builds_independent_caption_variant_files(tmp_path):
    strategy = Krea2TextEncoderOutputsCachingStrategy(True, 1, False, num_variants=2)
    captured_captions = []

    class Encoder:
        def __call__(self, captions):
            captured_captions.append(list(captions))
            value = len(captured_captions)
            hidden = torch.full((len(captions), 2, 12, 2560), value, dtype=torch.bfloat16)
            return hidden, torch.ones(len(captions), 2, dtype=torch.bool)

    image_path = tmp_path / "sample.png"
    info = SimpleNamespace(
        image_key="image",
        absolute_path=str(image_path),
        caption="original",
        caption_nl=None,
        text_encoder_outputs=None,
        text_encoder_outputs_npz=None,
    )
    dataset = object.__new__(BaseDataset)
    dataset.batch_size = 1
    dataset.image_data = {info.image_key: info}
    dataset.image_to_subset = {info.image_key: SimpleNamespace()}

    accelerator = SimpleNamespace(num_processes=1, process_index=0)
    with (
        patch.object(TextEncoderOutputsCachingStrategy, "_strategy", strategy),
        patch.object(TokenizeStrategy, "_strategy", Krea2TokenizeStrategy()),
        patch.object(TextEncodingStrategy, "_strategy", Krea2TextEncodingStrategy()),
        patch.object(
            dataset,
            "process_caption",
            side_effect=[("first variant", {}), ("second variant", {})],
        ),
    ):
        dataset.new_cache_text_encoder_outputs_variants(2, [Encoder()], accelerator)

    first_path = strategy.get_variant_outputs_npz_path(str(image_path), 0)
    second_path = strategy.get_variant_outputs_npz_path(str(image_path), 1)
    assert strategy.num_variants == 2
    assert captured_captions == [["first variant"], ["second variant"]]
    assert strategy.is_disk_cached_outputs_expected_for_caption(first_path, "first variant")
    assert strategy.is_disk_cached_outputs_expected_for_caption(second_path, "second variant")
    assert torch.all(strategy.load_outputs_npz(first_path)[0] == 1)
    assert torch.all(strategy.load_outputs_npz(second_path)[0] == 2)


def test_krea2_trainer_enables_disk_cache_for_stochastic_caption_variants():
    from krea2_train_network import Krea2NetworkTrainer

    subset = SimpleNamespace(
        token_warmup_step=0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        shuffle_caption=True,
        caption_tag_dropout_rate=0.1,
        special_caption_tag_dropout_rate=0.1,
        caption_mode="mixed",
        enable_wildcard=True,
    )

    class DatasetGroup:
        datasets = [SimpleNamespace(subsets=[subset], replacements={"color": ["red", "blue"]})]

        @staticmethod
        def is_text_encoder_output_cacheable():
            return False

        @staticmethod
        def verify_bucket_reso_steps(_steps):
            pass

    args = SimpleNamespace(
        pretrained_model_name_or_path="dit.safetensors",
        vae="vae.safetensors",
        text_encoder="text_encoder.safetensors",
        mixed_precision="bf16",
        network_module="networks.lora_krea2",
        network_train_text_encoder_only=False,
        network_train_unet_only=False,
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs_num_variants=10,
        weighted_captions=False,
        fp8_base=False,
        fp8_base_unet=False,
        fp8_scaled=False,
        cpu_offload_checkpointing=False,
        blocks_to_swap=None,
        compile=False,
        xformers=False,
        sdpa=True,
        attn_mode=None,
        split_attn=False,
    )

    with patch("krea2_train_network.flux_train_utils.log_timestep_sampling_info"):
        Krea2NetworkTrainer().assert_extra_args(args, DatasetGroup(), None)

    assert args.cache_text_encoder_outputs
    assert args.cache_text_encoder_outputs_to_disk
    assert args.network_train_unet_only


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("token_warmup_step", 1, "token_warmup_step"),
        ("caption_dropout_rate", 0.1, "caption_dropout_rate"),
        ("caption_dropout_every_n_epochs", 2, "caption_dropout_every_n_epochs"),
    ],
)
def test_krea2_caption_variants_reject_processing_that_the_pool_cannot_represent(
    option, value, message
):
    from krea2_train_network import Krea2NetworkTrainer

    subset = SimpleNamespace(
        token_warmup_step=0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
    )
    setattr(subset, option, value)
    dataset_group = SimpleNamespace(datasets=[SimpleNamespace(subsets=[subset])])

    with pytest.raises(ValueError, match=message):
        Krea2NetworkTrainer._assert_variant_text_cache(dataset_group, "training")


def test_tiny_krea2_checkpoint_loader_assigns_exact_weights(tmp_path):
    config = _tiny_config()
    original = SingleStreamDiT(config, attn_mode="torch")
    checkpoint = tmp_path / "tiny_krea2.safetensors"
    save_file({key: value.detach().contiguous() for key, value in original.state_dict().items()}, checkpoint)

    loaded = load_krea2_dit(
        str(checkpoint),
        device="cpu",
        dtype=torch.float32,
        config=config,
        disable_mmap=True,
    )

    assert original.state_dict().keys() == loaded.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key


def test_krea2_modelspec_metadata_has_flow_architecture_without_epsilon_prediction():
    metadata = build_metadata_dataclass(
        state_dict=None,
        v2=False,
        v_parameterization=False,
        sdxl=False,
        lora=True,
        textual_inversion=False,
        timestamp=0,
        model_config={"krea2": "raw"},
    )

    assert metadata.architecture == "Krea-2/lora"
    assert metadata.implementation == "https://github.com/krea-ai/krea-2"
    assert metadata.resolution == "1024x1024"
    assert metadata.prediction_type is None


def test_trainer_reconstructs_full_precision_timestep_from_sigma():
    from krea2_train_network import Krea2NetworkTrainer

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return torch.autocast("cpu", dtype=torch.bfloat16)

        @staticmethod
        def unwrap_model(model):
            return model

    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch")
    captured = {}
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        captured["t"] = kwargs["t"].detach().clone()
        return original_forward(*args, **kwargs)

    model.forward = capture_forward
    args = SimpleNamespace(
        timestep_sampling="krea2_shift",
        sigmoid_scale=1.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
        weighting_scheme="none",
    )
    latents = torch.randn(1, config.channels, 4, 4, dtype=torch.bfloat16)
    text = torch.randn(1, 3, config.txtlayers, config.txtdim, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False]])

    _, target, timesteps, weighting = Krea2NetworkTrainer().get_noise_pred_and_target(
        args,
        Accelerator(),
        _NoiseScheduler(),
        latents,
        {},
        [text, mask],
        model,
        None,
        torch.bfloat16,
        True,
    )

    assert timesteps.dtype == torch.float32
    assert torch.allclose(captured["t"], timesteps / 1000.0)
    assert torch.all(timesteps >= 1.0)
    assert target.shape == latents.shape
    assert weighting.shape == (1, 1, 1, 1)


def test_trainer_sigma_sampling_does_not_add_a_second_timestep_offset():
    from krea2_train_network import Krea2NetworkTrainer

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return torch.autocast("cpu", dtype=torch.bfloat16)

        @staticmethod
        def unwrap_model(model):
            return model

    config = _tiny_config()
    model = SingleStreamDiT(config, attn_mode="torch")
    captured = {}
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        captured["t"] = kwargs["t"].detach().clone()
        return original_forward(*args, **kwargs)

    model.forward = capture_forward
    scheduler = _NoiseScheduler()
    scheduler.timesteps = torch.arange(1000, 0, -1, dtype=torch.float32)
    scheduler.sigmas = scheduler.timesteps / 1000.0
    args = SimpleNamespace(
        timestep_sampling="sigma",
        weighting_scheme="none",
        logit_mean=None,
        logit_std=None,
        mode_scale=None,
        min_timestep=500,
        max_timestep=500,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
    )
    latents = torch.randn(1, config.channels, 4, 4, dtype=torch.bfloat16)
    text = torch.randn(1, 3, config.txtlayers, config.txtdim, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False]])

    _, _, timesteps, _ = Krea2NetworkTrainer().get_noise_pred_and_target(
        args,
        Accelerator(),
        scheduler,
        latents,
        {},
        [text, mask],
        model,
        None,
        torch.bfloat16,
        True,
    )

    assert timesteps.item() == pytest.approx(0.5 * 1000.0)
    assert captured["t"].item() == pytest.approx(0.5)


def test_shared_helper_can_preserve_float32_scheduler_timesteps_with_bfloat16_noise():
    scheduler = _NoiseScheduler()
    scheduler.timesteps = torch.tensor([999.125, 500.375], dtype=torch.float32)
    scheduler.sigmas = scheduler.timesteps / 1000.0
    args = SimpleNamespace(
        timestep_sampling="sigma",
        weighting_scheme="none",
        logit_mean=None,
        logit_std=None,
        mode_scale=None,
        min_timestep=0,
        max_timestep=0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
    )
    latents = torch.zeros(1, 2, 2, 2, dtype=torch.bfloat16)
    noise = torch.ones_like(latents)

    _, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        args,
        scheduler,
        latents,
        noise,
        device="cpu",
        dtype=torch.bfloat16,
        return_timesteps_float32=True,
    )

    assert timesteps.dtype == torch.float32
    assert timesteps.item() == pytest.approx(999.125)
    assert sigmas.dtype == torch.bfloat16
