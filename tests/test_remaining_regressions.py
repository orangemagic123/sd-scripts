import random
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from library import args as args_util, model_io
from library.dataset import (
    BaseDataset,
    BucketBatchIndex,
    ImageInfo,
    should_keep_vae_for_training,
)
from library.strategy_anima import AnimaTextEncoderOutputsCachingStrategy
from library.strategy_base import (
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from networks.ema_lora import _save_state_dict
from train_network import setup_parser


class _VariantCache:
    cache_to_disk = True
    batch_size = 2
    num_variants = 3
    is_partial = False

    def __init__(self):
        self.saved = {}
        self.validated = []

    def get_outputs_npz_path(self, image_abs_path):
        return f"{image_abs_path}.base.npz"

    def get_variant_outputs_npz_path(self, image_abs_path, variant_idx):
        return f"{image_abs_path}.v{variant_idx}.npz"

    def is_disk_cached_outputs_expected(self, npz_path):
        return npz_path in self.saved

    def is_disk_cached_outputs_expected_for_caption(self, npz_path, caption):
        self.validated.append((npz_path, caption))
        return self.saved.get(npz_path) == caption

    def cache_batch_outputs(
        self,
        tokenize_strategy,
        models,
        text_encoding_strategy,
        infos,
        captions=None,
        variant_idx=None,
    ):
        for info, caption in zip(infos, captions):
            path = self.get_variant_outputs_npz_path(
                info.absolute_path, variant_idx
            )
            self.saved[path] = caption


def test_caption_variants_are_stable_across_partial_rebuilds(monkeypatch):
    strategy = _VariantCache()
    monkeypatch.setattr(TokenizeStrategy, "_strategy", object())
    monkeypatch.setattr(TextEncodingStrategy, "_strategy", object())
    monkeypatch.setattr(
        TextEncoderOutputsCachingStrategy, "_strategy", strategy
    )

    dataset = BaseDataset((8, 8), 1.0, False, False)
    dataset.batch_size = 2
    dataset.seed = 12345

    subset = SimpleNamespace(options=("red", "green", "blue"))
    infos = [
        ImageInfo("first", 1, "first caption", False, "/dataset/first.png"),
        ImageInfo("second", 1, "second caption", False, "/dataset/second.png"),
    ]
    dataset.image_data = {info.image_key: info for info in infos}
    dataset.image_to_subset = {info.image_key: subset for info in infos}

    def process_caption(
        self, subset, caption, caption_nl, skip_caption_dropout=False
    ):
        suffix = f"{random.choice(subset.options)}-{random.randrange(1_000_000)}"
        return f"{caption}|{suffix}", {}

    dataset.process_caption = MethodType(process_caption, dataset)
    accelerator = SimpleNamespace(num_processes=1, process_index=0)

    random.seed(777)
    state_before = random.getstate()
    dataset.new_cache_text_encoder_outputs_variants(3, [object()], accelerator)
    assert random.getstate() == state_before
    original = dict(strategy.saved)
    assert len(original) == len(infos) * 3

    missing_path = strategy.get_variant_outputs_npz_path(
        infos[1].absolute_path, 1
    )
    del strategy.saved[missing_path]
    for _ in range(20):
        random.random()
    state_before = random.getstate()
    dataset.new_cache_text_encoder_outputs_variants(3, [object()], accelerator)

    assert random.getstate() == state_before
    assert strategy.saved == original
    assert strategy.validated

    infos[0].caption = "updated caption"
    dataset.new_cache_text_encoder_outputs_variants(3, [object()], accelerator)
    for variant_idx in range(3):
        path = strategy.get_variant_outputs_npz_path(
            infos[0].absolute_path, variant_idx
        )
        assert strategy.saved[path].startswith("updated caption|")


def _write_anima_cache(path, strategy, caption, *, include_identity):
    values = {
        "prompt_embeds": np.zeros((2, 4), dtype=np.float32),
        "attn_mask": np.ones((2,), dtype=np.int64),
        "t5_input_ids": np.ones((2,), dtype=np.int32),
        "t5_attn_mask": np.ones((2,), dtype=np.int32),
        "caption_dropout_rate": np.asarray(0.0, dtype=np.float32),
    }
    if include_identity:
        values["cache_version"] = np.asarray(
            strategy.CACHE_VERSION, dtype=np.int32
        )
        values["caption_sha256"] = np.asarray(
            strategy._caption_hash(caption)
        )
    np.savez(path, **values)


def test_anima_cache_tracks_the_encoded_caption(tmp_path):
    strategy = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, num_variants=2
    )
    cache_path = tmp_path / "caption_cache.npz"

    _write_anima_cache(
        cache_path, strategy, "old caption", include_identity=False
    )
    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))

    _write_anima_cache(
        cache_path, strategy, "old caption", include_identity=True
    )
    assert strategy.is_disk_cached_outputs_expected(str(cache_path))
    assert strategy.is_disk_cached_outputs_expected_for_caption(
        str(cache_path), "old caption"
    )
    assert not strategy.is_disk_cached_outputs_expected_for_caption(
        str(cache_path), "new caption"
    )


def test_posthoc_ema_save_recomputes_safetensors_hashes(tmp_path):
    output = tmp_path / "ema.safetensors"
    state_dict = {
        "lora_unet_test.lora_down.weight": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 3),
        "lora_unet_test.alpha": torch.tensor(2.0),
    }
    metadata = {
        "ss_network_module": "networks.lora",
        "sshs_model_hash": "stale-model-hash",
        "sshs_legacy_hash": "stale-legacy-hash",
    }

    _save_state_dict(
        str(output),
        {key: value.clone() for key, value in state_dict.items()},
        torch.float16,
        metadata,
    )

    saved_tensors = load_file(output)
    with safe_open(output, framework="pt", device="cpu") as handle:
        saved_metadata = handle.metadata()

    expected_model_hash, expected_legacy_hash = (
        model_io.precalculate_safetensors_hashes(
            saved_tensors, saved_metadata
        )
    )
    assert saved_metadata["sshs_model_hash"] == expected_model_hash
    assert saved_metadata["sshs_legacy_hash"] == expected_legacy_hash
    assert saved_metadata["sshs_model_hash"] != "stale-model-hash"
    assert saved_metadata["sshs_legacy_hash"] != "stale-legacy-hash"
    assert metadata["sshs_model_hash"] == "stale-model-hash"


def test_cached_latents_still_build_inpainting_inputs(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(64, 128, 192)).save(image_path)

    dataset = BaseDataset((8, 8), 1.0, True, False)
    dataset.enable_bucket = True
    dataset.batch_size = 1
    dataset.prior_loss_weight = 1.0
    dataset.bucket_manager = SimpleNamespace(buckets=[["image"]])
    dataset.buckets_indices = [BucketBatchIndex(0, 1, 0)]
    dataset._length = 1
    dataset.text_encoder_output_caching_strategy = SimpleNamespace(
        is_partial=False, num_variants=0
    )
    dataset.random_mask = lambda size: Image.new("L", size, color=255)

    subset = SimpleNamespace(
        custom_attributes={},
        flip_aug=False,
        alpha_mask=False,
        random_crop=False,
        color_aug=False,
    )
    info = ImageInfo("image", 1, "caption", False, str(image_path))
    info.bucket_reso = (8, 8)
    info.resized_size = (8, 8)
    info.latents = torch.zeros(4, 1, 1)
    info.latents_original_size = (8, 8)
    info.latents_crop_ltrb = (0, 0, 0, 0)
    info.text_encoder_outputs = [torch.zeros(1, 2)]

    dataset.image_data = {"image": info}
    dataset.image_to_subset = {"image": subset}

    example = dataset[0]

    assert example["images"] is None
    assert example["latents"].shape == (1, 4, 1, 1)
    assert example["masks"].shape == (1, 1, 8, 8)
    assert example["masked_images"].shape == (1, 3, 8, 8)
    assert torch.count_nonzero(example["masked_images"]) == 0


def test_inpainting_keeps_vae_available_with_cached_latents():
    assert should_keep_vae_for_training(False, False)
    assert should_keep_vae_for_training(False, True)
    assert should_keep_vae_for_training(True, True)
    assert not should_keep_vae_for_training(True, False)


def test_training_args_allow_inpainting_with_latent_cache():
    args = setup_parser().parse_args(
        ["--train_inpainting", "--cache_latents_to_disk"]
    )

    args_util.verify_training_args(args)

    assert args.train_inpainting
    assert args.cache_latents
    assert args.cache_latents_to_disk
