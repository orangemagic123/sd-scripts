from types import SimpleNamespace

import pytest

from library.dataset import BaseDataset, BucketManager


def _dataset():
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.current_epoch = 1
    dataset.current_step = 0
    dataset.max_train_steps = 100
    dataset.replacements = {}
    dataset.protected_tags_cache = {}
    return dataset


def _subset(**overrides):
    values = {
        "shuffle_caption": False,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": "|||",
        "secondary_separator": None,
        "enable_wildcard": False,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "special_caption_tag_dropout_rate": 0.0,
        "caption_mode": "tags",
        "mixed_weights": {"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25},
        "protected_tags_file": None,
        "caption_prefix": None,
        "caption_suffix": None,
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_keep_tokens_separator_is_removed_without_augmentation_and_uses_comma_space():
    caption, debug_info = _dataset().process_caption(
        _subset(),
        "fixed_one, fixed_two|||flex_one,flex_two|||suffix",
    )

    assert caption == "fixed_one, fixed_two, flex_one, flex_two, suffix"
    assert "|||" not in caption
    assert debug_info == {"caption_mode": "tags", "dropped_tags": []}


def test_flex_dropout_keeps_fixed_and_protected_flex_tags(tmp_path):
    protected_tags_file = tmp_path / "protected_tags.txt"
    protected_tags_file.write_text("flex_keep\n", encoding="utf-8")
    subset = _subset(
        caption_tag_dropout_rate=1.0,
        protected_tags_file=str(protected_tags_file),
    )

    caption, debug_info = _dataset().process_caption(
        subset,
        "fixed|||flex_drop, flex_keep",
    )

    assert caption == "fixed, flex_keep"
    assert debug_info["dropped_tags"] == ["flex_drop"]


def test_special_dropout_only_drops_unprotected_fixed_tokens(tmp_path):
    protected_tags_file = tmp_path / "protected_tags.txt"
    protected_tags_file.write_text("fixed_keep\n", encoding="utf-8")
    subset = _subset(
        special_caption_tag_dropout_rate=1.0,
        protected_tags_file=str(protected_tags_file),
    )

    caption, debug_info = _dataset().process_caption(
        subset,
        "fixed_drop, fixed_keep|||flex_survives|||suffix_drop",
    )

    assert caption == "fixed_keep, flex_survives"
    assert debug_info["dropped_tags"] == ["fixed_drop", "suffix_drop"]


@pytest.mark.parametrize(
    ("selected_mode", "expected_caption"),
    [
        ("tags", "fixed, tag_a, tag_b"),
        ("nl", "fixed, a natural description"),
        ("tags_nl", "fixed, tag_a, tag_b, a natural description"),
        ("nl_tags", "fixed, a natural description, tag_a, tag_b"),
    ],
)
def test_mixed_caption_one_hot_weights_select_each_mode(selected_mode, expected_caption):
    weights = {"tags": 0, "nl": 0, "tags_nl": 0, "nl_tags": 0}
    weights[selected_mode] = 1
    subset = _subset(caption_mode="mixed", mixed_weights=weights)

    caption, debug_info = _dataset().process_caption(
        subset,
        "fixed|||tag_a, tag_b",
        "a natural description",
    )

    assert caption == expected_caption
    assert debug_info["caption_mode"] == selected_mode


def test_mixed_caption_all_zero_weights_fall_back_to_tags():
    subset = _subset(
        caption_mode="mixed",
        mixed_weights={"tags": 0, "nl": 0, "tags_nl": 0, "nl_tags": 0},
    )

    caption, debug_info = _dataset().process_caption(
        subset,
        "fixed|||tag_a, tag_b",
        "a natural description",
    )

    assert caption == "fixed, tag_a, tag_b"
    assert debug_info["caption_mode"] == "tags"


def test_bucket_no_upscale_switches_at_max_area_boundary():
    manager = BucketManager(
        no_upscale=True,
        max_reso=(1024, 1024),
        min_size=256,
        max_size=2048,
        reso_steps=64,
    )
    predefined_resos = [(512, 1024), (1024, 1024), (1024, 512), (1024, 256)]
    manager.set_predefined_resos(predefined_resos)

    below_reso, below_resized, below_error = manager.select_bucket(640, 640)
    equal_reso, equal_resized, equal_error = manager.select_bucket(2048, 512)
    above_reso, above_resized, above_error = manager.select_bucket(2048, 1024)

    assert below_reso == below_resized == (640, 640)
    assert below_reso not in predefined_resos
    assert equal_reso == equal_resized == (1024, 256)
    assert equal_reso in predefined_resos
    assert above_reso == above_resized == (1024, 512)
    assert above_reso in predefined_resos
    assert below_error == pytest.approx(0.0)
    assert equal_error == pytest.approx(0.0)
    assert above_error == pytest.approx(0.0)
