"""MOSS JSON captions (caption_format = "moss_json")."""

import collections
import json
import random
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from library import moss_image_utils
from library.config_util import DreamBoothSubsetParams
from library.dataset import BaseDataset, BucketBatchIndex, ImageInfo
from library.dreambooth_dataset import DreamBoothDataset
from library.moss_json_caption import (
    MossCaptionChoice,
    get_choices,
    normalize_weights,
    parse_moss_caption,
    parse_moss_caption_text,
    read_moss_caption_text,
    render,
)
from library.strategy_base import TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy
from library.strategy_moss_image import (
    MossImageTextEncoderOutputsCachingStrategy,
    MossImageTextEncodingStrategy,
    MossImageTokenizeStrategy,
)
from library.subset import DreamBoothSubset

CAPTION = {
    "caption": {"short": "A girl.", "medium": "A girl in a garden.", "long": "A girl in a garden,\nfacing the viewer."},
    "tags": ["solo", "smile", "mychar", "smile", "outdoors"],
    "keep_tags": ["mychar", "1girl"],
    "watermark": ["logo"],
    "borders": ["none"],
}
CAPTION_TEXT = json.dumps(CAPTION)
NO_WEIGHT = {"tags": 0, "nl": 0, "tags_nl": 0, "nl_tags": 0}


def moss_subset(**overrides):
    params = dict(image_dir="images", caption_format="moss_json")
    params.update(overrides)
    return DreamBoothSubset(**asdict(DreamBoothSubsetParams(**params)))


def make_dataset():
    ds = BaseDataset((64, 64), 1.0, False, False)
    ds.batch_size = 1
    ds.prior_loss_weight = 1.0
    return ds


def process(subset, caption=CAPTION_TEXT, **kwargs):
    return make_dataset().process_caption_with_debug(subset, caption, **kwargs)


# ---------------------------------------------------------------------------
# parsing and weights
# ---------------------------------------------------------------------------


def test_parse_removes_duplicates_and_keep_tags_from_tags():
    caption = parse_moss_caption(CAPTION)
    assert caption.keep_tags == ("mychar", "1girl")
    assert caption.tags == ("solo", "smile", "outdoors")
    assert caption.long == "A girl in a garden,\nfacing the viewer."  # line breaks are kept
    assert caption.tag_caption() == "mychar, 1girl, solo, smile, outdoors"


def test_parse_tags_are_optional():
    caption = parse_moss_caption({"caption": CAPTION["caption"]})
    assert caption.keep_tags == () and caption.tags == ()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "caption` object"),
        ({"caption": "text"}, "caption` object"),
        ({"caption": {"short": "a", "medium": "b"}}, "caption.long"),
        ({"caption": {"short": "a", "medium": " ", "long": "c"}}, "caption.medium"),
        ({"caption": CAPTION["caption"], "tags": "a, b"}, "tags must be an array"),
        ({"caption": CAPTION["caption"], "keep_tags": ["a", ""]}, "keep_tags must be an array"),
    ],
)
def test_parse_rejects_invalid_captions(value, message):
    with pytest.raises(ValueError, match=message):
        parse_moss_caption(value, "x.json")


def test_read_removes_bom(tmp_path):
    (tmp_path / "img.json").write_text(CAPTION_TEXT, encoding="utf-8-sig")
    text = read_moss_caption_text(str(tmp_path / "img.png"), ".json")
    assert parse_moss_caption_text(text) == parse_moss_caption(CAPTION)
    assert read_moss_caption_text(str(tmp_path / "other.png"), ".json") is None


def test_normalize_weights():
    assert normalize_weights("w", None, ("a", "b")) == {"a": 1.0, "b": 1.0}
    assert normalize_weights("w", {"a": 3}, ("a", "b")) == {"a": 3.0, "b": 0.0}  # omitted is zero
    for weights, message in [
        ({"a": -1, "b": 2}, "negative"),
        ({"a": 0}, "positive"),
        ({"a": float("nan")}, "finite"),
        ({"a": "x"}, "finite"),
        ({"c": 1}, "only supports"),
        ([1, 2], "mapping"),
    ]:
        with pytest.raises(ValueError, match=message):
            normalize_weights("w", weights, ("a", "b"))


def test_choices_and_probabilities():
    caption = parse_moss_caption(CAPTION)
    lengths = {"short": 1, "medium": 0, "long": 3}
    choices = get_choices(caption, "mixed", {"tags": 2, "nl": 1, "tags_nl": 1, "nl_tags": 0}, lengths)
    probs = {choice.label: choice.probability for choice in choices}
    assert probs == pytest.approx(
        {"tags": 0.5, "nl:short": 0.0625, "nl:long": 0.1875, "tags_nl:short": 0.0625, "tags_nl:long": 0.1875}
    )
    # caption_mode = tags: keep_tags + tags only
    assert [c.label for c in get_choices(caption, "tags", {}, lengths)] == ["tags"]
    # an image without general tags always uses natural language
    no_tags = parse_moss_caption({"caption": CAPTION["caption"], "keep_tags": ["x"]})
    assert [c.label for c in get_choices(no_tags, "tags", {}, lengths)] == ["nl:short", "nl:long"]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("tags", "k, t1, t2"),
        ("nl", "k, A girl."),
        ("tags_nl", "k, t1, t2, A girl."),
        ("nl_tags", "k, A girl., t1, t2"),
    ],
)
def test_render_variants(variant, expected):
    caption = parse_moss_caption(CAPTION)
    length = None if variant == "tags" else "short"
    assert render(caption, MossCaptionChoice(variant, length, 1.0), ["k"], ["t1", "t2"]) == expected


# ---------------------------------------------------------------------------
# subset options
# ---------------------------------------------------------------------------


def test_subset_defaults_and_validation():
    subset = moss_subset()
    assert subset.caption_extension == ".json"
    assert subset.caption_length_weights == {"short": 1.0, "medium": 1.0, "long": 1.0}
    assert moss_subset(caption_extension=".mossjson").caption_extension == ".mossjson"

    subset = moss_subset(caption_mode="mixed", mixed_weights={"nl": 1})
    assert subset.mixed_weights == {"tags": 0.0, "nl": 1.0, "tags_nl": 0.0, "nl_tags": 0.0}

    for overrides, message in [
        (dict(enable_wildcard=True), "enable_wildcard"),
        (dict(enable_multiline_caption=True), "enable_multiline_caption"),
        (dict(token_warmup_step=10), "token_warmup_step"),
        (dict(caption_mode="mixed", mixed_weights=NO_WEIGHT), "mixed_weights"),
        (dict(caption_length_weights={"huge": 1}), "caption_length_weights"),
        (dict(caption_dropout_rate=0.6, protected_caption_dropout_rate=0.5), "cannot exceed 1"),
        (dict(caption_format="yaml"), "caption_format"),
        (dict(caption_format="text", protected_caption_dropout_rate=0.1), "moss_json"),
    ]:
        with pytest.raises(ValueError, match=message):
            moss_subset(**overrides)


# ---------------------------------------------------------------------------
# process_caption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("tags", "mychar, 1girl, solo, smile, outdoors"),
        ("nl", "mychar, 1girl, A girl in a garden,\nfacing the viewer."),
        ("tags_nl", "mychar, 1girl, solo, smile, outdoors, A girl in a garden,\nfacing the viewer."),
        ("nl_tags", "mychar, 1girl, A girl in a garden,\nfacing the viewer., solo, smile, outdoors"),
    ],
)
def test_process_caption_one_hot_weights(variant, expected):
    weights = dict(NO_WEIGHT, **{variant: 1})
    subset = moss_subset(caption_mode="mixed", mixed_weights=weights, caption_length_weights={"long": 1})
    caption, debug = process(subset)
    assert caption == expected
    assert debug["caption_mode"] == (variant if variant == "tags" else f"{variant}:long")


def test_process_caption_follows_weights():
    subset = moss_subset(
        caption_mode="mixed",
        mixed_weights={"tags": 1, "nl": 3},
        caption_length_weights={"short": 1, "long": 1},
        caption_dropout_rate=0.1,
        protected_caption_dropout_rate=0.2,
    )
    ds = make_dataset()
    random.seed(0)
    counts = collections.Counter(ds.process_caption_with_debug(subset, CAPTION_TEXT)[1]["caption_mode"] for _ in range(20000))
    rates = {label: n / 20000 for label, n in counts.items()}
    # dropout 0.1, protected 0.2, the rest 0.7: tags 0.7 * 1/4, nl 0.7 * 3/4 split by length
    expected = {"dropout": 0.1, "protected": 0.2, "tags": 0.175, "nl:short": 0.2625, "nl:long": 0.2625}
    assert rates == pytest.approx(expected, abs=0.015)


def test_tag_augmentation_changes_only_general_tags(tmp_path):
    protected_tags_file = tmp_path / "protected.txt"
    protected_tags_file.write_text("outdoors\n", encoding="utf-8")
    subset = moss_subset(
        caption_mode="mixed",
        mixed_weights={"tags_nl": 1},
        caption_length_weights={"short": 1},
        keep_tokens=1,
        shuffle_caption=True,
        caption_tag_dropout_rate=1.0,
        protected_tags_file=str(protected_tags_file),
    )
    caption, debug = process(subset)
    # keep_tags first, the first keep_tokens general tag is fixed, protected tags survive, the sentence is kept
    assert caption == "mychar, 1girl, solo, outdoors, A girl."
    assert debug["dropped_tags"] == ["smile"]


def test_dropout_protected_caption_prefix_and_replacements():
    subset = moss_subset(caption_dropout_rate=1.0)
    assert process(subset)[0] == ""

    subset = moss_subset(protected_caption_dropout_rate=1.0, keep_tokens=1, caption_prefix="pre", caption_suffix="suf")
    caption, debug = process(subset)
    assert caption == "pre mychar, 1girl, solo suf" and debug["caption_mode"] == "protected"

    # nothing protected: same as the full caption dropout, prefix and suffix are not added
    no_keep = json.dumps({"caption": CAPTION["caption"], "tags": ["a"]})
    assert process(moss_subset(protected_caption_dropout_rate=1.0, caption_prefix="pre"), no_keep)[0] == ""

    ds = make_dataset()
    ds.add_replacement("mychar", "Alice")
    assert ds.process_caption(moss_subset(caption_prefix="pre"), CAPTION_TEXT) == "pre Alice, 1girl, solo, smile, outdoors"


def test_skip_caption_dropout_keeps_overall_protected_rate():
    subset = moss_subset(caption_dropout_rate=0.5, protected_caption_dropout_rate=0.25)
    ds = make_dataset()
    random.seed(0)
    modes = [ds.process_caption_with_debug(subset, CAPTION_TEXT, skip_caption_dropout=True)[1]["caption_mode"] for _ in range(20000)]
    assert "dropout" not in modes
    # the full dropout (0.5) is applied later, so the protected caption is chosen with 0.25 / (1 - 0.5)
    assert modes.count("protected") / len(modes) == pytest.approx(0.5, abs=0.015)


# ---------------------------------------------------------------------------
# caching candidates
# ---------------------------------------------------------------------------


def test_cache_candidates_and_weights():
    subset = moss_subset(
        caption_mode="mixed",
        mixed_weights={"tags": 1, "nl_tags": 1},
        caption_length_weights={"short": 1, "long": 3},
        caption_dropout_rate=0.1,
        protected_caption_dropout_rate=0.2,
        caption_suffix="suf",
    )
    ds = make_dataset()
    candidates = ds.moss_caption_candidates_for_cache(subset, CAPTION_TEXT)
    assert [label for label, _, _ in candidates] == ["tags", "nl_tags:short", "nl_tags:long", "protected"]
    assert [weight for _, _, weight in candidates] == pytest.approx([0.35, 0.0875, 0.2625, 0.2])
    assert [text for _, text, _ in candidates] == [
        "mychar, 1girl, solo, smile, outdoors suf",
        "mychar, 1girl, A girl., solo, smile, outdoors suf",
        "mychar, 1girl, A girl in a garden,\nfacing the viewer., solo, smile, outdoors suf",
        "mychar, 1girl suf",
    ]
    assert ds.process_caption_candidates_for_cache(subset, CAPTION_TEXT) == [text for _, text, _ in candidates]

    random.seed(0)
    counts = collections.Counter(ds.choose_caption_candidate_for_cache(subset, CAPTION_TEXT)[0] for _ in range(20000))
    total = 0.9  # without the full caption dropout, which is applied to the cached outputs
    assert [counts[k] / 20000 for k in range(4)] == pytest.approx([0.35 / total, 0.0875 / total, 0.2625 / total, 0.2 / total], abs=0.015)


def test_cacheable_only_with_multiple_candidate_support():
    ds = make_dataset()
    subset = moss_subset(caption_mode="mixed")
    ds.subsets = [subset]
    ds.validate_caption_candidates(subset, [CAPTION_TEXT])
    assert ds.has_multiple_caption_candidates
    assert ds.is_text_encoder_output_cacheable(cache_supports_dropout=True, supports_multiple_caption_candidates=True)
    assert not ds.is_text_encoder_output_cacheable(cache_supports_dropout=True, supports_multiple_caption_candidates=False)

    # a single candidate can be cached by any model
    ds = make_dataset()
    subset = moss_subset()
    ds.subsets = [subset]
    ds.validate_caption_candidates(subset, [CAPTION_TEXT])
    assert not ds.has_multiple_caption_candidates and ds.is_text_encoder_output_cacheable()

    # random tag augmentation cannot be cached
    ds.subsets = [moss_subset(caption_mode="mixed", shuffle_caption=True)]
    assert not ds.is_text_encoder_output_cacheable(cache_supports_dropout=True, supports_multiple_caption_candidates=True)


# ---------------------------------------------------------------------------
# DreamBooth dataset loading
# ---------------------------------------------------------------------------


def make_image_dir(tmp_path, names):
    image_dir = tmp_path / "1_images"
    image_dir.mkdir()
    for name in names:
        Image.new("RGB", (64, 64)).save(image_dir / f"{name}.png")
    return image_dir


def make_dreambooth_dataset(image_dir, **overrides):
    subset = moss_subset(image_dir=str(image_dir), caption_mode="mixed", **overrides)
    return DreamBoothDataset([subset], True, 1, (64, 64), 1.0, False, 64, 64, 64, False, 1.0, False, False, 0.0, None, None)


def test_dreambooth_dataset_loads_json_captions(tmp_path):
    image_dir = make_image_dir(tmp_path, ["a", "b"])
    (image_dir / "a.json").write_text(CAPTION_TEXT, encoding="utf-8")
    (image_dir / "b.json").write_text(json.dumps({"caption": CAPTION["caption"], "tags": ["solo"]}), encoding="utf-8")
    (image_dir / "a_nl.json").write_text("not used", encoding="utf-8")

    ds = make_dreambooth_dataset(image_dir)
    infos = sorted(ds.image_data.values(), key=lambda info: info.image_key)
    assert parse_moss_caption_text(infos[0].caption) == parse_moss_caption(CAPTION)
    assert infos[0].caption_nl is None
    assert ds.has_multiple_caption_candidates
    assert ds.tag_frequency["1_images"] == {"mychar": 1, "1girl": 1, "solo": 2, "smile": 1, "outdoors": 1}


def test_dreambooth_dataset_lists_all_missing_and_invalid_json(tmp_path):
    image_dir = make_image_dir(tmp_path, ["a", "b", "c"])
    (image_dir / "a.json").write_text(CAPTION_TEXT, encoding="utf-8")
    (image_dir / "b.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="2 MOSS JSON caption file") as e:
        make_dreambooth_dataset(image_dir)
    message = str(e.value)
    assert "b.json: invalid JSON" in message and "c.json: file not found" in message


# ---------------------------------------------------------------------------
# end to end with the MOSS-Image Text Encoder outputs cache
# ---------------------------------------------------------------------------


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        ids = [ord(c) % 250 + 1 for c in text]
        return {"input_ids": torch.tensor([ids[:max_length] if truncation and max_length else ids], dtype=torch.int64)}


class FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, 8)

    def forward(self, input_ids, use_cache=False, return_dict=True):
        pos = torch.arange(input_ids.shape[1])[None, :, None].float()
        return SimpleNamespace(last_hidden_state=(self.embed(input_ids) + pos * 0.01).to(torch.bfloat16))


def make_moss_tokenize_strategy():
    tokenize_strategy = MossImageTokenizeStrategy.__new__(MossImageTokenizeStrategy)
    tokenize_strategy.tokenizer, tokenize_strategy.max_length, tokenize_strategy.pad_token_id = FakeTokenizer(), 4096, 0
    return tokenize_strategy


@pytest.mark.parametrize("cache_to_disk", [True, False])
def test_moss_image_cache_roundtrip(tmp_path, cache_to_disk):
    tokenize_strategy = make_moss_tokenize_strategy()
    encoding_strategy = MossImageTextEncodingStrategy()
    text_encoder = FakeTextEncoder()
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(cache_to_disk, 4, False)

    ds = make_dataset()
    ds.batch_size = 2
    subset = moss_subset(
        caption_mode="mixed",
        mixed_weights={"tags": 1, "nl": 1},
        caption_length_weights={"short": 1, "long": 1},
        protected_caption_dropout_rate=0.1,
    )
    ds.subsets = [subset]
    info = ImageInfo("img", 1, CAPTION_TEXT, False, str(tmp_path / "img.png"), subset.caption_dropout_rate)
    ds.register_image(info, subset)
    info.latents = torch.zeros(4, 8, 8)
    info.latents_original_size = (64, 64)
    info.latents_crop_ltrb = (0, 0, 0, 0)
    info.alpha_mask = None
    info.bucket_reso = (64, 64)

    with patch.object(TokenizeStrategy, "get_strategy", return_value=tokenize_strategy), patch.object(
        TextEncodingStrategy, "get_strategy", return_value=encoding_strategy
    ), patch.object(TextEncoderOutputsCachingStrategy, "get_strategy", return_value=caching_strategy):
        ds.new_cache_text_encoder_outputs([text_encoder], SimpleNamespace(num_processes=1, process_index=0))

    captions = ds.process_caption_candidates_for_cache(subset, CAPTION_TEXT)
    assert len(captions) == 4  # tags, nl:short, nl:long, protected
    if cache_to_disk:
        assert caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, captions)
    else:
        assert len(info.text_encoder_outputs) == len(captions)

    ds.text_encoder_output_caching_strategy = caching_strategy
    ds.bucket_manager = SimpleNamespace(buckets=[["img"]])
    ds.buckets_indices = [BucketBatchIndex(0, 1, 0)]
    seen = set()
    for seed in range(30):
        random.seed(seed)
        example = ds[0]
        caption = example["captions"][0]
        seen.add(caption)
        ids = moss_image_utils.tokenize_prompts(tokenize_strategy.tokenizer, [caption])[0]
        expected = text_encoder(ids[None]).last_hidden_state[0].float()
        embeds, mask = example["text_encoder_outputs_list"][:2]
        assert torch.equal(embeds[0, : int(mask[0].sum())], expected)  # the cached outputs match the caption
    assert seen <= set(captions) and len(seen) >= 3


# ---------------------------------------------------------------------------
# regressions from the MOSS JSON review (2026-10-22)
# ---------------------------------------------------------------------------


def cached_dataset(tmp_path, caption=CAPTION_TEXT, cache_to_disk=False, **overrides):
    """A dataset with one image whose Text Encoder outputs are cached by the MOSS-Image strategy (fake encoder)."""
    encoding_strategy = MossImageTextEncodingStrategy()
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(cache_to_disk, 4, False)
    subset = moss_subset(**overrides)
    ds = make_dataset()
    ds.subsets = [subset]
    info = ImageInfo("img", 1, caption, False, str(tmp_path / "img.png"), subset.caption_dropout_rate)
    ds.register_image(info, subset)
    info.latents = torch.zeros(4, 8, 8)
    info.latents_original_size = (64, 64)
    info.latents_crop_ltrb = (0, 0, 0, 0)
    info.alpha_mask = None
    info.bucket_reso = (64, 64)
    with patch.object(TokenizeStrategy, "get_strategy", return_value=make_moss_tokenize_strategy()), patch.object(
        TextEncodingStrategy, "get_strategy", return_value=encoding_strategy
    ), patch.object(TextEncoderOutputsCachingStrategy, "get_strategy", return_value=caching_strategy):
        ds.new_cache_text_encoder_outputs([FakeTextEncoder()], SimpleNamespace(num_processes=1, process_index=0))
    ds.text_encoder_output_caching_strategy = caching_strategy
    ds.bucket_manager = SimpleNamespace(buckets=[["img"]])
    ds.buckets_indices = [BucketBatchIndex(0, 1, 0)]
    ds.log_captions_every_n_steps = 1
    return ds, subset, encoding_strategy


def trained_embeds(example, encoding_strategy):
    """The embeddings the trainer uses after its caption dropout (moss_image_train_network.process_batch)."""
    values = example["text_encoder_outputs_list"]
    embeds, mask = encoding_strategy.drop_cached_text_encoder_outputs(*values[:-1], caption_dropout_rates=values[-1])
    return embeds[0][mask[0].bool()]


def unconditional_embeds(example):
    values = example["text_encoder_outputs_list"]
    return values[2][0][values[3][0].bool()]


def trained_label(example, encoding_strategy):
    if torch.equal(trained_embeds(example, encoding_strategy), unconditional_embeds(example)):
        return "dropout"
    return example["caption_debug_entries"][0]["caption_mode"]


def format_options(caption_format, **overrides):
    """The caption and the subset options of a MOSS JSON or a text caption, for the tests of both formats."""
    if caption_format == "moss_json":
        return CAPTION_TEXT, dict(caption_format="moss_json", **overrides)
    return "mychar, solo", dict(caption_format="text", **overrides)


def test_cli_and_config_precedence():
    from library.config_util import BlueprintGenerator, ConfigSanitizer
    from moss_image_train_network import setup_parser

    args = setup_parser().parse_args(["--caption_format", "moss_json", "--protected_caption_dropout_rate", "0.2"])
    generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
    config = {"datasets": [{"resolution": 64, "subsets": [{"image_dir": "images"}]}]}
    params = generator.generate(config, args).dataset_group.datasets[0].subsets[0].params
    assert params.caption_format == "moss_json" and params.protected_caption_dropout_rate == 0.2
    config["general"] = {"protected_caption_dropout_rate": 0.3}
    config["datasets"][0]["subsets"][0]["protected_caption_dropout_rate"] = 0.4
    assert generator.generate(config, args).dataset_group.datasets[0].subsets[0].params.protected_caption_dropout_rate == 0.4


def test_special_dropout_preserves_natural_language_and_file_protected_tags(tmp_path):
    protected = tmp_path / "protected.txt"
    protected.write_text("mychar\nsolo\n", encoding="utf-8")
    subset = moss_subset(
        caption_mode="mixed",
        mixed_weights={"tags_nl": 1},
        caption_length_weights={"long": 1},
        keep_tokens=1,
        caption_tag_dropout_rate=1.0,
        special_caption_tag_dropout_rate=1.0,
        protected_tags_file=str(protected),
    )
    assert process(subset)[0] == "mychar, solo, A girl in a garden,\nfacing the viewer."


def test_cached_and_uncached_probabilities_agree(tmp_path):
    ds, _, encoding_strategy = cached_dataset(
        tmp_path,
        caption_mode="mixed",
        mixed_weights={"tags": 1, "nl": 3},
        caption_length_weights={"short": 1, "long": 1},
        caption_dropout_rate=0.1,
        protected_caption_dropout_rate=0.2,
    )
    ds.current_epoch = 1
    random.seed(8143)
    counts = collections.Counter(trained_label(ds[0], encoding_strategy) for _ in range(5000))
    assert {k: v / 5000 for k, v in counts.items()} == pytest.approx(
        {"dropout": 0.1, "protected": 0.2, "tags": 0.175, "nl:short": 0.2625, "nl:long": 0.2625}, abs=0.025
    )


@pytest.mark.parametrize("caption_format", ["moss_json", "text"])
def test_reused_disk_cache_uses_current_dropout_settings(tmp_path, caption_format):
    # the cache is created with caption_dropout_rate 0.0 and reused with 0.5: the rate in the file must not be used
    protected = {"protected_caption_dropout_rate": 0.2} if caption_format == "moss_json" else {}
    caption, options = format_options(caption_format, cache_to_disk=True, **protected)
    cached_dataset(tmp_path, caption, caption_dropout_rate=0.0, **options)
    ds, _, encoding_strategy = cached_dataset(tmp_path, caption, caption_dropout_rate=0.5, **options)

    random.seed(0)
    labels = []
    for _ in range(4000):
        example = ds[0]
        assert float(example["text_encoder_outputs_list"][-1][0]) in (0.0, 1.0)  # decided by the dataset
        labels.append(trained_label(example, encoding_strategy))
    assert labels.count("dropout") / 4000 == pytest.approx(0.5, abs=0.025)
    if protected:
        assert labels.count("protected") / 4000 == pytest.approx(0.2, abs=0.025)


@pytest.mark.parametrize("caption_format", ["moss_json", "text"])
def test_epoch_dropout_matches_with_and_without_cache(tmp_path, caption_format):
    caption, options = format_options(caption_format, caption_dropout_every_n_epochs=2)
    ds, subset, encoding_strategy = cached_dataset(tmp_path, caption, **options)

    ds.current_epoch = 2
    assert ds.process_caption(subset, caption) == ""
    example = ds[0]
    assert torch.equal(trained_embeds(example, encoding_strategy), unconditional_embeds(example))
    assert example["captions"] == [""]

    ds.current_epoch = 3
    assert ds.process_caption(subset, caption) != ""
    example = ds[0]
    assert not torch.equal(trained_embeds(example, encoding_strategy), unconditional_embeds(example))


@pytest.mark.parametrize("caption_format", ["moss_json", "text"])
def test_cached_debug_caption_reports_full_dropout(tmp_path, caption_format):
    caption, options = format_options(caption_format, caption_dropout_rate=1.0)
    ds, _, encoding_strategy = cached_dataset(tmp_path, caption, **options)
    example = ds[0]
    assert torch.equal(trained_embeds(example, encoding_strategy), unconditional_embeds(example))
    assert example["caption_debug_entries"][0]["caption_mode"] == "dropout"
    assert example["captions"] == [""]


def test_cached_caption_dropout_keeps_the_type_of_the_rate():
    # Anima keeps the in-memory rate as a torch tensor, MOSS-Image and the disk caches as numpy arrays
    ds = make_dataset()
    outputs, dropped = ds._decide_cached_caption_dropout(moss_subset(caption_dropout_rate=1.0), (np.zeros(2), torch.tensor(0.0)))
    assert dropped and isinstance(outputs[-1], torch.Tensor) and float(outputs[-1]) == 1.0
    outputs, dropped = ds._decide_cached_caption_dropout(moss_subset(), [np.zeros(2), np.array(0.3, dtype=np.float32)])
    assert not dropped and isinstance(outputs[-1], np.ndarray) and float(outputs[-1]) == 0.0


@pytest.mark.parametrize(("initial_format", "next_format"), [("text", "moss_json"), ("moss_json", "text")])
def test_switching_caption_format_invalidates_metadata_cache(tmp_path, initial_format, next_format):
    image_dir = make_image_dir(tmp_path, ["a"])
    (image_dir / "a.caption").write_text("mychar, solo", encoding="utf-8")
    (image_dir / "a.json").write_text(CAPTION_TEXT, encoding="utf-8")
    make_dreambooth_dataset(image_dir, caption_format=initial_format, cache_info=True)
    ds = make_dreambooth_dataset(image_dir, caption_format=next_format, cache_info=True)
    caption = next(iter(ds.image_data.values())).caption
    if next_format == "moss_json":
        assert json.loads(caption)["keep_tags"] == ["mychar", "1girl"]
    else:
        assert caption == "mychar, solo"

    # the cache is updated with the new format and used again for the same format
    metas = json.loads((image_dir / "metadata_cache.json").read_text(encoding="utf-8"))
    assert [meta["caption_format"] for meta in metas.values()] == [next_format]
    (image_dir / "a.caption").write_text("changed", encoding="utf-8")
    (image_dir / "a.json").write_text("{}", encoding="utf-8")
    ds = make_dreambooth_dataset(image_dir, caption_format=next_format, cache_info=True)
    assert next(iter(ds.image_data.values())).caption == caption


@pytest.mark.parametrize(
    ("legacy_format", "current_format"), [("text", "moss_json"), ("moss_json", "moss_json"), ("moss_json", "text")]
)
def test_metadata_cache_without_caption_format_is_read_again(tmp_path, legacy_format, current_format):
    # caches of previous versions (including the first MOSS JSON implementation) have no caption_format
    image_dir = make_image_dir(tmp_path, ["a"])
    (image_dir / "a.caption").write_text("mychar, solo", encoding="utf-8")
    (image_dir / "a.json").write_text(CAPTION_TEXT, encoding="utf-8")
    legacy_caption = CAPTION_TEXT if legacy_format == "moss_json" else "mychar, solo"
    cache_file = image_dir / "metadata_cache.json"
    cache_file.write_text(
        json.dumps({str(image_dir / "a.png"): {"caption": legacy_caption, "resolution": [64, 64]}}), encoding="utf-8"
    )

    ds = make_dreambooth_dataset(image_dir, caption_format=current_format, cache_info=True)
    assert next(iter(ds.image_data.values())).caption == (CAPTION_TEXT if current_format == "moss_json" else "mychar, solo")
    metas = json.loads(cache_file.read_text(encoding="utf-8"))
    assert [meta["caption_format"] for meta in metas.values()] == [current_format]


def test_all_invalid_files_reported_even_with_invalid_utf8(tmp_path):
    image_dir = make_image_dir(tmp_path, ["a", "b", "c"])
    (image_dir / "a.json").write_bytes(b"\xff\xfeinvalid")
    (image_dir / "b.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="3 MOSS JSON caption file") as e:
        make_dreambooth_dataset(image_dir)
    message = str(e.value)
    assert "a.json: not UTF-8" in message and "b.json: invalid JSON" in message and "c.json: file not found" in message
