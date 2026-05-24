import json
from types import SimpleNamespace

from train_network import _build_caption_metadata


def test_build_caption_metadata_for_mixed_caption_mode():
    mixed_weights = {"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25}
    train_dataset_group = SimpleNamespace(
        datasets=[SimpleNamespace(subsets=[SimpleNamespace(caption_mode="mixed", mixed_weights=mixed_weights)])]
    )

    metadata = _build_caption_metadata(train_dataset_group)

    assert metadata["ss_caption_mode"] == "mixed"
    assert json.loads(metadata["ss_mixed_weights"]) == mixed_weights


def test_build_caption_metadata_for_multiple_caption_modes():
    mixed_weights = {"tags": 10, "nl": 20, "tags_nl": 30, "nl_tags": 40}
    train_dataset_group = SimpleNamespace(
        datasets=[
            SimpleNamespace(
                subsets=[
                    SimpleNamespace(caption_mode="tags", mixed_weights=None),
                    SimpleNamespace(caption_mode="mixed", mixed_weights=mixed_weights),
                ]
            )
        ]
    )

    metadata = _build_caption_metadata(train_dataset_group)

    assert json.loads(metadata["ss_caption_mode"]) == ["tags", "mixed"]
    assert json.loads(metadata["ss_mixed_weights"]) == mixed_weights
