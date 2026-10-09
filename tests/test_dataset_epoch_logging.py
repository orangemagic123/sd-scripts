import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from library.controlnet_dataset import ControlNetDataset
from library.dataset import BaseDataset, DatasetGroup


def _dataset(*protected_tags_files):
    dataset = BaseDataset((8, 8), 1.0, False, False)
    dataset.subsets = [SimpleNamespace(protected_tags_file=str(path)) for path in protected_tags_files]
    dataset.num_train_images = 1
    dataset.num_reg_images = 0
    dataset._length = 1
    dataset.shuffle_buckets = Mock()
    return dataset


@pytest.mark.parametrize("worker_id", [None, 0, 1, 7])
@pytest.mark.parametrize("rank", ["0", "1"])
def test_group_logs_once_per_epoch_and_keeps_all_datasets_updated(tmp_path, monkeypatch, caplog, worker_id, rank):
    tags_file = tmp_path / "protected_tags.txt"
    tags_file.write_text("chibi\nsketch\n", encoding="utf-8")
    datasets = [_dataset(tags_file, tags_file) for _ in range(3)]
    group = DatasetGroup(datasets)
    worker = None if worker_id is None else SimpleNamespace(id=worker_id)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker)
    monkeypatch.setenv("RANK", rank)

    with caplog.at_level(logging.INFO, logger="library.dataset"):
        group.set_current_epoch(1)
        group.set_current_epoch(1)
        group.set_current_epoch(2)

    messages = [record.message for record in caplog.records if record.name == "library.dataset"]
    if worker_id in (None, 0) and rank == "0":
        assert messages == [
            "epoch is incremented. current_epoch: 0, epoch: 1",
            f"[protected tags] epoch=1 file={tags_file} tags=['chibi', 'sketch']",
            "epoch is incremented. current_epoch: 1, epoch: 2",
            f"[protected tags] epoch=2 file={tags_file} tags=['chibi', 'sketch']",
        ]
    else:
        assert messages == []

    for dataset in datasets:
        assert dataset.current_epoch == 2
        assert dataset.shuffle_buckets.call_count == 2
        assert dataset._get_protected_tags(dataset.subsets[0]) == {"chibi", "sketch"}


def test_different_protected_tags_files_and_controlnet_delegate_are_logged(tmp_path, monkeypatch, caplog):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("chibi\n", encoding="utf-8")
    second_file.write_text("sketch\n", encoding="utf-8")
    first = _dataset(first_file)
    delegate = _dataset(first_file, second_file)
    controlnet = ControlNetDataset.__new__(ControlNetDataset)
    controlnet.dreambooth_dataset_delegate = delegate
    controlnet.image_data = delegate.image_data
    controlnet.num_train_images = delegate.num_train_images
    controlnet.num_reg_images = delegate.num_reg_images
    group = DatasetGroup([first, controlnet])
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    monkeypatch.delenv("RANK", raising=False)

    with caplog.at_level(logging.INFO, logger="library.dataset"):
        group.set_current_epoch(10)

    messages = [record.message for record in caplog.records if record.name == "library.dataset"]
    assert messages == [
        "epoch is incremented. current_epoch: 0, epoch: 10",
        f"[protected tags] epoch=10 file={first_file} tags=['chibi']",
        f"[protected tags] epoch=10 file={second_file} tags=['sketch']",
    ]
    for dataset in (first, delegate):
        assert dataset.current_epoch == 10
        dataset.shuffle_buckets.assert_called_once_with()


def test_standalone_dataset_keeps_epoch_logs(tmp_path, monkeypatch, caplog):
    tags_file = tmp_path / "protected_tags.txt"
    tags_file.write_text("chibi\n", encoding="utf-8")
    dataset = _dataset(tags_file)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    monkeypatch.delenv("RANK", raising=False)

    with caplog.at_level(logging.INFO, logger="library.dataset"):
        dataset.set_current_epoch(1)
        dataset.set_current_epoch(1)

    assert len(caplog.records) == 2
    assert dataset.current_epoch == 1
    dataset.shuffle_buckets.assert_called_once_with()
