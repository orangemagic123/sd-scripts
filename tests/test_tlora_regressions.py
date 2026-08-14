from types import SimpleNamespace

import pytest
import torch
from torch import nn

from networks.lora_tlora import TLoRAModule as SdTLoRAModule
from networks.lora_tlora import TLoRANetwork as SdTLoRANetwork
from networks.lora_tlora_anima import TLoRAModule as AnimaTLoRAModule
from networks.lora_tlora_anima import TLoRANetwork as AnimaTLoRANetwork
from networks.tlora_utils import build_rank_mask, timestep_progress, validate_tlora_schedule


class _RankNetwork:
    def __init__(self, progress, min_rank=1):
        self.current_timestep_progress = torch.as_tensor(progress, dtype=torch.float32)
        self.tlora_min_rank = min_rank

    def get_timestep_rank_mask(self, rank, batch_size, device, dtype):
        return build_rank_mask(
            self.current_timestep_progress,
            rank,
            self.tlora_min_rank,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )


@pytest.mark.parametrize("module_class", [SdTLoRAModule, AnimaTLoRAModule])
def test_plain_tlora_applies_rank_per_batch_item(module_class):
    original = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        original.weight.zero_()

    module = module_class(
        "test",
        original,
        lora_dim=2,
        alpha=2,
        network=_RankNetwork([1.0, 0.0]),
        is_unet=True,
    )
    with torch.no_grad():
        module.lora_down.weight.copy_(torch.eye(2))
        module.lora_up.weight.copy_(torch.eye(2))
    module.apply_to()
    module.train()

    output = original(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(output, torch.tensor([[1.0, 2.0], [3.0, 0.0]]))

    module.enabled = False
    assert torch.equal(original(torch.ones(2, 2)), torch.zeros(2, 2))


def _network_shell(network_class, max_timestep):
    network = object.__new__(network_class)
    nn.Module.__init__(network)
    network.lora_dim = 4
    network.tlora_min_rank = 1
    network.tlora_alpha_rank_scale = 1.0
    network.tlora_max_timestep = max_timestep
    network.current_sigma_mask = None
    network.current_sigma_r = None
    network.current_timestep_progress = None
    network._rank_mask_cache = {}
    return network


@pytest.mark.parametrize(
    "network_class,max_timestep,timesteps",
    [
        (SdTLoRANetwork, 1000.0, torch.tensor([0.0, 1000.0])),
        (AnimaTLoRANetwork, 1.0, torch.tensor([0.0, 1.0])),
    ],
)
def test_tlora_hook_keeps_every_batch_timestep(network_class, max_timestep, timesteps):
    network = _network_shell(network_class, max_timestep)
    network.train()
    sample = torch.zeros(2, 1)

    network._unet_forward_pre_hook(None, (sample, timesteps), {})

    mask = network.get_timestep_rank_mask(4, 2, sample.device, torch.float32)
    assert torch.equal(mask.sum(dim=1), torch.tensor([4.0, 1.0]))
    assert network.is_mergeable() is False


def test_rank_mask_repeats_for_classifier_free_guidance():
    progress = timestep_progress(torch.tensor([0.0, 1.0]), 1.0)
    mask = build_rank_mask(progress, rank=4, min_rank=1, batch_size=4)
    assert torch.equal(mask.sum(dim=1), torch.tensor([4.0, 1.0, 4.0, 1.0]))


def test_invalid_tlora_schedule_is_rejected():
    with pytest.raises(ValueError):
        validate_tlora_schedule(0.0, 1, 1.0)
    with pytest.raises(ValueError):
        validate_tlora_schedule(1.0, 0, 1.0)
    with pytest.raises(ValueError):
        validate_tlora_schedule(1.0, 1, -0.5)
