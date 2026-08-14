import contextlib
from types import SimpleNamespace

import torch
from torch import nn

import train_network
from train_network import NetworkTrainer, _network_eval_scope, _should_sample_at_step


def test_should_sample_at_step_matches_step_sampling_contract():
    args = SimpleNamespace(
        sample_at_first=True,
        sample_every_n_steps=10,
        sample_every_n_epochs=None,
    )
    assert _should_sample_at_step(args, 0)
    assert _should_sample_at_step(args, 10)
    assert not _should_sample_at_step(args, 9)

    args.sample_every_n_epochs = 1
    assert not _should_sample_at_step(args, 10)


def test_network_eval_scope_restores_training_mode():
    module = nn.Dropout()
    module.train()
    with _network_eval_scope(module):
        assert not module.training
    assert module.training

    module.eval()
    with _network_eval_scope(module):
        assert not module.training
    assert not module.training


class _Accelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()


class _Network:
    def __init__(self):
        self.multiplier = 0.75
        self.history = []

    def set_multiplier(self, value):
        self.multiplier = value
        self.history.append(value)


def test_differential_preservation_uses_inpainting_input_and_only_selected_batch(monkeypatch):
    trainer = NetworkTrainer()
    calls = []

    def fake_noise_inputs(args, scheduler, latents):
        noise = torch.zeros_like(latents)
        noisy = latents.clone()
        timesteps = torch.tensor([10, 20], dtype=torch.long)
        return noise, noisy, timesteps

    monkeypatch.setattr(train_network.loss_util, "get_noise_noisy_latents_and_timesteps", fake_noise_inputs)

    def fake_call_unet(args, accelerator, unet, model_input, timesteps, text_conds, batch, weight_dtype, **kwargs):
        calls.append((tuple(model_input.shape), tuple(timesteps.shape), tuple(text_conds[0].shape)))
        value = 1.0 if model_input.shape[0] == 1 else 0.0
        return torch.full((model_input.shape[0], 4, 2, 2), value)

    trainer.call_unet = fake_call_unet
    network = _Network()
    latents = torch.zeros(2, 4, 2, 2)
    batch = {
        "masks": torch.ones(2, 1, 4, 4),
        "masked_latents": torch.zeros(2, 4, 2, 2),
        "custom_attributes": [{}, {"diff_output_preservation": True}],
    }
    args = SimpleNamespace(gradient_checkpointing=False, v_parameterization=False)

    _, target, _, _ = trainer.get_noise_pred_and_target(
        args,
        _Accelerator(),
        object(),
        latents,
        batch,
        [torch.zeros(2, 3, 4)],
        object(),
        network,
        torch.float32,
        True,
    )

    assert calls == [((2, 9, 2, 2), (2,), (2, 3, 4)), ((1, 9, 2, 2), (1,), (1, 3, 4))]
    assert torch.equal(target[0], torch.zeros_like(target[0]))
    assert torch.equal(target[1], torch.ones_like(target[1]))
    assert network.history == [0.0, 0.75]
