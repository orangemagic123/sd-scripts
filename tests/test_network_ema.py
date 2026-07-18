import pytest
import torch
from torch import nn

from train_network import _ema_scope, _update_ema_model, setup_parser


class TinyNetwork(nn.Module):
    def __init__(self, weight, bias, running_mean, num_batches):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.register_buffer("running_mean", torch.tensor(running_mean, dtype=torch.float32))
        self.register_buffer("num_batches", torch.tensor(num_batches, dtype=torch.int64))


def _assert_parameters_equal(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        assert torch.equal(actual_parameters[name], expected_parameters[name])


def test_update_ema_model_averages_parameters_and_copies_buffers():
    ema_model = TinyNetwork(
        weight=[2.0, 4.0],
        bias=[1.0],
        running_mean=[1.0, 3.0],
        num_batches=2,
    )
    source_model = TinyNetwork(
        weight=[10.0, 20.0],
        bias=[5.0],
        running_mean=[7.0, 9.0],
        num_batches=11,
    )
    source_state = {name: value.detach().clone() for name, value in source_model.state_dict().items()}

    _update_ema_model(ema_model, source_model, decay=0.25)

    assert torch.allclose(ema_model.weight, torch.tensor([8.0, 16.0]))
    assert torch.allclose(ema_model.bias, torch.tensor([4.0]))
    assert torch.equal(ema_model.running_mean, source_model.running_mean)
    assert torch.equal(ema_model.num_batches, source_model.num_batches)
    for name, original_value in source_state.items():
        assert torch.equal(source_model.state_dict()[name], original_value)


def test_ema_scope_uses_ema_parameters_and_restores_normally():
    network = TinyNetwork(
        weight=[1.0, 2.0],
        bias=[3.0],
        running_mean=[4.0, 5.0],
        num_batches=6,
    )
    ema_network = TinyNetwork(
        weight=[10.0, 20.0],
        bias=[30.0],
        running_mean=[40.0, 50.0],
        num_batches=60,
    )
    original_parameters = {name: value.detach().clone() for name, value in network.named_parameters()}

    with _ema_scope(network, ema_network):
        _assert_parameters_equal(network, ema_network)
        with torch.no_grad():
            network.weight.add_(100.0)

    for name, parameter in network.named_parameters():
        assert torch.equal(parameter, original_parameters[name])


def test_ema_scope_restores_parameters_after_exception():
    network = TinyNetwork(
        weight=[1.0, 2.0],
        bias=[3.0],
        running_mean=[4.0, 5.0],
        num_batches=6,
    )
    ema_network = TinyNetwork(
        weight=[10.0, 20.0],
        bias=[30.0],
        running_mean=[40.0, 50.0],
        num_batches=60,
    )
    original_parameters = {name: value.detach().clone() for name, value in network.named_parameters()}

    with pytest.raises(RuntimeError, match="validation failed"):
        with _ema_scope(network, ema_network):
            _assert_parameters_equal(network, ema_network)
            with torch.no_grad():
                network.bias.zero_()
            raise RuntimeError("validation failed")

    for name, parameter in network.named_parameters():
        assert torch.equal(parameter, original_parameters[name])


def test_setup_parser_parses_ema_decay():
    parser = setup_parser()

    assert parser.parse_args([]).ema_decay is None
    assert parser.parse_args(["--ema_decay", "0.999"]).ema_decay == pytest.approx(0.999)
