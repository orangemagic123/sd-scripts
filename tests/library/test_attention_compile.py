import pytest
import torch

from library.attention import AttentionParams, attention


def test_sdpa_fullgraph_compilation_preserves_output_and_gradients():
    """Python 3.10 must not hit DELETE_DEREF when releasing q/k/v."""
    torch.manual_seed(42)
    inputs = [torch.randn(1, 8, 2, 4, requires_grad=True) for _ in range(3)]
    compiled_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    def forward(q, k, v):
        return attention([q, k, v], attn_params=AttentionParams("torch", False))

    expected = forward(*inputs)
    expected.square().sum().backward()
    # AOTAutograd exercises both graphs without requiring Triton or CUDA.
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    actual = compiled(*compiled_inputs)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip(compiled_inputs, inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_split_sdpa_matches_individual_sequences(noncontiguous):
    torch.manual_seed(42)
    inputs = [torch.randn(2, 8, 2, 8 if noncontiguous else 4) for _ in range(3)]
    if noncontiguous:
        inputs = [tensor[..., ::2] for tensor in inputs]
    inputs = [tensor.requires_grad_() for tensor in inputs]
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    params = AttentionParams("torch", True, seqlens=torch.tensor([5, 8]), max_seqlen=8)
    actual = attention(list(inputs), attn_params=params)
    expected_loss = 0
    for i, length in enumerate([5, 8]):
        expected = attention(
            [tensor[i : i + 1, :length] for tensor in expected_inputs],
            attn_params=AttentionParams("torch", False),
        )
        expected_loss = expected_loss + expected.square().sum()
        torch.testing.assert_close(actual[i : i + 1, :length], expected)
        assert torch.count_nonzero(actual[i, length:]) == 0
    actual.square().sum().backward()
    expected_loss.backward()
    for actual_input, expected_input in zip(inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
