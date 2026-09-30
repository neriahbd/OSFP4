import torch

from llmcompressor_osfp4.modifiers.optimization._activation_groups import (
    assemble_activation_quantization_groups,
)


def test_assemble_activation_quantization_groups_preserves_order():
    cached = [
        torch.arange(2 * 80, dtype=torch.float16).reshape(2, 80),
        torch.arange(3 * 80, dtype=torch.float16).reshape(3, 80).add_(1000),
    ]
    expected = torch.cat(cached).reshape(5, 5, 16).permute(1, 2, 0).float().contiguous()

    actual = assemble_activation_quantization_groups(
        cached,
        5,
        torch.device("cpu"),
    )

    assert torch.equal(actual, expected)
    assert actual.shape == (5, 16, 5)
    assert actual.is_contiguous()
