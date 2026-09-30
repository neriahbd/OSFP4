import pytest
import torch

from llmcompressor_osfp4.modifiers.optimization.sic import (
    accumulate_hessian,
    make_empty_hessian,
)


@pytest.mark.parametrize("shape", [(7, 16), (4, 3, 16)])
def test_osfp4_hessian_accumulates_raw_linear_gram(shape):
    generator = torch.Generator().manual_seed(70)
    module = torch.nn.Linear(16, 5, bias=False)
    activations = torch.randn(shape, generator=generator)
    hessian = make_empty_hessian(module)

    returned = accumulate_hessian(activations, module, hessian)
    flattened = activations.reshape(-1, 16).float()
    expected = torch.zeros_like(hessian)
    expected.addmm_(flattened.T, flattened)

    assert returned is hessian
    assert hessian.dtype is torch.float32
    assert hessian.device == module.weight.device
    assert torch.equal(hessian, expected)
    assert not torch.allclose(hessian, 2.0 * expected)


def test_osfp4_hessian_matches_ordered_addmm_reference_exactly():
    generator = torch.Generator().manual_seed(72)
    module = torch.nn.Linear(16, 5, bias=False)
    batches = [
        torch.randn(7, 16, generator=generator),
        torch.randn(2, 3, 16, generator=generator),
        torch.randn(11, 16, generator=generator),
    ]
    actual = make_empty_hessian(module)
    expected = torch.zeros_like(actual)

    for batch in batches:
        accumulate_hessian(batch, module, actual)
        flattened = batch.float().reshape(-1, 16)
        expected.addmm_(flattened.T, flattened)

    assert torch.equal(actual, expected)


def test_osfp4_hessian_is_invariant_to_calibration_batch_partitioning():
    generator = torch.Generator().manual_seed(71)
    module = torch.nn.Linear(16, 5, bias=False)
    activations = torch.randn(4, 3, 16, generator=generator)

    whole = accumulate_hessian(activations, module, make_empty_hessian(module))
    partitioned = make_empty_hessian(module)
    accumulate_hessian(activations[:1], module, partitioned)
    accumulate_hessian(activations[1:], module, partitioned)

    torch.testing.assert_close(partitioned, whole)
    assert not torch.allclose(partitioned, whole / activations.shape[0])
