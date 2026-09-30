import hashlib

import pytest
import torch

from llmcompressor_osfp4.modifiers import activation_subsampling
from llmcompressor_osfp4.modifiers.activation_subsampling import (
    subsample_activations,
)


def _batches() -> list[torch.Tensor]:
    rows = torch.arange(10, dtype=torch.float32).reshape(10, 1).expand(-1, 4)
    return [rows[:3].clone(), rows[3:7].clone(), rows[7:].clone()]


def test_fixed_sampling_is_global_deterministic_and_preserves_complete_rows():
    first = subsample_activations(_batches(), 4, output_rows=7)
    second = subsample_activations(_batches(), 4, output_rows=7)

    assert first.batches is not None
    assert second.batches is not None
    assert torch.equal(first.batches[0], second.batches[0])
    assert first.batches[0][:, 0].tolist() == [1.0, 2.0, 6.0, 8.0]
    assert torch.equal(
        first.batches[0],
        first.batches[0][:, :1].expand(-1, 4),
    )
    assert len(first.batches[0][:, 0].unique()) == 4
    assert first.provenance == second.provenance
    expected_indices = torch.tensor([1, 2, 6, 8], dtype=torch.int64)
    assert first.provenance == {
        "policy": "fixed",
        "seed": 42,
        "m": 7,
        "k": 4,
        "k1": 10,
        "index_sha256": hashlib.sha256(expected_indices.numpy().tobytes()).hexdigest(),
    }
    assert list(first.provenance) == ["policy", "seed", "m", "k", "k1", "index_sha256"]


def test_sampling_sizes_are_nested_for_the_private_seeded_ordering():
    smaller = subsample_activations(_batches(), 3, output_rows=7)
    larger = subsample_activations(_batches(), 6, output_rows=7)

    assert smaller.batches is not None
    assert larger.batches is not None
    assert set(smaller.batches[0][:, 0].tolist()) < set(
        larger.batches[0][:, 0].tolist()
    )


def test_fixed_size_clamps_to_all_cached_vectors():
    sampled = subsample_activations(_batches(), 4, output_rows=12)
    bypassed = subsample_activations(_batches(), 12, output_rows=4)

    assert sampled.batches is not None
    assert sampled.batches[0].shape == (4, 4)
    assert sampled.provenance["policy"] == "fixed"
    assert sampled.provenance["k"] == 4
    assert bypassed.batches is None
    assert bypassed.provenance["k"] == bypassed.provenance["k1"] == 10


@pytest.mark.parametrize("size", [1, 4, 10, 12])
def test_subsampling_does_not_consume_global_rng_state(size):
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()

    subsample_activations(_batches(), size, output_rows=7)

    assert torch.equal(torch.random.get_rng_state(), state_before)


@pytest.mark.parametrize("size", [1, 12])
def test_subsampling_rejects_cross_batch_dtype_changes(size):
    batches = [torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bfloat16)]

    with pytest.raises(ValueError, match="share one dtype"):
        subsample_activations(batches, size, output_rows=4)


def _tensor_bytes(tensor):
    flat = tensor.contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("lengths", [(3, 4, 3), (0, 2, 0, 7, 1, 0), (10,)])
def test_selection_preserves_bytes_across_batch_layouts(dtype, noncontiguous, lengths):
    generator = torch.Generator().manual_seed(19)
    rows = torch.randn(10, 8, generator=generator, dtype=dtype)
    rows[1, :2] = torch.tensor([0.0, -0.0], dtype=dtype)
    if noncontiguous:
        rows = rows.T.contiguous().T
        assert not rows.is_contiguous()
    batches = list(rows.split(lengths))
    originals = [_tensor_bytes(batch) for batch in batches]

    selected = subsample_activations(batches, 4, output_rows=12)

    expected = rows[[1, 2, 6, 8]].clone()
    assert selected.batches[0].dtype == dtype
    assert selected.batches[0].shape == (4, 8)
    assert _tensor_bytes(selected.batches[0]) == _tensor_bytes(expected)
    assert [_tensor_bytes(batch) for batch in batches] == originals
    rows.fill_(42)
    assert _tensor_bytes(selected.batches[0]) == _tensor_bytes(expected)


@pytest.mark.parametrize(
    "lengths,size", [((3, 4, 3), 10), ((3, 4, 3), 12), ((0, 0), 4)]
)
def test_full_row_bypass_does_not_sample_or_gather(monkeypatch, lengths, size):
    batches = [torch.ones(length, 4) for length in lengths]

    def unexpected(*args, **kwargs):
        pytest.fail("Full-row selection must not generate a permutation or gather")

    monkeypatch.setattr(torch, "randperm", unexpected)
    monkeypatch.setattr(activation_subsampling, "_gather_rows", unexpected)

    result = subsample_activations(batches, size, output_rows=7)

    assert result.batches is None
    total = sum(lengths)
    assert result.provenance["k"] == result.provenance["k1"] == total
    indices = torch.arange(total, dtype=torch.int64)
    assert result.provenance["index_sha256"] == hashlib.sha256(
        indices.numpy().tobytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "pinned",
    [
        (False, False, False),
        (True, True, True),
        (True, False, True),
        (False, True, True),
    ],
)
def test_sample_allocation_is_pinned_only_when_every_batch_is_pinned(
    monkeypatch, pinned
):
    batches = _batches()
    flags = {id(batch): flag for batch, flag in zip(batches, pinned)}
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: flags[id(tensor)])
    allocate = torch.empty
    allocations = []

    def record_allocation(*args, **kwargs):
        allocations.append(kwargs["pin_memory"])
        return allocate(*args, **{**kwargs, "pin_memory": False})

    monkeypatch.setattr(torch, "empty", record_allocation)

    result = subsample_activations(batches, 4, output_rows=7)

    assert allocations == [all(pinned)]
    assert result.batches[0][:, 0].tolist() == [1.0, 2.0, 6.0, 8.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned memory requires CUDA")
def test_sample_from_pinned_batches_is_pinned():
    result = subsample_activations(
        [batch.pin_memory() for batch in _batches()], 4, output_rows=7
    )

    assert result.batches[0].is_pinned()
    assert result.batches[0][:, 0].tolist() == [1.0, 2.0, 6.0, 8.0]
