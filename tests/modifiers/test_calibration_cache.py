# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

import pytest
import torch

from llmcompressor_osfp4.modifiers.calibration_cache import (
    OSFP4CalibrationCache,
)


def _tensor_bytes(tensor):
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("cache_activations", [False, True])
@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
@pytest.mark.parametrize("offload_hessians", [False, True])
def test_capture_preserves_exact_statistics_and_input_bytes(
    dtype,
    noncontiguous,
    cache_activations,
    optimization_mode,
    offload_hessians,
):
    generator = torch.Generator().manual_seed(23)
    cache = OSFP4CalibrationCache(offload_hessians=offload_hessians)
    layer = torch.nn.Linear(16, 4, bias=False, dtype=dtype)
    batches = []
    for rows in (2, 3):
        if noncontiguous:
            batch = torch.randn(
                rows, 16, 3, generator=generator, dtype=dtype
            ).transpose(1, 2)
            assert not batch.is_contiguous()
        else:
            batch = torch.randn(rows, 3, 16, generator=generator, dtype=dtype)
        batches.append(batch)

    hook = cache.make_capture_hook(
        "mapping",
        optimization_mode=optimization_mode,
        cache_activations=cache_activations,
    )
    expected_hessian = torch.zeros(16, 16)
    expected_sigma = torch.zeros(16)
    expected_inputs = []
    for batch in batches:
        flat = batch.detach().reshape(-1, batch.shape[-1])
        expected_hessian.addmm_(flat.float().T, flat.float())
        expected_sigma.add_(flat.float().square().sum(dim=0))
        expected_inputs.append(flat.cpu().clone())
        hook(layer, (batch,))

    assert cache.token_count["mapping"] == 15
    assert "mapping" not in cache.capture_events
    if optimization_mode == "sic":
        assert _tensor_bytes(cache.hessian["mapping"]) == _tensor_bytes(
            expected_hessian
        )
        assert cache.hessian["mapping"].device.type == "cpu"
    else:
        assert "mapping" not in cache.hessian
    if optimization_mode == "rtn":
        assert _tensor_bytes(cache.sigma_x_squared_sum["mapping"]) == _tensor_bytes(
            expected_sigma
        )
    else:
        assert "mapping" not in cache.sigma_x_squared_sum
    if cache_activations:
        for cached, expected, source in zip(
            cache.cached_activations["mapping"], expected_inputs, batches
        ):
            assert cached.data_ptr() != source.data_ptr()
            assert cached.dtype == expected.dtype
            assert _tensor_bytes(cached) == _tensor_bytes(expected)
        with torch.no_grad():
            batches[0].fill_(42)
        assert _tensor_bytes(cache.cached_activations["mapping"][0]) == _tensor_bytes(
            expected_inputs[0]
        )
    else:
        assert "mapping" not in cache.cached_activations


@pytest.mark.parametrize("cache_activations", [False, True])
@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
def test_empty_capture_preserves_readiness_semantics(
    cache_activations, optimization_mode
):
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(16, 4, bias=False)
    batch = torch.empty(0, 3, 16)

    cache.make_capture_hook(
        "mapping",
        optimization_mode=optimization_mode,
        cache_activations=cache_activations,
    )(layer, (batch,))

    assert cache.token_count["mapping"] == 0
    if optimization_mode == "sic":
        assert _tensor_bytes(cache.hessian["mapping"]) == _tensor_bytes(
            torch.zeros(16, 16)
        )
        assert not cache.sigma_x_squared_sum
    else:
        assert _tensor_bytes(cache.sigma_x_squared_sum["mapping"]) == _tensor_bytes(
            torch.zeros(16)
        )
        assert not cache.hessian
    assert cache.has_calibration_data("mapping") is cache_activations
    assert "mapping" not in cache.capture_events


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
def test_cleanup_preserves_other_mappings_and_clears_streams(optimization_mode):
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(16, 4, bias=False)
    hook = cache.make_capture_hook("first", optimization_mode=optimization_mode)
    hook(layer, (torch.randn(2, 16),))
    cache.cached_activations["second"] = [torch.ones(1, 16)]
    cache.capture_events["second"] = {}
    cache.token_count["second"] = 1
    cache.capture_streams[torch.device("cuda")] = object()

    cache.clear_mapping("first")

    assert set(cache.cached_activations) == {"second"}
    assert set(cache.capture_events) == {"second"}
    assert set(cache.token_count) == {"second"}
    assert not cache.hessian
    assert not cache.sigma_x_squared_sum
    assert cache.capture_streams

    cache.clear_all()

    assert not any(
        (
            cache.cached_activations,
            cache.capture_events,
            cache.hessian,
            cache.sigma_x_squared_sum,
            cache.token_count,
            cache.capture_streams,
        )
    )


def test_wait_uses_consumer_stream_or_host_synchronization(monkeypatch):
    calls = []

    class Event:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            calls.append(("synchronize", self.name))

    class Stream:
        def wait_event(self, event):
            calls.append(("wait_event", event.name))

    cache = OSFP4CalibrationCache(
        capture_events={
            "mapping": {
                torch.device("cuda:0"): Event("first"),
                torch.device("cuda:1"): Event("second"),
            }
        }
    )
    monkeypatch.setattr(
        torch.get_device_module("cuda"), "current_stream", lambda device: Stream()
    )

    cache.wait_for_capture("mapping", torch.device("cuda"))
    assert calls == [("wait_event", "first"), ("wait_event", "second")]

    calls.clear()
    cache.wait_for_capture("mapping", torch.device("cpu"))
    assert calls == [("synchronize", "first"), ("synchronize", "second")]
    calls.clear()
    cache.wait_for_capture("missing", torch.device("cpu"))
    assert calls == []


def test_capture_stream_is_created_once_per_device(monkeypatch):
    created = []

    def make_stream(device):
        stream = object()
        created.append((device, stream))
        return stream

    cache = OSFP4CalibrationCache()
    monkeypatch.setattr(torch.get_device_module("cuda"), "Stream", make_stream)
    first = torch.device("cuda:0")
    second = torch.device("cuda:1")

    assert cache._get_capture_stream(first) is cache._get_capture_stream(first)
    assert cache._get_capture_stream(second) is cache._get_capture_stream(second)
    assert [device for device, _ in created] == [first, second]


def test_calibration_cache_accumulates_raw_osfp4_hessian():
    generator = torch.Generator().manual_seed(20)
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 7, bias=False)
    batches = [
        torch.randn(2, 3, 16, generator=generator),
        torch.randn(1, 5, 16, generator=generator),
    ]
    hook = cache.make_capture_hook("mapping", optimization_mode="sic")
    for batch in batches:
        hook(module, (batch,))

    expected_H = sum(
        batch.reshape(-1, 16).T @ batch.reshape(-1, 16) for batch in batches
    )

    torch.testing.assert_close(cache.hessian["mapping"], expected_H)
    assert not torch.allclose(cache.hessian["mapping"], 2.0 * expected_H)
    assert [tuple(value.shape) for value in cache.cached_activations["mapping"]] == [
        (6, 16),
        (5, 16),
    ]


def test_calibration_cache_clears_hessian_state():
    cache = OSFP4CalibrationCache(offload_hessians=True)
    module = torch.nn.Linear(16, 4, bias=False)
    cache.make_capture_hook("mapping", optimization_mode="sic")(
        module, (torch.randn(2, 3, 16),)
    )

    assert cache.hessian["mapping"].device.type == "cpu"
    cache.clear_mapping("mapping")

    assert "mapping" not in cache.cached_activations
    assert "mapping" not in cache.hessian


def test_calibration_cache_does_not_duplicate_activation_statistics():
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    cache.make_capture_hook("mapping", optimization_mode="rtn")(
        module, (torch.randn(2, 3, 16),)
    )

    assert "mapping" in cache.cached_activations
    assert not hasattr(cache, "channel_absmax")


def test_weight_only_rtn_streams_sigma_x_squared_without_raw_inputs():
    generator = torch.Generator().manual_seed(21)
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    batches = [
        torch.randn(2, 3, 16, generator=generator),
        torch.randn(1, 5, 16, generator=generator),
    ]
    hook = cache.make_capture_hook(
        "mapping",
        optimization_mode="rtn",
        cache_activations=False,
    )
    for batch in batches:
        hook(module, (batch,))

    concatenated = torch.cat([batch.reshape(-1, 16) for batch in batches])
    torch.testing.assert_close(
        cache.sigma_x_squared_sum["mapping"],
        concatenated.square().sum(dim=0),
    )
    assert cache.token_count["mapping"] == concatenated.shape[0]
    assert cache.has_calibration_data("mapping")
    assert "mapping" not in cache.cached_activations
    assert "mapping" not in cache.hessian


def test_nvfp4_rtn_streams_sigma_x_squared_with_raw_inputs():
    generator = torch.Generator().manual_seed(22)
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    batches = [
        torch.randn(2, 3, 16, generator=generator),
        torch.randn(1, 5, 16, generator=generator),
    ]
    hook = cache.make_capture_hook(
        "mapping",
        optimization_mode="rtn",
        cache_activations=True,
    )
    for batch in batches:
        hook(module, (batch,))

    concatenated = torch.cat([batch.reshape(-1, 16) for batch in batches])
    torch.testing.assert_close(
        cache.sigma_x_squared_sum["mapping"],
        concatenated.square().sum(dim=0),
    )
    assert len(cache.cached_activations["mapping"]) == len(batches)
    assert cache.token_count["mapping"] == concatenated.shape[0]
    assert "mapping" not in cache.hessian


def test_weight_only_sic_streams_hessian_without_raw_inputs():
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    batch = torch.randn(2, 3, 16)
    cache.make_capture_hook(
        "mapping",
        optimization_mode="sic",
        cache_activations=False,
    )(module, (batch,))

    assert "mapping" not in cache.cached_activations
    assert "mapping" not in cache.sigma_x_squared_sum
    assert cache.token_count["mapping"] == 6
    torch.testing.assert_close(
        cache.hessian["mapping"],
        batch.reshape(-1, 16).T @ batch.reshape(-1, 16),
    )


@pytest.mark.parametrize("retain_all", [False, True])
@pytest.mark.parametrize("single_batch", [False, True])
def test_prepare_sampled_activations_waits_for_pending_cpu_buffers(
    retain_all, single_batch
):
    indices = torch.arange(5) if retain_all else torch.tensor([1, 3])
    expected = torch.arange(indices.numel() * 16, dtype=torch.float32).reshape(-1, 16)
    source_batches = (expected,) if single_batch else expected.split(1)
    batches = [torch.full_like(batch, -1) for batch in source_batches]

    class PendingCopy:
        def synchronize(self):
            for destination, source in zip(batches, source_batches):
                destination.copy_(source)

    cache = OSFP4CalibrationCache(
        cached_activations={"mapping": batches},
        capture_events={"mapping": {torch.device("cuda:0"): PendingCopy()}},
        sampled_indices={"mapping": indices},
        total_num_tokens={"mapping": 5},
        token_count={"mapping": 5},
    )
    result = cache.prepare_sampled_activations("mapping", output_rows=4)
    actual = torch.cat(result.batches or cache.cached_activations["mapping"])
    assert _tensor_bytes(actual) == _tensor_bytes(expected)
    assert (result.batches is None) is retain_all


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["single", "strided", "multiple"])
@pytest.mark.parametrize("retain_all", [False, True])
@pytest.mark.parametrize(
    "pinned",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.get_device_module("cuda").is_available(),
                reason="Pinned memory requires CUDA",
            ),
        ),
    ],
)
def test_prepare_sampled_activations_preserves_layout_values_and_cache_ownership(
    dtype, layout, retain_all, pinned
):
    total = 9
    indices = torch.arange(total) if retain_all else torch.tensor([0, 2, 4, 7])
    expected = torch.arange(indices.numel() * 16, dtype=dtype).reshape(-1, 16)
    expected[0, 0] = -0.0
    source = torch.empty(
        (indices.numel(), 32 if layout == "strided" else 16),
        dtype=dtype,
        pin_memory=pinned,
    )
    if layout == "strided":
        source = source[:, ::2]
    source.copy_(expected)
    batches = list(source.split(2)) if layout == "multiple" else [source]
    cache = OSFP4CalibrationCache(
        cached_activations={"mapping": batches},
        sampled_indices={"mapping": indices},
        total_num_tokens={"mapping": total},
        token_count={"mapping": total},
    )
    first = cache.prepare_sampled_activations("mapping", output_rows=4)
    second = cache.prepare_sampled_activations("mapping", output_rows=4)
    assert first.provenance == second.provenance
    if retain_all:
        assert first.batches is second.batches is None
        assert cache.cached_activations["mapping"] is batches
        actual = torch.cat(batches)
    else:
        actual = first.batches[0]
        assert actual is second.batches[0] is cache.cached_activations["mapping"][0]
        assert actual.is_contiguous()
        assert actual.is_pinned() is pinned
        assert (actual is source) is (layout == "single")
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert _tensor_bytes(actual) == _tensor_bytes(expected)
    assert all(
        batch.is_pinned() is pinned for batch in cache.cached_activations["mapping"]
    )


@pytest.mark.parametrize(
    "offload, failure",
    [
        (False, stage)
        for stage in ("allocation", "before_accumulation", "after_accumulation")
    ]
    + [
        (True, stage)
        for stage in (
            "allocation",
            "onload",
            "before_accumulation",
            "after_accumulation",
            "offload",
        )
    ],
)
def test_hessian_capture_failure_preserves_partial_cache(monkeypatch, offload, failure):
    from llmcompressor_osfp4.modifiers import calibration_cache

    cache = OSFP4CalibrationCache(offload_hessians=offload)
    layer = torch.nn.Linear(16, 4)
    inputs = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    buffers = []
    transfers = []
    error = RuntimeError("injected capture failure")
    allocate = calibration_cache.make_empty_hessian
    accumulate = calibration_cache.accumulate_hessian
    tensor_to = torch.Tensor.to

    def allocate_hessian(*args, **kwargs):
        if failure == "allocation":
            raise error
        hessian = allocate(*args, **kwargs)
        buffers.append(hessian)
        return hessian

    def transfer(tensor, *args, **kwargs):
        if not any(tensor is buffer for buffer in buffers):
            return tensor_to(tensor, *args, **kwargs)
        stage = "onload" if not transfers else "offload"
        transfers.append(stage)
        assert cache.hessian["mapping"] is tensor
        if failure == stage:
            raise error
        # Distinct CPU buffers model transfer ownership without requiring CUDA.
        copied = tensor.clone()
        buffers.append(copied)
        return copied

    def accumulate_with_failure(inputs, layer, hessian):
        assert cache.hessian["mapping"] is hessian
        if failure == "before_accumulation":
            raise error
        result = accumulate(inputs, layer, hessian)
        if failure == "after_accumulation":
            raise error
        return result

    monkeypatch.setattr(calibration_cache, "make_empty_hessian", allocate_hessian)
    monkeypatch.setattr(
        calibration_cache, "accumulate_hessian", accumulate_with_failure
    )
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    hook = cache.make_capture_hook("mapping", optimization_mode="sic")
    with pytest.raises(RuntimeError) as raised:
        hook(layer, (inputs,))
    assert raised.value is error
    assert cache.token_count["mapping"] == 3
    assert "mapping" not in cache.cached_activations
    if failure == "allocation":
        assert "mapping" not in cache.hessian
    else:
        assert cache.hessian["mapping"] is buffers[-1]
        expected = (
            inputs.T @ inputs
            if failure in ("after_accumulation", "offload")
            else torch.zeros(16, 16)
        )
        assert _tensor_bytes(cache.hessian["mapping"]) == _tensor_bytes(expected)
