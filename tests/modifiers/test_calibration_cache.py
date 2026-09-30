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
@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize(
    "capture_hessian,capture_sigma_x_squared",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("offload_hessians", [False, True])
def test_capture_preserves_exact_statistics_and_input_bytes(
    dtype,
    noncontiguous,
    cache_inputs,
    capture_hessian,
    capture_sigma_x_squared,
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
        capture_hessian=capture_hessian,
        cache_inputs=cache_inputs,
        capture_sigma_x_squared=capture_sigma_x_squared,
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

    assert cache.sample_count["mapping"] == 15
    assert "mapping" not in cache.events
    if capture_hessian:
        assert _tensor_bytes(cache.hessian["mapping"]) == _tensor_bytes(
            expected_hessian
        )
        assert cache.hessian["mapping"].device.type == "cpu"
    else:
        assert "mapping" not in cache.hessian
    if capture_sigma_x_squared:
        assert _tensor_bytes(cache.sigma_x_squared["mapping"]) == _tensor_bytes(
            expected_sigma
        )
    else:
        assert "mapping" not in cache.sigma_x_squared
    if cache_inputs:
        for cached, expected, source in zip(
            cache.inputs["mapping"], expected_inputs, batches
        ):
            assert cached.data_ptr() != source.data_ptr()
            assert cached.dtype == expected.dtype
            assert _tensor_bytes(cached) == _tensor_bytes(expected)
        with torch.no_grad():
            batches[0].fill_(42)
        assert _tensor_bytes(cache.inputs["mapping"][0]) == _tensor_bytes(
            expected_inputs[0]
        )
    else:
        assert "mapping" not in cache.inputs


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_empty_capture_preserves_readiness_semantics(cache_inputs):
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(16, 4, bias=False)
    batch = torch.empty(0, 3, 16)

    cache.make_capture_hook(
        "mapping",
        capture_hessian=True,
        cache_inputs=cache_inputs,
        capture_sigma_x_squared=True,
    )(layer, (batch,))

    assert cache.sample_count["mapping"] == 0
    assert _tensor_bytes(cache.hessian["mapping"]) == _tensor_bytes(torch.zeros(16, 16))
    assert _tensor_bytes(cache.sigma_x_squared["mapping"]) == _tensor_bytes(
        torch.zeros(16)
    )
    assert cache.has_observation("mapping") is cache_inputs
    assert "mapping" not in cache.events


def test_cleanup_preserves_other_mappings_and_clears_streams():
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(16, 4, bias=False)
    hook = cache.make_capture_hook(
        "first", capture_hessian=True, capture_sigma_x_squared=True
    )
    hook(layer, (torch.randn(2, 16),))
    cache.inputs["second"] = [torch.ones(1, 16)]
    cache.events["second"] = {}
    cache.sample_count["second"] = 1
    cache.capture_streams[torch.device("cuda")] = object()

    cache.clear_mapping("first")

    assert set(cache.inputs) == {"second"}
    assert set(cache.events) == {"second"}
    assert set(cache.sample_count) == {"second"}
    assert not cache.hessian
    assert not cache.sigma_x_squared
    assert cache.capture_streams

    cache.clear_all()

    assert not any(
        (
            cache.inputs,
            cache.events,
            cache.hessian,
            cache.sigma_x_squared,
            cache.sample_count,
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
        events={
            "mapping": {
                torch.device("cuda:0"): Event("first"),
                torch.device("cuda:1"): Event("second"),
            }
        }
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())

    cache.wait("mapping", torch.device("cuda"))
    assert calls == [("wait_event", "first"), ("wait_event", "second")]

    calls.clear()
    cache.wait("mapping", torch.device("cpu"))
    assert calls == [("synchronize", "first"), ("synchronize", "second")]
    calls.clear()
    cache.wait("missing", torch.device("cpu"))
    assert calls == []


def test_capture_stream_is_created_once_per_device(monkeypatch):
    created = []

    def make_stream(device):
        stream = object()
        created.append((device, stream))
        return stream

    cache = OSFP4CalibrationCache()
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
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
    hook = cache.make_capture_hook("mapping", capture_hessian=True)
    for batch in batches:
        hook(module, (batch,))

    expected_H = sum(
        batch.reshape(-1, 16).T @ batch.reshape(-1, 16) for batch in batches
    )

    torch.testing.assert_close(cache.hessian["mapping"], expected_H)
    assert not torch.allclose(cache.hessian["mapping"], 2.0 * expected_H)
    assert [tuple(value.shape) for value in cache.inputs["mapping"]] == [
        (6, 16),
        (5, 16),
    ]


def test_calibration_cache_clears_hessian_state():
    cache = OSFP4CalibrationCache(offload_hessians=True)
    module = torch.nn.Linear(16, 4, bias=False)
    cache.make_capture_hook("mapping", capture_hessian=True)(
        module, (torch.randn(2, 3, 16),)
    )

    assert cache.hessian["mapping"].device.type == "cpu"
    cache.clear_mapping("mapping")

    assert "mapping" not in cache.inputs
    assert "mapping" not in cache.hessian


def test_calibration_cache_does_not_duplicate_activation_statistics():
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    cache.make_capture_hook("mapping", capture_hessian=False)(
        module, (torch.randn(2, 3, 16),)
    )

    assert "mapping" in cache.inputs
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
        capture_hessian=False,
        cache_inputs=False,
        capture_sigma_x_squared=True,
    )
    for batch in batches:
        hook(module, (batch,))

    concatenated = torch.cat([batch.reshape(-1, 16) for batch in batches])
    torch.testing.assert_close(
        cache.sigma_x_squared["mapping"],
        concatenated.square().sum(dim=0),
    )
    assert cache.sample_count["mapping"] == concatenated.shape[0]
    assert cache.has_observation("mapping")
    assert "mapping" not in cache.inputs
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
        capture_hessian=False,
        cache_inputs=True,
        capture_sigma_x_squared=True,
    )
    for batch in batches:
        hook(module, (batch,))

    concatenated = torch.cat([batch.reshape(-1, 16) for batch in batches])
    torch.testing.assert_close(
        cache.sigma_x_squared["mapping"],
        concatenated.square().sum(dim=0),
    )
    assert len(cache.inputs["mapping"]) == len(batches)
    assert cache.sample_count["mapping"] == concatenated.shape[0]
    assert "mapping" not in cache.hessian


def test_weight_only_sic_streams_hessian_without_raw_inputs():
    cache = OSFP4CalibrationCache()
    module = torch.nn.Linear(16, 4, bias=False)
    batch = torch.randn(2, 3, 16)
    cache.make_capture_hook(
        "mapping",
        capture_hessian=True,
        cache_inputs=False,
    )(module, (batch,))

    assert "mapping" not in cache.inputs
    assert "mapping" not in cache.sigma_x_squared
    assert cache.sample_count["mapping"] == 6
    torch.testing.assert_close(
        cache.hessian["mapping"],
        batch.reshape(-1, 16).T @ batch.reshape(-1, 16),
    )
