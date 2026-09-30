import hashlib
from dataclasses import dataclass, field

import torch
from torch.nn import Linear

from .activation_subsampling import _ACTIVATION_SUBSAMPLE_SEED, ActivationSubsample
from .optimization.sic import accumulate_hessian, make_empty_hessian

__all__ = ["OSFP4CalibrationCache"]


@dataclass
class OSFP4CalibrationCache:
    """Own mapping inputs, raw statistic sums, sample counts, and capture state."""

    offload_hessians: bool = False
    inputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    events: dict[str, dict[torch.device, torch.cuda.Event]] = field(
        default_factory=dict
    )
    hessian: dict[str, torch.Tensor] = field(default_factory=dict)  # SIC statistic
    sigma_x_squared: dict[str, torch.Tensor] = field(default_factory=dict)  # RTN
    sample_count: dict[str, int] = field(default_factory=dict)
    capture_streams: dict[torch.device, torch.cuda.Stream] = field(default_factory=dict)
    sample_indices: dict[str, torch.Tensor] = field(default_factory=dict)
    expected_tokens: dict[str, int] = field(default_factory=dict)
    input_absmax: dict[str, torch.Tensor] = field(default_factory=dict)

    def validate_sampling(self, name: str) -> None:
        if name not in self.sample_indices:
            return
        observed = self.sample_count.get(name, 0)
        expected = self.expected_tokens[name]
        if observed != expected:
            raise ValueError(
                f"OSFP4 {name}: expected {expected} calibration tokens, got {observed}"
            )
        retained = sum(batch.shape[0] for batch in self.inputs.get(name, ()))
        if retained != self.sample_indices[name].numel():
            raise ValueError(f"OSFP4 {name}: incorrect retained activation count")

    def sampled_inputs(self, name: str, output_rows: int) -> ActivationSubsample:
        """Return the legacy single sampled batch and unchanged provenance."""
        self.validate_sampling(name)
        indices = self.sample_indices[name]
        total = self.expected_tokens[name]
        batches = self.inputs[name]
        selected = None
        if indices.numel() < total:
            sampled = torch.empty(
                (indices.numel(), batches[0].shape[1]),
                dtype=batches[0].dtype,
                device="cpu",
                pin_memory=all(batch.is_pinned() for batch in batches),
            )
            torch.cat(batches, out=sampled)
            self.inputs[name] = [sampled]
            selected = (sampled,)
        return ActivationSubsample(
            selected,
            {
                "policy": "fixed",
                "seed": _ACTIVATION_SUBSAMPLE_SEED,
                "m": output_rows,
                "k": indices.numel(),
                "k1": total,
                "index_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            },
        )

    def clear_all(self) -> None:
        """Clear all cached calibration data and capture streams."""
        self.inputs.clear()
        self.events.clear()
        self.hessian.clear()
        self.sigma_x_squared.clear()
        self.sample_count.clear()
        self.capture_streams.clear()
        self.sample_indices.clear()
        self.expected_tokens.clear()
        self.input_absmax.clear()

    def clear_mapping(self, mapping_name: str) -> None:
        """Clear cached calibration data for one mapping."""
        self.inputs.pop(mapping_name, None)
        self.events.pop(mapping_name, None)
        self.hessian.pop(mapping_name, None)
        self.sigma_x_squared.pop(mapping_name, None)
        self.sample_count.pop(mapping_name, None)
        self.sample_indices.pop(mapping_name, None)
        self.expected_tokens.pop(mapping_name, None)
        self.input_absmax.pop(mapping_name, None)

    def has_observation(self, mapping_name: str) -> bool:
        """Return whether a mapping has captured samples or inputs."""
        return self.sample_count.get(mapping_name, 0) > 0 or bool(
            self.inputs.get(mapping_name)
        )

    def _get_capture_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the reusable CUDA capture stream for a device."""
        stream = self.capture_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.capture_streams[device] = stream
        return stream

    def make_capture_hook(
        self,
        name: str,
        *,
        capture_hessian: bool,
        cache_inputs: bool = True,
        capture_sigma_x_squared: bool = False,
        expected_tokens: int | None = None,
        subsample_size: int | None = None,
    ):
        """Create a hook that captures the requested mapping statistics."""
        streaming = (
            cache_inputs and expected_tokens is not None and subsample_size is not None
        )
        if streaming and (expected_tokens <= 0 or subsample_size <= 0):
            raise ValueError(
                "expected tokens and activation sample size must be positive"
            )

        def update_statistics(
            layer: Linear,
            inputs: torch.Tensor,
            flat: torch.Tensor,
        ) -> None:
            """Accumulate enabled statistics from one captured activation batch."""
            if capture_hessian:
                if name not in self.hessian:
                    device: torch.device | str = (
                        "cpu" if self.offload_hessians else layer.weight.device
                    )
                    self.hessian[name] = make_empty_hessian(layer, device=device)
                if self.offload_hessians:
                    self.hessian[name] = self.hessian[name].to(
                        device=layer.weight.device
                    )
                self.hessian[name] = accumulate_hessian(
                    inputs,
                    layer,
                    self.hessian[name],
                )
                if self.offload_hessians:
                    self.hessian[name] = self.hessian[name].to(device="cpu")
            if capture_sigma_x_squared:
                sigma_x_squared = flat.float().square().sum(dim=0)
                if name not in self.sigma_x_squared:
                    self.sigma_x_squared[name] = torch.zeros_like(sigma_x_squared)
                self.sigma_x_squared[name].add_(sigma_x_squared)
            if streaming and flat.shape[0]:
                maximum = flat.float().abs().amax(dim=0)
                previous = self.input_absmax.get(name)
                self.input_absmax[name] = (
                    maximum if previous is None else torch.maximum(previous, maximum)
                )

        def capture(layer: Linear, args: tuple[torch.Tensor, ...]):
            """Capture one layer input batch and schedule any CUDA copies."""
            inputs = args[0].detach()
            flat = inputs.reshape(-1, inputs.shape[-1])
            start = self.sample_count.get(name, 0)
            self.sample_count[name] = start + flat.shape[0]
            indices = None
            if streaming:
                if name not in self.sample_indices:
                    size = min(expected_tokens, subsample_size)
                    generator = torch.Generator(device="cpu").manual_seed(
                        _ACTIVATION_SUBSAMPLE_SEED
                    )
                    self.sample_indices[name] = (
                        torch.arange(expected_tokens, dtype=torch.int64)
                        if size == expected_tokens
                        else torch.randperm(expected_tokens, generator=generator)[:size]
                        .sort()
                        .values
                    )
                    self.expected_tokens[name] = expected_tokens
                if self.sample_count[name] > expected_tokens:
                    self.validate_sampling(name)
                previous = self.inputs.get(name)
                if previous and previous[0].dtype != flat.dtype:
                    raise ValueError("cached activation batches must share one dtype")
                selected = self.sample_indices[name]
                if selected.numel() < expected_tokens:
                    lo, hi = torch.searchsorted(
                        selected, torch.tensor([start, start + flat.shape[0]])
                    ).tolist()
                    indices = selected[lo:hi] - start
            rows = flat.shape[0] if indices is None else indices.numel()
            retain = cache_inputs and (indices is None or rows > 0)

            if flat.device.type == "cuda":
                cached_activation = None
                if retain:
                    cached_activation = (
                        torch.empty_like(flat, device="cpu", pin_memory=True)
                        if indices is None
                        else torch.empty(
                            (rows, flat.shape[1]),
                            dtype=flat.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                    )
                compute_stream = torch.cuda.current_stream(flat.device)
                source_ready = torch.cuda.Event()
                source_ready.record(compute_stream)
                capture_stream = self._get_capture_stream(flat.device)
                with torch.cuda.stream(capture_stream):
                    capture_stream.wait_event(source_ready)
                    update_statistics(layer, inputs, flat)
                    if cached_activation is not None:
                        source = (
                            flat
                            if indices is None
                            else flat.index_select(0, indices.to(flat.device))
                        )
                        cached_activation.copy_(source, non_blocking=True)
                    ready_event = torch.cuda.Event()
                    ready_event.record(capture_stream)
                flat.record_stream(capture_stream)
                # One ordered capture stream per device: its newest event covers
                # all earlier captures for this mapping on that device.
                self.events.setdefault(name, {})[flat.device] = ready_event
            else:
                update_statistics(layer, inputs, flat)
                cached_activation = None
                if retain:
                    cached_activation = (
                        flat.to(device="cpu", copy=True)
                        if indices is None
                        else flat.index_select(0, indices.to(flat.device)).cpu()
                    )
            if cached_activation is not None:
                self.inputs.setdefault(name, []).append(cached_activation)

        return capture

    def wait(self, mapping_name: str, device: torch.device) -> None:
        """Wait for outstanding capture events on the consuming device."""
        events = self.events.get(mapping_name, {}).values()
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            for event in events:
                stream.wait_event(event)
            return
        for event in events:
            event.synchronize()
