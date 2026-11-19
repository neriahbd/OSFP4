# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.nn import Linear

from .activation_subsampling import (
    ActivationSubsample,
    _build_subsample_provenance,
    _select_activation_indices,
)
from .optimization.sic import accumulate_hessian, make_empty_hessian

__all__ = ["OSFP4CalibrationCache"]


@dataclass
class OSFP4CalibrationCache:
    """Store calibration data and capture state for each OSFP4 mapping.

    Flow: receive batch activations -> identify sampled token rows for optimization
    -> accumulate statistics -> prepare activations -> optimize in the modifier
    -> clear the mapping's cached data.

    Each batch contributes statistics from all tokens, while retained activation
    rows are cached to evaluate quantization reconstruction loss during
    optimization. Preparation waits for capture to finish and combines sampled
    batches for optimization. Weight-only calibration stores statistics without
    caching activations.
    """

    offload_hessians: bool = False

    # Total tokens used to select activation subsampling indices before capture.
    total_num_tokens: dict[str, int] = field(default_factory=dict)
    sampled_indices: dict[str, torch.Tensor] = field(default_factory=dict)
    token_count: dict[str, int] = field(default_factory=dict)

    # CUDA queues for accumulating statistics and copying activations to CPU.
    capture_streams: dict[torch.device, torch.Stream] = field(default_factory=dict)
    # Completion markers used to wait until captured data is ready to use.
    capture_events: dict[str, dict[torch.device, torch.Event]] = field(default_factory=dict)

    # Sum of token outer products; divide by token_count for Sigma_X (SIC).
    hessian: dict[str, torch.Tensor] = field(default_factory=dict)
    # Sum of squared activations per channel; divide by token_count for sigma_X² (RTN).
    sigma_x_squared_sum: dict[str, torch.Tensor] = field(default_factory=dict)
    # Used to update the input global scale after smoothing.
    input_channel_absmax: dict[str, torch.Tensor] = field(default_factory=dict)

    cached_activations: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def clear_all(self) -> None:
        """Clear all cached calibration data and capture streams."""
        self.cached_activations.clear()
        self.capture_events.clear()
        self.hessian.clear()
        self.sigma_x_squared_sum.clear()
        self.token_count.clear()
        self.capture_streams.clear()
        self.sampled_indices.clear()
        self.total_num_tokens.clear()
        self.input_channel_absmax.clear()

    def make_capture_hook(
        self,
        mapping_name: str,
        *,
        optimization_mode: Literal["rtn", "sic"],
        cache_activations: bool = True,
        total_num_tokens: int | None = None,
        activation_subsample_size: int | None = None,
    ) -> Callable[[Linear, tuple[torch.Tensor, ...]], None]:
        """Create a per-batch hook to collect calibration statistics and activations.

        Use all tokens for SIC/RTN statistics and retain activation rows as configured.
        The caller registers the returned hook on the mapping's first balance layer.
        """
        subsample_during_capture = (
            cache_activations
            and total_num_tokens is not None
            and activation_subsample_size is not None
        )
        if subsample_during_capture and (
            total_num_tokens <= 0 or activation_subsample_size <= 0
        ):
            raise ValueError(
                "expected tokens and activation sample size must be positive"
            )

        def capture_calibration_batch_hook(layer: Linear, args: tuple[torch.Tensor, ...]) -> None:
            """Count tokens and capture calibration data from one input batch.

            Accumulate statistics from all rows and cache selected activations
            for the quantization reconstruction loss. Full caching retains all
            rows; weight-only calibration retains statistics without activations.
            """
            activations = args[0].detach()
            activation_rows = activations.reshape(-1, activations.shape[-1])
            batch_token_offset = self.token_count.get(mapping_name, 0)
            self.token_count[mapping_name] = batch_token_offset + activation_rows.shape[0]
            batch_sampled_indices = None
            if subsample_during_capture:
                batch_sampled_indices = self._get_batch_sampled_indices(
                    mapping_name,
                    activation_rows,
                    batch_token_offset,
                    total_num_tokens,
                    activation_subsample_size,
                )
            num_rows_to_cache = (
                activation_rows.shape[0]
                if batch_sampled_indices is None
                else batch_sampled_indices.numel()
            )
            retain_activations = cache_activations and (
                batch_sampled_indices is None or num_rows_to_cache > 0
            )

            if activation_rows.device.type == "cuda":
                cached_batch = self._capture_calibration_batch_cuda(
                    mapping_name,
                    layer,
                    activations,
                    activation_rows,
                    batch_sampled_indices,
                    retain_activations=retain_activations,
                    optimization_mode=optimization_mode,
                    subsample_during_capture=subsample_during_capture,
                )
            else:
                self._accumulate_calibration_statistics(
                    mapping_name,
                    layer,
                    activations,
                    activation_rows,
                    optimization_mode=optimization_mode,
                    subsample_during_capture=subsample_during_capture,
                )
                cached_batch = None
                if retain_activations:
                    cached_batch = (
                        activation_rows.to(device="cpu", copy=True)
                        if batch_sampled_indices is None
                        else activation_rows.index_select(
                            0, batch_sampled_indices.to(activation_rows.device)
                        ).cpu()
                    )
            if cached_batch is not None:
                self.cached_activations.setdefault(mapping_name, []).append(
                    cached_batch
                )

        return capture_calibration_batch_hook

    def _get_batch_sampled_indices(
        self,
        mapping_name: str,
        activation_rows: torch.Tensor,
        batch_token_offset: int,
        total_num_tokens: int,
        activation_subsample_size: int,
    ) -> torch.Tensor | None:
        """Locate and return sampled token indices relative to the current batch.

        Return None to keep all rows, or an empty tensor to keep none.
        """
        if mapping_name not in self.sampled_indices:
            self.sampled_indices[mapping_name] = _select_activation_indices(
                total_num_tokens, activation_subsample_size
            )
            self.total_num_tokens[mapping_name] = total_num_tokens
        if self.token_count[mapping_name] > total_num_tokens:
            self.validate_sampling(mapping_name)
        cached_batches = self.cached_activations.get(mapping_name)
        if cached_batches and cached_batches[0].dtype != activation_rows.dtype:
            raise ValueError("cached activation batches must share one dtype")
        sampled_indices = self.sampled_indices[mapping_name]
        if sampled_indices.numel() < total_num_tokens:
            index_start, index_end = torch.searchsorted(
                sampled_indices,
                torch.tensor([batch_token_offset, batch_token_offset + activation_rows.shape[0]]),
            ).tolist()
            return sampled_indices[index_start:index_end] - batch_token_offset
        return None

    def _accumulate_calibration_statistics(
        self,
        mapping_name: str,
        layer: Linear,
        activations: torch.Tensor,
        activation_rows: torch.Tensor,
        *,
        optimization_mode: Literal["rtn", "sic"],
        subsample_during_capture: bool,
    ) -> None:
        """Accumulate enabled statistics from one captured activation batch."""
        if optimization_mode == "sic":
            if mapping_name not in self.hessian:
                device: torch.device | str = (
                    "cpu" if self.offload_hessians else layer.weight.device
                )
                self.hessian[mapping_name] = make_empty_hessian(layer, device=device)
            hessian = self.hessian[mapping_name]
            if self.offload_hessians:
                hessian = hessian.to(device=layer.weight.device)
                self.hessian[mapping_name] = hessian
            hessian = accumulate_hessian(activations, layer, hessian)
            self.hessian[mapping_name] = hessian
            if self.offload_hessians:
                self.hessian[mapping_name] = hessian.to(device="cpu")
        else:
            sigma_x_squared_sum = activation_rows.float().square().sum(dim=0)
            if mapping_name not in self.sigma_x_squared_sum:
                self.sigma_x_squared_sum[mapping_name] = torch.zeros_like(sigma_x_squared_sum)
            self.sigma_x_squared_sum[mapping_name].add_(sigma_x_squared_sum)
        if subsample_during_capture and activation_rows.shape[0]:
            input_channel_absmax = activation_rows.float().abs().amax(dim=0)
            previous_absmax = self.input_channel_absmax.get(mapping_name)
            self.input_channel_absmax[mapping_name] = (
                input_channel_absmax
                if previous_absmax is None
                else torch.maximum(previous_absmax, input_channel_absmax)
            )

    def _capture_calibration_batch_cuda(
        self,
        mapping_name: str,
        layer: Linear,
        activations: torch.Tensor,
        activation_rows: torch.Tensor,
        batch_sampled_indices: torch.Tensor | None,
        *,
        retain_activations: bool,
        optimization_mode: Literal["rtn", "sic"],
        subsample_during_capture: bool,
    ) -> torch.Tensor | None:
        """Schedule statistics and optional activation copies on the capture stream."""
        device = activation_rows.device
        cached_batch = None
        if retain_activations:
            if batch_sampled_indices is None:
                cached_batch = torch.empty_like(
                    activation_rows, device="cpu", pin_memory=True
                )
            else:
                cached_batch = torch.empty(
                    (batch_sampled_indices.numel(), activation_rows.shape[1]),
                    dtype=activation_rows.dtype,
                    device="cpu",
                    pin_memory=True,
                )

        cuda = torch.get_device_module(device)
        compute_stream = cuda.current_stream(device)
        activations_ready_event = cuda.Event()
        activations_ready_event.record(compute_stream)
        capture_stream = self._get_capture_stream(device)
        with cuda.stream(capture_stream):
            capture_stream.wait_event(activations_ready_event)
            self._accumulate_calibration_statistics(
                mapping_name,
                layer,
                activations,
                activation_rows,
                optimization_mode=optimization_mode,
                subsample_during_capture=subsample_during_capture,
            )
            if cached_batch is not None:
                selected_activations = activation_rows
                if batch_sampled_indices is not None:
                    selected_activations = activation_rows.index_select(
                        0, batch_sampled_indices.to(device)
                    )
                cached_batch.copy_(selected_activations, non_blocking=True)
            capture_complete_event = cuda.Event()
            capture_complete_event.record(capture_stream)

        # Prevent either buffer from being reused before capture finishes.
        activations.record_stream(capture_stream)
        activation_rows.record_stream(capture_stream)
        # The latest event covers all earlier work on this capture stream.
        self.capture_events.setdefault(mapping_name, {})[device] = (
            capture_complete_event
        )
        return cached_batch

    def _get_capture_stream(self, device: torch.device) -> torch.Stream:
        """Return the device's CUDA capture stream, creating it on first use."""
        capture_stream = self.capture_streams.get(device)
        if capture_stream is None:
            capture_stream = torch.get_device_module(device).Stream(device=device)
            self.capture_streams[device] = capture_stream
        return capture_stream

    def has_calibration_data(self, mapping_name: str) -> bool:
        """Return whether the mapping has processed tokens or cached activation batches."""
        return self.token_count.get(mapping_name, 0) > 0 or bool(
            self.cached_activations.get(mapping_name)
        )

    def validate_sampling(self, mapping_name: str) -> None:
        """Check token and retained-row counts match the streaming sampling plan."""
        if mapping_name not in self.sampled_indices:
            return
        token_count = self.token_count.get(mapping_name, 0)
        total_num_tokens = self.total_num_tokens[mapping_name]
        if token_count != total_num_tokens:
            raise ValueError(
                f"OSFP4 {mapping_name}: expected {total_num_tokens} calibration tokens, got {token_count}"
            )
        num_cached_tokens = sum(
            len(batch) for batch in self.cached_activations.get(mapping_name, ())
        )
        if num_cached_tokens != self.sampled_indices[mapping_name].numel():
            raise ValueError(
                f"OSFP4 {mapping_name}: incorrect retained activation count"
            )

    def prepare_sampled_activations(self, mapping_name: str, output_rows: int) -> ActivationSubsample:
        """Wait, validate, and combine already sampled rows for optimization.

        Return the subset and sampling metadata, or no subset if all tokens were kept.
        """
        self.wait_for_capture(mapping_name, torch.device("cpu"))
        self.validate_sampling(mapping_name)
        sampled_indices = self.sampled_indices[mapping_name]
        total_num_tokens = self.total_num_tokens[mapping_name]
        cached_activations = self.cached_activations[mapping_name]
        provenance = _build_subsample_provenance(
            sampled_indices, total_num_tokens, output_rows
        )
        if sampled_indices.numel() >= total_num_tokens:
            return ActivationSubsample(None, provenance)

        sampled_activations = cached_activations[0]
        if len(cached_activations) > 1 or not sampled_activations.is_contiguous():
            sampled_activations = torch.empty(
                (sampled_indices.numel(), sampled_activations.shape[1]),
                dtype=sampled_activations.dtype,
                device="cpu",
                pin_memory=all(batch.is_pinned() for batch in cached_activations),
            )
            torch.cat(cached_activations, out=sampled_activations)
        self.cached_activations[mapping_name] = [sampled_activations]
        return ActivationSubsample((sampled_activations,), provenance)

    def wait_for_capture(self, mapping_name: str, device: torch.device) -> None:
        """Wait for outstanding capture events on the consuming device."""
        capture_events = self.capture_events.get(mapping_name, {}).values()
        if device.type == "cuda":
            consumer_stream = torch.get_device_module(device).current_stream(device)
            for capture_event in capture_events:
                consumer_stream.wait_event(capture_event)
            return
        for capture_event in capture_events:
            capture_event.synchronize()

    def clear_mapping(self, mapping_name: str) -> None:
        """Clear cached calibration data for one mapping."""
        self.cached_activations.pop(mapping_name, None)
        self.capture_events.pop(mapping_name, None)
        self.hessian.pop(mapping_name, None)
        self.sigma_x_squared_sum.pop(mapping_name, None)
        self.token_count.pop(mapping_name, None)
        self.sampled_indices.pop(mapping_name, None)
        self.total_num_tokens.pop(mapping_name, None)
        self.input_channel_absmax.pop(mapping_name, None)
