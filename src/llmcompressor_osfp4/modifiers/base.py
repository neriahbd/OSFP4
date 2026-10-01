import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from compressed_tensors.offload.dist_utils import is_distributed
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationConfig,
    QuantizationStrategy,
)
from compressed_tensors.utils import (
    match_modules_set,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr
from torch.nn import Linear, Module
from torch.utils._pytree import tree_leaves
from torch.utils.data import IterableDataset
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_qparams
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    get_layer_mappings_from_model,
)
from llmcompressor.observers.min_max import StaticMinMaxObserver
from llmcompressor.utils.pytorch.module import get_module_to_name_dict

from .activation_subsampling import subsample_activations
from .calibration_cache import OSFP4CalibrationCache
from .osfp4_quantize import quantize_mapping
from .runtime_contract import attach_osfp4_runtime_contract

__all__ = ["OSFP4Mapping", "OSFP4Modifier"]

_OSFP4_OBSERVER = "osfp4"


@dataclass(frozen=True, slots=True)
class OSFP4Mapping:
    """One complete alpha optimization and SmoothQuant deployment unit."""

    mapping_name: str
    smooth_layer: Module | None
    balance_layers: tuple[Linear, ...]

    @property
    def requires_runtime_smoothing(self) -> bool:
        return self.smooth_layer is None


class OSFP4Modifier(Modifier, QuantizationMixin):
    """Optimize NVFP4 per mapping and deploy smooth-layer or runtime smoothing.

    Temporary runtime hooks preserve smoothing during sequential propagation;
    finalization removes the hooks and records the serving contract.
    """

    # Configuration

    # Calibration is always required, so pipeline inference selects sequential.
    requires_calibration_data: bool = True
    optimization_mode: Literal["rtn", "sic"] = "sic"
    steps: int = 80
    lr: float = 0.12
    dampening_frac: float = 0.01
    offload_hessians: bool = False
    activation_subsample_size: int | Literal["auto"] | None = "auto"

    _resolved_mappings: list[OSFP4Mapping] = PrivateAttr(default_factory=list)
    _calibration_dataloader: object = PrivateAttr(default=None)
    _calibration: OSFP4CalibrationCache = PrivateAttr(
        default_factory=OSFP4CalibrationCache
    )
    _optimized_mapping_names: set[str] = PrivateAttr(default_factory=set)
    _finalization_complete: bool = PrivateAttr(default=False)
    _deployment_failure: tuple[str, str] | None = PrivateAttr(default=None)
    _weight_only: bool = PrivateAttr(default=False)
    _activation_subsampling_records: dict[str, dict[str, int | str]] = PrivateAttr(
        default_factory=dict
    )
    _runtime_smoothing_hooks: dict[Module, RemovableHandle] = PrivateAttr(
        default_factory=dict
    )

    # 1. Initialization: resolve mappings, reset state, and initialize quantization.

    def on_initialize(self, state: State, **kwargs) -> bool:
        self._require_usable()
        self._resolved_mappings = self._resolve_mappings(state.model)
        self._reset_calibration_state()
        # State may contain a deep copy; count the loader the pipeline will consume.
        self._calibration_dataloader = kwargs.get("calib_data", state.data.calib)
        self._calibration.offload_hessians = self.offload_hessians
        QuantizationMixin.initialize_quantization(self, state.model)
        return True

    def _resolve_mappings(self, model: Module) -> list[OSFP4Mapping]:
        """Resolve smooth-layer mappings, then runtime mappings, in order."""
        selected = list(
            match_named_modules(model, self.resolved_weight_targets, self.ignore)
        )
        selected_balance_layers = {layer for _, layer in selected}
        module_names = get_module_to_name_dict(model)
        mappings = self._resolve_smooth_layer_mappings(
            model, selected_balance_layers, module_names
        )
        smooth_layer_count = len(mappings)
        covered = {layer for mapping in mappings for layer in mapping.balance_layers}
        mappings.extend(
            OSFP4Mapping(
                mapping_name=name,
                smooth_layer=None,
                balance_layers=(layer,),
            )
            for name, layer in selected
            if layer not in covered
        )
        logger.info(
            f"Resolved OSFP4 architecture {model.__class__.__name__}: "
            f"{smooth_layer_count} smooth-layer mappings and "
            f"{len(mappings) - smooth_layer_count} "
            "runtime mappings"
        )
        return mappings

    @property
    def resolved_weight_targets(self) -> set[str]:
        """Return all resolved weight targets across configuration groups."""
        return {
            target
            for group in self.resolved_config.config_groups.values()
            for target in group.targets
        }

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = QuantizationMixin.resolve_quantization_config(self)
        observer_kwargs = {
            "mode": self.optimization_mode,
            "num_iters": self.steps,
            "lr": self.lr,
        }
        weight_only_groups = set()
        for scheme in config.config_groups.values():
            inputs = scheme.input_activations
            weight_only_groups.add(inputs is None)
            scheme.weights.observer = _OSFP4_OBSERVER
            scheme.weights.observer_kwargs = observer_kwargs.copy()
        if len(weight_only_groups) > 1:
            raise ValueError("OSFP4 cannot mix NVFP4 and NVFP4A16 config groups")
        self._weight_only = next(iter(weight_only_groups), False)
        self.weight_observer = _OSFP4_OBSERVER
        if self._weight_only:
            self.activation_subsample_size = None
        return config

    def _resolve_smooth_layer_mappings(
        self,
        model: Module,
        selected_balance_layers: set[Module],
        module_names: dict[Module, str],
    ) -> list[OSFP4Mapping]:
        mappings = []
        for spec in get_layer_mappings_from_model(model):
            for *nested_balance_layers, smooth_layers in match_modules_set(
                model, tree_leaves(spec)
            ):
                if len(smooth_layers) > 1:
                    raise ValueError(
                        "OSFP4 must match a single smooth layer per mapping; "
                        f"got {[module_names.get(layer) for layer in smooth_layers]} "
                        f"for {spec}"
                    )
                if not smooth_layers:
                    continue
                smooth_layer = smooth_layers[0]
                balance_layers = tuple(tree_leaves(nested_balance_layers))
                if not balance_layers:
                    logger.warning(
                        f"Skipping OSFP4 mapping for {module_names[smooth_layer]}: "
                        "no balance layers"
                    )
                    continue
                if set(balance_layers) <= selected_balance_layers:
                    mappings.append(
                        OSFP4Mapping(
                            mapping_name=module_names[smooth_layer],
                            smooth_layer=smooth_layer,
                            balance_layers=balance_layers,
                        )
                    )
        return mappings

    def _reset_calibration_state(self) -> None:
        """Clear temporary calibration state before initialization."""
        self._calibration.clear_all()
        self._optimized_mapping_names.clear()
        self._activation_subsampling_records.clear()
        self._remove_runtime_smoothing_hooks()
        self._finalization_complete = False

    # 2. Calibration setup: register capture hooks for sequential forward passes.

    @staticmethod
    def _calibration_token_count(dataloader) -> int | None:
        """Count collated token rows without model forwards or advancing RNGs."""
        if (
            dataloader is None
            or isinstance(dataloader, Iterator)
            or isinstance(getattr(dataloader, "dataset", None), IterableDataset)
        ):
            return None
        # Persistent worker RNGs cannot be restored from the main process.
        if getattr(dataloader, "persistent_workers", False):
            return None
        generators = {
            generator
            for owner in (dataloader, getattr(dataloader, "sampler", None))
            if isinstance(
                generator := getattr(owner, "generator", None), torch.Generator
            )
        }
        generator_states = {g: g.get_state() for g in generators}
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_initialized()
            else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                count = 0
                for batch in dataloader:
                    if not isinstance(batch, Mapping):
                        return None
                    inputs = batch.get("input_ids")
                    if isinstance(inputs, torch.Tensor):
                        count += inputs.numel()
                    else:
                        inputs = batch.get("inputs_embeds")
                        if not isinstance(inputs, torch.Tensor):
                            return None
                        count += inputs.numel() // inputs.shape[-1]
                return count or None
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            for generator, rng in generator_states.items():
                generator.set_state(rng)

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        if is_distributed():
            raise NotImplementedError(
                "OSFP4 calibration is single-process; use tensor parallelism only "
                "when serving the produced checkpoint"
            )
        QuantizationMixin.start_calibration(self, state.model)
        expected_tokens = (
            self._calibration_token_count(self._calibration_dataloader)
            if self.activation_subsample_size is not None and not self._weight_only
            else None
        )
        self._calibration_dataloader = None
        if (
            expected_tokens is None
            and state.data.calib is not None
            and not self._weight_only
            and self.activation_subsample_size is not None
        ):
            logger.warning(
                "OSFP4 cannot count calibration token rows; "
                "using full activation caching"
            )
        subsample_sizes = [
            self._resolve_activation_subsample_size(mapping)[0]
            for mapping in self._resolved_mappings
        ]
        for mapping, subsample_size in zip(self._resolved_mappings, subsample_sizes):
            streaming = (
                expected_tokens is not None
                and subsample_size is not None
                and not self._weight_only
                and all(
                    type(getattr(layer, "input_observer", None)) is StaticMinMaxObserver
                    and layer.input_observer.args.strategy
                    == QuantizationStrategy.TENSOR_GROUP
                    and layer.input_observer.args.dynamic == DynamicType.LOCAL
                    for layer in mapping.balance_layers
                )
            )
            self.register_hook(
                mapping.balance_layers[0],
                self._calibration.make_capture_hook(
                    mapping.mapping_name,
                    capture_hessian=self.optimization_mode == "sic",
                    cache_inputs=not self._weight_only,
                    capture_sigma_x_squared=self.optimization_mode == "rtn",
                    expected_tokens=expected_tokens if streaming else None,
                    subsample_size=subsample_size,
                ),
                "forward_pre",
            )

    # 3. Optimization after each subgraph: sample, optimize, deploy, and clear caches.

    def on_sequential_epoch_end(
        self,
        state: State,
        event: Event,
        modules: list[Module],
        **kwargs,
    ):
        self._optimize_available_mappings()

    def _optimize_available_mappings(self) -> None:
        """Optimize each observed mapping that has not yet been processed."""
        self._require_usable()
        available = [
            mapping
            for mapping in self._resolved_mappings
            if self._calibration.has_observation(mapping.mapping_name)
            and mapping.mapping_name not in self._optimized_mapping_names
        ]
        for mapping in available:
            self._calibration.validate_sampling(mapping.mapping_name)
        for mapping in tqdm(available, desc="OSFP4 layer optimization"):
            self._optimize_mapping(mapping)

    def _optimize_mapping(self, mapping: OSFP4Mapping) -> None:
        optimization_input_batches = self._sample_optimization_inputs(mapping)
        self._deploy_mapping(mapping, optimization_input_batches)
        self._optimized_mapping_names.add(mapping.mapping_name)
        self._calibration.clear_mapping(mapping.mapping_name)

    def _sample_optimization_inputs(
        self, mapping: OSFP4Mapping
    ) -> tuple[torch.Tensor, ...] | None:
        subsample_size, input_width = self._resolve_activation_subsample_size(mapping)
        if subsample_size is None:
            return None

        # Synchronize D2H capture before sampling CPU rows from pinned buffers.
        self._calibration.wait(mapping.mapping_name, torch.device("cpu"))
        output_rows = sum(layer.weight.shape[0] for layer in mapping.balance_layers)
        if mapping.mapping_name in self._calibration.sample_indices:
            selection = self._calibration.sampled_inputs(
                mapping.mapping_name, output_rows
            )
        else:
            selection = subsample_activations(
                self._calibration.inputs[mapping.mapping_name],
                subsample_size,
                output_rows=output_rows,
            )
        provenance = selection.provenance.copy()
        if input_width is not None:
            provenance["policy"] = "auto"
            provenance["n"] = input_width
        self._activation_subsampling_records[mapping.mapping_name] = provenance
        return selection.batches

    def _resolve_activation_subsample_size(
        self, mapping: OSFP4Mapping
    ) -> tuple[int | None, int | None]:
        """Resolve the row cap and the input width recorded for auto sampling."""
        if self.activation_subsample_size != "auto":
            return self.activation_subsample_size, None
        input_widths = {layer.weight.shape[1] for layer in mapping.balance_layers}
        if len(input_widths) != 1:
            raise ValueError(
                f"OSFP4 mapping {mapping.mapping_name!r} cannot use auto activation "
                "subsampling because its balance layers have different input widths: "
                f"{sorted(input_widths)}"
            )
        input_width = input_widths.pop()
        return 2 * input_width, input_width

    def _deploy_mapping(
        self,
        mapping: OSFP4Mapping,
        optimization_input_batches: tuple[torch.Tensor, ...] | None,
    ) -> None:
        stage = None

        def deployment_stage(value: str) -> None:
            nonlocal stage
            stage = value

        try:
            deployment = quantize_mapping(
                mapping,
                self._calibration,
                mode=self.optimization_mode,
                dampening_frac=self.dampening_frac,
                weight_only=self._weight_only,
                optimization_input_batches=optimization_input_batches,
                _deployment_stage=deployment_stage,
            )
            deployment_stage("parameter installation")
            for layer, qparams in deployment:
                for name, value in qparams.items():
                    update_offload_parameter(layer, name, value)
            deployment_stage("runtime-hook registration")
            self._register_runtime_smoothing_hook(mapping)
        except BaseException as error:
            if stage is not None:
                # No rollback: even a failed first write may have changed offload state.
                self._deployment_failure = (mapping.mapping_name, stage)
                raise RuntimeError(self._deployment_failure_message()) from error
            raise

    def _register_runtime_smoothing_hook(self, mapping: OSFP4Mapping) -> None:
        """Temporarily apply a runtime SmoothQuant scale during local execution."""
        if not mapping.requires_runtime_smoothing:
            return
        layer = mapping.balance_layers[0]
        if layer in self._runtime_smoothing_hooks:
            return

        def scale_input(module: Module, args: tuple):
            """Multiply a layer input by its deployed runtime smoothing scale."""
            scale = module.smooth_quant_scale.to(
                device=args[0].device,
                dtype=args[0].dtype,
            )
            return args[0] * scale, *args[1:]

        self._runtime_smoothing_hooks[layer] = layer.register_forward_pre_hook(
            scale_input
        )

    # 4. Calibration completion: remove hooks, verify coverage, and freeze quantization.

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()
        self._require_complete_calibration()
        if self.kv_cache_scheme is not None:
            kv_modules = [
                module
                for _, module in match_named_modules(
                    state.model,
                    self.resolved_targets,
                    self.ignore,
                )
            ]
            update_qparams(kv_modules, ("q", "k", "v"))
        QuantizationMixin.end_calibration(self, state.model)

    # 5. Finalization: attach the runtime contract, remove hooks, and clear state.

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self._finalization_complete:
            return True
        self._require_complete_calibration()
        runtime_targets = [
            mapping.mapping_name
            for mapping in self._resolved_mappings
            if mapping.requires_runtime_smoothing
        ]
        attach_osfp4_runtime_contract(
            state.model,
            runtime_smooth_quant_scale_targets=runtime_targets,
        )
        self._remove_runtime_smoothing_hooks()
        self._calibration.clear_all()
        self._resolved_mappings.clear()
        self._finalization_complete = True
        return True

    # 6. Shared cleanup and guards: remove hooks, verify coverage, and reject failed reuse.

    def _remove_runtime_smoothing_hooks(self) -> None:
        """Remove runtime scale hooks without removing checkpoint state."""
        for handle in self._runtime_smoothing_hooks.values():
            handle.remove()
        self._runtime_smoothing_hooks.clear()

    def _require_complete_calibration(self) -> None:
        """Reject calibration that did not optimize every resolved mapping."""
        self._require_usable()
        expected = {mapping.mapping_name for mapping in self._resolved_mappings}
        missing = sorted(expected - self._optimized_mapping_names)
        if missing:
            raise RuntimeError(
                "OSFP4 calibration did not observe every required registry mapping; "
                f"missing {missing}. Use oneshot(..., pipeline='sequential')."
            )

    def _require_usable(self) -> None:
        if self._deployment_failure is not None:
            raise RuntimeError(self._deployment_failure_message())

    def _deployment_failure_message(self) -> str:
        assert self._deployment_failure is not None
        name, stage = self._deployment_failure
        return (
            f"OSFP4 deployment failed for mapping {name!r} during {stage}; "
            "the model may be partially modified. Reload a fresh model and create "
            "a fresh OSFP4Modifier; this modifier cannot be retried."
        )

    # 7. Reporting: expose sampling provenance for callers.

    @property
    def activation_subsampling_records(self) -> dict[str, dict[str, int | str]]:
        """Return JSON-serializable sampling provenance for optimized mappings."""
        return {
            name: record.copy()
            for name, record in self._activation_subsampling_records.items()
        }
