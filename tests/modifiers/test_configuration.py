import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, preset_name_to_scheme

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import OSFP4Modifier as _OSFP4Modifier
from llmcompressor_osfp4.observers import OSFP4Observer

from ._testing import LlamaForCausalLM, make_osfp4_modifier, run_modifier


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_optimization_mode_configures_one_observer(mode):
    modifier = _OSFP4Modifier(scheme="NVFP4", optimization_mode=mode, steps=7, lr=0.2)
    scheme = modifier.resolved_config.config_groups["group_0"]
    weights = scheme.weights

    assert modifier.weight_observer == "osfp4"
    assert weights.observer == "osfp4"
    assert weights.observer_kwargs == {
        "mode": mode,
        "num_iters": 7,
        "lr": 0.2,
    }
    assert scheme.input_activations.observer == "static_minmax"
    assert scheme.output_activations is None


@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_nvfp4a16_configures_weight_only_observer(mode):
    modifier = _OSFP4Modifier(
        scheme="NVFP4A16",
        optimization_mode=mode,
        steps=7,
        lr=0.2,
    )
    scheme = modifier.resolved_config.config_groups["group_0"]

    assert modifier._weight_only is True
    assert modifier.activation_subsample_size is None
    assert scheme.input_activations is None
    assert scheme.output_activations is None
    assert scheme.weights.observer == "osfp4"
    assert scheme.weights.observer_kwargs == {
        "mode": mode,
        "num_iters": 7,
        "lr": 0.2,
    }


@pytest.mark.parametrize("activation_subsample_size", [1, 128, 8192])
def test_nvfp4_accepts_activation_subsample_size(activation_subsample_size):
    modifier = _OSFP4Modifier(
        scheme="NVFP4",
        activation_subsample_size=activation_subsample_size,
    )

    assert modifier.activation_subsample_size == activation_subsample_size


@pytest.mark.parametrize("activation_subsample_size", [1, 8192])
def test_nvfp4a16_disables_activation_subsampling(activation_subsample_size):
    modifier = _OSFP4Modifier(
        scheme="NVFP4A16",
        activation_subsample_size=activation_subsample_size,
    )

    _ = modifier.resolved_config
    assert modifier.activation_subsample_size is None


def test_config_groups_reject_mixed_nvfp4_and_nvfp4a16():
    modifier = _OSFP4Modifier(
        config_groups={
            "a4": preset_name_to_scheme("NVFP4", ["model.layers.0"]),
            "a16": preset_name_to_scheme("NVFP4A16", ["model.layers.1"]),
        }
    )

    with pytest.raises(ValueError, match="cannot mix NVFP4 and NVFP4A16"):
        _ = modifier.resolved_config


def test_modifier_owns_config_group_weight_observer_kwargs():
    scheme = preset_name_to_scheme("NVFP4", ["Linear"])
    scheme.weights.observer = "osfp4"
    scheme.weights.observer_kwargs = {
        "mode": "sic",
        "num_iters": 7,
        "lr": 0.2,
    }

    modifier = _OSFP4Modifier(
        config_groups={"group": scheme},
        optimization_mode="rtn",
        steps=7,
        lr=0.2,
    )
    weights = modifier.resolved_config.config_groups["group"].weights
    assert weights.observer == "osfp4"
    assert weights.observer_kwargs == {"mode": "rtn", "num_iters": 7, "lr": 0.2}


def test_default_ignore_includes_lm_head_as_a_standalone_target():
    model, modifier = run_modifier(ignore=[])

    assert len(modifier._optimized_mapping_names) == 5
    assert hasattr(model.lm_head, "quantization_scheme")
    assert hasattr(model.lm_head, "smooth_quant_scale")


def test_compatible_input_observer_replays_post_alpha_inputs():
    model, _modifier = run_modifier(input_observer="memoryless_mse")
    layer = model.model.layers[0].self_attn.q_proj

    assert layer.quantization_scheme.input_activations.observer == "memoryless_mse"
    assert torch.isfinite(layer.input_global_scale).all()
    assert not hasattr(layer, "input_observer")


def test_input_observer_override_is_delegated_to_shared_registry():
    modifier = _OSFP4Modifier(scheme="NVFP4", input_observer="imatrix_mse")
    scheme = modifier.resolved_config.config_groups["group_0"]

    assert scheme.weights.observer == "osfp4"
    assert scheme.input_activations.observer == "imatrix_mse"


def test_unknown_input_observer_is_rejected_by_shared_registry():
    model = LlamaForCausalLM()
    modifier = _OSFP4Modifier(scheme="NVFP4", input_observer="not_registered")
    state = State(model=model)
    modifier.on_initialize(state)

    with pytest.raises(KeyError, match="Unable to find not-registered"):
        modifier.on_calibration_start(state, Event())


def test_observer_dict_preserves_input_and_modifier_owns_weight_observer():
    modifier = _OSFP4Modifier(
        scheme="NVFP4",
        observer={"input": "memoryless_mse"},
    )
    scheme = modifier.resolved_config.config_groups["group_0"]

    assert modifier.observer == {"input": "memoryless_mse"}
    assert scheme.weights.observer == "osfp4"
    assert scheme.input_activations.observer == "memoryless_mse"


def test_optional_kv_cache_scheme_uses_mixin_lifecycle():
    model, _modifier = run_modifier(
        kv_cache_scheme=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        )
    )
    attention = model.model.layers[0].self_attn

    assert torch.isfinite(attention.k_scale).all()
    assert torch.isfinite(attention.v_scale).all()
    assert attention.quantization_status.name == "FROZEN"
    assert not hasattr(attention, "k_observer")
    assert not hasattr(attention, "v_observer")


@pytest.mark.parametrize("with_kv_cache", [False, True])
def test_kv_qparam_update_is_explicitly_conditional(monkeypatch, with_kv_cache):
    from llmcompressor_osfp4.modifiers import base

    calls = []
    monkeypatch.setattr(
        base,
        "update_qparams",
        lambda modules, base_names: calls.append((modules, base_names)),
    )
    kwargs = {}
    if with_kv_cache:
        kwargs["kv_cache_scheme"] = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        )

    run_modifier(**kwargs)

    assert len(calls) == int(with_kv_cache)
    if with_kv_cache:
        assert calls[0][1] == ("q", "k", "v")


def test_nvfp4_config_groups_are_an_alternative_to_scheme():
    modifier = _OSFP4Modifier(
        config_groups={"decoder": preset_name_to_scheme("NVFP4", ["model.layers.0"])}
    )

    assert modifier.scheme is None
    assert modifier.resolved_weight_targets == {"model.layers.0"}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "scheme": "NVFP4",
                "config_groups": {"group": preset_name_to_scheme("NVFP4", ["Linear"])},
            },
            "either `scheme` or `config_groups`",
        ),
        (
            {"scheme": "NVFP4", "observer": {"unknown": "minmax"}},
            "Invalid observer key 'unknown'",
        ),
    ],
)
def test_quantization_configuration_rejects_incompatible_options(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _ = _OSFP4Modifier(**kwargs).resolved_config


def test_modifier_overrides_requested_weight_observer():
    modifier = _OSFP4Modifier(scheme="NVFP4", weight_observer="memoryless_minmax")
    weights = modifier.resolved_config.config_groups["group_0"].weights

    assert modifier.weight_observer == "osfp4"
    assert weights.observer == "osfp4"


def test_public_configuration_defaults_are_canonical():
    modifier = _OSFP4Modifier(scheme="NVFP4")
    _ = modifier.resolved_config
    assert modifier.optimization_mode == "sic"
    assert modifier.steps == 80
    assert modifier.lr == 0.12
    assert modifier.dampening_frac == 0.01
    assert modifier.offload_hessians is False
    assert modifier.activation_subsample_size == 16384
    assert modifier.targets == ["Linear"]
    assert modifier.ignore == []
    assert modifier.weight_observer == "osfp4"
    assert "resolved_mappings_" not in modifier.model_dump()
    assert "activation_subsample_seed" not in modifier.model_dump()
    assert "_resolved_mappings" not in modifier.model_dump()
    assert not hasattr(modifier, "_input_global_scales")

    removed = (
        "activation_global_scale_mode",
        "weight_scale_strategy",
        "fp8_scale_candidates_per_chunk",
        "group_size",
        "mappings",
        "smooth_quant_scale_targets",
        "optimize_unmapped_weight_scales",
        "loss_backend",
        "optimization_schedule",
        "gamma_updates_per_step",
        "alpha_updates_per_step",
        "optimization_groups_per_chunk",
        "independent_groups_per_chunk",
        "optimization_blocks_per_chunk",
        "optimization_blocks_per_batch",
        "optimization_quantization_groups_per_batch",
        "optimization_diagnostics_dir",
    )
    for field in removed:
        with pytest.raises(ValueError, match=field):
            make_osfp4_modifier(**{field: 1})


def test_osfp4_package_root_exports_only_modifier():
    from llmcompressor_osfp4 import modifiers as osfp4

    assert osfp4.__all__ == ["OSFP4Modifier"]
    assert not hasattr(osfp4, "OSFP4Observer")
    assert OSFP4Observer.__module__ == "llmcompressor_osfp4.observers.observer"


def test_modifier_configures_osfp4_local_hessian_offload():
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=0, offload_hessians=True)
    modifier.on_initialize(State(model=model))

    assert modifier._calibration.offload_hessians is True


def test_optimizer_configuration_has_no_range_validation():
    modifier = make_osfp4_modifier(
        steps=-1,
        lr=0.0,
        dampening_frac=-0.01,
        activation_subsample_size=0,
    )

    assert modifier.steps == -1
    assert modifier.lr == 0.0
    assert modifier.dampening_frac == -0.01
    assert modifier.activation_subsample_size == 0


@pytest.mark.parametrize("mode", ["var", "opt", "absmax"])
def test_modifier_rejects_legacy_optimization_modes(mode):
    with pytest.raises(ValueError, match="optimization_mode"):
        make_osfp4_modifier(optimization_mode=mode)


def test_modifier_rejects_removed_weight_scale_mode_parameter():
    with pytest.raises(ValueError, match="weight_scale_mode"):
        make_osfp4_modifier(weight_scale_mode="sic")


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("value", [None, -1, 0, -2, 1])
def test_one_shot_schedule_preserves_accepted_values(field, value):
    modifier = make_osfp4_modifier(**{field: value})
    state = State(model=LlamaForCausalLM())
    if value in (None, -1, 0):
        assert modifier.on_initialize(state)
    else:
        with pytest.raises(ValueError, match="one-shot"):
            modifier.on_initialize(state)


def test_resolved_groups_have_independent_optimizer_settings():
    modifier = _OSFP4Modifier(
        config_groups={
            "first": preset_name_to_scheme("NVFP4", ["first"]),
            "second": preset_name_to_scheme("NVFP4", ["second"]),
        },
        steps=7,
        lr=0.2,
    )
    first, second = modifier.resolved_config.config_groups.values()
    assert first.weights.observer_kwargs == second.weights.observer_kwargs
    assert first.weights.observer_kwargs is not second.weights.observer_kwargs
    first.weights.observer_kwargs["num_iters"] = 100
    assert second.weights.observer_kwargs["num_iters"] == 7
