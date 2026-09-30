import inspect

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor.observers import Observer
from llmcompressor_osfp4 import observers as osfp4_package
from llmcompressor_osfp4.observers import OSFP4Observer
from llmcompressor_osfp4.observers import observer as osfp4_module
from llmcompressor_osfp4.observers.scale_optimization import (
    QuantizationGroupScaleValues,
)


def _make_args(mode="sic", *, steps=7, lr=0.2):
    return preset_name_to_scheme("NVFP4", []).weights.model_copy(
        update={
            "observer": "osfp4",
            "observer_kwargs": {
                "mode": mode,
                "num_iters": steps,
                "lr": lr,
            },
        },
        deep=True,
    )


def _make_observer(mode="sic", *, steps=7, lr=0.2):
    args = _make_args(mode, steps=steps, lr=lr)
    return Observer.load_from_registry(args.observer, base_name="weight", args=args)


def test_registry_builds_attached_observer_from_quantization_args():
    observer = _make_observer("rtn", steps=11, lr=0.03)

    assert isinstance(observer, OSFP4Observer)
    assert observer.args.observer_kwargs == {
        "mode": "rtn",
        "num_iters": 11,
        "lr": 0.03,
    }
    assert osfp4_package.OSFP4Observer is OSFP4Observer


def test_observer_optimizer_interface_requires_modifier_owned_inputs():
    parameters = inspect.signature(
        OSFP4Observer.optimize_quantization_group_scales
    ).parameters
    assert parameters["weight_metric"].default is inspect.Parameter.empty
    assert parameters["alpha_dtype"].default is inspect.Parameter.empty
    assert "weight_only" not in parameters
    assert "trace_callback" not in parameters


def test_observer_uses_inherited_forward_for_lifecycle_statistics():
    observer = _make_observer("rtn")
    observed = torch.arange(32, dtype=torch.float32).reshape(2, 16)

    returned = observer(observed)

    assert returned is observer
    assert observer.has_statistics
    assert observer.min_vals.shape == observer.max_vals.shape == (2, 1)


def test_joint_optimization_and_scale_selection_are_separate(monkeypatch):
    rows = 3
    samples = 5
    weight = torch.arange(rows * 16, dtype=torch.float32).reshape(1, rows, 16)
    metric = torch.linspace(1.0, 2.0, 16).unsqueeze(0)
    activations = torch.arange(16 * samples, dtype=torch.float32).reshape(
        1, 16, samples
    )
    alpha = torch.linspace(0.5, 1.5, 16).reshape(1, 16)
    gamma_w = torch.ones(1, rows, 1)
    gamma_x = torch.ones(1, 1, samples)
    selected = torch.arange(1, rows + 1, dtype=torch.float32).reshape(1, rows, 1)
    seen = {}

    def optimize(raw_weight, raw_activations, **kwargs):
        seen.update(weight=raw_weight, activations=raw_activations, kwargs=kwargs)
        return QuantizationGroupScaleValues(alpha, gamma_w, gamma_x)

    def select(target, selected_alpha, selected_gamma, selected_metric):
        seen.update(
            target=target,
            alpha=selected_alpha,
            gamma=selected_gamma,
            metric=selected_metric,
        )
        return selected, torch.zeros((), dtype=torch.bool), torch.tensor(False)

    monkeypatch.setattr(osfp4_module, "_optimize_joint_scales", optimize)
    monkeypatch.setattr(osfp4_module, "_search_e4m3_gamma_star", select)
    observer = _make_observer("rtn")

    optimized = observer.optimize_quantization_group_scales(
        weight,
        activation_quantization_groups=activations,
        weight_metric=metric,
        alpha_dtype=torch.bfloat16,
    )
    scales, zero_points = observer.select_weight_qparams(
        weight,
        optimized,
        weight_metric=metric,
    )

    assert torch.equal(seen["weight"], weight)
    assert torch.equal(seen["activations"], activations)
    assert torch.equal(seen["kwargs"].pop("weight_metric"), metric)
    assert seen["kwargs"] == {"steps": 7, "lr": 0.2}
    assert torch.equal(optimized.alpha_star, alpha.to(torch.bfloat16).float())
    assert torch.equal(seen["alpha"], optimized.alpha_star)
    assert torch.equal(scales, selected)
    assert zero_points.dtype == observer.args.zp_dtype


def test_missing_activations_selects_weight_only_optimization(monkeypatch):
    weight = torch.ones(2, 3, 16)
    metric = torch.ones(2, 16)
    learned = QuantizationGroupScaleValues(torch.ones(2, 16), torch.ones(2, 3, 1), None)
    seen = {}

    def optimize(target, weight_metric, **kwargs):
        seen.update(target=target, metric=weight_metric, kwargs=kwargs)
        return learned

    monkeypatch.setattr(osfp4_module, "_optimize_weight_scales", optimize)
    observer = _make_observer("sic")

    actual = observer.optimize_quantization_group_scales(
        weight,
        weight_metric=metric,
        alpha_dtype=torch.float32,
    )

    assert torch.equal(actual.alpha_star, learned.alpha)
    assert torch.equal(actual.gamma_w, learned.gamma_w)
    assert seen["kwargs"] == {"steps": 7, "lr": 0.2}


def test_invalid_optimized_scales_are_rejected(monkeypatch):
    monkeypatch.setattr(
        osfp4_module,
        "_optimize_weight_scales",
        lambda *_args, **_kwargs: QuantizationGroupScaleValues(
            torch.full((1, 16), float("inf")), torch.ones(1, 2, 1), None
        ),
    )

    with pytest.raises(ValueError, match="invalid scales"):
        _make_observer().optimize_quantization_group_scales(
            torch.ones(1, 2, 16),
            weight_metric=torch.ones(1, 16),
            alpha_dtype=torch.float32,
        )


def test_invalid_selected_scales_are_rejected(monkeypatch):
    monkeypatch.setattr(
        osfp4_module,
        "_search_e4m3_gamma_star",
        lambda *_args: (
            torch.ones(1, 2, 1),
            torch.ones(1, dtype=torch.bool),
            torch.tensor(False),
        ),
    )
    optimized = osfp4_module.OptimizedQuantizationGroupScales(
        torch.ones(1, 16), torch.ones(1, 2, 1)
    )

    with pytest.raises(ValueError, match="invalid checkpoint scales"):
        _make_observer().select_weight_qparams(
            torch.ones(1, 2, 16),
            optimized,
            weight_metric=torch.ones(1, 16),
        )


@pytest.mark.parametrize(
    "dtype,value",
    [
        (torch.float16, 1e5),
        (torch.float16, 1e-9),
        (torch.bfloat16, torch.finfo(torch.float32).max),
        (torch.bfloat16, 1e-44),
    ],
)
def test_alpha_is_validated_after_deployment_rounding(monkeypatch, dtype, value):
    alpha = torch.full((1, 16), value)
    assert torch.isfinite(alpha).all() and (alpha > 0).all()
    monkeypatch.setattr(
        osfp4_module,
        "_optimize_weight_scales",
        lambda *_args, **_kwargs: QuantizationGroupScaleValues(
            alpha, torch.ones(1, 2, 1), None
        ),
    )
    with pytest.raises(ValueError, match="invalid scales"):
        _make_observer().optimize_quantization_group_scales(
            torch.ones(1, 2, 16),
            weight_metric=torch.ones(1, 16),
            alpha_dtype=dtype,
        )


@pytest.mark.parametrize("gamma,expected_warnings", [(1.0, 0), (1e20, 1), (1e-20, 1)])
def test_scale_fallback_logs_once_after_tiled_search(
    monkeypatch, gamma, expected_warnings
):
    from llmcompressor_osfp4.observers import scale_selection

    monkeypatch.setattr(scale_selection, "_E4M3_SEARCH_PRIMARY_TENSOR_BYTES", 16 * 4)
    warnings = []
    monkeypatch.setattr(osfp4_module.logger, "warning", warnings.append)
    optimized = osfp4_module.OptimizedQuantizationGroupScales(
        torch.ones(1, 16), torch.full((1, 3, 1), gamma)
    )
    scales, _ = _make_observer().select_weight_qparams(
        torch.ones(1, 3, 16), optimized, weight_metric=torch.ones(1, 16)
    )
    assert torch.isfinite(scales).all()
    assert (
        warnings
        == ["OSFP4: empty scale candidate set; applied boundary fallback."]
        * expected_warnings
    )
