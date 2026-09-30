import inspect
from functools import wraps

import torch
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor_osfp4.modifiers.optimization.rtn import (
    optimize_rtn as _optimize_rtn,
)
from llmcompressor.observers import Observer
from llmcompressor_osfp4.observers import OptimizedQuantizationGroupScales
from llmcompressor_osfp4.observers import observer as osfp4_observer
from llmcompressor_osfp4.observers.observer import _optimize_joint_scales


def _make_observer(
    mode,
    *,
    steps=0,
    lr=0.01,
    dtype=torch.float32,
    weight_only=False,
):
    args = preset_name_to_scheme("NVFP4", []).weights.model_copy(
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
    observer = Observer.load_from_registry(
        args.observer,
        base_name="weight",
        args=args,
    )
    observer._test_alpha_dtype = dtype
    return observer


@wraps(_optimize_rtn)
def optimize_rtn(*args, **kwargs):
    observer = kwargs.setdefault("observer", _make_observer("rtn"))
    kwargs.setdefault(
        "alpha_dtype",
        getattr(observer, "_test_alpha_dtype", torch.float32),
    )
    return _optimize_rtn(*args, **kwargs)


def _legacy_optimize_joint_scales(weight, activations, *, steps, lr):
    weight = weight.detach().to(dtype=torch.float32)
    activations = activations.detach().to(
        device=weight.device,
        dtype=torch.float32,
    )
    sigma_x_squared = activations.square().mean(dim=2)
    initialized = osfp4_observer.initialize_scale_values_from_absmax(
        weight,
        activations,
    )
    coefficients = osfp4_observer.build_joint_loss_coefficients(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        activation_reference=weight,
    )
    parameters = osfp4_observer.create_log_scale_parameters(initialized)

    def evaluate_loss():
        return osfp4_observer.compute_joint_loss(
            weight,
            activations,
            parameters.log_alpha.exp(),
            parameters.log_gamma_w.exp(),
            parameters.log_gamma_x.exp(),
            coefficients,
        ).sum()

    osfp4_observer.run_optimizer(
        parameters,
        evaluate_loss,
        steps=steps,
        lr=lr,
    )
    osfp4_observer.canonicalize_log_scale_parameters(parameters)
    return (
        osfp4_observer.materialize_scale_values(parameters),
        sigma_x_squared,
    )


def test_rtn_runner_separates_continuous_optimization_and_selection():
    W = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    activations = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    calls = []

    class RecordingObserver:
        def __init__(self):
            self.mode = "rtn"
            self.alpha_dtype = torch.float32
            self.weight_dtype = torch.float32

        def __call__(self, *_args, **_kwargs):
            raise AssertionError(
                "RTN must use optimize_quantization_group_scales, not observer forward"
            )

        def optimize_quantization_group_scales(
            self,
            weight,
            *,
            activation_quantization_groups,
            weight_metric,
            alpha_dtype,
        ):
            assert alpha_dtype is torch.float32
            calls.append(
                {
                    "event": "optimize",
                    "weight": weight.clone(),
                    "activation_quantization_groups": (
                        activation_quantization_groups.clone()
                    ),
                    "weight_metric": weight_metric.clone(),
                }
            )
            quantization_group_count, rows, width = weight.shape
            return OptimizedQuantizationGroupScales(
                torch.ones((quantization_group_count, width)),
                torch.full((quantization_group_count, rows, 1), 1.5),
            )

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            calls.append({"event": "select", "weight": weight.clone()})
            assert torch.equal(
                weight_metric,
                activations.T.unsqueeze(0).square().mean(dim=2),
            )
            scales = torch.full_like(optimized.gamma_w, 2.0)
            return scales, torch.zeros_like(scales)

    result = optimize_rtn(
        W,
        [activations],
        observer=RecordingObserver(),
        sigma_x_squared=activations.square().mean(dim=0),
    )

    assert [call["event"] for call in calls] == ["optimize", "select"]
    assert torch.equal(calls[0]["weight"], W.unsqueeze(0))
    assert torch.equal(
        calls[0]["activation_quantization_groups"], activations.T.unsqueeze(0)
    )
    assert torch.equal(
        calls[0]["weight_metric"],
        activations.T.unsqueeze(0).square().mean(dim=2),
    )
    assert torch.equal(calls[1]["weight"], W.unsqueeze(0))
    assert torch.equal(result.alpha_star, torch.ones(16))
    assert torch.equal(result.gamma_w_star, torch.full((2, 1), 2.0))
    assert torch.equal(result.weight_zero_point, torch.zeros((2, 1)))


def test_rtn_optimizes_all_groups_once_and_preserves_result_layout():
    W = torch.arange(96, dtype=torch.float32).reshape(2, 48)
    activations = torch.arange(192, dtype=torch.float32).reshape(4, 48)

    class RecordingObserver:
        mode = "rtn"
        calls = 0
        activation_quantization_groups = None

        def optimize_quantization_group_scales(
            self,
            weight,
            *,
            activation_quantization_groups,
            weight_metric,
            alpha_dtype,
        ):
            assert alpha_dtype is torch.float32
            assert activation_quantization_groups.shape[:2] == (weight.shape[0], 16)
            assert weight.shape == (3, 2, 16)
            assert torch.equal(
                weight_metric,
                activation_quantization_groups.square().mean(dim=2),
            )
            self.activation_quantization_groups = activation_quantization_groups.clone()
            self.calls += 1
            quantization_group_count, rows, width = weight.shape
            alpha = torch.arange(
                1,
                quantization_group_count * width + 1,
                dtype=torch.float32,
            ).reshape(quantization_group_count, width)
            gamma = torch.stack(
                [
                    torch.arange(
                        quantization_group_index + 1,
                        quantization_group_index + rows + 1,
                        dtype=torch.float32,
                    )
                    for quantization_group_index in range(quantization_group_count)
                ]
            ).unsqueeze(-1)
            return OptimizedQuantizationGroupScales(
                alpha,
                gamma,
            )

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            assert self.calls == 1
            assert weight.shape == (3, 2, 16)
            assert weight_metric.shape == (3, 16)
            return optimized.gamma_w, torch.zeros_like(
                optimized.gamma_w,
                dtype=torch.float8_e4m3fn,
            )

    observer = RecordingObserver()
    result = optimize_rtn(
        W,
        [activations],
        observer=observer,
        sigma_x_squared=activations.square().mean(dim=0),
    )

    assert observer.calls == 1
    for quantization_group_index in range(3):
        column_start = quantization_group_index * 16
        assert torch.equal(
            observer.activation_quantization_groups[
                quantization_group_index : quantization_group_index + 1
            ],
            activations[:, column_start : column_start + 16].T.unsqueeze(0),
        )
    expected_alpha_star = torch.arange(1, 49, dtype=torch.float32)
    expected_gamma_w_star = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    expected_weight_zero_point = torch.zeros_like(
        expected_gamma_w_star,
        dtype=torch.float8_e4m3fn,
    )
    assert torch.equal(result.alpha_star, expected_alpha_star)
    assert torch.equal(
        result.gamma_w_star,
        expected_gamma_w_star,
    )
    assert torch.equal(result.weight_zero_point, expected_weight_zero_point)


def test_rtn_omits_activation_groups_for_weight_only():
    seen = []

    class WeightOnlyObserver:
        def optimize_quantization_group_scales(
            self,
            weight,
            *,
            activation_quantization_groups,
            weight_metric,
            alpha_dtype,
        ):
            seen.append(weight.shape[0])
            assert activation_quantization_groups is None
            return OptimizedQuantizationGroupScales(
                torch.ones(3, 16),
                torch.ones(3, 2, 1),
            )

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            return optimized.gamma_w, torch.zeros_like(optimized.gamma_w)

    optimize_rtn(
        torch.ones(2, 48),
        None,
        observer=WeightOnlyObserver(),
        alpha_dtype=torch.float32,
        sigma_x_squared=torch.ones(48),
    )

    assert seen == [3]


def test_optimizer_is_deterministic_and_returns_finite_positive_scales():
    torch.manual_seed(8)
    weight = torch.randn(2, 4, 16)
    activations = torch.randn(2, 16, 12)
    sigma_x_squared = activations.square().mean(dim=2)
    first = _optimize_joint_scales(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        steps=1,
        lr=0.01,
    )
    second = _optimize_joint_scales(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        steps=1,
        lr=0.01,
    )
    for first_value, second_value in zip(first, second):
        assert torch.equal(first_value, second_value)
    for value in first:
        assert torch.isfinite(value).all()
        assert torch.all(value > 0)


def test_joint_optimizer_uses_supplied_weight_metric(monkeypatch):
    torch.manual_seed(9)
    weight = torch.randn(2, 4, 16)
    activations = torch.randn(2, 16, 12)
    sigma_x_squared = torch.rand(2, 16)
    seen = {}
    original = osfp4_observer.build_joint_loss_coefficients

    def record(*args, **kwargs):
        seen["weight_metric"] = kwargs["weight_metric"].clone()
        return original(*args, **kwargs)

    monkeypatch.setattr(osfp4_observer, "build_joint_loss_coefficients", record)
    _optimize_joint_scales(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        steps=0,
        lr=0.01,
    )

    assert torch.equal(seen["weight_metric"], sigma_x_squared)


def test_optimizer_signature_has_no_schedule_controls():
    runner_parameters = inspect.signature(optimize_rtn).parameters
    assert "quantization_groups_per_batch" not in runner_parameters
    assert "sample_count" not in runner_parameters
    assert "sigma_x_squared" in runner_parameters
    assert runner_parameters["sigma_x_squared"].default is inspect.Parameter.empty
    assert "weight_metric" not in runner_parameters
    assert "mode" not in runner_parameters
    assert "alpha_dtype" in runner_parameters
    assert runner_parameters["alpha_dtype"].default is inspect.Parameter.empty
    assert "weight_dtype" not in runner_parameters
    assert "steps" not in runner_parameters
    assert "lr" not in runner_parameters
    assert "optimization_schedule" not in runner_parameters
    assert "gamma_updates_per_step" not in runner_parameters
    assert "alpha_updates_per_step" not in runner_parameters


def test_parallel_and_one_quantization_group_optimization_reconstruct_identically():
    torch.manual_seed(10)
    weight = torch.randn(3, 4, 16)
    activations = torch.randn(3, 16, 12)
    sigma_x_squared = activations.square().mean(dim=2)
    kwargs = dict(steps=2, lr=0.01)
    parallel = _optimize_joint_scales(
        weight,
        activations,
        weight_metric=sigma_x_squared,
        **kwargs,
    )
    serial = [
        _optimize_joint_scales(
            weight[index : index + 1],
            activations[index : index + 1],
            weight_metric=sigma_x_squared[index : index + 1],
            **kwargs,
        )
        for index in range(3)
    ]
    for field, parallel_value in zip(parallel._fields, parallel):
        reconstructed = torch.cat([getattr(result, field) for result in serial])
        assert torch.allclose(reconstructed, parallel_value, atol=1e-6, rtol=1e-6)


def test_optimize_rtn_returns_canonical_alpha_and_final_e4m3_gamma():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(3, 32, generator=generator)
    batches = [torch.randn(17, 32, generator=generator)]

    result = optimize_rtn(
        weight,
        batches,
        observer=_make_observer("rtn", steps=0, lr=0.01, dtype=torch.bfloat16),
        sigma_x_squared=batches[0].square().mean(dim=0),
    )

    assert result._fields == ("alpha_star", "gamma_w_star", "weight_zero_point")
    assert result.alpha_star.shape == (32,)
    assert result.gamma_w_star.shape == (3, 2)
    assert result.weight_zero_point.shape == (3, 2)
    assert torch.equal(
        result.alpha_star,
        result.alpha_star.to(torch.bfloat16).float(),
    )
    assert torch.equal(
        result.gamma_w_star,
        result.gamma_w_star.to(torch.float8_e4m3fn).float(),
    )
    assert torch.count_nonzero(result.weight_zero_point.float()) == 0


def test_cached_sigma_x_squared_matches_legacy_rtn_numerically():
    generator = torch.Generator().manual_seed(25)
    weight = torch.randn(3, 32, generator=generator)
    batches = [
        torch.randn(7, 32, generator=generator),
        torch.randn(10, 32, generator=generator),
    ]
    activations = torch.cat(batches)
    cached_sigma_x_squared = (
        sum(batch.float().square().sum(dim=0) for batch in batches)
        / activations.shape[0]
    )
    weight_quantization_groups = weight.reshape(3, 2, 16).permute(1, 0, 2).contiguous()
    activation_quantization_groups = (
        activations.reshape(17, 2, 16).permute(1, 2, 0).contiguous()
    )
    learned, legacy_sigma_x_squared = _legacy_optimize_joint_scales(
        weight_quantization_groups,
        activation_quantization_groups,
        steps=2,
        lr=0.01,
    )
    legacy_observer = _make_observer(
        "rtn",
        steps=2,
        lr=0.01,
        dtype=torch.bfloat16,
    )
    legacy_optimized = OptimizedQuantizationGroupScales(
        learned.alpha.to(torch.bfloat16).float(),
        learned.gamma_w,
    )
    legacy_scales, legacy_zero_points = legacy_observer.select_weight_qparams(
        weight_quantization_groups,
        legacy_optimized,
        weight_metric=legacy_sigma_x_squared,
    )
    result = optimize_rtn(
        weight,
        batches,
        observer=_make_observer(
            "rtn",
            steps=2,
            lr=0.01,
            dtype=torch.bfloat16,
        ),
        sigma_x_squared=cached_sigma_x_squared,
    )

    torch.testing.assert_close(
        cached_sigma_x_squared,
        legacy_sigma_x_squared.reshape(32),
        atol=1e-5,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result.alpha_star,
        legacy_optimized.alpha_star.reshape(32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        result.gamma_w_star,
        legacy_scales.squeeze(-1).transpose(0, 1),
    )
    assert torch.equal(
        result.weight_zero_point,
        legacy_zero_points.squeeze(-1).transpose(0, 1),
    )
