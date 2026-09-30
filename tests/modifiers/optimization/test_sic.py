import inspect
from functools import wraps

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor_osfp4.modifiers.optimization import sic
from llmcompressor_osfp4.modifiers.optimization.sic import (
    optimize_sic as _optimize_sic,
)
from llmcompressor.observers import Observer
from llmcompressor_osfp4.observers import (
    OptimizedQuantizationGroupScales,
    scale_selection,
)
from llmcompressor_osfp4.observers import observer as osfp4_observer
from llmcompressor_osfp4.observers.fp4 import quantize_e2m1


def _make_observer(
    *,
    mode="sic",
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
    observer._test_weight_dtype = dtype
    return observer


@wraps(_optimize_sic)
def optimize_sic(*args, **kwargs):
    observer = kwargs.setdefault("observer", _make_observer())
    kwargs.setdefault(
        "alpha_dtype",
        getattr(observer, "_test_alpha_dtype", torch.float32),
    )
    kwargs.setdefault(
        "weight_dtype", getattr(observer, "_test_weight_dtype", torch.float32)
    )
    return _optimize_sic(*args, **kwargs)


def _reference_hessian(batches):
    sample_count = sum(batch.shape[0] for batch in batches)
    return sum(batch.T @ batch for batch in batches) / sample_count


def _reference_cholesky_state(weight, hessian, percdamp=0.01):
    weight = weight.detach().float()
    damped = hessian.detach().to(device=weight.device, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(damped))
    diag = torch.arange(damped.shape[0], device=damped.device)
    damped[diag, diag] += damp
    factor = torch.linalg.cholesky(damped, upper=True)
    return factor, factor @ weight.transpose(0, 1)


def _legacy_lower_sic_quantization(
    weight,
    hessian,
    alpha_star,
    gamma_w_star,
    *,
    weight_dtype,
    percdamp=0.01,
):
    weight = weight.detach().float()
    damped = hessian.detach().to(device=weight.device, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(damped))
    diag = torch.arange(damped.shape[0], device=damped.device)
    damped[diag, diag] += damp
    lower = torch.linalg.cholesky(damped, upper=False)
    residual = weight @ lower
    quantized_weight = torch.empty_like(weight)

    for index in range(weight.shape[1] - 1, -1, -1):
        scale = gamma_w_star[:, index // 16]
        target = residual[:, index] / lower[index, index]
        denominator = alpha_star[index] * scale
        code = quantize_e2m1(target / denominator)
        dequantized = (code * scale).to(weight_dtype).float()
        quantized_weight[:, index] = dequantized
        reconstructed = alpha_star[index] * dequantized
        residual[:, :index] -= reconstructed.unsqueeze(1) * lower[
            index, :index
        ].unsqueeze(0)

    return quantized_weight


def _exhaustive_sic_scale_search(
    target,
    alpha,
    gamma_w,
    u_quantization_group_diag,
):
    grid = scale_selection._get_e4m3_scale_grid(target.device, torch.float32)
    denominator = alpha[:, None, None, :] * grid[None, None, :, None]
    original = target[:, :, None, :]
    quantized = quantize_e2m1(original / denominator)
    reconstructed = denominator * quantized
    errors = (
        (original - reconstructed).square()
        * u_quantization_group_diag.square()[:, None, None, :]
    ).sum(dim=-1)
    beta = gamma_w / grid.view(1, 1, -1)
    valid = (
        (beta >= scale_selection._BETA_MIN)
        | torch.isclose(beta, torch.tensor(scale_selection._BETA_MIN))
    ) & (
        (beta <= scale_selection._BETA_MAX)
        | torch.isclose(beta, torch.tensor(scale_selection._BETA_MAX))
    )
    errors.masked_fill_(~valid, torch.inf)
    _, index = errors.min(dim=-1)
    return grid[index].unsqueeze(-1)


def _reference_sequential_scale_search(weight, U, optimized, weight_dtype):
    """Score the image's objective directly, using reconstructed right blocks."""
    rows, columns = weight.shape
    alpha = optimized.alpha_star.flatten()
    grid = scale_selection._get_e4m3_scale_grid(weight.device, torch.float32)
    selected = torch.empty(rows, columns // 16)
    reconstructed = torch.zeros_like(weight)
    stored = torch.zeros_like(weight)
    for group in range(columns // 16 - 1, -1, -1):
        start, end = group * 16, (group + 1) * 16
        d = U.diag()[start:end]
        # Known feedback only: current-block quantization has not started.
        proxy = d[:, None] * weight[:, start:end].T
        proxy += U[start:end, end:] @ (weight[:, end:] - reconstructed[:, end:]).T
        for row in range(rows):
            gamma = optimized.gamma_w[group, row, 0]
            beta = gamma / grid
            valid = (
                (beta >= scale_selection._BETA_MIN)
                | torch.isclose(beta, torch.tensor(scale_selection._BETA_MIN))
            ) & (
                (beta <= scale_selection._BETA_MAX)
                | torch.isclose(beta, torch.tensor(scale_selection._BETA_MAX))
            )
            candidates = grid[valid]
            denominator = candidates[:, None] * alpha[start:end] * d
            codes = quantize_e2m1(proxy[:, row] / denominator)
            errors = (proxy[:, row] - denominator * codes).square().sum(-1)
            selected[row, group] = candidates[errors.argmin()]

        for j in range(end - 1, start - 1, -1):
            # Recompute from actual reconstruction errors instead of mutating Y.
            target = (
                weight[:, j]
                + ((weight[:, j + 1 :] - reconstructed[:, j + 1 :]) @ U[j, j + 1 :])
                / U[j, j]
            )
            scale = selected[:, group]
            code = quantize_e2m1(target / (alpha[j] * scale))
            stored[:, j] = (code * scale).to(weight_dtype).float()
            reconstructed[:, j] = alpha[j] * stored[:, j]
    return selected, stored


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("groups", [1, 3])
def test_sic_matches_image_objective_and_feedback_reference(
    monkeypatch, weight_only, dtype, groups
):
    generator = torch.Generator().manual_seed(73)
    columns = groups * 16
    weight = torch.randn(5, columns, generator=generator).to(dtype).float()
    calibration = torch.randn(83, columns, generator=generator)
    hessian = calibration.T @ calibration / 83
    before_weight, before_hessian = weight.clone(), hessian.clone()
    observer = _make_observer(steps=2, dtype=dtype, weight_only=weight_only)
    original_optimize = observer.optimize_quantization_group_scales
    learned = []

    def capture(*args, **kwargs):
        result = original_optimize(*args, **kwargs)
        learned.append(result)
        return result

    monkeypatch.setattr(observer, "optimize_quantization_group_scales", capture)
    result = optimize_sic(
        weight,
        None if weight_only else [calibration],
        hessian,
        observer=observer,
    )
    assert len(learned) == 1
    U, _ = _reference_cholesky_state(weight, hessian)
    expected_scales, expected_weight = _reference_sequential_scale_search(
        weight, U, learned[0], dtype
    )
    torch.testing.assert_close(result.gamma_w_star, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(result.quantized_weight, expected_weight, rtol=0, atol=0)
    original_scales = (
        _exhaustive_sic_scale_search(
            weight.reshape(5, groups, 16).permute(1, 0, 2),
            learned[0].alpha_star,
            learned[0].gamma_w,
            U.diag().reshape(groups, 16),
        )
        .squeeze(-1)
        .T
    )
    assert torch.equal(result.gamma_w_star[:, -1], original_scales[:, -1])
    if groups > 1:
        assert not torch.equal(result.gamma_w_star[:, :-1], original_scales[:, :-1])
    assert torch.count_nonzero(result.weight_zero_point) == 0
    assert torch.equal(weight, before_weight)
    assert torch.equal(hessian, before_hessian)


@pytest.mark.parametrize("gamma", [1.0, 1e20])
def test_sic_zero_weight_ties_and_boundary_fallback(monkeypatch, gamma):
    observer = _make_observer()
    optimized = OptimizedQuantizationGroupScales(
        torch.ones(2, 16), torch.full((2, 3, 1), gamma)
    )
    monkeypatch.setattr(
        observer, "optimize_quantization_group_scales", lambda *a, **kw: optimized
    )
    weight = torch.zeros(3, 32)
    result = optimize_sic(weight, None, torch.eye(32), observer=observer)
    expected = _exhaustive_sic_scale_search(
        torch.zeros(2, 3, 16),
        optimized.alpha_star,
        optimized.gamma_w,
        torch.ones(2, 16),
    )
    if gamma == 1e20:
        expected.fill_(448.0)
    assert torch.equal(result.gamma_w_star, expected.squeeze(-1).T)
    assert torch.count_nonzero(result.quantized_weight) == 0
    assert torch.count_nonzero(result.weight_zero_point) == 0


def test_sic_runner_signature_has_no_optimizer_controls():
    parameters = inspect.signature(optimize_sic).parameters
    assert "quantization_groups_per_batch" not in parameters
    assert "sample_count" not in parameters
    assert "altered_qargs" not in parameters
    assert "alpha_dtype" in parameters
    assert "weight_dtype" in parameters
    assert "steps" not in parameters
    assert "lr" not in parameters
    assert "mode" not in parameters
    assert "optimization_schedule" not in parameters
    assert "gamma_updates_per_step" not in parameters
    assert "alpha_updates_per_step" not in parameters


def _run_optimize_sic(**overrides):
    arguments = {
        "W": torch.randn(2, 16),
        "cached_activation_batches": [torch.randn(4, 16)],
        "hessian": torch.eye(16),
        "observer": _make_observer(),
    }
    arguments.update(overrides)
    return optimize_sic(**arguments)


def test_sic_fp8_scale_selection_matches_exhaustive_grid():
    generator = torch.Generator().manual_seed(4)
    target = torch.randn(3, 5, 16, generator=generator)
    alpha = torch.rand(3, 16, generator=generator) + 0.5
    gamma_w = torch.rand(3, 5, 1, generator=generator) * 2 + 0.2
    u_quantization_group_diag = torch.rand(3, 16, generator=generator) + 0.1

    selected, invalid, fallback = scale_selection._search_e4m3_gamma_star(
        target,
        alpha,
        gamma_w,
        u_quantization_group_diag.square(),
    )
    expected_scales = _exhaustive_sic_scale_search(
        target,
        alpha,
        gamma_w,
        u_quantization_group_diag,
    )

    assert not invalid.any()
    assert torch.equal(selected, expected_scales)


def test_sic_alpha_initializes_to_ones_without_a_floor(monkeypatch):
    weight = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    batches = [torch.eye(16)]
    seen = {}
    original = osfp4_observer.create_log_scale_parameters

    def record(scales):
        seen["alpha_init"] = scales.alpha.detach().clone()
        return original(scales)

    monkeypatch.setattr(osfp4_observer, "create_log_scale_parameters", record)
    optimize_sic(
        weight,
        batches,
        _reference_hessian(batches),
    )

    assert torch.equal(seen["alpha_init"], torch.ones(1, 16))


def test_sic_uses_upper_cholesky_factorization():
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(3, 32, generator=generator)
    batches = [torch.randn(47, 32, generator=generator)]
    hessian = _reference_hessian(batches)
    U, _ = _reference_cholesky_state(weight, hessian)
    damped = hessian.float().clone()
    diag = torch.arange(damped.shape[0])
    damped[diag, diag] += 0.01 * torch.diag(damped).mean()

    assert torch.count_nonzero(torch.tril(U, diagonal=-1)) == 0
    torch.testing.assert_close(U.transpose(0, 1) @ U, damped)


def test_normalized_hessian_preserves_residual_targets():
    generator = torch.Generator().manual_seed(26)
    weight = torch.randn(3, 32, generator=generator)
    calibration = torch.randn(29, 32, generator=generator)
    raw_hessian = calibration.T @ calibration
    normalized_hessian = raw_hessian / calibration.shape[0]
    raw_U, raw_Y = _reference_cholesky_state(weight, raw_hessian)
    normalized_U, normalized_Y = _reference_cholesky_state(
        weight,
        normalized_hessian,
    )

    for index in range(weight.shape[1] - 1, -1, -1):
        torch.testing.assert_close(
            normalized_Y[index, :] / normalized_U[index, index],
            raw_Y[index, :] / raw_U[index, index],
            atol=2e-5,
            rtol=2e-5,
        )
        reconstructed = weight[:, index]
        normalized_Y[:index, :] -= normalized_U[:index, index].unsqueeze(
            1
        ) * reconstructed.unsqueeze(0)
        raw_Y[:index, :] -= raw_U[:index, index].unsqueeze(1) * reconstructed.unsqueeze(
            0
        )


def test_normalized_hessian_preserves_discrete_sic_outputs():
    generator = torch.Generator().manual_seed(27)
    weight = torch.randn(3, 32, generator=generator)
    calibration = torch.randn(31, 32, generator=generator)
    raw_hessian = calibration.T @ calibration
    normalized_hessian = raw_hessian / calibration.shape[0]

    normalized = optimize_sic(
        weight,
        None,
        normalized_hessian,
        observer=_make_observer(steps=0, weight_only=True),
    )
    raw = optimize_sic(
        weight,
        None,
        raw_hessian,
        observer=_make_observer(steps=0, weight_only=True),
    )

    assert torch.equal(normalized.gamma_w_star, raw.gamma_w_star)
    assert torch.equal(normalized.weight_zero_point, raw.weight_zero_point)
    assert torch.equal(normalized.quantized_weight, raw.quantized_weight)


@pytest.mark.parametrize("weight_only", [False, True])
def test_sic_scale_selection_includes_completed_blocks_feedback(
    monkeypatch, weight_only
):
    generator = torch.Generator().manual_seed(10)
    weight = torch.randn(4, 32, generator=generator)
    batches = [torch.randn(24, 32, generator=generator)]
    seen_targets = []
    seen_metrics = []
    original = osfp4_observer._search_e4m3_gamma_star

    def record_target(target, alpha, gamma_w, error_metric):
        seen_targets.append(target.detach().clone())
        seen_metrics.append(error_metric.detach().clone())
        return original(target, alpha, gamma_w, error_metric)

    monkeypatch.setattr(
        osfp4_observer,
        "_search_e4m3_gamma_star",
        record_target,
    )
    result = optimize_sic(
        weight,
        None if weight_only else batches,
        _reference_hessian(batches),
    )

    assert len(seen_targets) == 2
    U, _ = _reference_cholesky_state(weight, _reference_hessian(batches))
    torch.testing.assert_close(seen_targets[0][0], weight[:, 16:])
    reconstructed = result.quantized_weight * result.alpha_star
    feedback = U[:16, 16:] @ (weight[:, 16:] - reconstructed[:, 16:]).T
    expected_target = weight[:, :16] + (feedback / U.diag()[:16, None]).T
    torch.testing.assert_close(seen_targets[1][0], expected_target)
    assert not torch.allclose(seen_targets[1][0], weight[:, :16])
    torch.testing.assert_close(seen_metrics[0][0], U.diag()[16:].square())
    torch.testing.assert_close(seen_metrics[1][0], U.diag()[:16].square())


def test_sic_continuous_optimizer_uses_original_weight_quantization_groups(monkeypatch):
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(4, 32, generator=generator)
    batches = [torch.randn(24, 32, generator=generator)]
    optimized_weights = []
    original = osfp4_observer._optimize_joint_scales

    def record_weight(W_quantization_group, *args, **kwargs):
        optimized_weights.append(W_quantization_group.detach().clone())
        return original(W_quantization_group, *args, **kwargs)

    monkeypatch.setattr(osfp4_observer, "_optimize_joint_scales", record_weight)
    optimize_sic(
        weight,
        batches,
        _reference_hessian(batches),
    )

    assert len(optimized_weights) == 1
    torch.testing.assert_close(optimized_weights[0][0], weight[:, :16])
    torch.testing.assert_close(optimized_weights[0][1], weight[:, 16:])


def test_sic_optimizes_all_quantization_groups_together(monkeypatch):
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(3, 64, generator=generator)
    batches = [torch.arange(64, dtype=torch.float32).repeat(19, 1)]
    optimized_starts = []
    original_optimize = osfp4_observer._optimize_joint_scales

    def record_optimize(weight, activations, *args, **kwargs):
        assert activations.shape[1:] == (16, 19)
        optimized_starts.append(activations[:, 0, 0].int().tolist())
        return original_optimize(weight, activations, *args, **kwargs)

    monkeypatch.setattr(osfp4_observer, "_optimize_joint_scales", record_optimize)
    optimize_sic(
        weight,
        batches,
        _reference_hessian(batches),
    )

    assert optimized_starts == [[0, 16, 32, 48]]


def test_sic_selects_scales_after_full_mapping_optimization():
    generator = torch.Generator().manual_seed(21)
    weight = torch.randn(2, 32, generator=generator)
    batches = [torch.randn(12, 32, generator=generator)]
    events = []

    class RecordingObserver:
        mode = "sic"
        alpha_dtype = torch.float32
        weight_dtype = torch.float32

        def optimize_quantization_group_scales(
            self,
            weight,
            *,
            weight_metric,
            activation_quantization_groups,
            alpha_dtype,
        ):
            events.append("optimize")
            assert weight.shape == (2, 2, 16)
            assert activation_quantization_groups.shape == (2, 16, 12)
            assert weight_metric.shape == (2, 16)
            return OptimizedQuantizationGroupScales(
                torch.ones(2, 16),
                torch.ones(2, 2, 1),
            )

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            events.append("select")
            assert events[0] == "optimize"
            assert weight.shape == (1, 2, 16)
            assert optimized.alpha_star.shape == (1, 16)
            assert weight_metric.shape == (1, 16)
            return optimized.gamma_w, torch.zeros_like(optimized.gamma_w)

    _optimize_sic(
        weight,
        batches,
        _reference_hessian(batches),
        observer=RecordingObserver(),
        alpha_dtype=torch.float32,
        weight_dtype=torch.float32,
    )

    assert events == ["optimize", "select", "select"]


def test_sic_runner_matches_block_selection_and_residual_reference(
    monkeypatch,
):
    generator = torch.Generator().manual_seed(14)
    weight = torch.randn(2, 32, generator=generator)
    batch = torch.randn(48, 32, generator=generator)
    hessian = _reference_hessian([batch])
    calls = []

    class RecordingObserver:
        mode = "sic"
        alpha_dtype = torch.float32
        weight_dtype = torch.float32

        def __call__(self, *_args, **_kwargs):
            raise AssertionError(
                "SIC must use optimize_quantization_group_scales, not observer forward"
            )

        def optimize_quantization_group_scales(
            self,
            weight,
            *,
            weight_metric,
            activation_quantization_groups,
            alpha_dtype,
        ):
            assert alpha_dtype is torch.float32
            calls.append(
                {
                    "event": "optimize",
                    "weight": weight.clone(),
                    "weight_metric": weight_metric.clone(),
                    "activation_quantization_groups": (
                        activation_quantization_groups.clone()
                    ),
                }
            )
            alpha = torch.stack((torch.full((16,), 0.75), torch.ones(16)))
            gamma = torch.ones((2, 2, 1))
            return OptimizedQuantizationGroupScales(alpha, gamma)

        def select_weight_qparams(self, weight, optimized, *, weight_metric):
            # All 16 quantizations of the right block precede left selection.
            group = 1 - sum(call["event"] == "select" for call in calls)
            assert len(quantize_inputs) == (1 - group) * 16
            calls.append(
                {
                    "event": "select",
                    "weight": weight.clone(),
                    "optimized": optimized,
                    "weight_metric": weight_metric.clone(),
                }
            )
            scales = torch.tensor(
                [[[21.0], [22.0]], [[11.0], [12.0]]],
            )
            zero_points = torch.tensor(
                [[[31.0], [32.0]], [[41.0], [42.0]]],
            )
            return scales[group : group + 1], zero_points[group : group + 1]

    observer = RecordingObserver()
    quantize_inputs = []
    original_quantize = quantize_e2m1

    def record_quantize(values):
        quantize_inputs.append(values.clone())
        return original_quantize(values)

    monkeypatch.setattr(sic, "quantize_e2m1", record_quantize)
    result = _optimize_sic(
        weight,
        [batch],
        hessian,
        observer=observer,
        alpha_dtype=torch.float32,
        weight_dtype=torch.float32,
    )
    U, expected_Y = _reference_cholesky_state(weight, hessian)

    assert [call["event"] for call in calls] == ["optimize", "select", "select"]
    expected_weight_quantization_groups = weight.reshape(2, 2, 16).permute(1, 0, 2)
    expected_activation_quantization_groups = batch.reshape(48, 2, 16).permute(1, 2, 0)
    assert torch.equal(calls[0]["weight"], expected_weight_quantization_groups)
    assert torch.equal(
        calls[0]["activation_quantization_groups"],
        expected_activation_quantization_groups,
    )
    assert torch.equal(
        calls[0]["weight_metric"],
        torch.diag(U).square().reshape(2, 16),
    )
    assert not calls[0]["weight"].requires_grad
    assert len(quantize_inputs) == 32

    expected_alpha_star = torch.cat((torch.full((16,), 0.75), torch.ones(16)))
    expected_weight_scale = torch.tensor([[21.0, 11.0], [22.0, 12.0]])
    expected_weight_zero_point = torch.tensor([[31.0, 41.0], [32.0, 42.0]])
    expected_quantized_weight = torch.empty_like(weight)
    quantize_call_index = 0

    for quantization_group_index in range(1, -1, -1):
        column_start = quantization_group_index * 16
        column_end = column_start + 16
        expected_alpha = expected_alpha_star[column_start]
        expected_gamma_w_star = expected_weight_scale[:, quantization_group_index]
        block_u = U[column_start:column_end, column_start:column_end]
        proxy = expected_Y[column_start:column_end] - torch.triu(block_u, 1) @ (
            weight[:, column_start:column_end].T
        )
        selection = calls[2 - quantization_group_index]
        torch.testing.assert_close(
            selection["weight"][0], (proxy / block_u.diag()[:, None]).T
        )
        torch.testing.assert_close(
            selection["weight_metric"][0], block_u.diag().square()
        )

        for j in range(column_end - 1, column_start - 1, -1):
            column_target = expected_Y[j, :] / U[j, j]
            expected_ratio = column_target / (expected_alpha * expected_gamma_w_star)
            assert torch.equal(quantize_inputs[quantize_call_index], expected_ratio)
            quantized_weight = (
                quantize_e2m1(expected_ratio) * expected_gamma_w_star
            ).float()
            expected_quantized_weight[:, j] = quantized_weight
            reconstructed_weight = expected_alpha * quantized_weight
            expected_Y[:j, :] -= U[:j, j].unsqueeze(1) * reconstructed_weight.unsqueeze(
                0
            )
            quantize_call_index += 1

    assert torch.equal(result.alpha_star, expected_alpha_star)
    assert torch.equal(result.gamma_w_star, expected_weight_scale)
    assert torch.equal(result.weight_zero_point, expected_weight_zero_point)
    assert torch.equal(result.quantized_weight, expected_quantized_weight)


def test_sic_reverse_updates_and_quantized_weight_match_reference():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(3, 32, generator=generator)
    batches = [torch.randn(31, 32, generator=generator)]
    result = optimize_sic(
        weight,
        batches,
        _reference_hessian(batches),
        observer=_make_observer(dtype=torch.bfloat16),
    )
    U, Y = _reference_cholesky_state(weight, _reference_hessian(batches))
    transformed = Y.clone()
    expected_quantized_weight = torch.empty_like(weight)

    for index in range(31, -1, -1):
        scale = result.gamma_w_star[:, index // 16]
        target = transformed[index, :] / U[index, index]
        denominator = result.alpha_star[index] * scale
        code = quantize_e2m1(target / denominator)
        dequantized_weight = (code * scale).to(torch.bfloat16).float()
        actual = result.alpha_star[index] * dequantized_weight
        expected_quantized_weight[:, index] = dequantized_weight
        transformed[:index, :] -= U[:index, index].unsqueeze(1) * actual.unsqueeze(0)

    assert result._fields == (
        "alpha_star",
        "gamma_w_star",
        "weight_zero_point",
        "quantized_weight",
    )
    assert torch.equal(
        result.alpha_star,
        result.alpha_star.to(torch.bfloat16).float(),
    )
    assert torch.equal(result.quantized_weight, expected_quantized_weight)
    assert torch.count_nonzero(result.weight_zero_point) == 0


def test_upper_sic_matches_lower_quantization_with_sequentially_selected_scales():
    generator = torch.Generator().manual_seed(24)
    weight = torch.randn(3, 64, generator=generator)
    calibration = torch.randn(37, 64, generator=generator)
    hessian = calibration.transpose(0, 1) @ calibration / calibration.shape[0]
    result = optimize_sic(
        weight,
        None,
        hessian,
        observer=_make_observer(
            steps=2,
            lr=0.01,
            dtype=torch.bfloat16,
            weight_only=True,
        ),
    )
    legacy_weight = _legacy_lower_sic_quantization(
        weight,
        hessian,
        result.alpha_star,
        result.gamma_w_star,
        weight_dtype=torch.bfloat16,
    )

    assert torch.equal(result.quantized_weight, legacy_weight)
