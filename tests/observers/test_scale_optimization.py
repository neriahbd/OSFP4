import torch

from llmcompressor_osfp4.observers.scale_optimization import (  # noqa: E501
    LogScaleParameters,
    canonicalize_log_scale_parameters,
    create_log_scale_parameters,
    initialize_scale_values_from_absmax,
    materialize_scale_values,
    run_optimizer,
)


def test_normalization_preserves_weight_and_activation_scale_ratios():
    parameters = LogScaleParameters(
        log_alpha=torch.tensor([[0.1, 0.2]]),
        log_gamma_w=torch.tensor([[[0.4], [0.8]]]),
        log_gamma_x=torch.tensor([[[1.3, 1.7]]]),
    )
    weight_scale_before = (
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w
    ).clone()
    activation_scale_before = (
        parameters.log_alpha.unsqueeze(-1) - parameters.log_gamma_x
    ).clone()

    canonicalize_log_scale_parameters(parameters)

    torch.testing.assert_close(parameters.log_alpha, torch.tensor([[0.7, 0.8]]))
    torch.testing.assert_close(
        parameters.log_gamma_w,
        torch.tensor([[[-0.2], [0.2]]]),
    )
    torch.testing.assert_close(
        parameters.log_gamma_x,
        torch.tensor([[[1.9, 2.3]]]),
    )
    torch.testing.assert_close(parameters.log_gamma_w.mean(), torch.tensor(0.0))
    torch.testing.assert_close(
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w,
        weight_scale_before,
    )
    torch.testing.assert_close(
        parameters.log_alpha.unsqueeze(-1) - parameters.log_gamma_x,
        activation_scale_before,
    )


def test_scale_initialization_materializes_positive_quantization_group_scales():
    target = torch.arange(1, 2 * 3 * 16 + 1, dtype=torch.float32).reshape(
        2,
        3,
        16,
    )
    activations = torch.arange(1, 2 * 16 * 5 + 1, dtype=torch.float32).reshape(
        2,
        16,
        5,
    )

    initialized = initialize_scale_values_from_absmax(target, activations)
    parameters = create_log_scale_parameters(initialized)
    materialized = materialize_scale_values(parameters)

    assert initialized.alpha.shape == (2, 16)
    assert initialized.gamma_w.shape == (2, 3, 1)
    assert initialized.gamma_x.shape == (2, 1, 5)
    assert all(parameter.requires_grad for parameter in parameters)
    for initial, actual in zip(initialized, materialized):
        assert torch.isfinite(actual).all()
        assert torch.all(actual > 0)
        torch.testing.assert_close(actual, initial)


def test_run_optimizer_updates_trainable_scale_parameters():
    parameters = LogScaleParameters(
        log_alpha=torch.full((1, 16), 1.0, requires_grad=True),
        log_gamma_w=torch.full((1, 1, 1), 1.0, requires_grad=True),
        log_gamma_x=torch.full((1, 1, 1), 1.0, requires_grad=True),
    )
    before = tuple(parameter.detach().clone() for parameter in parameters)

    run_optimizer(
        parameters,
        lambda: sum(
            parameter.square().sum()
            for parameter in parameters
            if parameter is not None
        ),
        steps=1,
        lr=0.1,
    )

    for original, actual in zip(before, parameters):
        assert torch.all(actual.abs() < original.abs())


def test_run_optimizer_has_no_extra_final_evaluation():
    parameters = LogScaleParameters(
        log_alpha=torch.full((1, 16), 1.0, requires_grad=True),
        log_gamma_w=torch.full((1, 1, 1), 1.0, requires_grad=True),
        log_gamma_x=None,
    )
    evaluations = 0

    def evaluate():
        nonlocal evaluations
        evaluations += 1
        return (
            parameters.log_alpha.square().sum() + parameters.log_gamma_w.square().sum()
        )

    run_optimizer(parameters, evaluate, steps=2, lr=0.1)

    assert evaluations == 2


def test_weight_only_scales_allocate_alpha_and_gamma_w_only():
    target = torch.arange(1, 2 * 3 * 16 + 1, dtype=torch.float32).reshape(
        2,
        3,
        16,
    )
    initialized = initialize_scale_values_from_absmax(target, None)
    parameters = create_log_scale_parameters(initialized)
    weight_scale_before = (
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w
    ).clone()

    assert initialized.gamma_x is None
    assert parameters.log_gamma_x is None
    assert [parameter for parameter in parameters if parameter is not None] == [
        parameters.log_alpha,
        parameters.log_gamma_w,
    ]

    canonicalize_log_scale_parameters(parameters)
    materialized = materialize_scale_values(parameters)
    assert materialized.gamma_x is None
    torch.testing.assert_close(
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w,
        weight_scale_before,
    )


def test_weight_only_outlier_group_uses_plain_mean_centering():
    gamma_w = torch.full((1, 512, 1), 0.02)
    gamma_w[0, 0, 0] = 100.0
    parameters = LogScaleParameters(
        log_alpha=torch.zeros(1, 16),
        log_gamma_w=gamma_w.log(),
        log_gamma_x=None,
    )
    weight_scale_before = (
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w
    ).clone()
    expected_shift = parameters.log_gamma_w.mean(dim=(1, 2), keepdim=True)
    expected_gamma_w = parameters.log_gamma_w - expected_shift

    canonicalize_log_scale_parameters(parameters)

    torch.testing.assert_close(parameters.log_gamma_w, expected_gamma_w)
    torch.testing.assert_close(
        parameters.log_alpha,
        expected_shift.squeeze(-1).expand_as(parameters.log_alpha),
    )
    torch.testing.assert_close(
        parameters.log_gamma_w.mean(dim=(1, 2)),
        torch.zeros(1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        parameters.log_alpha.unsqueeze(1) + parameters.log_gamma_w,
        weight_scale_before,
    )


def test_representable_weight_only_group_keeps_plain_mean_centering():
    torch.manual_seed(0)
    gamma_w = (torch.randn(4, 256, 1) * 0.5).exp()
    parameters = LogScaleParameters(
        log_alpha=torch.zeros(4, 16),
        log_gamma_w=gamma_w.log(),
        log_gamma_x=None,
    )

    canonicalize_log_scale_parameters(parameters)

    torch.testing.assert_close(
        parameters.log_gamma_w.mean(dim=(1, 2)),
        torch.zeros(4),
        atol=1e-6,
        rtol=0,
    )
