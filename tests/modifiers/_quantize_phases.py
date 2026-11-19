# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Run production deployment while retaining parameters for phase comparisons."""

from unittest.mock import patch

from llmcompressor_osfp4.modifiers import base, osfp4_quantize


def run_phases(
    mapping,
    cache,
    *,
    optimization_mode,
    dampening_frac,
    weight_only,
    sampled_activations=None,
):
    result = osfp4_quantize.quantize_mapping(
        mapping,
        cache,
        optimization_mode=optimization_mode,
        dampening_frac=dampening_frac,
        weight_only=weight_only,
        sampled_activations=sampled_activations,
    )
    modifier = base.OSFP4Modifier(
        scheme="NVFP4A16" if weight_only else "NVFP4",
        optimization_mode=optimization_mode,
    )
    modifier._calibration_cache = cache
    modifier._weight_only = weight_only
    deployment = {}

    def record_parameter(layer, name, value):
        deployment.setdefault(layer, {})[name] = value

    # Keep smoothing writes, but defer parameter installation to the test caller.
    with patch.object(base, "update_offload_parameter", record_parameter):
        modifier._deploy_mapping(mapping, result)
    return list(deployment.items())
