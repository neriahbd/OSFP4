"""Pipeline and recipe integration with stock llm-compressor.

Replaces the fork's edit to ``tests/llmcompressor/pipelines/test_registry.py``;
upstream 0.14.0 infers pipelines from ``Modifier.requires_calibration_data``.
"""

import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.pipelines import CalibrationPipeline, SequentialPipeline
from llmcompressor_osfp4 import OSFP4Modifier


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_osfp4_infers_sequential_pipeline(scheme):
    modifiers = [OSFP4Modifier(scheme=scheme)]

    assert modifiers[0].requires_calibration_data is True
    assert isinstance(
        CalibrationPipeline.from_modifiers(modifiers), SequentialPipeline
    )


def test_osfp4_resolves_by_name_for_recipes():
    modifier = ModifierFactory.create(
        "OSFP4Modifier",
        allow_registered=True,
        allow_experimental=True,
        scheme="NVFP4",
        optimization_mode="rtn",
    )

    assert type(modifier) is OSFP4Modifier
    assert modifier.optimization_mode == "rtn"
