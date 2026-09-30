"""Byte-for-byte equivalence with outputs captured from the original fork."""

import json
from pathlib import Path

import pytest
import torch

from tests.golden.generate_golden import CASES, run_case

FIXTURES = Path(__file__).parent / "golden" / "fixtures"


@pytest.mark.parametrize("scheme,mode", CASES)
def test_matches_fork_outputs(scheme, mode):
    stem = f"{scheme.lower()}-{mode}"
    path = FIXTURES / f"{stem}.safetensors"
    if not path.is_file():
        pytest.skip(f"golden fixture {path.name} not generated yet")
    from safetensors.torch import load_file

    expected = load_file(path)
    expected_metadata = json.loads((FIXTURES / f"{stem}.json").read_text())

    metadata, actual = run_case(scheme, mode)

    assert json.loads(json.dumps(metadata)) == expected_metadata
    assert sorted(actual) == sorted(expected)
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        assert actual[name].shape == value.shape, name
        assert torch.equal(
            actual[name].reshape(-1).view(torch.uint8),
            value.reshape(-1).view(torch.uint8),
        ), name
