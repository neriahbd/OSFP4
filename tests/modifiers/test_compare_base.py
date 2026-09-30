import pytest
import torch

from .compare_base import Trace, compare


def test_comparison_preserves_signed_zero_and_empty_tensor_bytes(tmp_path):
    for name in ("reference", "candidate"):
        directory = tmp_path / name
        directory.mkdir()
        trace = Trace()
        trace.add("values", [torch.tensor([0.0, -0.0]), torch.empty(0, 16)])
        trace.save(directory)
    assert compare(tmp_path / "reference", tmp_path / "candidate") == 2


@pytest.mark.parametrize("change", ["signed_zero", "dtype", "shape", "order", "event"])
def test_comparison_identifies_byte_and_metadata_mismatches(tmp_path, change):
    before, after = Trace(), Trace()
    before.add("values", {"a": torch.tensor([0.0]), "b": 1})
    if change == "signed_zero":
        value = {"a": torch.tensor([-0.0]), "b": 1}
    elif change == "dtype":
        value = {"a": torch.tensor([0.0], dtype=torch.bfloat16), "b": 1}
    elif change == "shape":
        value = {"a": torch.tensor([[0.0]]), "b": 1}
    elif change == "order":
        value = {"b": 1, "a": torch.tensor([0.0])}
    else:
        value = {"a": torch.tensor([0.0]), "b": 1}
    after.add("values", value)
    if change == "event":
        after.add("extra", None)
    for name, trace in (("reference", before), ("candidate", after)):
        directory = tmp_path / name
        directory.mkdir()
        trace.save(directory)
    expected = "raw bytes differ at 0/values/a" if change == "signed_zero" else "event"
    with pytest.raises(AssertionError, match=expected):
        compare(tmp_path / "reference", tmp_path / "candidate")
