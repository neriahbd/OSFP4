import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
MINIMAL = REPO_ROOT / "examples/calibrate.py"
COMMON_CALIBRATION = REPO_ROOT / "examples/fpquant/common/calibration.py"


def test_minimal_example_is_direct_and_small():
    source = MINIMAL.read_text()
    tree = ast.parse(source)
    assert "Qwen/Qwen3-0.6B" in source
    assert "torch.bfloat16" in source
    assert "NUM_CALIBRATION_SAMPLES = 20" in source
    assert "MAX_SEQUENCE_LENGTH = 512" in source
    assert 'optimization_mode="sic"' in source
    assert 'pipeline="sequential"' in source
    assert "save_compressed=True" in source
    assert not any(
        isinstance(node, ast.Import) and node.names[0].name == "os"
        for node in tree.body
    )


def test_parallel_table_layout_and_thin_wrappers():
    common = COMMON_CALIBRATION.read_text()
    assert "def prepare_main(" in common
    assert "def calibrate_main(" in common
    for table in ("table1", "table7"):
        directory = REPO_ROOT / f"examples/fpquant/{table}"
        for name in (
            "README.md",
            "config.py",
            "prepare_data.py",
            "calibrate.py",
            "evaluate.py",
            "run_calibration.sh",
            "run_evaluation.sh",
        ):
            assert (directory / name).is_file()
        assert "calibrate_main(PROFILE)" in (directory / "calibrate.py").read_text()
        assert "prepare_main(PROFILE)" in (directory / "prepare_data.py").read_text()
        assert "main(PROFILE)" in (directory / "evaluate.py").read_text()
    assert (REPO_ROOT / "examples/fpquant/table7/aggregate.py").is_file()


def test_removed_workflows_have_no_compatibility_wrappers():
    removed = (
        "examples/calibration/calibrate_osfp4.py",
        "examples/calibration/build_calibration_dataset_cache.py",
        "examples/scripts/run_c4_256_calibration.sh",
        "examples/eval/run_table1_task.py",
    )
    assert all(not (REPO_ROOT / path).exists() for path in removed)
