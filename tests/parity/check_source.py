"""Compare installed OSFP4 sources with the pinned diagonal Git tree."""

import argparse
import difflib
import subprocess
from pathlib import Path

REFERENCE = "6a650621ac86842a3d8c08bb76b9350ebb5112d5"
PUBLIC = Path(__file__).resolve().parents[2] / "src/llmcompressor_osfp4"
SHARED_ROOTS = {
    "src/llmcompressor/modifiers/osfp4": "modifiers",
    "src/llmcompressor/observers/osfp4": "observers",
}
SMOOTH_SCALE_FILES = {"modifiers/base.py", "modifiers/smoothing.py"}
NATIVE_ONLY = {"modifiers/transforms.py"}
PUBLIC_ONLY = {"__init__.py", "modifiers/runtime_contract.py"}


def check(checkout):
    def git(*args):
        return subprocess.check_output(["git", "-C", checkout, *args])

    files = git("ls-tree", "-r", "--name-only", REFERENCE, "--", *SHARED_ROOTS)
    expected = set()
    smooth_scale_diff = []
    matched = 0
    errors = []
    for path in files.decode().splitlines():
        if not path.endswith(".py"):
            continue
        root = next(root for root in SHARED_ROOTS if path.startswith(root + "/"))
        name = SHARED_ROOTS[root] + path[len(root) :]
        expected.add(name)
        target = PUBLIC / name
        if name in NATIVE_ONLY or not target.is_file():
            continue
        reference = git("show", f"{REFERENCE}:{path}")
        public = target.read_bytes()
        if name in SMOOTH_SCALE_FILES:
            smooth_scale_diff.extend(
                difflib.diff_bytes(
                    difflib.unified_diff,
                    reference.splitlines(True),
                    public.splitlines(True),
                    f"a/{name}".encode(),
                    f"b/{name}".encode(),
                    n=0,
                )
            )
        elif reference == public:
            matched += 1
        else:
            errors.append(f"Source differs: {name}")

    actual = {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*.py")}
    wanted = (expected - NATIVE_ONLY) | PUBLIC_ONLY
    errors.extend(f"Unexpected public file: {name}" for name in sorted(actual - wanted))
    errors.extend(f"Missing public file: {name}" for name in sorted(wanted - actual))
    approved = Path(__file__).with_name("smooth-scale.patch").read_bytes()
    if b"".join(smooth_scale_diff) != approved:
        errors.append(
            "The base.py or smoothing.py diff differs from smooth-scale.patch"
        )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"{matched} shared files are byte-identical to {REFERENCE}.")
    print("All differences in base.py and smoothing.py match smooth-scale.patch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout", type=Path, help="LLM Compressor diagonal Git checkout"
    )
    check(parser.parse_args().checkout)
