"""Check the public calibration source against the pinned diagonal Git tree."""

import argparse
import ast
import subprocess
from pathlib import Path

REFERENCE = "6a650621ac86842a3d8c08bb76b9350ebb5112d5"
PUBLIC = Path(__file__).resolve().parents[2] / "src/llmcompressor_osfp4"
SHARED_ROOTS = {
    "src/llmcompressor/modifiers/osfp4": "modifiers",
    "src/llmcompressor/observers/osfp4": "observers",
}
# The public modifier registers with stock LLM Compressor and saves plugin alpha.
ADAPTED = {"modifiers/base.py", "modifiers/smoothing.py"}
NATIVE_ONLY = {"modifiers/transforms.py"}
PUBLIC_ONLY = {
    "__init__.py",
    "modifiers/runtime_contract.py",
}


# Stock integration and public runtime-scale storage extend these specific methods.
ADAPTED_BODIES = {
    "OSFP4Modifier.resolve_quantization_config",
    "OSFP4Modifier.on_initialize",
    "OSFP4Modifier._validate_initialization",
    "OSFP4Modifier._resolve_mappings",
    "OSFP4Modifier._reset_calibration_state",
    "OSFP4Modifier._deploy_mapping",
    "OSFP4Modifier.on_finalize",
    "_deploy_runtime_smoothing",
}
HOOK_MEMBERS = {
    "OSFP4Modifier._register_runtime_smoothing_hook",
    "OSFP4Modifier._remove_runtime_smoothing_hooks",
    "OSFP4Modifier._runtime_smoothing_hooks",
}


def declarations(source):
    """Compare declared fields, signatures and shared method bodies in adapted files."""
    result = {}

    def collect(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                collect(node.body, prefix + node.name + ".")
            elif isinstance(node, ast.FunctionDef):
                name = prefix + node.name
                if name in ADAPTED_BODIES:
                    node.body = []
                result[name] = ast.dump(node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                result[prefix + node.target.id] = ast.dump(node)

    collect(ast.parse(source).body)
    return result


def check(checkout):
    def git(*args):
        return subprocess.check_output(["git", "-C", checkout, *args])

    files = git("ls-tree", "-r", "--name-only", REFERENCE, "--", *SHARED_ROOTS)
    expected = set()
    matched = 0
    errors = []
    for path in files.decode().splitlines():
        if not path.endswith(".py"):
            continue
        root = next(root for root in SHARED_ROOTS if path.startswith(root + "/"))
        name = SHARED_ROOTS[root] + path[len(root) :]
        expected.add(name)
        if name in NATIVE_ONLY:
            continue
        data = git("show", f"{REFERENCE}:{path}")
        for source, public in SHARED_ROOTS.items():
            namespace = source.removeprefix("src/").replace("/", ".")
            data = data.replace(
                namespace.encode(), f"llmcompressor_osfp4.{public}".encode()
            )
        target = PUBLIC / name
        if name in ADAPTED and target.is_file():
            upstream = declarations(data)
            public = declarations(target.read_bytes())
            for key in sorted(upstream.keys() | public.keys()):
                if key in HOOK_MEMBERS and key not in upstream:
                    continue
                if upstream.get(key) != public.get(key):
                    errors.append(f"Declaration differs: {name} {key}")
            continue
        if not target.is_file() or target.read_bytes() != data:
            errors.append(f"Source differs: {name}")
        else:
            matched += 1

    actual = {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*.py")}
    wanted = (expected - NATIVE_ONLY) | PUBLIC_ONLY
    errors.extend(f"Unexpected public file: {name}" for name in sorted(actual - wanted))
    errors.extend(f"Missing public file: {name}" for name in sorted(wanted - actual))
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"{matched} shared modules match {REFERENCE} after import namespace translation."
    )
    for label, names in (
        ("Adapted", ADAPTED),
        ("Native only", NATIVE_ONLY),
        ("Public only", PUBLIC_ONLY),
    ):
        print(f"{label}: {', '.join(sorted(names))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout", type=Path, help="LLM Compressor diagonal Git checkout"
    )
    check(parser.parse_args().checkout)
