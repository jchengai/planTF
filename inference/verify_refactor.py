"""
verify_refactor.py
------------------
Post-refactor structural and functional verification for the inference/
directory reorganisation.

Checks performed
----------------
[FILE] File-existence checks for every expected script and document.
[IMPORT] Each module imports cleanly in an isolated subprocess.
[API] Public API of common/ modules (load_model, make_dummy_input) is intact.
[DEPS] Cross-track import direction: faithful → deploy only; deploy ↛ faithful.
[DEDUP] load_model / make_dummy_input defined only in common/ (not duplicated).
[ORCH] run_pipeline.py is pure orchestration (no PlanningModel / nn.Module subclass).
[PATH] Default --ckpt and --onnx arguments resolve to absolute paths.
[OUTDIR] outputs/ directory resolves under repo root, not CWD.
[ONNX] ONNX file convention: inference/planTF.onnx (warns if absent).

Usage (from repo root):
    python inference/verify_refactor.py
    python inference/verify_refactor.py --verbose   # show details on failures
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Optional

_here = os.path.dirname(os.path.abspath(__file__))
_inference = _here                                   # inference/
_repo_root  = os.path.dirname(_inference)            # planTF/


# ---------------------------------------------------------------------------
# Checker infrastructure
# ---------------------------------------------------------------------------

_results = []   # list of (category, label, status, detail)


def _record(category, label, passed, detail=""):
    """Store a result. `passed=None` means WARN."""
    if passed is True:
        status = "PASS"
    elif passed is None:
        status = "WARN"
    else:
        status = "FAIL"
    _results.append((category, label, status, detail))
    return passed


def _py(code: str) -> tuple[int, str]:
    """Run Python code in a subprocess; return (returncode, combined output)."""
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        timeout=60,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_files() -> None:
    """[FILE] Every expected file must exist."""
    expected = [
        ("common/run_inference.py",      "common utilities"),
        ("common/inspect_model.py",      "common utilities"),
        ("common/patches.py",            "common utilities"),
        ("deploy/export_onnx.py",        "deploy track"),
        ("deploy/benchmark_latency.py",    "deploy track"),
        ("deploy/benchmark_tensorrt.py",   "deploy track"),
        ("deploy/TENSORRT_REPORT.md",      "documentation"),
        ("faithful/compare_outputs.py",    "faithful track"),
        ("faithful/ablate_patches.py",     "faithful track"),
        ("run_pipeline.py",                "top-level orchestration"),
        ("README.md",                      "documentation"),
        ("FIDELITY_REPORT.md",             "documentation"),
    ]
    for rel, desc in expected:
        path = os.path.join(_inference, rel)
        _record("FILE", rel, os.path.isfile(path), desc)

    # Checkpoint is external — warn if absent, don't fail
    ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    if os.path.isfile(ckpt):
        _record("FILE", "checkpoints/planTF.ckpt", True, "checkpoint present")
    else:
        _record("FILE", "checkpoints/planTF.ckpt", None,
                "checkpoint absent — model-loading tests will be skipped")

    # ONNX is a build artifact — warn if absent
    onnx = os.path.join(_inference, "planTF.onnx")
    if os.path.isfile(onnx):
        size_mb = os.path.getsize(onnx) / 1e6
        _record("FILE", "inference/planTF.onnx", None, f"present ({size_mb:.1f} MB) — run deploy pipeline to regenerate")
    else:
        _record("FILE", "inference/planTF.onnx", None,
                "absent — run: python inference/deploy/export_onnx.py")

    # Old flat-level files must NOT exist (would shadow new ones)
    stale = [
        "run_inference.py", "inspect_model.py", "export_onnx.py",
        "benchmark_latency.py", "benchmark_tensorrt.py",
        "ablate_patches.py", "compare_outputs.py",
    ]
    for name in stale:
        path = os.path.join(_inference, name)
        exists = os.path.isfile(path)
        _record("FILE", f"stale root-level {name} absent",
                not exists,
                "" if not exists else f"FOUND at {path} — will shadow new scripts")


def check_imports() -> None:
    """[IMPORT] Each module must import cleanly in an isolated subprocess."""
    scripts = [
        ("common/run_inference",  os.path.join(_inference, "common"),  [_repo_root]),
        ("common/inspect_model",  os.path.join(_inference, "common"),  [_repo_root]),
        ("common/patches",         os.path.join(_inference, "common"),  [_repo_root]),
        ("deploy/export_onnx",    os.path.join(_inference, "deploy"),  [_repo_root, os.path.join(_inference, "common")]),
        ("deploy/benchmark_latency",   os.path.join(_inference, "deploy"), [_repo_root, os.path.join(_inference, "common")]),
        ("deploy/benchmark_tensorrt",  os.path.join(_inference, "deploy"), [_repo_root, os.path.join(_inference, "common")]),
        ("faithful/compare_outputs", os.path.join(_inference, "faithful"), [_repo_root, os.path.join(_inference, "common"), os.path.join(_inference, "deploy")]),
        ("faithful/ablate_patches",  os.path.join(_inference, "faithful"), [_repo_root, os.path.join(_inference, "common"), os.path.join(_inference, "deploy")]),
    ]
    for label, script_dir, extra_paths in scripts:
        module = os.path.basename(label).replace("/", "")
        path_setup = "\n".join(
            f'sys.path.insert(0, {repr(p)})' for p in [script_dir] + extra_paths
        )
        code = f"""
import sys
{path_setup}
import importlib
m = importlib.import_module({repr(module)})
print("OK:", m.__file__)
"""
        rc, out = _py(code)
        _record("IMPORT", label, rc == 0, out[:120] if rc != 0 else "")


def check_api() -> None:
    """[API] common/ must expose load_model and make_dummy_input."""
    code = f"""
import sys
sys.path.insert(0, {repr(_repo_root)})
sys.path.insert(0, {repr(os.path.join(_inference, 'common'))})
from run_inference import load_model, make_dummy_input
import inspect
sig = inspect.signature(make_dummy_input)
params = list(sig.parameters)
assert 'A' in params and 'M' in params and 'device' in params, f"unexpected params: {{params}}"
dummy = make_dummy_input(A=4, M=8)
assert dummy['agent']['position'].shape == (1, 4, 21, 2), "shape mismatch"
assert dummy['map']['valid_mask'].shape  == (1, 8, 20),   "shape mismatch"
print("API OK")
"""
    rc, out = _py(code)
    _record("API", "common/run_inference.load_model importable", rc == 0, out[:200] if rc != 0 else "")
    _record("API", "make_dummy_input(A, M, device) signature intact", rc == 0, "")
    _record("API", "make_dummy_input output shapes correct", rc == 0, "")


def check_dedup() -> None:
    """[DEDUP] load_model and make_dummy_input must only be defined in common/."""
    targets = ["def load_model", "def make_dummy_input"]
    for fn in targets:
        found_in = []
        for track in ["common", "deploy", "faithful"]:
            track_dir = os.path.join(_inference, track)
            for fname in os.listdir(track_dir):
                if not fname.endswith(".py"):
                    continue
                src = _read(os.path.join(track_dir, fname))
                if re.search(r"^" + re.escape(fn) + r"\b", src, re.MULTILINE):
                    found_in.append(f"{track}/{fname}")
        only_in_common = len(found_in) == 1 and found_in[0].startswith("common/")
        _record(
            "DEDUP", f"{fn} defined only in common/",
            only_in_common,
            f"found in: {found_in}" if not only_in_common else "",
        )


def check_orch() -> None:
    """[ORCH] run_pipeline.py must be pure orchestration (no model logic)."""
    src = _read(os.path.join(_inference, "run_pipeline.py"))

    forbidden_patterns = [
        (r"\bPlanningModel\b",         "defines / imports PlanningModel"),
        (r"\bclass\s+\w+\s*\(nn\.Module\)", "defines an nn.Module subclass"),
        (r"\bpatch_natten_for_onnx\b",  "contains patch logic (should import from deploy/)"),
        (r"\btorch\.onnx\.export\b",    "calls torch.onnx.export (should delegate to deploy/)"),
        (r"\bload_model\b",             "calls load_model directly (should delegate to sub-scripts)"),
    ]

    for pattern, description in forbidden_patterns:
        found = bool(re.search(pattern, src))
        _record("ORCH", f"run_pipeline has no: {description}", not found,
                f"pattern found: {pattern}" if found else "")

    # confirm subprocess orchestration is present
    has_subprocess = bool(re.search(r"\bsubprocess\b", src))
    _record("ORCH", "run_pipeline uses subprocess for delegation", has_subprocess, "")


def check_paths() -> None:
    """[PATH] Default --ckpt and --onnx/--out must use _repo_root (not bare CWD-relative strings)."""
    # Static analysis: confirm _repo_root appears before add_argument("--ckpt"/"--onnx"/"--out")
    # in each script.  This is simpler and more reliable than subprocess introspection of
    # local variables inside main().
    scripts_ckpt = [
        ("common", "run_inference.py"),
        ("common", "inspect_model.py"),
        ("deploy", "export_onnx.py"),
        ("deploy", "benchmark_latency.py"),
        ("faithful", "ablate_patches.py"),
        ("faithful", "compare_outputs.py"),
    ]
    for track, fname in scripts_ckpt:
        src = _read(os.path.join(_inference, track, fname))
        # After hardening: _repo_root must appear; bare default string must NOT
        has_bare  = bool(re.search(r'add_argument.*--ckpt.*default="checkpoints/', src))
        has_rooted = bool(re.search(r'_repo_root', src)) and bool(re.search(r'_default_ckpt', src))
        _record("PATH", f"{track}/{fname}: --ckpt default uses _repo_root (not bare CWD-relative)",
                has_rooted and not has_bare,
                ("still contains bare default" if has_bare else
                 "missing _repo_root/_default_ckpt" if not has_rooted else ""))

    for track, fname in [
        ("deploy",   "export_onnx.py"),
        ("deploy",   "benchmark_latency.py"),
        ("faithful", "compare_outputs.py"),
    ]:
        src = _read(os.path.join(_inference, track, fname))
        has_bare   = bool(re.search(r'default="inference/', src))
        has_rooted = bool(re.search(r'_inference.*planTF\.onnx|_default_onnx', src))
        _record("PATH", f"{track}/{fname}: --onnx/--out default uses _inference (not bare CWD-relative)",
                has_rooted and not has_bare,
                ("still contains bare default" if has_bare else
                 "missing _inference reference" if not has_rooted else ""))


def check_outdir() -> None:
    """[OUTDIR] outputs/ must resolve to <repo_root>/outputs, not a CWD-relative path."""
    # Check that scripts reference _repo_root for outputs, not a raw relative path
    scripts_with_outdir = [
        ("deploy/benchmark_latency.py",),
        ("faithful/compare_outputs.py",),
    ]
    for (rel,) in scripts_with_outdir:
        src = _read(os.path.join(_inference, rel))
        # Must use _repo_root variable, not a raw relative path like "../outputs"
        uses_repo_root = "_repo_root" in src and "outputs" in src
        bare_relative  = bool(re.search(r'["\']\.\./+outputs["\']', src))
        _record("OUTDIR", f"{rel} uses _repo_root for outputs/",
                uses_repo_root and not bare_relative,
                "uses bare relative path" if bare_relative else "")

    # Check the actual resolved path at runtime
    code = f"""
import sys, os
sys.path.insert(0, {repr(os.path.join(_inference, 'deploy'))})
sys.path.insert(0, {repr(os.path.join(_inference, 'common'))})
sys.path.insert(0, {repr(_repo_root)})
import benchmark_latency as bm
# The module-level _repo_root must match our expectation
assert hasattr(bm, '_repo_root'), "no _repo_root in module"
expected = os.path.normcase(os.path.normpath({repr(_repo_root)}))
actual   = os.path.normcase(os.path.normpath(bm._repo_root))
assert actual == expected, f"_repo_root mismatch: {{actual!r}} != {{expected!r}}"
out_dir = os.path.join(bm._repo_root, 'outputs')
assert os.path.isabs(out_dir), f"outputs dir not absolute: {{out_dir!r}}"
print("outputs dir:", out_dir)
"""
    rc, out = _py(code)
    _record("OUTDIR", "deploy/benchmark_latency._repo_root resolves correctly", rc == 0,
            out[:200] if rc != 0 else out.split("\n")[0])


def check_cross_track_deps() -> None:
    """[DEPS] Cross-track imports: faithful→deploy is allowed; deploy↛faithful."""
    deploy_dir   = os.path.join(_inference, "deploy")
    faithful_dir = os.path.join(_inference, "faithful")

    # Check deploy/ does NOT import from faithful/
    for fname in os.listdir(deploy_dir):
        if not fname.endswith(".py"):
            continue
        src = _read(os.path.join(deploy_dir, fname))
        imports_faithful = bool(re.search(
            r"import\s+(ablate_patches|compare_outputs)|from\s+(ablate_patches|compare_outputs)",
            src,
        ))
        _record("DEPS", f"deploy/{fname} does not import faithful/",
                not imports_faithful,
                "cross-track import found" if imports_faithful else "")

    # Check faithful/ imports deploy/ (expected) and note it
    for fname in os.listdir(faithful_dir):
        if not fname.endswith(".py"):
            continue
        src = _read(os.path.join(faithful_dir, fname))
        imports_deploy = bool(re.search(
            r"import\s+export_onnx|import\s+_exp\b|from\s+export_onnx",
            src,
        ))
        if imports_deploy:
            _record("DEPS", f"faithful/{fname} → deploy/export_onnx (intentional)",
                    None,  # WARN = informational
                    "faithful/ uses deploy/ patches — expected cross-track dependency")


def check_onnx_convention() -> None:
    """[ONNX] ONNX file must live at inference/planTF.onnx (not inside a subdir)."""
    onnx_path = os.path.join(_inference, "planTF.onnx")
    wrong_locations = [
        os.path.join(_inference, "deploy",   "planTF.onnx"),
        os.path.join(_inference, "faithful", "planTF.onnx"),
        os.path.join(_inference, "common",   "planTF.onnx"),
    ]
    for wrong in wrong_locations:
        _record("ONNX", f"planTF.onnx not in subdir {os.path.basename(os.path.dirname(wrong))}/",
                not os.path.isfile(wrong), "found in wrong location" if os.path.isfile(wrong) else "")

    # Verify each script's hardened default agrees with inference/planTF.onnx
    for track, fname, arg in [
        ("deploy",   "export_onnx.py",      "--out"),
        ("deploy",   "benchmark_latency.py", "--onnx"),
        ("faithful", "compare_outputs.py",   "--onnx"),
    ]:
        src = _read(os.path.join(_inference, track, fname))
        # After hardening, scripts use _inference / "planTF.onnx"
        uses_inference = bool(re.search(r'_inference.*planTF\.onnx|planTF\.onnx.*_inference', src))
        _record("ONNX", f"{track}/{fname} {arg} references _inference/planTF.onnx",
                uses_inference,
                "default ONNX path may not match convention" if not uses_inference else "")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(verbose: bool) -> int:
    """Pretty-print results; return exit code (0 = all PASS/WARN, 1 = any FAIL)."""
    cats = sorted(set(r[0] for r in _results))
    total_pass = sum(1 for r in _results if r[2] == "PASS")
    total_warn = sum(1 for r in _results if r[2] == "WARN")
    total_fail = sum(1 for r in _results if r[2] == "FAIL")

    print(f"\n{'='*72}")
    print(f"  planTF inference/ refactor verification")
    print(f"{'='*72}")

    for cat in cats:
        cat_results = [r for r in _results if r[0] == cat]
        print(f"\n  [{cat}]")
        for _, label, status, detail in cat_results:
            icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
            line = f"    {icon} [{status}]  {label}"
            print(line)
            if detail and (verbose or status == "FAIL"):
                print(f"           {detail}")

    print(f"\n{'='*72}")
    print(f"  Summary: {total_pass} PASS  {total_warn} WARN  {total_fail} FAIL")
    print(f"{'='*72}\n")

    if total_fail == 0:
        print("  Result: OVERALL PASS — structure is sound.\n")
        return 0
    else:
        print("  Result: FAILURES DETECTED — see items marked [FAIL] above.\n")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-refactor structural verification for inference/",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detail text for WARN items as well as FAIL",
    )
    args = parser.parse_args()

    print("Running refactor verification checks ...")

    check_files()
    check_imports()
    check_api()
    check_dedup()
    check_orch()
    check_paths()
    check_outdir()
    check_cross_track_deps()
    check_onnx_convention()

    rc = _print_report(args.verbose)
    sys.exit(rc)


if __name__ == "__main__":
    main()
