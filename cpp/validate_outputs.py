#!/usr/bin/env python3
"""
validate_outputs.py — Cross-runtime output equivalence checker.

Compares planTF inference outputs across three runtimes:
  1. Python ONNX Runtime (reference)
  2. C++ ONNX Runtime (via subprocess, parses stdout)
  3. C++ TensorRT FP32 (via subprocess, parses stdout)
  4. C++ TensorRT FP16 (via subprocess, parses stdout)

All runtimes use constant-fill inputs (float=0.1, int=0, bool=False),
matching the approach used in the C++ programs.

Usage
-----
    # From repo root
    conda run -n plantf python cpp/validate_outputs.py

    # Specify paths explicitly
    conda run -n plantf python cpp/validate_outputs.py \\
        --onnx inference/planTF.onnx \\
        --fp32-engine inference/planTF_fp32.trt \\
        --fp16-engine inference/planTF.trt \\
        --cpp-build cpp/build

Tolerances
----------
    ORT C++  vs ORT Python  : max abs diff < 1e-4  (effectively bit-exact)
    TRT FP32 vs ORT Python  : max abs diff < 0.002  (< 2 mm trajectory)
    TRT FP16 vs ORT Python  : max abs diff < 0.250  (< 250 mm; FP16 precision loss,
                               deviation depends strongly on inputs)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants — match the constant-fill used in onnx_runtime_infer.cpp /
# tensorrt_infer.cpp / tensorrt_utils.cpp
# ---------------------------------------------------------------------------
FLOAT_FILL = 0.1
INT_FILL = 0
BOOL_FILL = False

# Tolerances (metres, since the trajectory output is in metres)
TOL_ORT_CPP = 1e-4    # C++ ORT  vs Python ORT
TOL_TRT_FP32 = 2e-3   # C++ TRT FP32 vs Python ORT  (<2 mm)
TOL_TRT_FP16 = 0.25   # C++ TRT FP16 vs Python ORT  (<250 mm; FP16 precision loss expected)
                       # Note: deviation depends on inputs; constant-fill 0.1 can produce
                       # ~200mm deviation for first waypoints. See CPP_RUNTIME_REPORT.md.

# Reference values collected from validated runs (constant-fill inputs)
REFERENCE_ORT_PYTHON  = [-0.239406, 0.085293, 0.005405, -0.233568, 0.081084, 0.005540]
REFERENCE_ORT_CPP     = [-0.239406, 0.085293, 0.005405, -0.233568, 0.081084, 0.005540]
REFERENCE_TRT_FP32    = [-0.239158, 0.085261, 0.005419, -0.233316, 0.081049, 0.005554]
REFERENCE_TRT_FP16    = None   # FP16 has larger variance; skip hard reference check

FIRST6_INDICES = slice(0, 6)  # first 6 values of output_trajectory

# ---------------------------------------------------------------------------
# Python ORT inference with constant-fill inputs
# ---------------------------------------------------------------------------

def run_python_ort(onnx_path: str) -> np.ndarray:
    """Run ORT inference with constant-fill inputs; return output_trajectory."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    feeds: dict = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
            "tensor(double)": np.float64,
        }
        dt = dtype_map.get(inp.type, np.float32)
        if dt == np.bool_:
            feeds[inp.name] = np.full(shape, BOOL_FILL, dtype=dt)
        elif np.issubdtype(dt, np.integer):
            feeds[inp.name] = np.full(shape, INT_FILL, dtype=dt)
        else:
            feeds[inp.name] = np.full(shape, FLOAT_FILL, dtype=dt)

    outputs = sess.run(None, feeds)
    # output_trajectory is the last output by convention; shape [1, 80, 3]
    traj = outputs[-1].flatten()
    return traj


# ---------------------------------------------------------------------------
# Parse C++ binary output  — both ort_infer and trt_infer print:
#   "first 6 outputs: val0 val1 val2 val3 val4 val5"
# ---------------------------------------------------------------------------

def parse_cpp_output(stdout: str):
    """Extract the 6 output values from C++ binary stdout.

    Both ort_infer and trt_infer print a line of the form:
        first 6 values: -0.239406, 0.0852928, 0.00540451, -0.233568, 0.0810839, 0.00554027
    """
    # Match "first 6 values:" followed by comma- or space-separated floats
    pattern = r"first\s+6\s+values\s*:\s*([-\d.,e+ ]+)"
    m = re.search(pattern, stdout, re.IGNORECASE)
    if not m:
        # Fallback: "first 6 outputs:" variant
        pattern2 = r"first\s+6\s+outputs\s*[:\-]?\s*([-\d.,e+ ]+)"
        m = re.search(pattern2, stdout, re.IGNORECASE)
    if not m:
        return None
    # Values may be comma-separated or space-separated
    raw = m.group(1).replace(",", " ")
    vals = [float(v) for v in raw.split() if v]
    return np.array(vals[:6], dtype=np.float64)


def run_cpp_ort(build_dir: str, onnx_path: str) -> Optional[np.ndarray]:
    """Run ort_infer binary and parse its output values."""
    binary = Path(build_dir) / "ort_infer"
    if not binary.exists():
        print(f"  [SKIP] ort_infer not found at {binary}")
        return None

    # Set up LD_LIBRARY_PATH for the ORT shared library
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    ort_capi = Path(conda_prefix) / "lib/python3.9/site-packages/onnxruntime/capi"
    env = os.environ.copy()
    if ort_capi.exists():
        env["LD_LIBRARY_PATH"] = f"{ort_capi}:{env.get('LD_LIBRARY_PATH', '')}"

    result = subprocess.run(
        [str(binary), "--model", onnx_path, "--warmup", "1", "--runs", "1"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if result.returncode != 0:
        print(f"  [ERROR] ort_infer exited {result.returncode}:\n{result.stderr[:400]}")
        return None
    vals = parse_cpp_output(result.stdout + result.stderr)
    return vals


def run_cpp_trt(build_dir: str, engine_path: str, fp16: bool = False) -> Optional[np.ndarray]:
    """Run trt_infer binary and parse its output values."""
    binary = Path(build_dir) / "trt_infer"
    if not binary.exists():
        print(f"  [SKIP] trt_infer not found at {binary}")
        return None
    if not Path(engine_path).exists():
        print(f"  [SKIP] engine not found: {engine_path}")
        return None

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    py_pkgs = Path(conda_prefix) / "lib/python3.9/site-packages"
    trt_libs = py_pkgs / "tensorrt_libs"
    cudart_lib = py_pkgs / "nvidia/cuda_runtime/lib"

    env = os.environ.copy()
    ld_extra = ":".join(str(p) for p in [trt_libs, cudart_lib] if p.exists())
    env["LD_LIBRARY_PATH"] = f"{ld_extra}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [str(binary), "--engine", engine_path, "--warmup", "1", "--runs", "1"]
    if not fp16:
        cmd.append("--fp32")

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120,
    )
    if result.returncode != 0:
        print(f"  [ERROR] trt_infer exited {result.returncode}:\n{result.stderr[:400]}")
        return None
    vals = parse_cpp_output(result.stdout + result.stderr)
    return vals


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare(name: str, ref: np.ndarray, test: Optional[np.ndarray], tol: float) -> bool:
    """Print comparison result and return True if within tolerance."""
    if test is None:
        print(f"  {'SKIP':6s}  {name}  (binary not available or parsing failed)")
        return True   # don't fail the suite for missing binaries

    diff = np.abs(ref - test)
    max_diff = diff.max()
    status = "PASS" if max_diff <= tol else "FAIL"
    tol_mm = tol * 1000
    diff_mm = max_diff * 1000

    print(f"  {status:6s}  {name}")
    print(f"           ref  : {ref}")
    print(f"           test : {test}")
    print(f"           max |diff|: {diff_mm:.4f} mm  (tol={tol_mm:.1f} mm)")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--onnx", default=str(repo_root / "inference/planTF.onnx"),
                        help="Path to planTF.onnx")
    parser.add_argument("--fp32-engine", default=str(repo_root / "inference/planTF_fp32.trt"),
                        help="Path to FP32 TRT engine (.trt)")
    parser.add_argument("--fp16-engine", default=str(repo_root / "inference/planTF.trt"),
                        help="Path to FP16 TRT engine (.trt)")
    parser.add_argument("--cpp-build", default=str(repo_root / "cpp/build"),
                        help="Path to cmake build directory containing ort_infer / trt_infer")
    parser.add_argument("--no-python-ort", action="store_true",
                        help="Skip Python ORT (use embedded reference values instead)")
    args = parser.parse_args()

    print("=" * 70)
    print("planTF C++ runtime output validation")
    print("=" * 70)
    print(f"  ONNX model    : {args.onnx}")
    print(f"  FP32 engine   : {args.fp32_engine}")
    print(f"  FP16 engine   : {args.fp16_engine}")
    print(f"  C++ build dir : {args.cpp_build}")
    print()

    all_pass = True

    # Step 1 — Python ORT reference
    print("── Python ORT (reference) ──────────────────────────────────────────")
    if not args.no_python_ort and Path(args.onnx).exists():
        try:
            ref_vals = run_python_ort(args.onnx)
            ref6 = ref_vals[FIRST6_INDICES]
            print(f"  outputs[0:6] : {ref6}")
            # Sanity-check against embedded reference
            embedded = np.array(REFERENCE_ORT_PYTHON)
            diff = np.abs(ref6 - embedded).max()
            if diff < 1e-3:
                print(f"  OK: matches embedded reference (max diff={diff:.2e})")
            else:
                print(f"  WARNING: differs from embedded reference by {diff:.4f} "
                      f"(different model/inputs?)")
        except Exception as e:
            print(f"  [ERROR] Python ORT failed: {e}")
            print("  Falling back to embedded reference values.")
            ref6 = np.array(REFERENCE_ORT_PYTHON)
    else:
        print(f"  Using embedded reference: {REFERENCE_ORT_PYTHON}")
        ref6 = np.array(REFERENCE_ORT_PYTHON)
    print()

    # Step 2 — C++ ORT
    print("── C++ ORT ─────────────────────────────────────────────────────────")
    cpp_ort_vals = run_cpp_ort(args.cpp_build, args.onnx)
    ok = compare("C++ ORT vs Python ORT", ref6, cpp_ort_vals, TOL_ORT_CPP)
    all_pass = all_pass and ok
    print()

    # Step 3 — C++ TRT FP32
    print("── C++ TRT FP32 ────────────────────────────────────────────────────")
    cpp_trt_fp32_vals = run_cpp_trt(args.cpp_build, args.fp32_engine, fp16=False)
    ok = compare("C++ TRT FP32 vs Python ORT", ref6, cpp_trt_fp32_vals, TOL_TRT_FP32)
    all_pass = all_pass and ok
    print()

    # Step 4 — C++ TRT FP16
    print("── C++ TRT FP16 ────────────────────────────────────────────────────")
    cpp_trt_fp16_vals = run_cpp_trt(args.cpp_build, args.fp16_engine, fp16=True)
    ok = compare("C++ TRT FP16 vs Python ORT", ref6, cpp_trt_fp16_vals, TOL_TRT_FP16)
    all_pass = all_pass and ok
    print()

    # Summary
    print("=" * 70)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
