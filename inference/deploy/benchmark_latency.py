"""
benchmark_latency.py
--------------------
Measure and compare forward-pass latency across three model variants:
  1. Original PyTorch       — unmodified model
  2. Patched PyTorch        — same patches applied for ONNX export
  3. ONNX Runtime           — loaded from inference/planTF.onnx

All variants receive the same seeded dummy input (no data loading overhead).
Reports mean and p95 latency over N runs after warmup.

Usage (from repo root):
    python inference/deploy/benchmark_latency.py
    python inference/deploy/benchmark_latency.py --runs 100 --warmup 10
    python inference/deploy/benchmark_latency.py --onnx path/to.onnx
"""

import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_inference, "common"))

import numpy as np
import torch

from run_inference import load_model, make_dummy_input, _build_onnx_feeds
from patches import apply_onnx_export_patches


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_pytorch(model: torch.nn.Module, data: dict, warmup: int, runs: int) -> np.ndarray:
    """Return per-run latencies in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            model(data)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(data)
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _time_onnx(sess, feeds: dict, warmup: int, runs: int) -> np.ndarray:
    """Return per-run latencies in ms."""
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _stats(times: np.ndarray) -> str:
    return (
        f"mean={times.mean():.1f}ms  "
        f"median={np.median(times):.1f}ms  "
        f"p95={np.percentile(times, 95):.1f}ms  "
        f"min={times.min():.1f}ms  "
        f"max={times.max():.1f}ms"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    _default_onnx = os.path.join(_inference, "planTF.onnx")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=_default_ckpt)
    parser.add_argument("--onnx",     default=_default_onnx)
    parser.add_argument("--agents",   type=int, default=33)
    parser.add_argument("--polygons", type=int, default=152)
    parser.add_argument("--natten",   choices=["exact", "global"], default="exact",
                        help="NATTEN replacement: exact (default) or legacy global MHA")
    parser.add_argument("--warmup",   type=int, default=5,
                        help="Warmup runs (excluded from stats)")
    parser.add_argument("--runs",     type=int, default=50,
                        help="Timed runs per variant")
    parser.add_argument("--save",     action="store_true",
                        help="Save per-run latencies to outputs/latency_*.npy")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    torch.manual_seed(0)
    data = make_dummy_input(A=args.agents, M=args.polygons, device=device)

    print(f"Benchmark  A={args.agents}  M={args.polygons}"
          f"  warmup={args.warmup}  runs={args.runs}  device=cpu\n")

    results = {}

    # ------------------------------------------------------------------
    # 1. Original PyTorch
    # ------------------------------------------------------------------
    print("Loading original PyTorch model ...")
    model_orig = load_model(args.ckpt, device)
    print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
    t_orig = _time_pytorch(model_orig, data, args.warmup, args.runs)
    results["Original PyTorch"] = t_orig
    print("done")
    del model_orig

    # ------------------------------------------------------------------
    # 2. Patched PyTorch
    # ------------------------------------------------------------------
    print("Loading patched PyTorch model ...")
    model_patched = load_model(args.ckpt, device)
    apply_onnx_export_patches(model_patched, natten=args.natten)
    print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
    t_patched = _time_pytorch(model_patched, data, args.warmup, args.runs)
    results["Patched PyTorch"] = t_patched
    print("done")
    del model_patched

    # ------------------------------------------------------------------
    # 3. ONNX Runtime
    # ------------------------------------------------------------------
    if not os.path.exists(args.onnx):
        print(f"ONNX file not found at {args.onnx} — skipping ONNX benchmark.")
        print("Run:  python inference/export_onnx.py  to generate it.")
    else:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed — skipping ONNX benchmark.")
            print("Run:  pip install onnxruntime")
        else:
            print(f"Loading ONNX model from {args.onnx} ...")
            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            feeds = _build_onnx_feeds(sess, data)
            print(f"  Timing ({args.warmup} warmup, {args.runs} runs) ...", end=" ", flush=True)
            t_onnx = _time_onnx(sess, feeds, args.warmup, args.runs)
            results["ONNX Runtime"] = t_onnx
            print("done")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  {'Variant':<25}  {'mean':>8}  {'median':>8}  {'p95':>8}  {'min':>8}  {'max':>8}")
    print("  " + "-" * 63)
    for label, times in results.items():
        print(
            f"  {label:<25}"
            f"  {times.mean():>7.1f}ms"
            f"  {np.median(times):>7.1f}ms"
            f"  {np.percentile(times, 95):>7.1f}ms"
            f"  {times.min():>7.1f}ms"
            f"  {times.max():>7.1f}ms"
        )
    print("=" * 65)

    # Speedup relative to original
    if len(results) > 1:
        print()
        orig_mean = results["Original PyTorch"].mean()
        for label, times in results.items():
            if label == "Original PyTorch":
                continue
            ratio = orig_mean / times.mean()
            direction = f"{ratio:.2f}× faster" if ratio >= 1 else f"{1/ratio:.2f}× slower"
            print(f"  {label} vs Original: {direction}")

    # ------------------------------------------------------------------
    # Optional: save per-run latencies
    # ------------------------------------------------------------------
    if args.save:
        out_dir = os.path.join(_repo_root, "outputs", "fidelity")
        os.makedirs(out_dir, exist_ok=True)
        for label, times in results.items():
            fname = label.lower().replace(" ", "_")
            path = os.path.join(out_dir, f"latency_{fname}.npy")
            np.save(path, times)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
