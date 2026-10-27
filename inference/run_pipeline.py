"""
run_pipeline.py
---------------
Top-level CLI driver for the two planTF inference tracks.

  --mode deploy     Run the deployment pipeline:
                      1. Export to ONNX  (inference/deploy/export_onnx.py)
                      2. Latency benchmark (inference/deploy/benchmark_latency.py)

  --mode faithful   Run the fidelity pipeline:
                      1. Three-way output comparison (inference/faithful/compare_outputs.py)
                      2. Per-patch ablation           (inference/faithful/ablate_patches.py)

All sub-scripts are invoked via subprocess so each sets up its own
sys.path and imports cleanly.

Usage (from repo root):
    python inference/run_pipeline.py --mode deploy
    python inference/run_pipeline.py --mode faithful
    python inference/run_pipeline.py --mode deploy   --agents 33 --polygons 152
    python inference/run_pipeline.py --mode faithful --agents 33 --polygons 152
"""

import argparse
import os
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_deploy_dir  = os.path.join(_here, "deploy")
_faithful_dir = os.path.join(_here, "faithful")


def _run(script: str, extra_args: list[str]) -> int:
    """Run a Python script as a subprocess, returning its exit code."""
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*60}")
    print(f"  Running: {os.path.relpath(script)}")
    print(f"  Args   : {extra_args}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def run_deploy(args) -> None:
    """
    Deployment pipeline — ONNX export + latency benchmark.

    Track: inference/deploy/
    Branch: onnx-deployment
    Goal: validate that the ONNX export runs correctly and measure throughput.
    """
    extra = []
    if args.agents:
        extra += ["--agents", str(args.agents)]
    if args.polygons:
        extra += ["--polygons", str(args.polygons)]
    if args.ckpt:
        extra += ["--ckpt", args.ckpt]

    print("\n" + "#" * 60)
    print("#  DEPLOYMENT TRACK")
    print("#  Branch: onnx-deployment")
    print("#  Focus : ONNX export + throughput benchmarking")
    print("#" * 60)

    # Step 1: Export ONNX
    rc = _run(
        os.path.join(_deploy_dir, "export_onnx.py"),
        extra + ["--verify"],
    )
    if rc != 0:
        print(f"ERROR: export_onnx.py exited with code {rc}")
        sys.exit(rc)

    # Step 2: Latency benchmark
    bench_extra = extra + ["--warmup", "5", "--runs", "50", "--save"]
    if args.onnx:
        bench_extra += ["--onnx", args.onnx]
    rc = _run(
        os.path.join(_deploy_dir, "benchmark_latency.py"),
        bench_extra,
    )
    if rc != 0:
        print(f"ERROR: benchmark_latency.py exited with code {rc}")
        sys.exit(rc)

    print("\n" + "#" * 60)
    print("#  Deployment pipeline complete.")
    print("#  ONNX: inference/planTF.onnx")
    print("#  Latency data: outputs/latency_*.npy")
    print("#" * 60)


def run_faithful(args) -> None:
    """
    Fidelity pipeline — three-way comparison + per-patch ablation.

    Track: inference/faithful/
    Branch: faithful-natten
    Goal: quantify divergence from original model; guide NATTEN improvements.
    """
    extra = []
    if args.agents:
        extra += ["--agents", str(args.agents)]
    if args.polygons:
        extra += ["--polygons", str(args.polygons)]
    if args.ckpt:
        extra += ["--ckpt", args.ckpt]

    print("\n" + "#" * 60)
    print("#  FAITHFUL TRACK")
    print("#  Branch: faithful-natten")
    print("#  Focus : fidelity to original model / NATTEN accuracy")
    print("#" * 60)

    # Step 1: Three-way comparison (Original / Patched / ONNX)
    # Always re-export ONNX in faithful mode: the stored ONNX may have been
    # compiled with different input sizes (A=33, M=152).  A fresh export
    # guarantees the shapes match the comparison run.
    cmp_extra = extra + ["--save", "--reexport"]
    if args.onnx:
        cmp_extra += ["--onnx", args.onnx]
    rc = _run(
        os.path.join(_faithful_dir, "compare_outputs.py"),
        cmp_extra,
    )
    if rc != 0:
        print(f"ERROR: compare_outputs.py exited with code {rc}")
        sys.exit(rc)

    # Step 2: Per-patch ablation
    rc = _run(
        os.path.join(_faithful_dir, "ablate_patches.py"),
        extra,
    )
    if rc != 0:
        print(f"ERROR: ablate_patches.py exited with code {rc}")
        sys.exit(rc)

    print("\n" + "#" * 60)
    print("#  Faithful pipeline complete.")
    print("#  Saved outputs: outputs/{original,patched,onnx}.npy")
    print("#  See FIDELITY_REPORT.md for interpretation.")
    print("#" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="planTF inference pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  deploy    ONNX export + latency benchmark  (inference/deploy/)
  faithful  Output comparison + patch ablation (inference/faithful/)

Examples:
  python inference/run_pipeline.py --mode deploy
  python inference/run_pipeline.py --mode faithful
  python inference/run_pipeline.py --mode deploy --agents 33 --polygons 152
        """,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["deploy", "faithful"],
        help="Which pipeline track to run",
    )
    parser.add_argument("--ckpt",     default=None, help="Override checkpoint path")
    parser.add_argument("--onnx",     default=None, help="Override ONNX file path")
    parser.add_argument("--agents",   type=int, default=None)
    parser.add_argument("--polygons", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "deploy":
        run_deploy(args)
    elif args.mode == "faithful":
        run_faithful(args)


if __name__ == "__main__":
    main()
