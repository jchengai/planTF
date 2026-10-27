"""
compare_outputs.py
------------------
Fidelity validation pipeline for the planTF ONNX export.

Compares output_trajectory (shape: [1, 80, 3]) between:
  1. Original PyTorch model   — unmodified, loaded from checkpoint
  2. Patched PyTorch model    — same in-place patches applied before ONNX export
  3. ONNX Runtime model       — loaded from the exported .onnx file

Metrics reported per pair:
  - Max absolute error
  - Mean absolute error
  - Mean position error (L2 on x,y)
  - Mean heading error  (|Δθ|)
  - Final-step position error
  - Final-step heading error

Outputs are optionally saved to outputs/ as .npy files for offline inspection.

Usage:
    python inference/compare_outputs.py
    python inference/compare_outputs.py --reexport          # re-export ONNX first
    python inference/compare_outputs.py --onnx path/to.onnx # use existing ONNX
    python inference/compare_outputs.py --save              # save .npy outputs
    python inference/compare_outputs.py --agents 16 --polygons 64
    python inference/compare_outputs.py --natten global      # legacy export
    python inference/compare_outputs.py --valid-agents 33 --valid-polygons 152  # no padding

Inputs are padded by default (--valid-agents / --valid-polygons) because
real scenes are; with all-valid inputs, key-padding-mask bugs are invisible.
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_inference, "common"))
sys.path.insert(0, os.path.join(_inference, "deploy"))

import numpy as np
import torch

from run_inference import load_model, make_dummy_input, _build_onnx_feeds  # noqa: E402
from patches import apply_onnx_export_patches  # noqa: E402
from export_onnx import try_export


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def _apply_patches(model: torch.nn.Module, natten: str = "exact") -> None:
    """Apply all ONNX-compatibility patches in-place (mirrors export_onnx.py)."""
    apply_onnx_export_patches(model, natten=natten)


def _export_onnx(ckpt_path: str, out_path: str, agents: int, polygons: int,
                 natten: str = "exact") -> None:
    """Re-export ONNX using the same patches as export_onnx.py."""
    device = torch.device("cpu")
    model = load_model(ckpt_path, device)
    _apply_patches(model, natten=natten)
    data = make_dummy_input(A=agents, M=polygons, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try_export(model, data, out_path, opset=14)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compare(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> None:
    """Print a formatted comparison block between two output_trajectory arrays."""
    diff = np.abs(a - b)

    max_abs   = diff.max()
    mean_abs  = diff.mean()
    pos_err   = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean()
    hdg_err   = np.abs(a[..., 2] - b[..., 2]).mean()
    final_pos = np.linalg.norm(a[0, -1, :2] - b[0, -1, :2])
    final_hdg = abs(float(a[0, -1, 2]) - float(b[0, -1, 2]))

    print(f"\n{'='*50}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'='*50}")
    print(f"  Max abs error          : {max_abs:.6f}")
    print(f"  Mean abs error         : {mean_abs:.6f}")
    print(f"  Mean position error    : {pos_err:.6f}  m")
    print(f"  Mean heading error     : {hdg_err:.6f}  rad")
    print(f"  Final-step pos error   : {final_pos:.6f}  m")
    print(f"  Final-step hdg error   : {final_hdg:.6f}  rad")

    # Verdict
    if max_abs < 1e-4:
        verdict = "NUMERICALLY EQUIVALENT  (max < 1e-4)"
    elif max_abs < 1e-2:
        verdict = "CLOSE  (max < 1e-2, likely acceptable)"
    elif max_abs < 0.1:
        verdict = "NOTICEABLE DRIFT  (max < 0.1)"
    else:
        verdict = "SIGNIFICANT DIVERGENCE  (max >= 0.1)"
    print(f"\n  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    _default_onnx = os.path.join(_inference, "planTF.onnx")
    parser = argparse.ArgumentParser(description="Compare Original / Patched / ONNX outputs")
    parser.add_argument("--ckpt",     default=_default_ckpt)
    parser.add_argument("--onnx",     default=_default_onnx,
                        help="Path to .onnx file (will be auto-exported if missing or --reexport)")
    parser.add_argument("--reexport", action="store_true",
                        help="Re-export ONNX before comparison (keeps it in sync with patches)")
    parser.add_argument("--agents",   type=int, default=33)
    parser.add_argument("--polygons", type=int, default=152)
    parser.add_argument("--natten",   choices=["exact", "global"], default="exact",
                        help="NATTEN replacement: exact (default) or legacy global MHA")
    parser.add_argument("--valid-agents",   type=int, default=20,
                        help="Agents marked valid (rest padded); padding exercises key_padding_mask")
    parser.add_argument("--valid-polygons", type=int, default=100,
                        help="Polygons marked valid (rest padded)")
    parser.add_argument("--save",     action="store_true",
                        help="Save outputs to outputs/{original,patched,onnx}.npy")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Load Original model
    # ------------------------------------------------------------------
    print("Loading original model ...")
    model_orig = load_model(args.ckpt, device)

    # ------------------------------------------------------------------
    # 2. Load Patched model (fresh load + patches)
    # ------------------------------------------------------------------
    print("Loading patched model ...")
    model_patched = load_model(args.ckpt, device)
    _apply_patches(model_patched, natten=args.natten)

    # ------------------------------------------------------------------
    # 3. Export / load ONNX
    # ------------------------------------------------------------------
    onnx_missing = not os.path.exists(args.onnx)
    if args.reexport or onnx_missing:
        if onnx_missing:
            print(f"ONNX file not found at {args.onnx} — exporting ...")
        else:
            print(f"Re-exporting ONNX to {args.onnx} ...")
        _export_onnx(args.ckpt, args.onnx, args.agents, args.polygons, natten=args.natten)

    print(f"Loading ONNX model from {args.onnx} ...")
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"  ONNX inputs  ({len(sess.get_inputs())}):  "
          f"{[i.name for i in sess.get_inputs()]}")

    # ------------------------------------------------------------------
    # 4. Shared input (same seed → same tensors for all three models)
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    data = make_dummy_input(A=args.agents, M=args.polygons, device=device,
                            valid_agents=args.valid_agents,
                            valid_polygons=args.valid_polygons)
    print(f"\nInput: A={args.agents} agents ({args.valid_agents} valid), "
          f"M={args.polygons} polygons ({args.valid_polygons} valid)")

    # ------------------------------------------------------------------
    # 5. Run inference
    # ------------------------------------------------------------------
    with torch.no_grad():
        out_orig    = model_orig(data)["output_trajectory"].cpu().numpy()
        out_patched = model_patched(data)["output_trajectory"].cpu().numpy()

    feeds  = _build_onnx_feeds(sess, data)
    out_onnx = sess.run(None, feeds)[0]   # shape (1, 80, 3)

    print(f"\nOutput shapes:")
    print(f"  Original : {out_orig.shape}")
    print(f"  Patched  : {out_patched.shape}")
    print(f"  ONNX     : {out_onnx.shape}")

    # ------------------------------------------------------------------
    # 6. Comparison metrics
    # ------------------------------------------------------------------
    compare("Original (PyTorch)", out_orig,    "Patched (PyTorch)", out_patched)
    compare("Original (PyTorch)", out_orig,    "ONNX (Runtime)",    out_onnx)
    compare("Patched  (PyTorch)", out_patched, "ONNX (Runtime)",    out_onnx)

    # ------------------------------------------------------------------
    # 7. Optional: save .npy
    # ------------------------------------------------------------------
    if args.save:
        out_dir = os.path.join(_repo_root, "outputs", "fidelity")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "trajectory_original.npy"), out_orig)
        np.save(os.path.join(out_dir, "trajectory_patched.npy"),  out_patched)
        np.save(os.path.join(out_dir, "trajectory_onnx.npy"),     out_onnx)
        print(f"\nSaved outputs to {os.path.abspath(out_dir)}/")

    # ------------------------------------------------------------------
    # 8. Recommendation
    # ------------------------------------------------------------------
    max_orig_patched = np.abs(out_orig - out_patched).max()
    max_patched_onnx = np.abs(out_patched - out_onnx).max()
    max_orig_onnx    = np.abs(out_orig - out_onnx).max()

    print("\n" + "="*50)
    print("DIAGNOSIS")
    print("="*50)

    if max_patched_onnx < 1e-4:
        print("OK: Patched PyTorch ~ ONNX  (max {:.2e}) — ONNX export is faithful to the".format(max_patched_onnx))
        print("   patched model. No tracing artifacts.")
    else:
        print("WARNING: Patched PyTorch != ONNX  (max {:.2e}) — tracing introduced extra error.".format(max_patched_onnx))
        print("   Check TracerWarnings during export.")

    if max_orig_patched < 1e-2:
        print("Original ~ Patched  (max {:.2e}) — patches are safe transformations.".format(max_orig_patched))
        print("   -> Safe to proceed to TensorRT.")
    else:
        print("WARNING: Original != Patched  (max {:.2e}) — patches change model behaviour.".format(max_orig_patched))
        print("   Primary cause: NATTEN (windowed + RPB) → global MultiheadAttention (no RPB).")
        print("   The ONNX model is a faithful export of the *approximated* model, not the original.")
        if max_patched_onnx < 1e-4:
            print("\n   Options:")
            print("   A) Accept the approximation → proceed to TensorRT with current ONNX.")
            print("      Suitable if throughput benchmarking is the goal, not exact replication.")
            print("   B) Register NATTEN as a custom ONNX/TRT op for exact replication.")
            print("      See: https://github.com/SHI-Labs/NATTEN")
    print("="*50)


if __name__ == "__main__":
    main()
