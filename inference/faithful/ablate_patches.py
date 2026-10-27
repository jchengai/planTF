"""
ablate_patches.py
-----------------
Isolate which ONNX compatibility patch(es) cause divergence from the original model.

For each variant, a fresh model is loaded and ONLY that one patch is applied.
Output is compared against the unmodified original.

Variants tested (in order of increasing expected impact):
  1. original                          → baseline (self-comparison, errors = 0)
  2. only patch_planning_model_for_onnx → atan2 substitution
  3. only patch_boolean_indexing_for_onnx → mask → multiply / where
  4. only patch_agent_encoder_for_onnx   → AgentEncoder bool-index → gate
  5. only patch_mha_for_onnx             → batch_first + dropped key_padding_mask
  6. only patch_natten_for_onnx          → NATTEN → global MHA
  7. all patches                         → cumulative effect

Usage:
    python inference/ablate_patches.py
    python inference/ablate_patches.py --ckpt path/to/other.ckpt
    python inference/ablate_patches.py --agents 16 --polygons 64
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

from run_inference import load_model, make_dummy_input, _metrics
from patches import (
    patch_natten_for_onnx,
    patch_natten_faithful,
    patch_mha_for_onnx,
    patch_boolean_indexing_for_onnx,
    patch_agent_encoder_for_onnx,
    patch_planning_model_for_onnx,
    patch_natten_exact,
    apply_onnx_export_patches,
)


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _run(model: torch.nn.Module, data: dict) -> np.ndarray:
    with torch.no_grad():
        return model(data)["output_trajectory"].cpu().numpy()


def _row(label: str, m: dict, width: int = 42) -> str:
    return (
        f"  {label:{width}s}"
        f"  max={m['max_abs']:.4e}"
        f"  mean={m['mean_abs']:.4e}"
        f"  pos={m['pos_err_m']:.4f}m"
        f"  hdg={m['hdg_err_rad']:.4f}rad"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=_default_ckpt)
    parser.add_argument("--agents",   type=int, default=33)
    parser.add_argument("--polygons", type=int, default=152)
    parser.add_argument("--valid-agents",   type=int, default=20,
                        help="Agents marked valid (rest padded); padding exercises key_padding_mask")
    parser.add_argument("--valid-polygons", type=int, default=100,
                        help="Polygons marked valid (rest padded)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    # Shared input — same for every variant
    torch.manual_seed(0)
    data = make_dummy_input(A=args.agents, M=args.polygons, device=device,
                            valid_agents=args.valid_agents,
                            valid_polygons=args.valid_polygons)

    # ------------------------------------------------------------------
    # Baseline: original unmodified model
    # ------------------------------------------------------------------
    print(f"Loading baseline (original) model ...")
    baseline = load_model(args.ckpt, device)
    out_orig = _run(baseline, data)
    del baseline

    # ------------------------------------------------------------------
    # Define variants: (label, patch_fn or None)
    # ------------------------------------------------------------------
    variants = [
        ("original (self-check)",                  None),
        ("only patch_planning_model",              patch_planning_model_for_onnx),
        ("only patch_boolean_indexing",            patch_boolean_indexing_for_onnx),
        ("only patch_agent_encoder",               patch_agent_encoder_for_onnx),
        ("only patch_mha",                         patch_mha_for_onnx),
        ("only patch_natten (global MHA)",         patch_natten_for_onnx),
        ("only patch_natten_faithful (local MHA)", patch_natten_faithful),
        ("only patch_natten_exact",                patch_natten_exact),
        ("all patches, legacy global NATTEN",      lambda m: apply_onnx_export_patches(m, natten="global")),
        ("all patches (ONNX export config)",       lambda m: apply_onnx_export_patches(m, natten="exact")),
    ]

    # ------------------------------------------------------------------
    # Run each variant
    # ------------------------------------------------------------------
    print(f"\nRunning ablation (A={args.agents}, M={args.polygons}) ...\n")
    header = (
        f"  {'Variant':<42}"
        f"  {'max |Δ|':<14}"
        f"  {'mean |Δ|':<14}"
        f"  {'pos err':<12}"
        f"  {'hdg err'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for label, patch_fn in variants:
        model = load_model(args.ckpt, device)
        if patch_fn is not None:
            patch_fn(model)
        out = _run(model, data)
        m = _metrics(out_orig, out)
        results.append((label, m))
        print(_row(label, m))
        del model

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print()
    print("Verdict:")
    dominant = max(
        [(label, m) for label, m in results if "only" in label],
        key=lambda x: x[1]["max_abs"],
    )
    print(f"  Largest single-patch divergence: {dominant[0]}")
    print(f"    max abs error = {dominant[1]['max_abs']:.4e}")
    print(f"    mean pos error = {dominant[1]['pos_err_m']:.4f} m")

    safe = [
        label for label, m in results
        if "only" in label and m["max_abs"] < 1e-3
    ]
    if safe:
        print(f"  Patches with negligible impact (max < 1e-3): {safe}")


if __name__ == "__main__":
    main()
