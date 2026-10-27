"""
export_onnx.py
--------------
Exports planTF to ONNX with fixed input shapes.

NOTE: The model uses NATTEN (Neighborhood Attention), which has no ONNX symbolic.
Before export, `patch_natten_exact()` (default, `--natten exact`) replaces every
NeighborhoodAttention1D with a gather-based re-implementation that keeps the
relative positional bias and NATTEN's edge windows, so the exported graph matches
the original model to float precision. `--natten global` selects the legacy
global-MHA approximation (RPB dropped, ~0.35 m mean position error).

To restore NATTEN for nuPlan runs, simply reload the original checkpoint with
`load_model()` from run_inference.py (no in-place patching done there).

Usage (from repo root):
    python inference/deploy/export_onnx.py
    python inference/deploy/export_onnx.py --ckpt path/to/other.ckpt
    python inference/deploy/export_onnx.py --agents 33 --polygons 152 --opset 14
    python inference/deploy/export_onnx.py --natten global   # legacy approximation
"""

import argparse
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_inference, "common"))

import torch
from run_inference import load_model, make_dummy_input   # reuse helpers


# ---------------------------------------------------------------------------
# Patch functions and wrapper classes live in common/patches.py.
# Re-exported here so callers using `import export_onnx` continue to work.
# ---------------------------------------------------------------------------
from patches import (  # noqa: F401
    _MHAWrapper,
    _LocalWindowMHA,
    _ExactNA1D,
    _Bf1MHAWrapper,
    apply_onnx_export_patches,
    patch_natten_exact,
    patch_natten_for_onnx,
    patch_natten_faithful,
    patch_mha_for_onnx,
    patch_boolean_indexing_for_onnx,
    patch_agent_encoder_for_onnx,
    patch_planning_model_for_onnx,
)


def try_export(model, data, output_path, opset):
    """Attempt torch.onnx.export and return True on success."""
    try:
        # Flatten nested dict into positional args is not straightforward;
        # we wrap the model to accept a flat tuple of tensors.
        class WrappedModel(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(
                self,
                # agent
                a_position, a_heading, a_velocity, a_shape, a_category,
                a_valid_mask, a_target,
                # map
                m_point_position, m_point_vector, m_point_orientation,
                m_point_side, m_polygon_center, m_polygon_position,
                m_polygon_orientation, m_polygon_type, m_polygon_on_route,
                m_polygon_tl_status, m_polygon_has_speed_limit,
                m_polygon_speed_limit, m_valid_mask,
                # scalars
                current_state, origin, angle,
            ):
                data = {
                    "agent": {
                        "position": a_position,
                        "heading": a_heading,
                        "velocity": a_velocity,
                        "shape": a_shape,
                        "category": a_category,
                        "valid_mask": a_valid_mask,
                        "target": a_target,
                    },
                    "map": {
                        "point_position": m_point_position,
                        "point_vector": m_point_vector,
                        "point_orientation": m_point_orientation,
                        "point_side": m_point_side,
                        "polygon_center": m_polygon_center,
                        "polygon_position": m_polygon_position,
                        "polygon_orientation": m_polygon_orientation,
                        "polygon_type": m_polygon_type,
                        "polygon_on_route": m_polygon_on_route,
                        "polygon_tl_status": m_polygon_tl_status,
                        "polygon_has_speed_limit": m_polygon_has_speed_limit,
                        "polygon_speed_limit": m_polygon_speed_limit,
                        "valid_mask": m_valid_mask,
                    },
                    "current_state": current_state,
                    "origin": origin,
                    "angle": angle,
                }
                out = self.inner(data)
                # return only output_trajectory for simplicity
                return out["output_trajectory"]

        wrapped = WrappedModel(model)
        wrapped.eval()

        def flatten(d):
            ag = d["agent"]
            mp = d["map"]
            return (
                ag["position"], ag["heading"], ag["velocity"], ag["shape"],
                ag["category"], ag["valid_mask"], ag["target"],
                mp["point_position"], mp["point_vector"], mp["point_orientation"],
                mp["point_side"], mp["polygon_center"], mp["polygon_position"],
                mp["polygon_orientation"], mp["polygon_type"], mp["polygon_on_route"],
                mp["polygon_tl_status"], mp["polygon_has_speed_limit"],
                mp["polygon_speed_limit"], mp["valid_mask"],
                d["current_state"], d["origin"], d["angle"],
            )

        flat_inputs = flatten(data)
        input_names = [
            "a_position", "a_heading", "a_velocity", "a_shape", "a_category",
            "a_valid_mask", "a_target",
            "m_point_position", "m_point_vector", "m_point_orientation",
            "m_point_side", "m_polygon_center", "m_polygon_position",
            "m_polygon_orientation", "m_polygon_type", "m_polygon_on_route",
            "m_polygon_tl_status", "m_polygon_has_speed_limit",
            "m_polygon_speed_limit", "m_valid_mask",
            "current_state", "origin", "angle",
        ]

        # --- Strategy 1: direct torch.onnx.export ---
        print("  Trying direct torch.onnx.export ...", end=" ", flush=True)
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapped,
                    flat_inputs,
                    output_path,
                    opset_version=opset,
                    input_names=input_names,
                    output_names=["output_trajectory"],
                    do_constant_folding=False,
                    training=torch.onnx.TrainingMode.EVAL,
                    verbose=False,
                )
            print("OK")
            print(f"ONNX export SUCCESS → {output_path}")
            return True
        except RuntimeError as e1:
            print(f"FAILED: {e1}")

        # --- Strategy 2: torch.jit.trace first, then export ---
        print("  Trying jit.trace → onnx.export ...", end=" ", flush=True)
        try:
            with torch.no_grad():
                traced = torch.jit.trace(wrapped, flat_inputs, strict=False)
                torch.onnx.export(
                    traced,
                    flat_inputs,
                    output_path,
                    opset_version=opset,
                    input_names=input_names,
                    output_names=["output_trajectory"],
                    do_constant_folding=False,
                    verbose=False,
                )
            print("OK")
            print(f"ONNX export SUCCESS (via jit.trace) → {output_path}")
            return True
        except Exception as e2:
            print(f"FAILED: {e2}")

        raise RuntimeError("Both export strategies failed (see above).")

    except Exception as e:
        print(f"ONNX export FAILED: {type(e).__name__}: {e}")
        return False


def verify_onnx(output_path, data):
    """Verify ONNX model with onnxruntime — dynamically matches ONNX input names."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        print("\nONNX model inputs:")
        for inp in sess.get_inputs():
            print(f"  {inp.name}: {inp.shape} {inp.type}")

        # Build a flat pool of all possible tensors keyed by input_name
        ag = data["agent"]
        mp = data["map"]
        pool = {
            "a_position":             ag["position"].numpy(),
            "a_heading":              ag["heading"].numpy(),
            "a_velocity":             ag["velocity"].numpy(),
            "a_shape":                ag["shape"].numpy(),
            "a_category":             ag["category"].numpy(),
            "a_valid_mask":           ag["valid_mask"].numpy(),
            "a_target":               ag["target"].numpy(),
            "m_point_position":       mp["point_position"].numpy(),
            "m_point_vector":         mp["point_vector"].numpy(),
            "m_point_orientation":    mp["point_orientation"].numpy(),
            "m_point_side":           mp["point_side"].numpy(),
            "m_polygon_center":       mp["polygon_center"].numpy(),
            "m_polygon_position":     mp["polygon_position"].numpy(),
            "m_polygon_orientation":  mp["polygon_orientation"].numpy(),
            "m_polygon_type":         mp["polygon_type"].numpy(),
            "m_polygon_on_route":     mp["polygon_on_route"].numpy(),
            "m_polygon_tl_status":    mp["polygon_tl_status"].numpy(),
            "m_polygon_has_speed_limit": mp["polygon_has_speed_limit"].numpy(),
            "m_polygon_speed_limit":  mp["polygon_speed_limit"].numpy(),
            "m_valid_mask":           mp["valid_mask"].numpy(),
            "current_state":          data["current_state"].numpy(),
            "origin":                 data["origin"].numpy(),
            "angle":                  data["angle"].numpy(),
        }

        # Only feed what the ONNX model actually expects
        feeds = {inp.name: pool[inp.name] for inp in sess.get_inputs() if inp.name in pool}
        missing = [inp.name for inp in sess.get_inputs() if inp.name not in pool]
        if missing:
            print(f"  WARNING: unknown ONNX inputs not in pool: {missing}")

        result = sess.run(None, feeds)
        print(f"\nONNXRuntime inference OK. output_trajectory shape: {result[0].shape}")
    except ImportError:
        print("\nonnxruntime not installed — skipping verification.")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"\nONNXRuntime verification FAILED: {e}")


def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    _default_onnx = os.path.join(_inference, "planTF.onnx")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=_default_ckpt)
    parser.add_argument("--out",      default=_default_onnx)
    parser.add_argument("--agents",   type=int, default=33)
    parser.add_argument("--polygons", type=int, default=152)
    parser.add_argument("--opset",    type=int, default=14)
    parser.add_argument("--natten",   choices=["exact", "global"], default="exact",
                        help="NATTEN replacement: exact (default) or legacy global MHA")
    parser.add_argument("--verify",   action="store_true", default=True,
                        help="Verify with onnxruntime after export")
    args = parser.parse_args()

    device = torch.device("cpu")  # ONNX export always on CPU

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device)
    print("Model loaded OK")

    print(f"\nApplying ONNX export patches (NATTEN replacement: {args.natten}) ...")
    apply_onnx_export_patches(model, natten=args.natten)

    data = make_dummy_input(A=args.agents, M=args.polygons, device=device)

    print(f"\nAttempting ONNX export (opset={args.opset}) → {args.out}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with torch.no_grad():
        success = try_export(model, data, args.out, args.opset)

    if success and args.verify:
        verify_onnx(args.out, data)

    if not success:
        print("\n--- ONNX export failed. ---")
        print("NATTEN has been replaced with nn.MultiheadAttention, so the issue")
        print("is likely a different dynamic-shape or custom-op in the model.")
        print("Try: torch.jit.script(model) for TorchScript export instead.")


if __name__ == "__main__":
    main()
