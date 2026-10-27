"""
inspect_model.py
----------------
Loads planTF.ckpt and prints the full model architecture + parameter count.
No nuPlan dependency required.

Usage:
    python inference/inspect_model.py
    python inference/inspect_model.py --ckpt path/to/other.ckpt
"""

import argparse
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)

import torch
from src.models.planTF.planning_model import PlanningModel


def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=_default_ckpt,
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # checkpoint may be wrapped in a dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # strip leading "model." prefix if present (pytorch-lightning style)
        state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
    else:
        state_dict = ckpt

    model = PlanningModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("\n=== Architecture ===")
    print(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Parameters ===")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")

    print("\n=== State dict keys (first 20) ===")
    for i, k in enumerate(state_dict.keys()):
        print(f"  {k}")
        if i >= 19:
            print(f"  ... ({len(state_dict)} keys total)")
            break


if __name__ == "__main__":
    main()
