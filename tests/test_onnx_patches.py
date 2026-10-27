"""Numerical parity of the ONNX-export patches against the original model.

Needs the real NATTEN package (0.14.x) and PyTorch 1.12; skipped otherwise.
nuplan-devkit is not needed: tests/nuplan_stub.py stands in for it.
The ONNX round-trip test also needs onnxruntime.

Run with:  pytest tests/test_onnx_patches.py -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch

natten = pytest.importorskip("natten")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tests"), os.path.join(_REPO_ROOT, "inference", "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nuplan_stub  # noqa: E402

nuplan_stub.install()

from natten import NeighborhoodAttention1D  # noqa: E402

from inference.common import patches  # noqa: E402
from inference.common.run_inference import make_dummy_input  # noqa: E402
from src.models.planTF.layers.embedding import NATLayer  # noqa: E402
from src.models.planTF.planning_model import PlanningModel  # noqa: E402

N_SCENES = 3


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """PlanTF with the released architecture and random weights.

    The relative positional bias is initialised near zero by NATTEN; scale it
    up so that dropping or mis-indexing it shows up in the outputs.
    """
    torch.manual_seed(0)
    m = PlanningModel(
        dim=128, history_steps=21, future_steps=80, encoder_depth=4,
        num_heads=8, num_modes=6,
    ).eval()
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module, NATLayer):
                module.attn.rpb.normal_(0.0, 1.0)
    return m


@pytest.fixture(scope="module")
def scenes():
    """Padded inputs, as in real scenes (fewer agents/polygons than the max)."""
    torch.manual_seed(1)
    return [make_dummy_input(valid_agents=20, valid_polygons=100) for _ in range(N_SCENES)]


def _run(m, data):
    with torch.no_grad():
        return m(data)["output_trajectory"]


def _max_pos_err(a, b) -> float:
    return float((a[..., :2] - b[..., :2]).norm(dim=-1).max())


# ── module level ─────────────────────────────────────────────────────────────

# (L, kernel_size, dim, heads): the three NATSequenceEncoder levels for 21
# history steps (21 -> 11 -> 6), plus sequences shorter than the window.
@pytest.mark.parametrize(
    "length,kernel_size,dim,heads",
    [(21, 3, 32, 2), (11, 3, 64, 4), (6, 5, 128, 8), (5, 5, 32, 2), (3, 5, 32, 2), (2, 3, 16, 2)],
)
def test_exact_na1d_matches_natten(length, kernel_size, dim, heads):
    torch.manual_seed(length)
    ref = NeighborhoodAttention1D(dim, num_heads=heads, kernel_size=kernel_size).eval()
    with torch.no_grad():
        ref.rpb.normal_(0.0, 1.0)
        x = torch.randn(4, length, dim)
        expected = ref(x)
        actual = patches._ExactNA1D(ref)(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ── full model ───────────────────────────────────────────────────────────────

def test_exact_natten_patch_matches_original(model, scenes):
    patched = copy.deepcopy(model)
    patches.patch_natten_exact(patched)
    for data in scenes:
        assert _max_pos_err(_run(patched, data), _run(model, data)) < 1e-4


def test_global_natten_patch_is_only_approximate(model, scenes):
    """Guards the test itself: the legacy approximation must be detectable."""
    patched = copy.deepcopy(model)
    patches.patch_natten_for_onnx(patched)
    errs = [_max_pos_err(_run(patched, d), _run(model, d)) for d in scenes]
    assert max(errs) > 1e-3


def test_export_patches_respect_padding(model, scenes):
    """All export patches together must still ignore padded agents / polygons."""
    patched = copy.deepcopy(model)
    patches.apply_onnx_export_patches(patched, natten="exact")
    for data in scenes:
        assert _max_pos_err(_run(patched, data), _run(model, data)) < 1e-4


def test_padding_values_do_not_leak(model, scenes):
    """Changing features of padded entries must not change the plan."""
    patched = copy.deepcopy(model)
    patches.apply_onnx_export_patches(patched, natten="exact")
    data = scenes[0]
    noisy = copy.deepcopy(data)
    noisy["agent"]["position"][:, 20:] += 50.0
    noisy["map"]["point_position"][:, 100:] += 50.0
    noisy["map"]["polygon_center"][:, 100:] += 50.0
    assert _max_pos_err(_run(patched, noisy), _run(patched, data)) < 1e-4


def test_onnx_export_matches_original(model, scenes, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from export_onnx import try_export

    patched = copy.deepcopy(model)
    patches.apply_onnx_export_patches(patched, natten="exact")
    out = str(tmp_path / "planTF_exact.onnx")
    with torch.no_grad():
        assert try_export(patched, scenes[0], out, 14)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    for data in scenes:
        feeds = {}
        for inp in sess.get_inputs():
            name = inp.name
            if name.startswith("a_"):
                feeds[name] = data["agent"][name[2:]].numpy()
            elif name.startswith("m_"):
                feeds[name] = data["map"][name[2:]].numpy()
            else:
                feeds[name] = data[name].numpy()
        onnx_traj = sess.run(None, feeds)[0]
        ref = _run(model, data).numpy()
        err = np.linalg.norm(onnx_traj[..., :2] - ref[..., :2], axis=-1).max()
        assert err < 1e-4
