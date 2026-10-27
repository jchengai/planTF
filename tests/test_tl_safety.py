"""Unit tests for src/planners/tl_safety.py.

No nuPlan dependency — only torch and numpy are required.
Run with:  pytest tests/test_tl_safety.py -v
"""

import sys
import os

import numpy as np
import pytest
import torch

# Ensure repo root is on the path regardless of where pytest is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.planners.tl_safety import apply_tl_brake

# ── constants ────────────────────────────────────────────────────────────────
T = 80      # future steps
DT = 0.1    # seconds per step
P = 20      # points per polygon
RED = 2
GREEN = 0
STOP_MARGIN = 3.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _straight_trajectory(speed: float = 10.0) -> np.ndarray:
    """Constant-speed straight trajectory along +x, ego-local frame."""
    t = np.arange(1, T + 1) * DT
    traj = np.zeros((T, 3))
    traj[:, 0] = speed * t
    return traj


def _make_data(polygons) -> dict:
    """Build a minimal feature-data dict (batch=1).

    polygons: list of (x, tl_status, on_route, valid) tuples.
    """
    m = len(polygons)
    center = torch.zeros(1, m, 3)
    tl = torch.zeros(1, m, dtype=torch.int8)
    route = torch.zeros(1, m, dtype=torch.bool)
    valid = torch.zeros(1, m, P, dtype=torch.bool)
    for i, (x, status, on_route, is_valid) in enumerate(polygons):
        center[0, i, 0] = x
        tl[0, i] = status
        route[0, i] = on_route
        valid[0, i] = is_valid
    return {
        "map": {
            "polygon_center": center,
            "polygon_tl_status": tl,
            "polygon_on_route": route,
            "valid_mask": valid,
        }
    }


# ── no-op cases ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "polygons",
    [
        [(30.0, GREEN, True, True)],    # green light
        [(30.0, RED, False, True)],     # red but off-route
        [(30.0, RED, True, False)],     # red but invalid polygon
        [(-10.0, RED, True, True)],     # red behind ego
        [(80.0, RED, True, True)],      # red beyond look-ahead range
        [(2.0, RED, True, True)],       # stop target would be behind ego
    ],
    ids=["green", "off_route", "invalid", "behind", "too_far", "too_close"],
)
def test_unchanged_when_no_relevant_red(polygons):
    traj = _straight_trajectory()
    out = apply_tl_brake(traj, _make_data(polygons), current_speed=10.0, dt=DT)
    np.testing.assert_array_equal(out, traj)


def test_unchanged_when_trajectory_never_reaches_stop_line():
    traj = _straight_trajectory(speed=2.0)  # reaches only 16 m in 8 s
    data = _make_data([(40.0, RED, True, True)])
    out = apply_tl_brake(traj, data, current_speed=2.0, dt=DT)
    np.testing.assert_array_equal(out, traj)


# ── braking behaviour ────────────────────────────────────────────────────────

def test_stops_before_red_polygon():
    red_x = 30.0
    traj = _straight_trajectory()
    out = apply_tl_brake(
        traj, _make_data([(red_x, RED, True, True)]), current_speed=10.0, dt=DT
    )

    assert out.shape == traj.shape
    # Never passes the stop target (one waypoint of slack for discretisation).
    assert out[:, 0].max() <= red_x - STOP_MARGIN + 10.0 * DT
    # Comes to rest: the tail of the trajectory is stationary.
    assert np.allclose(out[-10:, 0], out[-1, 0])
    # Progress along the path is monotonic (no reversing).
    assert np.all(np.diff(out[:, 0]) >= -1e-9)


def test_nearest_red_polygon_wins():
    traj = _straight_trajectory()
    data = _make_data([(50.0, RED, True, True), (25.0, RED, True, True)])
    out = apply_tl_brake(traj, data, current_speed=10.0, dt=DT)
    assert out[:, 0].max() <= 25.0 - STOP_MARGIN + 10.0 * DT


def test_path_geometry_preserved_on_curve():
    """Braking only re-times the path; waypoints stay on the original curve."""
    t = np.arange(1, T + 1) * DT
    traj = np.stack([10.0 * t, 0.02 * (10.0 * t) ** 2, np.zeros(T)], axis=-1)
    out = apply_tl_brake(
        traj, _make_data([(30.0, RED, True, True)]), current_speed=10.0, dt=DT
    )
    # y(x) relation of the original parabola must still hold.
    np.testing.assert_allclose(out[:, 1], 0.02 * out[:, 0] ** 2, atol=0.05)


def test_stationary_ego_is_held_before_stop_line():
    red_x = 30.0
    traj = _straight_trajectory()
    out = apply_tl_brake(
        traj, _make_data([(red_x, RED, True, True)]), current_speed=0.0, dt=DT
    )
    assert out[:, 0].max() < red_x - STOP_MARGIN
