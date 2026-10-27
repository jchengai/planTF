"""
tl_safety.py
------------
Rule-based post-processor that overrides the model's planned trajectory with a
smooth constant-deceleration stop when a RED on-route polygon lies ahead.

No retraining required — this operates entirely on the already-computed
trajectory and the map feature tensors that the model built during inference.

Design
------
1. Look at polygon_tl_status / polygon_on_route / valid_mask in the feature dict.
2. Find the nearest RED on-route polygon whose centre is ahead of ego (x > -1 m)
   and within _AHEAD_MAX_M.
3. Compute a stop target = polygon_centre_x - _STOP_MARGIN_M.
4. Re-parametrise the trajectory via arc length and generate a new kinematic
   (constant decel) arc schedule that brings the vehicle to rest at arc_stop.
5. Re-sample x/y/heading from the original trajectory at the new arc positions.

If no relevant red light is found the trajectory is returned unchanged.
"""

import numpy as np

# ------------------------------------------------------------------
# Constants (tune these to adjust behaviour)
# ------------------------------------------------------------------
_RED_STATUS  = 2      # TrafficLightStatusType.RED (IntEnum value)
_DECEL_MAX   = 3.0    # m/s²  — upper bound on deceleration
_DECEL_MIN   = 0.5    # m/s²  — lower bound (prevents very lengthy creep-stops)
_STOP_MARGIN = 3.0    # m     — stop this far before the red polygon centre
_AHEAD_MIN   = -1.0   # m     — ignore polygons behind ego beyond this
_AHEAD_MAX   = 60.0   # m     — ignore polygons further than this


def apply_tl_brake(
    trajectory: np.ndarray,
    data: dict,
    current_speed: float,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Post-process a local-frame trajectory to stop before the nearest RED
    on-route polygon.

    Parameters
    ----------
    trajectory : np.ndarray, shape [T, 3]
        (x, y, heading) waypoints in ego-local frame, 0.1 s intervals.
    data : dict
        Feature dict produced by NuplanFeatureBuilder, on any device.
        Must contain map/polygon_center, map/polygon_tl_status,
        map/polygon_on_route, map/valid_mask — all with batch dim 0.
    current_speed : float
        Ego speed in m/s at the moment of planning.
    dt : float
        Time step between trajectory waypoints (seconds).

    Returns
    -------
    np.ndarray, shape [T, 3]
        Modified (or unchanged) trajectory.
    """
    polygon_center = data["map"]["polygon_center"][0].cpu().numpy()    # [M, 3]
    polygon_tl     = data["map"]["polygon_tl_status"][0].cpu().numpy() # [M] int
    polygon_route  = data["map"]["polygon_on_route"][0].cpu().numpy()  # [M] bool
    valid          = data["map"]["valid_mask"][0].any(-1).cpu().numpy() # [M] bool

    # ---- Find nearest RED, on-route, valid polygon ahead of ego ----
    red_mask = (polygon_tl == _RED_STATUS) & polygon_route & valid
    if not red_mask.any():
        return trajectory

    cx = polygon_center[red_mask, 0]   # x-coord in ego-local frame
    ahead = (cx > _AHEAD_MIN) & (cx < _AHEAD_MAX)
    if not ahead.any():
        return trajectory

    stop_target_x = float(cx[ahead].min()) - _STOP_MARGIN
    if stop_target_x <= 0.0:
        return trajectory  # stop would be behind or at ego — cannot help

    # ---- Map stop_target_x to an arc-length distance along trajectory ----
    # Find the first waypoint where x crosses stop_target_x (forward driving).
    cross = np.where(trajectory[:, 0] >= stop_target_x)[0]
    if len(cross) == 0:
        return trajectory  # trajectory never reaches the stop line — no action

    # Arc length cumulation
    diffs  = np.diff(trajectory[:, :2], axis=0)           # [T-1, 2]
    segs   = np.linalg.norm(diffs, axis=-1)               # [T-1]
    arc    = np.concatenate([[0.0], np.cumsum(segs)])      # [T]

    arc_stop = float(arc[cross[0]])
    if arc_stop <= 0.0:
        return trajectory

    # ---- Kinematics: constant decel that stops at arc_stop ----
    v = max(float(current_speed), 0.0)

    if v < 0.05:
        # Already barely moving — just freeze at the last valid point before stop
        new_traj = trajectory.copy()
        stop_idx = max(cross[0], 1)
        new_traj[stop_idx:] = new_traj[stop_idx - 1]
        return new_traj

    # a such that  v² = 2 * a * arc_stop
    required_decel = v ** 2 / (2.0 * arc_stop)
    decel = float(np.clip(required_decel, _DECEL_MIN, _DECEL_MAX))

    # Build new arc-position schedule via forward Euler
    new_arc = np.zeros(len(trajectory))
    s, vi = 0.0, v
    for i in range(len(trajectory)):
        new_arc[i] = s
        if vi <= 0.0 or s >= arc_stop:
            new_arc[i:] = arc_stop
            break
        ds = vi * dt - 0.5 * decel * dt ** 2
        s  = min(s + max(ds, 0.0), arc_stop)
        vi = max(vi - decel * dt, 0.0)

    # ---- Re-sample x / y / heading at new arc positions ----
    new_traj = np.empty_like(trajectory)
    for col in range(3):
        new_traj[:, col] = np.interp(new_arc, arc, trajectory[:, col])

    return new_traj
