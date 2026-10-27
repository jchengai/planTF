"""Mode re-ranking post-processor for PlanTF.

The base model selects the trajectory mode with the highest predicted
probability (``probability.argmax()``).  This module re-scores each candidate
mode by how well it stays within the on-route drivable area and blends that
geometric score with the model's predicted probabilities before committing to
a mode.

Motivation
----------
Failures on ``traversing_traffic_light_intersection`` scenarios are dominated
by lateral path errors:
  * ego trajectory leaves the drivable area while turning
  * ego trajectory enters an opposing-traffic lane
  * ego trajectory clips the kerb / island at a complex intersection

The hypothesis is that in some of these cases the model produces at least
one mode that avoids the failure, but a worse mode has a slightly higher
predicted probability. Re-ranking by route coverage can correct that without
retraining. It cannot help when no mode is correct: the four zero-score TL
intersection scenarios studied in the README stay at 0.0 after re-ranking
because none of the K=6 modes follows the right path.

Enabled with ``planner.imitation_planner.use_mode_reranker=true``; the
weight is ``planner.imitation_planner.reranker_lambda``.

Algorithm
---------
1. Obtain all K=6 candidate trajectories (ego-local frame) and their
   log-probabilities from the model output.
2. Obtain the on-route polygon centres from the feature data.
   These are already in the same ego-local frame (transformed by
   ``NuplanFeature.normalize``).
3. For each mode k, compute::

       coverage[k] = fraction of waypoints whose minimum distance to
                     any on-route polygon centre ≤ ROUTE_DIST_THRESHOLD

4. Blend::

       score[k] = (1 - lambda_) * softmax(prob)[k]  +  lambda_ * coverage[k]

5. Select ``argmax(score)`` as the output trajectory.

If no on-route polygons are present the original ``probability.argmax()``
result is returned unchanged as a safe fallback.
"""

import numpy as np
import torch

# A waypoint is considered "on route" if it is within this distance of any
# on-route polygon centre (proxy for lane centreline).  Typical half-lane
# width is ~1.9 m; 3.5 m gives comfortable leeway for wide turns.
ROUTE_DIST_THRESHOLD: float = 3.5  # metres

# Weight given to geometric coverage in the blended score.
# 0.0 → pure probability (no change from baseline)
# 1.0 → pure coverage (ignore model probability)
# 0.3 is the default: nudges the model away from out-of-route modes while
# still respecting its learned preferences for the most part.
DEFAULT_LAMBDA: float = 0.3


def rerank_modes(
    out: dict,
    data: dict,
    lambda_: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Select the best trajectory mode by blending model probability with
    on-route coverage.

    Parameters
    ----------
    out : dict
        Model output dict (inference mode, batch size 1).
        Required keys:

        * ``"trajectory"``   — ``Tensor[1, K, T, 4]``
          All K candidate trajectories.  Channel order: x, y, cos_h, sin_h.
        * ``"probability"``  — ``Tensor[1, K]``
          Unnormalised mode log-scores.
        * ``"output_trajectory"`` — ``Tensor[1, T, 3]``
          Pre-computed argmax trajectory (used as fallback).

    data : dict
        Batched feature dict (batch size 1).
        Required keys inside ``data["map"]``:

        * ``"polygon_center"``    — ``Tensor[1, M, 3]``  (x, y, heading)
        * ``"polygon_on_route"``  — ``Tensor[1, M]``     (bool / int)

    lambda_ : float
        Coverage weight in [0, 1].

    Returns
    -------
    np.ndarray, shape ``[T, 3]``, dtype float64
        Selected trajectory in ego-local frame: (x, y, heading) per step.
    """
    # ── model outputs (batch index 0) ──────────────────────────────────────
    traj = out["trajectory"][0]       # [K, T, 4]
    prob = out["probability"][0]       # [K]

    # ── on-route polygon centres (batch index 0) ───────────────────────────
    polygon_center   = data["map"]["polygon_center"][0]    # [M, 3]
    polygon_on_route = data["map"]["polygon_on_route"][0]  # [M]

    on_route_mask = polygon_on_route.bool()
    if on_route_mask.sum() == 0:
        # No route information available — return pre-computed baseline.
        return out["output_trajectory"][0].cpu().numpy().astype(np.float64)

    route_centers = polygon_center[on_route_mask, :2]  # [R, 2]

    # ── per-mode coverage scores ───────────────────────────────────────────
    K = traj.shape[0]
    coverage = torch.zeros(K, dtype=torch.float32, device=traj.device)

    for k in range(K):
        waypoints = traj[k, :, :2]                              # [T, 2]
        dists = torch.cdist(waypoints, route_centers)           # [T, R]
        min_dists = dists.min(dim=-1).values                    # [T]
        coverage[k] = (min_dists <= ROUTE_DIST_THRESHOLD).float().mean()

    # ── blended score and mode selection ──────────────────────────────────
    prob_norm = torch.softmax(prob, dim=-1)                     # [K]
    score = (1.0 - lambda_) * prob_norm + lambda_ * coverage   # [K]

    best_mode = score.argmax()
    selected  = traj[best_mode]                                 # [T, 4]
    angle     = torch.atan2(selected[..., 3], selected[..., 2])
    result    = torch.cat([selected[..., :2], angle.unsqueeze(-1)], dim=-1)

    return result.cpu().numpy().astype(np.float64)
