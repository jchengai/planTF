import numba
import numpy as np


def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return points @ rotate_mat


def interpolate_polyline(points: np.ndarray, t: int) -> np.ndarray:
    """copy from av2-api"""

    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: np.ndarray = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    chordlen[tbins - 1] = np.where(
        chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
    )

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp
