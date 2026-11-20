# Compute the numerical inputs shared by training and evaluation. Rows follow time order and
# paired hand measurements use left then right order.

"""NumPy input features for the Bi-CAST authority predictor."""

import numpy as np


def compute_fsr_features(force: np.ndarray, var_window: int = 5) -> np.ndarray:
    """Stack force level, one-sample difference and trailing population variance
    for each hand. The first difference is zero. A single force series returns
    (T, 3); bilateral input returns (T, 2, 3).
    """

    force = np.asarray(force, dtype=np.float32)
    single = force.ndim == 1
    if single:
        force = force[:, None]
    T, A = force.shape

    dF = np.zeros_like(force)
    dF[1:] = force[1:] - force[:-1]

    sigma = np.zeros_like(force)
    for t in range(T):
        lo = max(0, t - var_window + 1)
        sigma[t] = force[lo : t + 1].var(axis=0)

    feats = np.stack([force, dF, sigma], axis=-1)
    return feats[:, 0, :] if single else feats


def recording_features(force, iou, wall_distance, frame_range, var_window=5, legacy=False):
    """Build the selected segment; legacy mode reproduces full-recording preprocessing."""
    lo, hi = frame_range
    if legacy:
        scale = max(float(np.max(wall_distance)), 1e-6)
        fsr = compute_fsr_features(force, var_window)[lo:hi]
        distance = wall_distance[lo:hi] / scale
    else:
        fsr = compute_fsr_features(force[lo:hi], var_window)
        distance = wall_distance[lo:hi]
    return fsr, np.stack([iou[lo:hi], distance], axis=1)
