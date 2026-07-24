from __future__ import annotations

import numpy as np


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


def layer_trajectory_metrics(hidden_states: np.ndarray) -> dict[str, np.ndarray]:
    """Compute descriptive layer-trajectory metrics for each token.

    Parameters
    ----------
    hidden_states:
        Array shaped ``[tokens, layers, hidden_dim]``. The extraction point
        (pre/post normalization, model revision, prefill/decode) must be
        documented in the accompanying manifest.
    """
    h = np.asarray(hidden_states, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError("hidden_states must have shape [tokens, layers, hidden_dim]")
    if h.shape[1] < 3:
        raise ValueError("at least three layers are required")
    if not np.isfinite(h).all():
        raise ValueError("hidden_states contains NaN or infinity")

    velocity = np.diff(h, axis=1)
    speed = _safe_norm(velocity)
    acceleration = np.diff(velocity, axis=1)
    acceleration_norm = _safe_norm(acceleration)

    v0 = velocity[:, :-1]
    v1 = velocity[:, 1:]
    denom = _safe_norm(v0) * _safe_norm(v1)
    cosine = np.divide(
        np.sum(v0 * v1, axis=-1),
        denom,
        out=np.ones_like(denom),
        where=denom > 0,
    )
    turning_angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    path_length = speed.sum(axis=1)
    displacement = _safe_norm(h[:, -1] - h[:, 0])
    tortuosity = np.divide(
        path_length,
        displacement,
        out=np.full_like(path_length, np.nan),
        where=displacement > 0,
    )

    return {
        "path_length": path_length,
        "displacement": displacement,
        "tortuosity": tortuosity,
        "mean_speed": speed.mean(axis=1),
        "speed_cv": np.divide(
            speed.std(axis=1),
            speed.mean(axis=1),
            out=np.zeros(h.shape[0]),
            where=speed.mean(axis=1) > 0,
        ),
        "mean_acceleration": acceleration_norm.mean(axis=1),
        "mean_turning_angle_rad": turning_angle.mean(axis=1),
    }


def probability_baselines(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Compute entropy and top-1/top-2 margin baselines per token."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    if z.shape[1] < 2:
        raise ValueError("logits requires at least two vocabulary entries")
    if not np.isfinite(z).all():
        raise ValueError("logits contains NaN or infinity")

    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    p = exp / exp.sum(axis=1, keepdims=True)
    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return {
        "entropy_nats": entropy,
        "top1_probability": top2[:, 1],
        "top1_top2_margin": top2[:, 1] - top2[:, 0],
    }
