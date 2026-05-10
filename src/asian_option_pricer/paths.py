"""
Path construction utilities for GBM under arbitrary monitoring windows.

Two constructions from N iid standard normals to a discrete Brownian path are
supported:

* ``incremental`` (a.k.a. forward / random-walk construction)

* ``brownian_bridge``

This version supports averaging windows [T1, T2] instead of only [0, T].
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from .models import AsianOptionParams
from .utils import monitoring_times


PathMethod = str  # "incremental" | "brownian_bridge"


@lru_cache(maxsize=32)
def brownian_bridge_matrix_from_times(
    times_tuple: tuple[float, ...]
) -> np.ndarray:
    """
    Return Brownian bridge matrix for arbitrary monitoring times.

    If z ~ N(0, I), then

        W = B @ z

    produces Brownian motion evaluated at the specified times.
    """

    times = np.asarray(
        times_tuple,
        dtype=float,
    )

    N = len(times)

    if N <= 0:
        raise ValueError(
            "times must be non-empty."
        )

    if np.any(times <= 0):
        raise ValueError(
            "monitoring times must be positive."
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "monitoring times must be strictly increasing."
        )

    # prepend t=0
    t_ext = np.concatenate(
        ([0.0], times)
    )

    B = np.zeros((N, N))

    # terminal point
    B[N - 1, 0] = np.sqrt(
        times[-1]
    )

    queue: deque[tuple[int, int]] = deque()

    queue.append((0, N))

    z_idx = 1

    while queue and z_idx < N:

        l, r = queue.popleft()

        m = (l + r) // 2

        if m == l or m == r:
            continue

        t_l = t_ext[l]
        t_m = t_ext[m]
        t_r = t_ext[r]

        coef_l = (
            (t_r - t_m)
            / (t_r - t_l)
        )

        coef_r = (
            (t_m - t_l)
            / (t_r - t_l)
        )

        std = np.sqrt(
            (t_m - t_l)
            * (t_r - t_m)
            / (t_r - t_l)
        )

        row = np.zeros(N)

        if l > 0:
            row += coef_l * B[l - 1, :]

        if r > 0:
            row += coef_r * B[r - 1, :]

        row[z_idx] = std

        B[m - 1, :] = row

        z_idx += 1

        queue.append((l, m))
        queue.append((m, r))

    return B


@lru_cache(maxsize=32)
def brownian_bridge_matrix(
    N: int,
    T: float,
) -> np.ndarray:
    """
    Backward-compatible equally spaced bridge matrix.

    Equivalent to monitoring times:

        T/N, 2T/N, ..., T
    """

    if N <= 0:
        raise ValueError(
            "N must be positive."
        )

    if T <= 0:
        raise ValueError(
            "T must be positive."
        )

    times = tuple(
        (
            np.arange(1, N + 1)
            * (T / N)
        ).tolist()
    )

    return brownian_bridge_matrix_from_times(
        times
    )


def build_paths(
    params: AsianOptionParams,
    z: np.ndarray,
    method: PathMethod = "incremental",
) -> np.ndarray:
    """
    Turn (n_paths, N) standard normals into GBM price paths.

    Supports arbitrary monitoring windows [T1, T2].
    """

    params.validate()

    if (
        z.ndim != 2
        or z.shape[1] != params.N
    ):
        raise ValueError(
            f"z must have shape "
            f"(n_paths, {params.N}); "
            f"got {z.shape}."
        )

    # monitoring dates
    t = monitoring_times(params)

    drift_term = (
        params.r
        - 0.5 * params.sigma ** 2
    ) * t

    if method == "incremental":

        # Brownian increments must match
        # actual monitoring schedule
        dt = np.diff(
            np.concatenate(([0.0], t))
        )

        W = np.cumsum(
            np.sqrt(dt) * z,
            axis=1,
        )

    elif method == "brownian_bridge":

        B = brownian_bridge_matrix_from_times(
            tuple(float(x) for x in t)
        )

        # each row corresponds to one path
        W = z @ B.T

    else:
        raise ValueError(
            f"Unknown path construction "
            f"'{method}'. "
            "Use 'incremental' or "
            "'brownian_bridge'."
        )

    log_paths = (
        np.log(params.S0)
        + drift_term[None, :]
        + params.sigma * W
    )

    return np.exp(log_paths)


def payoff_from_paths(
    paths: np.ndarray,
    K: float,
    option_type: str,
) -> np.ndarray:
    """
    Arithmetic-average Asian payoff.
    """

    if paths.ndim != 2:
        raise ValueError(
            "paths must be 2-dimensional."
        )

    avg = paths.mean(axis=1)

    if option_type == "call":
        return np.maximum(
            avg - K,
            0.0,
        )

    if option_type == "put":
        return np.maximum(
            K - avg,
            0.0,
        )

    raise ValueError(
        f"Unknown option_type "
        f"'{option_type}'."
    )


def geometric_payoff_from_paths(
    paths: np.ndarray,
    K: float,
    option_type: str,
) -> np.ndarray:
    """
    Geometric-average Asian payoff.
    """

    if paths.ndim != 2:
        raise ValueError(
            "paths must be 2-dimensional."
        )

    geom = np.exp(
        np.log(paths).mean(axis=1)
    )

    if option_type == "call":
        return np.maximum(
            geom - K,
            0.0,
        )

    if option_type == "put":
        return np.maximum(
            K - geom,
            0.0,
        )

    raise ValueError(
        f"Unknown option_type "
        f"'{option_type}'."
    )
