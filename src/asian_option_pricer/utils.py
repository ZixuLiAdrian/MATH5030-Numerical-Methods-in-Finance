"""Utility helpers for Asian option pricing."""

from __future__ import annotations

import numpy as np

from .models import AsianOptionParams


def monitoring_times(
    params: AsianOptionParams,
) -> np.ndarray:
    """
    Return equally spaced monitoring dates.

    Default behavior:
        t_i = iT/N

    Generalized delayed averaging:
        t_i = T1 + i(T2 - T1)/N
    """

    params.validate()

    T2 = params.averaging_end

    dt = (
        T2 - params.T1
    ) / params.N

    return (
        params.T1
        + np.arange(
            1,
            params.N + 1,
        ) * dt
    )


def discount_factor(
    params: AsianOptionParams,
) -> float:
    """
    Risk-neutral discount factor exp(-rT).
    """

    params.validate()

    return float(
        np.exp(
            -params.r * params.T
        )
    )
