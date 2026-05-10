"""
Closed-form / semi-analytic prices for Asian calls under GBM.

Supports arbitrary averaging windows [T1, T2].
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .models import AsianOptionParams
from .utils import monitoring_times


def _discrete_geometric_log_moments(
    params: AsianOptionParams,
) -> tuple[float, float]:
    """
    Return mean and variance of log geometric average.

    For arbitrary monitoring times t_1, ..., t_N:

        G = exp( (1/N) sum log S(t_i) )

    log(G) is normal with:

        mean
            = log(S0)
              + (r - 0.5 sigma^2)
                * average(t_i)

        variance
            = sigma^2 / N^2
              * sum_i sum_j min(t_i, t_j)
    """

    params.validate()

    t = monitoring_times(params)

    t_i = t[:, None]
    t_j = t[None, :]

    mean_log_g = (
        np.log(params.S0)
        + (
            params.r
            - 0.5 * params.sigma ** 2
        )
        * float(np.mean(t))
    )

    var_log_g = float(
        params.sigma ** 2
        * np.mean(
            np.minimum(t_i, t_j)
        )
    )

    return (
        mean_log_g,
        var_log_g,
    )


def geometric_asian_call_price(
    params: AsianOptionParams,
) -> float:
    """
    Exact Kemna-Vorst price for discretely monitored
    geometric Asian call under arbitrary monitoring dates.
    """

    params.validate()

    if params.option_type != "call":
        raise NotImplementedError(
            "Closed-form geometric benchmark "
            "currently implemented for call only."
        )

    mean_log_g, var_log_g = (
        _discrete_geometric_log_moments(
            params
        )
    )

    df = np.exp(
        -params.r * params.T
    )

    # E[G]
    forward_g = np.exp(
        mean_log_g
        + 0.5 * var_log_g
    )

    # deterministic limit
    if var_log_g == 0.0:

        return float(
            df
            * max(
                forward_g - params.K,
                0.0,
            )
        )

    vol_g = np.sqrt(
        var_log_g
    )

    d1 = (
        mean_log_g
        + var_log_g
        - np.log(params.K)
    ) / vol_g

    d2 = d1 - vol_g

    return float(
        df
        * (
            forward_g * norm.cdf(d1)
            - params.K * norm.cdf(d2)
        )
    )


def levy_approx_call_price(
    params: AsianOptionParams,
) -> float:
    """
    Levy (1992) lognormal moment-matching approximation
    for arithmetic Asian call.

    Supports arbitrary monitoring windows [T1, T2].
    """

    params.validate()

    if params.option_type != "call":
        raise NotImplementedError(
            "Levy approximation currently "
            "implemented for call only."
        )

    t = monitoring_times(params)

    S0 = params.S0
    r = params.r
    sigma = params.sigma
    T = params.T
    K = params.K

    # First moment of arithmetic average
    M1 = float(
        np.mean(
            S0 * np.exp(r * t)
        )
    )

    # Second moment
    t_i = t[:, None]
    t_j = t[None, :]

    M2 = float(
        np.mean(
            S0 ** 2
            * np.exp(
                r * (t_i + t_j)
                + sigma ** 2
                * np.minimum(
                    t_i,
                    t_j,
                )
            )
        )
    )

    df = np.exp(
        -r * T
    )

    # lognormal variance
    v = np.log(
        M2 / (M1 ** 2)
    )

    # deterministic limit
    if v <= 0.0:

        return float(
            df
            * max(
                M1 - K,
                0.0,
            )
        )

    sqrt_v = np.sqrt(v)

    d1 = (
        np.log(M1 / K)
        + v
    ) / sqrt_v

    d2 = d1 - sqrt_v

    return float(
        df
        * (
            M1 * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
    )
