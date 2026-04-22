"""Closed-form / semi-analytic prices for Asian calls under GBM.

Two prices are provided:

* :func:`geometric_asian_call_price` -- exact price for the discretely
  monitored geometric-average Asian call. This is the Kemna-Vorst (1990)
  identity applied to the discrete case: the geometric average of lognormals
  is again lognormal, so a Black-Scholes-style formula holds exactly with a
  reduced volatility and a drift correction.

* :func:`levy_approx_call_price` -- Levy (1992) moment-matching approximation
  for the arithmetic-average Asian call. The arithmetic average of lognormals
  is not lognormal, but matching its first two moments to a lognormal gives a
  fast and usually very accurate approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .models import AsianOptionParams
from .utils import monitoring_times


def _discrete_geometric_moments(params: AsianOptionParams) -> tuple[float, float]:
    """Return ``(sigma_g_sq_T, b_g_T)`` for the discretely-monitored
    geometric average.

    For equally-spaced monitoring times ``t_i = i * T / N`` (``i = 1..N``):

    * ``log G`` is normal with variance
      ``sigma^2 * T * (N+1)(2N+1) / (6 N^2)`` and mean
      ``log S0 + (r - sigma^2/2) * T * (N+1) / (2N)``.
    * Writing ``E[G] = S0 * exp(b_g * T)`` gives
      ``b_g = (r - sigma^2/2) * (N+1)/(2N) + sigma_g^2 / 2``.
    """
    N = params.N
    sigma_sq_T_g = params.sigma ** 2 * params.T * (N + 1) * (2 * N + 1) / (6 * N ** 2)
    b_g_T = (
        (params.r - 0.5 * params.sigma ** 2) * params.T * (N + 1) / (2 * N)
        + 0.5 * sigma_sq_T_g
    )
    return float(sigma_sq_T_g), float(b_g_T)


def geometric_asian_call_price(params: AsianOptionParams) -> float:
    """Kemna-Vorst exact price for the discretely-monitored geometric Asian
    call.

    This is both a stand-alone benchmark and the control-mean used by
    :mod:`asian_option_pricer.control_variate`.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "Closed-form geometric benchmark currently implemented for call only."
        )

    sigma_sq_T_g, b_g_T = _discrete_geometric_moments(params)
    df = np.exp(-params.r * params.T)
    forward_g = params.S0 * np.exp(b_g_T)

    if sigma_sq_T_g == 0.0:
        return float(df * max(forward_g - params.K, 0.0))

    sigma_g_sqrt_T = np.sqrt(sigma_sq_T_g)
    d1 = (np.log(params.S0 / params.K) + b_g_T + 0.5 * sigma_sq_T_g) / sigma_g_sqrt_T
    d2 = d1 - sigma_g_sqrt_T
    return float(df * (forward_g * norm.cdf(d1) - params.K * norm.cdf(d2)))


def levy_approx_call_price(params: AsianOptionParams) -> float:
    """Levy (1992) lognormal moment-matching approximation for the arithmetic
    Asian call.

    ``M1`` and ``M2`` are the exact first two moments of the arithmetic
    average ``A = (1/N) sum S(t_i)`` under GBM; we then price as if ``A``
    were lognormal with those moments.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "Levy approximation currently implemented for call only."
        )

    t = monitoring_times(params)
    S0, r, sigma, T, K = params.S0, params.r, params.sigma, params.T, params.K

    # First moment of A: (1/N) sum_i S0 * exp(r t_i).
    M1 = float(np.mean(S0 * np.exp(r * t)))

    # Second moment of A uses E[S(t_i) S(t_j)] = S0^2 * exp(r(t_i+t_j) + sigma^2 min(t_i,t_j)).
    t_i = t[:, None]
    t_j = t[None, :]
    M2 = float(
        np.mean(S0 ** 2 * np.exp(r * (t_i + t_j) + sigma ** 2 * np.minimum(t_i, t_j)))
    )

    sigma_eff_sq = np.log(M2 / (M1 ** 2)) / T
    sigma_eff = float(np.sqrt(max(sigma_eff_sq, 0.0)))
    df = np.exp(-r * T)

    if sigma_eff == 0.0:
        return float(df * max(M1 - K, 0.0))

    d1 = (np.log(M1 / K) + 0.5 * sigma_eff_sq * T) / (sigma_eff * np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)
    return float(df * (M1 * norm.cdf(d1) - K * norm.cdf(d2)))
