"""Parameter-validation and robustness sanity tests."""
from __future__ import annotations

import numpy as np
import pytest

from asian_option_pricer import (
    AsianOptionParams,
    antithetic_cv_price,
    antithetic_mc_price,
    control_variate_price,
    rqmc_sobol_price,
    sobol_qmc_price,
    standard_mc_price,
)
from asian_option_pricer.utils import monitoring_times


def test_invalid_sigma_raises():
    with pytest.raises(ValueError):
        AsianOptionParams(S0=100, K=100, r=0.05, sigma=-0.1, T=1.0, N=52).validate()


def test_invalid_T_raises():
    with pytest.raises(ValueError):
        AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=0.0, N=52).validate()


def test_invalid_strike_raises():
    with pytest.raises(ValueError):
        AsianOptionParams(S0=100, K=0, r=0.05, sigma=0.2, T=1.0, N=52).validate()


def test_deep_itm_price_is_positive():
    """Deep in-the-money call should have a clearly positive price."""
    params = AsianOptionParams(S0=200, K=100, r=0.05, sigma=0.2, T=1.0, N=52)
    price = control_variate_price(params, 16_384, seed=0)["price"]
    assert price > 80.0


def test_deep_otm_price_is_near_zero():
    """Deep OTM prices must be non-negative and small."""
    params = AsianOptionParams(S0=100, K=400, r=0.05, sigma=0.2, T=1.0, N=52)
    for fn in [
        standard_mc_price,
        control_variate_price,
        antithetic_cv_price,
        sobol_qmc_price,
    ]:
        out = fn(params, 8192, seed=0)
        assert out["price"] >= 0.0
        assert out["price"] < 1e-2


@pytest.mark.parametrize(
    "estimator",
    [
        lambda p, n, s: control_variate_price(p, n, seed=s),
        lambda p, n, s: antithetic_cv_price(p, n, seed=s),
        lambda p, n, s: rqmc_sobol_price(p, n, seed=s, n_replications=8),
    ],
)
def test_price_is_monotone_in_strike(estimator):
    """Asian call price must be (weakly) decreasing in the strike."""
    base = dict(S0=100.0, r=0.05, sigma=0.25, T=1.0, N=52)
    K_grid = np.linspace(70, 130, 13)
    prices = []
    for K in K_grid:
        params = AsianOptionParams(K=float(K), **base)
        prices.append(estimator(params, 8192, 0)["price"])
    diffs = np.diff(prices)
    assert diffs.max() < 1e-6


@pytest.mark.parametrize(
    "estimator",
    [
        lambda p, n, s: control_variate_price(p, n, seed=s),
        lambda p, n, s: antithetic_cv_price(p, n, seed=s),
    ],
)
def test_price_is_monotone_in_volatility(estimator):
    """Asian call price must be non-decreasing in volatility."""
    base = dict(S0=100.0, K=100.0, r=0.05, T=1.0, N=52)
    sigma_grid = np.linspace(0.05, 0.70, 14)
    prices = []
    for sigma in sigma_grid:
        params = AsianOptionParams(sigma=float(sigma), **base)
        prices.append(estimator(params, 8192, 0)["price"])
    diffs = np.diff(prices)
    assert diffs.min() > -1e-3


def test_put_call_parity():
    """Arithmetic Asian put-call parity: C - P = e^{-rT}(E[A] - K).

    The right-hand side is exact: E[A] = (1/N) sum_i S0 * exp(r * t_i).
    Verified using antithetic MC for both call and put at 500k paths.
    """
    params = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=52)
    params_put = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=52,
                                   option_type="put")

    call = antithetic_mc_price(params,     n_paths=500_000, seed=0)["price"]
    put  = antithetic_mc_price(params_put, n_paths=500_000, seed=0)["price"]

    t = monitoring_times(params)
    expected_avg = float(np.mean(params.S0 * np.exp(params.r * t)))
    parity_rhs = np.exp(-params.r * params.T) * (expected_avg - params.K)

    assert abs((call - put) - parity_rhs) < 0.01
