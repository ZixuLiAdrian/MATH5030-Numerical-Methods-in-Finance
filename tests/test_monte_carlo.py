"""Tests for the Monte Carlo and QMC estimators."""
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


@pytest.fixture
def params() -> AsianOptionParams:
    return AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, N=12)


ALL_ESTIMATORS = [
    standard_mc_price,
    antithetic_mc_price,
    control_variate_price,
    antithetic_cv_price,
    sobol_qmc_price,
    rqmc_sobol_price,
]


@pytest.mark.parametrize("func", ALL_ESTIMATORS)
def test_estimators_return_expected_keys(params, func):
    out = func(params, n_paths=1024, seed=1)
    assert "price" in out
    assert "std_err" in out
    assert "runtime_s" in out
    assert "n_paths" in out
    assert out["price"] >= 0.0
    assert out["std_err"] >= 0.0


@pytest.mark.parametrize("func", ALL_ESTIMATORS)
def test_estimators_are_deterministic_with_seed(params, func):
    out1 = func(params, n_paths=2048, seed=42)
    out2 = func(params, n_paths=2048, seed=42)
    assert out1["price"] == out2["price"]


def test_variance_reduction_ordering(params):
    """At a moderate budget we expect the SE ordering
    standard_mc >= antithetic >> control_variate ~ antithetic_cv > rqmc_bridge.
    This is a statistical claim, but the margins are large enough that the
    ordering is robust across reasonable seeds at n_paths >= ~20k."""
    n = 32_768
    seed = 12345
    mc = standard_mc_price(params, n, seed=seed)
    anti = antithetic_mc_price(params, n, seed=seed)
    cv = control_variate_price(params, n, seed=seed)
    acv = antithetic_cv_price(params, n, seed=seed)
    rqmc = rqmc_sobol_price(params, n, seed=seed, n_replications=8)

    assert anti["std_err"] < mc["std_err"]
    assert cv["std_err"] < anti["std_err"]
    # The combined CV+antithetic estimator doesn't always beat CV alone on
    # this payoff, but its SE should be within a small factor.
    assert acv["std_err"] < 2.0 * cv["std_err"]
    # RQMC should be clearly better than plain MC and at least competitive
    # with CV. At low dimension (small N) its edge over CV is modest, so we
    # only demand a meaningful win versus unaugmented MC here.
    assert rqmc["std_err"] < 0.1 * mc["std_err"]
    assert rqmc["std_err"] < 2.0 * cv["std_err"]


def test_rqmc_requires_multiple_replications(params):
    with pytest.raises(ValueError):
        rqmc_sobol_price(params, 1024, n_replications=1)


def test_sobol_bridge_beats_incremental_on_asian():
    """Brownian-bridge construction should reduce RMSE versus incremental
    Sobol for a path-dependent payoff."""
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.30, T=1.0, N=50)
    # Repeat with different scramble seeds and compare spread of the estimator
    # (which is what QMC variance actually measures).
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    n = 8192
    inc_prices = [
        sobol_qmc_price(params, n, seed=s, path_method="incremental")["price"]
        for s in seeds
    ]
    bri_prices = [
        sobol_qmc_price(params, n, seed=s, path_method="brownian_bridge")["price"]
        for s in seeds
    ]
    assert np.std(bri_prices, ddof=1) < np.std(inc_prices, ddof=1)


def test_antithetic_rejects_tiny_budget(params):
    with pytest.raises(ValueError):
        antithetic_mc_price(params, n_paths=1)


@pytest.mark.parametrize("func", [standard_mc_price, control_variate_price])
def test_estimators_reject_invalid_n_paths(params, func):
    with pytest.raises(ValueError):
        func(params, n_paths=0)
