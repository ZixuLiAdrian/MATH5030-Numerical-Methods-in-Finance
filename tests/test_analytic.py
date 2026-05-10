"""Tests for the closed-form / semi-analytic prices."""

from __future__ import annotations

import numpy as np
import pytest

from asian_option_pricer import (
    AsianOptionParams,
    build_paths,
    geometric_asian_call_price,
    geometric_payoff_from_paths,
    levy_approx_call_price,
)

from asian_option_pricer.utils import (
    discount_factor,
    monitoring_times,
)


def test_prices_are_non_negative():

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=52,
    )

    assert geometric_asian_call_price(params) >= 0.0
    assert levy_approx_call_price(params) >= 0.0


def test_geometric_closed_form_matches_mc():
    """
    Closed-form geometric Asian price should match MC.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=52,
    )

    closed = geometric_asian_call_price(params)

    rng = np.random.default_rng(2024)

    n_pairs = 400_000

    z = rng.standard_normal(
        (n_pairs, params.N)
    )

    df = discount_factor(params)

    pv_pos = (
        df
        * geometric_payoff_from_paths(
            build_paths(
                params,
                z,
                method="incremental",
            ),
            params.K,
            params.option_type,
        )
    )

    pv_neg = (
        df
        * geometric_payoff_from_paths(
            build_paths(
                params,
                -z,
                method="incremental",
            ),
            params.K,
            params.option_type,
        )
    )

    pair = 0.5 * (pv_pos + pv_neg)

    mc_price = pair.mean()

    mc_se = (
        pair.std(ddof=1)
        / np.sqrt(n_pairs)
    )

    assert abs(closed - mc_price) < 3.0 * mc_se, (
        f"closed={closed}, mc={mc_price} +/- {mc_se}"
    )


def test_geometric_closed_form_matches_mc_delayed_averaging():
    """
    Closed-form geometric Asian should also match MC
    under delayed averaging windows [T1, T2].
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.30,
        T=1.0,
        N=50,
        T1=0.25,
        T2=1.0,
    )

    rng = np.random.default_rng(123)

    n_paths = 300_000

    z = rng.standard_normal(
        (n_paths, params.N)
    )

    paths = build_paths(
        params,
        z,
        method="incremental",
    )

    pv = (
        discount_factor(params)
        * geometric_payoff_from_paths(
            paths,
            params.K,
            params.option_type,
        )
    )

    mc_price = pv.mean()

    mc_se = (
        pv.std(ddof=1)
        / np.sqrt(n_paths)
    )

    closed = geometric_asian_call_price(
        params
    )

    assert abs(closed - mc_price) < 4.0 * mc_se


def test_geometric_single_monitor_reduces_to_black_scholes():
    """
    With N=1 the geometric average reduces to S(T),
    so the result should equal Black-Scholes.
    """

    from scipy.stats import norm

    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    params = AsianOptionParams(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        N=1,
    )

    geom = geometric_asian_call_price(params)

    d1 = (
        np.log(S0 / K)
        + (r + 0.5 * sigma ** 2) * T
    ) / (
        sigma * np.sqrt(T)
    )

    d2 = d1 - sigma * np.sqrt(T)

    bs = (
        S0 * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    assert abs(geom - bs) < 1e-10


def test_geometric_zero_volatility_limit():
    """
    At sigma=0 the geometric Asian price is deterministic.
    """

    params = AsianOptionParams(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.0,
        T=1.0,
        N=52,
    )

    price = geometric_asian_call_price(params)

    t = monitoring_times(params)

    deterministic_geom = np.exp(
        np.mean(
            np.log(
                params.S0 * np.exp(params.r * t)
            )
        )
    )

    expected = (
        np.exp(-params.r * params.T)
        * max(
            deterministic_geom - params.K,
            0.0,
        )
    )

    assert abs(price - expected) < 1e-10


def test_geometric_zero_volatility_delayed_averaging():
    """
    Zero-volatility limit should also hold
    under delayed averaging windows.
    """

    params = AsianOptionParams(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.0,
        T=1.0,
        N=12,
        T1=0.5,
        T2=1.0,
    )

    price = geometric_asian_call_price(params)

    t = monitoring_times(params)

    deterministic_geom = np.exp(
        np.mean(
            np.log(
                params.S0 * np.exp(params.r * t)
            )
        )
    )

    expected = (
        np.exp(-params.r * params.T)
        * max(
            deterministic_geom - params.K,
            0.0,
        )
    )

    assert abs(price - expected) < 1e-10


def test_levy_approx_close_to_mc_at_moderate_vol():
    """
    Levy approximation should stay close to MC.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=52,
    )

    levy = levy_approx_call_price(params)

    rng = np.random.default_rng(11)

    n_pairs = 200_000

    z = rng.standard_normal(
        (n_pairs, params.N)
    )

    paths_p = build_paths(
        params,
        z,
        method="incremental",
    )

    paths_n = build_paths(
        params,
        -z,
        method="incremental",
    )

    p_pos = np.maximum(
        paths_p.mean(1) - params.K,
        0.0,
    )

    p_neg = np.maximum(
        paths_n.mean(1) - params.K,
        0.0,
    )

    df = discount_factor(params)

    pair = 0.5 * df * (p_pos + p_neg)

    mc_price = pair.mean()

    mc_se = (
        pair.std(ddof=1)
        / np.sqrt(n_pairs)
    )

    assert abs(levy - mc_price) / mc_price < 0.01

    assert abs(levy - mc_price) < (
        10 * mc_se + 0.05
    )


def test_levy_approx_delayed_averaging():
    """
    Levy approximation should remain stable
    under delayed averaging windows.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=52,
        T1=0.25,
        T2=1.0,
    )

    levy = levy_approx_call_price(params)

    assert levy > 0.0


def test_put_not_implemented_is_explicit():

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=52,
        option_type="put",
    )

    with pytest.raises(NotImplementedError):
        geometric_asian_call_price(params)

    with pytest.raises(NotImplementedError):
        levy_approx_call_price(params)
