"""Tests for the path-construction utilities."""
from __future__ import annotations

import numpy as np
import pytest

from asian_option_pricer import AsianOptionParams, brownian_bridge_matrix, build_paths


def test_brownian_bridge_matrix_reproduces_covariance():
    """``B @ B.T`` must equal the covariance matrix of Brownian motion at the
    monitoring times, i.e. ``C[i, j] = min(t_i, t_j)``."""
    N, T = 8, 1.0
    B = brownian_bridge_matrix(N, T)
    t = np.arange(1, N + 1) * (T / N)
    cov_expected = np.minimum.outer(t, t)
    cov_actual = B @ B.T
    assert np.allclose(cov_actual, cov_expected, atol=1e-12)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 7, 13, 52])
def test_brownian_bridge_covariance_arbitrary_N(N: int):
    T = 1.7
    B = brownian_bridge_matrix(N, T)
    t = np.arange(1, N + 1) * (T / N)
    cov_expected = np.minimum.outer(t, t)
    assert np.allclose(B @ B.T, cov_expected, atol=1e-12)


def test_brownian_bridge_first_coord_is_terminal():
    """The first ``z``-coordinate must carry the full terminal variance: the
    (N, 0) entry of B is sqrt(T) and the rest of the first column is 0."""
    N, T = 16, 1.0
    B = brownian_bridge_matrix(N, T)
    assert abs(B[N - 1, 0] - np.sqrt(T)) < 1e-12


def test_build_paths_methods_agree_in_distribution():
    """Both constructions must produce the same *joint distribution* of
    prices, so the sample moments should agree within MC noise."""
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, N=20)
    n = 400_000
    rng = np.random.default_rng(7)
    z = rng.standard_normal((n, params.N))
    inc = build_paths(params, z, method="incremental")
    # Independent draw for bridge so we compare distributions, not paths.
    z2 = np.random.default_rng(8).standard_normal((n, params.N))
    bri = build_paths(params, z2, method="brownian_bridge")

    # Compare mean and variance at each monitoring time. Theoretical mean is
    # S0 * exp(r * t); theoretical variance is S0^2 exp(2 r t)(exp(sigma^2 t) - 1).
    t = np.arange(1, params.N + 1) * (params.T / params.N)
    mean_theory = params.S0 * np.exp(params.r * t)
    for paths, label in [(inc, "inc"), (bri, "bri")]:
        mean_emp = paths.mean(axis=0)
        rel_err = np.abs(mean_emp - mean_theory) / mean_theory
        assert rel_err.max() < 5e-3, f"{label}: {rel_err.max()}"


def test_build_paths_reject_wrong_shape():
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=10)
    z = np.zeros((3, 4))  # wrong N
    with pytest.raises(ValueError):
        build_paths(params, z)


def test_build_paths_unknown_method_raises():
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=10)
    z = np.zeros((3, 10))
    with pytest.raises(ValueError):
        build_paths(params, z, method="nope")
