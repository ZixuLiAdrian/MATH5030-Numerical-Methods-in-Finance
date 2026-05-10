"""Tests for the path-construction utilities."""

from __future__ import annotations

import numpy as np
import pytest

from asian_option_pricer import (
    AsianOptionParams,
    brownian_bridge_matrix,
    build_paths,
)

from asian_option_pricer.paths import (
    brownian_bridge_matrix_from_times,
)

from asian_option_pricer.utils import (
    monitoring_times,
)


def test_brownian_bridge_matrix_reproduces_covariance():
    """
    B @ B.T must equal Brownian covariance matrix.
    """

    N = 8
    T = 1.0

    B = brownian_bridge_matrix(N, T)

    t = np.arange(
        1,
        N + 1,
    ) * (T / N)

    cov_expected = np.minimum.outer(
        t,
        t,
    )

    cov_actual = B @ B.T

    assert np.allclose(
        cov_actual,
        cov_expected,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "N",
    [1, 2, 3, 4, 5, 7, 13, 52],
)
def test_brownian_bridge_covariance_arbitrary_N(
    N: int,
):

    T = 1.7

    B = brownian_bridge_matrix(
        N,
        T,
    )

    t = np.arange(
        1,
        N + 1,
    ) * (T / N)

    cov_expected = np.minimum.outer(
        t,
        t,
    )

    assert np.allclose(
        B @ B.T,
        cov_expected,
        atol=1e-12,
    )


def test_brownian_bridge_covariance_delayed_averaging():
    """
    Generalized bridge covariance should work
    for arbitrary monitoring times.
    """

    times = np.array([
        0.375,
        0.500,
        0.625,
        0.750,
    ])

    B = brownian_bridge_matrix_from_times(
        tuple(times.tolist())
    )

    cov_expected = np.minimum.outer(
        times,
        times,
    )

    cov_actual = B @ B.T

    assert np.allclose(
        cov_actual,
        cov_expected,
        atol=1e-12,
    )


def test_monitoring_times_support_delayed_averaging():
    """
    monitoring_times should correctly generate
    dates between T1 and T2.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=4,
        T1=0.25,
        T2=0.75,
    )

    expected = np.array([
        0.375,
        0.500,
        0.625,
        0.750,
    ])

    actual = monitoring_times(params)

    assert np.allclose(
        actual,
        expected,
    )


def test_default_monitoring_times_preserve_old_behavior():
    """
    Default behavior should remain identical
    to the original implementation.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=4,
    )

    expected = np.array([
        0.25,
        0.50,
        0.75,
        1.00,
    ])

    actual = monitoring_times(params)

    assert np.allclose(
        actual,
        expected,
    )


def test_brownian_bridge_first_coord_is_terminal():
    """
    First bridge coordinate should carry
    terminal variance.
    """

    N = 16
    T = 1.0

    B = brownian_bridge_matrix(
        N,
        T,
    )

    assert abs(
        B[N - 1, 0]
        - np.sqrt(T)
    ) < 1e-12


def test_build_paths_methods_agree_in_distribution():
    """
    Incremental and bridge constructions
    should produce the same distribution.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=20,
    )

    n = 400_000

    rng = np.random.default_rng(7)

    z = rng.standard_normal(
        (n, params.N)
    )

    inc = build_paths(
        params,
        z,
        method="incremental",
    )

    z2 = np.random.default_rng(8).standard_normal(
        (n, params.N)
    )

    bri = build_paths(
        params,
        z2,
        method="brownian_bridge",
    )

    t = monitoring_times(params)

    mean_theory = (
        params.S0
        * np.exp(params.r * t)
    )

    for paths, label in [
        (inc, "inc"),
        (bri, "bri"),
    ]:

        mean_emp = paths.mean(axis=0)

        rel_err = (
            np.abs(mean_emp - mean_theory)
            / mean_theory
        )

        assert rel_err.max() < 5e-3, (
            f"{label}: {rel_err.max()}"
        )


def test_build_paths_methods_agree_in_distribution_delayed():
    """
    Incremental and bridge constructions
    should still agree under delayed averaging.
    """

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=20,
        T1=0.25,
        T2=1.0,
    )

    n = 400_000

    rng = np.random.default_rng(77)

    z = rng.standard_normal(
        (n, params.N)
    )

    inc = build_paths(
        params,
        z,
        method="incremental",
    )

    z2 = np.random.default_rng(88).standard_normal(
        (n, params.N)
    )

    bri = build_paths(
        params,
        z2,
        method="brownian_bridge",
    )

    t = monitoring_times(params)

    mean_theory = (
        params.S0
        * np.exp(params.r * t)
    )

    for paths, label in [
        (inc, "inc"),
        (bri, "bri"),
    ]:

        mean_emp = paths.mean(axis=0)

        rel_err = (
            np.abs(mean_emp - mean_theory)
            / mean_theory
        )

        assert rel_err.max() < 5e-3, (
            f"{label}: {rel_err.max()}"
        )


def test_build_paths_reject_wrong_shape():

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=10,
    )

    z = np.zeros((3, 4))

    with pytest.raises(ValueError):

        build_paths(params, z)


def test_build_paths_unknown_method_raises():

    params = AsianOptionParams(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=10,
    )

    z = np.zeros((3, 10))

    with pytest.raises(ValueError):

        build_paths(
            params,
            z,
            method="nope",
        )
