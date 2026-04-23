"""Convenience ``benchmark_suite`` used by :mod:`experiments.run_benchmarks`.

Runs every estimator on a single canonical test case (S0=100, K=100, r=5%,
sigma=20%, T=1, N=52 weekly monitorings). The output is a flat dict ready for
pretty-printing or tabular export.
"""
from __future__ import annotations

from .analytic import geometric_asian_call_price, levy_approx_call_price
from .control_variate import antithetic_cv_price, control_variate_price
from .models import AsianOptionParams
from .monte_carlo import antithetic_mc_price, standard_mc_price
from .qmc import rqmc_sobol_price, sobol_qmc_price

DEFAULT_PARAMS = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, N=52)
DEFAULT_PATHS = 100_000
DEFAULT_SEED = 123


def benchmark_suite(
    params: AsianOptionParams = DEFAULT_PARAMS,
    n_paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run every estimator on the given parameters and return a results dict."""
    return {
        "params": {
            "S0": params.S0,
            "K": params.K,
            "r": params.r,
            "sigma": params.sigma,
            "T": params.T,
            "N": params.N,
        },
        "geometric_exact": geometric_asian_call_price(params),
        "levy": levy_approx_call_price(params),
        "standard_mc": standard_mc_price(params, n_paths, seed=seed),
        "antithetic": antithetic_mc_price(params, n_paths, seed=seed),
        "control_variate": control_variate_price(params, n_paths, seed=seed),
        "antithetic_cv": antithetic_cv_price(params, n_paths, seed=seed),
        "sobol_qmc_incremental": sobol_qmc_price(
            params, n_paths, seed=seed, path_method="incremental"
        ),
        "sobol_qmc_bridge": sobol_qmc_price(
            params, n_paths, seed=seed, path_method="brownian_bridge"
        ),
        "rqmc_bridge": rqmc_sobol_price(
            params, n_paths, seed=seed, n_replications=16, path_method="brownian_bridge"
        ),
    }
