"""Literature/reference validation for the Asian-option estimators.

Two validation tables are produced:

1. ``geometric_validation.csv``
   Compares the Kemna-Vorst closed-form geometric Asian price against a
   high-precision antithetic Monte Carlo estimate across a grid of
   ``(sigma, K)`` scenarios. This stress-tests the closed-form implementation
   itself: the two prices must agree up to MC noise.

2. ``arithmetic_validation.csv``
   For the same grid, computes a high-precision arithmetic reference price
   (RQMC + antithetic-CV, ~2M paths each) and then reports the error of each
   estimator (standard MC, antithetic, control variate, antithetic+CV, Sobol
   QMC with both path constructions, RQMC) at a fixed moderate budget of
   ``N_PATHS_METHOD`` paths per method. Error magnitudes are reported both in
   absolute terms and in basis points of the reference price.

Both tables are written to ``results/tables/`` and the key columns are
pretty-printed to stdout for quick inspection.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import (
    TABLES_DIR,
    AsianOptionParams,
    high_precision_reference,
)
from asian_option_pricer import (
    antithetic_cv_price,
    antithetic_mc_price,
    control_variate_price,
    geometric_asian_call_price,
    geometric_payoff_from_paths,
    levy_approx_call_price,
    rqmc_sobol_price,
    sobol_qmc_price,
    standard_mc_price,
    build_paths,
)
from asian_option_pricer.utils import discount_factor


# Scenario grid: spans moneyness and volatility regimes that appear in Levy
# (1992) and Turnbull-Wakeman (1991). Parameters held fixed across scenarios:
# r = 5%, T = 1 year, N = 50 equally spaced monitoring dates, S0 = 100.
GRID_SIGMA = [0.10, 0.20, 0.30, 0.50]
GRID_STRIKE = [90.0, 100.0, 110.0]
FIXED = dict(S0=100.0, r=0.05, T=1.0, N=50)
N_PATHS_METHOD = 100_000
N_PATHS_MC_GEOM = 200_000
SEED = 20240420


def _geom_mc_antithetic(params: AsianOptionParams, n_paths: int, seed: int) -> dict:
    """Antithetic MC for the geometric Asian call (used only here to
    independently verify the closed-form price)."""
    half = n_paths // 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((half, params.N))
    df = discount_factor(params)
    pv_p = df * geometric_payoff_from_paths(
        build_paths(params, z, method="incremental"), params.K, params.option_type
    )
    pv_n = df * geometric_payoff_from_paths(
        build_paths(params, -z, method="incremental"), params.K, params.option_type
    )
    pv_pair = 0.5 * (pv_p + pv_n)
    return {
        "price": float(pv_pair.mean()),
        "std_err": float(pv_pair.std(ddof=1) / np.sqrt(half)),
    }


def _method_runs(params: AsianOptionParams) -> dict[str, dict]:
    """Run every estimator once at ``N_PATHS_METHOD`` paths."""
    return {
        "levy_approx": {
            "price": levy_approx_call_price(params),
            "std_err": 0.0,
            "runtime_s": 0.0,
            "n_paths": 0,
        },
        "standard_mc": standard_mc_price(params, N_PATHS_METHOD, seed=SEED),
        "antithetic": antithetic_mc_price(params, N_PATHS_METHOD, seed=SEED),
        "control_variate": control_variate_price(params, N_PATHS_METHOD, seed=SEED),
        "antithetic_cv": antithetic_cv_price(params, N_PATHS_METHOD, seed=SEED),
        "sobol_incremental": sobol_qmc_price(
            params, N_PATHS_METHOD, seed=SEED, path_method="incremental"
        ),
        "sobol_bridge": sobol_qmc_price(
            params, N_PATHS_METHOD, seed=SEED, path_method="brownian_bridge"
        ),
        "rqmc_bridge": rqmc_sobol_price(
            params,
            N_PATHS_METHOD,
            seed=SEED,
            n_replications=16,
            path_method="brownian_bridge",
        ),
    }


def validate_geometric() -> pd.DataFrame:
    """Closed-form vs. antithetic MC for the geometric Asian call."""
    rows = []
    for sigma in GRID_SIGMA:
        for K in GRID_STRIKE:
            params = AsianOptionParams(sigma=sigma, K=K, **FIXED)
            closed = geometric_asian_call_price(params)
            mc = _geom_mc_antithetic(params, N_PATHS_MC_GEOM, seed=SEED)
            abs_err = closed - mc["price"]
            rows.append(
                {
                    "sigma": sigma,
                    "K": K,
                    "closed_form": closed,
                    "mc_price": mc["price"],
                    "mc_std_err": mc["std_err"],
                    "abs_error": abs_err,
                    "z_score": abs_err / mc["std_err"] if mc["std_err"] > 0 else float("nan"),
                }
            )
    df = pd.DataFrame(rows)
    return df


def validate_arithmetic() -> pd.DataFrame:
    """Reference-vs-method error for the arithmetic Asian call."""
    rows = []
    for sigma in GRID_SIGMA:
        for K in GRID_STRIKE:
            params = AsianOptionParams(sigma=sigma, K=K, **FIXED)
            ref_price, ref_se = high_precision_reference(params)
            runs = _method_runs(params)
            for name, out in runs.items():
                abs_err = out["price"] - ref_price
                rel_err_bps = 1.0e4 * abs_err / ref_price if ref_price > 0 else float("nan")
                rows.append(
                    {
                        "sigma": sigma,
                        "K": K,
                        "method": name,
                        "price": out["price"],
                        "std_err": out.get("std_err", float("nan")),
                        "runtime_s": out.get("runtime_s", float("nan")),
                        "n_paths": out.get("n_paths", 0),
                        "reference_price": ref_price,
                        "reference_std_err": ref_se,
                        "abs_error": abs_err,
                        "rel_error_bps": rel_err_bps,
                    }
                )
    return pd.DataFrame(rows)


def validate_pyfeng_convergence() -> pd.DataFrame:
    """Discrete arithmetic Asian converges to the continuous limit as N → ∞.

    Uses PyFENG's BsmAsianJsu (Johnson's SU approximation for the continuous
    arithmetic Asian) as an independent reference. Prices are computed with
    RQMC Brownian-bridge at 131k paths / 16 scrambles; the continuous limit is
    the same call at N → ∞ under GBM.
    """
    try:
        from pyfeng.asian import BsmAsianJsu
    except ImportError:
        print("pyfeng not installed — skipping continuous-limit validation.")
        return pd.DataFrame()

    model = BsmAsianJsu(sigma=0.20, intr=0.05)
    continuous_price = float(model.price(strike=100.0, spot=100.0, texp=1.0))

    rows = []
    for N in [52, 250, 500, 1000]:
        params = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, N=N)
        out = rqmc_sobol_price(
            params, n_paths=131_072, seed=SEED, n_replications=16,
            path_method="brownian_bridge",
        )
        rows.append({
            "N": N,
            "discrete_price": round(out["price"], 6),
            "std_err": round(out["std_err"], 6),
            "pyfeng_jsu": round(continuous_price, 6),
            "gap": round(out["price"] - continuous_price, 6),
        })
    return pd.DataFrame(rows)


def _pretty_geometric(df: pd.DataFrame) -> pd.DataFrame:
    return df[["sigma", "K", "closed_form", "mc_price", "mc_std_err", "abs_error", "z_score"]]


def _pretty_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["sigma", "K"],
        columns="method",
        values="abs_error",
        aggfunc="first",
    )
    return pivot


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 78)
    print("GEOMETRIC ASIAN VALIDATION (closed-form vs antithetic MC)")
    print("=" * 78)
    geom_df = validate_geometric()
    geom_path = TABLES_DIR / "geometric_validation.csv"
    geom_df.to_csv(geom_path, index=False)
    print(_pretty_geometric(geom_df).to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    max_z = geom_df["z_score"].abs().max()
    print(f"\nMax |z|-score across scenarios: {max_z:.2f}  (rule of thumb: < 3)")
    print(f"Table saved to {geom_path.relative_to(TABLES_DIR.parent.parent)}")

    print()
    print("=" * 78)
    print("ARITHMETIC ASIAN VALIDATION (method error vs high-precision reference)")
    print("=" * 78)
    arith_df = validate_arithmetic()
    arith_path = TABLES_DIR / "arithmetic_validation.csv"
    arith_df.to_csv(arith_path, index=False)
    print("\nAbsolute error by (sigma, K, method):")
    pivot = _pretty_arithmetic(arith_df)
    print(pivot.to_string(float_format=lambda x: f"{x: .5f}"))

    print("\nStandard error by (sigma, K, method):")
    pivot_se = arith_df.pivot_table(
        index=["sigma", "K"], columns="method", values="std_err", aggfunc="first"
    )
    print(pivot_se.to_string(float_format=lambda x: f"{x: .5f}"))

    print()
    print("=" * 78)
    print("CONTINUOUS-LIMIT VALIDATION (discrete → BsmAsianJsu as N → ∞)")
    print("=" * 78)
    pyfeng_df = validate_pyfeng_convergence()
    if not pyfeng_df.empty:
        pyfeng_path = TABLES_DIR / "pyfeng_convergence.csv"
        pyfeng_df.to_csv(pyfeng_path, index=False)
        print(pyfeng_df.to_string(index=False))
        print(f"\nTable saved to {pyfeng_path.relative_to(TABLES_DIR.parent.parent)}")

    elapsed = time.perf_counter() - t0
    print(f"\nValidation complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
