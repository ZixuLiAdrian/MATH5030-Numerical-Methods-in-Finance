"""Option Greeks: pathwise vs likelihood-ratio vs finite-difference comparison.

Three methods are benchmarked across four parameter regimes
(ATM, ITM, OTM, high-vol):

* Pathwise (IPA): delta, vega, rho with per-Greek standard errors.
* Likelihood-ratio: delta, vega; std_err compared to pathwise to quantify
  the variance overhead of the general-purpose LR approach.
* Finite-difference (bump-and-reprice with common random numbers):
  delta, gamma, vega, rho.

Results are written to ``results/tables/greeks_comparison.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import TABLES_DIR, AsianOptionParams
from asian_option_pricer import fd_greeks, lr_greeks, pathwise_greeks


SCENARIOS = [
    {"label": "ATM",      "K": 100.0, "sigma": 0.20},
    {"label": "ITM",      "K":  90.0, "sigma": 0.20},
    {"label": "OTM",      "K": 110.0, "sigma": 0.20},
    {"label": "high_vol", "K": 100.0, "sigma": 0.50},
]
FIXED = dict(S0=100.0, r=0.05, T=1.0, N=52)
N_PATHS_PW_LR = 200_000
N_PATHS_FD = 40_000   # 5 re-pricings per Greek; total paths ≈ N_PATHS_PW_LR
SEED = 20240420


def main() -> None:
    t0 = time.perf_counter()
    rows = []

    for sc in SCENARIOS:
        params_kwargs = {
            key: value
            for key, value in sc.items()
            if key != "label"
        }
        params = AsianOptionParams(**params_kwargs, **FIXED)
        print(f"\n  {sc['label']:<10} (K={sc['K']}, sigma={sc['sigma']})")

        pw = pathwise_greeks(params, N_PATHS_PW_LR, seed=SEED)
        lr = lr_greeks(params, N_PATHS_PW_LR, seed=SEED)
        fd = fd_greeks(params, N_PATHS_FD, seed=SEED)

        # Variance ratio: how many more paths does LR need to match pathwise SE?
        delta_vr = (
            (lr["std_err_delta"] / pw["std_err_delta"]) ** 2
            if pw["std_err_delta"] > 1e-12 else float("nan")
        )
        vega_vr = (
            (lr["std_err_vega"] / pw["std_err_vega"]) ** 2
            if pw["std_err_vega"] > 1e-12 else float("nan")
        )

        rows.append({
            "scenario": sc["label"],
            "sigma": sc["sigma"],
            "K": sc["K"],
            # Pathwise
            "pw_price":     round(pw["price"], 6),
            "pw_delta":     round(pw["delta"], 6),
            "pw_delta_se":  round(pw["std_err_delta"], 6),
            "pw_vega":      round(pw["vega"], 6),
            "pw_vega_se":   round(pw["std_err_vega"], 6),
            "pw_rho":       round(pw["rho"], 6),
            "pw_rho_se":    round(pw["std_err_rho"], 6),
            "pw_runtime_ms": round(pw["runtime_s"] * 1e3, 1),
            # LR
            "lr_delta":     round(lr["delta"], 6),
            "lr_delta_se":  round(lr["std_err_delta"], 6),
            "lr_vega":      round(lr["vega"], 6),
            "lr_vega_se":   round(lr["std_err_vega"], 6),
            "lr_runtime_ms": round(lr["runtime_s"] * 1e3, 1),
            # LR vs pathwise variance ratio (paths needed to match SE)
            "lr_pw_var_ratio_delta": round(delta_vr, 1) if np.isfinite(delta_vr) else float("nan"),
            "lr_pw_var_ratio_vega":  round(vega_vr, 1)  if np.isfinite(vega_vr)  else float("nan"),
            # FD
            "fd_delta":     round(fd["delta"], 6),
            "fd_gamma":     round(fd["gamma"], 6),
            "fd_vega":      round(fd["vega"], 6),
            "fd_rho":       round(fd["rho"], 6),
            "fd_runtime_ms": round(fd["runtime_s"] * 1e3, 1),
        })

        print(
            f"    pathwise  delta={pw['delta']:.5f} (se={pw['std_err_delta']:.5f})  "
            f"vega={pw['vega']:.5f}  rho={pw['rho']:.5f}"
        )
        print(
            f"    LR        delta={lr['delta']:.5f} (se={lr['std_err_delta']:.5f})  "
            f"var_ratio={delta_vr:.1f}x"
        )
        print(
            f"    FD        delta={fd['delta']:.5f}  gamma={fd['gamma']:.5f}  "
            f"vega={fd['vega']:.5f}  rho={fd['rho']:.5f}"
        )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "greeks_comparison.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nGreeks comparison complete in {elapsed:.1f}s")
    print(f"  table -> {out_path}")


if __name__ == "__main__":
    main()
