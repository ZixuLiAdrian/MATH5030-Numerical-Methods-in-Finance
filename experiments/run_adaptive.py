"""Adaptive estimator selection: which method wins in each parameter regime?

``smart_price`` is run across six scenarios (ATM, ITM, OTM, high-vol,
short-maturity, long-maturity) with ``N_REPS`` independent repetitions.
For each scenario we record:

* which method was selected in each replication,
* the RMSE of the adaptive price against a high-precision reference,
* the pilot efficiency scores for every candidate.

Results are written to ``results/tables/adaptive_selection.csv``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import TABLES_DIR, AsianOptionParams, high_precision_reference
from asian_option_pricer import smart_price


SCENARIOS = [
    {"label": "ATM",        "K": 100.0, "sigma": 0.20, "T": 1.0, "N": 52},
    {"label": "ITM",        "K":  90.0, "sigma": 0.20, "T": 1.0, "N": 52},
    {"label": "OTM",        "K": 115.0, "sigma": 0.20, "T": 1.0, "N": 52},
    {"label": "high_vol",   "K": 100.0, "sigma": 0.50, "T": 1.0, "N": 52},
    {"label": "short_mat",  "K": 100.0, "sigma": 0.20, "T": 0.1, "N": 10},
    {"label": "long_mat",   "K": 100.0, "sigma": 0.20, "T": 3.0, "N": 156},
]
FIXED = dict(S0=100.0, r=0.05)
N_PATHS = 50_000
N_REPS = 8
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
        print(f"\n  {sc['label']:<12} (K={sc['K']}, sigma={sc['sigma']}, T={sc['T']})")

        ref, ref_se = high_precision_reference(params)
        print(f"    reference = {ref:.6f}  (se={ref_se:.2e})")

        prices: list[float] = []
        methods: list[str] = []
        for rep in range(N_REPS):
            out = smart_price(params, N_PATHS, seed=SEED + rep * 1_000)
            prices.append(out["price"])
            methods.append(out["selected_method"])

        # Pilot scores from one canonical run
        pilot_out = smart_price(params, N_PATHS, seed=SEED)
        pilot = pilot_out["pilot_scores"]

        most_selected = max(set(methods), key=methods.count)
        consistency = methods.count(most_selected) / N_REPS
        rmse = float(np.sqrt(np.mean((np.array(prices) - ref) ** 2)))

        print(
            f"    selected={most_selected:<15}  "
            f"consistency={consistency:.0%}  rmse={rmse:.5f}"
        )
        print(f"    pilot efficiencies: " + "  ".join(
            f"{k}={v['efficiency']:.2e}" for k, v in pilot.items()
        ))

        row: dict = {
            "scenario": sc["label"],
            "sigma": sc["sigma"],
            "K": sc["K"],
            "T": sc["T"],
            "reference_price": round(ref, 6),
            "smart_mean_price": round(float(np.mean(prices)), 6),
            "rmse": round(rmse, 6),
            "most_selected_method": most_selected,
            "selection_consistency": round(consistency, 2),
            "pilot_n_paths": pilot_out["pilot_n_paths"],
            "main_n_paths": pilot_out["main_n_paths"],
            "total_n_paths": pilot_out["n_paths"],
        }
        for name, scores in pilot.items():
            row[f"pilot_eff_{name}"] = (
                round(scores["efficiency"], 4)
                if np.isfinite(scores["efficiency"]) else float("inf")
            )
            row[f"pilot_se_{name}"] = round(scores["std_err"], 6)
            row[f"pilot_n_{name}"] = scores["n_paths"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "adaptive_selection.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nAdaptive selection study complete in {elapsed:.1f}s")
    print(f"  table -> {out_path}")

    print("\nSummary:")
    print(df[["scenario", "most_selected_method", "selection_consistency", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
