"""
Pilot-based adaptive estimator selector for arithmetic Asian options.

:func:`smart_price` allocates a small fraction of the path budget to a pilot
phase that benchmarks several candidate estimators, then routes the remaining
budget to whichever candidate achieves the highest efficiency:

    efficiency = 1 / (std_err² × runtime_s)

which is proportional to variance⁻¹ per unit time — the natural measure of
how quickly an estimator reduces error.

The key design choice for QMC candidates: a single-scramble Sobol run reports
a proxy ``std_err`` (sample std / √n) that is unreliable at small pilot sizes
because it ignores the low-discrepancy dependence structure.  We address this
by running the RQMC candidate with multiple independent scrambles during the
pilot, giving an honest cross-replication standard error.
"""
from __future__ import annotations

import time

import numpy as np

from .control_variate import antithetic_cv_price, control_variate_price
from .models import AsianOptionParams
from .monte_carlo import antithetic_mc_price, standard_mc_price
from .qmc import rqmc_sobol_price


# ── Candidate pools ───────────────────────────────────────────────────────────

_CALL_CANDIDATES: tuple[str, ...] = (
    "antithetic_cv",
    "rqmc_bridge",
    "control_variate",
    "antithetic",
)

_PUT_CANDIDATES: tuple[str, ...] = (
    "rqmc_bridge",
    "antithetic",
    "standard_mc",
)

_MIN_PER_CANDIDATE: int = 128   # minimum paths each candidate receives in the pilot
_MIN_MAIN_PATHS: int = 64


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_candidate(
    name: str,
    params: AsianOptionParams,
    n_paths: int,
    seed: int,
    n_replications: int,
) -> dict:
    """Dispatch a single candidate estimator."""
    if name == "antithetic_cv":
        return antithetic_cv_price(params, n_paths, seed=seed)
    if name == "control_variate":
        return control_variate_price(params, n_paths, seed=seed)
    if name == "antithetic":
        return antithetic_mc_price(params, n_paths, seed=seed)
    if name == "standard_mc":
        return standard_mc_price(params, n_paths, seed=seed)
    if name == "rqmc_bridge":
        return rqmc_sobol_price(
            params,
            n_paths,
            seed=seed,
            n_replications=n_replications,
            path_method="brownian_bridge",
        )
    raise ValueError(f"Unknown candidate '{name}'.")


def _floor_pow2_at_least_2(n: int) -> int:
    """Return the largest power of two in ``[2, n]``."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    return 1 << (n.bit_length() - 1)


def _candidate_request_n_paths(
    name: str,
    budget: int,
    n_replications: int,
) -> int:
    """Return a request size whose realised path count fits within ``budget``."""
    if budget <= 0:
        return 0

    if name == "rqmc_bridge":
        min_budget = 2 * n_replications
        if budget < min_budget:
            return 0
        per_rep_budget = budget // n_replications
        per_rep = _floor_pow2_at_least_2(per_rep_budget)
        return per_rep * n_replications

    if name in {"antithetic_cv", "antithetic"}:
        return budget if budget >= 2 else 0

    return budget


def _split_integer_budget(
    total_budget: int,
    n_parts: int,
) -> list[int]:
    """Split an integer budget into near-equal integer parts."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive.")
    base = total_budget // n_parts
    remainder = total_budget % n_parts
    return [
        base + (1 if idx < remainder else 0)
        for idx in range(n_parts)
    ]


def _efficiency(std_err: float, runtime_s: float) -> float:
    """Precision per unit time — higher is better.

    Returns ``inf`` when ``std_err == 0`` (degenerate zero-variance case,
    e.g. deep-OTM where every simulated payoff is zero).  Ties at ``inf``
    are broken by runtime in the selection step.
    """
    if runtime_s <= 0.0:
        return 0.0
    if std_err <= 0.0:
        return float("inf")
    return 1.0 / (std_err ** 2 * runtime_s)


# ── Public API ────────────────────────────────────────────────────────────────

def smart_price(
    params: AsianOptionParams,
    n_paths: int,
    pilot_fraction: float = 0.05,
    n_pilot_replications: int = 8,
    n_main_replications: int = 16,
    seed: int = 123,
) -> dict:
    """Adaptive estimator: pilot-select the best method, then price with the rest.

    Two-phase algorithm
    -------------------
    1. **Pilot** (total budget ``pilot_fraction × n_paths``, minimum 512 when
       the overall budget is large enough):
       Split the pilot budget across the candidate estimators, then record each
       candidate's ``std_err`` and ``runtime_s``.
       For the RQMC candidate, use ``n_pilot_replications`` independent scrambles
       to obtain a cross-replication standard error (unbiased), rather than the
       unreliable proxy from a single scramble.

    2. **Main** (remaining paths):
       Run the winner with a fresh seed and report its result as the final price.

    Selection metric
    ----------------
    ``efficiency = 1 / (std_err² × runtime_s)`` — proportional to variance⁻¹
    per unit time.  When all candidates return zero variance (deep-OTM), ties
    are broken by runtime (fastest wins).

    Candidate pools
    ---------------
    Calls: antithetic_cv, rqmc_bridge, control_variate, antithetic.
    Puts:  rqmc_bridge, antithetic, standard_mc.

    Parameters
    ----------
    params : AsianOptionParams
    n_paths : int
        Total GBM path budget across pilot and main phases combined.
    pilot_fraction : float
        Fraction of budget spent on the pilot (default 0.05).
    n_pilot_replications : int
        Scrambles used for the RQMC candidate in the pilot (default 8).
    n_main_replications : int
        Scrambles used if RQMC wins the main phase (default 16).
    seed : int
        Base seed.  Pilot uses ``seed``; main uses ``seed + 1``.

    Returns
    -------
    dict with keys
        ``price``, ``std_err``, ``runtime_s``, ``n_paths``,
        ``selected_method``, ``pilot_scores``, ``pilot_n_paths``,
        ``main_n_paths``.
        ``pilot_scores`` is a nested dict mapping each candidate name to
        ``{std_err, runtime_s, efficiency, n_paths}``.
    """
    params.validate()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if not 0.0 < pilot_fraction < 1.0:
        raise ValueError("pilot_fraction must be in (0, 1).")

    t_total = time.perf_counter()
    candidates = _CALL_CANDIDATES if params.option_type == "call" else _PUT_CANDIDATES

    pilot_budget = int(n_paths * pilot_fraction)
    min_pilot_total = _MIN_PER_CANDIDATE * len(candidates)
    if n_paths >= min_pilot_total + _MIN_MAIN_PATHS:
        pilot_budget = max(pilot_budget, min_pilot_total)
    pilot_budget = min(pilot_budget, max(n_paths - _MIN_MAIN_PATHS, 0))

    # ── Pilot phase ───────────────────────────────────────────────────────────
    pilot_scores: dict[str, dict] = {}
    pilot_outputs: dict[str, dict] = {}
    pilot_actual_n = 0
    pilot_budgets = _split_integer_budget(pilot_budget, len(candidates))

    for name, candidate_budget in zip(candidates, pilot_budgets):
        request_n = _candidate_request_n_paths(
            name,
            candidate_budget,
            n_pilot_replications,
        )
        if request_n <= 0:
            out = {
                "std_err": float("nan"),
                "runtime_s": float("nan"),
                "n_paths": 0,
            }
            eff = 0.0
        else:
            try:
                out = _run_candidate(
                    name,
                    params,
                    request_n,
                    seed,
                    n_pilot_replications,
                )
                eff = _efficiency(out["std_err"], out["runtime_s"])
            except Exception:
                eff = 0.0
                out = {
                    "std_err": float("nan"),
                    "runtime_s": float("nan"),
                    "n_paths": 0,
                }

        pilot_outputs[name] = out
        pilot_actual_n += int(out.get("n_paths", 0))
        pilot_scores[name] = {
            "std_err": float(out.get("std_err", float("nan"))),
            "runtime_s": float(out.get("runtime_s", float("nan"))),
            "efficiency": eff,
            "n_paths": int(out.get("n_paths", 0)),
        }

    # ── Selection ─────────────────────────────────────────────────────────────
    available = [
        k for k in candidates
        if pilot_scores[k]["n_paths"] > 0
    ]
    if not available:
        available = list(candidates)

    inf_candidates = [
        k for k in available
        if pilot_scores[k]["efficiency"] == float("inf")
    ]
    if inf_candidates:
        # Zero-variance tie: pick fastest to minimise overhead
        best = min(
            inf_candidates,
            key=lambda k: pilot_scores[k]["runtime_s"],
        )
    else:
        best = max(available, key=lambda k: pilot_scores[k]["efficiency"])

    # ── Main phase ────────────────────────────────────────────────────────────
    main_seed = seed + 1
    remaining_budget = max(n_paths - pilot_actual_n, 0)
    main_request_n = _candidate_request_n_paths(
        best,
        remaining_budget,
        n_main_replications,
    )

    if main_request_n > 0:
        out = _run_candidate(
            best,
            params,
            main_request_n,
            main_seed,
            n_main_replications,
        )
        main_actual_n = int(out["n_paths"])
    else:
        out = pilot_outputs.get(best, {
            "price": float("nan"),
            "std_err": float("nan"),
            "runtime_s": 0.0,
            "n_paths": 0,
        })
        main_actual_n = 0

    elapsed = time.perf_counter() - t_total
    return {
        "price": out["price"],
        "std_err": out["std_err"],
        "runtime_s": elapsed,
        "n_paths": pilot_actual_n + main_actual_n,
        "selected_method": best,
        "pilot_scores": pilot_scores,
        "pilot_n_paths": pilot_actual_n,
        "main_n_paths": main_actual_n,
    }
