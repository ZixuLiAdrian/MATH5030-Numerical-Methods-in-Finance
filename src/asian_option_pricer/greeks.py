"""
Pathwise (IPA), likelihood-ratio (LR), and finite-difference Greeks
for discretely monitored arithmetic Asian calls under GBM.

Three estimators are provided:

* :func:`pathwise_greeks` -- Infinitesimal Perturbation Analysis (IPA).
  Differentiates the payoff path-by-path via the chain rule through the
  GBM dynamics.  Exact (unbiased) for Asian options because the arithmetic
  average A_N is a.s. smooth in S₀, σ, r (the kink at A_N = K has measure
  zero).  Returns delta, vega, rho with a per-Greek Monte Carlo standard
  error.

* :func:`lr_greeks` -- Score-function / likelihood-ratio estimator.
  Differentiates the simulation density rather than the payoff.  Valid for
  any payoff including discontinuous ones (e.g. digitals), at the cost of
  higher variance than the pathwise estimator for smooth payoffs like the
  Asian call.  Included as a methodological comparison: the variance ratio
  LR/pathwise quantifies how much the kink-free structure of the Asian payoff
  helps.

* :func:`fd_greeks` -- Central finite-difference bump-and-reprice.
  Uses :func:`antithetic_cv_price` with a common random seed for all
  re-pricings (common random numbers technique) to cancel most Monte Carlo
  noise.  Provides delta, gamma, vega, rho; gamma is not available from
  pathwise without an additional second-order estimator.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from .control_variate import antithetic_cv_price
from .models import AsianOptionParams
from .paths import PathMethod, brownian_bridge_matrix_from_times
from .utils import discount_factor, monitoring_times


# ── Internal helper ───────────────────────────────────────────────────────────

def _simulate_with_brownian(
    params: AsianOptionParams,
    z: np.ndarray,
    method: PathMethod,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(paths, W)`` from ``(n_paths, N)`` standard normals.

    ``paths[i, j] = S(t_j)`` on simulation path ``i``.
    ``W[i, j] = W(t_j)`` is the sigma-free Brownian motion evaluated at the
    j-th monitoring time on path ``i``.
    """
    t = monitoring_times(params)
    dt = np.diff(np.concatenate(([0.0], t)))

    if method == "incremental":
        W = np.cumsum(np.sqrt(dt) * z, axis=1)
    elif method == "brownian_bridge":
        B = brownian_bridge_matrix_from_times(tuple(float(x) for x in t))
        W = z @ B.T
    else:
        raise ValueError(
            f"Unknown path_method '{method}'. "
            "Use 'incremental' or 'brownian_bridge'."
        )

    log_paths = (
        np.log(params.S0)
        + (params.r - 0.5 * params.sigma ** 2) * t[None, :]
        + params.sigma * W
    )
    return np.exp(log_paths), W


# ── Public estimators ─────────────────────────────────────────────────────────

def pathwise_greeks(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "incremental",
) -> dict:
    """Pathwise (IPA) delta, vega, and rho for the arithmetic Asian call.

    Derivatives are obtained by differentiating the GBM path through the
    payoff function on each simulated trajectory.  Because the arithmetic
    average A_N is almost surely smooth in S₀, σ, r, the interchange of
    differentiation and expectation is valid and the estimator is unbiased.

    Formulas
    --------
    Let A = (1/N) Σᵢ S(tᵢ), W(tᵢ) = sigma-free BM at tᵢ.

    Delta:
        ∂C/∂S₀ = e^{-rT} E[ (A/S₀) · 1{A > K} ]

    Vega:
        ∂C/∂σ  = e^{-rT} E[ 1{A > K} · (1/N) Σᵢ S(tᵢ)(W(tᵢ) − σ tᵢ) ]

    Rho:
        ∂C/∂r  = e^{-rT} E[ 1{A > K} · (1/N) Σᵢ S(tᵢ) tᵢ ] − T · C

    Returns a dict with keys ``price``, ``delta``, ``vega``, ``rho``,
    ``std_err_delta``, ``std_err_vega``, ``std_err_rho``,
    ``runtime_s``, ``n_paths``.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "Pathwise Greeks are implemented for calls only."
        )
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    t0 = time.perf_counter()
    t = monitoring_times(params)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, params.N))
    paths, W = _simulate_with_brownian(params, z, path_method)

    A = paths.mean(axis=1)                          # (n_paths,)
    df = discount_factor(params)
    payoffs = np.maximum(A - params.K, 0.0)         # (n_paths,)
    itm = (A > params.K).astype(float)              # (n_paths,)

    price = float(df * payoffs.mean())

    # Delta: ∂A/∂S₀ = A/S₀  (each S(tᵢ) is linear in S₀)
    delta_s = df * (A / params.S0) * itm

    # Vega: ∂S(tᵢ)/∂σ = S(tᵢ)(W(tᵢ) − σ tᵢ); chain through arithmetic mean
    vega_s = df * (paths * (W - params.sigma * t[None, :])).mean(axis=1) * itm

    # Rho: ∂S(tᵢ)/∂r = S(tᵢ) tᵢ; discount factor ∂/∂r also contributes −T·C
    rho_s = (
        df * (paths * t[None, :]).mean(axis=1) * itm
        - params.T * df * payoffs
    )

    n = n_paths
    elapsed = time.perf_counter() - t0
    return {
        "price": price,
        "delta": float(delta_s.mean()),
        "vega": float(vega_s.mean()),
        "rho": float(rho_s.mean()),
        "std_err_delta": float(delta_s.std(ddof=1) / np.sqrt(n)),
        "std_err_vega": float(vega_s.std(ddof=1) / np.sqrt(n)),
        "std_err_rho": float(rho_s.std(ddof=1) / np.sqrt(n)),
        "runtime_s": elapsed,
        "n_paths": n_paths,
    }


def lr_greeks(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
) -> dict:
    """Likelihood-ratio (score-function) delta and vega for the arithmetic Asian call.

    Differentiates the joint simulation density w.r.t. S₀ and σ rather than
    the payoff function.  This approach is valid even for discontinuous payoffs
    (e.g. digitals), where pathwise differentiation fails.  For the smooth
    Asian payoff it has higher variance than the pathwise estimator; the ratio
    ``std_err_lr / std_err_pathwise`` quantifies the cost of generality.

    The incremental path construction is used so that the standard normal
    draws ``z`` have a direct interpretation as Brownian increments.

    Score functions
    ---------------
    Delta score (only the first conditional density p(S(t₁)|S₀) depends on S₀
    when the path is held fixed; subsequent densities p(S(tᵢ)|S(t_{i-1})) treat
    S(t_{i-1}) as a fixed observed value):
        ∂/∂S₀ log p = z₁ / (S₀ · σ · √Δt₁)

    Vega score (two terms per step: the chi-squared term from the variance, plus
    a drift-correction from differentiating the GBM mean (r−σ²/2)Δtᵢ w.r.t. σ):
        ∂/∂σ log p  = Σᵢ [ (zᵢ² − 1) / σ − zᵢ √Δtᵢ ]

    Estimators:
        δ_LR = e^{-rT} E[ (A−K)⁺ · z₁ / (S₀ σ √Δt₁) ]
        ν_LR = e^{-rT} E[ (A−K)⁺ · Σᵢ ( (zᵢ²−1)/σ − zᵢ√Δtᵢ ) ]

    Returns a dict with keys ``delta``, ``vega``, ``std_err_delta``,
    ``std_err_vega``, ``runtime_s``, ``n_paths``.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "LR Greeks are implemented for calls only."
        )
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if params.sigma <= 0.0:
        raise ValueError(
            "lr_greeks requires sigma > 0 because the "
            "likelihood-ratio scores divide by sigma."
        )

    t0 = time.perf_counter()
    t = monitoring_times(params)
    dt = np.diff(np.concatenate(([0.0], t)))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, params.N))

    # Incremental construction: z[:, i] is the i-th normalised BM increment
    W = np.cumsum(np.sqrt(dt) * z, axis=1)
    log_paths = (
        np.log(params.S0)
        + (params.r - 0.5 * params.sigma ** 2) * t[None, :]
        + params.sigma * W
    )
    paths = np.exp(log_paths)

    A = paths.mean(axis=1)
    df = discount_factor(params)
    payoffs = np.maximum(A - params.K, 0.0)

    # Delta score: ∂/∂S₀ log p(S(t₁)|S₀) = z₁ / (S₀ σ √Δt₁)
    dt1 = float(dt[0])
    score_delta = z[:, 0] / (params.S0 * params.sigma * np.sqrt(dt1))
    delta_s = df * payoffs * score_delta

    # Vega score: ∂/∂σ log p = Σᵢ [(zᵢ²−1)/σ − zᵢ√Δtᵢ]
    # The second term arises because the GBM drift (r−σ²/2)Δtᵢ depends on σ.
    # Omitting it produces a biased estimator for vega.
    score_vega = (
        (z ** 2 - 1).sum(axis=1) / params.sigma
        - (np.sqrt(dt) * z).sum(axis=1)
    )
    vega_s = df * payoffs * score_vega

    n = n_paths
    elapsed = time.perf_counter() - t0
    return {
        "delta": float(delta_s.mean()),
        "vega": float(vega_s.mean()),
        "std_err_delta": float(delta_s.std(ddof=1) / np.sqrt(n)),
        "std_err_vega": float(vega_s.std(ddof=1) / np.sqrt(n)),
        "runtime_s": elapsed,
        "n_paths": n_paths,
    }


def fd_greeks(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    bump_pct: float = 0.01,
) -> dict:
    """Central finite-difference Greeks via antithetic+CV bump-and-reprice.

    Each Greek is computed by bumping one parameter up and down by
    ``bump_pct`` of its value, and taking the central difference.  All
    re-pricings use the same ``seed`` (common random numbers), which cancels
    most Monte Carlo noise and makes the bump signal detectable at a fraction
    of the paths needed by naive finite differences.

    Bump sizes:
        h_S0    = bump_pct × S0
        h_sigma = max(bump_pct × sigma, 0.001)
        h_r     = max(bump_pct × |r|, 0.001)

    Gamma uses three evaluations (base + up + down in S0):
        γ = (C(S₀+h) − 2C(S₀) + C(S₀−h)) / h²

    Returns a dict with keys ``base_price``, ``delta``, ``gamma``,
    ``vega``, ``rho``, ``runtime_s``, ``n_paths``, ``bump_pct``.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "FD Greeks are implemented for calls only."
        )

    t0 = time.perf_counter()

    def price(p: AsianOptionParams) -> float:
        return antithetic_cv_price(p, n_paths, seed=seed)["price"]

    h_s = bump_pct * params.S0
    h_sigma = max(bump_pct * params.sigma, 0.001)
    h_r = max(bump_pct * abs(params.r), 0.001)

    base = price(params)

    # Delta and Gamma: bump S0
    c_up_s = price(dataclasses.replace(params, S0=params.S0 + h_s))
    c_dn_s = price(dataclasses.replace(params, S0=params.S0 - h_s))
    delta = (c_up_s - c_dn_s) / (2.0 * h_s)
    gamma = (c_up_s - 2.0 * base + c_dn_s) / (h_s ** 2)

    # Vega: bump sigma
    vega = (
        price(dataclasses.replace(params, sigma=params.sigma + h_sigma))
        - price(dataclasses.replace(params, sigma=params.sigma - h_sigma))
    ) / (2.0 * h_sigma)

    # Rho: bump r
    rho = (
        price(dataclasses.replace(params, r=params.r + h_r))
        - price(dataclasses.replace(params, r=params.r - h_r))
    ) / (2.0 * h_r)

    elapsed = time.perf_counter() - t0
    return {
        "base_price": float(base),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "rho": float(rho),
        "runtime_s": elapsed,
        "n_paths": n_paths,
        "bump_pct": bump_pct,
    }
