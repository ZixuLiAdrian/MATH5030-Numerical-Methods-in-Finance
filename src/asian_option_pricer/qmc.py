"""Quasi-Monte Carlo estimators based on scrambled Sobol sequences.

Two estimators are provided:

* :func:`sobol_qmc_price` -- a single scrambled Sobol sample. The reported
  ``std_err`` is a *proxy* (sample std / sqrt(n)) that ignores the QMC
  dependence structure; it is useful as a rough indicator only.

* :func:`rqmc_sobol_price` -- randomised QMC, which runs several
  independently scrambled Sobol replications and reports an unbiased
  cross-replication standard error. This is the theoretically correct way to
  quote a confidence interval for a QMC estimator (Owen 1997).

Both estimators accept ``path_method="brownian_bridge"``, which places the
terminal/coarse-scale Brownian movements on the low-index Sobol coordinates
(where the sequence has the best equidistribution) and typically reduces the
integrand's effective dimension.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm, qmc

from .models import AsianOptionParams
from .paths import PathMethod, build_paths, payoff_from_paths
from .utils import discount_factor


_PPF_CLIP = 1e-12


def _round_up_pow2(n: int) -> int:
    """Return the smallest power of two that is >= max(n, 2)."""
    return int(2 ** np.ceil(np.log2(max(n, 2))))


def _sobol_normals(d: int, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` scrambled Sobol points in ``d`` dimensions, mapped to
    standard normals via the inverse CDF. ``n`` should be a power of two for
    Sobol balance properties to hold."""
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random(n)
    # Guard against 0 and 1 which would map to +/- inf under the inverse CDF.
    u = np.clip(u, _PPF_CLIP, 1.0 - _PPF_CLIP)
    return norm.ppf(u)


def sobol_qmc_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "brownian_bridge",
) -> dict:
    """Single-replication scrambled Sobol QMC estimator.

    ``n_paths`` is rounded up to the next power of two. Defaults to
    ``path_method='brownian_bridge'`` because that is the construction where
    Sobol shows its largest advantage for Asian-style path functionals.
    """
    params.validate()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    t0 = time.perf_counter()
    n_qmc = _round_up_pow2(n_paths)
    z = _sobol_normals(params.N, n_qmc, seed=seed)
    paths = build_paths(params, z, method=path_method)
    pv = discount_factor(params) * payoff_from_paths(paths, params.K, params.option_type)
    elapsed = time.perf_counter() - t0
    return {
        "price": float(pv.mean()),
        # Proxy SE: classical MC formula applied to QMC samples. Under-states
        # precision for smooth integrands and over-states it for integrands
        # with kinks, so treat as informational.
        "std_err": float(pv.std(ddof=1) / np.sqrt(n_qmc)),
        "runtime_s": elapsed,
        "n_paths": n_qmc,
        "path_method": path_method,
    }


def rqmc_sobol_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    n_replications: int = 16,
    path_method: PathMethod = "brownian_bridge",
) -> dict:
    """Randomised QMC with multiple independent scrambles.

    Runs ``n_replications`` independently scrambled Sobol sets, each of size
    ``ceil_pow2(n_paths / n_replications)`` points, and reports the
    cross-replication mean and standard error. This yields an unbiased
    confidence interval, in contrast to the single-scramble ``std_err`` from
    :func:`sobol_qmc_price`.
    """
    params.validate()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2 to estimate a standard error.")

    per_rep = _round_up_pow2(max(n_paths // n_replications, 1))
    t0 = time.perf_counter()
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(n_replications)
    df = discount_factor(params)

    replicate_means = np.empty(n_replications)
    for k, child_seed in enumerate(child_seeds):
        z = _sobol_normals(params.N, per_rep, seed=int(child_seed))
        paths = build_paths(params, z, method=path_method)
        pv = df * payoff_from_paths(paths, params.K, params.option_type)
        replicate_means[k] = pv.mean()

    elapsed = time.perf_counter() - t0
    price = float(replicate_means.mean())
    std_err = float(replicate_means.std(ddof=1) / np.sqrt(n_replications))
    return {
        "price": price,
        "std_err": std_err,
        "runtime_s": elapsed,
        "n_paths": per_rep * n_replications,
        "n_replications": n_replications,
        "path_method": path_method,
    }
