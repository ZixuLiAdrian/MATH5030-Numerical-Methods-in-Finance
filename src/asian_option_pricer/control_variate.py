"""Control-variate estimators using the geometric-average Asian as control.

The arithmetic and geometric payoffs on the same GBM path are strongly
positively correlated, and the geometric price has a Kemna-Vorst closed form
(:func:`asian_option_pricer.analytic.geometric_asian_call_price`). This makes
the geometric Asian an excellent control for the arithmetic price.

Two estimators are exposed:

* :func:`control_variate_price` -- single-sample CV with an optimal
  coefficient ``beta`` estimated from the sample covariance.
* :func:`antithetic_cv_price` -- combines antithetic sampling with the
  geometric control, stacking two orthogonal variance-reduction techniques.
"""
from __future__ import annotations

import time

import numpy as np

from .analytic import geometric_asian_call_price
from .models import AsianOptionParams
from .paths import (
    PathMethod,
    build_paths,
    geometric_payoff_from_paths,
    payoff_from_paths,
)
from .utils import discount_factor


def _discounted_payoff_pair(
    params: AsianOptionParams, paths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return discounted arithmetic and geometric payoffs for each path."""
    df = discount_factor(params)
    x = df * payoff_from_paths(paths, params.K, params.option_type)
    y = df * geometric_payoff_from_paths(paths, params.K, params.option_type)
    return x, y


def _optimal_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Sample estimates of the CV coefficient and variance reduction factor.

    Returns ``(beta, var_reduction)`` where
    ``beta = Cov(X,Y) / Var(Y)`` and
    ``var_reduction = Var(X) / Var(X - beta*(Y - E[Y]))``.
    Both are computed from a single pass using dot products.
    Returns ``(0.0, 1.0)`` when ``Var(Y) == 0``.
    """
    n = len(y)
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_y = float(np.dot(y_c, y_c) / (n - 1))
    if var_y <= 0.0:
        return 0.0, 1.0
    cov_xy = float(np.dot(x_c, y_c) / (n - 1))
    beta = cov_xy / var_y
    var_x = float(np.dot(x_c, x_c) / (n - 1))
    # Var(X - beta*Y) = Var(X) - 2*beta*Cov(X,Y) + beta^2*Var(Y)
    var_cv = var_x - 2 * beta * cov_xy + beta ** 2 * var_y
    var_reduction = var_x / var_cv if var_cv > 0.0 else float("inf")
    return float(beta), float(var_reduction)


def control_variate_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "incremental",
) -> dict:
    """Geometric-Asian control-variate estimator for the arithmetic call.

    The estimator is ``X - beta * (Y - E[Y])`` where ``X`` is the discounted
    arithmetic payoff, ``Y`` is the discounted geometric payoff, and
    ``E[Y]`` is computed in closed form. ``beta`` is estimated from the same
    sample, which introduces a negligible bias in return for a large variance
    reduction.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "Geometric control-variate is currently implemented for calls only."
        )
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, params.N))
    paths = build_paths(params, z, method=path_method)
    x, y = _discounted_payoff_pair(params, paths)
    mu_y = geometric_asian_call_price(params)
    beta, var_reduction = _optimal_beta(x, y)
    cv_samples = x - beta * (y - mu_y)
    elapsed = time.perf_counter() - t0
    return {
        "price": float(cv_samples.mean()),
        "std_err": float(cv_samples.std(ddof=1) / np.sqrt(n_paths)),
        "runtime_s": elapsed,
        "n_paths": n_paths,
        "beta": beta,
        "var_reduction": var_reduction,
    }


def antithetic_cv_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "incremental",
) -> dict:
    """Combined antithetic + geometric-CV estimator.

    For each of ``half = n_paths // 2`` normal draws ``z`` we simulate both
    ``z`` and ``-z`` paths, average the discounted arithmetic and geometric
    payoffs within the pair, and then apply the geometric control variate on
    the paired samples. This stacks two complementary variance reductions:
    antithetic cancels linear antisymmetry, the control variate removes the
    component explained by the geometric payoff.
    """
    params.validate()
    if params.option_type != "call":
        raise NotImplementedError(
            "Antithetic+CV estimator is currently implemented for calls only."
        )
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")

    half = n_paths // 2
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((half, params.N))
    paths_pos = build_paths(params, z, method=path_method)
    paths_neg = build_paths(params, -z, method=path_method)

    x_pos, y_pos = _discounted_payoff_pair(params, paths_pos)
    x_neg, y_neg = _discounted_payoff_pair(params, paths_neg)
    x_pair = 0.5 * (x_pos + x_neg)
    y_pair = 0.5 * (y_pos + y_neg)

    mu_y = geometric_asian_call_price(params)
    beta, var_reduction = _optimal_beta(x_pair, y_pair)
    cv_samples = x_pair - beta * (y_pair - mu_y)
    elapsed = time.perf_counter() - t0
    return {
        "price": float(cv_samples.mean()),
        "std_err": float(cv_samples.std(ddof=1) / np.sqrt(half)),
        "runtime_s": elapsed,
        "n_paths": 2 * half,
        "beta": beta,
        "var_reduction": var_reduction,
    }
