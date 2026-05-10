# asian-option-pricer

**Fast, accurate pricing of arithmetic Asian options under GBM — with variance reduction and generalized averaging windows.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZixuLiAdrian/MATH5030-Numerical-Methods-in-Finance/blob/main/notebooks/demo.ipynb)

*MATH 5030 — Numerical Methods in Finance — Columbia University, Spring 2026*

---

Pricing an arithmetic Asian option requires simulation because the average of lognormal prices has no closed-form distribution.

This package implements analytic benchmarks together with Monte Carlo and Quasi-Monte Carlo estimators for discretely monitored Asian options under GBM. In addition to standard equally spaced averaging from time 0 to maturity T, the framework now supports generalized averaging windows [T1, T2], allowing averaging to begin after inception — a common feature in commodity-linked contracts and structured products.

This package implements two analytic benchmarks and six Monte Carlo / Quasi-Monte Carlo estimators, and shows that combining a **geometric-Asian control variate** with **Brownian-bridge Sobol QMC** reduces pricing error by up to **82×** versus plain Monte Carlo at the same computational cost.

---

## Innovations

While the core methods are classical, several aspects of this project go beyond textbook reproduction:

- **Continuous-limit cross-validation** — we independently verify that our discrete prices converge to the continuous arithmetic Asian price at the expected $O(1/N)$ rate, using PyFENG's `BsmAsianJsu` as an external benchmark. This is not a standard validation step.
- **Empirical variance reduction measurement** — the control variate estimators report a `var_reduction` key computed from the sample, making the speedup measurable rather than just claimed.
- **Combined antithetic + CV estimator** — `antithetic_cv_price` stacks antithetic sampling and the geometric control variate together, which standard references treat separately.
- **RMSE vs. CPU time efficiency study** — we measure root-mean-square error across 12 independent replications at five path budgets and plot against both paths and wall-clock time, giving a fairer comparison than RMSE vs. paths alone.
- **Generalized averaging windows** — the pricing framework supports arbitrary averaging intervals `[T1, T2]` instead of only `[0, T]`. All Monte Carlo, QMC, Brownian-bridge, and analytic benchmark implementations were generalized consistently and validated against covariance identities and Monte Carlo cross-checks.

---

## Installation

```bash
pip install asian-option-pricer
```

Or from source:

```bash
git clone https://github.com/ZixuLiAdrian/MATH5030-Numerical-Methods-in-Finance
cd MATH5030-Numerical-Methods-in-Finance
pip install -e ".[test]"
```

---

## Quick start

```python
from asian_option_pricer import (
    AsianOptionParams,
    antithetic_cv_price,
    rqmc_sobol_price,
)

# Standard averaging window [0, T]
params = AsianOptionParams(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.30,
    T=1.0,
    N=50,
)

print(
    antithetic_cv_price(
        params,
        n_paths=100_000,
    )
)

# Delayed averaging window [T1, T2]
delayed_params = AsianOptionParams(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.30,
    T=1.0,
    N=50,
    T1=0.25,
    T2=1.0,
)

print(
    rqmc_sobol_price(
        delayed_params,
        n_paths=131_072,
        n_replications=16,
        path_method="brownian_bridge",
    )
)
```

---

## API reference

### `AsianOptionParams`

Frozen dataclass holding all option parameters. Validated automatically by every pricing function.

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Initial stock price |
| `K` | `float` | Strike price (> 0) |
| `r` | `float` | Risk-free rate |
| `sigma` | `float` | Volatility (≥ 0) |
| `T` | `float` | Time to maturity in years (> 0) |
| `N` | `int` | Number of monitoring dates (≥ 1) |
| `option_type` | `str` | `"call"` (default) or `"put"` |
| `T1` | `float` | Averaging start time. Default `0.0`. |
| `T2` | `float \| None` | Averaging end time. Default `T`. |

### Analytic benchmarks

| Function | Returns | Description |
| --- | --- | --- |
| `geometric_asian_call_price(params)` | `float` | Exact Kemna–Vorst closed form for the discrete geometric Asian call under arbitrary monitoring schedules. |
| `levy_approx_call_price(params)` | `float` | Levy (1992) lognormal moment-matching approximation. Accurate for `sigma ≲ 0.3`. |

### MC / QMC estimators

All estimators return a `dict` with at minimum `{price, std_err, runtime_s, n_paths}`.

| Function | Extra keys | Description |
| --- | --- | --- |
| `standard_mc_price(params, n_paths, seed, path_method)` | — | Plain Monte Carlo. |
| `antithetic_mc_price(params, n_paths, seed, path_method)` | — | Antithetic variates; pairs each draw `z` with `−z`. |
| `control_variate_price(params, n_paths, seed, path_method)` | `beta`, `var_reduction` | Geometric-Asian control variate with sample-estimated `beta`. |
| `antithetic_cv_price(params, n_paths, seed, path_method)` | `beta`, `var_reduction` | Antithetic + control variate combined. |
| `sobol_qmc_price(params, n_paths, seed, path_method)` | — | Single scrambled Sobol replication. `std_err` is a proxy only. |
| `rqmc_sobol_price(params, n_paths, seed, n_replications, path_method)` | `n_replications` | Multiple scrambles; reports an honest cross-replication SE. |

**`path_method`** — `"incremental"` (default) or `"brownian_bridge"`.

The bridge construction concentrates path variance on the best-equidistributed Sobol coordinates and is recommended for QMC estimators. The implementation supports arbitrary monitoring schedules induced by delayed averaging windows `[T1, T2]`.

---

## Mathematical setup

Under the risk-neutral measure the underlying follows geometric Brownian motion

$$
\frac{dS_t}{S_t}
=
r\,dt + \sigma\,dW_t,
\qquad
S_0 \text{ given}.
$$

For a general averaging window $[T_1,T_2]$, the monitoring dates are

$$
t_i
=
T_1
+
i\frac{T_2-T_1}{N},
\qquad
i=1,\dots,N.
$$

The arithmetic average is

$$
A_N
=
\frac{1}{N}
\sum_{i=1}^N S(t_i).
$$

The Asian call payoff is

$$
(A_N - K)^+.
$$

The arbitrage-free price is

$$
C
=
e^{-rT}
\mathbb{E}
\left[
(A_N-K)^+
\right].
$$

Because the arithmetic average is a sum of correlated lognormal variables, no elementary closed form exists.

The auxiliary geometric average

$$
G_N
=
\left(
\prod_{i=1}^N S(t_i)
\right)^{1/N}
$$

remains lognormal even under generalized monitoring schedules, allowing an exact Kemna–Vorst-style closed form.

The original textbook setting corresponds to

$$
T_1 = 0,
\qquad
T_2 = T.
$$

Backward compatibility with that setting is fully preserved.

---

## Validation

### Delayed averaging validation

The generalized averaging framework `[T1, T2]` was validated through:

- covariance reconstruction tests for the generalized Brownian bridge,
- Monte Carlo versus closed-form comparisons for geometric Asian options,
- deterministic zero-volatility edge cases,
- and distributional agreement tests between incremental and bridge path constructions.

All tests passed within Monte Carlo error tolerance.

---

## Robustness

### Edge cases

- `sigma → 0`: all MC/QMC estimators converge to the deterministic intrinsic value.
- Deep OTM (`K ∈ {200, 400}`): every estimator returns exactly `0.0`.
- Dense monitoring (`N = 500`): variance-reduced estimators agree within a few basis points.
- Delayed averaging (`T1 > 0`): all MC/QMC estimators, Brownian-bridge constructions, and analytic benchmarks remain stable and internally consistent.

---

## Repository layout

```text
MATH5030-Numerical-Methods-in-Finance/
├─ src/asian_option_pricer/
│  ├─ analytic.py
│  ├─ control_variate.py
│  ├─ monte_carlo.py
│  ├─ paths.py
│  ├─ qmc.py
│  ├─ models.py
│  ├─ utils.py
│  └─ benchmarks.py
├─ experiments/
├─ tests/
├─ notebooks/demo.ipynb
├─ results/
├─ pyproject.toml
├─ LICENSE
└─ README.md
```

---

## References

1. Kemna, A. G. Z., & Vorst, A. C. F. (1990). *A pricing method for options based on average asset values.*
2. Levy, E. (1992). *Pricing European average rate currency options.*
3. Turnbull, S. M., & Wakeman, L. M. (1991). *A quick algorithm for pricing European average options.*
4. Caflisch, R. E., Morokoff, W. J., & Owen, A. B. (1997). *Valuation of mortgage-backed securities using Brownian bridges to reduce effective dimension.*
5. Owen, A. B. (1997). *Scrambled net variance for integrals of smooth functions.*
6. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.*
