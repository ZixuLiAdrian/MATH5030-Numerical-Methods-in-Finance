# asian-option-pricer

**Fast, accurate pricing of arithmetic Asian options under GBM — with variance reduction, generalized averaging windows, option Greeks, and adaptive estimator selection.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZixuLiAdrian/MATH5030-Numerical-Methods-in-Finance/blob/main/notebooks/demo.ipynb)

*MATH 5030 — Numerical Methods in Finance — Columbia University, Spring 2026*

---

Pricing an arithmetic Asian option requires simulation because the average of lognormal prices has no closed-form distribution.

This package implements analytic benchmarks together with Monte Carlo and Quasi-Monte Carlo estimators for discretely monitored Asian options under GBM. In addition to standard equally spaced averaging from time 0 to maturity T, the framework supports generalized averaging windows [T1, T2], allowing averaging to begin after inception — a common feature in commodity-linked contracts and structured products.

This package implements two analytic benchmarks, six MC/QMC estimators, an adaptive pilot-based selector, and three Greek estimators. Combining a **geometric-Asian control variate** with **Brownian-bridge Sobol QMC** reduces pricing error by up to **82×** versus plain Monte Carlo at the same computational cost.

---

## Innovations

While the core methods are classical, several aspects of this project go beyond textbook reproduction:

- **Continuous-limit cross-validation** — we independently verify that our discrete prices converge to the continuous arithmetic Asian price at the expected $O(1/N)$ rate, using PyFENG's `BsmAsianJsu` as an external benchmark. This is not a standard validation step.
- **Empirical variance reduction measurement** — the control variate estimators report a `var_reduction` key computed from the sample, making the speedup measurable rather than just claimed.
- **Combined antithetic + CV estimator** — `antithetic_cv_price` stacks antithetic sampling and the geometric control variate together, which standard references treat separately.
- **RMSE vs. CPU time efficiency study** — we measure root-mean-square error across 12 independent replications at five path budgets and plot against both paths and wall-clock time, giving a fairer comparison than RMSE vs. paths alone.
- **Generalized averaging windows** — the pricing framework supports arbitrary averaging intervals `[T1, T2]` instead of only `[0, T]`. All Monte Carlo, QMC, Brownian-bridge, and analytic benchmark implementations were generalized consistently and validated against covariance identities and Monte Carlo cross-checks.
- **Option Greeks via pathwise, likelihood-ratio, and finite-difference** — `pathwise_greeks` computes delta, vega, and rho via Infinitesimal Perturbation Analysis (IPA); `lr_greeks` computes delta and vega via the score-function method and quantifies its variance overhead relative to pathwise; `fd_greeks` adds gamma via central finite differences with common random numbers. The three-way comparison is methodologically instructive: pathwise is most efficient for the smooth Asian payoff, LR generalises to discontinuous payoffs at a variance cost, and FD is the only route to gamma without a second-order score estimator.
- **Pilot-based adaptive estimator selection** — `smart_price` reserves about 5% of the *total* path budget for a pilot phase, splits that pilot budget across the candidate estimators, selects the winner by efficiency ($1/\text{SE}^2/\text{runtime}$), then prices with the remaining budget. The returned `n_paths` now reflects the actual combined pilot + main path usage, so efficiency comparisons remain fair. The RQMC candidate uses multiple scrambles in the pilot to report a cross-replication SE (not the unreliable single-scramble proxy), making the comparison fair. The selector routes to different estimators across parameter regimes — for example switching away from the geometric control variate in deep-OTM cases where all payoffs are zero — turning the pricer into an adaptive algorithm rather than a fixed-method demonstration.

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

### Adaptive estimator

| Function | Extra keys | Description |
| --- | --- | --- |
| `smart_price(params, n_paths, pilot_fraction, n_pilot_replications, n_main_replications, seed)` | `selected_method`, `pilot_scores`, `pilot_n_paths`, `main_n_paths` | Pilot-benchmarks candidate estimators, selects the most efficient, prices with the remaining budget while honoring the total path budget. |

`pilot_fraction` (default 0.05) controls the total pilot budget. When the overall budget is large enough, the pilot reserves at least 512 total paths and leaves at least 64 for the main run. `pilot_scores` is a nested dict mapping each candidate to `{std_err, runtime_s, efficiency, n_paths}`; `pilot_n_paths` and `main_n_paths` report the realised pilot and main allocations, and their sum equals the returned `n_paths`.

### Option Greeks

All Greek functions accept calls only and return a `dict`.

| Function | Greeks returned | Method |
| --- | --- | --- |
| `pathwise_greeks(params, n_paths, seed, path_method)` | `delta`, `vega`, `rho` + per-Greek `std_err_*` | IPA — differentiates the payoff path-by-path. Lowest variance for smooth payoffs. |
| `lr_greeks(params, n_paths, seed)` | `delta`, `vega` + per-Greek `std_err_*` | Score-function — differentiates the simulation density. Valid for discontinuous payoffs; higher variance than pathwise for the Asian call. Requires `sigma > 0`. |
| `fd_greeks(params, n_paths, seed, bump_pct)` | `delta`, `gamma`, `vega`, `rho` | Central finite-difference with common random numbers. Only method providing gamma. |

---

## Results at a glance

Reference: `S0 = 100`, `K = 100`, `r = 5%`, `sigma = 30%`, `T = 1 year`, `N = 50`.

### Pricing — RMSE vs. plain Monte Carlo

RMSE measured across 12 independent replications at 524k paths; see `experiments/run_efficiency.py`.

| Method | RMSE @ 524k paths | Gain over plain MC |
| --- | ---: | ---: |
| Standard Monte Carlo | 1.9 × 10⁻² | 1× |
| Antithetic variates | 8.2 × 10⁻³ | ≈ 2.6× |
| Sobol QMC — incremental | 1.8 × 10⁻³ | ≈ 10× |
| Control variate (geometric Asian) | 7.3 × 10⁻⁴ | ≈ 26× |
| Antithetic + control variate | 7.8 × 10⁻⁴ | ≈ 24× |
| Randomised QMC — Brownian bridge | 3.5 × 10⁻⁴ | ≈ 54× |
| **Sobol QMC — Brownian bridge** | **2.3 × 10⁻⁴** | **≈ 82×** |
| `smart_price` (pilot-adaptive) | ≈ 7.9 × 10⁻⁴ | ≈ 24× *(selects antithetic+CV or control variate at ATM — both near-optimal)* |

The adaptive selector trails the single best fixed method because ≈5% of the budget is spent on the pilot. In exchange it is parameter-agnostic: it routes to the cheapest estimator automatically across moneyness and volatility regimes.

![Efficiency: RMSE vs paths and vs CPU time](results/figures/efficiency_rmse_vs_time.png)

### Option Greeks — pathwise vs. likelihood-ratio vs. finite-difference

`N_PATHS = 500 000` for pathwise and LR; `N_PATHS = 100 000` for finite-difference (5 re-pricings per Greek, common random numbers). Same canonical params as above.

| Greek | Pathwise (SE) | LR (SE) | Finite-diff | LR/PW variance ratio |
| --- | ---: | ---: | ---: | ---: |
| Delta (∂C/∂S₀) | 0.5727 (0.00079) | 0.5739 (0.00509) | 0.5718 | **41×** |
| Vega (∂C/∂σ) | 22.265 (0.062) | 22.271 (0.721) | 22.233 | **136×** |
| Rho (∂C/∂r) | 22.170 (0.031) | — | 22.148 | — |
| Gamma (∂²C/∂S₀²) | — | — | 0.02128 | — |

Key observations:
- All three methods agree within 1–2 SE (same expectation, different variance).
- The LR/PW variance ratio (41× for delta, 136× for vega) shows the efficiency cost of the general-purpose score-function approach for a payoff that is already smooth — pathwise IPA exploits that smoothness directly.
- Gamma is uniquely available from finite-difference; pathwise and LR would require second-order estimators.
- The large LR vega SE arises from the chi-squared score $\sum_i(z_i^2-1)/\sigma$, which has variance $2N/\sigma^2$ independently of the payoff; the drift-correction term $-z_i\sqrt{\Delta t_i}$ is required for unbiasedness (it accounts for the $\sigma$-dependence of the GBM drift) and adds negligible extra variance compared to the chi-squared term.

---

## Mathematical setup

Under the risk-neutral measure the underlying follows geometric Brownian motion

$$
\frac{dS_t}{S_t} = r\,dt + \sigma\,dW_t,\qquad S_0 \text{ given}
$$

For a general averaging window $[T_1,T_2]$, the monitoring dates are

$$
t_i = T_1 + i\frac{T_2-T_1}{N}, \qquad i=1,\dots,N.
$$

The arithmetic average is

$$
A_N = \frac{1}{N} \sum_{i=1}^N S(t_i).
$$

The Asian call payoff is

$$
(A_N - K)^+.
$$

The arbitrage-free price is

$$
C = e^{-rT} \mathbb{E} \left[(A_N-K)^+\right].
$$

Because the arithmetic average is a sum of correlated lognormal variables, no elementary closed form exists.

The auxiliary geometric average

$$
G_N = \left(\prod_{i=1}^N S(t_i)\right)^{1/N}
$$

remains lognormal even under generalized monitoring schedules, allowing an exact Kemna–Vorst-style closed form. We use this result both as an analytic benchmark and as the control mean in our variance reduction estimators.

The original textbook setting corresponds to

$$
T_1 = 0, \qquad T_2 = T.
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
│  ├─ analytic.py          # Kemna-Vorst + Levy closed forms
│  ├─ control_variate.py   # CV and antithetic+CV estimators
│  ├─ monte_carlo.py       # plain MC and antithetic MC
│  ├─ paths.py             # incremental + Brownian-bridge path construction
│  ├─ qmc.py               # Sobol QMC + RQMC
│  ├─ adaptive.py          # smart_price adaptive selector  ← new
│  ├─ greeks.py            # pathwise / LR / FD Greeks       ← new
│  ├─ models.py            # AsianOptionParams dataclass
│  ├─ utils.py             # monitoring_times, discount_factor
│  └─ benchmarks.py        # benchmark_suite convenience function
├─ experiments/
│  ├─ run_validation.py
│  ├─ run_efficiency.py
│  ├─ run_robustness.py
│  ├─ run_greeks.py        # Greeks comparison experiment     ← new
│  ├─ run_adaptive.py      # adaptive selection experiment    ← new
│  └─ _common.py
├─ tests/
├─ notebooks/demo.ipynb
├─ results/
│  ├─ tables/
│  │  ├─ greeks_comparison.csv   ← new
│  │  └─ adaptive_selection.csv  ← new
│  └─ figures/
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
6. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* (Ch. 7 for pathwise and likelihood-ratio Greeks.)
7. Broadie, M., & Glasserman, P. (1996). *Estimating security price derivatives using simulation.* Management Science, 42(2), 269–285.
