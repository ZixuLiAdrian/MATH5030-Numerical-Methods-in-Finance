"""Tests for the Greeks estimators."""

from __future__ import annotations

import pytest

from asian_option_pricer import AsianOptionParams, lr_greeks


def test_lr_greeks_rejects_zero_volatility():
    params = AsianOptionParams(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.0,
        T=1.0,
        N=12,
    )

    with pytest.raises(ValueError, match="sigma > 0"):
        lr_greeks(params, n_paths=1_024, seed=123)
