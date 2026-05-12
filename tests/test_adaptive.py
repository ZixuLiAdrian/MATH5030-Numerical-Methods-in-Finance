"""Tests for the adaptive estimator selector."""

from __future__ import annotations

from asian_option_pricer import AsianOptionParams, smart_price


def test_smart_price_respects_total_path_budget():
    params = AsianOptionParams(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=52,
    )

    requested_n_paths = 5_000
    out = smart_price(
        params,
        n_paths=requested_n_paths,
        seed=123,
    )

    assert out["selected_method"] in out["pilot_scores"]
    assert out["pilot_n_paths"] >= 0
    assert out["main_n_paths"] >= 0
    assert out["pilot_n_paths"] + out["main_n_paths"] == out["n_paths"]
    assert out["n_paths"] <= requested_n_paths

    pilot_total = sum(
        score["n_paths"]
        for score in out["pilot_scores"].values()
    )
    assert pilot_total == out["pilot_n_paths"]


def test_smart_price_small_budget_still_reports_honestly():
    params = AsianOptionParams(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        N=12,
    )

    requested_n_paths = 16
    out = smart_price(
        params,
        n_paths=requested_n_paths,
        seed=7,
    )

    assert out["price"] >= 0.0
    assert out["std_err"] >= 0.0
    assert out["n_paths"] <= requested_n_paths
    assert out["pilot_n_paths"] + out["main_n_paths"] == out["n_paths"]

