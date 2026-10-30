"""Offline tests for constrained portfolio optimization."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer as app


def synthetic_prices(n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    index = pd.bdate_range("2022-01-03", periods=n)
    market = rng.normal(0.0004, 0.009, n)
    defensive = rng.normal(0.00025, 0.004, n)
    growth = rng.normal(0.0007, 0.016, n)
    return pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(1 + market + rng.normal(0, 0.003, n)),
            "BBB": 100 * np.cumprod(1 + market * 0.85 + defensive),
            "CCC": 100 * np.cumprod(1 + growth),
            "DDD": 100 * np.cumprod(1 + market * 0.5 + rng.normal(0, 0.006, n)),
        },
        index=index,
    )


def test_projection_respects_budget_asset_caps_and_group_caps() -> None:
    group_indices = {
        "growth": np.array([0, 1]),
        "defensive": np.array([2, 3]),
    }
    projected = app._project_group_capped_budget(
        np.array([0.8, 0.7, 0.6, 0.5]),
        asset_caps=np.array([0.4, 0.4, 0.4, 0.4]),
        group_indices=group_indices,
        group_caps={"growth": 0.45, "defensive": 0.5},
        budget=0.9,
    )
    assert projected.sum() == pytest.approx(0.9, abs=1e-7)
    assert (projected >= 0).all()
    assert (projected <= 0.4 + 1e-8).all()
    assert projected[:2].sum() <= 0.45 + 1e-7
    assert projected[2:].sum() <= 0.5 + 1e-7


def test_minimum_variance_optimizer_respects_constraints_and_risk_accounting() -> None:
    prices = synthetic_prices()
    weights, summary = app.optimize_portfolio(
        prices,
        risky_budget=0.9,
        default_asset_max_weight=0.6,
        groups={"AAA": "growth", "BBB": "growth", "CCC": "defensive", "DDD": "defensive"},
        group_max_weights={"growth": 0.45, "defensive": 0.6},
        minimum_observations=100,
    )
    assert summary["convergence"]["converged"] is True
    assert weights["weight"].sum() == pytest.approx(0.9, abs=1e-7)
    assert (weights["weight"] >= 0).all()
    assert (weights["weight"] <= weights["asset_max_weight"] + 1e-8).all()
    exposures = weights.groupby("group")["weight"].sum().to_dict()
    assert exposures["growth"] <= 0.45 + 1e-7
    assert exposures["defensive"] <= 0.6 + 1e-7
    assert weights["risk_contribution"].sum() == pytest.approx(1.0, abs=1e-6)
    assert summary["cash_weight"] == pytest.approx(0.1)
    assert "volatility_reduction_vs_constrained_equal_weight" in summary


def test_minimum_variance_is_not_worse_than_same_constraints_equal_weight() -> None:
    weights, summary = app.optimize_portfolio(
        synthetic_prices(), risky_budget=0.9, default_asset_max_weight=0.6,
        minimum_observations=100, tolerance=1e-8,
    )
    assert (
        summary["optimized_portfolio"]["variance_annualized"]
        <= summary["constrained_equal_weight_baseline"]["variance_annualized"] + 1e-7
    )
    assert weights["weight"].sum() == pytest.approx(0.9, abs=1e-7)


def test_mean_variance_objective_uses_historical_return_weight() -> None:
    _, summary = app.optimize_portfolio(
        synthetic_prices(), risky_budget=0.9, default_asset_max_weight=0.6,
        objective="mean_variance", return_weight=0.02, minimum_observations=100,
    )
    assert summary["objective"] == "mean_variance"
    assert summary["return_weight"] == pytest.approx(0.02)


def test_optimizer_rejects_infeasible_asset_and_group_caps() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="infeasible"):
        app.optimize_portfolio(
            prices, risky_budget=0.9, default_asset_max_weight=0.1,
            minimum_observations=100,
        )
    with pytest.raises(ValueError, match="infeasible"):
        app.optimize_portfolio(
            prices, risky_budget=0.9, default_asset_max_weight=0.6,
            groups={"AAA": "A", "BBB": "A", "CCC": "B", "DDD": "B"},
            group_max_weights={"A": 0.2, "B": 0.2},
            minimum_observations=100,
        )


def test_optimizer_rejects_invalid_constraints_and_short_history() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="default_asset_max_weight"):
        app.optimize_portfolio(prices, default_asset_max_weight=0)
    with pytest.raises(ValueError, match="objective"):
        app.optimize_portfolio(prices, objective="max_sharpe")
    with pytest.raises(ValueError, match="Only"):
        app.optimize_portfolio(prices.iloc[:40], minimum_observations=60)


def test_cli_writes_optimizer_artifacts(tmp_path: Path) -> None:
    prices = synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)
    asset_caps_path = tmp_path / "asset_caps.json"
    asset_caps_path.write_text(json.dumps({"AAA": 0.6}), encoding="utf-8")
    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps({"AAA": "growth", "BBB": "growth"}), encoding="utf-8")
    group_caps_path = tmp_path / "group_caps.json"
    group_caps_path.write_text(json.dumps({"growth": 0.7}), encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    script = Path(__file__).resolve().parents[1] / "portfolio_optimizer.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--prices", str(price_path),
            "--risky-budget", "0.9", "--asset-max-weight", "0.6",
            "--asset-caps-json", str(asset_caps_path),
            "--groups-json", str(group_map_path),
            "--group-caps-json", str(group_caps_path),
            "--minimum-observations", "100", "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "portfolio_optimization_summary.json").read_text())
    assert summary["convergence"]["converged"] is True
    for filename in (
        "portfolio_optimized_weights.csv",
        "portfolio_optimization_summary.json",
        "portfolio_optimization_report.md",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert "optimized_portfolio" in result.stdout


def test_asset_and_group_maps_reject_unknown_entries() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="unknown tickers"):
        app.optimize_portfolio(
            prices, asset_max_weights={"UNKNOWN": 0.1}, minimum_observations=100
        )
    with pytest.raises(ValueError, match="unknown groups"):
        app.optimize_portfolio(
            prices, groups={"AAA": "growth"}, group_max_weights={"missing": 0.5},
            minimum_observations=100,
        )
