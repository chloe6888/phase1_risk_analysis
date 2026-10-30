"""Offline regression tests for the reproducible cadence/stability report."""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy_stability_report as app


def synthetic_prices(n: int = 340) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=n)
    t = np.arange(n)
    return pd.DataFrame({
        "AAA": 100 * np.exp(0.0008 * t + 0.05 * np.sin(t / 11)),
        "BBB": 100 * np.exp(0.0003 * t + 0.04 * np.sin(t / 17 + 0.7)),
        "CCC": 100 * np.exp(0.0005 * t + 0.03 * np.cos(t / 13)),
        "SPY": 100 * np.exp(0.0004 * t + 0.018 * np.sin(t / 20 + 0.2)),
    }, index=index)


def test_stability_analysis_compares_three_cadences_with_same_baseline_signals() -> None:
    prices = synthetic_prices()
    result = app.run_stability_analysis(
        prices,
        test_start=prices.index[230],
        benchmark="SPY",
        baseline_lookback=20,
        baseline_skip_recent=3,
        baseline_top_n=1,
        lookbacks=[20, 30],
        skip_recents=[3],
        top_ns=[1, 2],
        rebalance_freqs=["daily", "weekly", "monthly"],
        train_window=80,
        test_window=25,
    )

    cadence = result["cadence_metrics"]
    assert set(cadence["rebalance_freq"]) == {"daily", "weekly", "monthly"}
    assert len(cadence) == 3
    assert cadence["observations"].nunique() == 1
    assert {"new_total_return", "legacy_total_return", "new_minus_legacy_total_return",
            "total_cost_drag", "terminal_liquidation_cost"}.issubset(cadence.columns)
    assert (cadence["terminal_liquidation_cost"] >= 0).all()

    grid = result["cadence_grid"]
    assert len(grid) == 2 * 1 * 2 * 3
    assert (grid["status"] == "OK").all()
    assert set(grid["rebalance_freq"]) == {"daily", "weekly", "monthly"}
    assert set(result["frequency_grid_summary"]["rebalance_freq"]) == {
        "daily", "weekly", "monthly"
    }

    folds = result["walk_forward_folds"]
    assert not folds.empty
    assert (folds["status"] == "OK").all()
    assert "selected_rebalance_freq" in folds.columns
    assert set(result["parameter_stability"]["parameter"]) == {
        "lookback", "skip_recent", "top_n", "rebalance_freq"
    }
    assert result["walk_forward_summary"]["folds_successful"] == len(folds)
    assert {"daily", "weekly", "monthly"}.issubset(result["daily_returns"].columns)
    assert {"daily_legacy", "weekly_legacy", "monthly_legacy"}.issubset(result["daily_returns"].columns)


def test_stability_report_writes_reproducible_artifacts_and_manifest(tmp_path) -> None:
    prices = synthetic_prices()
    price_path = tmp_path / "adjusted_close_panel.csv"
    prices.to_csv(price_path)
    output_dir = tmp_path / "stability"

    result = app.run_and_save_stability_report(
        price_path=price_path,
        output_dir=output_dir,
        test_start=str(prices.index[230].date()),
        benchmark="SPY",
        baseline_lookback=20,
        baseline_skip_recent=3,
        baseline_top_n=1,
        lookbacks=[20],
        skip_recents=[3],
        top_ns=[1],
        rebalance_freqs=["daily", "weekly", "monthly"],
        train_window=80,
        test_window=25,
    )

    expected = [
        "cadence_comparison.csv",
        "cadence_daily_returns.csv",
        "cadence_equity_curves.csv",
        "strategy_parameter_sweep.csv",
        "strategy_yearly_performance.csv",
        "frequency_grid_summary.csv",
        "walk_forward_folds.csv",
        "walk_forward_oos_returns.csv",
        "walk_forward_summary.json",
        "walk_forward_parameter_stability.csv",
        "stability_run_manifest.json",
        "stability_report.md",
    ]
    for filename in expected:
        assert (output_dir / filename).is_file()
        assert (output_dir / filename).stat().st_size > 0

    manifest = pd.read_json(output_dir / "stability_run_manifest.json", typ="series")
    assert manifest["price_panel_sha256"]
    assert manifest["price_panel_rows"] == len(prices)
    assert manifest["test_start"] == str(prices.index[230].date())
    assert result["cadence_metrics"].shape[0] == 3


def test_parameter_stability_table_counts_only_successful_folds() -> None:
    folds = pd.DataFrame({
        "status": ["OK", "OK", "SKIPPED"],
        "selected_lookback": [63, 126, None],
        "selected_skip_recent": [5, 5, None],
        "selected_top_n": [1, 2, None],
        "selected_rebalance_freq": ["weekly", "weekly", None],
    })
    result = app.parameter_stability_table(folds)
    weekly = result[(result["parameter"] == "rebalance_freq") & (result["value"] == "weekly")]
    assert weekly.iloc[0]["folds_selected"] == 2
    assert weekly.iloc[0]["selection_pct"] == 100.0


def test_stability_analysis_can_skip_walk_forward_explicitly() -> None:
    prices = synthetic_prices()
    result = app.run_stability_analysis(
        prices,
        test_start=prices.index[230],
        benchmark="SPY",
        baseline_lookback=20,
        baseline_skip_recent=3,
        baseline_top_n=1,
        lookbacks=[20],
        skip_recents=[3],
        top_ns=[1],
        rebalance_freqs=["weekly"],
        skip_walk_forward=True,
    )
    assert result["walk_forward_summary"]["status"] == "SKIPPED_BY_USER"
    assert result["parameter_stability"].empty
