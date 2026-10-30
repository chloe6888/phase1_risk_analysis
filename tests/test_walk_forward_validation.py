"""Offline tests for rolling walk-forward validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import walk_forward_validation as app


def synthetic_prices(n: int = 240) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=n)
    t = np.arange(n)
    return pd.DataFrame({
        "AAA": 100 * np.exp(0.0008 * t + 0.05 * np.sin(t / 11)),
        "BBB": 100 * np.exp(0.0003 * t + 0.04 * np.sin(t / 17 + 0.7)),
        "CCC": 100 * np.exp(0.0005 * t + 0.03 * np.cos(t / 13)),
        "SPY": 100 * np.exp(0.0004 * t + 0.018 * np.sin(t / 20 + 0.2)),
    }, index=index)


def test_walk_forward_uses_non_overlapping_forward_test_windows() -> None:
    prices = synthetic_prices()
    folds, oos, summary = app.run_walk_forward(
        prices, benchmark="SPY", lookbacks=[20, 30], skip_recents=[3], top_ns=[1, 2],
        rebalance_freqs=["daily"], train_window=60, test_window=25, selection_metric="sharpe_ratio",
    )
    assert len(folds) >= 2
    assert (folds["status"] == "OK").all()
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["test_start"])).all()
    assert set(folds["selected_lookback"]).issubset({20, 30})
    assert set(folds["selected_top_n"]).issubset({1, 2})
    assert oos.index.is_unique
    assert oos.index.is_monotonic_increasing
    assert summary["oos_observations"] == len(oos)
    assert summary["folds_successful"] == len(folds)
    assert "out_of_sample_strategy" in summary
    assert "out_of_sample_benchmark" in summary


def test_walk_forward_rejects_short_history_and_bad_metric() -> None:
    prices = synthetic_prices(90)
    with pytest.raises(ValueError, match="at least"):
        app.run_walk_forward(
            prices, lookbacks=[40], skip_recents=[10], top_ns=[1],
            train_window=60, test_window=50,
        )
    with pytest.raises(ValueError, match="selection_metric"):
        app.run_walk_forward(
            synthetic_prices(), lookbacks=[20], skip_recents=[3], top_ns=[1],
            train_window=60, test_window=20, selection_metric="total_return",
        )


def test_walk_forward_rejects_missing_benchmark_and_bad_windows() -> None:
    with pytest.raises(ValueError, match="Benchmark"):
        app.run_walk_forward(
            synthetic_prices(), benchmark="QQQ", lookbacks=[20], skip_recents=[3], top_ns=[1],
            train_window=60, test_window=20,
        )
    with pytest.raises(ValueError, match="train_window"):
        app.run_walk_forward(
            synthetic_prices(), lookbacks=[20], skip_recents=[3], top_ns=[1],
            train_window=1, test_window=20,
        )


def test_cli_writes_walk_forward_outputs(tmp_path) -> None:
    prices = synthetic_prices(220)
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "walk_forward_validation.py"),
         "--prices", str(path), "--benchmark", "SPY", "--lookbacks", "20,30",
         "--skip-recents", "3", "--top-ns", "1,2", "--rebalance-freqs", "daily", "--train-window", "60",
         "--test-window", "20", "--selection-metric", "sharpe_ratio", "--output-dir", str(output_dir)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "walk_forward_summary.json").read_text())
    assert summary["requested_combinations"] == 4
    assert summary["folds_successful"] > 0
    for filename in ("walk_forward_folds.csv", "walk_forward_oos_returns.csv", "walk_forward_summary.json"):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert '"folds_successful"' in result.stdout


def test_walk_forward_selects_rebalance_frequency_inside_training_only() -> None:
    prices = synthetic_prices(260)
    folds, oos, summary = app.run_walk_forward(
        prices, benchmark="SPY", lookbacks=[20], skip_recents=[3], top_ns=[1],
        rebalance_freqs=["daily", "weekly", "monthly"],
        train_window=80, test_window=25,
    )
    assert summary["requested_combinations"] == 3
    assert set(folds.loc[folds["status"] == "OK", "selected_rebalance_freq"]).issubset(
        {"daily", "weekly", "monthly"}
    )
    assert "risk_free_return" in oos.columns
