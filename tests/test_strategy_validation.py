"""Offline tests for parameter and out-of-sample validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategy_validation as app


def synthetic_prices(n: int = 420) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=n)
    t = np.arange(n)
    return pd.DataFrame({
        "AAA": 100 * np.exp(0.0007 * t + 0.04 * np.sin(t / 18)),
        "BBB": 100 * np.exp(0.0003 * t + 0.035 * np.sin(t / 11 + 1)),
        "CCC": 100 * np.exp(0.0005 * t + 0.02 * np.cos(t / 15)),
        "SPY": 100 * np.exp(0.0004 * t + 0.018 * np.sin(t / 20 + 0.3)),
    }, index=index)


def test_validate_parameter_grid_requires_unique_positive_integers() -> None:
    app.validate_parameter_grid([63, 126], [5], [1, 2])
    with pytest.raises(ValueError, match="positive integers"):
        app.validate_parameter_grid([0], [5], [1])
    with pytest.raises(ValueError, match="duplicates"):
        app.validate_parameter_grid([63, 63], [5], [1])


def test_parameter_sweep_separates_train_and_test_periods() -> None:
    prices = synthetic_prices()
    split = prices.index[300]
    grid, yearly = app.run_parameter_sweep(
        prices, test_start=split, benchmark="SPY",
        lookbacks=[30, 60], skip_recents=[5], top_ns=[1, 2], rebalance_freqs=["daily"],
        transaction_cost_bps=5, slippage_bps=5,
    )
    assert len(grid) == 4
    assert (grid["status"] == "OK").all()
    assert (pd.to_datetime(grid["train_end"]) < split).all()
    assert (pd.to_datetime(grid["test_start"]) >= split).all()
    assert (grid["train_observations"] >= 2).all()
    assert (grid["test_observations"] >= 2).all()
    assert {"year", "strategy_total_return", "benchmark_total_return"}.issubset(yearly.columns)
    assert set(yearly["lookback"].unique()) == {30, 60}


def test_parameter_sweep_reports_unusable_splits_instead_of_faking_metrics() -> None:
    prices = synthetic_prices(100)
    grid, yearly = app.run_parameter_sweep(
        prices, test_start=prices.index[60], benchmark="SPY",
        lookbacks=[80], skip_recents=[15], top_ns=[1],
    )
    assert len(grid) == 1
    assert grid.iloc[0]["status"] == "SKIPPED"
    assert "reason" in grid.columns
    assert yearly.empty


def test_parameter_sweep_rejects_missing_benchmark() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="Benchmark"):
        app.run_parameter_sweep(
            prices, test_start=prices.index[300], benchmark="QQQ",
            lookbacks=[30], skip_recents=[5], top_ns=[1],
        )


def test_cli_writes_sweep_and_yearly_outputs(tmp_path) -> None:
    prices = synthetic_prices(260)
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "strategy_validation.py"),
         "--prices", str(path), "--benchmark", "SPY", "--test-start", str(prices.index[180].date()),
         "--lookbacks", "30,45", "--skip-recents", "5", "--top-ns", "1,2",
         "--rebalance-freqs", "daily", "--output-dir", str(output_dir)],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads((output_dir / "strategy_validation_metadata.json").read_text())
    assert metadata["requested_combinations"] == 4
    assert metadata["successful_combinations"] == 4
    for filename in ("strategy_parameter_sweep.csv", "strategy_yearly_performance.csv", "strategy_validation_metadata.json"):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert '"requested_combinations"' in result.stdout


def test_parse_int_list_rejects_duplicates_and_nonpositive() -> None:
    assert app.parse_int_list("63,126") == [63, 126]
    with pytest.raises(Exception):
        app.parse_int_list("63,63")
    with pytest.raises(Exception):
        app.parse_int_list("0,126")


def test_parameter_sweep_includes_rebalance_frequency_as_grid_dimension() -> None:
    prices = synthetic_prices()
    grid, _ = app.run_parameter_sweep(
        prices, test_start=prices.index[300], benchmark="SPY",
        lookbacks=[30], skip_recents=[5], top_ns=[1],
        rebalance_freqs=["daily", "weekly", "monthly"],
    )
    assert len(grid) == 3
    assert set(grid["rebalance_freq"]) == {"daily", "weekly", "monthly"}
    with pytest.raises(ValueError, match="rebalance_freqs"):
        app.run_parameter_sweep(
            prices, test_start=prices.index[300], benchmark="SPY",
            lookbacks=[30], skip_recents=[5], top_ns=[1], rebalance_freqs=["quarterly"],
        )
