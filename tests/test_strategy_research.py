"""Offline tests for multi-strategy comparison and overfitting audit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategy_research as app


def synthetic_prices(n: int = 520) -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=n)
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "AAA": 80 * np.exp(0.0008 * t + 0.05 * np.sin(t / 13)),
            "BBB": 100 * np.exp(0.0005 * t + 0.04 * np.sin(t / 19 + 0.7)),
            "CCC": 120 * np.exp(0.0003 * t + 0.06 * np.cos(t / 17)),
            "DDD": 95 * np.exp(0.0006 * t + 0.03 * np.sin(t / 9 + 1.1)),
            "SPY": 100 * np.exp(0.0004 * t + 0.018 * np.sin(t / 21 + 0.2)),
        },
        index=index,
    )


@pytest.mark.parametrize("strategy", app.SUPPORTED_STRATEGIES)
def test_all_strategies_generate_valid_long_only_weights(strategy: str) -> None:
    weights = app.build_strategy_weights(
        synthetic_prices(), strategy=strategy, benchmark="SPY",
        lookback=40, skip_recent=5, top_n=2,
    )
    assert weights.index.is_monotonic_increasing
    assert "SPY" not in weights.columns
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights.to_numpy() >= 0).all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (weights.sum(axis=1) > 0).any()


def test_strategy_signals_do_not_use_future_prices() -> None:
    prices = synthetic_prices()
    changed = prices.copy()
    cutoff = prices.index[300]
    changed.loc[changed.index > cutoff, ["AAA", "BBB", "CCC", "DDD"]] *= 1.7
    for strategy in app.SUPPORTED_STRATEGIES:
        original = app.build_strategy_weights(
            prices, strategy=strategy, benchmark="SPY", lookback=50,
            skip_recent=5, top_n=2,
        )
        modified = app.build_strategy_weights(
            changed, strategy=strategy, benchmark="SPY", lookback=50,
            skip_recent=5, top_n=2,
        )
        pd.testing.assert_frame_equal(
            original.loc[:cutoff], modified.loc[:cutoff],
            check_exact=False, atol=1e-12, rtol=1e-12,
        )


def test_build_weights_rejects_invalid_strategy_and_bad_inputs() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="Unsupported strategy"):
        app.build_strategy_weights(prices, strategy="black_box", benchmark="SPY")
    with pytest.raises(ValueError, match="Benchmark"):
        app.build_strategy_weights(prices.drop(columns=["SPY"]), strategy="equal_weight")
    with pytest.raises(ValueError, match="lookback"):
        app.build_strategy_weights(prices, strategy="low_volatility", lookback=0)


def test_deflated_sharpe_is_bounded_and_more_trials_raise_selection_bar() -> None:
    rng = np.random.default_rng(19)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    one_trial = app.calculate_deflated_sharpe_ratio(returns, n_trials=1)
    many_trials = app.calculate_deflated_sharpe_ratio(returns, n_trials=20)
    assert 0 <= one_trial["deflated_sharpe_ratio"] <= 1
    assert 0 <= many_trials["deflated_sharpe_ratio"] <= 1
    assert (
        many_trials["expected_max_daily_sharpe_under_null"]
        > one_trial["expected_max_daily_sharpe_under_null"]
    )
    assert many_trials["deflated_sharpe_ratio"] <= one_trial["deflated_sharpe_ratio"]


def test_deflated_sharpe_rejects_invalid_trials_and_short_history() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        app.calculate_deflated_sharpe_ratio(pd.Series([0.01, -0.01]), n_trials=0)
    with pytest.raises(ValueError, match="At least two"):
        app.calculate_deflated_sharpe_ratio(pd.Series([0.01]), n_trials=1)


def test_strategy_research_selects_using_training_only_and_keeps_holdout_separate() -> None:
    prices = synthetic_prices()
    split = prices.index[360]
    strategies = [
        "equal_weight", "cross_sectional_momentum", "time_series_momentum",
        "trend_following", "mean_reversion", "low_volatility",
    ]
    comparison, summary, selected_oos = app.run_strategy_research(
        prices, test_start=split, benchmark="SPY", strategies=strategies,
        lookback=50, skip_recent=5, top_n=2, rebalance_freq="weekly",
        transaction_cost_bps=5, slippage_bps=5,
    )
    assert len(comparison) == len(strategies)
    assert comparison["selected_on_training_only"].sum() == 1
    winner = comparison.loc[comparison["selected_on_training_only"]].iloc[0]
    assert winner["strategy"] == summary["selected_strategy"]
    assert pd.Timestamp(summary["train_end"]) < split
    assert pd.Timestamp(summary["test_start"]) >= split
    assert set(selected_oos.columns) >= {
        "strategy_return", "benchmark_return", "risk_free_return",
        "excess_return_vs_benchmark",
    }
    assert (selected_oos.index >= split).all()
    assert len(selected_oos) == summary["test_observations"]
    assert summary["selection_metric"] == "training_period_sharpe_ratio"
    assert summary["deflated_sharpe_audit_training_only"]["n_trials"] == len(strategies)


def test_strategy_research_rejects_duplicate_candidates_and_tiny_split() -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="unique"):
        app.run_strategy_research(
            prices, test_start=prices.index[300], strategies=["equal_weight", "equal_weight"]
        )
    with pytest.raises(ValueError, match="two common"):
        app.run_strategy_research(
            prices, test_start=prices.index[1], strategies=["equal_weight"]
        )


def test_cli_writes_comparison_audit_and_holdout_artifacts(tmp_path: Path) -> None:
    prices = synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)
    output_dir = tmp_path / "artifacts"
    script = Path(__file__).resolve().parents[1] / "strategy_research.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--prices", str(price_path),
            "--benchmark", "SPY", "--test-start", str(prices.index[360].date()),
            "--lookback", "50", "--skip-recent", "5", "--top-n", "2",
            "--rebalance-freq", "weekly", "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "strategy_research_summary.json").read_text())
    assert summary["candidate_count"] == len(app.SUPPORTED_STRATEGIES)
    assert summary["selected_strategy"] in app.SUPPORTED_STRATEGIES
    for filename in (
        "strategy_comparison.csv", "strategy_selected_oos_returns.csv",
        "strategy_research_summary.json", "strategy_research_report.md",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert "selected_strategy" in result.stdout


def test_parse_strategy_list_rejects_unknown_and_duplicate_names() -> None:
    assert app.parse_strategy_list("equal_weight,low_volatility") == [
        "equal_weight", "low_volatility"
    ]
    with pytest.raises(Exception):
        app.parse_strategy_list("equal_weight,equal_weight")
    with pytest.raises(Exception):
        app.parse_strategy_list("made_up")
