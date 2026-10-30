"""Offline tests for the v1 strategy backtest engine."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import backtest_engine as app


def prices_for_returns(returns: dict[str, list[float]]) -> pd.DataFrame:
    return_frame = pd.DataFrame(returns, index=pd.bdate_range("2024-01-01", periods=len(next(iter(returns.values())))))
    return 100.0 * (1.0 + return_frame).cumprod()


def test_momentum_signal_selects_top_n_and_excludes_benchmark() -> None:
    index = pd.bdate_range("2023-01-02", periods=180)
    fast = pd.Series(np.linspace(100, 220, len(index)), index=index)
    slow = pd.Series(np.linspace(100, 130, len(index)), index=index)
    benchmark = pd.Series(np.linspace(100, 120, len(index)), index=index)
    prices = pd.DataFrame({"FAST": fast, "SLOW": slow, "SPY": benchmark})
    weights = app.generate_cross_sectional_momentum_weights(prices, lookback=60, skip_recent=10, top_n=1, benchmark="SPY")
    assert (weights["FAST"] == 1.0).sum() > 0
    assert (weights["SLOW"] == 1.0).sum() == 0
    assert (weights["SPY"] == 0.0).all() if "SPY" in weights.columns else True
    assert (weights.sum(axis=1) <= 1.0 + 1e-12).all()


def test_momentum_signal_requires_valid_parameters_and_benchmark() -> None:
    prices = pd.DataFrame({"AAA": [100, 101, 102], "SPY": [100, 100, 101]}, index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="lookback"):
        app.generate_cross_sectional_momentum_weights(prices, lookback=0)
    with pytest.raises(ValueError, match="absent"):
        app.generate_cross_sectional_momentum_weights(prices, benchmark="QQQ")


def test_backtest_lags_signal_by_one_period_to_avoid_lookahead() -> None:
    returns = {"AAA": [0.0, 0.10, -0.10, 0.0], "SPY": [0.0, 0.01, 0.01, 0.01]}
    prices = prices_for_returns(returns)
    weights = pd.DataFrame({"AAA": [1.0, 0.0, 0.0, 0.0]}, index=prices.index)
    summary, daily, applied, equity = app.run_backtest(
        prices, weights, benchmark="SPY", transaction_cost_bps=0, slippage_bps=0,
    )
    # A close-time signal is lagged: the target at day 0 is applied to day 1's return.
    assert applied.loc[prices.index[1], "AAA"] == pytest.approx(1.0)
    assert applied.loc[prices.index[2], "AAA"] == pytest.approx(0.0)
    assert daily.loc[prices.index[1], "strategy_return"] == pytest.approx(0.10)
    assert daily.loc[prices.index[2], "strategy_return"] == pytest.approx(0.0)
    assert equity.loc[daily.index[-1], "strategy_equity"] > 0
    assert summary["observations"] == len(daily)


def test_backtest_charges_cost_for_target_weight_change() -> None:
    prices = prices_for_returns({"AAA": [0.0, 0.0, 0.0, 0.0, 0.0], "SPY": [0.0, 0.0, 0.0, 0.0, 0.0]})
    weights = pd.DataFrame({"AAA": [0.0, 1.0, 1.0, 0.0, 0.0]}, index=prices.index)
    _, daily, _, _ = app.run_backtest(prices, weights, benchmark="SPY", transaction_cost_bps=10, slippage_bps=0)
    # Trading costs occur in the return interval after a close-time target change.
    assert daily.loc[prices.index[2], "transaction_cost_return"] == pytest.approx(0.001)
    assert daily.loc[prices.index[4], "transaction_cost_return"] == pytest.approx(0.001)


def test_backtest_rejects_invalid_weights_and_unknown_assets() -> None:
    prices = prices_for_returns({"AAA": [0.0, 0.01, 0.0], "SPY": [0.0, 0.01, 0.0]})
    with pytest.raises(ValueError, match="at most 1.0"):
        app.run_backtest(prices, pd.DataFrame({"AAA": [1.2, 1.2, 1.2]}, index=prices.index), benchmark="SPY")
    with pytest.raises(ValueError, match="absent from the price panel"):
        app.run_backtest(prices, pd.DataFrame({"MISSING": [1.0, 1.0, 1.0]}, index=prices.index), benchmark="SPY")


def test_backtest_rejects_no_invested_signal() -> None:
    prices = prices_for_returns({"AAA": [0.0, 0.01, 0.0], "SPY": [0.0, 0.01, 0.0]})
    weights = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=prices.index)
    with pytest.raises(ValueError, match="no invested observations"):
        app.run_backtest(prices, weights, benchmark="SPY")


def test_cli_writes_backtest_outputs(tmp_path) -> None:
    index = pd.bdate_range("2022-01-03", periods=220)
    prices = pd.DataFrame({
        "AAA": np.linspace(100, 160, len(index)),
        "BBB": np.linspace(100, 130, len(index)) + 2 * np.sin(np.arange(len(index)) / 5),
        "SPY": np.linspace(100, 125, len(index)) + np.sin(np.arange(len(index)) / 7),
    }, index=index)
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(app.ROOT / "backtest_engine.py"), "--prices", str(path),
         "--benchmark", "SPY", "--lookback", "40", "--skip-recent", "5", "--top-n", "1",
         "--transaction-cost-bps", "5", "--slippage-bps", "2", "--output-dir", str(output_dir)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "backtest_summary.json").read_text())
    assert summary["strategy_name"] == "cross_sectional_momentum_v2_drift_aware"
    assert summary["benchmark_ticker"] == "SPY"
    assert summary["observations"] > 0
    for filename in ("backtest_summary.json", "backtest_daily_returns.csv", "backtest_applied_weights.csv", "backtest_equity_curve.csv"):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert '"strategy"' in result.stdout


def test_drifted_weights_create_real_turnover_and_cost_when_targets_do_not_change() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({
        "AAA": [100.0, 200.0, 200.0, 200.0, 200.0],
        "BBB": [100.0, 100.0, 100.0, 100.0, 100.0],
        "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
    }, index=index)
    targets = pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=index)
    _, daily, applied, equity = app.run_backtest(
        prices, targets, benchmark="SPY", transaction_cost_bps=10, slippage_bps=0,
    )
    assert daily.loc[index[2], "turnover"] == pytest.approx(1.0 / 3.0)
    assert daily.loc[index[2], "transaction_cost_return"] > 0
    assert daily.loc[index[2], "legacy_turnover"] == pytest.approx(0.0)
    assert daily.loc[index[2], "legacy_transaction_cost_return"] == pytest.approx(0.0)
    assert applied.loc[index[2], "AAA"] == pytest.approx(0.5)
    assert equity["legacy_strategy_equity"].iloc[-1] != pytest.approx(equity["strategy_equity"].iloc[-1])


def test_weekly_and_monthly_rebalances_use_last_available_trading_close() -> None:
    index = pd.bdate_range("2024-01-01", periods=45)
    prices = pd.DataFrame({
        "AAA": 100.0 + np.arange(len(index)),
        "BBB": 120.0 - 0.2 * np.arange(len(index)),
        "SPY": 100.0 + 0.1 * np.arange(len(index)),
    }, index=index)
    targets = pd.DataFrame({"AAA": np.linspace(0.2, 0.8, len(index)), "BBB": 0.0}, index=index)
    _, weekly, _, _ = app.run_backtest(prices, targets, benchmark="SPY", rebalance_freq="weekly")
    _, monthly, _, _ = app.run_backtest(prices, targets, benchmark="SPY", rebalance_freq="monthly")
    n_weekly = int((~pd.Series(index.to_period("W-FRI")).duplicated(keep="last")).sum())
    n_monthly = int((~pd.Series(index.to_period("M")).duplicated(keep="last")).sum())
    assert int(weekly["rebalance_executed"].sum()) == n_weekly - 1
    assert int(monthly["rebalance_executed"].sum()) == n_monthly - 1


def test_backtest_accepts_dated_annualized_risk_free_yield_series() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({
        "AAA": [100, 101, 102, 103, 104, 105],
        "SPY": [100, 100.5, 101, 101.5, 102, 102.5],
    }, index=index)
    targets = pd.DataFrame({"AAA": 1.0}, index=index)
    yields = pd.Series(0.05, index=index, name="annual_yield")
    summary, daily, _, _ = app.run_backtest(prices, targets, benchmark="SPY", risk_free_yield=yields)
    expected = (1.05 ** (1 / app.TRADING_DAYS_PER_YEAR)) - 1
    assert daily["risk_free_return"].iloc[0] == pytest.approx(expected)
    assert summary["risk_free_rate_source"] == "dated annualized yield series"


def test_load_risk_free_yield_requires_explicit_decimal_column(tmp_path) -> None:
    path = tmp_path / "rf.csv"
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "annual_yield": [0.04, 0.041]}).to_csv(path, index=False)
    loaded = app.load_risk_free_yield(path)
    assert loaded.loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.041)
    pd.DataFrame({"Date": ["2024-01-01"], "yield_pct": [4.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="annual_yield"):
        app.load_risk_free_yield(path)
