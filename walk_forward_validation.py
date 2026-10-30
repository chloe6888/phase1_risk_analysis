#!/usr/bin/env python3
"""Rolling walk-forward validation for the v1 momentum backtest.

Each fold selects parameters using only the immediately preceding training window
and evaluates the selected parameters on the next non-overlapping test window.
This reduces (but does not eliminate) overfitting and does not promise future
profitability. Keep an additional untouched period for final confirmation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest_engine import (
    _metric_summary, generate_cross_sectional_momentum_weights, load_risk_free_yield,
    parse_rebalance_freqs, run_backtest,
)
from portfolio_risk import DEFAULT_OUTPUT_DIR, DEFAULT_PRICE_PANEL, TRADING_DAYS_PER_YEAR, load_price_panel
from strategy_validation import parse_int_list, validate_parameter_grid


def run_walk_forward(
    adjusted_close: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_recents: Sequence[int] = (5, 21),
    top_ns: Sequence[int] = (1, 2, 3),
    train_window: int = 252,
    test_window: int = 63,
    selection_metric: str = "sharpe_ratio",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield: pd.Series | None = None,
    rebalance_freqs: Sequence[str] = ("daily",),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run a rolling parameter-selection process, testing each selection forward.

    train_window and test_window are measured in aligned daily return observations.
    Test windows are contiguous and non-overlapping. The strategy candidate set is
    fixed before running; parameters are reselected on each training fold only.
    """
    validate_parameter_grid(lookbacks, skip_recents, top_ns, rebalance_freqs)
    for name, value in (("train_window", train_window), ("test_window", test_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2.")
    if selection_metric not in {"sharpe_ratio", "annualized_return", "sortino_ratio"}:
        raise ValueError("selection_metric must be sharpe_ratio, annualized_return, or sortino_ratio.")
    if benchmark not in adjusted_close.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")

    combos = list(itertools.product(lookbacks, skip_recents, top_ns, rebalance_freqs))
    strategy_returns: dict[tuple[int, int, int, str], pd.Series] = {}
    benchmark_returns: dict[tuple[int, int, int, str], pd.Series] = {}
    risk_free_returns: dict[tuple[int, int, int, str], pd.Series] = {}
    for lookback, skip_recent, top_n, rebalance_freq in combos:
        key = (lookback, skip_recent, top_n, rebalance_freq)
        try:
            target = generate_cross_sectional_momentum_weights(
                adjusted_close, lookback=lookback, skip_recent=skip_recent,
                top_n=top_n, benchmark=benchmark,
            )
            _, daily, _, _ = run_backtest(
                adjusted_close, target, benchmark=benchmark,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps, periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free_rate_annual,
                risk_free_yield=risk_free_yield, rebalance_freq=rebalance_freq,
            )
            strategy_returns[key] = daily["strategy_return"]
            benchmark_returns[key] = daily["benchmark_return"]
            risk_free_returns[key] = daily["risk_free_return"]
        except (ValueError, KeyError, TypeError):
            # The fold loop will make insufficient/invalid parameter sets explicit.
            continue
    if not strategy_returns:
        raise ValueError("No valid parameter combinations could be backtested.")

    common = None
    for key in strategy_returns:
        dates = set(strategy_returns[key].index).intersection(benchmark_returns[key].dropna().index)
        common = dates if common is None else common.intersection(dates)
    common_index = pd.DatetimeIndex(sorted(common or []))
    required = train_window + test_window
    if len(common_index) < required:
        raise ValueError(
            f"Only {len(common_index)} common return observations are available; "
            f"at least {required} are required for one train/test fold. "
            "Use shorter windows or more history."
        )
    returns = {
        key: strategy_returns[key].reindex(common_index)
        for key in strategy_returns
    }
    bench = benchmark_returns[next(iter(benchmark_returns))].reindex(common_index)
    rf_daily = risk_free_returns[next(iter(risk_free_returns))].reindex(common_index)
    if rf_daily.isna().any():
        raise ValueError("Risk-free return series contains missing values in the common evaluation window.")
    if bench.isna().any():
        raise ValueError("Benchmark contains missing values in the common evaluation window.")

    fold_rows: list[dict[str, object]] = []
    oos_chunks: list[pd.DataFrame] = []
    fold_number = 0
    test_start_pos = train_window
    while test_start_pos + test_window <= len(common_index):
        fold_number += 1
        train_dates = common_index[test_start_pos - train_window:test_start_pos]
        test_dates = common_index[test_start_pos:test_start_pos + test_window]
        scores: list[tuple[float, tuple[int, int, int, str], dict[str, object]]] = []
        for key in combos:
            if key not in returns:
                continue
            train = returns[key].reindex(train_dates)
            if train.isna().any() or len(train) < 2:
                continue
            metrics = _metric_summary(
                train, periods_per_year, risk_free_rate_annual,
                risk_free_daily_returns=rf_daily.reindex(train.index),
            )
            score = metrics.get(selection_metric)
            if score is None or not pd.notna(score):
                continue
            scores.append((float(score), key, metrics))
        if not scores:
            fold_rows.append({
                "fold": fold_number,
                "train_start": train_dates.min().date().isoformat(),
                "train_end": train_dates.max().date().isoformat(),
                "test_start": test_dates.min().date().isoformat(),
                "test_end": test_dates.max().date().isoformat(),
                "status": "SKIPPED",
                "reason": f"No parameter combination had a defined {selection_metric} in the training window.",
            })
            test_start_pos += test_window
            continue
        # Deterministic tie-break: Python max preserves the first tuple on equal scores.
        best_score, selected_key, train_metrics = max(scores, key=lambda item: item[0])
        test = returns[selected_key].reindex(test_dates)
        test_benchmark = bench.reindex(test_dates)
        if test.isna().any() or test_benchmark.isna().any():
            fold_rows.append({
                "fold": fold_number, "status": "SKIPPED",
                "reason": "Missing strategy or benchmark returns in test window.",
                "train_start": train_dates.min().date().isoformat(),
                "train_end": train_dates.max().date().isoformat(),
                "test_start": test_dates.min().date().isoformat(),
                "test_end": test_dates.max().date().isoformat(),
            })
            test_start_pos += test_window
            continue
        test_rf = rf_daily.reindex(test_dates)
        test_metrics = _metric_summary(test, periods_per_year, risk_free_rate_annual, risk_free_daily_returns=test_rf)
        benchmark_metrics = _metric_summary(test_benchmark, periods_per_year, risk_free_rate_annual, risk_free_daily_returns=test_rf)
        fold_rows.append({
            "fold": fold_number, "status": "OK",
            "train_start": train_dates.min().date().isoformat(),
            "train_end": train_dates.max().date().isoformat(),
            "test_start": test_dates.min().date().isoformat(),
            "test_end": test_dates.max().date().isoformat(),
            "train_observations": len(train_dates), "test_observations": len(test_dates),
            "selected_lookback": selected_key[0],
            "selected_skip_recent": selected_key[1],
            "selected_top_n": selected_key[2],
            "selected_rebalance_freq": selected_key[3],
            "selection_metric": selection_metric,
            "training_selection_score": best_score,
            "train_total_return": train_metrics["total_return"],
            "train_annualized_return": train_metrics["annualized_return"],
            "train_sharpe_ratio": train_metrics["sharpe_ratio"],
            "test_total_return": test_metrics["total_return"],
            "test_annualized_return": test_metrics["annualized_return"],
            "test_sharpe_ratio": test_metrics["sharpe_ratio"],
            "test_sortino_ratio": test_metrics["sortino_ratio"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_benchmark_total_return": benchmark_metrics["total_return"],
            "test_excess_total_return": test_metrics["total_return"] - benchmark_metrics["total_return"],
        })
        fold_oos = pd.DataFrame({
            "fold": fold_number,
            "strategy_return": test,
            "benchmark_return": test_benchmark,
            "selected_lookback": selected_key[0],
            "selected_skip_recent": selected_key[1],
            "selected_top_n": selected_key[2],
            "selected_rebalance_freq": selected_key[3],
            "risk_free_return": test_rf,
        }, index=test_dates)
        oos_chunks.append(fold_oos)
        test_start_pos += test_window

    folds = pd.DataFrame(fold_rows)
    folds = folds.sort_values("fold", kind="stable") if not folds.empty else folds
    oos = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.DataFrame()
    successful = folds[folds.get("status", pd.Series(dtype=str)) == "OK"] if not folds.empty else pd.DataFrame()
    summary: dict[str, object] = {
        "benchmark": benchmark,
        "selection_metric": selection_metric,
        "train_window_observations": train_window,
        "test_window_observations": test_window,
        "requested_combinations": len(combos),
        "rebalance_freqs": list(rebalance_freqs),
        "risk_free_rate_source": "dated annualized yield series" if risk_free_yield is not None else "constant annual rate",
        "usable_combinations": len(strategy_returns),
        "folds_total": int(len(folds)),
        "folds_successful": int(len(successful)),
        "folds_skipped": int(len(folds) - len(successful)),
        "oos_observations": int(len(oos)),
        "first_oos_date": oos.index.min().date().isoformat() if not oos.empty else None,
        "last_oos_date": oos.index.max().date().isoformat() if not oos.empty else None,
        "selection_warning": "Parameters are selected on prior training windows only. Walk-forward tests remain subject to multiple-testing bias and do not replace a final untouched holdout.",
    }
    if not oos.empty:
        oos_rf = oos["risk_free_return"]
        strategy_metrics = _metric_summary(oos["strategy_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=oos_rf)
        benchmark_metrics = _metric_summary(oos["benchmark_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=oos_rf)
        summary["out_of_sample_strategy"] = strategy_metrics
        summary["out_of_sample_benchmark"] = benchmark_metrics
        summary["out_of_sample_excess_total_return"] = strategy_metrics["total_return"] - benchmark_metrics["total_return"]
    return folds, oos, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling walk-forward validation for momentum strategies.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--lookbacks", type=parse_int_list, default=[63, 126, 252])
    parser.add_argument("--skip-recents", type=parse_int_list, default=[5, 21])
    parser.add_argument("--top-ns", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--rebalance-freqs", type=parse_rebalance_freqs, default=["daily", "weekly", "monthly"])
    parser.add_argument("--risk-free-yield-csv", type=Path, help="CSV with Date,annual_yield (decimal annualized yield)")
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--selection-metric", choices=["sharpe_ratio", "annualized_return", "sortino_ratio"], default="sharpe_ratio")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    prices = load_price_panel(args.prices)
    risk_free_yield = load_risk_free_yield(args.risk_free_yield_csv) if args.risk_free_yield_csv else None
    folds, oos, summary = run_walk_forward(
        prices, benchmark=args.benchmark, lookbacks=args.lookbacks,
        skip_recents=args.skip_recents, top_ns=args.top_ns, rebalance_freqs=args.rebalance_freqs,
        train_window=args.train_window, test_window=args.test_window,
        selection_metric=args.selection_metric,
        transaction_cost_bps=args.transaction_cost_bps, slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate, risk_free_yield=risk_free_yield,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "walk_forward_folds.csv", index=False, float_format="%.10f")
    oos.to_csv(args.output_dir / "walk_forward_oos_returns.csv", index_label="Date", float_format="%.10f")
    (args.output_dir / "walk_forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
