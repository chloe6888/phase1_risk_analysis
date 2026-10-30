#!/usr/bin/env python3
"""Parameter sensitivity and out-of-sample reporting for the backtest engine.

This module compares an explicitly provided parameter grid. It does not choose
an investment strategy or infer future returns. Test-period metrics are reported
separately from metrics before the split to make overfitting easier to detect.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence
import pandas as pd
from backtest_engine import (
    _metric_summary, generate_cross_sectional_momentum_weights, load_risk_free_yield,
    parse_rebalance_freqs, run_backtest,
)
from portfolio_risk import DEFAULT_PRICE_PANEL, DEFAULT_OUTPUT_DIR, TRADING_DAYS_PER_YEAR, load_price_panel


def validate_parameter_grid(
    lookbacks: Sequence[int],
    skip_recents: Sequence[int],
    top_ns: Sequence[int],
    rebalance_freqs: Sequence[str] = ("daily",),
) -> None:
    for name, values in (("lookbacks", lookbacks), ("skip_recents", skip_recents), ("top_ns", top_ns)):
        if not values:
            raise ValueError(f"{name} must contain at least one value.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"{name} must contain positive integers.")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates.")
    allowed = {"daily", "weekly", "monthly"}
    if not rebalance_freqs or any(value not in allowed for value in rebalance_freqs):
        raise ValueError("rebalance_freqs must contain daily, weekly, and/or monthly.")
    if len(set(rebalance_freqs)) != len(rebalance_freqs):
        raise ValueError("rebalance_freqs must not contain duplicates.")


def run_parameter_sweep(
    adjusted_close: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp,
    benchmark: str = "SPY",
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_recents: Sequence[int] = (5, 21),
    top_ns: Sequence[int] = (1, 2, 3),
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield: pd.Series | None = None,
    rebalance_freqs: Sequence[str] = ("daily",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each parameter set and report train/test plus per-calendar-year results.

    test_start is inclusive for out-of-sample returns. Signals use trailing
    history only and the backtest engine applies them with a one-observation lag.
    No best parameter is selected automatically.
    """
    validate_parameter_grid(lookbacks, skip_recents, top_ns, rebalance_freqs)
    try:
        split = pd.Timestamp(test_start)
    except (TypeError, ValueError) as exc:
        raise ValueError("test_start must be a valid date.") from exc
    if pd.isna(split):
        raise ValueError("test_start must be a valid date.")
    if benchmark not in adjusted_close.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")
    rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        for skip_recent in skip_recents:
            for top_n in top_ns:
                for rebalance_freq in rebalance_freqs:
                    key = {"lookback": lookback, "skip_recent": skip_recent, "top_n": top_n,
                           "rebalance_freq": rebalance_freq}
                    try:
                        weights = generate_cross_sectional_momentum_weights(
                            adjusted_close, lookback=lookback, skip_recent=skip_recent,
                            top_n=top_n, benchmark=benchmark,
                        )
                        summary, daily, _, _ = run_backtest(
                            adjusted_close, weights, benchmark=benchmark,
                            transaction_cost_bps=transaction_cost_bps,
                            slippage_bps=slippage_bps, periods_per_year=periods_per_year,
                            risk_free_rate_annual=risk_free_rate_annual,
                            risk_free_yield=risk_free_yield, rebalance_freq=rebalance_freq,
                        )
                        train = daily.loc[daily.index < split]
                        test = daily.loc[daily.index >= split]
                        if len(train) < 2 or len(test) < 2:
                            raise ValueError(
                                f"split requires at least two return observations in both periods; "
                                f"train={len(train)}, test={len(test)}"
                            )
                        train_rf, test_rf = train["risk_free_return"], test["risk_free_return"]
                        train_strategy = _metric_summary(train["strategy_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=train_rf)
                        train_benchmark = _metric_summary(train["benchmark_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=train_rf)
                        test_strategy = _metric_summary(test["strategy_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=test_rf)
                        test_benchmark = _metric_summary(test["benchmark_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=test_rf)
                        rows.append({
                            **key, "status": "OK",
                            "train_start": train.index.min().date().isoformat(),
                            "train_end": train.index.max().date().isoformat(),
                            "test_start": test.index.min().date().isoformat(),
                            "test_end": test.index.max().date().isoformat(),
                            "train_observations": len(train), "test_observations": len(test),
                            "train_total_return": train_strategy["total_return"],
                            "train_annualized_return": train_strategy["annualized_return"],
                            "train_sharpe_ratio": train_strategy["sharpe_ratio"],
                            "train_max_drawdown": train_strategy["max_drawdown"],
                            "train_benchmark_total_return": train_benchmark["total_return"],
                            "test_total_return": test_strategy["total_return"],
                            "test_annualized_return": test_strategy["annualized_return"],
                            "test_sharpe_ratio": test_strategy["sharpe_ratio"],
                            "test_max_drawdown": test_strategy["max_drawdown"],
                            "test_benchmark_total_return": test_benchmark["total_return"],
                            "test_excess_total_return": test_strategy["total_return"] - test_benchmark["total_return"],
                            "full_total_return": summary["strategy"]["total_return"],
                            "full_annualized_return": summary["strategy"]["annualized_return"],
                            "average_daily_turnover": summary["average_daily_turnover"],
                            "annualized_turnover": summary["annualized_turnover"],
                            "legacy_full_total_return": summary["legacy_strategy"]["total_return"],
                            "new_minus_legacy_total_return": summary["legacy_comparison"]["total_return_delta_new_minus_legacy"],
                        })
                        for year, group in daily.groupby(daily.index.year):
                            if len(group) < 2:
                                continue
                            year_rf = group["risk_free_return"]
                            year_strategy = _metric_summary(group["strategy_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=year_rf)
                            year_benchmark = _metric_summary(group["benchmark_return"], periods_per_year, risk_free_rate_annual, risk_free_daily_returns=year_rf)
                            yearly_rows.append({
                                **key, "year": int(year), "observations": len(group),
                                "strategy_total_return": year_strategy["total_return"],
                                "strategy_annualized_return": year_strategy["annualized_return"],
                                "strategy_sharpe_ratio": year_strategy["sharpe_ratio"],
                                "strategy_max_drawdown": year_strategy["max_drawdown"],
                                "benchmark_total_return": year_benchmark["total_return"],
                                "excess_total_return": year_strategy["total_return"] - year_benchmark["total_return"],
                            })
                    except (ValueError, KeyError, TypeError) as exc:
                        rows.append({**key, "status": "SKIPPED", "reason": str(exc)})
    grid = pd.DataFrame(rows).sort_values(
        ["status", "lookback", "skip_recent", "top_n", "rebalance_freq"], kind="stable"
    )
    yearly = pd.DataFrame(yearly_rows)
    if not yearly.empty:
        yearly = yearly.sort_values(
            ["lookback", "skip_recent", "top_n", "rebalance_freq", "year"], kind="stable"
        )
    return grid, yearly


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers, e.g. 63,126,252.") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("List must contain positive integers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("List must not contain duplicates.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parameter sweep and out-of-sample validation report.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--test-start", required=True, help="Inclusive start date for out-of-sample test returns")
    parser.add_argument("--lookbacks", type=parse_int_list, default=[63, 126, 252])
    parser.add_argument("--skip-recents", type=parse_int_list, default=[5, 21])
    parser.add_argument("--top-ns", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--rebalance-freqs", type=parse_rebalance_freqs, default=["daily", "weekly", "monthly"])
    parser.add_argument("--risk-free-yield-csv", type=Path, help="CSV with Date,annual_yield (decimal annualized yield)")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    prices = load_price_panel(args.prices)
    risk_free_yield = load_risk_free_yield(args.risk_free_yield_csv) if args.risk_free_yield_csv else None
    grid, yearly = run_parameter_sweep(
        prices, test_start=args.test_start, benchmark=args.benchmark,
        lookbacks=args.lookbacks, skip_recents=args.skip_recents, top_ns=args.top_ns,
        transaction_cost_bps=args.transaction_cost_bps, slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate, risk_free_yield=risk_free_yield,
        rebalance_freqs=args.rebalance_freqs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "strategy_parameter_sweep.csv", index=False, float_format="%.10f")
    yearly.to_csv(args.output_dir / "strategy_yearly_performance.csv", index=False, float_format="%.10f")
    metadata = {
        "test_start_inclusive": pd.Timestamp(args.test_start).date().isoformat(),
        "benchmark": args.benchmark,
        "lookbacks": args.lookbacks, "skip_recents": args.skip_recents, "top_ns": args.top_ns,
        "rebalance_freqs": args.rebalance_freqs,
        "requested_combinations": len(args.lookbacks) * len(args.skip_recents) * len(args.top_ns) * len(args.rebalance_freqs),
        "successful_combinations": int((grid["status"] == "OK").sum()) if "status" in grid else 0,
        "skipped_combinations": int((grid["status"] == "SKIPPED").sum()) if "status" in grid else 0,
        "selection_warning": "All parameter/frequency combinations are reported; do not select a strategy using test-period performance. Reserve another untouched period or use walk-forward validation.",
        "risk_free_rate_source": "dated annualized yield series" if risk_free_yield is not None else "constant annual rate",
    }
    (args.output_dir / "strategy_validation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
