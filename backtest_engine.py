#!/usr/bin/env python3
"""A transparent, long-only research backtest engine for daily price panels.

Version 1 is a vectorized strategy research tool, not an order simulator. Signals
are computed from closing prices and lagged by one trading observation before
being applied to returns. Trading costs are charged against absolute changes in
target weights; intra-period holdings drift, taxes, fill prices, and final
liquidation costs are not fully simulated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk import DEFAULT_PRICE_PANEL, DEFAULT_OUTPUT_DIR, TRADING_DAYS_PER_YEAR, load_price_panel, max_drawdown_series

ROOT = Path(__file__).resolve().parent


def generate_cross_sectional_momentum_weights(
    adjusted_close: pd.DataFrame,
    *,
    lookback: int = 126,
    skip_recent: int = 21,
    top_n: int = 2,
    benchmark: str | None = "SPY",
) -> pd.DataFrame:
    """Generate equal-weight target allocations to the strongest past-return assets.

    Momentum is measured from t - skip_recent back to t - skip_recent - lookback.
    The latest skip_recent observations are skipped. Weights are determined at
    date t's close and applied no earlier than the next observed return interval.
    """
    for name, value in (("lookback", lookback), ("skip_recent", skip_recent), ("top_n", top_n)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if adjusted_close.empty or adjusted_close.index.has_duplicates or adjusted_close.columns.has_duplicates:
        raise ValueError("Price panel must be non-empty and have unique dates and ticker columns.")
    prices = adjusted_close.copy().sort_index().apply(pd.to_numeric, errors="coerce")
    for column in prices.columns:
        observed = prices[column].dropna()
        if observed.empty or not np.isfinite(observed.to_numpy(dtype=float)).all() or (observed <= 0).any():
            raise ValueError(f"{column}: prices must be finite and positive where present.")
    if benchmark is not None and benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")
    universe = [ticker for ticker in prices.columns if ticker != benchmark]
    if not universe:
        raise ValueError("No strategy assets remain after excluding the benchmark.")

    scores = prices[universe].shift(skip_recent) / prices[universe].shift(lookback + skip_recent) - 1.0
    ranks = scores.rank(axis=1, method="first", ascending=False)
    selected = ranks.le(top_n) & scores.notna()
    selected_counts = selected.sum(axis=1).replace(0, np.nan)
    weights = selected.astype(float).div(selected_counts, axis=0).fillna(0.0)
    weights.index.name = "Date"
    weights.columns.name = "asset"
    return weights


def _metric_summary(returns: pd.Series, periods_per_year: int, risk_free_rate_annual: float) -> dict[str, float | int | str | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("Cannot summarize an empty return series.")
    if not np.isfinite(values.to_numpy()).all() or (values < -1).any():
        raise ValueError("Returns must be finite simple returns no lower than -100%.")
    growth = float((1.0 + values).prod())
    annualized_return = growth ** (periods_per_year / len(values)) - 1.0 if growth > 0 else None
    daily_vol = float(values.std(ddof=1)) if len(values) >= 2 else 0.0
    annualized_vol = daily_vol * math.sqrt(periods_per_year)
    daily_rf = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess = values - daily_rf
    sharpe = float(excess.mean() / daily_vol * math.sqrt(periods_per_year)) if daily_vol > 0 else None
    downside = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean()))
    sortino = float(excess.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else None
    drawdown = max_drawdown_series(values)
    trough_date = drawdown["drawdown"].idxmin()
    wealth_before_trough = drawdown.loc[:trough_date, "portfolio_wealth_index"]
    peak_level = float(drawdown.loc[trough_date, "running_peak"])
    matching_peak_dates = wealth_before_trough[wealth_before_trough >= peak_level - 1e-12]
    peak_date = matching_peak_dates.index[-1] if not matching_peak_dates.empty else wealth_before_trough.idxmax()
    return {
        "observations": int(len(values)),
        "total_return": growth - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdown["drawdown"].min()),
        "max_drawdown_peak_date": pd.Timestamp(peak_date).date().isoformat(),
        "max_drawdown_trough_date": pd.Timestamp(trough_date).date().isoformat(),
        "win_rate": float((values > 0).mean()),
        "best_period_return": float(values.max()),
        "worst_period_return": float(values.min()),
    }


def run_backtest(
    adjusted_close: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest long-only target weights, with one-period signal lag and costs.

    Output tuple: summary, aligned daily returns, applied weights, and equity curve.
    Target weights can be provided only on rebalance dates; the last target is
    carried forward. Row weights must sum to at most one; unused weight is cash.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("slippage_bps must be finite and non-negative.")
    if not np.isfinite(risk_free_rate_annual) or risk_free_rate_annual <= -1:
        raise ValueError("risk_free_rate_annual must be finite and greater than -100%.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if benchmark not in adjusted_close.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")
    if target_weights.empty or target_weights.index.has_duplicates or target_weights.columns.has_duplicates:
        raise ValueError("Target weights must be non-empty with unique dates and asset columns.")
    unknown = sorted(set(target_weights.columns) - set(adjusted_close.columns))
    if unknown:
        raise ValueError(f"Target weights contain assets absent from the price panel: {unknown}")

    prices = adjusted_close.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.has_duplicates:
        raise ValueError("Price panel contains duplicate dates.")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    for column in prices.columns:
        observed = prices[column].dropna()
        if observed.empty or not np.isfinite(observed.to_numpy(dtype=float)).all() or (observed <= 0).any():
            raise ValueError(f"{column}: prices must be finite and positive where present.")
    supplied = target_weights.copy()
    supplied.index = pd.to_datetime(supplied.index)
    if supplied.index.has_duplicates:
        raise ValueError("Target weights contain duplicate dates after date normalization.")
    supplied = supplied.sort_index()
    weights = supplied.reindex(index=prices.index, columns=supplied.columns).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)
    weight_values = weights.to_numpy(dtype=float)
    if not np.isfinite(weight_values).all():
        raise ValueError("Target weights must be finite.")
    if (weight_values < -1e-12).any():
        raise ValueError("Negative target weights are not supported; this version is long-only.")
    if (weights.sum(axis=1) > 1.0 + 1e-9).any():
        raise ValueError("Each target-weight row must sum to at most 1.0; residual weight is treated as cash.")

    asset_returns = prices.pct_change(fill_method=None)
    applied_weights = weights.shift(1).fillna(0.0)
    active_missing = (applied_weights > 0) & asset_returns.isna()
    weighted_returns = asset_returns.where(applied_weights > 0, 0.0) * applied_weights
    gross = weighted_returns.sum(axis=1)
    gross.loc[active_missing.any(axis=1)] = np.nan
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    # A target change known at a close impacts the next return interval.
    applied_turnover = turnover.shift(1).fillna(0.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    transaction_cost = applied_turnover * cost_rate
    net = gross - transaction_cost

    first_active_rows = applied_weights.sum(axis=1) > 0
    if not first_active_rows.any():
        raise ValueError("Backtest has no invested observations; check lookback/window parameters.")
    first_active_date = first_active_rows[first_active_rows].index[0]
    daily = pd.DataFrame({
        "strategy_gross_return": gross,
        "transaction_cost_return": transaction_cost,
        "strategy_return": net,
        "benchmark_return": asset_returns[benchmark],
        "turnover": applied_turnover,
    }).loc[first_active_date:]
    daily = daily.dropna(subset=["strategy_return", "benchmark_return"])
    if len(daily) < 2:
        raise ValueError("At least two aligned strategy/benchmark return observations are required.")
    applied_weights = applied_weights.reindex(daily.index)
    strategy_metrics = _metric_summary(daily["strategy_return"], periods_per_year, risk_free_rate_annual)
    benchmark_metrics = _metric_summary(daily["benchmark_return"], periods_per_year, risk_free_rate_annual)
    active_returns = daily["strategy_return"] - daily["benchmark_return"]
    tracking_error_daily = float(active_returns.std(ddof=1))
    tracking_error_annual = tracking_error_daily * math.sqrt(periods_per_year)
    information_ratio = float(active_returns.mean() / tracking_error_daily * math.sqrt(periods_per_year)) if tracking_error_daily > 0 else None
    benchmark_variance = float(daily["benchmark_return"].var(ddof=1))
    beta = float(daily["strategy_return"].cov(daily["benchmark_return"]) / benchmark_variance) if benchmark_variance > 0 else None
    summary: dict[str, object] = {
        "strategy_name": "cross_sectional_momentum_v1",
        "benchmark_ticker": benchmark,
        "start_date": daily.index.min().date().isoformat(),
        "end_date": daily.index.max().date().isoformat(),
        "observations": int(len(daily)),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "annualized_turnover": float(daily["turnover"].mean() * periods_per_year),
        "cumulative_transaction_cost_return": float(daily["transaction_cost_return"].sum()),
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "excess_total_return": float(strategy_metrics["total_return"] - benchmark_metrics["total_return"]),
        "tracking_error_annualized": tracking_error_annual,
        "information_ratio": information_ratio,
        "beta_to_benchmark": beta,
        "methodology_note": (
            "Signals are lagged one observed return interval. Target weights are carried forward; "
            "costs use absolute changes in target weights. This is a research approximation, not a full holdings/fill simulator."
        ),
    }
    equity = pd.DataFrame({
        "strategy_equity": (1.0 + daily["strategy_return"]).cumprod(),
        "benchmark_equity": (1.0 + daily["benchmark_return"]).cumprod(),
    }, index=daily.index)
    daily.index.name = "Date"
    applied_weights.index.name = "Date"
    equity.index.name = "Date"
    return summary, daily, applied_weights, equity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple, cost-aware cross-sectional momentum backtest.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL, help="Adjusted-close panel generated by phase1_risk_analysis.py")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker present in the price panel")
    parser.add_argument("--lookback", type=int, default=126, help="Momentum lookback observations")
    parser.add_argument("--skip-recent", type=int, default=21, help="Most recent observations excluded from momentum score")
    parser.add_argument("--top-n", type=int, default=2, help="Number of highest-ranked assets to hold equally")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Cost per traded dollar in basis points")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Additional slippage per traded dollar in basis points")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate as decimal")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    prices = load_price_panel(args.prices)
    target_weights = generate_cross_sectional_momentum_weights(
        prices, lookback=args.lookback, skip_recent=args.skip_recent,
        top_n=args.top_n, benchmark=args.benchmark,
    )
    summary, daily, applied_weights, equity = run_backtest(
        prices, target_weights, benchmark=args.benchmark,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    daily.to_csv(args.output_dir / "backtest_daily_returns.csv", float_format="%.10f")
    applied_weights.to_csv(args.output_dir / "backtest_applied_weights.csv", float_format="%.10f")
    equity.to_csv(args.output_dir / "backtest_equity_curve.csv", float_format="%.10f")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
