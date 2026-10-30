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


def _metric_summary(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate_annual: float,
    *,
    risk_free_daily_returns: pd.Series | None = None,
) -> dict[str, float | int | str | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("Cannot summarize an empty return series.")
    if not np.isfinite(values.to_numpy()).all() or (values < -1).any():
        raise ValueError("Returns must be finite simple returns no lower than -100%.")
    growth = float((1.0 + values).prod())
    annualized_return = growth ** (periods_per_year / len(values)) - 1.0 if growth > 0 else None
    daily_vol = float(values.std(ddof=1)) if len(values) >= 2 else 0.0
    annualized_vol = daily_vol * math.sqrt(periods_per_year)
    if risk_free_daily_returns is None:
        daily_rf: float | pd.Series = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    else:
        daily_rf = pd.to_numeric(risk_free_daily_returns, errors="coerce").reindex(values.index)
        if daily_rf.isna().any() or not np.isfinite(daily_rf.to_numpy(dtype=float)).all():
            raise ValueError("Risk-free daily returns must cover every strategy return date with finite values.")
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


def load_risk_free_yield(path: Path) -> pd.Series:
    """Load an annualized government-bill yield series from a Date,annual_yield CSV.

    annual_yield is a decimal annualized yield (for example 0.045 for 4.5%),
    not a percentage number and not a daily return.
    """
    frame = pd.read_csv(path)
    required = {"Date", "annual_yield"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Risk-free yield CSV is missing required columns: {missing}")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    values = pd.to_numeric(frame["annual_yield"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Risk-free yield CSV must have valid, unique Date values.")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= -1).any():
        raise ValueError("annual_yield values must be finite decimals greater than -1.")
    return pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name="annual_yield").sort_index()


def _daily_risk_free_returns(
    dates: pd.DatetimeIndex,
    *,
    risk_free_rate_annual: float,
    risk_free_yield: pd.Series | None,
    periods_per_year: int,
) -> pd.Series:
    """Align annualized yields and convert them to effective daily returns."""
    if risk_free_yield is None:
        daily_rate = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
        return pd.Series(daily_rate, index=dates, name="risk_free_return")
    yields = pd.to_numeric(risk_free_yield.copy(), errors="coerce")
    yields.index = pd.to_datetime(yields.index)
    if yields.index.has_duplicates:
        raise ValueError("Risk-free yield series must have unique dates.")
    yields = yields.sort_index()
    if yields.isna().any() or not np.isfinite(yields.to_numpy(dtype=float)).all() or (yields <= -1).any():
        raise ValueError("Risk-free annualized yields must be finite decimals greater than -1.")
    aligned = yields.reindex(dates).ffill()
    # The first price row has no return; backfill only that leading, unused rate.
    aligned = aligned.bfill()
    if aligned.isna().any():
        first_missing = aligned.index[aligned.isna()][0].date().isoformat()
        raise ValueError(f"Risk-free yield series does not cover the backtest; first missing date is {first_missing}.")
    daily = np.power(1.0 + aligned, 1.0 / periods_per_year) - 1.0
    return pd.Series(daily.to_numpy(dtype=float), index=dates, name="risk_free_return")


def parse_rebalance_freqs(value: str) -> list[str]:
    """Parse a comma-separated subset of daily,weekly,monthly."""
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"daily", "weekly", "monthly"}
    if not values or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError("Rebalance frequencies must be a comma-separated subset of daily,weekly,monthly.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Rebalance frequencies must not contain duplicates.")
    return values


def _rebalance_mask(index: pd.DatetimeIndex, rebalance_freq: str) -> np.ndarray:
    """Select each daily close or the last available trading close of each period."""
    if rebalance_freq == "daily":
        return np.ones(len(index), dtype=bool)
    if rebalance_freq == "weekly":
        periods = index.to_period("W-FRI")
    elif rebalance_freq == "monthly":
        periods = index.to_period("M")
    else:
        raise ValueError("rebalance_freq must be daily, weekly, or monthly.")
    return ~pd.Series(periods, index=index).duplicated(keep="last").to_numpy()


def run_backtest(
    adjusted_close: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield: pd.Series | None = None,
    rebalance_freq: str = "daily",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest signals with drifted holdings and costs on actual traded notional.

    Signals formed at close t are executed at that close for the next return
    interval. Between rebalances, actual risky-asset weights drift with relative
    returns. Weekly/monthly rebalances use the last available trading close of
    each calendar period. Turnover is the sum of absolute risky-position changes;
    cash is the residual weight and earns zero.

    The old target-weight-rebalanced approximation is retained in daily/equity
    outputs and summary for an apples-to-apples comparison using the same signals.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("slippage_bps must be finite and non-negative.")
    if not np.isfinite(risk_free_rate_annual) or risk_free_rate_annual <= -1:
        raise ValueError("risk_free_rate_annual must be finite and greater than -100%.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if rebalance_freq not in {"daily", "weekly", "monthly"}:
        raise ValueError("rebalance_freq must be daily, weekly, or monthly.")
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
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target weights must be finite.")
    if (values < -1e-12).any():
        raise ValueError("Negative target weights are not supported; this version is long-only.")
    if (weights.sum(axis=1) > 1.0 + 1e-9).any():
        raise ValueError("Each target-weight row must sum to at most 1.0; residual weight is treated as cash.")

    schedule = _rebalance_mask(prices.index, rebalance_freq)
    scheduled_targets = weights.where(schedule, np.nan).ffill().fillna(0.0)
    asset_returns = prices.pct_change(fill_method=None)
    risk_free_daily = _daily_risk_free_returns(
        asset_returns.index, risk_free_rate_annual=risk_free_rate_annual,
        risk_free_yield=risk_free_yield, periods_per_year=periods_per_year,
    )
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0

    # Actual holdings simulation.
    end_weights = pd.Series(0.0, index=prices.columns, dtype=float)
    daily_rows: list[dict[str, object]] = []
    applied_rows: list[pd.Series] = []
    applied_dates: list[pd.Timestamp] = []
    for position in range(1, len(prices.index)):
        signal_position = position - 1
        pre_trade = end_weights.copy()
        do_rebalance = bool(schedule[signal_position])
        if do_rebalance:
            start_weights = weights.iloc[signal_position].astype(float).copy()
            traded_notional = float((start_weights - pre_trade).abs().sum())
        else:
            start_weights = pre_trade
            traded_notional = 0.0

        period_returns = asset_returns.iloc[position]
        missing_held = [
            ticker for ticker in prices.columns
            if start_weights[ticker] > 1e-12 and pd.isna(period_returns[ticker])
        ]
        if missing_held:
            raise ValueError(
                f"Cannot calculate drifted holdings because prices are missing for held assets on "
                f"{prices.index[position].date().isoformat()}: {missing_held}"
            )
        safe_returns = period_returns.fillna(0.0).astype(float)
        gross = float(start_weights.dot(safe_returns))
        cost_fraction = traded_notional * cost_rate
        if cost_fraction >= 1.0:
            raise ValueError("Estimated trading cost reaches 100% of portfolio value; check cost assumptions.")
        net = (1.0 + gross) * (1.0 - cost_fraction) - 1.0
        cost_drag = gross - net
        if gross <= -1.0 + 1e-15:
            end_weights = pd.Series(0.0, index=prices.columns, dtype=float)
        else:
            # Costs reduce wealth pro rata; market returns then cause natural drift.
            end_weights = start_weights * (1.0 + safe_returns) / (1.0 + gross)

        applied_rows.append(start_weights.rename(prices.index[position]))
        applied_dates.append(prices.index[position])
        daily_rows.append({
            "strategy_gross_return": gross,
            "transaction_cost_return": cost_drag,
            "strategy_return": net,
            "benchmark_return": period_returns[benchmark],
            "turnover": traded_notional,
            "rebalance_executed": do_rebalance,
            "risk_free_return": float(risk_free_daily.iloc[position]),
        })

    if not daily_rows:
        raise ValueError("At least two price observations are required for a backtest.")
    daily = pd.DataFrame(daily_rows, index=pd.DatetimeIndex(applied_dates))
    applied_weights = pd.DataFrame(applied_rows, index=pd.DatetimeIndex(applied_dates)).reindex(columns=prices.columns)

    # Former vectorized method, run on identical signal/rebalance dates.
    legacy_applied = scheduled_targets.shift(1).fillna(0.0)
    legacy_turnover = scheduled_targets.diff().abs().sum(axis=1)
    if len(legacy_turnover):
        legacy_turnover.iloc[0] = float(scheduled_targets.iloc[0].abs().sum())
    legacy_applied_turnover = legacy_turnover.shift(1).fillna(0.0)
    legacy_gross = (asset_returns.where(legacy_applied > 0, 0.0) * legacy_applied).sum(axis=1)
    legacy_missing = (legacy_applied > 0) & asset_returns.isna()
    legacy_gross.loc[legacy_missing.any(axis=1)] = np.nan
    legacy_cost = legacy_applied_turnover * cost_rate
    legacy_net = legacy_gross - legacy_cost
    legacy_data = pd.DataFrame({
        "legacy_strategy_gross_return": legacy_gross,
        "legacy_transaction_cost_return": legacy_cost,
        "legacy_strategy_return": legacy_net,
        "legacy_turnover": legacy_applied_turnover,
    }).reindex(daily.index)
    daily = daily.join(legacy_data)

    first_active = (applied_weights.sum(axis=1) > 0)
    if not first_active.any():
        raise ValueError("Backtest has no invested observations; check lookback/window parameters.")
    first_active_date = first_active[first_active].index[0]
    daily = daily.loc[first_active_date:]
    applied_weights = applied_weights.reindex(daily.index)
    daily = daily.dropna(subset=["strategy_return", "benchmark_return"])
    if len(daily) < 2:
        raise ValueError("At least two aligned strategy/benchmark return observations are required.")
    applied_weights = applied_weights.reindex(daily.index)

    daily_rf = daily["risk_free_return"]
    strategy_metrics = _metric_summary(
        daily["strategy_return"], periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=daily_rf,
    )
    benchmark_metrics = _metric_summary(
        daily["benchmark_return"], periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=daily_rf,
    )
    legacy_returns = daily["legacy_strategy_return"].dropna()
    legacy_metrics = _metric_summary(
        legacy_returns, periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=daily_rf.reindex(legacy_returns.index),
    )
    active = daily["strategy_return"] - daily["benchmark_return"]
    tracking_daily = float(active.std(ddof=1))
    tracking_annual = tracking_daily * math.sqrt(periods_per_year)
    information_ratio = float(active.mean() / tracking_daily * math.sqrt(periods_per_year)) if tracking_daily > 0 else None
    benchmark_variance = float(daily["benchmark_return"].var(ddof=1))
    beta = float(daily["strategy_return"].cov(daily["benchmark_return"]) / benchmark_variance) if benchmark_variance > 0 else None
    summary: dict[str, object] = {
        "strategy_name": "cross_sectional_momentum_v2_drift_aware",
        "benchmark_ticker": benchmark,
        "rebalance_freq": rebalance_freq,
        "start_date": daily.index.min().date().isoformat(),
        "end_date": daily.index.max().date().isoformat(),
        "observations": int(len(daily)),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "actual_rebalance_count": int(daily["rebalance_executed"].sum()),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "annualized_turnover": float(daily["turnover"].mean() * periods_per_year),
        "cumulative_transaction_cost_return": float(daily["transaction_cost_return"].sum()),
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "legacy_strategy": legacy_metrics,
        "legacy_comparison": {
            "total_return_delta_new_minus_legacy": float(strategy_metrics["total_return"] - legacy_metrics["total_return"]),
            "sharpe_delta_new_minus_legacy": (
                float(strategy_metrics["sharpe_ratio"] - legacy_metrics["sharpe_ratio"])
                if strategy_metrics["sharpe_ratio"] is not None and legacy_metrics["sharpe_ratio"] is not None else None
            ),
            "legacy_total_return": float(legacy_metrics["total_return"]),
            "new_total_return": float(strategy_metrics["total_return"]),
            "legacy_max_drawdown": float(legacy_metrics["max_drawdown"]),
            "new_max_drawdown": float(strategy_metrics["max_drawdown"]),
            "comparison_scope": "Same signals and rebalance dates; legacy assumes target weights for every interval and misses drift-driven trading costs.",
        },
        "excess_total_return": float(strategy_metrics["total_return"] - benchmark_metrics["total_return"]),
        "tracking_error_annualized": tracking_annual,
        "information_ratio": information_ratio,
        "beta_to_benchmark": beta,
        "risk_free_rate_source": "dated annualized yield series" if risk_free_yield is not None else "constant annual rate",
        "methodology_note": (
            "Close-time signals are executed for the next return interval. Holdings drift between scheduled rebalances. "
            "Turnover is actual absolute risky-position traded notional; costs are charged on that notional. "
            "Weekly/monthly rebalances occur on the last available trading close of each period. "
            "The previous target-weight-rebalanced calculation is retained for comparison."
        ),
    }
    equity = pd.DataFrame({
        "strategy_equity": (1.0 + daily["strategy_return"]).cumprod(),
        "legacy_strategy_equity": (1.0 + daily["legacy_strategy_return"]).cumprod(),
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
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Fallback annual risk-free rate as decimal")
    parser.add_argument("--risk-free-yield-csv", type=Path, help="CSV with Date,annual_yield (decimal annualized yield)")
    parser.add_argument("--rebalance-freq", choices=["daily", "weekly", "monthly"], default="daily")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    prices = load_price_panel(args.prices)
    target_weights = generate_cross_sectional_momentum_weights(
        prices, lookback=args.lookback, skip_recent=args.skip_recent,
        top_n=args.top_n, benchmark=args.benchmark,
    )
    risk_free_yield = load_risk_free_yield(args.risk_free_yield_csv) if args.risk_free_yield_csv else None
    summary, daily, applied_weights, equity = run_backtest(
        prices, target_weights, benchmark=args.benchmark,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate,
        risk_free_yield=risk_free_yield,
        rebalance_freq=args.rebalance_freq,
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
