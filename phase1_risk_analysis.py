#!/usr/bin/env python3
"""Phase 1: download daily data and calculate basic risk metrics.

Uses Yahoo Finance through yfinance for learning purposes, not as an audited
institutional data feed. The return series is based on Adj Close as a historical
total-return proxy; it is not a simulation of executable prices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

# Start with the three reference tickers. Add researched tickers only after this
# first run works; use Yahoo Finance symbols, e.g. "GOOGL" for Alphabet Class A.
TICKERS = ["SPY", "SOXX", "NVDA", "GOOGL"]
BENCHMARK = "SPY"
START_DATE = "2015-01-01"

# Yahoo's end parameter is exclusive. Excluding today's New York date avoids
# accidentally using a partially formed daily bar when the script runs mid-day.
END_DATE = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
TRADING_DAYS_PER_YEAR = 252
DEFAULT_MAX_CACHE_AGE_HOURS = 24.0
ABNORMAL_DAILY_RETURN_THRESHOLD = 0.35  # Review unusual moves; do not silently delete observations.



def cache_metadata_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}_daily.meta.json"


def validate_price_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize and validate daily data without silently repairing prices."""
    if data is None or data.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check connectivity, ticker symbol, "
            "Yahoo Finance availability, or try again later."
        )

    if isinstance(data.columns, pd.MultiIndex):
        field_names = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Dividends", "Stock Splits", "Capital Gains"
        }
        flattened = []
        for col in data.columns:
            parts = [str(x) for x in col]
            field = next((part for part in parts if part in field_names), parts[0])
            flattened.append(field)
        data.columns = flattened
    data = data.loc[:, ~pd.Index(data.columns).duplicated()].copy()
    data = validate_price_frame(data, ticker)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DIR / f"{ticker}_daily.csv", float_format="%.8f")
    metadata = {
        "ticker": ticker,
        "source": "yfinance/Yahoo Finance",
        "requested_start": start,
        "requested_end": end,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_data_date": data.index.min().date().isoformat(),
        "last_data_date": data.index.max().date().isoformat(),
        "row_count": int(len(data)),
        "auto_adjust": False,
        "actions": True,
    }
    cache_metadata_path(ticker).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return data



def load_cached_ticker(ticker: str) -> pd.DataFrame | None:
    """Load and validate a previously downloaded CSV; return None if absent."""
    path = RAW_DIR / f"{ticker}_daily.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return validate_price_frame(data, ticker)


def get_ticker_data(
    ticker: str,
    start: str,
    end: str,
    refresh: bool = False,
    allow_stale_cache: bool = False,
    max_cache_age_hours: float = DEFAULT_MAX_CACHE_AGE_HOURS,
) -> pd.DataFrame:
    """Use a cache only when recent and covering the requested date range.

    Stale cache is refreshed automatically. With --allow-stale-cache, a failed
    refresh may fall back to local data, but it emits an explicit warning.
    """
    cached = None if refresh else load_cached_ticker(ticker)
    cache_reason = "refresh requested" if refresh else "no local cache"
    if cached is not None:
        usable, cache_reason = cache_is_usable(
            ticker, cached, start=start, end=end, max_age_hours=max_cache_age_hours
        )
        if usable:
            print(
                f"Using {ticker} cache ({cache_reason}): "
                f"{cached.index.min().date()} to {cached.index.max().date()}"
            )
            return cached
        print(f"Cache for {ticker} is stale or incomplete: {cache_reason}; refreshing.")

    try:
        return download_one_ticker(ticker, start, end)
    except Exception as exc:
        if allow_stale_cache and cached is not None:
            print(
                f"WARNING: refresh failed for {ticker}; using stale cache ({cache_reason}). "
                f"Do not treat the results as current. Cause: {exc}"
            )
            return cached
        hint = " Use --allow-stale-cache only if you deliberately accept stale data."
        raise RuntimeError(
            f"Could not refresh {ticker}; cache status: {cache_reason}.{hint} Original error: {exc}"
        ) from exc


def compute_risk_metrics(adjusted_close: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
def compute_risk_metrics(adjusted_close: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Calculate return, CAGR, annualized volatility, max drawdown and beta."""
    if benchmark not in adjusted_close.columns:
        raise ValueError(f"Benchmark {benchmark!r} is not in the price data.")
    if adjusted_close.empty or adjusted_close.index.has_duplicates or not adjusted_close.index.is_monotonic_increasing:
        raise ValueError("Price panel must be non-empty, date-sorted, and free of duplicate dates.")

    rows: list[dict] = []
    all_returns = adjusted_close.apply(pd.to_numeric, errors="coerce").pct_change(fill_method=None)
    benchmark_returns = all_returns[benchmark].dropna()

    for ticker in adjusted_close.columns:
        raw_prices = pd.to_numeric(adjusted_close[ticker], errors="coerce")
        prices = raw_prices.dropna().astype(float)
        if not np.isfinite(prices.to_numpy(dtype=float)).all():
            raise ValueError(f"{ticker}: price panel contains non-finite values.")
        # Compute returns before dropping missing prices so a missing bar does
        # not get silently bridged and mislabelled as a one-day return.
        returns = all_returns[ticker].dropna().astype(float)
        if len(prices) < 2 or returns.empty:
            raise ValueError(f"Not enough valid observations to calculate metrics for {ticker}.")
        if (prices <= 0).any():
            raise ValueError(f"Non-positive adjusted prices found for {ticker}.")

        elapsed_days = (prices.index[-1] - prices.index[0]).days
        years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
        cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
        annualized_vol = returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        wealth = prices / prices.iloc[0]
        drawdown = wealth / wealth.cummax() - 1

        paired = pd.concat(
            [returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        benchmark_variance = paired["benchmark"].var(ddof=1)
        beta = (
            paired["asset"].cov(paired["benchmark"]) / benchmark_variance
            if len(paired) >= 2 and pd.notna(benchmark_variance) and benchmark_variance > 0
            else np.nan
        )

        rows.append({
            "ticker": ticker,
            "start_date": prices.index[0].date().isoformat(),
            "end_date": prices.index[-1].date().isoformat(),
            "price_observations": int(prices.shape[0]),
            "missing_price_observations": int(raw_prices.isna().sum()),
            "daily_return_observations": int(returns.shape[0]),
            "mean_daily_return": returns.mean(),
            "CAGR": cagr,
            "annualized_volatility": annualized_vol,
            "maximum_drawdown": drawdown.min(),
            f"beta_vs_{benchmark}": beta,
        })

    return pd.DataFrame(rows).set_index("ticker")



def max_drawdown_details(price_series: pd.Series) -> dict[str, str | float | int | None]:
    """Return max drawdown, peak/trough dates, and calendar days to recovery."""
    prices = pd.to_numeric(price_series, errors="coerce").dropna().astype(float).sort_index()
    if len(prices) < 2 or (prices <= 0).any():
        raise ValueError("At least two positive valid prices are required for drawdown details.")
    peaks = prices.cummax()
    drawdown = prices / peaks - 1.0
    trough_date = drawdown.idxmin()
    trough_price = float(prices.loc[trough_date])
    peak_price = float(peaks.loc[trough_date])
    prior_prices = prices.loc[:trough_date]
    peak_date = prior_prices[prior_prices == peak_price].index[-1]
    recovered = prices.loc[trough_date:][lambda x: x >= peak_price]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "maximum_drawdown": float(drawdown.min()),
        "peak_date": peak_date.date().isoformat(),
        "peak_price": peak_price,
        "trough_date": trough_date.date().isoformat(),
        "trough_price": trough_price,
        "recovery_date": recovery_date.date().isoformat() if recovery_date is not None else None,
        "recovery_days_calendar": int((recovery_date - peak_date).days) if recovery_date is not None else None,
    }


def build_correlation_observation_counts(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise non-null observation count behind each correlation."""
    columns = list(returns.columns)
    counts = pd.DataFrame(index=columns, columns=columns, dtype=int)
    for left in columns:
        for right in columns:
            counts.loc[left, right] = int(returns[[left, right]].dropna().shape[0])
    return counts.astype(int)


def build_data_quality_report(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize missing fields and flag unusually large daily adjusted-price moves."""
    rows = []
    for ticker, data in raw_data.items():
        adj = pd.to_numeric(data["Adj Close"], errors="coerce")
        adj_returns = adj.pct_change(fill_method=None)
        extreme = adj_returns[adj_returns.abs() >= ABNORMAL_DAILY_RETURN_THRESHOLD]
        rows.append({
            "ticker": ticker,
            "rows": len(data),
            "first_date": data.index.min().date().isoformat(),
            "last_date": data.index.max().date().isoformat(),
            "duplicate_dates": int(data.index.duplicated().sum()),
            "missing_adj_close": int(adj.isna().sum()),
            "nonpositive_adj_close": int((adj.dropna() <= 0).sum()),
            "max_abs_daily_return": float(adj_returns.abs().max()) if adj_returns.notna().any() else np.nan,
            "abnormal_return_days_ge_35pct": int(len(extreme)),
            "abnormal_return_dates": ";".join(extreme.index.strftime("%Y-%m-%d").tolist()),
            "missing_open": int(data["Open"].isna().sum()) if "Open" in data.columns else len(data),
            "missing_close": int(data["Close"].isna().sum()) if "Close" in data.columns else len(data),
            "missing_volume": int(data["Volume"].isna().sum()) if "Volume" in data.columns else len(data),
            "has_dividends_column": "Dividends" in data.columns,
            "has_splits_column": "Stock Splits" in data.columns,
        })
    return pd.DataFrame(rows).set_index("ticker")


def save_plots(adjusted_close: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = adjusted_close.dropna(how="any")
    if common.shape[0] < 2:
        raise ValueError("Not enough common dates across tickers to generate comparable plots.")

    # 1. Price chart: one panel per ticker prevents the largest nominal price
    # from visually flattening the other symbols.
    fig, axes = plt.subplots(len(adjusted_close.columns), 1, figsize=(11, 3.2 * len(adjusted_close.columns)), sharex=False)
    if len(adjusted_close.columns) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, adjusted_close.columns):
        series = adjusted_close[ticker].dropna()
        ax.plot(series.index, series, label=ticker)
        ax.set_title(f"{ticker} adjusted-close series")
        ax.set_ylabel("USD")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "closing_prices.png", dpi=150)
    plt.close(fig)

    # 2. Indexed growth: each ticker starts at 100 on the same common start date.
    wealth = common.div(common.iloc[0]).mul(100)
    fig, ax = plt.subplots(figsize=(11, 6))
    wealth.plot(ax=ax)
    ax.set_title("Growth of 100 — adjusted-close total-return proxy")
    ax.set_ylabel("Index (start = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "net_value.png", dpi=150)
    plt.close(fig)

    # 3. Drawdown from each asset's prior peak over the same common period.
    drawdown = wealth.div(wealth.cummax()).sub(1)
    fig, ax = plt.subplots(figsize=(11, 6))
    drawdown.plot(ax=ax)
    ax.set_title("Drawdown from prior peak")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drawdown.png", dpi=150)
    plt.close(fig)

    # 4. Daily-return correlation matrix.
    # Use pairwise available daily returns consistently with the CSV report.
    returns = adjusted_close.pct_change(fill_method=None)
    corr = returns.corr(min_periods=60)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center")
    ax.set_title("Daily-return correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def main(refresh: bool = False, allow_stale_cache: bool = False, max_cache_age_hours: float = DEFAULT_MAX_CACHE_AGE_HOURS) -> None:
    if BENCHMARK not in TICKERS:
        raise ValueError(
            f"Benchmark {BENCHMARK!r} must be included in TICKERS. "
            "Update the configuration at the top of this file."
        )
    if len(set(TICKERS)) != len(TICKERS):
        raise ValueError("TICKERS contains duplicates; each ticker must appear only once.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Tickers: {', '.join(TICKERS)}")
    if refresh:
        print(f"Refreshing data: {START_DATE} to before {END_DATE} (end date exclusive)")
    else:
        print("Using local CSV cache when available; missing tickers will be downloaded.")
    raw_data: dict[str, pd.DataFrame] = {}
    series: dict[str, pd.Series] = {}

    for ticker in TICKERS:
        print(f"Loading {ticker} ...")
        try:
            data = get_ticker_data(ticker, START_DATE, END_DATE, refresh=refresh, allow_stale_cache=allow_stale_cache, max_cache_age_hours=max_cache_age_hours)
        except Exception as exc:
            raise RuntimeError(f"Download/validation failed for {ticker}: {exc}") from exc
        raw_data[ticker] = data
        series[ticker] = data["Adj Close"].rename(ticker).astype(float)

    data_quality = build_data_quality_report(raw_data)
    data_quality.to_csv(OUT_DIR / "data_quality_report.csv")

    adjusted_close = pd.concat(series.values(), axis=1).sort_index()
    adjusted_close.index.name = "Date"
    adjusted_close.to_csv(RAW_DIR / "adjusted_close_panel.csv", float_format="%.8f")

    daily_returns = adjusted_close.pct_change(fill_method=None)
    daily_returns.to_csv(OUT_DIR / "daily_returns.csv", float_format="%.10f")

    metrics = compute_risk_metrics(adjusted_close, BENCHMARK)
    metrics.to_csv(OUT_DIR / "risk_metrics.csv", float_format="%.10f")

    correlation = daily_returns.corr(min_periods=60)
    correlation.to_csv(OUT_DIR / "correlation_matrix.csv", float_format="%.6f")
    correlation_counts = build_correlation_observation_counts(daily_returns)
    correlation_counts.to_csv(OUT_DIR / "correlation_observation_counts.csv")

    drawdown_rows = []
    for ticker in adjusted_close.columns:
        details = max_drawdown_details(adjusted_close[ticker])
        details["ticker"] = ticker
        drawdown_rows.append(details)
    pd.DataFrame(drawdown_rows).set_index("ticker").to_csv(OUT_DIR / "max_drawdown_details.csv")

    analysis_metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance/Yahoo Finance",
        "tickers": TICKERS,
        "benchmark": BENCHMARK,
        "requested_start": START_DATE,
        "requested_end_exclusive": END_DATE,
        "data_last_dates": {ticker: data.index.max().date().isoformat() for ticker, data in raw_data.items()},
        "max_cache_age_hours": max_cache_age_hours,
        "allow_stale_cache": allow_stale_cache,
        "return_price_field": "Adj Close",
    }
    (OUT_DIR / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    save_plots(adjusted_close)

    print("\n=== Data quality ===")
    print(data_quality.to_string())
    print("\n=== Risk metrics ===")
    display_metrics = metrics.copy()
    for col in ["mean_daily_return", "CAGR", "annualized_volatility", "maximum_drawdown"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
    display_metrics[f"beta_vs_{BENCHMARK}"] = display_metrics[f"beta_vs_{BENCHMARK}"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
    )
    print(display_metrics.to_string())
    print("\n=== Daily-return correlation ===")
    print(correlation.round(3).to_string())
    print("\n=== Pairwise observation counts ===")
    print(correlation_counts.to_string())
    print("\n=== Maximum drawdown details ===")
    print(pd.DataFrame(drawdown_rows).set_index("ticker").to_string())
    print(f"\nFiles saved under: {ROOT}")
    print("Charts: closing_prices.png, net_value.png, drawdown.png, correlation_matrix.png")
    print("Tables: correlation_observation_counts.csv, max_drawdown_details.csv, analysis_metadata.json")
    print("Reminder: Adj Close is a historical total-return proxy, not an executable fill price.")
    print("Reminder: these historical metrics are not forecasts or future-loss limits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download/cache US market data and calculate basic risk metrics."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload all configured tickers instead of using local CSV files",
    )
    parser.add_argument(
        "--allow-stale-cache",
        action="store_true",
        help="if refresh fails, explicitly allow stale local CSV data (results will be warned as stale)",
    )
    parser.add_argument(
        "--max-cache-age-hours",
        type=float,
        default=DEFAULT_MAX_CACHE_AGE_HOURS,
        help=f"maximum cache age before refresh (default: {DEFAULT_MAX_CACHE_AGE_HOURS:g} hours)",
    )
    args = parser.parse_args()
    if args.max_cache_age_hours <= 0:
        parser.error("--max-cache-age-hours must be positive")
    main(refresh=args.refresh, allow_stale_cache=args.allow_stale_cache, max_cache_age_hours=args.max_cache_age_hours)
