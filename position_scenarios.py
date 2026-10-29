#!/usr/bin/env python3
"""Position sizing and add-price scenarios for an existing stock holding.

This tool is intentionally generic: personal holdings are supplied at runtime
and are not stored in source control. It does not predict prices or make buy
signals. Optional daily history can add moving averages and volatility context.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def position_summary(shares: float, average_cost: float, current_price: float) -> dict[str, float]:
    if shares <= 0 or average_cost <= 0 or current_price <= 0:
        raise ValueError("shares, average_cost and current_price must all be positive.")
    cost = shares * average_cost
    value = shares * current_price
    pnl = value - cost
    return {
        "shares": shares,
        "average_cost": average_cost,
        "current_price": current_price,
        "total_cost": cost,
        "market_value": value,
        "unrealized_pnl": pnl,
        "unrealized_return_pct": pnl / cost,
        "price_vs_average_cost_pct": current_price / average_cost - 1,
    }


def build_add_scenarios(
    shares: float,
    average_cost: float,
    reference_price: float,
    entry_prices: list[float],
    add_share_counts: list[int],
    stress_prices: list[float],
) -> pd.DataFrame:
    if any(p <= 0 for p in [average_cost, reference_price, *entry_prices, *stress_prices]):
        raise ValueError("All prices must be positive.")
    if shares <= 0 or any(n <= 0 for n in add_share_counts):
        raise ValueError("Existing shares and additional share counts must be positive.")

    existing_cost = shares * average_cost
    rows: list[dict[str, float | int]] = []
    for entry in entry_prices:
        for add_shares in add_share_counts:
            new_shares = shares + add_shares
            add_cost = entry * add_shares
            total_cost = existing_cost + add_cost
            new_avg = total_cost / new_shares
            row: dict[str, float | int] = {
                "entry_price": entry,
                "add_shares": add_shares,
                "cash_required": add_cost,
                "total_shares_after_add": new_shares,
                "new_average_cost": new_avg,
                "average_cost_change_pct": new_avg / average_cost - 1,
                "position_value_at_reference_price": new_shares * reference_price,
                "pnl_at_reference_price": new_shares * reference_price - total_cost,
                "return_at_reference_price_pct": (new_shares * reference_price - total_cost) / total_cost,
            }
            for stress in stress_prices:
                row[f"pnl_if_price_{stress:g}"] = new_shares * stress - total_cost
                row[f"return_if_price_{stress:g}_pct"] = (new_shares * stress - total_cost) / total_cost
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_history(csv_path: Path) -> dict[str, float | str | int]:
    data = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"]).sort_index()
    if data.empty:
        raise ValueError(f"History CSV is empty: {csv_path}")
    price_col = "Close" if "Close" in data.columns else "Adj Close" if "Adj Close" in data.columns else None
    if price_col is None:
        raise ValueError("History CSV must contain a 'Close' or 'Adj Close' column.")
    close = pd.to_numeric(data[price_col], errors="coerce").dropna()
    if len(close) < 2:
        raise ValueError("History CSV needs at least two valid closing prices.")
    result: dict[str, float | str | int] = {
        "history_first_date": close.index.min().date().isoformat(),
        "history_last_date": close.index.max().date().isoformat(),
        "history_observations": int(len(close)),
        "history_price_field": price_col,
        "history_latest_close": float(close.iloc[-1]),
        "history_min_close_in_sample": float(close.min()),
        "history_max_close_in_sample": float(close.max()),
    }
    if len(close) >= 20:
        result["sma20"] = float(close.tail(20).mean())
    if len(close) >= 50:
        result["sma50"] = float(close.tail(50).mean())
    returns = close.pct_change(fill_method=None).dropna()
    if len(returns) >= 20:
        result["annualized_volatility_20d"] = float(returns.tail(20).std(ddof=1) * math.sqrt(252))
    if len(close) >= 252:
        result["high_52w"] = float(close.tail(252).max())
        result["low_52w"] = float(close.tail(252).min())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="GOOGL")
    parser.add_argument("--shares", type=float, required=True, help="Current number of shares")
    parser.add_argument("--avg-cost", type=float, required=True, help="Average cost per share in USD")
    parser.add_argument("--price", type=float, required=True, help="Reference market price in USD; state if after-hours")
    parser.add_argument("--levels", type=float, nargs="+", required=True, help="Candidate entry prices to scenario-test")
    parser.add_argument("--add-shares", type=int, nargs="+", default=[1, 2, 3], help="Share counts to test at every entry price")
    parser.add_argument("--stress-prices", type=float, nargs="*", default=[], help="Downside prices for stress-testing")
    parser.add_argument("--history-csv", type=Path, default=None, help="Optional daily CSV with Date plus Close or Adj Close")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = position_summary(args.shares, args.avg_cost, args.price)
    scenario = build_add_scenarios(
        args.shares, args.avg_cost, args.price, args.levels, args.add_shares, args.stress_prices
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = args.output_dir / f"{args.ticker.lower()}_add_scenarios.csv"
    scenario.to_csv(scenario_path, index=False, float_format="%.6f")

    summary_df = pd.DataFrame([summary])
    summary_path = args.output_dir / f"{args.ticker.lower()}_position_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"=== {args.ticker.upper()} position snapshot ===")
    print(f"Reference price: ${args.price:,.2f} (confirm whether regular-session or extended-hours quote)")
    print(f"Shares: {args.shares:g} | average cost: ${args.avg_cost:,.2f}")
    print(f"Cost basis: ${summary['total_cost']:,.2f}")
    print(f"Reference market value: ${summary['market_value']:,.2f}")
    print(f"Unrealized P/L: ${summary['unrealized_pnl']:,.2f} ({summary['unrealized_return_pct']:.2%})")

    if args.history_csv:
        history = analyze_history(args.history_csv)
        print("\n=== Historical technical context (based only on supplied CSV) ===")
        for key, value in history.items():
            if key.endswith("volatility_20d"):
                print(f"{key}: {value:.2%}")
            elif isinstance(value, float):
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value}")

    print("\n=== Add-price scenarios ===")
    print(scenario.round(2).to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {scenario_path}")
    print("This is arithmetic and scenario analysis, not a price forecast or automated buy signal.")


if __name__ == "__main__":
    main()
