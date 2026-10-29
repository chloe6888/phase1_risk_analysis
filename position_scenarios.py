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
    values = [shares, average_cost, current_price]
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("shares, average_cost and current_price must all be finite and positive.")
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



def _require_positive_finite(name: str, values: list[float]) -> None:
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"{name} must contain only finite positive values.")


def build_staged_plan(
    shares: float,
    average_cost: float,
    reference_price: float,
    tranche_prices: list[float],
    tranche_shares: list[int],
    stress_prices: list[float] | None = None,
    max_add_cash: float | None = None,
    max_total_shares: float | None = None,
    max_stress_loss_pct: float | None = None,
) -> pd.DataFrame:
    """Calculate a sequential, cumulative plan for multiple planned tranches.

    Each row includes prior tranches, unlike build_add_scenarios(), where each
    row is an independent alternative against the initial position.
    A tranche with zero shares is treated as intentionally skipped.
    """
    stress_prices = stress_prices or []
    _require_positive_finite("average_cost/reference_price/tranche_prices", [average_cost, reference_price, *tranche_prices])
    if stress_prices:
        _require_positive_finite("stress_prices", stress_prices)
    if not np.isfinite(shares) or shares <= 0:
        raise ValueError("Existing shares must be finite and positive.")
    if len(tranche_prices) != len(tranche_shares) or not tranche_prices:
        raise ValueError("tranche_prices and tranche_shares must have the same non-zero length.")
    if any((not isinstance(qty, (int, np.integer))) or qty < 0 for qty in tranche_shares):
        raise ValueError("Each tranche share count must be a non-negative whole number; use 0 to skip a tranche.")
    if max_add_cash is not None and (not np.isfinite(max_add_cash) or max_add_cash < 0):
        raise ValueError("max_add_cash must be finite and non-negative.")
    if max_total_shares is not None and (not np.isfinite(max_total_shares) or max_total_shares < shares):
        raise ValueError("max_total_shares must be finite and no smaller than the current share count.")
    if max_stress_loss_pct is not None and (not np.isfinite(max_stress_loss_pct) or not 0 <= max_stress_loss_pct <= 1):
        raise ValueError("max_stress_loss_pct must be between 0 and 1 (e.g. 0.10 for 10%).")
    if max_stress_loss_pct is not None and not stress_prices:
        raise ValueError("Stress prices are required when max_stress_loss_pct is set.")

    original_cost = shares * average_cost
    cumulative_cash = 0.0
    cumulative_shares = 0
    rows: list[dict[str, float | int | str | bool]] = []
    for i, (price, quantity) in enumerate(zip(tranche_prices, tranche_shares), start=1):
        tranche_cash = float(price) * int(quantity)
        next_cash = cumulative_cash + tranche_cash
        next_shares = shares + cumulative_shares + int(quantity)
        if max_add_cash is not None and next_cash > max_add_cash + 1e-9:
            raise ValueError(
                f"Tranche {i} would bring additional cash to {chr(36)}{next_cash:,.2f}, "
                f"above max_add_cash {chr(36)}{max_add_cash:,.2f}."
            )
        if max_total_shares is not None and next_shares > max_total_shares:
            raise ValueError(
                f"Tranche {i} would bring total shares to {next_shares:g}, "
                f"above max_total_shares {max_total_shares:g}."
            )
        next_total_cost = original_cost + next_cash
        if max_stress_loss_pct is not None and quantity > 0:
            for stress in stress_prices:
                stress_return = (next_shares * stress - next_total_cost) / next_total_cost
                if stress_return < -max_stress_loss_pct - 1e-12:
                    raise ValueError(
                        f"Tranche {i} breaches the stress-loss limit at price {stress:g}: "
                        f"loss {-stress_return:.2%} exceeds {max_stress_loss_pct:.2%}."
                    )

        cumulative_cash = next_cash
        cumulative_shares += int(quantity)
        total_shares = shares + cumulative_shares
        total_cost_basis = original_cost + cumulative_cash
        average_cost_after = total_cost_basis / total_shares
        pnl_reference = total_shares * reference_price - total_cost_basis
        row: dict[str, float | int | str | bool] = {
            "tranche": i,
            "status": "SKIPPED" if quantity == 0 else "PLANNED",
            "entry_price": float(price),
            "tranche_shares": int(quantity),
            "tranche_cash": tranche_cash,
            "cumulative_added_shares": cumulative_shares,
            "cumulative_additional_cash": cumulative_cash,
            "total_shares_after_tranche": total_shares,
            "total_cost_basis_after_tranche": total_cost_basis,
            "average_cost_after_tranche": average_cost_after,
            "market_value_at_reference_price": total_shares * reference_price,
            "pnl_at_reference_price": pnl_reference,
            "return_at_reference_price_pct": pnl_reference / total_cost_basis,
            "budget_check": "PASS",
        }
        for stress in stress_prices:
            pnl = total_shares * stress - total_cost_basis
            row[f"pnl_if_price_{stress:g}"] = pnl
            row[f"return_if_price_{stress:g}_pct"] = pnl / total_cost_basis
        rows.append(row)
    return pd.DataFrame(rows)


def build_add_scenarios(
    shares: float,
    average_cost: float,
    reference_price: float,
    entry_prices: list[float],
    add_share_counts: list[int],
    stress_prices: list[float],
    max_add_cash: float | None = None,
    max_total_shares: float | None = None,
    max_stress_loss_pct: float | None = None,
) -> pd.DataFrame:
    all_prices = [average_cost, reference_price, *entry_prices, *stress_prices]
    if not all(np.isfinite(p) and p > 0 for p in all_prices):
        raise ValueError("All prices must be finite and positive.")
    if not np.isfinite(shares) or shares <= 0 or not entry_prices or not add_share_counts:
        raise ValueError("Existing shares, entry prices and add-share counts must be non-empty and positive.")
    if any((not isinstance(n, (int, np.integer))) or n <= 0 for n in add_share_counts):
        raise ValueError("Additional share counts must be positive whole numbers.")

    if max_add_cash is not None and (not np.isfinite(max_add_cash) or max_add_cash < 0):
        raise ValueError("max_add_cash must be finite and non-negative.")
    if max_total_shares is not None and (not np.isfinite(max_total_shares) or max_total_shares < shares):
        raise ValueError("max_total_shares must be finite and no smaller than the current share count.")
    if max_stress_loss_pct is not None and (not np.isfinite(max_stress_loss_pct) or not 0 <= max_stress_loss_pct <= 1):
        raise ValueError("max_stress_loss_pct must be between 0 and 1 (e.g. 0.10 for 10%).")
    if max_stress_loss_pct is not None and not stress_prices:
        raise ValueError("Stress prices are required when max_stress_loss_pct is set.")

    existing_cost = shares * average_cost
    rows: list[dict[str, float | int | str | bool]] = []
    for entry in entry_prices:
        for add_shares in add_share_counts:
            new_shares = shares + add_shares
            add_cost = entry * add_shares
            total_cost = existing_cost + add_cost
            new_avg = total_cost / new_shares
            stress_returns = {stress: (new_shares * stress - total_cost) / total_cost for stress in stress_prices}
            breached_stress = [
                stress for stress, result in stress_returns.items()
                if max_stress_loss_pct is not None and result < -max_stress_loss_pct - 1e-12
            ]
            within_budget = (
                (max_add_cash is None or add_cost <= max_add_cash)
                and (max_total_shares is None or new_shares <= max_total_shares)
                and not breached_stress
            )
            reasons = [
                f"cash {chr(36)}{add_cost:,.2f} exceeds limit {chr(36)}{max_add_cash:,.2f}" if max_add_cash is not None and add_cost > max_add_cash else "",
                f"total shares {new_shares:g} exceeds limit {max_total_shares:g}" if max_total_shares is not None and new_shares > max_total_shares else "",
            ]
            reasons.extend(
                f"stress price {chr(36)}{stress:g} implies loss {-stress_returns[stress]:.2%} above limit {max_stress_loss_pct:.2%}"
                for stress in breached_stress
            )
            row: dict[str, float | int | str | bool] = {
                "entry_price": entry,
                "add_shares": add_shares,
                "cash_required": add_cost,
                "total_shares_after_add": new_shares,
                "new_average_cost": new_avg,
                "average_cost_change_pct": new_avg / average_cost - 1,
                "position_value_at_reference_price": new_shares * reference_price,
                "pnl_at_reference_price": new_shares * reference_price - total_cost,
                "return_at_reference_price_pct": (new_shares * reference_price - total_cost) / total_cost,
                "within_budget": within_budget,
                "budget_reason": "; ".join(reason for reason in reasons if reason) or "within limits",
                "worst_stress_return_pct": min(stress_returns.values()) if stress_returns else np.nan,
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
    parser.add_argument("--levels", type=float, nargs="+", help="Independent candidate entry prices to scenario-test")
    parser.add_argument("--add-shares", type=int, nargs="+", default=[1, 2, 3], help="Share counts to test at every independent entry price")
    parser.add_argument("--staged-prices", type=float, nargs="+", help="Sequential entry prices for a cumulative plan")
    parser.add_argument("--staged-shares", type=int, nargs="+", help="Share counts for each sequential tranche; use 0 to skip a tranche")
    parser.add_argument("--max-add-cash", type=float, help="Maximum total additional cash budget in USD")
    parser.add_argument("--max-total-shares", type=float, help="Maximum allowed total share count after adding")
    parser.add_argument("--max-stress-loss-pct", type=float, help="Maximum permitted loss fraction at each stress price (0.10 = 10%%)")
    parser.add_argument("--stress-prices", type=float, nargs="*", default=[], help="Downside prices for stress-testing")
    parser.add_argument("--history-csv", type=Path, default=None, help="Optional daily CSV with Date plus Close or Adj Close")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = position_summary(args.shares, args.avg_cost, args.price)
    has_staged_prices = args.staged_prices is not None
    has_staged_shares = args.staged_shares is not None
    if has_staged_prices != has_staged_shares:
        parser.error("--staged-prices and --staged-shares must be provided together")
    if has_staged_prices and args.levels is not None:
        parser.error("Choose either --levels for independent scenarios or --staged-prices for a cumulative plan")
    if not has_staged_prices and args.levels is None:
        parser.error("Provide --levels or both --staged-prices and --staged-shares")

    if has_staged_prices:
        scenario = build_staged_plan(
            args.shares, args.avg_cost, args.price, args.staged_prices, args.staged_shares,
            args.stress_prices, max_add_cash=args.max_add_cash, max_total_shares=args.max_total_shares,
            max_stress_loss_pct=args.max_stress_loss_pct,
        )
        output_suffix = "staged_plan"
    else:
        scenario = build_add_scenarios(
            args.shares, args.avg_cost, args.price, args.levels, args.add_shares, args.stress_prices,
            max_add_cash=args.max_add_cash, max_total_shares=args.max_total_shares,
            max_stress_loss_pct=args.max_stress_loss_pct,
        )
        output_suffix = "add_scenarios"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = args.output_dir / f"{args.ticker.lower()}_{output_suffix}.csv"
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
    if args.max_add_cash is not None:
        print(f"Additional cash budget limit: {chr(36)}{args.max_add_cash:,.2f}")
    if args.max_total_shares is not None:
        print(f"Total share limit: {args.max_total_shares:g}")
    if args.max_stress_loss_pct is not None:
        print(f"Stress-loss limit: {args.max_stress_loss_pct:.2%} at all supplied stress prices")


if __name__ == "__main__":
    main()
