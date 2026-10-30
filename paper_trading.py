#!/usr/bin/env python3
"""Offline paper rebalancing: target weights -> simulated whole-share orders.

The module never connects to a broker. Orders are simulated at a reference close
with a configurable adverse slippage assumption and transaction-cost rate.
The optional ledger is local JSON and is only updated with --commit-ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk import ROOT, load_price_panel

DEFAULT_CLOSE_PRICE_PANEL = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_LEDGER = DEFAULT_OUTPUT_DIR / "paper_trading_ledger.json"
ORDER_COLUMNS = [
    "ticker", "side", "current_shares", "desired_target_shares",
    "filled_shares", "reference_price", "simulated_execution_price",
    "gross_notional", "transaction_fee", "slippage_cost", "status",
]


def normalize_holdings(holdings: pd.DataFrame) -> dict[str, float]:
    if not isinstance(holdings, pd.DataFrame):
        raise ValueError("holdings must be a DataFrame.")
    if not {"ticker", "shares"}.issubset(holdings.columns):
        raise ValueError("Holdings CSV must contain ticker and shares columns.")
    frame = holdings[["ticker", "shares"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    if frame["ticker"].eq("").any() or frame["shares"].isna().any():
        raise ValueError("Holdings must have non-empty tickers and numeric shares.")
    if frame["ticker"].duplicated().any():
        raise ValueError("Holdings must not contain duplicate tickers.")
    values = frame["shares"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Holding shares must be finite and non-negative.")
    return {
        row.ticker: float(row.shares)
        for row in frame.itertuples(index=False)
        if float(row.shares) > 0
    }


def normalize_target_weights(targets: pd.DataFrame | Mapping[str, float]) -> dict[str, float]:
    if isinstance(targets, pd.DataFrame):
        if not {"ticker", "weight"}.issubset(targets.columns):
            raise ValueError("Target-weight CSV must contain ticker and weight columns.")
        frame = targets[["ticker", "weight"]].copy()
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        if frame["ticker"].eq("").any() or frame["weight"].isna().any():
            raise ValueError("Target weights must have non-empty tickers and numeric weights.")
        if frame["ticker"].duplicated().any():
            raise ValueError("Target weights must not contain duplicate tickers.")
        weights = dict(zip(frame["ticker"], frame["weight"].astype(float)))
    else:
        weights = {str(k).strip().upper(): float(v) for k, v in dict(targets).items()}
    if not weights:
        raise ValueError("At least one target weight is required.")
    values = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Target weights must be finite and between 0 and 1.")
    if sum(weights.values()) > 1.0 + 1e-9:
        raise ValueError("Target risky weights must sum to no more than 100%; residual is cash.")
    return weights


def latest_price_series(price_panel: pd.DataFrame) -> pd.Series:
    if not isinstance(price_panel, pd.DataFrame) or price_panel.empty:
        raise ValueError("Price panel must be non-empty.")
    prices = price_panel.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.isna().any() or prices.index.has_duplicates:
        raise ValueError("Price panel must have valid, unique dates.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    latest: dict[str, float] = {}
    for ticker in prices.columns:
        values = prices[ticker].dropna()
        if values.empty:
            continue
        value = float(values.iloc[-1])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{ticker}: latest price must be finite and positive.")
        latest[str(ticker).upper()] = value
    return pd.Series(latest, dtype=float, name="reference_price")


def simulate_rebalance(
    holdings: pd.DataFrame,
    target_weights: pd.DataFrame | Mapping[str, float],
    latest_prices: pd.Series | Mapping[str, float],
    *,
    cash: float,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    fractional_shares: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Simulate sells first, then fund buys from available cash.

    A cash shortfall scales requested buy quantities down proportionally. Whole
    shares are rounded down; fractional shares are optional. The final holdings
    file is a paper-trading snapshot, not a brokerage instruction.
    """
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be finite and non-negative.")
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps < 10000:
        raise ValueError("slippage_bps must be between 0 and 10000.")
    current = normalize_holdings(holdings)
    targets = normalize_target_weights(target_weights)
    price_map = {
        str(k).strip().upper(): float(v)
        for k, v in dict(latest_prices).items()
    }
    if any(not math.isfinite(v) or v <= 0 for v in price_map.values()):
        raise ValueError("All supplied prices must be finite and positive.")
    tickers = sorted(set(current) | set(targets))
    missing = sorted(set(tickers) - set(price_map))
    if missing:
        raise ValueError(f"Missing current prices for: {missing}")

    nav_before = float(cash + sum(current.get(t, 0.0) * price_map[t] for t in tickers))
    if not math.isfinite(nav_before) or nav_before <= 0:
        raise ValueError("Account net asset value must be positive.")
    fee_rate = transaction_cost_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0

    desired_shares: dict[str, float] = {}
    for ticker in tickers:
        target_weight = float(targets.get(ticker, 0.0))
        raw = nav_before * target_weight / price_map[ticker]
        desired_shares[ticker] = raw if fractional_shares else float(math.floor(raw + 1e-10))

    sell_quantities = {
        ticker: max(0.0, current.get(ticker, 0.0) - desired_shares[ticker])
        for ticker in tickers
    }
    buy_quantities = {
        ticker: max(0.0, desired_shares[ticker] - current.get(ticker, 0.0))
        for ticker in tickers
    }

    orders: list[dict[str, object]] = []
    cash_after_sells = float(cash)
    total_fees = 0.0
    total_slippage = 0.0

    for ticker, quantity in sell_quantities.items():
        if quantity <= 1e-12:
            continue
        reference = price_map[ticker]
        execution = reference * (1.0 - slippage_rate)
        notional = quantity * execution
        fee = notional * fee_rate
        slippage_cost = quantity * (reference - execution)
        cash_after_sells += notional - fee
        total_fees += fee
        total_slippage += slippage_cost
        orders.append({
            "ticker": ticker, "side": "SELL",
            "current_shares": current.get(ticker, 0.0),
            "desired_target_shares": desired_shares[ticker],
            "filled_shares": quantity,
            "reference_price": reference,
            "simulated_execution_price": execution,
            "gross_notional": notional,
            "transaction_fee": fee,
            "slippage_cost": slippage_cost,
            "status": "FILLED",
        })

    requested_buy_cost = sum(
        quantity * price_map[ticker] * (1.0 + slippage_rate) * (1.0 + fee_rate)
        for ticker, quantity in buy_quantities.items()
    )
    buy_scale = min(1.0, cash_after_sells / requested_buy_cost) if requested_buy_cost > 0 else 1.0
    final_shares = {
        ticker: current.get(ticker, 0.0) - sell_quantities[ticker]
        for ticker in tickers
    }

    for ticker, desired_quantity in buy_quantities.items():
        if desired_quantity <= 1e-12:
            continue
        scaled_quantity = desired_quantity * buy_scale
        quantity = scaled_quantity if fractional_shares else float(math.floor(scaled_quantity + 1e-10))
        if quantity <= 1e-12:
            continue
        reference = price_map[ticker]
        execution = reference * (1.0 + slippage_rate)
        notional = quantity * execution
        fee = notional * fee_rate
        slippage_cost = quantity * (execution - reference)
        total_cost = notional + fee
        if total_cost > cash_after_sells + 1e-7:
            raise RuntimeError("Buy simulation exceeded available cash after scaling.")
        cash_after_sells -= total_cost
        total_fees += fee
        total_slippage += slippage_cost
        final_shares[ticker] += quantity
        orders.append({
            "ticker": ticker, "side": "BUY",
            "current_shares": current.get(ticker, 0.0),
            "desired_target_shares": desired_shares[ticker],
            "filled_shares": quantity,
            "reference_price": reference,
            "simulated_execution_price": execution,
            "gross_notional": notional,
            "transaction_fee": fee,
            "slippage_cost": slippage_cost,
            "status": "FILLED" if quantity >= desired_quantity - 1e-9 else "PARTIALLY_FILLED",
        })

    cash_after = max(0.0, float(cash_after_sells))
    holdings_rows: list[dict[str, object]] = []
    for ticker in tickers:
        shares = float(final_shares[ticker])
        value = shares * price_map[ticker]
        holdings_rows.append({
            "ticker": ticker,
            "shares": shares,
            "reference_price": price_map[ticker],
            "market_value": value,
            "target_weight": float(targets.get(ticker, 0.0)),
        })
    post_market_value = sum(row["market_value"] for row in holdings_rows)
    nav_after = float(cash_after + post_market_value)
    holdings_after = pd.DataFrame(holdings_rows)
    if nav_after > 0:
        holdings_after["actual_weight"] = holdings_after["market_value"] / nav_after
    else:
        holdings_after["actual_weight"] = 0.0

    orders_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    summary: dict[str, object] = {
        "nav_before": nav_before,
        "nav_after": nav_after,
        "cash_before": float(cash),
        "cash_after": cash_after,
        "cash_target_weight": max(0.0, 1.0 - sum(targets.values())),
        "target_risky_weight": float(sum(targets.values())),
        "total_transaction_fees": float(total_fees),
        "total_slippage_cost": float(total_slippage),
        "total_simulated_friction": float(total_fees + total_slippage),
        "orders_count": int(len(orders_frame)),
        "buy_orders_count": int((orders_frame["side"] == "BUY").sum()) if not orders_frame.empty else 0,
        "sell_orders_count": int((orders_frame["side"] == "SELL").sum()) if not orders_frame.empty else 0,
        "buy_scale_due_to_cash": float(buy_scale),
        "fractional_shares": bool(fractional_shares),
        "target_weights": targets,
        "methodology": {
            "sell_before_buy": True,
            "whole_shares_rounded_down": not fractional_shares,
            "buy_shortfall": "requested buys are scaled proportionally to cash after sells",
            "execution_price": "reference close adjusted adversely by the configured slippage",
            "commission": "charged on simulated execution notional",
            "no_broker_connection": True,
        },
        "warnings": [
            "This is an offline paper simulation at a reference close, not a live quote or guaranteed fill.",
            "Taxes, market impact, halts, price limits, intraday fills, and queue priority are not modeled.",
            "Review order quantities and the latest reference prices before using the output for any real-world decision.",
        ],
    }
    return orders_frame, holdings_after, summary


def read_account(
    ledger_path: Path,
    *,
    initial_holdings_csv: Path | None,
    initial_cash: float,
) -> tuple[pd.DataFrame, float, bool]:
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read paper ledger: {exc}") from exc
        if not isinstance(ledger, dict) or not isinstance(ledger.get("positions"), list):
            raise ValueError("Paper ledger must contain a positions list.")
        cash = float(ledger.get("cash", 0))
        holdings = pd.DataFrame(ledger["positions"], columns=["ticker", "shares"])
        normalize_holdings(holdings)
        if not math.isfinite(cash) or cash < 0:
            raise ValueError("Paper ledger cash must be finite and non-negative.")
        return holdings, cash, True
    if initial_holdings_csv is None:
        raise ValueError(
            "No paper ledger exists; supply --holdings-csv to initialize the paper account."
        )
    if not math.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError("initial_cash must be finite and non-negative.")
    holdings = pd.read_csv(initial_holdings_csv)
    normalize_holdings(holdings)
    return holdings[["ticker", "shares"]].copy(), float(initial_cash), False


def write_account_ledger(
    path: Path,
    holdings_after: pd.DataFrame,
    cash_after: float,
    *,
    last_simulation_summary: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "cash": float(cash_after),
        "positions": holdings_after[["ticker", "shares"]].to_dict(orient="records"),
        "last_simulation": dict(last_simulation_summary),
        "note": "Local paper-trading state only; no brokerage orders were sent.",
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline paper orders from target weights; never connects to a broker."
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_CLOSE_PRICE_PANEL)
    parser.add_argument("--targets-csv", type=Path, required=True, help="CSV with ticker,weight columns.")
    parser.add_argument("--holdings-csv", type=Path, help="Initial holdings CSV (ticker,shares), used if ledger is absent.")
    parser.add_argument("--cash", type=float, default=0.0, help="Initial cash; used only when creating a new ledger.")
    parser.add_argument("--ledger-file", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--commit-ledger", action="store_true", help="Persist the simulated state; default is preview only.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fractional-shares", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    account, cash, ledger_exists = read_account(
        args.ledger_file,
        initial_holdings_csv=args.holdings_csv,
        initial_cash=args.cash,
    )
    target_frame = pd.read_csv(args.targets_csv)
    price_panel = load_price_panel(args.prices)
    latest = latest_price_series(price_panel)
    orders, after, summary = simulate_rebalance(
        account, target_frame, latest,
        cash=cash,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        fractional_shares=args.fractional_shares,
    )
    if args.commit_ledger:
        write_account_ledger(
            args.ledger_file, after, float(summary["cash_after"]),
            last_simulation_summary=summary,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(args.output_dir / "paper_trade_orders.csv", index=False, float_format="%.10f")
    after.to_csv(args.output_dir / "paper_trading_holdings_after.csv", index=False, float_format="%.10f")
    summary["ledger_file"] = str(args.ledger_file)
    summary["ledger_existed_before_run"] = ledger_exists
    summary["ledger_committed"] = bool(args.commit_ledger)
    (args.output_dir / "paper_trading_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
