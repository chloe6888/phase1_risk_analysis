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
    "requested_shares", "filled_shares", "unfilled_shares",
    "liquidity_limited_shares", "cash_limited_shares",
    "average_daily_volume", "participation_rate",
    "reference_price", "market_impact_bps", "effective_slippage_bps",
    "simulated_execution_price", "gross_notional",
    "transaction_fee", "slippage_cost", "status",
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


def load_average_daily_volume(
    volume_dir: Path,
    tickers: list[str],
    *,
    window: int = 20,
    max_data_age_calendar_days: int = 7,
    minimum_observations: int = 5,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Load trailing average share volume from local *_daily.csv files."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("volume window must be a positive integer.")
    if isinstance(max_data_age_calendar_days, bool) or not isinstance(max_data_age_calendar_days, int) or max_data_age_calendar_days < 0:
        raise ValueError("maximum volume-data age must be a non-negative integer.")
    if isinstance(minimum_observations, bool) or not isinstance(minimum_observations, int) or minimum_observations < 1:
        raise ValueError("minimum volume observations must be a positive integer.")

    adv: dict[str, float] = {}
    metadata: dict[str, dict[str, object]] = {}
    today = pd.Timestamp.now().date()
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        path = Path(volume_dir) / f"{ticker}_daily.csv"
        if not path.is_file():
            raise ValueError(f"Missing volume history file for {ticker}: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Volume history for {ticker} is empty.")
        date_column = next(
            (column for column in frame.columns if str(column).strip().lower() in {"date", "datetime"}),
            frame.columns[0],
        )
        volume_column = next(
            (column for column in frame.columns if str(column).strip().lower() == "volume"),
            None,
        )
        if volume_column is None:
            raise ValueError(f"Volume history for {ticker} has no Volume column.")
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        volumes = pd.to_numeric(frame[volume_column], errors="coerce")
        series = pd.Series(volumes.to_numpy(dtype=float), index=pd.DatetimeIndex(dates))
        series = series.loc[~series.index.isna()]
        if series.index.has_duplicates:
            raise ValueError(f"Volume history for {ticker} has duplicate dates.")
        series = series.sort_index().dropna()
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Volume history for {ticker} must be finite and non-negative.")
        trailing = series.tail(window)
        if len(trailing) < minimum_observations:
            raise ValueError(
                f"Only {len(trailing)} valid volume observations for {ticker}; "
                f"at least {minimum_observations} are required."
            )
        latest_date = pd.Timestamp(trailing.index[-1]).date()
        age_days = (today - latest_date).days
        if age_days < 0 or age_days > max_data_age_calendar_days:
            raise ValueError(
                f"Volume history for {ticker} is {age_days} calendar days old; "
                f"maximum allowed is {max_data_age_calendar_days}."
            )
        average = float(trailing.mean())
        if not math.isfinite(average) or average <= 0:
            raise ValueError(f"Average daily volume for {ticker} must be positive.")
        adv[ticker] = average
        metadata[ticker] = {
            "average_daily_volume_shares": average,
            "observations": int(len(trailing)),
            "latest_volume_date": latest_date.isoformat(),
            "age_calendar_days": int(age_days),
        }
    return adv, metadata


def simulate_rebalance(
    holdings: pd.DataFrame,
    target_weights: pd.DataFrame | Mapping[str, float],
    latest_prices: pd.Series | Mapping[str, float],
    *,
    cash: float,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    fractional_shares: bool = False,
    average_daily_volume: Mapping[str, float] | None = None,
    max_participation_rate: float = 0.05,
    market_impact_coefficient_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Simulate sell-first orders with optional ADV caps and square-root impact."""
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be finite and non-negative.")
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps < 10000:
        raise ValueError("slippage_bps must be between 0 and 10000.")
    if not math.isfinite(max_participation_rate) or not 0 < max_participation_rate <= 1:
        raise ValueError("max_participation_rate must be in (0, 1].")
    if not math.isfinite(market_impact_coefficient_bps) or market_impact_coefficient_bps < 0:
        raise ValueError("market_impact_coefficient_bps must be finite and non-negative.")

    current = normalize_holdings(holdings)
    targets = normalize_target_weights(target_weights)
    price_map = {str(k).strip().upper(): float(v) for k, v in dict(latest_prices).items()}
    if any(not math.isfinite(v) or v <= 0 for v in price_map.values()):
        raise ValueError("All supplied prices must be finite and positive.")
    tickers = sorted(set(current) | set(targets))
    missing = sorted(set(tickers) - set(price_map))
    if missing:
        raise ValueError(f"Missing current prices for: {missing}")

    adv_map: dict[str, float] | None = None
    if average_daily_volume is not None:
        adv_map = {str(k).strip().upper(): float(v) for k, v in dict(average_daily_volume).items()}
        missing_adv = sorted(set(tickers) - set(adv_map))
        if missing_adv:
            raise ValueError(f"Missing average daily volume for: {missing_adv}")
        if any(not math.isfinite(v) or v <= 0 for v in adv_map.values()):
            raise ValueError("Average daily volume must be finite and positive.")
    elif market_impact_coefficient_bps > 0:
        raise ValueError("market impact requires average_daily_volume data.")

    nav_before = float(cash + sum(current.get(t, 0.0) * price_map[t] for t in tickers))
    if not math.isfinite(nav_before) or nav_before <= 0:
        raise ValueError("Account net asset value must be positive.")
    fee_rate = transaction_cost_bps / 10000.0

    def round_shares(quantity: float) -> float:
        return float(quantity) if fractional_shares else float(math.floor(quantity + 1e-10))

    def liquidity_capacity(ticker: str, requested: float) -> float:
        if adv_map is None:
            return requested
        return min(requested, round_shares(adv_map[ticker] * max_participation_rate))

    def impact_bps(ticker: str, quantity: float) -> float:
        if adv_map is None or quantity <= 0 or market_impact_coefficient_bps == 0:
            return 0.0
        return market_impact_coefficient_bps * math.sqrt(min(1.0, quantity / adv_map[ticker]))

    desired_shares: dict[str, float] = {}
    for ticker in tickers:
        raw_target = nav_before * float(targets.get(ticker, 0.0)) / price_map[ticker]
        desired_shares[ticker] = round_shares(raw_target)

    sell_requested = {
        ticker: max(0.0, current.get(ticker, 0.0) - desired_shares[ticker])
        for ticker in tickers
    }
    buy_requested = {
        ticker: max(0.0, desired_shares[ticker] - current.get(ticker, 0.0))
        for ticker in tickers
    }
    sell_fills = {
        ticker: liquidity_capacity(ticker, quantity)
        for ticker, quantity in sell_requested.items()
    }
    buy_capacities = {
        ticker: liquidity_capacity(ticker, quantity)
        for ticker, quantity in buy_requested.items()
    }

    orders: list[dict[str, object]] = []
    cash_after_sells = float(cash)
    total_fees = 0.0
    total_slippage = 0.0
    final_shares = dict(current)
    liquidity_limited_orders = 0
    total_sell_shortfall = 0.0

    for ticker, requested in sell_requested.items():
        if requested <= 1e-12:
            continue
        fill = sell_fills[ticker]
        adv = adv_map.get(ticker) if adv_map is not None else None
        liquidity_shortfall = max(0.0, requested - fill)
        impact = impact_bps(ticker, fill)
        effective_slippage = slippage_bps + impact
        if effective_slippage >= 10000:
            raise ValueError("Effective sell slippage must remain below 10000 bps.")
        reference = price_map[ticker]
        execution = reference * (1.0 - effective_slippage / 10000.0) if fill > 0 else None
        notional = fill * float(execution) if fill > 0 else 0.0
        fee = notional * fee_rate
        slippage_cost = fill * (reference - float(execution)) if fill > 0 else 0.0
        cash_after_sells += notional - fee
        total_fees += fee
        total_slippage += slippage_cost
        final_shares[ticker] = current.get(ticker, 0.0) - fill
        total_sell_shortfall += liquidity_shortfall
        if liquidity_shortfall > 1e-9:
            liquidity_limited_orders += 1
            status = "PARTIALLY_FILLED_LIQUIDITY" if fill > 0 else "UNFILLED_NO_LIQUIDITY"
        else:
            status = "FILLED"
        orders.append({
            "ticker": ticker, "side": "SELL",
            "current_shares": current.get(ticker, 0.0),
            "desired_target_shares": desired_shares[ticker],
            "requested_shares": requested, "filled_shares": fill,
            "unfilled_shares": liquidity_shortfall,
            "liquidity_limited_shares": liquidity_shortfall, "cash_limited_shares": 0.0,
            "average_daily_volume": adv,
            "participation_rate": fill / adv if adv is not None else None,
            "reference_price": reference, "market_impact_bps": impact,
            "effective_slippage_bps": effective_slippage if fill > 0 else None,
            "simulated_execution_price": execution, "gross_notional": notional,
            "transaction_fee": fee, "slippage_cost": slippage_cost, "status": status,
        })

    estimated_buy_cost = 0.0
    for ticker, capacity in buy_capacities.items():
        if capacity <= 0:
            continue
        estimate_slippage = slippage_bps + impact_bps(ticker, capacity)
        if estimate_slippage >= 10000:
            raise ValueError("Effective buy slippage must remain below 10000 bps.")
        estimated_buy_cost += (
            capacity * price_map[ticker]
            * (1.0 + estimate_slippage / 10000.0)
            * (1.0 + fee_rate)
        )
    buy_scale = min(1.0, cash_after_sells / estimated_buy_cost) if estimated_buy_cost > 0 else 1.0

    total_buy_shortfall = 0.0
    liquidity_limited_buy_shares = 0.0
    for ticker, requested in buy_requested.items():
        if requested <= 1e-12:
            continue
        adv = adv_map.get(ticker) if adv_map is not None else None
        capacity = buy_capacities[ticker]
        liquidity_shortfall = max(0.0, requested - capacity)
        fill = round_shares(capacity * buy_scale)
        cash_shortfall = max(0.0, capacity - fill)
        impact = impact_bps(ticker, fill)
        effective_slippage = slippage_bps + impact
        if effective_slippage >= 10000:
            raise ValueError("Effective buy slippage must remain below 10000 bps.")
        reference = price_map[ticker]
        execution = reference * (1.0 + effective_slippage / 10000.0) if fill > 0 else None
        notional = fill * float(execution) if fill > 0 else 0.0
        fee = notional * fee_rate
        slippage_cost = fill * (float(execution) - reference) if fill > 0 else 0.0
        total_cost = notional + fee
        if total_cost > cash_after_sells + 1e-7:
            raise RuntimeError("Buy simulation exceeded available cash after scaling.")
        cash_after_sells -= total_cost
        total_fees += fee
        total_slippage += slippage_cost
        final_shares[ticker] = final_shares.get(ticker, 0.0) + fill
        cash_shortfall = max(0.0, capacity - fill)
        unfilled = max(0.0, requested - fill)
        total_buy_shortfall += unfilled
        liquidity_limited_buy_shares += liquidity_shortfall

        if liquidity_shortfall > 1e-9 and cash_shortfall > 1e-9:
            status = "PARTIALLY_FILLED_LIQUIDITY_AND_CASH" if fill > 0 else "UNFILLED_LIQUIDITY_AND_CASH"
        elif liquidity_shortfall > 1e-9:
            status = "PARTIALLY_FILLED_LIQUIDITY" if fill > 0 else "UNFILLED_NO_LIQUIDITY"
        elif cash_shortfall > 1e-9:
            status = "PARTIALLY_FILLED_CASH" if fill > 0 else "UNFILLED_NO_CASH"
        else:
            status = "FILLED"
        if liquidity_shortfall > 1e-9:
            liquidity_limited_orders += 1
        orders.append({
            "ticker": ticker, "side": "BUY",
            "current_shares": current.get(ticker, 0.0),
            "desired_target_shares": desired_shares[ticker],
            "requested_shares": requested, "filled_shares": fill,
            "unfilled_shares": unfilled,
            "liquidity_limited_shares": liquidity_shortfall, "cash_limited_shares": cash_shortfall,
            "average_daily_volume": adv,
            "participation_rate": fill / adv if adv is not None else None,
            "reference_price": reference, "market_impact_bps": impact,
            "effective_slippage_bps": effective_slippage if fill > 0 else None,
            "simulated_execution_price": execution, "gross_notional": notional,
            "transaction_fee": fee, "slippage_cost": slippage_cost, "status": status,
        })

    cash_after = max(0.0, float(cash_after_sells))
    holdings_rows: list[dict[str, object]] = []
    for ticker in tickers:
        shares = float(final_shares.get(ticker, 0.0))
        holdings_rows.append({
            "ticker": ticker, "shares": shares,
            "reference_price": price_map[ticker],
            "market_value": shares * price_map[ticker],
            "target_weight": float(targets.get(ticker, 0.0)),
        })
    nav_after = float(cash_after + sum(row["market_value"] for row in holdings_rows))
    holdings_after = pd.DataFrame(holdings_rows)
    holdings_after["actual_weight"] = (
        holdings_after["market_value"] / nav_after if nav_after > 0 else 0.0
    )
    orders_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    max_observed_participation = (
        float(orders_frame["participation_rate"].max())
        if not orders_frame.empty and orders_frame["participation_rate"].notna().any()
        else None
    )
    summary: dict[str, object] = {
        "nav_before": nav_before, "nav_after": nav_after,
        "cash_before": float(cash), "cash_after": cash_after,
        "cash_target_weight": max(0.0, 1.0 - sum(targets.values())),
        "target_risky_weight": float(sum(targets.values())),
        "total_transaction_fees": float(total_fees),
        "total_slippage_cost": float(total_slippage),
        "total_simulated_friction": float(total_fees + total_slippage),
        "orders_count": int(len(orders_frame)),
        "buy_orders_count": int((orders_frame["side"] == "BUY").sum()) if not orders_frame.empty else 0,
        "sell_orders_count": int((orders_frame["side"] == "SELL").sum()) if not orders_frame.empty else 0,
        "partial_orders_count": int(orders_frame["status"].str.startswith("PARTIALLY").sum()) if not orders_frame.empty else 0,
        "unfilled_orders_count": int(orders_frame["status"].str.startswith("UNFILLED").sum()) if not orders_frame.empty else 0,
        "liquidity_limited_orders_count": int(liquidity_limited_orders),
        "total_sell_unfilled_shares": float(total_sell_shortfall),
        "total_buy_unfilled_shares": float(total_buy_shortfall),
        "liquidity_limited_buy_shares": float(liquidity_limited_buy_shares),
        "buy_scale_due_to_cash": float(buy_scale),
        "max_participation_rate": float(max_participation_rate) if adv_map is not None else None,
        "max_observed_participation_rate": max_observed_participation,
        "average_daily_volume": adv_map,
        "market_impact_coefficient_bps": float(market_impact_coefficient_bps),
        "fractional_shares": bool(fractional_shares),
        "target_weights": targets,
        "methodology": {
            "sell_before_buy": True,
            "whole_shares_rounded_down": not fractional_shares,
            "buy_shortfall": "buy capacities are limited by ADV participation then by available cash",
            "liquidity": "each order is capped at ADV * participation rate" if adv_map is not None else "not applied; volume files not supplied",
            "market_impact": "additional slippage = coefficient_bps * sqrt(filled shares / ADV)",
            "execution_price": "reference close adjusted adversely by base slippage and optional impact",
            "commission": "charged on simulated execution notional",
            "no_broker_connection": True,
        },
        "warnings": [
            "This is an offline paper simulation at a reference close, not a live quote or guaranteed fill.",
            "ADV participation and square-root impact are configurable approximations, not calibrated intraday microstructure models.",
            "Taxes, halts, price limits, queue priority, intraday fills, and broker routing are not modeled.",
            "Review order quantities and reference prices before any real-world decision.",
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
    parser.add_argument("--volume-dir", type=Path, help="Optional directory containing TICKER_daily.csv files with Volume.")
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--max-volume-age-days", type=int, default=7)
    parser.add_argument("--max-participation-rate", type=float, default=0.05)
    parser.add_argument("--market-impact-coefficient-bps", type=float, default=0.0)
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
    average_daily_volume = None
    volume_metadata = {}
    if args.volume_dir is not None:
        tickers = sorted(
            set(normalize_holdings(account))
            | set(normalize_target_weights(target_frame))
        )
        average_daily_volume, volume_metadata = load_average_daily_volume(
            args.volume_dir, tickers,
            window=args.volume_window,
            max_data_age_calendar_days=args.max_volume_age_days,
        )
    orders, after, summary = simulate_rebalance(
        account, target_frame, latest,
        cash=cash,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        fractional_shares=args.fractional_shares,
        average_daily_volume=average_daily_volume,
        max_participation_rate=args.max_participation_rate,
        market_impact_coefficient_bps=args.market_impact_coefficient_bps,
    )
    summary["volume_data"] = volume_metadata
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
