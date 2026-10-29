"""Import and analyze a user's own long-only stock/ETF holdings from CSV.

The module contains no account credentials or brokerage integration. A CSV may
contain sensitive personal financial data and should remain outside public Git.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"ticker", "shares", "average_cost", "snapshot_price"}
OPTIONAL_COLUMNS = {"name", "risk_group", "subgroup", "currency", "notes", "quantity_source"}
DEFAULT_CURRENCY = "USD"


def load_holdings(path_or_buffer: str | Path | object) -> pd.DataFrame:
    """Load a holdings CSV and validate all positions are long-only and in one currency."""
    try:
        frame = pd.read_csv(path_or_buffer)
    except Exception as exc:
        raise ValueError(f"Could not read holdings CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("Holdings file is empty.")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Holdings CSV is missing required columns: {missing}")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    if frame["ticker"].eq("").any() or frame["ticker"].eq("NAN").any():
        raise ValueError("Every position must have a ticker.")
    if frame["ticker"].duplicated().any():
        duplicated = sorted(frame.loc[frame["ticker"].duplicated(keep=False), "ticker"].unique())
        raise ValueError(f"Duplicate tickers must be consolidated into one position: {duplicated}")
    for column in ("shares", "average_cost", "snapshot_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if (frame[column] <= 0).any():
            raise ValueError(f"{column} must contain only values greater than zero.")
    if "name" not in frame:
        frame["name"] = frame["ticker"]
    else:
        frame["name"] = frame["name"].fillna(frame["ticker"]).astype(str)
    if "risk_group" not in frame:
        frame["risk_group"] = "unclassified"
    else:
        frame["risk_group"] = frame["risk_group"].fillna("unclassified").astype(str).str.strip().str.lower()
        frame.loc[frame["risk_group"].eq(""), "risk_group"] = "unclassified"
    if "subgroup" not in frame:
        frame["subgroup"] = frame["risk_group"]
    else:
        frame["subgroup"] = frame["subgroup"].fillna(frame["risk_group"]).astype(str).str.strip().str.lower()
        mask = frame["subgroup"].eq("")
        frame.loc[mask, "subgroup"] = frame.loc[mask, "risk_group"]
    if "currency" not in frame:
        frame["currency"] = DEFAULT_CURRENCY
    else:
        frame["currency"] = frame["currency"].fillna(DEFAULT_CURRENCY).astype(str).str.upper().str.strip()
    if frame["currency"].eq("").any():
        frame.loc[frame["currency"].eq(""), "currency"] = DEFAULT_CURRENCY
    currencies = sorted(frame["currency"].unique())
    if len(currencies) != 1:
        raise ValueError(
            f"Mixed currencies are not supported in one portfolio calculation: {currencies}. "
            "Convert values to a single base currency first."
        )
    if "notes" not in frame:
        frame["notes"] = ""
    else:
        frame["notes"] = frame["notes"].fillna("").astype(str)
    if "quantity_source" not in frame:
        frame["quantity_source"] = "user_provided"
    else:
        frame["quantity_source"] = frame["quantity_source"].fillna("user_provided").astype(str)
    return frame.sort_values("ticker").reset_index(drop=True)


def summarize_holdings(
    holdings: pd.DataFrame,
    *,
    reference_prices: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Calculate per-position snapshot values, weights, unrealized P/L, and group concentration.

    If reference_prices is supplied, its valid values override snapshot_price per
    ticker. These are values for exposure analysis, not executable trading prices.
    """
    frame = holdings.copy()
    if frame.empty:
        raise ValueError("At least one holding is required.")
    prices: list[float] = []
    for row in frame.itertuples(index=False):
        reference = (reference_prices or {}).get(row.ticker, row.snapshot_price)
        if not np.isfinite(reference) or float(reference) <= 0:
            raise ValueError(f"{row.ticker}: reference price must be finite and positive.")
        prices.append(float(reference))
    frame["reference_price"] = prices
    frame["market_value"] = frame["shares"] * frame["reference_price"]
    frame["cost_basis"] = frame["shares"] * frame["average_cost"]
    frame["unrealized_pnl"] = frame["market_value"] - frame["cost_basis"]
    frame["unrealized_return_pct"] = frame["unrealized_pnl"] / frame["cost_basis"] * 100.0
    total_value = float(frame["market_value"].sum())
    if total_value <= 0:
        raise ValueError("Total portfolio market value must be positive.")
    frame["weight_pct"] = frame["market_value"] / total_value * 100.0
    group_values = frame.groupby("risk_group", dropna=False)["market_value"].sum()
    group_weights = group_values / total_value * 100.0
    subgroup_values = frame.groupby("subgroup", dropna=False)["market_value"].sum()
    subgroup_weights = subgroup_values / total_value * 100.0
    top = frame.sort_values("weight_pct", ascending=False).iloc[0]
    summary: dict[str, float | int | str] = {
        "currency": str(frame["currency"].iloc[0]),
        "position_count": int(len(frame)),
        "market_value": total_value,
        "cost_basis": float(frame["cost_basis"].sum()),
        "unrealized_pnl": float(frame["unrealized_pnl"].sum()),
        "unrealized_return_pct": float(frame["unrealized_pnl"].sum() / frame["cost_basis"].sum() * 100.0),
        "largest_position_ticker": str(top["ticker"]),
        "largest_position_weight_pct": float(top["weight_pct"]),
        "largest_risk_group": str(group_weights.idxmax()),
        "largest_risk_group_weight_pct": float(group_weights.max()),
        "risk_group_exposures_pct": {str(key): float(value) for key, value in group_weights.items()},
        "subgroup_exposures_pct": {str(key): float(value) for key, value in subgroup_weights.items()},
    }
    frame = frame.sort_values("market_value", ascending=False).reset_index(drop=True)
    return frame, summary


def weights_from_price_panel(
    holdings: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    *,
    max_staleness_calendar_days: int = 7,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Use each held ticker's last valid adjusted close to calculate portfolio weights.

    Different tickers can have different latest dates if their market calendars
    differ; last-price dates are returned explicitly and stale observations are
    rejected beyond max_staleness_calendar_days.
    """
    if max_staleness_calendar_days < 0:
        raise ValueError("max_staleness_calendar_days cannot be negative.")
    if adjusted_close.empty or adjusted_close.index.has_duplicates:
        raise ValueError("Adjusted-close panel must be non-empty with unique dates.")
    prices = adjusted_close.copy()
    prices.index = pd.to_datetime(prices.index)
    panel_end = pd.Timestamp(prices.index.max())
    unknown = sorted(set(holdings["ticker"]) - set(prices.columns))
    if unknown:
        raise ValueError(f"Holdings tickers absent from adjusted-close panel: {unknown}")
    rows: list[dict[str, object]] = []
    for row in holdings.itertuples(index=False):
        series = pd.to_numeric(prices[row.ticker], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"{row.ticker}: no valid adjusted-close data.")
        price_date = pd.Timestamp(series.index.max())
        price = float(series.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"{row.ticker}: last adjusted close must be finite and positive.")
        age_days = (panel_end.normalize() - price_date.normalize()).days
        if age_days > max_staleness_calendar_days:
            raise ValueError(
                f"{row.ticker}: last valid price is {age_days} calendar days older than the panel end; "
                "refresh/check data before analyzing the portfolio."
            )
        rows.append({
            "ticker": row.ticker,
            "shares": float(row.shares),
            "price": price,
            "price_date": price_date.date().isoformat(),
            "market_value": float(row.shares) * price,
            "currency": str(getattr(row, "currency", DEFAULT_CURRENCY)),
        })
    valuation = pd.DataFrame(rows)
    total_value = float(valuation["market_value"].sum())
    if total_value <= 0:
        raise ValueError("Total value based on latest adjusted prices must be positive.")
    valuation["weight"] = valuation["market_value"] / total_value
    weights = dict(zip(valuation["ticker"], valuation["weight"].astype(float)))
    return weights, valuation.sort_values("market_value", ascending=False).reset_index(drop=True)


def risk_guardrail_alerts(
    positions: pd.DataFrame,
    *,
    single_position_limit_pct: float = 20.0,
    risk_group_limit_pct: float = 40.0,
) -> list[dict[str, str | float]]:
    """Identify concentration-rule breaches for single tickers, themes, and subgroups."""
    if not 0 < single_position_limit_pct <= 100:
        raise ValueError("single_position_limit_pct must be in (0, 100].")
    if not 0 < risk_group_limit_pct <= 100:
        raise ValueError("risk_group_limit_pct must be in (0, 100].")
    alerts: list[dict[str, str | float]] = []
    for row in positions.itertuples(index=False):
        weight = float(row.weight_pct)
        if weight > single_position_limit_pct:
            alerts.append({
                "level": "warning",
                "kind": "single_position_concentration",
                "name": str(row.ticker),
                "weight_pct": weight,
                "message": (
                    f"{row.ticker} weighs {weight:.1f}% of listed holdings, above the "
                    f"{single_position_limit_pct:.1f}% review threshold."
                ),
            })
    for column, label, kind in (
        ("risk_group", "risk theme", "risk_group_concentration"),
        ("subgroup", "industry subgroup", "subgroup_concentration"),
    ):
        if column not in positions.columns:
            continue
        group_weights = positions.groupby(column)["market_value"].sum() / positions["market_value"].sum() * 100.0
        for group, weight in group_weights.items():
            if float(weight) > risk_group_limit_pct:
                alerts.append({
                    "level": "warning",
                    "kind": kind,
                    "group_type": label,
                    "name": str(group),
                    "weight_pct": float(weight),
                    "message": (
                        f"{label.capitalize()} '{group}' weighs {float(weight):.1f}% of listed holdings, above the "
                        f"{risk_group_limit_pct:.1f}% review threshold."
                    ),
                })
    return alerts

def reconcile_snapshot(
    calculated_market_value: float,
    calculated_unrealized_pnl: float,
    *,
    reported_market_value: float | None = None,
    reported_unrealized_pnl: float | None = None,
    tolerance: float = 1.0,
) -> dict[str, float | bool | None]:
    """Compare reconstructed rows to aggregate values shown by the broker UI."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative.")
    market_delta = None if reported_market_value is None else calculated_market_value - reported_market_value
    pnl_delta = None if reported_unrealized_pnl is None else calculated_unrealized_pnl - reported_unrealized_pnl
    return {
        "calculated_market_value": float(calculated_market_value),
        "reported_market_value": None if reported_market_value is None else float(reported_market_value),
        "market_value_delta": None if market_delta is None else float(market_delta),
        "calculated_unrealized_pnl": float(calculated_unrealized_pnl),
        "reported_unrealized_pnl": None if reported_unrealized_pnl is None else float(reported_unrealized_pnl),
        "unrealized_pnl_delta": None if pnl_delta is None else float(pnl_delta),
        "within_tolerance": bool(
            (market_delta is None or abs(market_delta) <= tolerance)
            and (pnl_delta is None or abs(pnl_delta) <= tolerance)
        ),
    }
