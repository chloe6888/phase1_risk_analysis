#!/usr/bin/env python3
"""Explainable position-sizing scenarios for overlapping memory-stock exposure.

This module is a transparent risk-budget aid, not a price predictor. It uses
latest validated close prices and user-editable target weights. It never places
orders. Fundamental valuation, official intraday quotes, taxes, margin and the
full portfolio outside the imported holdings must be considered separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from holdings_manager import load_holdings, weights_from_price_panel
from portfolio_risk import DEFAULT_OUTPUT_DIR, load_price_panel

ROOT = Path(__file__).resolve().parent
DEFAULT_HOLDINGS_CSV = ROOT / "data" / "private" / "holdings_snapshot.csv"
DEFAULT_CLOSE_PANEL = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_ADJ_CLOSE_PANEL = ROOT / "data" / "raw" / "adjusted_close_panel.csv"

MEMORY_TICKERS = ("MU", "SKHY", "DRAM")
DEFAULT_TARGET_WEIGHTS_PCT = {"MU": 8.0, "SKHY": 6.0, "DRAM": 5.0}
DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "balanced": {"MU": 8.0, "SKHY": 6.0, "DRAM": 5.0},
    "etf_core": {"MU": 0.0, "SKHY": 0.0, "DRAM": 15.0},
    "direct_only": {"MU": 8.0, "SKHY": 8.0, "DRAM": 0.0},
}
# Indicative look-through exposure using the 2026-10-29 daily holdings file:
# Micron direct + swap lines = 27.87%; SK hynix direct + swap + ADR lines = 21.82%.
# These are configuration assumptions, not fixed weights; refresh before trading.
DEFAULT_DRAM_LOOKTHROUGH = {"MU": 0.2787, "SKHY": 0.2182}
DEFAULT_LOOKTHROUGH_ASOF = "2026-10-29"
DEFAULT_ADD_BUFFER_PCT = 3.0
DEFAULT_MEMORY_LIMIT_PCT = 25.0
DEFAULT_ISSUER_LIMIT_PCT = 12.0


def validate_percent(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be finite and between 0 and 100.")


def parse_target_weights(value: str) -> dict[str, float]:
    """Parse a target such as MU=8,SKHY=6,DRAM=5 (percent of reference portfolio)."""
    result: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("Target weights must use TICKER=percent, e.g. MU=8,SKHY=6,DRAM=5.")
        ticker, raw = item.split("=", 1)
        ticker = ticker.strip().upper()
        if ticker not in MEMORY_TICKERS or ticker in result:
            raise argparse.ArgumentTypeError(f"Unsupported or duplicate target ticker: {ticker!r}.")
        try:
            pct = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid target percentage for {ticker}: {raw!r}.") from exc
        if not np.isfinite(pct) or not 0 <= pct <= 100:
            raise argparse.ArgumentTypeError(f"Target percentage for {ticker} must be between 0 and 100.")
        result[ticker] = pct
    if not result:
        raise argparse.ArgumentTypeError("At least one ticker target is required.")
    for ticker in MEMORY_TICKERS:
        result.setdefault(ticker, 0.0)
    return result


def calculate_trend_diagnostics(adjusted_close: pd.DataFrame | None) -> pd.DataFrame:
    """Return trailing returns, realized volatility and a descriptive trend flag."""
    rows: list[dict[str, object]] = []
    for ticker in MEMORY_TICKERS:
        row: dict[str, object] = {
            "ticker": ticker,
            "return_20d_pct": None,
            "return_60d_pct": None,
            "realized_volatility_20d_annualized_pct": None,
            "price_vs_ma20_pct": None,
            "price_vs_ma60_pct": None,
            "trend_flag": "INSUFFICIENT_HISTORY",
        }
        if adjusted_close is None or ticker not in adjusted_close.columns:
            rows.append(row)
            continue
        series = pd.to_numeric(adjusted_close[ticker], errors="coerce").dropna()
        if len(series) >= 21:
            row["return_20d_pct"] = float((series.iloc[-1] / series.iloc[-21] - 1.0) * 100.0)
            ma20 = float(series.tail(20).mean())
            row["price_vs_ma20_pct"] = float((series.iloc[-1] / ma20 - 1.0) * 100.0) if ma20 else None
        if len(series) >= 61:
            row["return_60d_pct"] = float((series.iloc[-1] / series.iloc[-61] - 1.0) * 100.0)
            ma60 = float(series.tail(60).mean())
            row["price_vs_ma60_pct"] = float((series.iloc[-1] / ma60 - 1.0) * 100.0) if ma60 else None
        returns = series.pct_change(fill_method=None).dropna().tail(20)
        if len(returns) >= 10:
            row["realized_volatility_20d_annualized_pct"] = float(returns.std(ddof=1) * np.sqrt(252) * 100.0)
        r20, r60, vs20 = row["return_20d_pct"], row["return_60d_pct"], row["price_vs_ma20_pct"]
        if r20 is not None and r60 is not None and vs20 is not None:
            if r20 < 0 and r60 < 0 and vs20 < 0:
                row["trend_flag"] = "WEAK_TREND_NO_ADD"
            elif r20 > 0 and r60 > 0 and vs20 > 0:
                row["trend_flag"] = "POSITIVE_TREND_BUT_SIZE_LIMITS_STILL_APPLY"
            else:
                row["trend_flag"] = "MIXED_TREND"
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def _target_shares(target_value: float, price: float, *, fractional_shares: bool) -> float | int:
    quantity = max(target_value / price, 0.0)
    if fractional_shares:
        return round(quantity, 4)
    return int(math.floor(quantity + 0.5))


def analyze_memory_positions(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    adjusted_close: pd.DataFrame | None = None,
    portfolio_value: float | None = None,
    max_memory_weight_pct: float = DEFAULT_MEMORY_LIMIT_PCT,
    max_issuer_exposure_pct: float = DEFAULT_ISSUER_LIMIT_PCT,
    target_scenarios: Mapping[str, Mapping[str, float]] | None = None,
    dram_lookthrough: Mapping[str, float] | None = None,
    lookthrough_asof: str = DEFAULT_LOOKTHROUGH_ASOF,
    fractional_shares: bool = False,
    max_staleness_calendar_days: int = 7,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Calculate overlapping exposure and share-count scenarios without placing orders.

    portfolio_value should be net portfolio equity including cash if known.
    Otherwise the denominator is the market value of imported positions only.
    Target weights are percentages of that denominator, not buy/sell signals.
    """
    validate_percent(max_memory_weight_pct, "max_memory_weight_pct")
    validate_percent(max_issuer_exposure_pct, "max_issuer_exposure_pct")
    if portfolio_value is not None and (not np.isfinite(portfolio_value) or portfolio_value <= 0):
        raise ValueError("portfolio_value must be finite and positive when provided.")
    if holdings.empty:
        raise ValueError("Holdings cannot be empty.")
    if not set(MEMORY_TICKERS).issubset(set(holdings["ticker"])):
        missing = sorted(set(MEMORY_TICKERS) - set(holdings["ticker"]))
        raise ValueError(f"Holdings file must include MU, SKHY and DRAM to analyze their overlap; missing: {missing}.")

    lookthrough = dict(DEFAULT_DRAM_LOOKTHROUGH if dram_lookthrough is None else dram_lookthrough)
    for ticker in ("MU", "SKHY"):
        if ticker not in lookthrough:
            raise ValueError(f"DRAM look-through configuration is missing {ticker}.")
        fraction = float(lookthrough[ticker])
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f"DRAM look-through for {ticker} must be between 0 and 1.")
        lookthrough[ticker] = fraction
    if sum(float(lookthrough[ticker]) for ticker in ("MU", "SKHY")) > 1.0 + 1e-9:
        raise ValueError("DRAM look-through exposures for MU and SKHY cannot sum above 100%.")

    _, valuations = weights_from_price_panel(
        holdings, close_prices, max_staleness_calendar_days=max_staleness_calendar_days
    )
    valuations = valuations.copy()
    valuations["market_value"] = valuations["shares"] * valuations["price"]
    calculated_listed_value = float(valuations["market_value"].sum())
    denominator = float(portfolio_value) if portfolio_value is not None else calculated_listed_value
    denominator_type = "user_supplied_net_portfolio_equity" if portfolio_value is not None else "imported_listed_positions_only"
    value_map = dict(zip(valuations["ticker"], valuations["market_value"]))
    price_map = dict(zip(valuations["ticker"], valuations["price"]))
    shares_map = dict(zip(valuations["ticker"], valuations["shares"]))
    date_map = dict(zip(valuations["ticker"], valuations["price_date"]))

    current_memory_value = float(sum(value_map.get(ticker, 0.0) for ticker in MEMORY_TICKERS))
    current_memory_weight = current_memory_value / denominator * 100.0
    permitted_memory_value = denominator * max_memory_weight_pct / 100.0
    excess_value = max(0.0, current_memory_value - permitted_memory_value)

    dram_value = float(value_map["DRAM"])
    lookthrough_values = {
        "MU": float(value_map["MU"]) + dram_value * float(lookthrough["MU"]),
        "SKHY": float(value_map["SKHY"]) + dram_value * float(lookthrough["SKHY"]),
    }
    issuer_weights = {ticker: value / denominator * 100.0 for ticker, value in lookthrough_values.items()}
    diagnostics = calculate_trend_diagnostics(adjusted_close)
    position_rows: list[dict[str, object]] = []
    for ticker in MEMORY_TICKERS:
        current_shares = float(shares_map[ticker])
        position_value = float(value_map[ticker])
        direct_weight = position_value / denominator * 100.0
        additional_lookthrough = dram_value * float(lookthrough.get(ticker, 0.0)) if ticker in ("MU", "SKHY") else 0.0
        trend = diagnostics.loc[ticker].to_dict()
        if current_memory_weight > max_memory_weight_pct:
            add_permission = "NO_ADD_MEMORY_LIMIT"
        elif current_memory_weight > max(0.0, max_memory_weight_pct - DEFAULT_ADD_BUFFER_PCT):
            add_permission = "WAIT_FOR_MEMORY_HEADROOM"
        elif ticker in issuer_weights and issuer_weights[ticker] > max_issuer_exposure_pct:
            add_permission = "NO_ADD_ISSUER_LOOKTHROUGH_LIMIT"
        elif trend.get("trend_flag") != "POSITIVE_TREND_BUT_SIZE_LIMITS_STILL_APPLY":
            add_permission = "WAIT_FOR_TREND_CONFIRMATION"
        else:
            add_permission = "STAGED_ADD_REVIEW_ONLY"
        position_rows.append({
            "ticker": ticker,
            "current_shares": current_shares,
            "latest_close": float(price_map[ticker]),
            "price_date": date_map[ticker],
            "market_value": position_value,
            "weight_of_reference_portfolio_pct": direct_weight,
            "dram_lookthrough_additional_value": additional_lookthrough,
            "lookthrough_issuer_exposure_value": lookthrough_values.get(ticker, position_value),
            "lookthrough_issuer_exposure_pct": issuer_weights.get(ticker, direct_weight),
            "add_permission": add_permission,
            **trend,
        })
    current_positions = pd.DataFrame(position_rows).set_index("ticker")

    scenarios = {name: dict(values) for name, values in (target_scenarios or DEFAULT_SCENARIOS).items()}
    if "balanced" not in scenarios:
        raise ValueError("At least a 'balanced' target scenario is required.")
    # Minimum-trim option: preserve the current DRAM ETF quantity and scale
    # only direct MU/SKHY holdings. The scale must satisfy both the combined
    # memory-sleeve cap and the look-through single-issuer caps. Whole-share
    # quantities are rounded down so the resulting scenario cannot cross a
    # hard limit merely because of rounding.
    if "cap_only_preserve_etf" not in scenarios:
        dram_value = float(value_map["DRAM"])
        dram_weight_pct = dram_value / denominator * 100.0
        memory_direct_budget = max(0.0, denominator * max_memory_weight_pct / 100.0 - dram_value)
        issuer_budgets = {
            ticker: max(
                0.0,
                denominator * max_issuer_exposure_pct / 100.0
                - dram_value * float(lookthrough[ticker]),
            )
            for ticker in ("MU", "SKHY")
        }
        direct_values = {ticker: float(value_map[ticker]) for ticker in ("MU", "SKHY")}
        direct_total_value = sum(direct_values.values())
        scales = [1.0]
        if direct_total_value > 0:
            scales.append(memory_direct_budget / direct_total_value)
        for ticker in ("MU", "SKHY"):
            if direct_values[ticker] > 0:
                scales.append(issuer_budgets[ticker] / direct_values[ticker])
        proportional_scale = min(max(value, 0.0) for value in scales)
        cap_quantities: dict[str, float] = {}
        if fractional_shares:
            for ticker in ("MU", "SKHY"):
                quantity_limit = min(
                    float(shares_map[ticker]) * proportional_scale,
                    issuer_budgets[ticker] / float(price_map[ticker]),
                )
                cap_quantities[ticker] = round(max(0.0, quantity_limit), 4)
        else:
            max_mu = min(
                int(math.floor(float(shares_map["MU"]) + 1e-9)),
                int(math.floor(issuer_budgets["MU"] / float(price_map["MU"]) + 1e-9)),
            )
            max_skhy = min(
                int(math.floor(float(shares_map["SKHY"]) + 1e-9)),
                int(math.floor(issuer_budgets["SKHY"] / float(price_map["SKHY"]) + 1e-9)),
            )
            desired = {
                ticker: float(shares_map[ticker]) * proportional_scale
                for ticker in ("MU", "SKHY")
            }
            best_value = -1.0
            best_distance = float("inf")
            best_pair = (0, 0)
            for mu_quantity in range(max_mu + 1):
                mu_value = mu_quantity * float(price_map["MU"])
                for skhy_quantity in range(max_skhy + 1):
                    skhy_value = skhy_quantity * float(price_map["SKHY"])
                    direct_value = mu_value + skhy_value
                    if direct_value > memory_direct_budget + 1e-8:
                        continue
                    distance = (
                        (mu_quantity - desired["MU"]) ** 2
                        + (skhy_quantity - desired["SKHY"]) ** 2
                    )
                    if direct_value > best_value + 1e-8 or (
                        abs(direct_value - best_value) <= 1e-8 and distance < best_distance
                    ):
                        best_value = direct_value
                        best_distance = distance
                        best_pair = (mu_quantity, skhy_quantity)
            cap_quantities["MU"], cap_quantities["SKHY"] = map(float, best_pair)
        cap_targets = {
            ticker: cap_quantities[ticker] * float(price_map[ticker]) / denominator * 100.0
            for ticker in ("MU", "SKHY")
        }
        cap_targets["DRAM"] = dram_weight_pct
        scenarios["cap_only_preserve_etf"] = cap_targets
    scenario_rows: list[dict[str, object]] = []
    for scenario_name, targets in scenarios.items():
        normalized_targets: dict[str, float] = {}
        for ticker in MEMORY_TICKERS:
            value = float(targets.get(ticker, 0.0))
            validate_percent(value, f"{scenario_name} target weight for {ticker}")
            normalized_targets[ticker] = value
        scenario_total = sum(normalized_targets.values())
        for ticker in MEMORY_TICKERS:
            price = float(price_map[ticker])
            current_shares = float(shares_map[ticker])
            target_weight = normalized_targets[ticker]
            target_value = denominator * target_weight / 100.0
            target_shares = _target_shares(target_value, price, fractional_shares=fractional_shares)
            if float(target_shares) < current_shares - 1e-9:
                delta = float(target_shares) - current_shares
                action = "REDUCE_TO_TARGET"
                reason = f"Current {ticker} position exceeds this scenario's target weight ({target_weight:.1f}%)."
            elif float(target_shares) > current_shares + 1e-9:
                delta = float(target_shares) - current_shares
                action = "HOLD_NO_AUTO_ADD"
                reason = "Below target; no automatic addition without a separate entry rule."
            else:
                delta = 0.0
                action = "HOLD"
                reason = "At the rounded target quantity for this scenario."
            scenario_rows.append({
                "scenario": scenario_name,
                "scenario_total_target_weight_pct": scenario_total,
                "scenario_exceeds_memory_limit": bool(scenario_total > max_memory_weight_pct + 1e-9),
                "ticker": ticker,
                "current_shares": current_shares,
                "latest_close": price,
                "target_weight_pct": target_weight,
                "target_value": target_value,
                "target_shares": target_shares,
                "share_delta": delta,
                "action": action,
                "reason": reason,
            })
    decision_table = pd.DataFrame(scenario_rows)

    # Recalculate look-through issuer exposure for every proposed target mix.
    # This is essential when a scenario preserves an ETF whose own holdings can
    # already breach an issuer cap even after direct shares are reduced to zero.
    scenario_lookthrough_exposures_pct: dict[str, dict[str, float]] = {}
    scenario_issuer_limit_status: dict[str, dict[str, bool]] = {}
    scenario_memory_weights_pct: dict[str, float] = {}
    for scenario_name, targets in scenarios.items():
        normalized_targets = {
            ticker: float(targets.get(ticker, 0.0)) for ticker in MEMORY_TICKERS
        }
        scenario_memory_weights_pct[scenario_name] = sum(normalized_targets.values())
        scenario_lookthrough_exposures_pct[scenario_name] = {
            ticker: normalized_targets[ticker] + normalized_targets["DRAM"] * float(lookthrough[ticker])
            for ticker in ("MU", "SKHY")
        }
        scenario_issuer_limit_status[scenario_name] = {
            ticker: exposure > max_issuer_exposure_pct + 1e-9
            for ticker, exposure in scenario_lookthrough_exposures_pct[scenario_name].items()
        }

    issuer_alerts = [
        {
            "ticker": ticker,
            "lookthrough_exposure_pct": float(weight),
            "threshold_pct": float(max_issuer_exposure_pct),
            "message": f"Estimated look-through exposure to {ticker} is {weight:.1f}%, above the {max_issuer_exposure_pct:.1f}% review threshold.",
        }
        for ticker, weight in issuer_weights.items()
        if weight > max_issuer_exposure_pct
    ]
    if current_memory_weight > max_memory_weight_pct:
        overall_action = "REDUCE_MEMORY_SLEEVE"
        overall_reason = (
            f"MU + SKHY + DRAM represent {current_memory_weight:.1f}% of the reference portfolio, "
            f"above the {max_memory_weight_pct:.1f}% policy limit. Current excess is approximately {excess_value:,.2f} in portfolio currency."
        )
    else:
        overall_action = "WITHIN_MEMORY_LIMIT"
        overall_reason = (
            f"MU + SKHY + DRAM represent {current_memory_weight:.1f}% of the reference portfolio, "
            f"within the {max_memory_weight_pct:.1f}% policy limit."
        )

    summary: dict[str, object] = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "reference_portfolio_value": denominator,
        "reference_portfolio_value_type": denominator_type,
        "imported_listed_positions_market_value": calculated_listed_value,
        "currency": str(holdings["currency"].iloc[0]) if "currency" in holdings.columns else "USD",
        "denominator_warning": (
            "Percentages are relative to the full net portfolio equity supplied by the user."
            if portfolio_value is not None else
            "Cash, bonds, other accounts and financing are not included. Percentages apply only to imported listed positions; supply net portfolio equity to use account-level weights."
        ),
        "latest_price_dates": date_map,
        "memory_tickers": list(MEMORY_TICKERS),
        "memory_sleeve_market_value": current_memory_value,
        "memory_sleeve_weight_pct": current_memory_weight,
        "max_memory_weight_pct": max_memory_weight_pct,
        "memory_value_above_limit": excess_value,
        "overall_action": overall_action,
        "overall_reason": overall_reason,
        "max_issuer_exposure_pct": max_issuer_exposure_pct,
        "lookthrough_asof": lookthrough_asof,
        "dram_lookthrough_weights": {key: float(value) for key, value in lookthrough.items()},
        "lookthrough_exposure_values": lookthrough_values,
        "lookthrough_issuer_weights_pct": issuer_weights,
        "issuer_alerts": issuer_alerts,
        "default_balanced_targets_pct": DEFAULT_TARGET_WEIGHTS_PCT,
        "scenario_memory_weights_pct": scenario_memory_weights_pct,
        "scenario_lookthrough_exposures_pct": scenario_lookthrough_exposures_pct,
        "scenario_issuer_limit_status": scenario_issuer_limit_status,
        "decision_method_note": (
            "Position-sizing guardrails and trailing price diagnostics only. No company valuation, earnings forecast, "
            "or probability-of-return model is included. Target scenarios are editable policy assumptions, not optimized weights."
        ),
    }
    return summary, current_positions, decision_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze overlapping MU/SKHY/DRAM positions using risk-budget rules; no orders are placed."
    )
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS_CSV)
    parser.add_argument("--prices", type=Path, default=DEFAULT_CLOSE_PANEL, help="Raw close price panel")
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_ADJ_CLOSE_PANEL, help="Adjusted close panel for trend diagnostics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--portfolio-value", type=float, help="Optional full net portfolio equity including cash and liabilities")
    parser.add_argument("--max-memory-weight-pct", type=float, default=DEFAULT_MEMORY_LIMIT_PCT)
    parser.add_argument("--max-issuer-exposure-pct", type=float, default=DEFAULT_ISSUER_LIMIT_PCT)
    parser.add_argument("--target-weights", type=parse_target_weights, default=None,
                        help="Balanced scenario weights as percentages, e.g. MU=8,SKHY=6,DRAM=5")
    parser.add_argument("--dram-mu-exposure", type=float, default=DEFAULT_DRAM_LOOKTHROUGH["MU"],
                        help="Assumed MU look-through share of DRAM ETF, decimal from 0 to 1")
    parser.add_argument("--dram-skhy-exposure", type=float, default=DEFAULT_DRAM_LOOKTHROUGH["SKHY"],
                        help="Assumed SKHY look-through share of DRAM ETF, decimal from 0 to 1")
    parser.add_argument("--lookthrough-asof", default=DEFAULT_LOOKTHROUGH_ASOF,
                        help="Date of the manually verified DRAM holdings assumption")
    parser.add_argument("--fractional-shares", action="store_true",
                        help="Allow 4-decimal target quantities; default rounds to whole shares")
    args = parser.parse_args()
    if args.dram_mu_exposure < 0 or args.dram_mu_exposure > 1 or args.dram_skhy_exposure < 0 or args.dram_skhy_exposure > 1:
        parser.error("DRAM look-through fractions must be between zero and one.")
    holdings = load_holdings(args.holdings_csv)
    close_prices = load_price_panel(args.prices)
    adjusted = load_price_panel(args.adjusted_prices) if args.adjusted_prices.exists() else None
    scenarios = {name: dict(values) for name, values in DEFAULT_SCENARIOS.items()}
    if args.target_weights is not None:
        scenarios["balanced"] = args.target_weights
    summary, positions, decisions = analyze_memory_positions(
        holdings,
        close_prices,
        adjusted_close=adjusted,
        portfolio_value=args.portfolio_value,
        max_memory_weight_pct=args.max_memory_weight_pct,
        max_issuer_exposure_pct=args.max_issuer_exposure_pct,
        target_scenarios=scenarios,
        dram_lookthrough={"MU": args.dram_mu_exposure, "SKHY": args.dram_skhy_exposure},
        lookthrough_asof=args.lookthrough_asof,
        fractional_shares=args.fractional_shares,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(args.output_dir / "memory_position_diagnostics.csv", float_format="%.8f")
    decisions.to_csv(args.output_dir / "memory_allocation_scenarios.csv", index=False, float_format="%.8f")
    (args.output_dir / "memory_allocation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
