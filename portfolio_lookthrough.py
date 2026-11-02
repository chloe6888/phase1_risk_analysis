"""Combine direct holdings with dated ETF look-through exposure snapshots.

The result is exposure by underlying issuer as a percentage of total portfolio
equity. ETF holdings are multiplied by the portfolio weight of that ETF; direct
holdings are then added once. This is an overlap analysis, not a portfolio
revaluation or a buy/sell signal.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from etf_lookthrough import load_issuer_snapshot

DEFAULT_FUND_TICKERS = ("DRAM", "QQQ")


def _as_date(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("as_of_date must use YYYY-MM-DD.") from exc


def analyze_portfolio_lookthrough(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    snapshot_paths: Sequence[str | Path],
    *,
    fund_tickers: Iterable[str] = DEFAULT_FUND_TICKERS,
    cash: float | None = None,
    as_of_date: date | str | None = None,
    max_price_age_calendar_days: int = 7,
    max_snapshot_age_calendar_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Calculate combined direct + indirect issuer exposure using dated snapshots.

    If cash is omitted, it is temporarily treated as zero and the summary flags
    that the denominator covers the imported securities only.
    """
    if holdings.empty:
        raise ValueError("Holdings cannot be empty.")
    required = {"ticker", "shares"}
    missing = sorted(required - set(holdings.columns))
    if missing:
        raise ValueError(f"Holdings are missing required columns: {missing}")
    if isinstance(max_price_age_calendar_days, bool) or max_price_age_calendar_days < 0:
        raise ValueError("max_price_age_calendar_days must be a non-negative integer.")
    if isinstance(max_snapshot_age_calendar_days, bool) or max_snapshot_age_calendar_days < 0:
        raise ValueError("max_snapshot_age_calendar_days must be a non-negative integer.")

    reference_date = _as_date(as_of_date)
    cash_value = 0.0 if cash is None else float(cash)
    if not np.isfinite(cash_value) or cash_value < 0:
        raise ValueError("cash must be finite and non-negative.")

    positions = holdings.copy()
    positions["ticker"] = positions["ticker"].astype(str).str.strip().str.upper()
    positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce")
    if positions["ticker"].eq("").any() or positions["ticker"].duplicated().any():
        raise ValueError("Holdings must have unique, non-empty tickers.")
    quantities = positions["shares"].to_numpy(dtype=float)
    if not np.isfinite(quantities).all() or (quantities <= 0).any():
        raise ValueError("Holding shares must be finite and positive.")

    prices = close_prices.copy()
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Close-price panel must be non-empty with unique dates.")
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.hasnans:
        raise ValueError("Close-price panel contains invalid dates.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")

    fund_set = {str(ticker).strip().upper() for ticker in fund_tickers}
    held_tickers = set(positions["ticker"])
    missing_prices = sorted(held_tickers - set(map(str, prices.columns)))
    if missing_prices:
        raise ValueError(f"Close-price panel is missing held tickers: {missing_prices}.")

    valuation_rows: list[dict[str, object]] = []
    for row in positions.itertuples(index=False):
        series = prices[row.ticker].dropna()
        if series.empty:
            raise ValueError(f"{row.ticker}: no valid raw close price.")
        last_date = pd.Timestamp(series.index[-1]).date()
        age = (reference_date - last_date).days
        if age < 0:
            raise ValueError(f"{row.ticker}: latest raw close is later than the analysis date.")
        if age > max_price_age_calendar_days:
            raise ValueError(
                f"{row.ticker}: latest raw close is {age} days old, above the {max_price_age_calendar_days}-day limit."
            )
        price = float(series.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"{row.ticker}: latest raw close must be finite and positive.")
        valuation_rows.append({
            "ticker": row.ticker,
            "shares": float(row.shares),
            "price": price,
            "price_date": last_date.isoformat(),
            "market_value": float(row.shares) * price,
        })

    valuation = pd.DataFrame(valuation_rows)
    listed_value = float(valuation["market_value"].sum())
    equity = listed_value + cash_value
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("Portfolio equity including cash must be positive.")
    valuation["portfolio_weight_pct"] = valuation["market_value"] / equity * 100.0
    weight_map = dict(zip(valuation["ticker"], valuation["portfolio_weight_pct"]))

    issuer_direct: dict[str, float] = defaultdict(float)
    issuer_indirect: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ticker, weight in weight_map.items():
        if ticker not in fund_set:
            issuer_direct[ticker] += float(weight)

    snapshot_by_fund: dict[str, tuple[dict[str, float], str, dict[str, object], Path]] = {}
    warnings: list[str] = []
    fund_rows: list[dict[str, object]] = []
    for raw_path in snapshot_paths:
        source_path = Path(raw_path)
        issuer_weights, snapshot_asof, manifest = load_issuer_snapshot(source_path)
        fund = str(manifest.get("fund_ticker", "")).strip().upper()
        if not fund:
            raise ValueError(f"{source_path.name}: snapshot manifest is missing fund_ticker.")
        if fund in snapshot_by_fund:
            raise ValueError(f"More than one snapshot was supplied for {fund}; select one dated snapshot.")
        if fund not in fund_set:
            raise ValueError(f"{fund}: snapshot ticker is not in configured fund_tickers.")
        if fund not in held_tickers:
            raise ValueError(f"{fund}: snapshot was supplied but this ETF is not in the current holdings.")
        asof = _as_date(snapshot_asof)
        age = (reference_date - asof).days
        if age < 0:
            warnings.append(f"{fund} 快照日期 {snapshot_asof} 晚于分析日期，需核对日期。")
        elif age > max_snapshot_age_calendar_days:
            warnings.append(
                f"{fund} 持仓快照已过期 {age} 天（阈值 {max_snapshot_age_calendar_days} 天）；请刷新官方披露文件。"
            )
        for warning in manifest.get("warnings", []):
            warnings.append(f"{fund}：{warning}")
        snapshot_by_fund[fund] = (issuer_weights, snapshot_asof, manifest, source_path)

    for fund in sorted(fund_set & held_tickers):
        position_weight = float(weight_map[fund])
        if fund not in snapshot_by_fund:
            fund_rows.append({
                "fund_ticker": fund,
                "portfolio_weight_pct": position_weight,
                "snapshot_as_of": None,
                "snapshot_age_days": None,
                "mapped_target_issuer_weight_pct": None,
                "unmapped_weight_pct": None,
                "cash_or_cash_equivalent_weight_pct": None,
                "status": "missing_snapshot",
                "source_name": None,
            })
            warnings.append(f"{fund} 尚无带日期的 ETF 官方持仓快照；当前未计入其底层发行人敞口。")
            continue
        issuer_weights, snapshot_asof, manifest, source_path = snapshot_by_fund[fund]
        snapshot_age = (reference_date - _as_date(snapshot_asof)).days
        for issuer, issuer_fraction in issuer_weights.items():
            issuer_indirect[issuer][fund] += position_weight * float(issuer_fraction)
        fund_rows.append({
            "fund_ticker": fund,
            "portfolio_weight_pct": position_weight,
            "snapshot_as_of": snapshot_asof,
            "snapshot_age_days": snapshot_age,
            "mapped_target_issuer_weight_pct": manifest.get("mapped_target_issuer_weight_pct"),
            "unmapped_weight_pct": manifest.get("unmapped_weight_pct"),
            "cash_or_cash_equivalent_weight_pct": manifest.get("cash_or_cash_equivalent_weight_pct"),
            "status": (
                "future_snapshot" if snapshot_age < 0
                else "stale_snapshot" if snapshot_age > max_snapshot_age_calendar_days
                else "loaded"
            ),
            "source_name": manifest.get("source_name"),
            "source_file": manifest.get("source_file", source_path.name),
        })

    issuer_set = set(issuer_direct)
    issuer_set.update(issuer_indirect)
    fund_columns = [f"{fund}_indirect_weight_pct" for fund in sorted(fund_set)]
    rows: list[dict[str, object]] = []
    for issuer in sorted(issuer_set):
        direct = float(issuer_direct.get(issuer, 0.0))
        by_fund = issuer_indirect.get(issuer, {})
        row: dict[str, object] = {
            "issuer": issuer,
            "direct_weight_pct": direct,
        }
        total = direct
        for fund in sorted(fund_set):
            value = float(by_fund.get(fund, 0.0))
            row[f"{fund}_indirect_weight_pct"] = value
            total += value
        row["total_effective_exposure_pct"] = total
        rows.append(row)
    exposure_columns = ["issuer", "direct_weight_pct", *fund_columns, "total_effective_exposure_pct"]
    exposure_frame = pd.DataFrame(rows, columns=exposure_columns)
    if not exposure_frame.empty:
        exposure_frame = exposure_frame.sort_values(
            "total_effective_exposure_pct", ascending=False, kind="stable"
        ).reset_index(drop=True)

    fund_frame = pd.DataFrame(fund_rows)
    missing_funds = sorted(
        fund for fund in fund_set & held_tickers if fund not in snapshot_by_fund
    )
    summary: dict[str, object] = {
        "as_of_date": reference_date.isoformat(),
        "position_count": int(len(positions)),
        "portfolio_equity_value": float(equity),
        "cash_value": cash_value,
        "cash_source": "user_provided" if cash is not None else "not_provided_assumed_zero",
        "listed_positions_value": listed_value,
        "latest_price_dates": dict(zip(valuation["ticker"], valuation["price_date"])),
        "fund_tickers_held": sorted(fund_set & held_tickers),
        "fund_snapshots_loaded": sorted(snapshot_by_fund),
        "fund_snapshots_missing": missing_funds,
        "warnings": warnings,
        "issuer_exposure_denominator": "portfolio equity including supplied cash; if cash was omitted, listed positions only",
        "methodology": "effective issuer exposure = direct issuer weight + sum(ETF portfolio weight * dated ETF issuer weight)",
        "limitations": [
            "Only issuers recognized by the supplied ETF issuer-matching maps are shown; unmapped holdings remain outside issuer rows.",
            "ETF snapshots are dated and can become stale; stale snapshots are flagged, not silently refreshed.",
            "Effective issuer exposures overlap and must not be summed across issuers as if they were mutually exclusive portfolio weights.",
            "This analysis does not calculate ETF sector exposure or look through ETFs without a supplied holdings snapshot.",
        ],
    }
    return exposure_frame, fund_frame, summary


if __name__ == "__main__":
    import argparse
    import json

    from holdings_manager import load_holdings
    from portfolio_risk import ROOT, load_price_panel

    parser = argparse.ArgumentParser(description="Combine direct portfolio positions with dated ETF issuer look-through.")
    parser.add_argument("--holdings-csv", type=Path, default=ROOT / "data/private/holdings_snapshot.csv")
    parser.add_argument("--prices", type=Path, default=ROOT / "data/raw/close_price_panel.csv")
    parser.add_argument("--snapshot", type=Path, action="append", default=[], help="Repeat for each dated ETF JSON manifest.")
    parser.add_argument("--cash", type=float, default=None, help="Optional account cash in the same base currency.")
    parser.add_argument("--as-of-date", help="Analysis date YYYY-MM-DD; defaults to today's local date.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/personal_lookthrough")
    args = parser.parse_args()
    holdings = load_holdings(args.holdings_csv)
    prices = load_price_panel(args.prices)
    exposures, funds, summary = analyze_portfolio_lookthrough(
        holdings, prices, args.snapshot, cash=args.cash, as_of_date=args.as_of_date
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(args.output_dir / "portfolio_lookthrough_exposures.csv", index=False, float_format="%.10f")
    funds.to_csv(args.output_dir / "portfolio_lookthrough_funds.csv", index=False, float_format="%.10f")
    (args.output_dir / "portfolio_lookthrough_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
