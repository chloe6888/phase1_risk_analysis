#!/usr/bin/env python3
"""Run a personalized, local-only holdings + price-data acceptance workflow.

Private holdings and downloaded price panels remain in ignored local folders.
The script never uploads holdings, share quantities, cost basis, or account values.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

import phase1_risk_analysis as market_data
import real_data_acceptance as acceptance
from holdings_manager import load_holdings

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "portfolio_universe.json"
DEFAULT_HOLDINGS = ROOT / "data" / "private" / "holdings_snapshot.csv"
DEFAULT_CLOSE = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_ADJUSTED = ROOT / "data" / "raw" / "adjusted_close_panel.csv"
DEFAULT_REPORTS = ROOT / "outputs" / "acceptance"


def load_universe(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the non-private portfolio ticker universe."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read portfolio universe configuration: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Portfolio universe configuration must be a schema_version=1 JSON object.")

    base_currency = str(payload.get("base_currency", "")).strip().upper()
    benchmark = str(payload.get("benchmark", "")).strip().upper()
    raw_symbols = payload.get("symbols")
    reference_tickers = payload.get("reference_tickers")
    start_date = str(payload.get("start_date", "")).strip()
    try:
        pd.Timestamp(start_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("Portfolio start_date must be a valid YYYY-MM-DD date.") from exc
    if not base_currency or len(base_currency) != 3 or not base_currency.isalpha():
        raise ValueError("base_currency must be a three-letter currency code.")
    if not isinstance(raw_symbols, dict) or not raw_symbols:
        raise ValueError("Portfolio universe must define a non-empty symbols object.")
    if not isinstance(reference_tickers, list) or not reference_tickers:
        raise ValueError("reference_tickers must be a non-empty list.")

    symbols: dict[str, dict[str, Any]] = {}
    for raw_ticker, raw_metadata in raw_symbols.items():
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in symbols or not isinstance(raw_metadata, dict):
            raise ValueError(f"Invalid or duplicate symbol metadata for {raw_ticker!r}.")
        metadata = dict(raw_metadata)
        metadata["role"] = str(metadata.get("role", "")).strip().lower()
        metadata["currency"] = str(metadata.get("currency", "")).strip().upper()
        if metadata["role"] not in {"holding", "benchmark", "sector_reference", "reference"}:
            raise ValueError(f"{ticker}: unsupported symbol role {metadata['role']!r}.")
        if metadata["currency"] != base_currency:
            raise ValueError(f"{ticker}: configured currency must equal portfolio base currency {base_currency}.")
        if metadata["role"] == "holding":
            for field in ("risk_group", "subgroup"):
                if not str(metadata.get(field, "")).strip():
                    raise ValueError(f"{ticker}: holding metadata requires {field}.")
        symbols[ticker] = metadata

    refs = [str(item).strip().upper() for item in reference_tickers]
    if any(not item for item in refs) or len(set(refs)) != len(refs):
        raise ValueError("reference_tickers must contain unique, non-empty ticker symbols.")
    if any(item not in symbols for item in refs):
        missing = sorted(set(refs) - set(symbols))
        raise ValueError(f"Reference tickers are missing from symbols metadata: {missing}.")
    if benchmark not in refs or benchmark not in symbols:
        raise ValueError("The configured benchmark must be included in reference_tickers and symbols.")
    if symbols[benchmark]["role"] != "benchmark":
        raise ValueError(f"{benchmark}: the configured benchmark must have role='benchmark'.")
    holding_tickers = sorted(ticker for ticker, item in symbols.items() if item["role"] == "holding")
    if not holding_tickers:
        raise ValueError("Portfolio universe does not define any holdings.")
    minimum_common = int(payload.get("minimum_common_observations", 60))
    target_common = int(payload.get("target_common_observations", 126))
    if minimum_common < 2 or target_common < minimum_common:
        raise ValueError("Common-history thresholds must satisfy 2 <= minimum <= target.")
    payload["base_currency"] = base_currency
    payload["benchmark"] = benchmark
    payload["reference_tickers"] = refs
    payload["symbols"] = symbols
    payload["holding_tickers"] = holding_tickers
    payload["minimum_common_observations"] = minimum_common
    payload["target_common_observations"] = target_common
    return payload


def validate_holdings_file(
    path: str | Path,
    universe: dict[str, Any],
) -> pd.DataFrame:
    """Require exact ticker coverage, explicit currencies, and matching risk taxonomy."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(
            f"Private holdings file not found: {source}. Copy your holdings CSV to data/private/holdings_snapshot.csv."
        )
    try:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Cannot read private holdings CSV: {exc}") from exc
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    if "currency" not in raw.columns:
        raise ValueError("Private holdings CSV must contain an explicit currency column; do not infer USD silently.")
    currencies = raw["currency"].astype(str).str.strip().str.upper()
    if currencies.eq("").any() or currencies.isin({"NAN", "NONE", "NULL", "<NA>"}).any():
        raise ValueError("Every holding must have a non-empty explicit currency value.")
    if set(currencies) != {universe["base_currency"]}:
        raise ValueError(
            f"All holdings must use {universe['base_currency']} in this pipeline; mixed currencies are blocked."
        )

    holdings = load_holdings(source)
    actual = set(holdings["ticker"].astype(str))
    expected = set(universe["holding_tickers"])
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Holdings ticker coverage mismatch; missing={missing}, unexpected={unexpected}.")

    metadata_by_ticker = universe["symbols"]
    for row in holdings.itertuples(index=False):
        metadata = metadata_by_ticker[row.ticker]
        if str(row.currency).upper() != metadata["currency"]:
            raise ValueError(f"{row.ticker}: holding currency does not match configured currency.")
        if str(row.risk_group).strip().lower() != metadata["risk_group"]:
            raise ValueError(f"{row.ticker}: risk_group does not match portfolio_universe.json.")
        if str(row.subgroup).strip().lower() != metadata["subgroup"]:
            raise ValueError(f"{row.ticker}: subgroup does not match portfolio_universe.json.")
    return holdings


def run_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    holdings_path: str | Path = DEFAULT_HOLDINGS,
    close_prices_path: str | Path = DEFAULT_CLOSE,
    adjusted_prices_path: str | Path = DEFAULT_ADJUSTED,
    output_dir: str | Path = DEFAULT_REPORTS,
    refresh: bool = True,
    as_of_date: str | date | None = None,
) -> dict[str, Any]:
    """Refresh the market panels and run the local real-data acceptance checks."""
    universe = load_universe(config_path)
    holdings = validate_holdings_file(holdings_path, universe)
    tickers = list(dict.fromkeys(
        [universe["benchmark"], *universe["reference_tickers"], *holdings["ticker"].astype(str).tolist()]
    ))
    short_history_tickers = sorted(
        ticker for ticker, metadata in universe["symbols"].items()
        if metadata.get("regular_way_start_date")
    )

    market_data.main(
        refresh=refresh,
        allow_stale_cache=False,
        max_cache_age_hours=market_data.DEFAULT_MAX_CACHE_AGE_HOURS,
        tickers=tickers,
        start_date=universe["start_date"],
        end_date=market_data.END_DATE,
        benchmark=universe["benchmark"],
        short_history_tickers=short_history_tickers,
    )
    result = acceptance.run_acceptance(
        holdings_path,
        close_prices_path,
        adjusted_prices_path,
        output_dir,
        as_of_date=as_of_date,
        max_staleness_days=7,
        minimum_common_observations=universe["minimum_common_observations"],
        target_common_observations=universe["target_common_observations"],
    )
    return {
        "status": result["status"],
        "as_of_date": result["as_of_date"],
        "position_count": result.get("overview", {}).get("position_count"),
        "price_tickers": tickers,
        "short_history_tickers": short_history_tickers,
        "common_return_observations": result.get("overview", {}).get("common_return_observations"),
        "report_markdown": str(Path(output_dir) / "real_data_acceptance.md"),
        "report_json": str(Path(output_dir) / "real_data_acceptance.json"),
        "target_common_observations": universe["target_common_observations"],
        "private_values_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the configured portfolio prices and run local-only data acceptance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS)
    parser.add_argument("--close-prices", type=Path, default=DEFAULT_CLOSE)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--as-of-date", help="Override acceptance date (YYYY-MM-DD); normally leave unset.")
    refresh_mode = parser.add_mutually_exclusive_group()
    refresh_mode.add_argument(
        "--refresh",
        dest="refresh",
        action="store_true",
        help="Force a fresh market-data download (the default).",
    )
    refresh_mode.add_argument(
        "--use-cache",
        dest="refresh",
        action="store_false",
        help="Reuse cache only when fresh and complete; stale or incomplete data is refreshed.",
    )
    parser.set_defaults(refresh=True)
    args = parser.parse_args()
    try:
        result = run_pipeline(
            config_path=args.config,
            holdings_path=args.holdings_csv,
            close_prices_path=args.close_prices,
            adjusted_prices_path=args.adjusted_prices,
            output_dir=args.output_dir,
            refresh=args.refresh,
            as_of_date=args.as_of_date,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
