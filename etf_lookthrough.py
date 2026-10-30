#!/usr/bin/env python3
"""Import dated ETF holdings and aggregate issuer look-through exposures.

The importer accepts a CSV downloaded from an ETF issuer (or another explicitly
named provider). It normalizes common column names, identifies issuer rows across
direct shares, swaps, and depositary receipts, records provenance, and writes an
immutable dated snapshot. It does not fetch an undocumented issuer endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk import DEFAULT_OUTPUT_DIR, ROOT

DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR
COLUMN_ALIASES = {
    "ticker": {"ticker", "symbol", "stockticker", "holdingticker"},
    "name": {"name", "description", "security", "holding", "holdingname", "securityname"},
    "identifier": {"identifier", "cusip", "isin", "sedol", "securityid"},
    "weight": {
        "weight", "etfweight", "weightpct", "weightpercent", "portfolioweight",
        "percentageofportfolio", "pctportfolio", "portfolioallocation", "allocation",
    },
    "shares": {"shares", "sharesheld", "quantity", "units", "amount"},
    "market_value": {"marketvalue", "marketvalueusd", "value", "marketvalue($)"},
}
DEFAULT_ISSUER_PATTERNS = {
    "MU": [
        "MICRON", "MICRON TECHNOLOGY", "MU US", "MU UW", "MU.O",
    ],
    "SKHY": [
        "SK HYNIX", "SKHY", "SK HYNIX INC", "000660", "000660 KS",
    ],
}


def _normalize_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_columns(columns: list[object]) -> dict[str, object]:
    normalized = {_normalize_header(column): column for column in columns}
    result: dict[str, object] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                result[canonical] = normalized[alias]
                break
    if "weight" not in result:
        raise ValueError(
            "ETF holdings CSV must contain a weight column (e.g. ETF Weight, Weight, or % Portfolio)."
        )
    if "ticker" not in result and "name" not in result:
        raise ValueError("ETF holdings CSV must contain at least a ticker or name/description column.")
    return result


def _parse_weight(value: object) -> tuple[float, bool]:
    """Return weight as percentage points and whether the row explicitly used '%'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("ETF holding weight is missing.")
    raw = str(value).strip().replace(",", "").replace("$", "")
    if not raw:
        raise ValueError("ETF holding weight is empty.")
    explicit_percent = raw.endswith("%")
    if explicit_percent:
        raw = raw[:-1].strip()
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ETF holding weight: {value!r}") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"ETF holding weight must be finite and non-negative: {value!r}")
    return number, explicit_percent


def _canonical_issuer(
    ticker: str,
    name: str,
    identifier: str,
    issuer_patterns: Mapping[str, list[str]],
) -> str | None:
    combined = " ".join((ticker, name, identifier)).upper()
    for issuer, patterns in issuer_patterns.items():
        normalized_issuer = str(issuer).strip().upper()
        if not normalized_issuer:
            continue
        for pattern in patterns:
            candidate = str(pattern).strip().upper()
            if candidate and candidate in combined:
                return normalized_issuer
    return None


def _exposure_type(ticker: str, name: str, identifier: str, issuer: str | None) -> str:
    combined = " ".join((ticker, name, identifier)).upper()
    if not issuer:
        if any(token in combined for token in ("TREASURY", "CASH", "MONEY MARKET", "USD", "KRW", "CNY", "TWD")):
            return "cash_or_cash_equivalent"
        return "other"
    if "SWAP" in combined or "TRS" in combined or "TOTAL RETURN" in combined:
        return "swap"
    if ticker.strip().upper() == "SKHY" or "ADR" in combined or "DEPOSITARY" in combined:
        return "adr"
    return "direct_equity"


def _normalize_pattern_map(
    issuer_patterns: Mapping[str, list[str]] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    source = issuer_patterns or DEFAULT_ISSUER_PATTERNS
    for issuer, patterns in source.items():
        key = str(issuer).strip().upper()
        if not key:
            raise ValueError("Issuer names in the mapping must not be empty.")
        if isinstance(patterns, str):
            patterns = [patterns]
        cleaned = [str(item).strip().upper() for item in patterns if str(item).strip()]
        if not cleaned:
            raise ValueError(f"Issuer '{key}' must have at least one matching pattern.")
        result[key] = cleaned
    return result


def import_etf_holdings(
    csv_path: Path,
    *,
    as_of: str,
    source_name: str = "ETF issuer CSV",
    issuer_patterns: Mapping[str, list[str]] | None = None,
    decimal_weights: bool | None = None,
    complete_weight_range: tuple[float, float] = (95.0, 105.0),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Parse a holdings CSV and aggregate selected issuer exposure by row type.

    decimal_weights=None auto-detects a decimal series when its total is <=1.5;
    otherwise values are treated as percent points. Set True/False to override.
    Completeness is reported, not enforced: some issuer exports can omit cash or
    non-security lines, and users should inspect the manifest warning.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise ValueError(f"ETF holdings CSV does not exist: {path}")
    try:
        asof_date = pd.Timestamp(as_of).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of must be a valid date (YYYY-MM-DD).") from exc
    if not source_name.strip():
        raise ValueError("source_name must not be empty.")
    low, high = map(float, complete_weight_range)
    if not np.isfinite([low, high]).all() or low < 0 or high <= low:
        raise ValueError("complete_weight_range must be finite and increasing.")

    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot parse ETF holdings CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("ETF holdings CSV has no data rows.")

    columns = _find_columns(list(frame.columns))
    patterns = _normalize_pattern_map(issuer_patterns)
    rows: list[dict[str, object]] = []
    raw_weights: list[float] = []
    any_explicit_percent = False
    for row_number, row in frame.iterrows():
        ticker = str(row.get(columns.get("ticker"), "") if "ticker" in columns else "").strip().upper()
        name = str(row.get(columns.get("name"), "") if "name" in columns else "").strip()
        identifier = str(row.get(columns.get("identifier"), "") if "identifier" in columns else "").strip()
        if not ticker and not name and not identifier:
            continue
        weight_value = row.get(columns["weight"], "")
        weight, explicit_percent = _parse_weight(weight_value)
        raw_weights.append(weight)
        any_explicit_percent = any_explicit_percent or explicit_percent
        issuer = _canonical_issuer(ticker, name, identifier, patterns)
        rows.append({
            "source_row_number": int(row_number) + 2,
            "ticker": ticker,
            "name": name,
            "identifier": identifier,
            "issuer": issuer,
            "exposure_type": _exposure_type(ticker, name, identifier, issuer),
            "_raw_weight": weight,
            "shares": str(row.get(columns.get("shares"), "") if "shares" in columns else "").strip(),
            "market_value": str(row.get(columns.get("market_value"), "") if "market_value" in columns else "").strip(),
        })
    if not rows:
        raise ValueError("No usable holdings rows were found.")
    total_raw = float(sum(raw_weights))
    use_decimal = bool(decimal_weights) if decimal_weights is not None else (
        not any_explicit_percent and total_raw <= 1.5
    )
    multiplier = 100.0 if use_decimal else 1.0
    for row in rows:
        row["weight_pct"] = float(row.pop("_raw_weight")) * multiplier
        if row["weight_pct"] > 100.0 + 1e-9:
            raise ValueError(
                f"ETF holding row {row['source_row_number']} weight exceeds 100%; verify weight units."
            )

    holdings = pd.DataFrame(rows)
    total_weight_pct = float(holdings["weight_pct"].sum())
    if total_weight_pct > 105.0 + 1e-6:
        raise ValueError(
            f"Total ETF holdings weight is {total_weight_pct:.2f}%; likely duplicate rows or wrong weight units."
        )
    issuer_rows = holdings[holdings["issuer"].notna()].copy()
    if issuer_rows.empty:
        raise ValueError(
            "No target issuers matched the mapping. Check ticker/name fields or provide --issuer-map-json."
        )

    issuer_summary = (
        issuer_rows.groupby(["issuer", "exposure_type"], dropna=False)["weight_pct"]
        .sum().unstack(fill_value=0.0).reset_index()
    )
    for kind in ("direct_equity", "swap", "adr"):
        if kind not in issuer_summary.columns:
            issuer_summary[kind] = 0.0
    issuer_summary["total_issuer_exposure_pct"] = issuer_summary[
        ["direct_equity", "swap", "adr"]
    ].sum(axis=1)
    issuer_summary = issuer_summary.sort_values("issuer", kind="stable").reset_index(drop=True)

    exposure_map = {
        str(row.issuer): float(row.total_issuer_exposure_pct)
        for row in issuer_summary.itertuples(index=False)
    }
    complete = low <= total_weight_pct <= high
    warnings: list[str] = []
    if not complete:
        warnings.append(
            f"Total parsed holdings weight is {total_weight_pct:.2f}%; expected complete range is {low:.2f}%–{high:.2f}%. "
            "Check whether the source export omits cash or other holdings."
        )
    unmatched_rows = holdings[
        holdings["issuer"].isna()
        & holdings["exposure_type"].ne("cash_or_cash_equivalent")
    ]
    unmapped_weight = float(unmatched_rows["weight_pct"].sum())
    cash_equivalent_weight = float(
        holdings.loc[holdings["exposure_type"].eq("cash_or_cash_equivalent"), "weight_pct"].sum()
    )
    if unmapped_weight > 0:
        warnings.append(
            f"{unmapped_weight:.2f}% of non-cash holdings weight was not mapped to the configured issuer set; "
            "review the full holdings file before relying on issuer exposure totals."
        )
    manifest: dict[str, object] = {
        "fund_ticker": "DRAM",
        "as_of": asof_date.isoformat(),
        "source_name": source_name,
        "source_file": path.name,
        "source_sha256": sha256,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "holding_rows": int(len(holdings)),
        "mapped_rows": int(len(issuer_rows)),
        "total_weight_pct": total_weight_pct,
        "mapped_target_issuer_weight_pct": float(issuer_rows["weight_pct"].sum()),
        "unmapped_weight_pct": unmapped_weight,
        "cash_or_cash_equivalent_weight_pct": cash_equivalent_weight,
        "weights_were_decimal": use_decimal,
        "complete_weight_range_pct": [low, high],
        "weight_total_in_range": bool(complete),
        "issuer_weights_pct": exposure_map,
        "warnings": warnings,
        "limitations": [
            "Issuer aggregation relies on the configured text/ticker matching map; inspect unmapped rows and the source file.",
            "Underlying shares, swaps and depositary receipts are grouped by issuer; they are not interchangeable instruments in their own right.",
            "The snapshot is dated and must be refreshed when the issuer publishes updated holdings.",
        ],
    }
    return holdings.drop(columns=[]), issuer_summary, manifest


def load_issuer_snapshot(path: Path) -> tuple[dict[str, float], str, dict[str, object]]:
    """Read a previously imported snapshot manifest for use in memory allocation."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ETF issuer snapshot JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("ETF issuer snapshot must be a JSON object.")
    raw_map = manifest.get("issuer_weights_pct")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("ETF issuer snapshot is missing issuer_weights_pct.")
    normalized: dict[str, float] = {}
    for issuer, value in raw_map.items():
        key = str(issuer).strip().upper()
        try:
            fraction = float(value) / 100.0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid issuer exposure for {issuer}.") from exc
        if not key or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f"Issuer exposure for {issuer} must be between 0 and 100%.")
        normalized[key] = fraction
    if sum(normalized.values()) > 1.0 + 1e-9:
        raise ValueError("Mapped issuer exposures cannot sum above 100% of the ETF.")
    as_of = str(manifest.get("as_of", "")).strip()
    try:
        pd.Timestamp(as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError("Snapshot manifest has an invalid as_of date.") from exc
    return normalized, as_of, manifest


def _load_issuer_map(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read issuer map JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Issuer map JSON must be an object.")
    return {str(key): value if isinstance(value, list) else [str(value)] for key, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a dated ETF holdings CSV and calculate issuer look-through exposure; no market orders are placed."
    )
    parser.add_argument("--holdings-csv", type=Path, required=True, help="CSV downloaded from the ETF issuer.")
    parser.add_argument("--as-of", required=True, help="Date represented by the source holdings, YYYY-MM-DD.")
    parser.add_argument("--source-name", default="Roundhill issuer holdings CSV")
    parser.add_argument("--issuer-map-json", type=Path, help="Optional JSON mapping issuer -> matching text patterns.")
    parser.add_argument("--decimal-weights", action="store_true", help="Interpret un-suffixed weight values as fractions, e.g. 0.25 = 25%.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "etf_holdings")
    args = parser.parse_args()
    holdings, issuers, manifest = import_etf_holdings(
        args.holdings_csv,
        as_of=args.as_of,
        source_name=args.source_name,
        issuer_patterns=_load_issuer_map(args.issuer_map_json),
        decimal_weights=True if args.decimal_weights else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dated_name = f"DRAM_holdings_{manifest['as_of']}.csv"
    holdings.to_csv(args.output_dir / dated_name, index=False, float_format="%.8f")
    issuers.to_csv(args.output_dir / f"DRAM_issuer_exposures_{manifest['as_of']}.csv", index=False, float_format="%.8f")
    (args.output_dir / f"DRAM_lookthrough_{manifest['as_of']}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
