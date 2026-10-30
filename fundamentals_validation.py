#!/usr/bin/env python3
"""Auditable freshness, currency, and per-share-basis checks for valuation data."""
from __future__ import annotations

from datetime import date, datetime, timezone
import math
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_date(value: Any) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_fundamentals_snapshot(
    snapshot: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    *,
    as_of_date: date | str | None = None,
    max_quote_age_days: int = 7,
    max_financial_age_days: int = 180,
    max_cache_age_hours: float = 168.0,
    market_cap_relative_tolerance: float = 0.30,
) -> dict[str, Any]:
    """Return explicit blockers/warnings and DCF/PE readiness gates.

    A quote and financial period are checked independently. A recent cache
    retrieval timestamp does not make an old financial statement current.
    DCF and PE receive separate currency/unit gates so a KRW FCF mismatch need
    not suppress a USD-denominated EPS multiple when its per-share basis is
    separately verified.
    """
    if not isinstance(snapshot, Mapping) or not isinstance(assumptions, Mapping):
        raise ValueError("snapshot and assumptions must be mappings.")
    if as_of_date is None:
        as_of = date.today()
    elif isinstance(as_of_date, date):
        as_of = as_of_date
    else:
        parsed_as_of = _parse_date(as_of_date)
        if parsed_as_of is None:
            raise ValueError("as_of_date must be a valid date.")
        as_of = parsed_as_of
    if isinstance(max_quote_age_days, bool) or not isinstance(max_quote_age_days, int) or max_quote_age_days < 0:
        raise ValueError("max_quote_age_days must be a non-negative integer.")
    if isinstance(max_financial_age_days, bool) or not isinstance(max_financial_age_days, int) or max_financial_age_days < 0:
        raise ValueError("max_financial_age_days must be a non-negative integer.")
    if not math.isfinite(max_cache_age_hours) or max_cache_age_hours <= 0:
        raise ValueError("max_cache_age_hours must be finite and positive.")
    if not math.isfinite(market_cap_relative_tolerance) or market_cap_relative_tolerance < 0:
        raise ValueError("market_cap_relative_tolerance must be finite and non-negative.")

    findings: list[dict[str, str]] = []

    def add(
        code: str,
        severity: str,
        message: str,
        gates: tuple[str, ...] = ("dcf", "pe"),
    ) -> None:
        findings.append({
            "code": code,
            "severity": severity,
            "message": message,
            "gates": ",".join(gates),
        })

    quote_currency = str(snapshot.get("currency") or "").strip().upper()
    financial_currency = str(snapshot.get("financial_currency") or quote_currency).strip().upper()
    eps_currency = str(snapshot.get("eps_currency") or quote_currency).strip().upper()
    price = _number(snapshot.get("current_price"))
    fcf = _number(
        assumptions.get("normalized_fcf")
        if assumptions.get("normalized_fcf") is not None
        else snapshot.get("free_cash_flow")
    )
    fcf_currency = str(
        assumptions.get("normalized_fcf_currency")
        if assumptions.get("normalized_fcf") is not None and assumptions.get("normalized_fcf_currency")
        else financial_currency
    ).strip().upper()
    eps = _number(snapshot.get("forward_eps"))
    shares = _number(snapshot.get("diluted_shares"))
    market_cap = _number(snapshot.get("market_cap"))

    quote_age_days: int | None = None
    price_date = _parse_date(snapshot.get("price_date"))
    if price is None or price <= 0:
        add("PRICE_MISSING_OR_INVALID", "BLOCKER", "No positive current quote is available.")
    if price_date is None:
        add("QUOTE_DATE_MISSING", "BLOCKER", "The quote has no valid price date.")
    else:
        quote_age_days = (as_of - price_date).days
        if quote_age_days < 0:
            add("QUOTE_DATE_IN_FUTURE", "BLOCKER", f"Quote date {price_date.isoformat()} is after the audit date.")
        elif quote_age_days > max_quote_age_days:
            add(
                "QUOTE_STALE", "BLOCKER",
                f"Quote is {quote_age_days} calendar days old; configured maximum is {max_quote_age_days} days.",
            )

    financial_age_days: int | None = None
    financial_period = _parse_date(snapshot.get("financial_period"))
    if financial_period is None:
        add(
            "FINANCIAL_PERIOD_MISSING", "WARNING",
            "A statement period was not provided; the retrieval timestamp cannot substitute for a reporting period.",
        )
    else:
        financial_age_days = (as_of - financial_period).days
        if financial_age_days < 0:
            add("FINANCIAL_PERIOD_IN_FUTURE", "BLOCKER", "Financial statement period is after the audit date.")
        elif financial_age_days > max_financial_age_days:
            add(
                "FINANCIAL_PERIOD_STALE", "BLOCKER",
                f"Financial statement period is {financial_age_days} days old; configured maximum is {max_financial_age_days} days.",
            )

    fetched = _parse_datetime(snapshot.get("fetched_at_utc") or snapshot.get("data_fetched_at_utc"))
    cache_age = _number(snapshot.get("cache_age_hours"))
    if cache_age is None and fetched is not None:
        cache_age = max(0.0, (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0)
    if bool(snapshot.get("stale_data", False)):
        add("STALE_CACHE_FLAG", "WARNING", "The data adapter marked this snapshot as stale.")
    elif cache_age is not None and cache_age > max_cache_age_hours:
        add(
            "CACHE_STALE", "WARNING",
            f"Snapshot cache is {cache_age:.1f} hours old; configured maximum is {max_cache_age_hours:.1f} hours.",
        )
    if fetched is None:
        add("FETCH_TIMESTAMP_MISSING", "WARNING", "Snapshot retrieval timestamp is missing or invalid.")

    if not quote_currency:
        add("QUOTE_CURRENCY_MISSING", "BLOCKER", "Quote currency is missing.")
    if not financial_currency:
        add("FINANCIAL_CURRENCY_MISSING", "BLOCKER", "Financial statement currency is missing.", ("dcf",))
    elif quote_currency and fcf_currency != quote_currency:
        add(
            "DCF_CURRENCY_MISMATCH", "BLOCKER",
            f"DCF cash-flow currency {fcf_currency} differs from quote currency {quote_currency}; no FX conversion is applied.",
            ("dcf",),
        )
    if not eps_currency:
        add("EPS_CURRENCY_MISSING", "BLOCKER", "EPS currency is missing.", ("pe",))
    elif quote_currency and eps_currency != quote_currency:
        add(
            "PE_CURRENCY_MISMATCH", "BLOCKER",
            f"EPS currency {eps_currency} differs from quote currency {quote_currency}; no FX conversion is applied.",
            ("pe",),
        )

    per_share_verified = bool(assumptions.get("per_share_basis_verified", True))
    if not per_share_verified:
        note = str(assumptions.get("per_share_basis_note") or "EPS/share-count units have not been verified.")
        add("PER_SHARE_BASIS_UNVERIFIED", "BLOCKER", note, ("dcf", "pe"))
    adr_ratio = _number(assumptions.get("adr_underlying_shares_per_ads"))
    if adr_ratio is not None and adr_ratio <= 0:
        add("ADR_RATIO_INVALID", "BLOCKER", "ADR underlying shares per ADS must be positive.", ("dcf", "pe"))

    if eps is None or eps <= 0:
        add("FORWARD_EPS_UNAVAILABLE", "BLOCKER", "No positive forward EPS is available.", ("pe",))
    eps_source = str(snapshot.get("eps_source") or "").lower()
    trailing_eps_fallback = "trailing" in eps_source and (
        "fallback" in eps_source
        or "unavailable" in eps_source
        or "substitut" in eps_source
    )
    if trailing_eps_fallback:
        add(
            "TRAILING_EPS_FALLBACK", "BLOCKER",
            "The provider substituted trailing EPS because forward EPS is unavailable; this does not satisfy the forward-PE input requirement.",
            ("pe",),
        )
    if fcf is None or fcf <= 0:
        add("POSITIVE_FCF_UNAVAILABLE", "BLOCKER", "No positive DCF cash-flow input is available.", ("dcf",))
    if shares is None or shares <= 0:
        add("SHARE_COUNT_UNAVAILABLE", "BLOCKER", "No positive diluted share count is available.", ("dcf",))

    if price is not None and price > 0 and shares is not None and shares > 0 and market_cap is not None and market_cap > 0:
        implied_market_cap = price * shares
        relative_error = abs(market_cap - implied_market_cap) / max(market_cap, implied_market_cap)
        if relative_error > market_cap_relative_tolerance:
            add(
                "MARKET_CAP_SHARE_COUNT_DISAGREEMENT", "WARNING",
                f"Market cap differs from price × diluted shares by {relative_error:.1%}; confirm share-count units and ADS basis.",
            )

    if str(snapshot.get("data_quality") or "").startswith("partial"):
        add("PARTIAL_VENDOR_SNAPSHOT", "WARNING", str(snapshot.get("data_quality")))
    provider_warnings = snapshot.get("fetch_warnings") or []
    if provider_warnings:
        add(
            "PROVIDER_WARNINGS", "WARNING",
            "Provider adapter reported: " + " | ".join(str(item) for item in provider_warnings),
        )

    blockers = [item for item in findings if item["severity"] == "BLOCKER"]
    warnings = [item for item in findings if item["severity"] == "WARNING"]
    dcf_blockers = [item for item in blockers if "dcf" in item["gates"].split(",")]
    pe_blockers = [item for item in blockers if "pe" in item["gates"].split(",")]
    status = "BLOCKED" if blockers else ("WARNING" if warnings else "PASS")
    return {
        "ticker": str(snapshot.get("ticker") or "").upper(),
        "audit_date": as_of.isoformat(),
        "status": status,
        "dcf_ready": not dcf_blockers,
        "pe_ready": not pe_blockers,
        "quote_age_days": quote_age_days,
        "financial_age_days": financial_age_days,
        "cache_age_hours": cache_age,
        "quote_currency": quote_currency or None,
        "financial_currency": financial_currency or None,
        "eps_currency": eps_currency or None,
        "fcf_currency": fcf_currency or None,
        "per_share_basis_verified": per_share_verified,
        "findings_count": len(findings),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "findings": findings,
        "limitations": [
            "This audit validates dates, units, currencies and internal consistency; it does not independently verify vendor data against company filings.",
            "A recent retrieval timestamp does not guarantee a recent financial statement period.",
            "A reported market-cap/share-count discrepancy is a warning requiring unit review, not proof that either vendor field is wrong.",
        ],
    }


def apply_validation_gates(row: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    """Apply data-readiness gates to a valuation row without rewriting model inputs."""
    result = dict(row)
    for prefix, gate in (("dcf", "dcf_ready"), ("pe", "pe_ready")):
        if not bool(validation.get(gate, False)):
            for field in (f"{prefix}_bear", f"{prefix}_base", f"{prefix}_bull",
                          f"{prefix}_buy_below", f"{prefix}_sell_review_above"):
                result[field] = None
            result[f"{prefix}_view"] = "unavailable_data_validation"
    findings = validation.get("findings", [])
    result["data_validation_status"] = validation.get("status", "UNKNOWN")
    result["data_validation_dcf_ready"] = bool(validation.get("dcf_ready", False))
    result["data_validation_pe_ready"] = bool(validation.get("pe_ready", False))
    result["data_validation_quote_age_days"] = validation.get("quote_age_days")
    result["data_validation_financial_age_days"] = validation.get("financial_age_days")
    result["data_validation_findings"] = " | ".join(
        f"{item.get('code')}: {item.get('message')}" for item in findings
    )
    if validation.get("status") != "PASS":
        result["overall_confidence"] = "low"
        existing = str(result.get("limitations") or "")
        finding_text = result["data_validation_findings"]
        result["limitations"] = "；".join(part for part in (existing, f"数据核验：{finding_text}") if part)
    return result
