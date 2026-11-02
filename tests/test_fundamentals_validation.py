"""Tests for valuation snapshot freshness, currency, and unit validation."""
from __future__ import annotations

from datetime import date

import pytest

from fundamentals_validation import apply_validation_gates, validate_fundamentals_snapshot


def base_snapshot() -> dict[str, object]:
    return {
        "ticker": "TEST",
        "currency": "USD",
        "financial_currency": "USD",
        "eps_currency": "USD",
        "current_price": 100.0,
        "price_date": "2026-10-29",
        "fetched_at_utc": "2026-10-30T00:00:00+00:00",
        "financial_period": "2026-09-30",
        "forward_eps": 5.0,
        "eps_source": "Yahoo forwardEps",
        "free_cash_flow": 1000.0,
        "diluted_shares": 10.0,
        "market_cap": 1000.0,
        "data_quality": "complete_vendor_snapshot",
        "fetch_warnings": [],
        "stale_data": False,
        "cache_age_hours": 1.0,
    }


def base_assumptions() -> dict[str, object]:
    return {
        "per_share_basis_verified": True,
        "normalized_fcf": None,
        "normalized_fcf_currency": None,
        "adr_underlying_shares_per_ads": None,
    }


def test_fresh_consistent_snapshot_passes_both_valuation_gates() -> None:
    result = validate_fundamentals_snapshot(
        base_snapshot(), base_assumptions(), as_of_date="2026-10-30"
    )
    assert result["status"] == "PASS"
    assert result["dcf_ready"] is True
    assert result["pe_ready"] is True
    assert result["quote_age_days"] == 1
    assert result["financial_age_days"] == 30


def test_stale_quote_blocks_both_models_but_keeps_age_evidence() -> None:
    snapshot = base_snapshot()
    snapshot["price_date"] = "2026-10-01"
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30", max_quote_age_days=7
    )
    assert result["status"] == "BLOCKED"
    assert result["dcf_ready"] is False
    assert result["pe_ready"] is False
    assert any(item["code"] == "QUOTE_STALE" for item in result["findings"])


def test_old_financial_period_blocks_valuation_even_when_cache_is_recent() -> None:
    snapshot = base_snapshot()
    snapshot["financial_period"] = "2025-12-31"
    snapshot["cache_age_hours"] = 0.1
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30", max_financial_age_days=180
    )
    assert result["status"] == "BLOCKED"
    assert result["dcf_ready"] is False
    assert result["pe_ready"] is False
    assert any(item["code"] == "FINANCIAL_PERIOD_STALE" for item in result["findings"])


def test_currency_mismatch_blocks_only_affected_valuation_model() -> None:
    snapshot = base_snapshot()
    snapshot["financial_currency"] = "KRW"
    snapshot["eps_currency"] = "USD"
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30"
    )
    assert result["dcf_ready"] is False
    assert result["pe_ready"] is True
    assert any(item["code"] == "DCF_CURRENCY_MISMATCH" for item in result["findings"])


def test_trailing_eps_fallback_is_not_accepted_as_forward_pe_input() -> None:
    snapshot = base_snapshot()
    snapshot["eps_source"] = "Yahoo trailingEps fallback; forward EPS unavailable"
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30"
    )
    assert result["dcf_ready"] is True
    assert result["pe_ready"] is False
    assert any(item["code"] == "TRAILING_EPS_FALLBACK" for item in result["findings"])


def test_unverified_adr_per_share_basis_blocks_both_methods() -> None:
    result = validate_fundamentals_snapshot(
        base_snapshot(),
        {
            **base_assumptions(),
            "per_share_basis_verified": False,
            "adr_underlying_shares_per_ads": 10,
        },
        as_of_date="2026-10-30",
    )
    assert result["dcf_ready"] is False
    assert result["pe_ready"] is False
    assert any(item["code"] == "PER_SHARE_BASIS_UNVERIFIED" for item in result["findings"])


def test_market_cap_disagreement_is_a_warning_not_a_false_certainty() -> None:
    snapshot = base_snapshot()
    snapshot["market_cap"] = 100_000.0
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30"
    )
    assert result["status"] == "WARNING"
    assert result["dcf_ready"] is True
    assert result["pe_ready"] is True
    assert any(item["code"] == "MARKET_CAP_SHARE_COUNT_DISAGREEMENT" for item in result["findings"])


def test_validation_rejects_future_dates_and_bad_settings() -> None:
    snapshot = base_snapshot()
    snapshot["price_date"] = "2026-11-01"
    result = validate_fundamentals_snapshot(
        snapshot, base_assumptions(), as_of_date="2026-10-30"
    )
    assert result["dcf_ready"] is False
    with pytest.raises(ValueError, match="max_quote_age_days"):
        validate_fundamentals_snapshot(snapshot, base_assumptions(), max_quote_age_days=-1)


def test_apply_validation_gates_clears_only_blocked_outputs() -> None:
    validation = {
        "status": "BLOCKED",
        "dcf_ready": False,
        "pe_ready": True,
        "quote_age_days": 1,
        "financial_age_days": 30,
        "findings": [{"code": "DCF_CURRENCY_MISMATCH", "message": "DCF currency mismatch"}],
    }
    row = {
        "dcf_base": 10.0, "dcf_view": "between_thresholds",
        "dcf_buy_below": 8.0, "dcf_sell_review_above": 12.0,
        "pe_base": 20.0, "pe_view": "between_thresholds",
        "pe_buy_below": 15.0, "pe_sell_review_above": 24.0,
        "overall_confidence": "medium", "limitations": "",
    }
    updated = apply_validation_gates(row, validation)
    assert updated["dcf_base"] is None
    assert updated["dcf_view"] == "unavailable_data_validation"
    assert updated["pe_base"] == 20.0
    assert updated["overall_confidence"] == "low"
