"""Regression tests for the local review-rule module."""
import pandas as pd
import pytest
from position_decision_rules import evaluate_position_decisions, _load_policy

def policy():
    return {"single_name_max_weight": 0.5, "group_max_weights": {"semiconductor_related": 0.8}, "subgroup_max_weights": {"memory": 0.6}, "max_price_age_calendar_days": 5, "max_quote_age_days": 7, "max_financial_age_days": 180, "min_valid_valuation_methods_for_add": 2, "acceptable_confidence_for_add": ["medium", "high"], "require_confirmed_cash_for_add": True, "require_quote_price_match": True, "quote_price_max_deviation_pct": 3.0}

def inputs():
    holdings = pd.DataFrame([{"ticker":"AAA","name":"Test","shares":1,"risk_group":"semiconductor_related","subgroup":"memory"}])
    prices = pd.DataFrame({"AAA":[99.0,100.0]}, index=pd.to_datetime(["2026-10-29","2026-10-30"]))
    valuation = pd.DataFrame([{"ticker":"AAA","current_price":100.0,"price_date":"2026-10-30","data_validation_status":"PASS","data_validation_dcf_ready":True,"data_validation_pe_ready":True,"data_validation_quote_age_days":0,"data_validation_financial_age_days":30,"dcf_buy_below":105.0,"dcf_sell_review_above":150.0,"pe_buy_below":104.0,"pe_sell_review_above":145.0,"overall_confidence":"medium"}])
    return holdings, prices, valuation

def test_add_watch_requires_cash_confirmation():
    holdings, prices, valuation = inputs()
    p = policy()
    # This test isolates the cash gate; concentration precedence is tested separately.
    p["single_name_max_weight"] = 1.0
    p["group_max_weights"]["semiconductor_related"] = 1.0
    p["subgroup_max_weights"]["memory"] = 1.0
    result, _ = evaluate_position_decisions(holdings, prices, valuation, p, cash=1000.0, as_of_date="2026-10-30", etf_tickers=())
    assert result.iloc[0]["decision"] == "ADD_RESEARCH_WATCH"
    result, summary = evaluate_position_decisions(holdings, prices, valuation, p, cash=None, as_of_date="2026-10-30", etf_tickers=())
    assert result.iloc[0]["decision"] == "BLOCK_ADD_CASH_UNVERIFIED"
    assert summary["cash_source"] == "not_provided"

def test_risk_cap_breach_takes_precedence():
    holdings, prices, valuation = inputs()
    p = policy()
    p["single_name_max_weight"] = 0.5
    result, _ = evaluate_position_decisions(holdings, prices, valuation, p, as_of_date="2026-10-30", etf_tickers=())
    assert result.iloc[0]["decision"] == "REDUCE_REVIEW"

def test_missing_second_method_prevents_add_watch():
    holdings, prices, valuation = inputs()
    valuation.loc[0, "data_validation_pe_ready"] = False
    result, _ = evaluate_position_decisions(holdings, prices, valuation, policy(), cash=1000.0, as_of_date="2026-10-30", etf_tickers=())
    assert result.iloc[0]["decision"] == "HOLD_DATA_BLOCKED"

def test_policy_has_balanced_limits():
    p = _load_policy()
    assert p["single_name_max_weight"] == pytest.approx(0.15)
    assert p["group_max_weights"]["semiconductor_related"] == pytest.approx(0.45)
    assert p["subgroup_max_weights"]["memory"] == pytest.approx(0.25)



def test_quote_date_and_vendor_price_must_match_local_close() -> None:
    holdings, prices, valuation = inputs()
    valuation.loc[0, "current_price"] = 108.0
    result, _ = evaluate_position_decisions(
        holdings, prices, valuation, policy(), cash=1000.0,
        as_of_date="2026-10-30", etf_tickers=(),
    )
    assert result.iloc[0]["decision"] == "HOLD_DATA_BLOCKED"


def test_etf_is_classified_as_lookthrough_not_company_valuation() -> None:
    holdings, prices, _ = inputs()
    etf = holdings.iloc[[0]].copy()
    etf.loc[:, "ticker"] = "DRAM"
    etf.loc[:, "name"] = "Memory ETF"
    etf.loc[:, "risk_group"] = "semiconductor_related"
    etf.loc[:, "subgroup"] = "memory"
    etf_prices = prices[["AAA"]].rename(columns={"AAA":"DRAM"})
    result, _ = evaluate_position_decisions(
        etf, etf_prices, pd.DataFrame(), policy(), cash=1000.0,
        as_of_date="2026-10-30", etf_tickers=("DRAM",),
    )
    assert result.iloc[0]["decision"] == "ETF_LOOKTHROUGH_ONLY"
