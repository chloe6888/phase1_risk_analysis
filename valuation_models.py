#!/usr/bin/env python3
"""Transparent DCF and forward-PE valuation models.

All assumptions are explicit inputs. This module does not predict future prices.
DCF uses a five-year constant-growth forecast followed by a Gordon-growth
terminal value. PE valuation applies an explicitly configured multiple to EPS.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCENARIOS = ("bear", "base", "bull")


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _required_positive(value: Any, field: str) -> float:
    number = finite_number(value)
    if number is None or number <= 0:
        raise ValueError(f"{field} must be a finite number greater than zero.")
    return number


def dcf_valuation(
    *,
    free_cash_flow: float,
    diluted_shares: float,
    cash: float,
    debt: float,
    annual_growth: float,
    discount_rate: float,
    terminal_growth: float,
    forecast_years: int = 5,
) -> dict[str, float]:
    """Return FCFF-based enterprise/equity value and per-share intrinsic value.

    Cash flow and balance-sheet amounts must use the same currency and units.
    The cash-flow input should be unlevered FCF. Yahoo's reported freeCashflow is
    a vendor field and may not be identical to audited unlevered FCFF.
    """
    fcf0 = _required_positive(free_cash_flow, "free_cash_flow")
    shares = _required_positive(diluted_shares, "diluted_shares")
    cash_value = finite_number(cash)
    debt_value = finite_number(debt)
    if cash_value is None or cash_value < 0:
        raise ValueError("cash must be a finite non-negative number.")
    if debt_value is None or debt_value < 0:
        raise ValueError("debt must be a finite non-negative number.")
    growth = finite_number(annual_growth)
    wacc = finite_number(discount_rate)
    terminal = finite_number(terminal_growth)
    if growth is None or not -0.75 <= growth <= 1.0:
        raise ValueError("annual_growth must be between -75% and 100%.")
    if wacc is None or not 0 < wacc <= 0.5:
        raise ValueError("discount_rate must be greater than 0% and at most 50%.")
    if terminal is None or not -0.05 <= terminal <= 0.10:
        raise ValueError("terminal_growth must be between -5% and 10%.")
    if wacc <= terminal:
        raise ValueError("discount_rate must be greater than terminal_growth.")
    if isinstance(forecast_years, bool) or not isinstance(forecast_years, int) or not 1 <= forecast_years <= 20:
        raise ValueError("forecast_years must be an integer between 1 and 20.")

    projected: list[float] = []
    present_values: list[float] = []
    for year in range(1, forecast_years + 1):
        projected_fcf = fcf0 * ((1.0 + growth) ** year)
        projected.append(projected_fcf)
        present_values.append(projected_fcf / ((1.0 + wacc) ** year))

    terminal_value = projected[-1] * (1.0 + terminal) / (wacc - terminal)
    pv_terminal = terminal_value / ((1.0 + wacc) ** forecast_years)
    pv_explicit = sum(present_values)
    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value + cash_value - debt_value
    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": equity_value / shares,
        "present_value_explicit_period": pv_explicit,
        "terminal_value": terminal_value,
        "present_value_terminal_value": pv_terminal,
        "forecast_years": float(forecast_years),
        "forecast_year_1_fcf": projected[0],
        "forecast_year_n_fcf": projected[-1],
    }


def pe_valuation(*, forward_eps: float, fair_pe: float) -> dict[str, float]:
    """Apply an explicit forward PE multiple to positive forward EPS."""
    eps = _required_positive(forward_eps, "forward_eps")
    multiple = _required_positive(fair_pe, "fair_pe")
    if multiple > 100:
        raise ValueError("fair_pe must not exceed 100.")
    return {"forward_eps": eps, "fair_pe": multiple, "intrinsic_value_per_share": eps * multiple}


def validate_assumptions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one ticker's editable valuation assumptions."""
    if not isinstance(config, Mapping):
        raise ValueError("Valuation assumptions must be a JSON object.")
    margin = finite_number(config.get("margin_of_safety", 0.25))
    premium = finite_number(config.get("overvaluation_premium", 0.20))
    years_raw = config.get("forecast_years", 5)
    if margin is None or not 0 <= margin < 0.8:
        raise ValueError("margin_of_safety must be between 0% and 80%.")
    if premium is None or not 0 < premium <= 3:
        raise ValueError("overvaluation_premium must be greater than 0% and at most 300%.")
    if isinstance(years_raw, bool) or not isinstance(years_raw, int) or not 1 <= years_raw <= 20:
        raise ValueError("forecast_years must be an integer between 1 and 20.")
    scenarios_raw = config.get("scenarios")
    if not isinstance(scenarios_raw, Mapping):
        raise ValueError("assumptions.scenarios must contain bear, base, and bull.")
    normalized: dict[str, Any] = {
        "margin_of_safety": margin,
        "overvaluation_premium": premium,
        "forecast_years": years_raw,
        "cyclical": bool(config.get("cyclical", False)),
        "normalized_fcf": finite_number(config.get("normalized_fcf")),
        "notes": str(config.get("notes", "")),
        "assumption_source": str(config.get("assumption_source", "user-configured starter assumptions")),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        row = scenarios_raw.get(scenario)
        if not isinstance(row, Mapping):
            raise ValueError(f"Missing {scenario} valuation assumptions.")
        growth = finite_number(row.get("fcf_growth"))
        wacc = finite_number(row.get("discount_rate"))
        terminal = finite_number(row.get("terminal_growth"))
        pe = finite_number(row.get("fair_pe"))
        if growth is None or not -0.75 <= growth <= 1.0:
            raise ValueError(f"{scenario}.fcf_growth must be between -75% and 100%.")
        if wacc is None or not 0 < wacc <= 0.5:
            raise ValueError(f"{scenario}.discount_rate must be greater than 0% and at most 50%.")
        if terminal is None or not -0.05 <= terminal <= 0.10 or wacc <= terminal:
            raise ValueError(f"{scenario}.terminal_growth must be between -5% and 10% and below discount_rate.")
        if pe is None or not 1 <= pe <= 100:
            raise ValueError(f"{scenario}.fair_pe must be between 1 and 100.")
        normalized["scenarios"][scenario] = {
            "fcf_growth": growth,
            "discount_rate": wacc,
            "terminal_growth": terminal,
            "fair_pe": pe,
        }
    if normalized["normalized_fcf"] is not None and normalized["normalized_fcf"] <= 0:
        raise ValueError("normalized_fcf must be positive when supplied.")
    return normalized


def value_ticker(
    ticker: str,
    fundamentals: Mapping[str, Any],
    assumptions: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate scenario values and price bands for one company."""
    cfg = validate_assumptions(assumptions)
    symbol = ticker.strip().upper()
    price = finite_number(fundamentals.get("current_price"))
    shares = finite_number(fundamentals.get("diluted_shares"))
    cash = finite_number(fundamentals.get("cash"))
    debt = finite_number(fundamentals.get("debt"))
    forward_eps = finite_number(fundamentals.get("forward_eps"))
    reported_fcf = finite_number(fundamentals.get("free_cash_flow"))
    normalized_fcf = cfg["normalized_fcf"]
    valuation_fcf = normalized_fcf if normalized_fcf is not None else reported_fcf
    fcf_basis = "user_normalized_fcf" if normalized_fcf is not None else "reported_vendor_fcf"

    row: dict[str, Any] = {
        "ticker": symbol,
        "company_name": fundamentals.get("company_name"),
        "currency": fundamentals.get("currency", "USD"),
        "current_price": price,
        "price_date": fundamentals.get("price_date"),
        "data_fetched_at_utc": fundamentals.get("fetched_at_utc"),
        "financial_period": fundamentals.get("financial_period"),
        "data_source": fundamentals.get("data_source", "Yahoo Finance via yfinance"),
        "forward_eps": forward_eps,
        "eps_source": fundamentals.get("eps_source"),
        "free_cash_flow": reported_fcf,
        "valuation_fcf": valuation_fcf,
        "fcf_basis": fcf_basis if valuation_fcf is not None else "unavailable",
        "fcf_source": fundamentals.get("fcf_source"),
        "diluted_shares": shares,
        "cash": cash,
        "debt": debt,
        "market_cap": finite_number(fundamentals.get("market_cap")),
        "margin_of_safety": cfg["margin_of_safety"],
        "overvaluation_premium": cfg["overvaluation_premium"],
        "cyclical": cfg["cyclical"],
        "assumption_source": cfg["assumption_source"],
        "assumption_notes": cfg["notes"],
        "data_quality": fundamentals.get("data_quality", "unverified_vendor_snapshot"),
        "dcf_bear": None,
        "dcf_base": None,
        "dcf_bull": None,
        "pe_bear": None,
        "pe_base": None,
        "pe_bull": None,
        "dcf_buy_below": None,
        "dcf_sell_review_above": None,
        "pe_buy_below": None,
        "pe_sell_review_above": None,
        "dcf_view": "unavailable",
        "pe_view": "unavailable",
        "overall_confidence": "low",
        "limitations": "",
    }

    dcf_results: dict[str, float] = {}
    if valuation_fcf is not None and valuation_fcf > 0 and shares is not None and shares > 0 and cash is not None and debt is not None:
        for scenario, config in cfg["scenarios"].items():
            try:
                result = dcf_valuation(
                    free_cash_flow=valuation_fcf,
                    diluted_shares=shares,
                    cash=cash,
                    debt=debt,
                    annual_growth=config["fcf_growth"],
                    discount_rate=config["discount_rate"],
                    terminal_growth=config["terminal_growth"],
                    forecast_years=cfg["forecast_years"],
                )
                dcf_results[scenario] = result["intrinsic_value_per_share"]
                row[f"dcf_{scenario}"] = result["intrinsic_value_per_share"]
            except ValueError:
                continue
        base = dcf_results.get("base")
        if base is not None and base > 0:
            row["dcf_buy_below"] = base * (1.0 - cfg["margin_of_safety"])
            row["dcf_sell_review_above"] = base * (1.0 + cfg["overvaluation_premium"])
            if price is not None:
                row["dcf_view"] = _price_view(price, row["dcf_buy_below"], row["dcf_sell_review_above"])

    pe_results: dict[str, float] = {}
    if forward_eps is not None and forward_eps > 0:
        for scenario, config in cfg["scenarios"].items():
            try:
                result = pe_valuation(forward_eps=forward_eps, fair_pe=config["fair_pe"])
                pe_results[scenario] = result["intrinsic_value_per_share"]
                row[f"pe_{scenario}"] = result["intrinsic_value_per_share"]
            except ValueError:
                continue
        base = pe_results.get("base")
        if base is not None and base > 0:
            row["pe_buy_below"] = base * (1.0 - cfg["margin_of_safety"])
            row["pe_sell_review_above"] = base * (1.0 + cfg["overvaluation_premium"])
            if price is not None:
                row["pe_view"] = _price_view(price, row["pe_buy_below"], row["pe_sell_review_above"])

    confidence = "medium"
    limitations: list[str] = []
    if price is None or price <= 0:
        limitations.append("当前股价缺失，不能判断价格所处区间。")
        confidence = "low"
    if forward_eps is None or forward_eps <= 0:
        limitations.append("缺少正数远期 EPS，PE 估值不可用。")
        confidence = "low"
    if valuation_fcf is None or valuation_fcf <= 0:
        limitations.append("缺少正数自由现金流，DCF 估值不可用。")
        confidence = "low"
    if shares is None or shares <= 0 or cash is None or debt is None:
        limitations.append("股数、现金或债务字段不完整，DCF 估值不可用。")
        confidence = "low"
    if cfg["cyclical"]:
        limitations.append("周期行业：单年自由现金流和盈利可能处于周期高点或低点，应优先审查正常化 FCF/EPS。")
        confidence = "low"
    if normalized_fcf is not None:
        limitations.append("DCF 使用用户输入的正常化自由现金流，不是数据源报告值。")
    if fundamentals.get("stale_data"):
        limitations.append("本次使用了过期缓存，价格和财务数据日期必须人工核对。")
        confidence = "low"
    if fundamentals.get("price_date") is None:
        limitations.append("缺少报价日期。")
        confidence = "low"
    row["overall_confidence"] = confidence
    row["limitations"] = "；".join(limitations) if limitations else "估值依赖情景假设，不是价格预测；请核对最新财报和数据源。"
    return row


def _price_view(price: float, buy_below: float | None, sell_above: float | None) -> str:
    if buy_below is not None and price <= buy_below:
        return "below_buy_threshold"
    if sell_above is not None and price >= sell_above:
        return "above_sell_review_threshold"
    return "between_thresholds"
