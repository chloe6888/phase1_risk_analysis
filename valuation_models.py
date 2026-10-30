#!/usr/bin/env python3
"""Transparent DCF and forward-PE valuation models.

All assumptions are explicit inputs. This module does not predict future prices.
DCF uses a five-year equity-cash-flow proxy forecast followed by a Gordon-growth
terminal value. Yahoo's CFO-less-CapEx FCF is not a fully reconciled FCFE series,
so this is an indicative proxy; the discount rate is an editable cost-of-equity
assumption and cash/debt are not added or subtracted again. PE valuation applies
an explicitly configured multiple to EPS.
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
    annual_growth: float,
    cost_of_equity: float,
    terminal_growth: float,
    forecast_years: int = 5,
) -> dict[str, float]:
    """Return indicative equity value and per-share value from the FCF proxy.

    The cash-flow input is treated as an equity-cash-flow proxy. Yahoo's reported
    CFO-less-CapEx FCF is not fully reconciled FCFE; discount at cost of equity and
    do not add cash or subtract debt again. Treat the result as indicative only.
    """
    fcf0 = _required_positive(free_cash_flow, "free_cash_flow")
    shares = _required_positive(diluted_shares, "diluted_shares")
    growth = finite_number(annual_growth)
    cost = finite_number(cost_of_equity)
    terminal = finite_number(terminal_growth)
    if growth is None or not -0.75 <= growth <= 1.0:
        raise ValueError("annual_growth must be between -75% and 100%.")
    if cost is None or not 0 < cost <= 0.5:
        raise ValueError("cost_of_equity must be greater than 0% and at most 50%.")
    if terminal is None or not -0.05 <= terminal <= 0.10:
        raise ValueError("terminal_growth must be between -5% and 10%.")
    if cost <= terminal:
        raise ValueError("cost_of_equity must be greater than terminal_growth.")
    if isinstance(forecast_years, bool) or not isinstance(forecast_years, int) or not 1 <= forecast_years <= 20:
        raise ValueError("forecast_years must be an integer between 1 and 20.")

    projected: list[float] = []
    present_values: list[float] = []
    for year in range(1, forecast_years + 1):
        projected_fcf = fcf0 * ((1.0 + growth) ** year)
        projected.append(projected_fcf)
        present_values.append(projected_fcf / ((1.0 + cost) ** year))

    terminal_value = projected[-1] * (1.0 + terminal) / (cost - terminal)
    pv_terminal = terminal_value / ((1.0 + cost) ** forecast_years)
    pv_explicit = sum(present_values)
    equity_value = pv_explicit + pv_terminal
    return {
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
        "per_share_basis_verified": bool(config.get("per_share_basis_verified", True)),
        "per_share_basis_note": str(config.get("per_share_basis_note", "")),
        "adr_underlying_shares_per_ads": finite_number(config.get("adr_underlying_shares_per_ads")),
        "normalized_fcf": finite_number(config.get("normalized_fcf")),
        "normalized_fcf_currency": str(config.get("normalized_fcf_currency", "")).upper() or None,
        "notes": str(config.get("notes", "")),
        "assumption_source": str(config.get("assumption_source", "user-configured starter assumptions")),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        row = scenarios_raw.get(scenario)
        if not isinstance(row, Mapping):
            raise ValueError(f"Missing {scenario} valuation assumptions.")
        growth = finite_number(row.get("fcf_growth"))
        cost = finite_number(row.get("cost_of_equity"))
        terminal = finite_number(row.get("terminal_growth"))
        pe = finite_number(row.get("fair_pe"))
        if growth is None or not -0.75 <= growth <= 1.0:
            raise ValueError(f"{scenario}.fcf_growth must be between -75% and 100%.")
        if cost is None or not 0 < cost <= 0.5:
            raise ValueError(f"{scenario}.cost_of_equity must be greater than 0% and at most 50%.")
        if terminal is None or not -0.05 <= terminal <= 0.10 or cost <= terminal:
            raise ValueError(f"{scenario}.terminal_growth must be between -5% and 10% and below cost_of_equity.")
        if pe is None or not 1 <= pe <= 100:
            raise ValueError(f"{scenario}.fair_pe must be between 1 and 100.")
        normalized["scenarios"][scenario] = {
            "fcf_growth": growth,
            "cost_of_equity": cost,
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
    currency_mismatch = bool(fundamentals.get("currency_mismatch", False))
    dcf_currency_mismatch = bool(fundamentals.get("dcf_currency_mismatch", currency_mismatch))
    pe_currency_mismatch = bool(fundamentals.get("pe_currency_mismatch", currency_mismatch))
    reported_fcf = finite_number(fundamentals.get("free_cash_flow"))
    normalized_fcf = cfg["normalized_fcf"]
    valuation_fcf = normalized_fcf if normalized_fcf is not None else reported_fcf
    fcf_basis = "user_normalized_fcf" if normalized_fcf is not None else "reported_vendor_fcf"
    valuation_fcf_currency = (
        cfg["normalized_fcf_currency"]
        if normalized_fcf is not None and cfg["normalized_fcf_currency"]
        else fundamentals.get("financial_currency", fundamentals.get("currency", "USD"))
    )
    if normalized_fcf is not None:
        dcf_currency_mismatch = str(valuation_fcf_currency).upper() != str(fundamentals.get("currency", "USD")).upper()

    row: dict[str, Any] = {
        "ticker": symbol,
        "company_name": fundamentals.get("company_name"),
        "currency": fundamentals.get("currency", "USD"),
        "financial_currency": fundamentals.get("financial_currency", fundamentals.get("currency", "USD")),
        "eps_currency": fundamentals.get("eps_currency", fundamentals.get("currency", "USD")),
        "currency_mismatch": currency_mismatch,
        "dcf_currency_mismatch": dcf_currency_mismatch,
        "pe_currency_mismatch": pe_currency_mismatch,
        "current_price": price,
        "price_date": fundamentals.get("price_date"),
        "data_fetched_at_utc": fundamentals.get("fetched_at_utc"),
        "cache_age_hours": finite_number(fundamentals.get("cache_age_hours")),
        "stale_data": bool(fundamentals.get("stale_data", False)),
        "financial_period": fundamentals.get("financial_period"),
        "data_source": fundamentals.get("data_source", "Yahoo Finance via yfinance"),
        "forward_eps": forward_eps,
        "eps_source": fundamentals.get("eps_source"),
        "free_cash_flow": reported_fcf,
        "valuation_fcf": valuation_fcf,
        "valuation_fcf_currency": valuation_fcf_currency,
        "fcf_basis": fcf_basis if valuation_fcf is not None else "unavailable",
        "fcf_source": fundamentals.get("fcf_source"),
        "diluted_shares": shares,
        "cash": cash,
        "debt": debt,
        "market_cap": finite_number(fundamentals.get("market_cap")),
        "revenue": finite_number(fundamentals.get("revenue")),
        "gross_profit": finite_number(fundamentals.get("gross_profit")),
        "operating_income": finite_number(fundamentals.get("operating_income")),
        "net_income": finite_number(fundamentals.get("net_income")),
        "ebitda": finite_number(fundamentals.get("ebitda")),
        "revenue_growth_yoy": finite_number(fundamentals.get("revenue_growth_yoy")),
        "gross_margin": finite_number(fundamentals.get("gross_margin")),
        "operating_margin": finite_number(fundamentals.get("operating_margin")),
        "net_margin": finite_number(fundamentals.get("net_margin")),
        "return_on_equity": finite_number(fundamentals.get("return_on_equity")),
        "return_on_assets": finite_number(fundamentals.get("return_on_assets")),
        "trailing_pe": finite_number(fundamentals.get("trailing_pe")),
        "forward_pe": finite_number(fundamentals.get("forward_pe")),
        "financial_metric_sources": fundamentals.get("financial_metric_sources", {}),
        "margin_of_safety": cfg["margin_of_safety"],
        "overvaluation_premium": cfg["overvaluation_premium"],
        "cyclical": cfg["cyclical"],
        "per_share_basis_verified": cfg["per_share_basis_verified"],
        "per_share_basis_note": cfg["per_share_basis_note"],
        "adr_underlying_shares_per_ads": cfg["adr_underlying_shares_per_ads"],
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
    if cfg["per_share_basis_verified"] and not dcf_currency_mismatch and valuation_fcf is not None and valuation_fcf > 0 and shares is not None and shares > 0:
        for scenario, config in cfg["scenarios"].items():
            try:
                result = dcf_valuation(
                    free_cash_flow=valuation_fcf,
                    diluted_shares=shares,
                    annual_growth=config["fcf_growth"],
                    cost_of_equity=config["cost_of_equity"],
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
    if cfg["per_share_basis_verified"] and not pe_currency_mismatch and forward_eps is not None and forward_eps > 0:
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
    if not cfg["per_share_basis_verified"]:
        limitations.append(
            "每股估值口径尚未核验：" + (cfg["per_share_basis_note"] or "数据源的 EPS/股数是否按当前上市证券单位披露尚未确认。") + "；DCF/PE 均已暂停。"
        )
        confidence = "low"
    if dcf_currency_mismatch:
        limitations.append(
            f"报价币种 {fundamentals.get('currency')} 与自由现金流报告币种 {fundamentals.get('financial_currency')} 不一致，未进行汇率换算；DCF 已暂停。"
        )
        confidence = "low"
    if pe_currency_mismatch:
        limitations.append(
            f"报价币种 {fundamentals.get('currency')} 与 EPS 币种 {fundamentals.get('eps_currency')} 不一致，未进行汇率换算；PE 已暂停。"
        )
        confidence = "low"
    if price is None or price <= 0:
        limitations.append("当前股价缺失，不能判断价格所处区间。")
        confidence = "low"
    if forward_eps is None or forward_eps <= 0:
        limitations.append("缺少正数远期 EPS，PE 估值不可用。")
        confidence = "low"
    if valuation_fcf is None or valuation_fcf <= 0:
        limitations.append("缺少正数自由现金流，DCF 估值不可用。")
        confidence = "low"
    if shares is None or shares <= 0:
        limitations.append("缺少有效股数，DCF 估值不可用。")
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
