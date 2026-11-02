"""Rule-based, read-only add/hold/reduce review for current portfolio positions.

The engine combines:
- current raw Close * confirmed shares;
- single-name, parent-group and subgroup concentration limits;
- fresh/valid DCF and forward-PE outputs;
- valuation confidence and quote-price reconciliation;
- optional, explicitly supplied account cash.

It produces research statuses only. It never places orders or sends private
holdings to an external service.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from holdings_manager import load_holdings
from portfolio_risk import load_price_panel
from run_personal_portfolio import DEFAULT_CONFIG, DEFAULT_HOLDINGS, load_universe, validate_holdings_file

ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = ROOT / "risk_profiles" / "position_decision_policy.json"
DEFAULT_PRICES = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_VALUATION = ROOT / "outputs" / "valuation" / "valuation_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "position_decisions"

ACTION_LABELS = {
    "REDUCE_REVIEW": "减仓/再平衡复核",
    "ADD_RESEARCH_WATCH": "加仓研究观察（非交易指令）",
    "BLOCK_ADD_CASH_UNVERIFIED": "暂不加仓：现金未核实",
    "BLOCK_ADD_NO_CASH": "暂不加仓：没有可用现金",
    "BLOCK_ADD_RISK_CAP": "暂不加仓：风险额度不足",
    "HOLD_DATA_BLOCKED": "继续观察：数据未通过校验",
    "HOLD_LOW_CONFIDENCE": "继续观察：估值可信度不足",
    "HOLD_RESEARCH_REQUIRED": "继续观察：估值证据不足",
    "HOLD_VALUATION_MISSING": "继续观察：尚无估值结果",
    "ETF_LOOKTHROUGH_ONLY": "ETF：使用底层持仓穿透分析",
    "HOLD": "按当前规则继续持有观察",
}


def _load_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    source = Path(path)
    try:
        policy = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read position decision policy: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("Position decision policy must be a schema_version=1 JSON object.")

    def _cap_map(key: str) -> dict[str, float]:
        raw = policy.get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"{key} must be a JSON object.")
        result = {str(k).strip(): float(v) for k, v in raw.items()}
        if any(not k or not math.isfinite(v) or not 0 <= v <= 1 for k, v in result.items()):
            raise ValueError(f"{key} contains an invalid key or cap; values must be in [0, 1].")
        return result

    single_cap = float(policy.get("single_name_max_weight", float("nan")))
    if not math.isfinite(single_cap) or not 0 < single_cap <= 1:
        raise ValueError("single_name_max_weight must be finite and in (0, 1].")
    policy["single_name_max_weight"] = single_cap
    policy["group_max_weights"] = _cap_map("group_max_weights")
    policy["subgroup_max_weights"] = _cap_map("subgroup_max_weights")
    for key in ("max_price_age_calendar_days", "max_quote_age_days", "max_financial_age_days"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    method_count = policy.get("min_valid_valuation_methods_for_add", 2)
    if isinstance(method_count, bool) or not isinstance(method_count, int) or method_count < 1:
        raise ValueError("min_valid_valuation_methods_for_add must be a positive integer.")
    acceptable = policy.get("acceptable_confidence_for_add", ["medium", "high"])
    if not isinstance(acceptable, list) or not acceptable:
        raise ValueError("acceptable_confidence_for_add must be a non-empty list.")
    if not set(map(str, acceptable)).issubset({"low", "medium", "high"}):
        raise ValueError("acceptable_confidence_for_add may contain only low, medium, high.")
    policy["acceptable_confidence_for_add"] = [str(x) for x in acceptable]
    deviation = float(policy.get("quote_price_max_deviation_pct", 3.0))
    if not math.isfinite(deviation) or not 0 <= deviation <= 50:
        raise ValueError("quote_price_max_deviation_pct must be between 0 and 50.")
    policy["quote_price_max_deviation_pct"] = deviation
    return policy


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ready"}


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_date(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of_date must be a valid date.") from exc


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def evaluate_position_decisions(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    valuation_summary: pd.DataFrame | None,
    policy: Mapping[str, Any],
    *,
    cash: float | None = None,
    as_of_date: date | str | None = None,
    etf_tickers: tuple[str, ...] = ("DRAM", "QQQ"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one row per position and return a portfolio-level rule summary."""
    if holdings.empty:
        raise ValueError("Holdings cannot be empty.")
    if not {"ticker", "shares", "risk_group", "subgroup"}.issubset(holdings.columns):
        raise ValueError("Holdings require ticker, shares, risk_group and subgroup.")
    if cash is not None and (not math.isfinite(float(cash)) or float(cash) < 0):
        raise ValueError("cash must be finite and non-negative.")

    reference_date = _as_date(as_of_date)
    prices = close_prices.copy()
    if prices.empty:
        raise ValueError("Close-price panel cannot be empty.")
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.has_duplicates or prices.index.hasnans:
        raise ValueError("Close-price panel must have unique valid dates.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")

    positions = holdings.copy()
    positions["ticker"] = positions["ticker"].astype(str).str.strip().str.upper()
    if positions["ticker"].eq("").any() or positions["ticker"].duplicated().any():
        raise ValueError("Holdings tickers must be unique and non-empty.")
    positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce")
    if not np.isfinite(positions["shares"].to_numpy(dtype=float)).all() or (positions["shares"] <= 0).any():
        raise ValueError("Holdings shares must be finite and positive.")

    missing = sorted(set(positions["ticker"]) - set(map(str, prices.columns)))
    if missing:
        raise ValueError(f"Close-price panel is missing holdings: {missing}")

    price_rows: list[dict[str, Any]] = []
    for ticker in positions["ticker"]:
        series = prices[ticker].dropna()
        if series.empty:
            raise ValueError(f"{ticker}: no valid raw close price.")
        price_date = pd.Timestamp(series.index[-1]).date()
        age = (reference_date - price_date).days
        if age < 0:
            raise ValueError(f"{ticker}: latest close is later than the analysis date.")
        if age > int(policy["max_price_age_calendar_days"]):
            raise ValueError(
                f"{ticker}: close price is {age} days old, over policy limit "
                f"{policy['max_price_age_calendar_days']}."
            )
        price = float(series.iloc[-1])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"{ticker}: latest close must be finite and positive.")
        price_rows.append({"ticker": ticker, "current_price": price, "price_date": price_date.isoformat(), "price_age_days": age})
    price_frame = pd.DataFrame(price_rows)

    positions = positions.merge(price_frame, on="ticker", how="left", validate="one_to_one")
    positions["market_value"] = positions["shares"] * positions["current_price"]
    listed_value = float(positions["market_value"].sum())
    cash_value = 0.0 if cash is None else float(cash)
    net_equity = listed_value + cash_value
    if not math.isfinite(net_equity) or net_equity <= 0:
        raise ValueError("Listed value plus cash must be positive.")
    positions["weight"] = positions["market_value"] / net_equity

    group_exposure = positions.groupby("risk_group")["weight"].sum().to_dict()
    subgroup_exposure = positions.groupby("subgroup")["weight"].sum().to_dict()
    valuation = valuation_summary.copy() if valuation_summary is not None else pd.DataFrame()
    if not valuation.empty:
        if "ticker" not in valuation.columns:
            raise ValueError("Valuation summary must contain ticker.")
        valuation["ticker"] = valuation["ticker"].astype(str).str.strip().str.upper()
        if valuation["ticker"].eq("").any() or valuation["ticker"].duplicated().any():
            raise ValueError("Valuation summary tickers must be unique and non-empty.")
    valuation_by_ticker = (
        {row["ticker"]: row for row in valuation.to_dict("records")}
        if not valuation.empty else {}
    )

    output: list[dict[str, Any]] = []
    for row in positions.itertuples(index=False):
        ticker = row.ticker
        group = str(row.risk_group)
        subgroup = str(row.subgroup)
        current_weight = float(row.weight)
        price = float(row.current_price)
        reasons: list[dict[str, str]] = []
        flags: list[str] = []

        asset_cap = float(policy["single_name_max_weight"])
        group_cap = policy["group_max_weights"].get(group)
        subgroup_cap = policy["subgroup_max_weights"].get(subgroup)
        asset_breach = current_weight > asset_cap + 1e-9
        group_breach = group_cap is not None and float(group_exposure.get(group, 0.0)) > float(group_cap) + 1e-9
        subgroup_breach = subgroup_cap is not None and float(subgroup_exposure.get(subgroup, 0.0)) > float(subgroup_cap) + 1e-9
        if asset_breach:
            flags.append("single_name_cap")
            reasons.append(_reason("SINGLE_NAME_CAP_BREACH", f"当前权重 {current_weight:.2%}，上限 {asset_cap:.2%}。"))
        if group_breach:
            flags.append("group_cap")
            reasons.append(_reason("GROUP_CAP_BREACH", f"{group} 主题占比 {float(group_exposure[group]):.2%}，上限 {float(group_cap):.2%}。"))
        if subgroup_breach:
            flags.append("subgroup_cap")
            reasons.append(_reason("SUBGROUP_CAP_BREACH", f"{subgroup} 子主题占比 {float(subgroup_exposure[subgroup]):.2%}，上限 {float(subgroup_cap):.2%}。"))

        group_headroom = float(group_cap) - float(group_exposure.get(group, 0.0)) if group_cap is not None else 1.0
        subgroup_headroom = float(subgroup_cap) - float(subgroup_exposure.get(subgroup, 0.0)) if subgroup_cap is not None else 1.0
        headroom = max(0.0, min(asset_cap - current_weight, group_headroom, subgroup_headroom))

        v = valuation_by_ticker.get(ticker)
        if ticker in set(etf_tickers):
            valuation_status = "ETF_LOOKTHROUGH_ONLY"
            methods: dict[str, dict[str, Any]] = {}
            data_blockers = []
            valuation_signal = "not_applicable"
            reasons.append(_reason("ETF_NOT_COMPANY_VALUATION", "ETF 不使用公司 DCF/PE；应核验最新基金持仓与底层发行人敞口。"))
        elif v is None:
            valuation_status = "missing"
            methods = {}
            data_blockers = ["尚无估值结果。"]
            valuation_signal = "insufficient"
            reasons.append(_reason("VALUATION_MISSING", "当前没有该标的估值报告。"))
        else:
            valuation_status = str(v.get("data_validation_status", "BLOCKED")).upper()
            dcf_ready = _as_bool(v.get("data_validation_dcf_ready"))
            pe_ready = _as_bool(v.get("data_validation_pe_ready"))
            quote_age = _as_float(v.get("data_validation_quote_age_days"))
            financial_age = _as_float(v.get("data_validation_financial_age_days"))
            confidence = str(v.get("overall_confidence", "low")).strip().lower()
            vendor_price = _as_float(v.get("current_price"))
            price_date_value = v.get("price_date")
            data_blockers = []
            if not dcf_ready:
                data_blockers.append("DCF 数据校验未通过或不可用。")
            if not pe_ready:
                data_blockers.append("远期 PE 数据校验未通过或不可用。")
            if valuation_status == "BLOCKED":
                data_blockers.append("估值数据质量状态为 BLOCKED。")
            if quote_age is None or quote_age > float(policy["max_quote_age_days"]):
                data_blockers.append("估值报价过期或缺少报价日期。")
            if financial_age is None or financial_age > float(policy["max_financial_age_days"]):
                data_blockers.append("财务数据过期或缺少财务期间。")
            if price_date_value is not None and not pd.isna(price_date_value):
                try:
                    valuation_price_date = pd.Timestamp(price_date_value).date()
                    if abs((valuation_price_date - row.price_date).days) > 1:
                        data_blockers.append("估值报价日期与本地行情面板日期不一致。")
                except (TypeError, ValueError):
                    data_blockers.append("估值报价日期格式无效。")
            else:
                data_blockers.append("估值结果缺少 price_date。")
            if policy.get("require_quote_price_match", True) and vendor_price is not None:
                deviation = abs(vendor_price - price) / price * 100.0
                if deviation > float(policy["quote_price_max_deviation_pct"]):
                    data_blockers.append(
                        f"估值供应商报价与本地收盘价偏差 {deviation:.2f}%，超过 "
                        f"{policy['quote_price_max_deviation_pct']:.2f}% 核对阈值。"
                    )
            methods = {}
            for method, ready, buy_column, sell_column in (
                ("dcf", dcf_ready, "dcf_buy_below", "dcf_sell_review_above"),
                ("pe", pe_ready, "pe_buy_below", "pe_sell_review_above"),
            ):
                buy = _as_float(v.get(buy_column))
                sell = _as_float(v.get(sell_column))
                if ready and buy is not None and sell is not None and buy > 0 and sell > buy:
                    methods[method] = {"buy_below": buy, "sell_review_above": sell}
                elif ready:
                    data_blockers.append(f"{method.upper()} 观察价格线无效或缺失。")
            if not methods:
                valuation_signal = "insufficient"
            elif data_blockers or len(methods) < int(policy["min_valid_valuation_methods_for_add"]):
                valuation_signal = "insufficient"
            elif all(price <= x["buy_below"] for x in methods.values()):
                valuation_signal = "attractive"
            elif all(price >= x["sell_review_above"] for x in methods.values()):
                valuation_signal = "above_review"
            else:
                valuation_signal = "within_or_mixed"

        # Risk-limit breaches take precedence over valuation-based add watches.
        action = "HOLD"
        if asset_breach or group_breach or subgroup_breach:
            action = "REDUCE_REVIEW"
            flags.append("reduce_review")
        elif valuation_status == "ETF_LOOKTHROUGH_ONLY":
            action = "ETF_LOOKTHROUGH_ONLY"
        elif valuation_signal == "above_review":
            action = "REDUCE_REVIEW"
            flags.append("valuation_above_review")
            reasons.append(_reason("BOTH_VALUATION_MODELS_ABOVE_REVIEW", "DCF 与远期 PE 均达到卖出复核线以上；请复核增长假设和仓位，不是自动卖出信号。"))
        elif valuation_signal == "insufficient":
            action = "HOLD_DATA_BLOCKED" if data_blockers else "HOLD_RESEARCH_REQUIRED"
            if data_blockers:
                flags.append("valuation_data_blocked")
            reasons.extend(_reason("VALUATION_DATA_LIMITATION", item) for item in data_blockers)
        elif valuation_signal == "attractive":
            confidence = str(v.get("overall_confidence", "low")).strip().lower() if v is not None else "low"
            if confidence not in policy["acceptable_confidence_for_add"]:
                action = "HOLD_LOW_CONFIDENCE"
                reasons.append(_reason("VALUATION_CONFIDENCE_TOO_LOW", f"当前估值可信度为 {confidence}，加仓观察要求 {policy['acceptable_confidence_for_add']}。"))
            elif policy.get("require_confirmed_cash_for_add", True) and cash is None:
                action = "BLOCK_ADD_CASH_UNVERIFIED"
                reasons.append(_reason("CASH_NOT_CONFIRMED", "未提供经过核实的账户现金余额，不能计算可用加仓金额。"))
            elif cash_value <= 0:
                action = "BLOCK_ADD_NO_CASH"
                reasons.append(_reason("NO_AVAILABLE_CASH", "已确认现金余额不大于 0。"))
            elif headroom <= 1e-9:
                action = "BLOCK_ADD_RISK_CAP"
                reasons.append(_reason("NO_RISK_BUDGET_HEADROOM", "单股、主题或子主题上限没有剩余额度。"))
            else:
                action = "ADD_RESEARCH_WATCH"
                reasons.append(_reason("VALUATION_AND_RISK_HEADROOM_PASS", "双模型安全边际、数据质量、可信度与风险限额条件通过；仍需人工研究和手动决策。"))

        max_add_amount = min(cash_value, net_equity * headroom) if cash is not None else None
        output.append({
            "ticker": ticker,
            "name": getattr(row, "name", ticker),
            "current_price": price,
            "price_date": row.price_date,
            "shares": float(row.shares),
            "market_value": float(row.market_value),
            "portfolio_weight": current_weight,
            "risk_group": group,
            "group_exposure": float(group_exposure.get(group, 0.0)),
            "group_cap": group_cap,
            "subgroup": subgroup,
            "subgroup_exposure": float(subgroup_exposure.get(subgroup, 0.0)),
            "subgroup_cap": subgroup_cap,
            "single_name_cap": asset_cap,
            "max_add_weight_headroom": headroom,
            "valuation_status": valuation_status,
            "valuation_signal": valuation_signal,
            "valuation_methods_ready": sorted(methods),
            "valuation_confidence": str(v.get("overall_confidence", "low")) if v is not None else None,
            "decision": action,
            "decision_label": ACTION_LABELS[action],
            "max_add_amount_usd": max_add_amount,
            "cash_source": "user_provided" if cash is not None else "not_provided",
            "flags": sorted(set(flags)),
            "reasons": json.dumps(reasons, ensure_ascii=False),
        })

    result = pd.DataFrame(output).sort_values(
        ["decision", "portfolio_weight"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
    counts = result["decision"].value_counts().to_dict()
    summary = {
        "as_of_date": reference_date.isoformat(),
        "position_count": int(len(result)),
        "listed_positions_market_value": listed_value,
        "cash_value": cash_value,
        "cash_source": "user_provided" if cash is not None else "not_provided",
        "net_equity_denominator": net_equity,
        "denominator_note": (
            "Cash was provided by the user and included."
            if cash is not None else
            "Cash was not provided; weights use listed holdings only. This does not assert account cash is zero."
        ),
        "decision_counts": {str(k): int(v) for k, v in counts.items()},
        "policy_single_name_cap": float(policy["single_name_max_weight"]),
        "limitations": [
            "This engine produces review statuses, not orders or personalized guarantees.",
            "A concentration breach triggers a reduce/rebalance review; no immediate sale is implied.",
            "Add research watch requires both DCF and forward-PE thresholds to pass; the underlying assumptions may still be wrong.",
            "Cash is unverified when not supplied, and no add amount can be calculated.",
            "Current weights use the imported listed holdings and any explicitly supplied cash; other accounts, liabilities and assets are outside this denominator.",
            "Price and fundamental freshness are checked against policy, but Yahoo/yfinance data is not independently audited against filings.",
            "ETF holdings are not valued with company DCF/PE. Use dated DRAM/QQQ issuer look-through for indirect exposure.",
            "No tax, spread, slippage, liquidity, order-lot, market-impact or trade-timing analysis is performed.",
        ],
    }
    return result, summary


def write_decision_report(result: pd.DataFrame, summary: Mapping[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.to_csv(target / "position_decision_review.csv", index=False)
    (target / "position_decision_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 持仓加减仓规则复核",
        "",
        f"- 分析日期：{summary['as_of_date']}",
        f"- 持仓数量：{summary['position_count']}",
        f"- 股票/ETF 市值：{summary['listed_positions_market_value']:,.2f} USD",
        f"- 现金口径：{summary['denominator_note']}",
        "",
        "## 规则输出",
        "",
        "| 标的 | 当前权重 | 估值信号 | 规则结论 | 可用风险额度 |",
        "|---|---:|---|---|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.portfolio_weight:.2%} | {row.valuation_signal} | "
            f"{row.decision_label} | {row.max_add_weight_headroom:.2%} |"
        )
    lines += ["", "## 逐项复核依据", ""]
    for row in result.itertuples(index=False):
        lines += [f"### {row.ticker}", "", f"- 结论：{row.decision_label}"]
        reasons = json.loads(row.reasons)
        if reasons:
            lines.extend(f"- {item['detail']}" for item in reasons)
        else:
            lines.append("- 当前未触发特殊复核规则。")
        lines.append("")
    lines += ["## 适用限制", ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    (target / "position_decision_review.md").write_text("\n".join(lines), encoding="utf-8")


def run_position_decision_review(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    holdings_path: str | Path = DEFAULT_HOLDINGS,
    prices_path: str | Path = DEFAULT_PRICES,
    valuation_path: str | Path = DEFAULT_VALUATION,
    policy_path: str | Path = DEFAULT_POLICY,
    output_dir: str | Path = DEFAULT_OUTPUT,
    cash: float | None = None,
    as_of_date: date | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    universe = load_universe(config_path)
    holdings = validate_holdings_file(holdings_path, universe)
    prices = load_price_panel(prices_path)
    valuation_file = Path(valuation_path)
    valuations = pd.read_csv(valuation_file) if valuation_file.is_file() else pd.DataFrame()
    policy = _load_policy(policy_path)
    result, summary = evaluate_position_decisions(
        holdings,
        prices,
        valuations,
        policy,
        cash=cash,
        as_of_date=as_of_date,
        etf_tickers=tuple(
            ticker for ticker, meta in universe["symbols"].items()
            if meta.get("instrument_type") == "etf"
        ),
    )
    write_decision_report(result, summary, output_dir)
    summary["report_path"] = str(Path(output_dir) / "position_decision_review.md")
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only add/hold/reduce rule review. This script never submits orders."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--valuation-summary", type=Path, default=DEFAULT_VALUATION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cash", type=float, default=None, help="Optional verified available cash in USD.")
    parser.add_argument("--as-of-date", help="Analysis date YYYY-MM-DD.")
    args = parser.parse_args()
    try:
        _, summary = run_position_decision_review(
            config_path=args.config,
            holdings_path=args.holdings_csv,
            prices_path=args.prices,
            valuation_path=args.valuation_summary,
            policy_path=args.policy,
            output_dir=args.output_dir,
            cash=args.cash,
            as_of_date=args.as_of_date,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
