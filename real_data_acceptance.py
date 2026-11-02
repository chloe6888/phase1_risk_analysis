#!/usr/bin/env python3
"""Local-only acceptance checks for real holdings and market-price data.

The report deliberately excludes share quantities, average costs, market values,
and account balances. It never uploads or transmits private inputs.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from holdings_manager import (
    load_holdings,
    reconcile_snapshot,
    risk_guardrail_alerts,
    summarize_holdings,
    weights_from_price_panel,
)
from portfolio_risk import ROOT, load_price_panel

DEFAULT_HOLDINGS = ROOT / "data" / "private" / "holdings_snapshot.csv"
DEFAULT_CLOSE = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_ADJUSTED = ROOT / "data" / "raw" / "adjusted_close_panel.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "acceptance"


def _currency_diagnostic(path: Path) -> list[str]:
    """Best-effort currency discovery when the canonical loader rejects a file."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "currency" not in frame:
        return ["USD"]
    values = (
        frame["currency"]
        .fillna("USD")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", "USD")
    )
    return sorted(set(values.tolist()))


def _explicit_currency_issue(path: Path) -> str | None:
    """Require a declared quote currency and verify common exchange suffixes."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "ticker" not in frame:
        return None
    if "currency" not in frame.columns:
        return (
            "持仓 CSV 缺少 currency 列；请逐行明确填写证券报价币种（如 USD、HKD、KRW），"
            "不要依赖导入器将缺省币种设为 USD。"
        )
    expected_by_suffix = {
        ".HK": "HKD",
        ".KS": "KRW",
        ".KQ": "KRW",
        ".T": "JPY",
        ".L": "GBP",
        ".TO": "CAD",
        ".AX": "AUD",
        ".SS": "CNY",
        ".SZ": "CNY",
        ".TW": "TWD",
    }
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip().upper()
        expected = next(
            (currency for suffix, currency in expected_by_suffix.items() if ticker.endswith(suffix)),
            None,
        )
        raw_currency = row.get("currency")
        if pd.isna(raw_currency):
            return f"{ticker or '某一持仓'} 的 currency 为空；请填写该证券实际报价币种。"
        supplied = str(raw_currency).strip().upper()
        if not supplied or supplied in {"NAN", "NONE", "NULL", "<NA>"}:
            return f"{ticker or '某一持仓'} 的 currency 为空；请填写该证券实际报价币种。"
        if len(supplied) != 3 or not supplied.isalpha():
            return f"{ticker or '某一持仓'} 的 currency 格式无效；请填写三位字母币种代码（如 USD、HKD、KRW）。"
        if expected is not None and supplied != expected:
            return f"{ticker} 的代码后缀对应 {expected}，但文件填写为 {supplied}；请先核对报价币种。"
    return None


def _parse_as_of(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("as_of_date must use YYYY-MM-DD.") from exc


def _status(checks: list[dict[str, Any]]) -> str:
    levels = {str(item["level"]) for item in checks}
    if "BLOCKER" in levels:
        return "BLOCKED"
    if "WARNING" in levels:
        return "WARNING"
    return "PASS"


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 真实数据本地验收报告",
        "",
        f"- 生成时间（UTC）：{payload['generated_at_utc']}",
        f"- 验收状态：**{payload['status']}**",
        "- 隐私说明：本报告不包含持股数量、成本价、市值或账户余额；原始文件仅在本机读取。",
        "",
        "## 检查结果",
        "",
        "| 检查项 | 级别 | 结果 |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(
            f"| {item['code']} | {item['level']} | {item['message']} |"
        )
    lines += ["", "## 数据概况", ""]
    overview = payload.get("overview") or {}
    if not overview:
        lines.append("由于前置检查未通过，尚未形成完整数据概况。")
    else:
        for key, label in (
            ("position_count", "持仓行数"),
            ("currency_count", "币种数量"),
            ("currencies", "币种"),
            ("raw_price_last_date", "未调整收盘价面板最新日期"),
            ("adjusted_price_last_date", "调整收盘价面板最新日期"),
            ("common_return_observations", "共同收益观测数"),
            ("stale_ticker_count", "过期未调整价格标的数量"),
            ("stale_adjusted_ticker_count", "过期调整价格标的数量"),
            ("missing_raw_ticker_count", "未调整价格缺失标的数量"),
            ("missing_adjusted_ticker_count", "调整价格缺失标的数量"),
            ("concentration_alert_count", "集中度提醒数量"),
            ("abnormal_return_observation_count", "超过 35% 的单日收益观测数"),
            ("broker_reconciliation", "券商汇总核对"),
        ):
            if key in overview:
                value = overview[key]
                if isinstance(value, list):
                    value = "、".join(str(item) for item in value)
                lines.append(f"- {label}：{value}")
    lines += [
        "",
        "## 使用边界",
        "",
        "- 这是数据接入与口径验收，不是投资建议或买卖信号。",
        "- 当前组合导入仅支持单一计价币种；USD、HKD 等混合币种必须先完成有日期依据的汇率换算，不能直接合并。",
        "- 风险权重只覆盖导入的股票和 ETF；未导入的现金、债券、其他账户和负债不会自动纳入。",
        "- 行情来自本地价格面板；上游数据源及公司行为调整仍需核对，不能视为审计级行情。",
        "- 通过本验收不代表策略有效，也不代表估值输入已通过基本面核验。",
        "",
    ]
    return "\n".join(lines)


def run_acceptance(
    holdings_path: str | Path = DEFAULT_HOLDINGS,
    close_prices_path: str | Path = DEFAULT_CLOSE,
    adjusted_prices_path: str | Path = DEFAULT_ADJUSTED,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    as_of_date: date | str | None = None,
    max_staleness_days: int = 7,
    minimum_common_observations: int = 60,
) -> dict[str, Any]:
    """Validate local holdings and price panels without exposing position amounts."""
    as_of = _parse_as_of(as_of_date)
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days must be non-negative.")
    if minimum_common_observations < 2:
        raise ValueError("minimum_common_observations must be at least 2.")

    checks: list[dict[str, Any]] = []

    def add(code: str, level: str, message: str) -> None:
        checks.append({"code": code, "level": level, "message": message})

    holdings_file = Path(holdings_path)
    close_file = Path(close_prices_path)
    adjusted_file = Path(adjusted_prices_path)
    overview: dict[str, Any] = {}

    if not holdings_file.is_file():
        add("HOLDINGS_FILE_MISSING", "BLOCKER", "未找到持仓 CSV；请确认本地路径。")
        holdings = None
    else:
        currency_issue = _explicit_currency_issue(holdings_file)
        if currency_issue:
            holdings = None
            add("EXCHANGE_CURRENCY_UNVERIFIED", "BLOCKER", currency_issue)
        else:
            try:
                holdings = load_holdings(holdings_file)
                currencies = sorted(holdings["currency"].astype(str).str.upper().unique().tolist())
                overview.update({
                    "position_count": int(len(holdings)),
                    "currency_count": int(len(currencies)),
                    "currencies": currencies,
                })
                add("HOLDINGS_SCHEMA", "PASS", "必需字段、数值、正股数、唯一代码和币种检查通过。")
                if len(currencies) != 1:
                    add(
                        "MIXED_CURRENCIES",
                        "BLOCKER",
                        "当前版本不支持混合计价币种；请先建立同一基准币种的、带日期的汇率换算流程。",
                    )
            except Exception as exc:
                holdings = None
                currencies = _currency_diagnostic(holdings_file)
                if len(currencies) > 1:
                    add(
                        "MIXED_CURRENCIES",
                        "BLOCKER",
                        f"检测到多个计价币种（{', '.join(currencies)}），当前版本拒绝合并计算。",
                    )
                else:
                    add("HOLDINGS_SCHEMA", "BLOCKER", f"持仓文件未通过结构检查：{exc}")

    close_panel: pd.DataFrame | None = None
    adjusted_panel: pd.DataFrame | None = None
    if not close_file.is_file():
        add("RAW_CLOSE_PANEL_MISSING", "BLOCKER", "未找到未调整收盘价面板。")
    else:
        try:
            close_panel = load_price_panel(close_file)
            add("RAW_CLOSE_PANEL", "PASS", "未调整收盘价面板结构、日期和正价格检查通过。")
            overview["raw_price_last_date"] = pd.Timestamp(close_panel.index.max()).date().isoformat()
        except Exception as exc:
            add("RAW_CLOSE_PANEL", "BLOCKER", f"未调整收盘价面板无法读取：{exc}")

    if not adjusted_file.is_file():
        add("ADJUSTED_CLOSE_PANEL_MISSING", "BLOCKER", "未找到调整收盘价面板。")
    else:
        try:
            adjusted_panel = load_price_panel(adjusted_file)
            add("ADJUSTED_CLOSE_PANEL", "PASS", "调整收盘价面板结构、日期和正价格检查通过。")
            overview["adjusted_price_last_date"] = pd.Timestamp(adjusted_panel.index.max()).date().isoformat()
        except Exception as exc:
            add("ADJUSTED_CLOSE_PANEL", "BLOCKER", f"调整收盘价面板无法读取：{exc}")

    if holdings is not None and close_panel is not None:
        held = set(holdings["ticker"].astype(str))
        missing = sorted(held - set(map(str, close_panel.columns)))
        overview["missing_raw_ticker_count"] = len(missing)
        if missing:
            add("RAW_TICKER_COVERAGE", "BLOCKER", f"未调整价格面板缺少持仓标的：{', '.join(missing)}。")
        else:
            add("RAW_TICKER_COVERAGE", "PASS", "未调整价格面板覆盖全部持仓标的。")

        panel_end = pd.Timestamp(close_panel.index.max()).date()
        stale: list[str] = []
        future: list[str] = []
        for ticker in sorted(held - set(missing)):
            observed = pd.to_numeric(close_panel[ticker], errors="coerce").dropna()
            if observed.empty:
                stale.append(ticker)
                continue
            last_date = pd.Timestamp(observed.index.max()).date()
            age = (as_of - last_date).days
            if age < 0:
                future.append(ticker)
            elif age > max_staleness_days:
                stale.append(ticker)
        overview["stale_ticker_count"] = len(stale)
        if future:
            add("RAW_PRICE_FUTURE_DATE", "BLOCKER", f"未调整价格面板存在晚于验收日的数据：{', '.join(future)}。")
        if stale:
            add("RAW_PRICE_STALE", "BLOCKER", f"持仓标的最新未调整价格超过 {max_staleness_days} 个自然日：{', '.join(stale)}。")
        if not stale and not future and not missing:
            add("RAW_PRICE_FRESHNESS", "PASS", "所有持仓标的的未调整收盘价均在允许日期范围内。")

        if panel_end > as_of:
            add("RAW_PANEL_FUTURE_DATE", "BLOCKER", "未调整价格面板最新日期晚于验收日期，请检查本机日期或数据文件。")

    if holdings is not None and adjusted_panel is not None:
        held = set(holdings["ticker"].astype(str))
        missing_adj = sorted(held - set(map(str, adjusted_panel.columns)))
        overview["missing_adjusted_ticker_count"] = len(missing_adj)
        if missing_adj:
            add("ADJUSTED_TICKER_COVERAGE", "BLOCKER", f"调整价格面板缺少持仓标的：{', '.join(missing_adj)}。")
        else:
            add("ADJUSTED_TICKER_COVERAGE", "PASS", "调整价格面板覆盖全部持仓标的。")
            active = adjusted_panel.loc[:, sorted(held)].apply(pd.to_numeric, errors="coerce")
            returns = active.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            common = returns.dropna(how="any")
            common_count = int(len(common))
            overview["common_return_observations"] = common_count
            if common_count < minimum_common_observations:
                add(
                    "COMMON_RETURN_HISTORY",
                    "BLOCKER",
                    f"持仓标的共同收益观测仅 {common_count} 个，低于最低要求 {minimum_common_observations} 个。",
                )
            elif common_count < 252:
                add(
                    "COMMON_RETURN_HISTORY",
                    "WARNING",
                    f"共同收益观测为 {common_count} 个；满足最低计算要求，但不足约一年的 252 个交易日。",
                )
            else:
                add("COMMON_RETURN_HISTORY", "PASS", f"共同收益观测为 {common_count} 个，达到一年左右的基础历史长度。")

            abnormal_count = int((common.abs() > 0.35).sum().sum())
            overview["abnormal_return_observation_count"] = abnormal_count
            if abnormal_count:
                add(
                    "ABNORMAL_DAILY_RETURNS",
                    "WARNING",
                    f"发现 {abnormal_count} 个绝对值超过 35% 的单日收益观测；请核对公司行为和行情异常。",
                )
            else:
                add("ABNORMAL_DAILY_RETURNS", "PASS", "未发现绝对值超过 35% 的单日收益观测。")

        stale_adjusted: list[str] = []
        future_adjusted: list[str] = []
        for ticker in sorted(held - set(missing_adj)):
            observed = pd.to_numeric(adjusted_panel[ticker], errors="coerce").dropna()
            if observed.empty:
                stale_adjusted.append(ticker)
                continue
            last_date = pd.Timestamp(observed.index.max()).date()
            age = (as_of - last_date).days
            if age < 0:
                future_adjusted.append(ticker)
            elif age > max_staleness_days:
                stale_adjusted.append(ticker)
        overview["stale_adjusted_ticker_count"] = len(stale_adjusted)
        if future_adjusted:
            add(
                "ADJUSTED_PRICE_FUTURE_DATE",
                "BLOCKER",
                f"调整价格面板存在晚于验收日的持仓标的数据：{', '.join(future_adjusted)}。",
            )
        if stale_adjusted:
            add(
                "ADJUSTED_PRICE_TICKER_STALE",
                "BLOCKER",
                f"持仓标的最新调整收盘价超过 {max_staleness_days} 个自然日：{', '.join(stale_adjusted)}。",
            )
        elif not future_adjusted:
            add("ADJUSTED_PRICE_TICKER_FRESHNESS", "PASS", "所有持仓标的的调整收盘价均在允许日期范围内。")

        adjusted_end = pd.Timestamp(adjusted_panel.index.max()).date()
        if adjusted_end > as_of:
            add("ADJUSTED_PANEL_FUTURE_DATE", "BLOCKER", "调整价格面板最新日期晚于验收日期。")
        elif (as_of - adjusted_end).days > max_staleness_days:
            add("ADJUSTED_PANEL_STALE", "WARNING", "调整价格面板整体日期已超过允许的新鲜度窗口。")

    if holdings is not None and close_panel is not None:
        held = set(holdings["ticker"].astype(str))
        if held.issubset(set(map(str, close_panel.columns))):
            try:
                weights, latest_valuation = weights_from_price_panel(
                    holdings, close_panel, max_staleness_calendar_days=max_staleness_days
                )
                latest_prices = dict(zip(latest_valuation["ticker"], latest_valuation["price"]))
                positions, _ = summarize_holdings(holdings, reference_prices=latest_prices)
                alerts = risk_guardrail_alerts(positions)
                overview["concentration_alert_count"] = len(alerts)
                add(
                    "HOLDINGS_MARK_TO_MARKET",
                    "PASS",
                    f"持仓市值与权重可由最新未调整收盘价重建（覆盖 {len(weights)} 个标的）；报告不保存金额和股数。",
                )
                if alerts:
                    add(
                        "CONCENTRATION_GUARDRAILS",
                        "WARNING",
                        f"有 {len(alerts)} 项单一持仓、主题或行业集中度提醒，需要结合账户完整资产复核。",
                    )
                else:
                    add("CONCENTRATION_GUARDRAILS", "PASS", "未触发默认单一持仓、主题和行业集中度阈值。")
            except Exception as exc:
                add("HOLDINGS_MARK_TO_MARKET", "BLOCKER", f"无法基于最新未调整收盘价重建持仓权重：{exc}")

        meta_path = Path(str(holdings_file) + ".meta.json")
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                _, snapshot_summary = summarize_holdings(holdings)
                reported_market = meta.get("reported_market_value")
                reported_pnl = meta.get("reported_unrealized_pnl")
                if reported_market is None and reported_pnl is None:
                    overview["broker_reconciliation"] = "未提供券商汇总值"
                    add("BROKER_RECONCILIATION", "WARNING", "元数据文件未提供可用于核对的券商市值或浮动盈亏汇总。")
                else:
                    result = reconcile_snapshot(
                        float(snapshot_summary["market_value"]),
                        float(snapshot_summary["unrealized_pnl"]),
                        reported_market_value=None if reported_market is None else float(reported_market),
                        reported_unrealized_pnl=None if reported_pnl is None else float(reported_pnl),
                        tolerance=float(meta.get("reconciliation_tolerance", 1.0)),
                    )
                    within = bool(result["within_tolerance"])
                    overview["broker_reconciliation"] = "在容差内" if within else "不在容差内"
                    add(
                        "BROKER_RECONCILIATION",
                        "PASS" if within else "BLOCKER",
                        "持仓快照与券商汇总值在容差范围内。" if within else "持仓快照与券商汇总值不一致；请核对股数、报价时点、费用和遗漏项目。",
                    )
            except Exception as exc:
                overview["broker_reconciliation"] = "元数据无法读取"
                add("BROKER_RECONCILIATION", "BLOCKER", f"券商汇总元数据无法核验：{exc}")
        else:
            overview["broker_reconciliation"] = "未提供"
            add("BROKER_RECONCILIATION", "WARNING", "未提供券商汇总值，无法确认 CSV 是否完整反映账户快照。")

    add(
        "ACCOUNT_SCOPE",
        "INFO",
        "当前统计口径仅覆盖导入的股票和 ETF；请另行确认现金、债券、其他账户及负债是否需要纳入。",
    )
    add(
        "DATA_SOURCE_SCOPE",
        "INFO",
        "价格面板属于研究用途的行情数据；异常跳变、公司行为和复权口径仍需人工抽查。",
    )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of.isoformat(),
        "status": _status(checks),
        "checks": checks,
        "overview": overview,
        "privacy": {
            "position_details_included": False,
            "share_quantities_included": False,
            "cost_basis_included": False,
            "market_values_included": False,
            "account_balances_included": False,
            "private_inputs_uploaded": False,
        },
        "thresholds": {
            "max_staleness_calendar_days": max_staleness_days,
            "minimum_common_return_observations": minimum_common_observations,
            "one_year_reference_observations": 252,
            "abnormal_daily_return_absolute_threshold": 0.35,
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_data_acceptance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "real_data_acceptance.md").write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local-only acceptance checks on private holdings and cached stock-price panels."
    )
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS)
    parser.add_argument("--close-prices", type=Path, default=DEFAULT_CLOSE)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of-date", help="Override local validation date (YYYY-MM-DD), primarily for repeatable tests.")
    parser.add_argument("--max-staleness-days", type=int, default=7)
    parser.add_argument("--minimum-common-observations", type=int, default=60)
    args = parser.parse_args()
    try:
        result = run_acceptance(
            args.holdings_csv,
            args.close_prices,
            args.adjusted_prices,
            args.output_dir,
            as_of_date=args.as_of_date,
            max_staleness_days=args.max_staleness_days,
            minimum_common_observations=args.minimum_common_observations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": result["status"],
        "check_counts": {
            level: sum(item["level"] == level for item in result["checks"])
            for level in ("PASS", "WARNING", "BLOCKER", "INFO")
        },
        "report_markdown": str((args.output_dir / "real_data_acceptance.md")),
        "report_json": str((args.output_dir / "real_data_acceptance.json")),
        "private_values_included": False,
    }, ensure_ascii=False, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
