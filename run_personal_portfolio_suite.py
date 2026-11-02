#!/usr/bin/env python3
"""Run the local personal-portfolio workflow in a guarded, auditable order.

Sequence:
1. Refresh/reuse market data and run local real-data acceptance.
2. Stop all downstream work if acceptance is BLOCKED.
3. Generate personal risk and ETF look-through reports.
4. Compare actual holdings with balanced/conservative reference allocations and walk-forward tests.
5. Refresh DCF/forward-PE research for held operating companies (ETFs excluded).
6. Run rule-based add/hold/reduce review only after a successful valuation refresh.

No private holdings are uploaded and no broker orders are placed.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import portfolio_allocation_comparison as allocation
import position_decision_rules as decision_rules
import run_personal_portfolio as data_pipeline
import run_personal_risk_analysis as risk_analysis
import run_valuation as valuation_pipeline
from etf_lookthrough import load_issuer_snapshot

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "personal_suite"
DEFAULT_VALUATION_OUTPUT = ROOT / "outputs" / "valuation"
DEFAULT_DECISION_OUTPUT = ROOT / "outputs" / "position_decisions"
DEFAULT_ETF_SNAPSHOT_DIR = ROOT / "outputs" / "etf_holdings"




def discover_latest_etf_snapshots(
    snapshot_dir: str | Path = DEFAULT_ETF_SNAPSHOT_DIR,
    *,
    fund_tickers: Sequence[str] = ("DRAM", "QQQ"),
    as_of_date: str | date | None = None,
) -> list[Path]:
    """Select the newest valid dated issuer snapshot per held fund.

    Only snapshots dated on or before the analysis date are eligible. Duplicate
    manifests for the same fund and latest date are rejected as ambiguous.
    Invalid manifests are not silently ignored.
    """
    reference_date = date.today() if as_of_date is None else (
        as_of_date if isinstance(as_of_date, date) else date.fromisoformat(str(as_of_date))
    )
    target_funds = {str(item).strip().upper() for item in fund_tickers if str(item).strip()}
    if not target_funds:
        return []
    directory = Path(snapshot_dir)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"ETF snapshot path is not a directory: {directory}")

    candidates: dict[str, list[tuple[date, Path]]] = {}
    for path in sorted(directory.glob("*_lookthrough_*.json")):
        try:
            _, snapshot_as_of, manifest = load_issuer_snapshot(path)
        except ValueError as exc:
            raise ValueError(f"Invalid ETF snapshot {path.name}: {exc}") from exc
        fund = str(manifest.get("fund_ticker", "")).strip().upper()
        if fund not in target_funds:
            continue
        snapshot_date = date.fromisoformat(snapshot_as_of)
        if snapshot_date > reference_date:
            continue
        candidates.setdefault(fund, []).append((snapshot_date, path))

    selected: list[Path] = []
    for fund in sorted(target_funds):
        rows = candidates.get(fund, [])
        if not rows:
            continue
        latest_date = max(item[0] for item in rows)
        latest_paths = [path for snapshot_date, path in rows if snapshot_date == latest_date]
        if len(latest_paths) != 1:
            raise ValueError(
                f"{fund} has multiple ETF snapshots dated {latest_date.isoformat()}; "
                "remove or rename duplicates after reviewing their provenance."
            )
        selected.append(latest_paths[0])
    return selected


def _write_summary(payload: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "personal_portfolio_suite.json"
    markdown_path = target / "personal_portfolio_suite.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    stages = payload["stages"]
    lines = [
        "# 个人组合一键研究报告",
        "",
        f"- 整体状态：**{payload['status']}**",
        f"- 运行时间（UTC）：{payload['generated_at_utc']}",
        f"- 现金口径：{payload['cash_note']}",
        "",
        "## 执行结果",
        "",
    ]
    acceptance = stages["data_acceptance"]
    lines.append(
        f"- 数据验收：{acceptance['status']}；共同日收益观测："
        f"{acceptance.get('common_return_observations', 'N/A')}"
    )
    risk = stages["risk_analysis"]
    if risk["status"] == "SKIPPED":
        lines.append(f"- 风险分析：未执行（{risk.get('reason', '数据验收未通过')}）")
    elif risk["status"] == "FAILED":
        lines.append(f"- 风险分析：失败（{risk.get('error', '未知错误')}）")
    else:
        lines.append(
            f"- 风险分析：已生成；历史质量：{risk.get('history_quality', 'N/A')}；"
            f"ETF 穿透：{risk.get('lookthrough_status', 'N/A')}；"
            f"风险告警：{risk.get('alert_count', 'N/A')} 项"
        )
    compare = stages["allocation_comparison"]
    if compare["status"] == "SKIPPED":
        lines.append(f"- 配置比较：未执行（{compare.get('reason', '数据验收未通过')}）")
    elif compare["status"] == "FAILED":
        lines.append(f"- 配置比较：失败（{compare.get('error', '未知错误')}）")
    else:
        walk_forward = compare.get("walk_forward", {})
        lines.append(
            f"- 配置比较：已生成；共同历史：{compare.get('common_return_observations', 'N/A')} 个；"
            f"Walk-forward：{walk_forward.get('status', 'N/A')}（"
            f"{walk_forward.get('fold_count', 'N/A')} 折）"
        )
    valuation = stages["valuation_research"]
    if valuation["status"] == "SKIPPED":
        lines.append(f"- 基本面估值：未执行（{valuation.get('reason', '数据验收未通过')}）")
    elif valuation["status"] == "FAILED":
        lines.append(f"- 基本面估值：失败（{valuation.get('error', '未知错误')}）")
    else:
        lines.append(
            f"- 基本面估值：已生成；覆盖 {valuation.get('ticker_count', 'N/A')} 家经营性公司；"
            f"数据状态统计：{valuation.get('validation_status_counts', {})}"
        )
    decision = stages["position_decision_review"]
    if decision["status"] == "SKIPPED":
        lines.append(f"- 加减仓规则复核：未执行（{decision.get('reason', '估值流程未完成')}）")
    elif decision["status"] == "FAILED":
        lines.append(f"- 加减仓规则复核：失败（{decision.get('error', '未知错误')}）")
    else:
        lines.append(
            f"- 加减仓规则复核：已生成；规则结果统计：{decision.get('decision_counts', {})}"
        )

    if payload["warnings"]:
        lines += ["", "## 需要留意", ""]
        lines.extend(f"- {item}" for item in payload["warnings"])
    lines += [
        "",
        "## 本次输出位置",
        "",
        f"- 数据验收报告：{acceptance.get('report_markdown', 'N/A')}",
        f"- 风险报告：{risk.get('report_markdown', 'N/A')}",
        f"- 配置比较报告：{compare.get('report_path', 'N/A')}",
        f"- 基本面估值报告：{valuation.get('report_path', 'N/A')}",
        f"- 加减仓规则报告：{decision.get('report_path', 'N/A')}",
        "",
        "## 使用边界",
        "",
        "- 本工具只在本机读取个人持仓和价格文件，不上传持仓股数、成本、账户余额或私有 CSV。",
        "- 数据验收 BLOCKED 时不会继续计算风险与配置；WARNING 会保留告警并继续生成研究结果。",
        "- 未提供现金时，当前持仓比例仅按导入证券计算；不会声称现金余额为零或代表已核实的全部资产。",
        "- 未导入有日期的 DRAM/QQQ ETF 持仓快照时，ETF 穿透会标记缺失或不完整，不会用固定比例代替真实持仓。",
        "- DCF 和远期 PE 是估值交叉检查，不是目标价；财报单位、周期性现金流或报价不合格时必须保留 BLOCKED/低置信度。",
        "- 加仓研究观察需要经过核实的现金余额；现金未提供时不会生成加仓金额。减仓复核也不是自动卖出指令。",
        "- 风险、压力情景和优化结果用于研究复核，不构成买卖指令；程序不会连接券商或下单。",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def run_portfolio_suite(
    *,
    config_path: str | Path = data_pipeline.DEFAULT_CONFIG,
    holdings_path: str | Path = data_pipeline.DEFAULT_HOLDINGS,
    close_prices_path: str | Path = data_pipeline.DEFAULT_CLOSE,
    adjusted_prices_path: str | Path = data_pipeline.DEFAULT_ADJUSTED,
    acceptance_output_dir: str | Path = data_pipeline.DEFAULT_REPORTS,
    risk_output_dir: str | Path = risk_analysis.DEFAULT_OUTPUT,
    allocation_output_dir: str | Path = allocation.DEFAULT_OUTPUT,
    group_caps_path: str | Path = risk_analysis.DEFAULT_GROUP_CAPS,
    subgroup_caps_path: str | Path = risk_analysis.DEFAULT_SUBGROUP_CAPS,
    stress_profile_path: str | Path = risk_analysis.DEFAULT_STRESS_PROFILE,
    allocation_profiles_path: str | Path = allocation.DEFAULT_PROFILES,
    valuation_config_path: str | Path = valuation_pipeline.DEFAULT_CONFIG,
    valuation_cache_dir: str | Path = valuation_pipeline.DEFAULT_CACHE_DIR,
    valuation_output_dir: str | Path = DEFAULT_VALUATION_OUTPUT,
    decision_policy_path: str | Path = decision_rules.DEFAULT_POLICY,
    decision_output_dir: str | Path = DEFAULT_DECISION_OUTPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    valuation_max_cache_age_hours: float = 168.0,
    valuation_max_quote_age_days: int = 7,
    valuation_max_financial_age_days: int = 180,
    refresh: bool = True,
    cash: float | None = None,
    snapshot_paths: Sequence[str | Path] = (),
    etf_snapshot_dir: str | Path = DEFAULT_ETF_SNAPSHOT_DIR,
    as_of_date: str | date | None = None,
) -> dict[str, Any]:
    """Run acceptance, risk, allocation, valuation and position review in order."""
    if cash is not None:
        try:
            cash_value = float(cash)
        except (TypeError, ValueError) as exc:
            raise ValueError("cash must be a finite non-negative USD amount.") from exc
        if not math.isfinite(cash_value) or cash_value < 0:
            raise ValueError("cash must be a finite non-negative USD amount.")
    else:
        cash_value = None

    acceptance = data_pipeline.run_pipeline(
        config_path=config_path,
        holdings_path=holdings_path,
        close_prices_path=close_prices_path,
        adjusted_prices_path=adjusted_prices_path,
        output_dir=acceptance_output_dir,
        refresh=refresh,
        as_of_date=as_of_date,
    )
    acceptance_status = str(acceptance.get("status", "UNKNOWN")).upper()
    if acceptance_status not in {"PASS", "WARNING", "BLOCKED"}:
        raise ValueError(f"Unexpected data-acceptance status: {acceptance_status!r}.")

    universe = data_pipeline.load_universe(config_path)
    held_funds = [
        ticker for ticker in ("DRAM", "QQQ") if ticker in set(universe["holding_tickers"])
    ]
    effective_snapshot_paths = list(snapshot_paths) if snapshot_paths else discover_latest_etf_snapshots(
        etf_snapshot_dir,
        fund_tickers=held_funds,
        as_of_date=as_of_date or acceptance.get("as_of_date"),
    )

    warnings: list[str] = []
    if acceptance_status == "WARNING":
        warnings.append("数据验收存在 WARNING；请先阅读验收报告中的历史长度、行情异常和对账提示。")

    stages: dict[str, dict[str, Any]] = {
        "data_acceptance": {
            "status": acceptance_status,
            "as_of_date": acceptance.get("as_of_date"),
            "position_count": acceptance.get("position_count"),
            "common_return_observations": acceptance.get("common_return_observations"),
            "short_history_tickers": acceptance.get("short_history_tickers", []),
            "report_markdown": acceptance.get("report_markdown"),
            "report_json": acceptance.get("report_json"),
        },
        "risk_analysis": {"status": "SKIPPED"},
        "allocation_comparison": {"status": "SKIPPED"},
        "valuation_research": {"status": "SKIPPED"},
        "position_decision_review": {"status": "SKIPPED"},
    }

    if acceptance_status == "BLOCKED":
        stages["risk_analysis"]["reason"] = "数据验收为 BLOCKED，禁止继续解释风险指标。"
        stages["allocation_comparison"]["reason"] = "数据验收为 BLOCKED，禁止继续生成配置方案。"
        stages["valuation_research"]["reason"] = "数据验收为 BLOCKED，禁止刷新或解释基本面估值。"
        stages["position_decision_review"]["reason"] = "数据验收为 BLOCKED，禁止输出加减仓规则结论。"
        payload: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "cash_source": "not_provided" if cash_value is None else "user_provided",
            "cash_note": (
                "未提供现金余额；如后续分析，仅按已导入证券计算比例。"
                if cash_value is None else "已提供并将在分析中纳入现金余额。"
            ),
            "stages": stages,
            "warnings": warnings + ["数据验收未通过；请修正 BLOCKER 项后重新运行。"],
            "privacy": {
                "private_holdings_uploaded": False,
                "share_quantities_included_in_suite_manifest": False,
                "cost_basis_included_in_suite_manifest": False,
                "account_values_included_in_suite_manifest": False,
                "broker_orders_sent": False,
            },
        }
        _write_summary(payload, output_dir)
        return payload

    risk_result: dict[str, Any] | None = None
    try:
        risk_result = risk_analysis.run_personal_risk_report(
            config_path=config_path,
            holdings_path=holdings_path,
            close_prices_path=close_prices_path,
            adjusted_prices_path=adjusted_prices_path,
            group_caps_path=group_caps_path,
            subgroup_caps_path=subgroup_caps_path,
            stress_profile_path=stress_profile_path,
            output_dir=risk_output_dir,
            cash=cash_value,
            snapshot_paths=effective_snapshot_paths,
            as_of_date=as_of_date,
        )
        risk_summary = risk_result["summary"]
        stages["risk_analysis"] = {
            "status": "PASS",
            "report_markdown": risk_summary.get("report_markdown"),
            "history_quality": risk_summary.get("history_quality"),
            "portfolio_return_observations": risk_summary.get("portfolio_return_observations"),
            "alert_count": risk_summary.get("alert_count"),
            "lookthrough_status": risk_summary.get("lookthrough_status"),
            "lookthrough_funds_missing": risk_summary.get("lookthrough_funds_missing", []),
            "etf_snapshot_files": [str(path) for path in effective_snapshot_paths],
        }
        if risk_summary.get("lookthrough_status") == "partial_or_missing_or_stale":
            warnings.append("DRAM/QQQ 的一项或多项有日期 ETF 持仓快照缺失、过期或不完整；发行人穿透敞口尚不能视为完整。")
        if risk_summary.get("history_quality") == "under_one_trading_year":
            warnings.append("组合共同历史不足约一年；波动率、相关性、VaR/CVaR 和风险贡献仅作初步估计。")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        stages["risk_analysis"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    try:
        allocation_result = allocation.run_allocation_comparison(
            config_path=config_path,
            holdings_path=holdings_path,
            close_prices_path=close_prices_path,
            adjusted_prices_path=adjusted_prices_path,
            profiles_path=allocation_profiles_path,
            output_dir=allocation_output_dir,
            cash=cash_value,
            as_of_date=as_of_date,
        )
        allocation_summary = allocation_result["summary"]
        walk_forward = allocation_summary.get("walk_forward", {})
        stages["allocation_comparison"] = {
            "status": "PASS",
            "report_path": allocation_summary.get("report_path"),
            "common_return_observations": allocation_summary.get("common_return_observations"),
            "walk_forward": walk_forward,
        }
        if walk_forward.get("status") != "multiple_oos_folds":
            warnings.append(
                "Walk-forward 样本外证据不足或不可用；不要把最小方差权重当成稳定最优配置。"
            )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        stages["allocation_comparison"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    valuation_rows: list[dict[str, Any]] | None = None
    try:
        universe = data_pipeline.load_universe(config_path)
        valuation_config = valuation_pipeline.load_config(valuation_config_path)
        operating_companies = [
            ticker for ticker in universe["holding_tickers"]
            if universe["symbols"][ticker].get("instrument_type") != "etf"
        ]
        missing_valuation_config = sorted(set(operating_companies) - set(valuation_config["tickers"]))
        if missing_valuation_config:
            raise ValueError(
                "Valuation assumptions are missing for held operating companies: "
                f"{missing_valuation_config}."
            )
        valuation_rows = valuation_pipeline.run(
            operating_companies,
            config_path=Path(valuation_config_path),
            cache_dir=Path(valuation_cache_dir),
            output_dir=Path(valuation_output_dir),
            refresh=refresh,
            allow_stale_cache=False,
            max_cache_age_hours=valuation_max_cache_age_hours,
            max_quote_age_days=valuation_max_quote_age_days,
            max_financial_age_days=valuation_max_financial_age_days,
        )
        if not valuation_rows:
            raise RuntimeError("Valuation pipeline returned no rows.")
        validation_status_counts: dict[str, int] = {}
        blocked_valuation_tickers: list[str] = []
        for row in valuation_rows:
            status = str(row.get("data_validation_status", "BLOCKED")).upper()
            validation_status_counts[status] = validation_status_counts.get(status, 0) + 1
            if status != "PASS":
                blocked_valuation_tickers.append(str(row.get("ticker", "UNKNOWN")))
        valuation_status = "WARNING" if blocked_valuation_tickers else "PASS"
        stages["valuation_research"] = {
            "status": valuation_status,
            "ticker_count": len(valuation_rows),
            "validation_status_counts": validation_status_counts,
            "non_pass_tickers": blocked_valuation_tickers,
            "report_path": str(Path(valuation_output_dir) / "valuation_report.md"),
            "summary_path": str(Path(valuation_output_dir) / "valuation_summary.csv"),
        }
        if blocked_valuation_tickers:
            warnings.append(
                "以下经营性公司估值数据未达到 PASS，规则引擎会将相关标的保留为观察/数据阻断，不会推断可加仓："
                + ", ".join(blocked_valuation_tickers)
            )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        stages["valuation_research"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    if stages["valuation_research"]["status"] in {"PASS", "WARNING"}:
        try:
            _, decision_summary = decision_rules.run_position_decision_review(
                config_path=config_path,
                holdings_path=holdings_path,
                prices_path=close_prices_path,
                valuation_path=Path(valuation_output_dir) / "valuation_summary.csv",
                policy_path=decision_policy_path,
                output_dir=decision_output_dir,
                cash=cash_value,
                as_of_date=as_of_date,
            )
            decision_counts = dict(decision_summary.get("decision_counts", {}))
            stages["position_decision_review"] = {
                "status": "COMPLETED",
                "position_count": decision_summary.get("position_count"),
                "decision_counts": decision_counts,
                "cash_source": decision_summary.get("cash_source"),
                "report_path": decision_summary.get("report_path"),
            }
            if cash_value is None:
                warnings.append("现金余额未提供；加仓规则会标记现金未核实，不会计算可用加仓金额。")
            if int(decision_counts.get("REDUCE_REVIEW", 0)) > 0:
                warnings.append("规则复核发现达到减仓/再平衡复核门槛的持仓；请阅读逐项依据，不代表要求立即卖出。")
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            stages["position_decision_review"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    else:
        stages["position_decision_review"] = {
            "status": "SKIPPED",
            "reason": "估值刷新失败；为防止误用旧估值文件，本次跳过加减仓规则复核。",
        }

    failed_count = sum(
        stages[key]["status"] == "FAILED"
        for key in ("risk_analysis", "allocation_comparison", "valuation_research", "position_decision_review")
    )
    completed_count = sum(
        stages[key]["status"] in {"PASS", "WARNING", "COMPLETED"}
        for key in ("risk_analysis", "allocation_comparison", "valuation_research", "position_decision_review")
    )
    if failed_count and completed_count:
        overall_status = "PARTIAL"
    elif failed_count:
        overall_status = "FAILED"
    elif warnings:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "cash_source": "not_provided" if cash_value is None else "user_provided",
        "cash_note": (
            "未提供现金余额；当前持仓比例仅按导入证券计算，不代表已核实的全部资产。"
            if cash_value is None else "已提供现金余额并纳入组合净资产分母。"
        ),
        "stages": stages,
        "warnings": warnings,
        "privacy": {
            "private_holdings_uploaded": False,
            "share_quantities_included_in_suite_manifest": False,
            "cost_basis_included_in_suite_manifest": False,
            "account_values_included_in_suite_manifest": False,
            "broker_orders_sent": False,
        },
    }
    _write_summary(payload, output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local personal portfolio acceptance, risk analysis and allocation comparison."
    )
    parser.add_argument("--config", type=Path, default=data_pipeline.DEFAULT_CONFIG)
    parser.add_argument("--holdings-csv", type=Path, default=data_pipeline.DEFAULT_HOLDINGS)
    parser.add_argument("--close-prices", type=Path, default=data_pipeline.DEFAULT_CLOSE)
    parser.add_argument("--adjusted-prices", type=Path, default=data_pipeline.DEFAULT_ADJUSTED)
    parser.add_argument("--acceptance-output-dir", type=Path, default=data_pipeline.DEFAULT_REPORTS)
    parser.add_argument("--risk-output-dir", type=Path, default=risk_analysis.DEFAULT_OUTPUT)
    parser.add_argument("--allocation-output-dir", type=Path, default=allocation.DEFAULT_OUTPUT)
    parser.add_argument("--group-caps", type=Path, default=risk_analysis.DEFAULT_GROUP_CAPS)
    parser.add_argument("--subgroup-caps", type=Path, default=risk_analysis.DEFAULT_SUBGROUP_CAPS)
    parser.add_argument("--stress-profile", type=Path, default=risk_analysis.DEFAULT_STRESS_PROFILE)
    parser.add_argument("--allocation-profiles", type=Path, default=allocation.DEFAULT_PROFILES)
    parser.add_argument("--valuation-config", type=Path, default=valuation_pipeline.DEFAULT_CONFIG)
    parser.add_argument("--valuation-cache-dir", type=Path, default=valuation_pipeline.DEFAULT_CACHE_DIR)
    parser.add_argument("--valuation-output-dir", type=Path, default=DEFAULT_VALUATION_OUTPUT)
    parser.add_argument("--decision-policy", type=Path, default=decision_rules.DEFAULT_POLICY)
    parser.add_argument("--decision-output-dir", type=Path, default=DEFAULT_DECISION_OUTPUT)
    parser.add_argument("--valuation-max-cache-age-hours", type=float, default=168.0)
    parser.add_argument("--valuation-max-quote-age-days", type=int, default=7)
    parser.add_argument("--valuation-max-financial-age-days", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cash", type=float, default=None, help="Optional verified cash balance in USD.")
    parser.add_argument(
        "--lookthrough-snapshot",
        type=Path,
        action="append",
        default=[],
        help="Explicit dated DRAM/QQQ issuer snapshot JSON; overrides auto-discovery when supplied.",
    )
    parser.add_argument(
        "--etf-snapshot-dir",
        type=Path,
        default=DEFAULT_ETF_SNAPSHOT_DIR,
        help="Directory containing dated *_lookthrough_*.json manifests; newest eligible snapshot per held fund is selected.",
    )
    parser.add_argument("--as-of-date", help="Analysis date YYYY-MM-DD.")
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh", dest="refresh", action="store_true", help="Force fresh market data.")
    refresh.add_argument("--use-cache", dest="refresh", action="store_false", help="Reuse only valid fresh cache.")
    parser.set_defaults(refresh=True)
    args = parser.parse_args()
    try:
        result = run_portfolio_suite(
            config_path=args.config,
            holdings_path=args.holdings_csv,
            close_prices_path=args.close_prices,
            adjusted_prices_path=args.adjusted_prices,
            acceptance_output_dir=args.acceptance_output_dir,
            risk_output_dir=args.risk_output_dir,
            allocation_output_dir=args.allocation_output_dir,
            group_caps_path=args.group_caps,
            subgroup_caps_path=args.subgroup_caps,
            stress_profile_path=args.stress_profile,
            allocation_profiles_path=args.allocation_profiles,
            valuation_config_path=args.valuation_config,
            valuation_cache_dir=args.valuation_cache_dir,
            valuation_output_dir=args.valuation_output_dir,
            decision_policy_path=args.decision_policy,
            decision_output_dir=args.decision_output_dir,
            valuation_max_cache_age_hours=args.valuation_max_cache_age_hours,
            valuation_max_quote_age_days=args.valuation_max_quote_age_days,
            valuation_max_financial_age_days=args.valuation_max_financial_age_days,
            output_dir=args.output_dir,
            refresh=args.refresh,
            cash=args.cash,
            snapshot_paths=args.lookthrough_snapshot,
            etf_snapshot_dir=args.etf_snapshot_dir,
            as_of_date=args.as_of_date,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] in {"BLOCKED", "FAILED", "PARTIAL"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
