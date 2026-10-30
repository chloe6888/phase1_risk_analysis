#!/usr/bin/env python3
"""One-command orchestration for a personal quantitative research run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_risk import DEFAULT_OUTPUT_DIR, load_price_panel, parse_scenario
from strategy_validation import parse_int_list
from backtest_engine import load_risk_free_yield, parse_rebalance_freqs
from holdings_manager import load_holdings, summarize_holdings, weights_from_price_panel, risk_guardrail_alerts, reconcile_snapshot
from memory_allocation import analyze_memory_positions

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def fmt_pct(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def fmt_num(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def metrics_table(metrics: dict[str, Any]) -> list[str]:
    specs = [
        ("total_return", "累计收益率", True),
        ("annualized_return", "年化收益率", True),
        ("annualized_volatility", "年化波动率", True),
        ("sharpe_ratio", "夏普比率", False),
        ("sortino_ratio", "索提诺比率", False),
        ("max_drawdown", "最大回撤", True),
    ]
    rows = []
    for key, label, is_percent in specs:
        if key in metrics:
            value = fmt_pct(metrics[key]) if is_percent else fmt_num(metrics[key])
            rows.append(f"| {label} | {value} |")
    return rows


def _localized_holdings_alert(alert: dict[str, Any]) -> str:
    """Render concentration alerts in Chinese from their structured fields."""
    kind = str(alert.get("kind", ""))
    name = str(alert.get("name", "未知标的"))
    weight = alert.get("weight_pct")
    threshold = alert.get("threshold_pct")
    if kind == "single_position_concentration" and weight is not None and threshold is not None:
        return f"{name} 占导入股票/ETF持仓 {float(weight):.1f}%，超过 {float(threshold):.1f}% 的单一持仓复核阈值。"
    if kind in {"risk_group_concentration", "subgroup_concentration"} and weight is not None and threshold is not None:
        group_label = "风险主题" if kind == "risk_group_concentration" else "行业分类"
        return f"{group_label}“{name}”占导入股票/ETF持仓 {float(weight):.1f}%，超过 {float(threshold):.1f}% 的复核阈值。"
    return str(alert.get("message", ""))


def render_report(
    *,
    generated_at: str,
    weights: str,
    benchmark: str,
    data: dict[str, Any] | None,
    portfolio: dict[str, Any] | None,
    backtest: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    walk_forward: dict[str, Any] | None,
    holdings_summary: dict[str, Any] | None = None,
    holdings_alerts: list[dict[str, Any]] | None = None,
    holdings_reconciliation: dict[str, Any] | None = None,
    memory_summary: dict[str, Any] | None = None,
    memory_decisions: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# 个人投资研究报告",
        "",
        f"- 生成时间（UTC）： {generated_at}",
        f"- 对比基准： {benchmark}",
        f"- 组合权重： {weights}",
        "- 报告用途：投资研究与风险评估，不构成买卖指令。",
        "",
    ]
    if data:
        lines += [
            "## 行情数据覆盖", "",
            f"- 数据来源： {data.get('source', 'N/A')}",
            f"- 请求数据区间： {data.get('requested_start', 'N/A')} to {data.get('requested_end_exclusive', 'N/A')}（结束日期不包含在内）",
            f"- 各标的最后数据日期： {json.dumps(data.get('data_last_dates', {}), ensure_ascii=False)}", "",
        ]
    if holdings_summary:
        lines += [
            "## 导入持仓与集中度检查", "",
            f"- 持仓数量： {holdings_summary.get('position_count', 'N/A')}",
            f"- 最近收盘价估算市值： {fmt_num(holdings_summary.get('market_value'), 2)} {holdings_summary.get('currency', '')}",
            f"- 按持仓数量重建的成本： {fmt_num(holdings_summary.get('cost_basis'), 2)} {holdings_summary.get('currency', '')}",
            f"- 按最近收盘价估算的浮动盈亏： {fmt_num(holdings_summary.get('unrealized_pnl'), 2)} {holdings_summary.get('currency', '')} ({fmt_pct(float(holdings_summary.get('unrealized_return_pct', 0.0)) / 100.0)})",
            f"- 最大单一持仓： {holdings_summary.get('largest_position_ticker', 'N/A')} ({fmt_pct(float(holdings_summary.get('largest_position_weight_pct', 0.0)) / 100.0)})",
            f"- 最大风险主题： {holdings_summary.get('largest_risk_group', 'N/A')} ({fmt_pct(float(holdings_summary.get('largest_risk_group_weight_pct', 0.0)) / 100.0)})", "",
        ]
        theme_exposures = holdings_summary.get("risk_group_exposures_pct", {})
        subgroup_exposures = holdings_summary.get("subgroup_exposures_pct", {})
        if theme_exposures:
            lines.append("- 风险主题敞口：" + "; ".join(f"{name}: {float(weight):.1f}%" for name, weight in theme_exposures.items()))
        if subgroup_exposures:
            lines.append("- 行业分类敞口：" + "; ".join(f"{name}: {float(weight):.1f}%" for name, weight in subgroup_exposures.items()))
        lines.append("")
        if holdings_reconciliation:
            lines.extend([
                "### 持仓快照核对", "",
                f"- 估算市值与券商汇总差额： {fmt_num(holdings_reconciliation.get('market_value_delta'), 2)}",
                f"- 浮动盈亏估算与券商汇总差额： {fmt_num(holdings_reconciliation.get('unrealized_pnl_delta'), 2)}",
                f"- 券商截图记录的当日盈亏： {fmt_num(holdings_reconciliation.get('reported_today_pnl'), 2)}",
                f"- 是否在设定误差范围内对齐： {'是' if holdings_reconciliation.get('within_tolerance') else '否——请核对持股数量、报价时间、遗漏持仓和费用'}", "",
            ])
        if holdings_alerts:
            lines.extend(["### 集中度风险提醒", ""])
            for alert in holdings_alerts:
                level = {"warning": "提醒", "critical": "严重"}.get(str(alert.get("level", "info")), "提示")
                lines.append(f"- **{level}** {_localized_holdings_alert(alert)}")
            lines.append("")
        else:
            lines.extend(["- 导入持仓未触发单一持仓或分组集中度阈值。", ""])
        lines.extend([
            "持仓比例仅按已导入股票和 ETF 计算；除非单独导入，否则不包含现金、债券、其他账户、外币负债和融资负债。",
            "最近收盘价仅用于历史估值参考，不代表实时可成交报价。采取行动前，请核实每个标的的持股数量和报价日期。", "",
        ])
    if memory_summary:
        denominator_warning = str(memory_summary.get("denominator_warning", "") or "")
        if denominator_warning.startswith("Percentages are relative to the full net portfolio equity"):
            denominator_warning = "本页占比以所提供的账户总净资产为分母；请确认该金额已包含现金、其他资产及负债。"
        elif denominator_warning.startswith("Cash, bonds, other accounts and financing are not included"):
            denominator_warning = "当前占比仅按导入的股票和 ETF 持仓计算，尚未包含现金、债券、其他账户及融资负债。"
        elif denominator_warning.startswith("Imported listed positions only"):
            denominator_warning = "当前占比仅按导入的股票和 ETF 持仓计算，不包含其他未导入资产。"
        memory_action = str(memory_summary.get("overall_action", "N/A"))
        memory_current = float(memory_summary.get("memory_sleeve_weight_pct", 0.0))
        memory_limit = float(memory_summary.get("max_memory_weight_pct", 0.0))
        if memory_action == "REDUCE_MEMORY_SLEEVE":
            memory_reason = f"当前存储主题占参考组合 {memory_current:.1f}%，高于 {memory_limit:.1f}% 的风险预算上限；应复核主题仓位和发行人穿透敞口。"
        elif memory_action == "WITHIN_MEMORY_LIMIT":
            memory_reason = f"当前存储主题占参考组合 {memory_current:.1f}%，未超过 {memory_limit:.1f}% 的风险预算上限；仍需持续跟踪 ETF 穿透持仓。"
        else:
            memory_reason = "规则判断原因未提供，请查看原始风险诊断明细。"
        lines += [
            "## MU / SKHY / DRAM 存储主题分析", "",
            f"- 当前存储主题占比： {fmt_pct(memory_current / 100.0)}",
            f"- 风险预算上限： {fmt_pct(memory_limit / 100.0)}",
            f"- 规则检查结果： { {'REDUCE_MEMORY_SLEEVE': '存储主题仓位超限', 'WITHIN_MEMORY_LIMIT': '存储主题仓位在限制内'}.get(memory_action, '需人工复核')}",
            f"- 判断依据： {memory_reason}",
            f"- DRAM 穿透持仓假设日期： {memory_summary.get('lookthrough_asof', 'N/A')}",
            f"- 发行人穿透敞口：" + "; ".join(
                f"{ticker}: {float(weight):.1f}%" for ticker, weight in memory_summary.get("lookthrough_issuer_weights_pct", {}).items()
            ),
            "",
            "本模块用于检查风险预算，不是价格预测或交易指令。默认平衡方案以参考组合计，目标权重为 MU 8%、SKHY 6%、DRAM 5%。执行前应更新 DRAM 穿透持仓并复核公司基本面。",
            "",
        ]
        if memory_decisions is not None and not memory_decisions.empty:
            balanced = memory_decisions[memory_decisions["scenario"] == "balanced"]
            if not balanced.empty:
                lines += [
                    "| 平衡方案 | 当前股数 | 目标股数 | 股数变化 | 建议动作 |",
                    "|---|---:|---:|---:|---|",
                ]
                for row in balanced.itertuples(index=False):
                    current = float(row.current_shares)
                    target = float(row.target_shares)
                    change = float(row.share_delta)
                    action_label = {"REDUCE_TO_TARGET":"按目标仓位减仓复核","HOLD_NO_AUTO_ADD":"暂不自动加仓","HOLD":"维持当前股数","NO_ADD_MEMORY_LIMIT":"存储主题仓位超限，暂不加仓","WAIT_FOR_MEMORY_HEADROOM":"等待仓位回到安全缓冲区","NO_ADD_ISSUER_LOOKTHROUGH_LIMIT":"单一发行人穿透敞口超限","WAIT_FOR_TREND_CONFIRMATION":"等待趋势确认","POSITIVE_TREND_BUT_SIZE_LIMITS_STILL_APPLY":"趋势转正，但仍受仓位限制","STAGED_ADD_REVIEW_ONLY":"满足条件后再评估加仓"}.get(str(row.action), str(row.action))
                    lines.append(
                        f"| {row.ticker} | {current:g} | {target:g} | {change:+g} | {action_label} |"
                    )
                lines.append("")
        lines += [
            denominator_warning,
            "近期价格动量和波动率仅是描述性筛选，不会覆盖已触发的仓位上限，也不能替代基本面估值。",
            "",
        ]
    if portfolio:
        lines += [
            "## 组合风险分析", "",
            f"- 统计区间： {portfolio.get('start_date', 'N/A')} 至 {portfolio.get('end_date', 'N/A')}",
            f"- 有效收益观测： {portfolio.get('observations', 'N/A')}",
            f"- 无风险收益率来源： { {'constant annual rate': '固定年化无风险收益率', 'dated annualized yield series': '按日期变化的年化收益率序列'}.get(str(portfolio.get('risk_free_rate_source', 'constant annual rate')), portfolio.get('risk_free_rate_source', '暂无记录'))}", "",
            "| 指标 | 数值 |", "|---|---:|", *metrics_table(portfolio),
            f"| 历史单期 VaR | {fmt_pct(portfolio.get('historical_var_one_period'))} |",
            f"| 历史单期 CVaR | {fmt_pct(portfolio.get('historical_cvar_one_period'))} |", "",
            "VaR/CVaR 是基于历史单期收益估算的尾部风险指标，不是最大亏损保证。", "",
            *(["风险提醒：共同收益观测少于 252 个交易日，年化风险指标基于偏短的历史样本，可能对个别行情阶段较敏感。"] if int(portfolio.get("observations", 0)) < 252 else []),
        ]
    if backtest:
        strategy, base = backtest.get("strategy", {}), backtest.get("benchmark", {})
        lines += [
            "## 基准动量策略历史回测（仅供参考）", "",
            f"- 统计区间： {backtest.get('start_date', 'N/A')} 至 {backtest.get('end_date', 'N/A')}",
            f"- 有效收益观测： {backtest.get('observations', 'N/A')}",
            f"- 成本假设： {backtest.get('transaction_cost_bps', 'N/A')} 个基点交易成本，滑点 {backtest.get('slippage_bps', 'N/A')} 个基点（按成交金额估算）", "",
            "| 指标 | 策略 | 对比基准 |", "|---|---:|---:|",
        ]
        for key, label, percentage in [
            ("total_return", "累计收益率", True), ("annualized_return", "年化收益率", True),
            ("annualized_volatility", "年化波动率", True), ("sharpe_ratio", "夏普比率", False),
            ("sortino_ratio", "索提诺比率", False), ("max_drawdown", "最大回撤", True),
        ]:
            left = fmt_pct(strategy.get(key)) if percentage else fmt_num(strategy.get(key))
            right = fmt_pct(base.get(key)) if percentage else fmt_num(base.get(key))
            lines.append(f"| {label} | {left} | {right} |")
        lines += [
            "", f"- 相对基准累计收益差： {fmt_pct(backtest.get('excess_total_return'))}",
            f"- 年化换手率估算（考虑权重漂移）： {fmt_num(backtest.get('annualized_turnover'), 2)}",
            f"- 实际调仓次数： {backtest.get('actual_rebalance_count', 'N/A')}",
            f"- 期末清仓成本： {fmt_pct(backtest.get('terminal_liquidation_cost_return'))}", "",
        ]
        comparison = backtest.get("legacy_comparison", {})
        if comparison:
            lines += [
                "### 考虑权重漂移的新旧核算口径对照", "",
                "| 指标 | 考虑权重漂移的新模型 | 旧目标权重模型 |",
                "|---|---:|---:|",
                f"| 累计收益率 | {fmt_pct(comparison.get('new_total_return'))} | {fmt_pct(comparison.get('legacy_total_return'))} |",
                f"| 最大回撤 | {fmt_pct(comparison.get('new_max_drawdown'))} | {fmt_pct(comparison.get('legacy_max_drawdown'))} |",
                f"| 新模型减旧模型的累计收益差 | {fmt_pct(comparison.get('total_return_delta_new_minus_legacy'))} | — |",
                "",
                "该对照使用相同信号和调仓日期，仅用于隔离核算方法变化，并非两套分别优化策略的比较。",
                "",
            ]
        lines += [
            "考虑权重漂移的模型仍是成交近似，假设成本按比例减少组合资产；它不模拟真实成交股数、市场冲击、税费、停牌/涨跌停或期末清仓成交细节。", "",
        ]
    if validation:
        lines += [
            "## 参数敏感性检验", "",
            f"- 请求测试的参数组合数： {validation.get('requested_combinations', 'N/A')}",
            f"- 成功完成的参数组合数： {validation.get('successful_combinations', 'N/A')}",
            f"- 因数据不足等原因跳过的组合数： {validation.get('skipped_combinations', 'N/A')}", "",
            "不要仅凭用于调参的同一段测试期结果选择策略；反复查看并调整测试期，会破坏样本外检验的独立性。", "",
        ]
    if walk_forward:
        lines += [
            "## 滚动样本外验证", "",
            f"- 训练窗口： {walk_forward.get('train_window_observations', 'N/A')} 个收益观测",
            f"- 检验窗口： {walk_forward.get('test_window_observations', 'N/A')} 个收益观测",
            f"- 成功验证轮次： {walk_forward.get('folds_successful', 'N/A')} / {walk_forward.get('folds_total', 'N/A')}",
            f"- 样本外有效观测： {walk_forward.get('oos_observations', 'N/A')}", "",
        ]
        if "out_of_sample_strategy" in walk_forward:
            strategy = walk_forward["out_of_sample_strategy"]
            base = walk_forward["out_of_sample_benchmark"]
            lines += ["| 指标 | 滚动样本外策略 | 对比基准 |", "|---|---:|---:|"]
            for key, label, percentage in [
                ("total_return", "累计收益率", True), ("annualized_return", "年化收益率", True),
                ("annualized_volatility", "年化波动率", True), ("sharpe_ratio", "夏普比率", False),
                ("max_drawdown", "最大回撤", True),
            ]:
                left = fmt_pct(strategy.get(key)) if percentage else fmt_num(strategy.get(key))
                right = fmt_pct(base.get(key)) if percentage else fmt_num(base.get(key))
                lines.append(f"| {label} | {left} | {right} |")
            lines += ["", f"- 样本外相对基准累计收益差： {fmt_pct(walk_forward.get('out_of_sample_excess_total_return'))}", ""]
        lines += [
            "滚动样本外验证可以降低对单次切分的依赖，但不能消除多重试验偏差、幸存者偏差或市场状态变化的影响。建议额外保留一段最终未触碰的留出数据。", "",
        ]
    lines += [
        "## 研究结果文件", "",
        "outputs 目录还包含风险贡献、回撤、滚动风险、每日收益、应用权重、参数网格、年度表现和 walk-forward 折次等 CSV/JSON 明细文件。", "",
        "## 方法局限与使用提醒", "",
        "本报告总结历史观测和研究假设，不构成预测、不执行交易，也不能证明策略具有正的未来期望收益。", "",
    ]
    return "\n".join(lines)


def write_report(output_dir: Path, *, weights: str, benchmark: str) -> Path:
    data = read_json(output_dir / "analysis_metadata.json")
    portfolio = read_json(output_dir / "portfolio_risk_summary.json")
    backtest = read_json(output_dir / "backtest_summary.json")
    validation = read_json(output_dir / "strategy_validation_metadata.json")
    walk_forward = read_json(output_dir / "walk_forward_summary.json")
    memory_summary = read_json(output_dir / "memory_allocation_summary.json")
    memory_decisions_path = output_dir / "memory_allocation_scenarios.csv"
    memory_decisions = pd.read_csv(memory_decisions_path) if memory_decisions_path.exists() else None
    holdings_summary = read_json(output_dir / "holdings_summary.json")
    alerts_payload = read_json(output_dir / "holdings_risk_alerts.json")
    reconciliation = read_json(output_dir / "holdings_reconciliation.json")
    report = render_report(
        generated_at=datetime.now(timezone.utc).isoformat(), weights=weights, benchmark=benchmark,
        data=data, portfolio=portfolio, backtest=backtest, validation=validation, walk_forward=walk_forward,
        holdings_summary=holdings_summary,
        holdings_alerts=(alerts_payload or {}).get('alerts', []),
        holdings_reconciliation=reconciliation,
        memory_summary=memory_summary,
        memory_decisions=memory_decisions,
    )
    path = output_dir / "research_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_step(title: str, command: list[str]) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full offline/online personal quant research pipeline.")
    parser.add_argument("--weights", default=None, help="Target portfolio weights such as GOOGL=0.6,SPY=0.4")
    parser.add_argument("--holdings-csv", type=Path, default=None, help="Private holdings CSV; if given, weights are calculated from latest validated prices")
    parser.add_argument("--test-start", required=True, help="Inclusive single-split out-of-sample start date")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh market data")
    parser.add_argument("--allow-stale-cache", action="store_true", help="Allow stale cache only if refreshing fails")
    parser.add_argument("--max-cache-age-hours", type=float, default=24.0)
    parser.add_argument("--lookbacks", type=parse_int_list, default=[63, 126, 252])
    parser.add_argument("--skip-recents", type=parse_int_list, default=[5, 21])
    parser.add_argument("--top-ns", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--backtest-lookback", type=int, default=126)
    parser.add_argument("--backtest-skip-recent", type=int, default=21)
    parser.add_argument("--backtest-top-n", type=int, default=2)
    parser.add_argument("--backtest-rebalance-freq", choices=["daily", "weekly", "monthly"], default="daily")
    parser.add_argument("--rebalance-freqs", type=parse_rebalance_freqs, default=["daily", "weekly", "monthly"])
    parser.add_argument("--risk-free-yield-csv", type=Path, help="CSV with Date,annual_yield (decimal annualized yield)")
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--selection-metric", choices=["sharpe_ratio", "annualized_return", "sortino_ratio"], default="sharpe_ratio")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--portfolio-value", type=float, help="Optional portfolio value for stress-case dollar impact")
    parser.add_argument("--stress-scenario", action="append", type=parse_scenario, default=[],
                        help="Repeatable NAME:TICKER=return[,TICKER=return...] stress case")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward when history is too short")
    args = parser.parse_args()
    if bool(args.weights) == bool(args.holdings_csv):
        parser.error("Provide exactly one of --weights or --holdings-csv.")
    if args.holdings_csv is not None and not args.holdings_csv.is_file():
        parser.error(f"Holdings CSV does not exist: {args.holdings_csv}")
    if args.risk_free_yield_csv is not None and not args.risk_free_yield_csv.is_file():
        parser.error(f"Risk-free yield CSV does not exist: {args.risk_free_yield_csv}")
    if args.max_cache_age_hours <= 0:
        parser.error("--max-cache-age-hours must be positive.")
    if args.transaction_cost_bps < 0 or args.slippage_bps < 0:
        parser.error("Transaction costs and slippage must be non-negative.")
    if min(args.backtest_lookback, args.backtest_skip_recent, args.backtest_top_n) <= 0:
        parser.error("Backtest lookback, skip-recent, and top-N must be positive.")
    if args.train_window < 2 or args.test_window < 2:
        parser.error("Train and test windows must be at least two observations.")

    py = sys.executable
    risk_free_yield = load_risk_free_yield(args.risk_free_yield_csv) if args.risk_free_yield_csv else None
    imported_holdings = load_holdings(args.holdings_csv) if args.holdings_csv else None
    data_cmd = [py, str(ROOT / "phase1_risk_analysis.py"), "--max-cache-age-hours", str(args.max_cache_age_hours)]
    if imported_holdings is not None:
        required = sorted(set(["SPY", "SOXX", "NVDA"] + imported_holdings["ticker"].tolist()))
        data_cmd += ["--tickers", ",".join(required)]
    if args.refresh:
        data_cmd.append("--refresh")
    if args.allow_stale_cache:
        data_cmd.append("--allow-stale-cache")
    run_step("1/5 Data download, cache, and validation", data_cmd)

    weights_text = args.weights
    user_portfolio_value = args.portfolio_value
    holdings_value = None
    if imported_holdings is not None:
        price_panel = load_price_panel(ROOT / "data" / "raw" / "close_price_panel.csv")
        calculated_weights, latest_valuation = weights_from_price_panel(imported_holdings, price_panel)
        weights_text = ",".join(f"{ticker}={weight:.12f}" for ticker, weight in calculated_weights.items())
        latest_prices = dict(zip(latest_valuation["ticker"], latest_valuation["price"]))
        valued_positions, current_holdings_summary = summarize_holdings(imported_holdings, reference_prices=latest_prices)
        snapshot_positions, snapshot_summary = summarize_holdings(imported_holdings)
        valued_positions.to_csv(OUTPUT_DIR / "holdings_valuation.csv", index=False, float_format="%.10f")
        latest_valuation.to_csv(OUTPUT_DIR / "holdings_latest_price_weights.csv", index=False, float_format="%.10f")
        (OUTPUT_DIR / "holdings_summary.json").write_text(
            json.dumps(current_holdings_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        adjusted_price_panel = load_price_panel(ROOT / "data" / "raw" / "adjusted_close_panel.csv")
        memory_summary, memory_positions, memory_decisions = analyze_memory_positions(
            imported_holdings,
            price_panel,
            adjusted_close=adjusted_price_panel,
            portfolio_value=user_portfolio_value,
        )
        memory_positions.to_csv(OUTPUT_DIR / "memory_position_diagnostics.csv", float_format="%.8f")
        memory_decisions.to_csv(OUTPUT_DIR / "memory_allocation_scenarios.csv", index=False, float_format="%.8f")
        (OUTPUT_DIR / "memory_allocation_summary.json").write_text(
            json.dumps(memory_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        alerts = risk_guardrail_alerts(valued_positions)
        (OUTPUT_DIR / "holdings_risk_alerts.json").write_text(
            json.dumps({"alerts": alerts}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        snapshot_meta_path = Path(str(args.holdings_csv) + ".meta.json")
        snapshot_meta = read_json(snapshot_meta_path) if snapshot_meta_path.exists() else {}
        reconciliation = reconcile_snapshot(
            float(snapshot_summary["market_value"]), float(snapshot_summary["unrealized_pnl"]),
            reported_market_value=snapshot_meta.get("reported_market_value"),
            reported_unrealized_pnl=snapshot_meta.get("reported_unrealized_pnl"),
            tolerance=float(snapshot_meta.get("reconciliation_tolerance", 1.0)),
        )
        reconciliation["reported_today_pnl"] = snapshot_meta.get("reported_today_pnl")
        (OUTPUT_DIR / "holdings_reconciliation.json").write_text(
            json.dumps(reconciliation, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        holdings_value = float(current_holdings_summary["market_value"])
        if args.portfolio_value is None:
            args.portfolio_value = holdings_value

    risk_cmd = [py, str(ROOT / "portfolio_risk.py"), "--weights", weights_text, "--risk-free-rate", str(args.risk_free_rate)]
    if args.portfolio_value is not None:
        risk_cmd += ["--portfolio-value", str(args.portfolio_value)]
    if args.risk_free_yield_csv:
        risk_cmd += ["--risk-free-yield-csv", str(args.risk_free_yield_csv)]
    for name, shocks in args.stress_scenario:
        packed = ",".join(f"{ticker}={shock}" for ticker, shock in shocks.items())
        risk_cmd += ["--scenario", f"{name}:{packed}"]
    run_step("2/5 Portfolio risk and stress cases", risk_cmd)

    backtest_cmd = [
        py, str(ROOT / "backtest_engine.py"), "--benchmark", args.benchmark,
        "--lookback", str(args.backtest_lookback), "--skip-recent", str(args.backtest_skip_recent),
        "--top-n", str(args.backtest_top_n), "--rebalance-freq", args.backtest_rebalance_freq,
        "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--slippage-bps", str(args.slippage_bps), "--risk-free-rate", str(args.risk_free_rate),
    ]
    if args.risk_free_yield_csv:
        backtest_cmd += ["--risk-free-yield-csv", str(args.risk_free_yield_csv)]
    run_step("3/5 Baseline backtest", backtest_cmd)

    grid_cmd = [
        py, str(ROOT / "strategy_validation.py"), "--benchmark", args.benchmark, "--test-start", args.test_start,
        "--lookbacks", ",".join(map(str, args.lookbacks)), "--skip-recents", ",".join(map(str, args.skip_recents)),
        "--top-ns", ",".join(map(str, args.top_ns)),
        "--rebalance-freqs", ",".join(args.rebalance_freqs),
        "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--slippage-bps", str(args.slippage_bps), "--risk-free-rate", str(args.risk_free_rate),
    ]
    if args.risk_free_yield_csv:
        grid_cmd += ["--risk-free-yield-csv", str(args.risk_free_yield_csv)]
    run_step("4/5 Parameter sensitivity and single split", grid_cmd)

    if not args.skip_walk_forward:
        walk_cmd = [
            py, str(ROOT / "walk_forward_validation.py"), "--benchmark", args.benchmark,
            "--lookbacks", ",".join(map(str, args.lookbacks)), "--skip-recents", ",".join(map(str, args.skip_recents)),
            "--top-ns", ",".join(map(str, args.top_ns)),
            "--rebalance-freqs", ",".join(args.rebalance_freqs),
            "--train-window", str(args.train_window),
            "--test-window", str(args.test_window), "--selection-metric", args.selection_metric,
            "--transaction-cost-bps", str(args.transaction_cost_bps), "--slippage-bps", str(args.slippage_bps),
            "--risk-free-rate", str(args.risk_free_rate),
        ]
        if args.risk_free_yield_csv:
            walk_cmd += ["--risk-free-yield-csv", str(args.risk_free_yield_csv)]
        run_step("5/5 Rolling walk-forward validation", walk_cmd)

    report_path = write_report(OUTPUT_DIR, weights=weights_text, benchmark=args.benchmark)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark, "portfolio_weights": weights_text, "single_split_test_start": args.test_start,
        "parameters": {
            "lookbacks": args.lookbacks, "skip_recents": args.skip_recents, "top_ns": args.top_ns,
            "backtest_lookback": args.backtest_lookback, "backtest_skip_recent": args.backtest_skip_recent,
            "backtest_top_n": args.backtest_top_n, "backtest_rebalance_freq": args.backtest_rebalance_freq,
            "rebalance_freqs": args.rebalance_freqs,
            "risk_free_yield_csv": str(args.risk_free_yield_csv) if args.risk_free_yield_csv else None,
            "risk_free_rate_source": "dated annualized yield series" if risk_free_yield is not None else "constant annual rate",
            "train_window": args.train_window, "test_window": args.test_window,
            "selection_metric": args.selection_metric, "transaction_cost_bps": args.transaction_cost_bps,
            "slippage_bps": args.slippage_bps, "risk_free_rate": args.risk_free_rate,
            "has_portfolio_value": args.portfolio_value is not None,
            "holdings_csv_imported": imported_holdings is not None,
            "holdings_snapshot_positions": int(len(imported_holdings)) if imported_holdings is not None else 0,
            "stress_case_names": [name for name, _ in args.stress_scenario],
            "walk_forward_enabled": not args.skip_walk_forward,
        },
        "report_path": str(report_path.relative_to(ROOT)),
        "note": "Historical quantitative research only; not a trading instruction or return guarantee.",
    }
    (OUTPUT_DIR / "research_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nResearch run completed. Report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
