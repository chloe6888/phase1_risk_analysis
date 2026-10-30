#!/usr/bin/env python3
"""Local Streamlit dashboard for browsing outputs from the quant research pipeline."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from holdings_manager import load_holdings, summarize_holdings, risk_guardrail_alerts
from task_manager import (
    TaskAlreadyRunning, active_output_dir, get_current_task, get_task_log, start_task,
)

try:
    import streamlit as st
except ImportError:  # keep helper functions importable in minimal/test environments
    st = None

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PIPELINE_SCRIPT = ROOT / "run_research.py"
PRIVATE_HOLDINGS_PATH = ROOT / "data" / "private" / "holdings_snapshot.csv"


def read_json(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON object if present; return None for an absent result file."""
    source = Path(path)
    if not source.exists():
        return None
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return payload


def read_csv(path: str | Path, *, date_index: bool = False) -> pd.DataFrame | None:
    """Read an output CSV; return None when a step has not generated it yet."""
    source = Path(path)
    if not source.exists():
        return None
    if date_index:
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        frame = pd.read_csv(source)
    return frame


def output_status(output_dir: str | Path | None = None) -> dict[str, bool]:
    """Return availability of key research artifacts in the last successful run."""
    directory = Path(output_dir) if output_dir is not None else active_output_dir(ROOT)
    names = [
        "analysis_metadata.json",
        "portfolio_risk_summary.json",
        "portfolio_risk_contributions.csv",
        "backtest_summary.json",
        "backtest_equity_curve.csv",
        "strategy_validation_metadata.json",
        "walk_forward_summary.json",
        "holdings_summary.json",
        "holdings_valuation.csv",
        "holdings_latest_price_weights.csv",
        "holdings_risk_alerts.json",
        "memory_allocation_summary.json",
        "memory_position_diagnostics.csv",
        "memory_allocation_scenarios.csv",
        "research_report.md",
    ]
    return {name: (directory / name).is_file() for name in names}


def make_pipeline_command(
    *,
    weights: str | None,
    test_start: date | str,
    holdings_csv: str | Path | None = None,
    benchmark: str = "SPY",
    lookbacks: str = "63,126,252",
    skip_recents: str = "5,21",
    top_ns: str = "1,2,3",
    backtest_lookback: int = 126,
    backtest_skip_recent: int = 21,
    backtest_top_n: int = 2,
    train_window: int = 252,
    test_window: int = 63,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    risk_free_rate: float = 0.0,
    refresh: bool = False,
    allow_stale_cache: bool = False,
    skip_walk_forward: bool = False,
) -> list[str]:
    """Build a safe argv list; never uses shell=True."""
    start = test_start.isoformat() if isinstance(test_start, date) else str(test_start)
    if bool(weights and weights.strip()) == bool(holdings_csv):
        raise ValueError("Provide exactly one of manual weights or imported holdings CSV.")
    command = [sys.executable, str(PIPELINE_SCRIPT)]
    if holdings_csv:
        command += ["--holdings-csv", str(holdings_csv)]
    else:
        command += ["--weights", str(weights).strip()]
    command += [
        "--test-start", start,
        "--benchmark", benchmark,
        "--lookbacks", lookbacks,
        "--skip-recents", skip_recents,
        "--top-ns", top_ns,
        "--backtest-lookback", str(backtest_lookback),
        "--backtest-skip-recent", str(backtest_skip_recent),
        "--backtest-top-n", str(backtest_top_n),
        "--train-window", str(train_window),
        "--test-window", str(test_window),
        "--transaction-cost-bps", str(transaction_cost_bps),
        "--slippage-bps", str(slippage_bps),
        "--risk-free-rate", str(risk_free_rate),
    ]
    if refresh:
        command.append("--refresh")
    if allow_stale_cache:
        command.append("--allow-stale-cache")
    if skip_walk_forward:
        command.append("--skip-walk-forward")
    return command



COLUMN_LABELS = {
    "ticker": "股票代码",
    "name": "名称",
    "shares": "持股数量",
    "average_cost": "平均成本",
    "snapshot_price": "截图参考价",
    "reference_price": "最新参考价",
    "price": "参考价格",
    "price_date": "价格日期",
    "market_value": "持仓市值",
    "cost_basis": "持仓成本",
    "unrealized_pnl": "浮动盈亏",
    "unrealized_return_pct": "浮动收益率",
    "weight_pct": "持仓占比",
    "weight_of_reference_portfolio_pct": "参考组合权重",
    "dram_lookthrough_additional_value": "DRAM 穿透增加金额",
    "lookthrough_issuer_exposure_value": "穿透发行人敞口金额",
    "lookthrough_issuer_exposure_pct": "穿透发行人敞口比例",
    "add_permission": "加仓条件检查",
    "trend_flag": "趋势状态",
    "return_20d": "近 20 日收益率",
    "return_60d": "近 60 日收益率",
    "annualized_volatility_20d": "近 20 日年化波动率",
    "scenario_exceeds_memory_limit": "是否超过主题上限",
    "target_value": "目标持仓金额",
    "threshold_pct": "复核阈值",
    "lookback": "回看窗口（交易日）",
    "skip_recent": "跳过近期窗口（交易日）",
    "top_n": "持有标的数量",
    "year": "年份",
    "risk_group": "风险主题",
    "subgroup": "行业分类",
    "scenario": "方案",
    "scenario_total_target_weight_pct": "方案目标仓位",
    "target_weight_pct": "目标仓位",
    "current_shares": "当前股数",
    "latest_close": "最新收盘价",
    "target_shares": "目标股数",
    "share_delta": "股数调整",
    "action": "建议动作",
    "reason": "判断依据",
    "issuer": "发行人",
    "lookthrough_exposure_pct": "穿透敞口占比",
    "percentage_of_portfolio_volatility": "组合波动贡献",
    "contribution_to_portfolio_volatility": "波动贡献",
    "portfolio_weight": "组合权重",
    "annualized_volatility": "年化波动率",
    "rolling_annualized_volatility": "滚动年化波动率",
    "rolling_max_drawdown": "滚动最大回撤",
    "drawdown": "组合回撤",
    "window_observations": "观察窗口（交易日）",
    "date": "日期",
    "fold": "验证轮次",
    "status": "验证状态",
    "selected_lookback": "选中回看窗口",
    "selected_skip_recent": "选中跳过窗口",
    "selected_top_n": "选中持仓数量",
    "train_start": "训练开始日期",
    "train_end": "训练结束日期",
    "test_start": "测试开始日期",
    "test_end": "测试结束日期",
    "strategy_return": "策略收益率",
    "benchmark_return": "基准收益率",
    "strategy_gross_return": "策略毛收益率",
    "transaction_cost": "交易成本",
    "turnover": "换手率",
    "total_return": "累计收益率",
    "annualized_return": "年化收益率",
    "sharpe_ratio": "夏普比率",
    "sortino_ratio": "索提诺比率",
    "max_drawdown": "最大回撤",
    "tracking_error": "跟踪误差",
    "information_ratio": "信息比率",
    "beta": "贝塔系数",
    "volatility": "波动率",
    "value_at_risk": "风险价值估计（VaR）",
    "conditional_value_at_risk": "尾部平均损失（CVaR）",
    "portfolio_loss_pct": "组合损失比例",
    "portfolio_loss_value": "组合损失金额",
    "asset": "资产",
    "weight": "权重",
    "return": "收益率",
    "Date": "日期",
}
VALUE_LABELS = {
    "balanced": "平衡方案",
    "etf_core": "ETF 核心方案",
    "direct_only": "直接持股方案",
    "cap_only_preserve_etf": "保留 ETF 的最小调整方案",
    "REDUCE_TO_TARGET": "按目标仓位减仓复核",
    "HOLD_NO_AUTO_ADD": "暂不自动加仓",
    "HOLD": "维持当前股数",
    "NO_ADD_MEMORY_LIMIT": "存储主题仓位超限，暂不加仓",
    "WAIT_FOR_MEMORY_HEADROOM": "等待仓位回到安全缓冲区",
    "NO_ADD_ISSUER_LOOKTHROUGH_LIMIT": "单一发行人穿透敞口超限",
    "WAIT_FOR_TREND_CONFIRMATION": "等待趋势确认",
    "POSITIVE_TREND_BUT_SIZE_LIMITS_STILL_APPLY": "趋势转正，但仍受仓位限制",
    "STAGED_ADD_REVIEW_ONLY": "满足条件后再评估加仓",
    "REDUCE_MEMORY_SLEEVE": "存储主题仓位超限",
    "WITHIN_MEMORY_LIMIT": "存储主题仓位在限制内",
    "OK": "通过",
    "SKIP": "跳过",
}
FILE_LABELS = {
    "analysis_metadata.json": "行情数据与更新时间",
    "portfolio_risk_summary.json": "组合风险汇总",
    "portfolio_risk_contributions.csv": "各持仓风险贡献",
    "backtest_summary.json": "策略回测汇总",
    "backtest_equity_curve.csv": "策略净值曲线",
    "strategy_validation_metadata.json": "策略参数验证记录",
    "walk_forward_summary.json": "滚动样本外验证汇总",
    "holdings_summary.json": "持仓汇总",
    "holdings_valuation.csv": "持仓估值明细",
    "holdings_latest_price_weights.csv": "持仓价格日期与权重",
    "holdings_risk_alerts.json": "持仓集中度提醒",
    "memory_allocation_summary.json": "存储主题风险汇总",
    "memory_position_diagnostics.csv": "存储主题持仓诊断",
    "memory_allocation_scenarios.csv": "存储主题调整方案",
    "research_report.md": "综合研究报告",
}


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{float(value) * 100:.2f}%"


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{float(value):,.{digits}f}"


def _display_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a presentation-only Chinese copy; never mutate stored research data."""
    if frame is None:
        return None
    result = frame.copy()
    result = result.rename(columns={key: value for key, value in COLUMN_LABELS.items() if key in result.columns})
    for column in ("方案", "建议动作", "验证状态", "加仓条件检查", "趋势状态"):
        if column in result.columns:
            result[column] = result[column].replace(VALUE_LABELS)
    if "判断依据" in result.columns:
        result["判断依据"] = result["判断依据"].map(_localized_decision_reason)
    if "是否超过主题上限" in result.columns:
        result["是否超过主题上限"] = result["是否超过主题上限"].map(lambda value: "是" if bool(value) else "否")
    return result


def _localized_decision_reason(reason: Any) -> str:
    """Translate fixed explanatory sentences emitted by the memory-allocation model."""
    value = str(reason or "")
    if value.startswith("Current ") and " position exceeds this scenario's target weight" in value:
        ticker = value.split(" ", 2)[1]
        weight = value.rsplit("(", 1)[-1].rstrip(").")
        return f"当前 {ticker} 持仓高于此方案目标仓位（{weight}）。"
    if value.startswith("Below target;"):
        return "当前仓位低于目标；在满足独立入场条件前，系统不会自动加仓。"
    if value.startswith("At the rounded target quantity"):
        return "当前股数已达到此方案按整股取整后的目标数量。"
    return value


def _memory_overall_reason(summary: dict[str, Any]) -> str:
    current = float(summary.get("memory_sleeve_weight_pct", 0.0))
    limit = float(summary.get("max_memory_weight_pct", 0.0))
    excess = float(summary.get("memory_value_above_limit", 0.0))
    if summary.get("overall_action") == "REDUCE_MEMORY_SLEEVE":
        return f"MU、SKHY 与 DRAM 合计约占参考组合 {current:.1f}%，高于设定的 {limit:.1f}% 上限；估算超出金额约为 {excess:,.2f}。建议先复核主题仓位和单一发行人敞口。"
    return f"MU、SKHY 与 DRAM 合计约占参考组合 {current:.1f}%，目前未超过设定的 {limit:.1f}% 上限。仍需关注 ETF 穿透持仓变化和单一发行人集中度。"


def _memory_denominator_note(note: str) -> str:
    value = str(note or "")
    if value.startswith("Percentages are relative to the full net portfolio equity"):
        return "本页占比以你提供的账户净资产为分母；请确认该数值已包含现金、其他资产及负债。"
    if value.startswith("Cash, bonds, other accounts and financing are not included"):
        return "注意：当前比例仅按导入的股票和 ETF 持仓计算，尚未包含现金、债券、其他账户及融资负债。若要查看整个账户的真实仓位比例，请提供完整净资产。"
    return value or "组合权重的统计范围尚未记录，请核对持仓文件和账户净资产口径。"


def _localized_report(markdown: str) -> str:
    """Translate common report headings and labels for the dashboard view."""
    replacements = {
        "# Personal Quantitative Research Report": "# 综合投资研究报告",
        "## Data coverage": "## 行情数据覆盖",
        "## Imported holdings and concentration controls": "## 持仓估值与集中度检查",
        "### Snapshot reconciliation": "### 持仓数据核对",
        "### Concentration review flags": "### 集中度风险提醒",
        "## MU / SKHY / DRAM allocation review": "## MU / SKHY / DRAM 存储主题分析",
        "### Balanced scenario": "### 平衡方案",
        "## Portfolio risk": "## 组合风险分析",
        "## Baseline momentum backtest": "## 基准策略回测",
        "## Parameter sensitivity": "## 参数敏感性检验",
        "## Rolling walk-forward evaluation": "## 滚动样本外验证",
        "## Output files": "## 研究结果文件",
        "## Limitations": "## 方法局限与使用提醒",
        "- Generated at (UTC):": "- 生成时间（UTC）：",
        "- Benchmark:": "- 对比基准：",
        "- Portfolio weights:": "- 组合权重：",
        "- Purpose:": "- 报告用途：",
        "- Source:": "- 数据来源：",
        "- Requested dates:": "- 请求数据区间：",
        "- Last date per ticker:": "- 各标的最后数据日期：",
        "- Period:": "- 统计区间：",
        "- Return observations:": "- 有效收益观测：",
        "- Positions:": "- 持仓数量：",
        "- Latest-price market value:": "- 最近价格估算市值：",
        "- Reconstructed cost basis:": "- 持仓成本估算：",
        "- Unrealized P/L at latest available closes:": "- 按最近收盘价估算的浮动盈亏：",
        "- Largest position:": "- 最大单一持仓：",
        "- Largest risk group:": "- 最大风险主题：",
        "- Calculated market value minus broker summary:": "- 持仓估算市值与券商汇总差额：",
        "- Calculated P/L minus broker summary:": "- 浮动盈亏估算与券商汇总差额：",
        "- Broker-reported daily P/L (snapshot only):": "- 券商截图显示的当日盈亏：",
        "- Reconciled within configured tolerance:": "- 是否在设定误差范围内对齐：",
        "- Successful folds:": "- 成功验证轮次：",
        "- Out-of-sample observations:": "- 样本外有效观测：",
    }
    result = markdown
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def _task_step_label(step: str) -> str:
    labels = {
        "1/5 Data download, cache, and validation": "第 1 步：行情下载与数据质量检查",
        "2/5 Portfolio risk and stress cases": "第 2 步：组合风险与压力测试",
        "3/5 Baseline backtest": "第 3 步：基准策略回测",
        "4/5 Parameter sensitivity and single split": "第 4 步：参数敏感性与样本外检验",
        "5/5 Rolling walk-forward validation": "第 5 步：滚动样本外验证",
        "等待后台进程启动": "任务已提交，正在启动后台进程",
        "1/5 初始化研究流程": "正在初始化研究流程",
        "研究完成": "研究已完成",
        "研究失败": "研究未能完成",
    }
    for source, target in labels.items():
        if step.startswith(source):
            return step.replace(source, target, 1)
    return step or "后台准备中"


def render_task_monitor() -> None:
    """Show task state in plain Chinese, with progress and expandable diagnostics."""
    if st is None:
        return

    @st.fragment(run_every="3s")
    def _monitor() -> None:
        task = get_current_task(ROOT)
        with st.container(border=True):
            st.subheader("研究任务状态")
            if task is None:
                st.caption("尚未提交研究任务。完成下方持仓设置后，点击“开始研究”。")
                return

            state = str(task.get("state", "unknown"))
            task_id = str(task.get("task_id", "unknown"))
            step = _task_step_label(str(task.get("current_step") or "后台准备中"))
            message = str(task.get("message") or "")
            created = str(task.get("created_at") or "")

            if state in {"queued", "running"}:
                st.info("研究正在后台运行。你可以继续浏览页面；上一次完整成功的报告会保留。")
                st.write(f"**当前进度：** {step}")
                raw_step = str(task.get("current_step") or "")
                try:
                    current = int(raw_step.split("/", 1)[0])
                    st.progress(min(max(current / 5, 0.0), 1.0), text=f"研究流程进度：第 {current} / 5 步")
                except (ValueError, ZeroDivisionError):
                    st.progress(0.05, text="后台进程启动中")
            elif state == "succeeded":
                st.success("最近一次研究已成功完成，新的完整报告已经发布。")
                st.caption(f"完成时间（UTC）：{task.get('finished_at', '暂无记录')}")
                loaded = st.session_state.get("loaded_task_id")
                if loaded != task_id:
                    st.session_state["loaded_task_id"] = task_id
                    st.rerun()
            elif state == "failed":
                st.error("本次研究未能完成。请展开运行日志查看原因；上一次成功报告不会被覆盖。")
                if message:
                    st.caption(message)
            else:
                st.caption(f"任务状态：{state} · {message}")

            st.caption(f"任务编号：{task_id} · 提交时间（UTC）：{created or '暂无记录'}")
            with st.expander("查看运行日志（排查问题时使用）", expanded=state == "failed"):
                st.code(get_task_log(task_id, ROOT), language="text")

    _monitor()


def main() -> None:
    if st is None:
        raise RuntimeError(
            "未安装网页仪表盘依赖。请运行：python -m pip install -r requirements-dashboard.txt"
        )
    st.set_page_config(page_title="个人投资研究工作台", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem 1rem; border-radius: .65rem;}
        [data-testid="stTabs"] button {font-size: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("个人投资研究工作台")
    st.caption("以中文呈现持仓、风险与策略研究结果。所有数据用于辅助分析，不连接券商、不执行真实交易。")
    st.markdown(
        "**使用流程：** ① 导入持仓或填写组合权重 → ② 核对研究设置 → ③ 开始研究 → ④ 查看风险与报告。"
    )

    render_task_monitor()

    with st.container(border=True):
        st.subheader("开始一次研究")
        st.caption("建议首次使用先导入券商持仓文件；若暂时没有文件，也可以手动填写股票代码和组合权重。")
        with st.form("research_run_form"):
            upload_col, manual_col = st.columns([1.15, 1])
            with upload_col:
                uploaded_holdings = st.file_uploader(
                    "上传持仓文件（CSV）",
                    type=["csv"],
                    help="必需字段：ticker（股票代码）、shares（股数）、average_cost（平均成本）、snapshot_price（截图参考价）。",
                )
                use_saved_holdings = st.checkbox(
                    "没有新文件时，继续使用上次保存的持仓",
                    value=PRIVATE_HOLDINGS_PATH.is_file(),
                    help="持仓文件只保存在本地 data/private/ 目录，不会提交到 GitHub 仓库。",
                )
                st.caption("上传新文件会覆盖本地保存的持仓快照。请确认文件内容和股数准确。")
            with manual_col:
                weights = st.text_input(
                    "手动填写组合权重",
                    value="",
                    placeholder="GOOG=0.60,SPY=0.40",
                    help="格式为“股票代码=权重”，多个标的用英文逗号分隔；权重用小数表示，合计建议为 1。",
                )
                st.caption("例如：GOOG=0.60,SPY=0.40，表示组合中 GOOG 占 60%、SPY 占 40%。选择持仓文件后，此项会被忽略。")

            with st.expander("高级研究设置（一般保持默认即可）", expanded=False):
                setting_a, setting_b = st.columns(2)
                with setting_a:
                    benchmark = st.text_input("对比基准（用于衡量相对表现）", value="SPY").strip().upper() or "SPY"
                    test_start = st.date_input("样本外检验开始日期", value=date(2023, 1, 1))
                    lookbacks = st.text_input("动量观察区间（交易日）", value="63,126,252")
                    skips = st.text_input("跳过最近的交易日数", value="5,21")
                    top_ns = st.text_input("每期选取的持仓数量", value="1,2,3")
                with setting_b:
                    train_window = st.number_input("训练区间长度（交易日）", min_value=20, max_value=2000, value=252)
                    test_window = st.number_input("每轮检验区间长度（交易日）", min_value=10, max_value=500, value=63)
                    cost_bps = st.number_input("单边交易成本估计（基点）", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
                    slippage_bps = st.number_input("滑点估计（基点）", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
                    risk_free_rate = st.number_input("年化无风险利率（小数）", min_value=-0.5, max_value=1.0, value=0.0, step=0.005, format="%.3f")
                refresh = st.checkbox("强制重新下载全部行情（耗时可能更长）", value=False)
                allow_stale = st.checkbox(
                    "行情刷新失败时，允许使用旧缓存",
                    value=False,
                    help="旧缓存可能过期。只有明确接受数据时效风险时才勾选。",
                )
                skip_walk_forward = st.checkbox(
                    "跳过滚动样本外验证",
                    value=False,
                    help="仅在历史数据长度不足、无法完成滚动验证时使用。",
                )
            submitted = st.form_submit_button("开始研究", type="primary", use_container_width=True)

    if submitted:
        imported_path = None
        if uploaded_holdings is not None:
            PRIVATE_HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PRIVATE_HOLDINGS_PATH.write_bytes(uploaded_holdings.getvalue())
            imported_path = PRIVATE_HOLDINGS_PATH
        elif use_saved_holdings and PRIVATE_HOLDINGS_PATH.is_file():
            imported_path = PRIVATE_HOLDINGS_PATH

        if imported_path is None and not weights.strip():
            st.error("请上传持仓文件、选择继续使用已保存的持仓，或填写手动组合权重。")
        else:
            command = make_pipeline_command(
                weights=None if imported_path else weights.strip(),
                holdings_csv=imported_path,
                test_start=test_start,
                benchmark=benchmark,
                lookbacks=lookbacks.strip(),
                skip_recents=skips.strip(),
                top_ns=top_ns.strip(),
                train_window=int(train_window),
                test_window=int(test_window),
                transaction_cost_bps=float(cost_bps),
                slippage_bps=float(slippage_bps),
                risk_free_rate=float(risk_free_rate),
                refresh=refresh,
                allow_stale_cache=allow_stale,
                skip_walk_forward=skip_walk_forward,
            )
            try:
                task = start_task(command, repository_root=ROOT)
                st.success(f"研究任务已提交（编号：{task['task_id']}）。页面会自动刷新任务状态。")
                st.rerun()
            except TaskAlreadyRunning as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"无法启动研究任务：{type(exc).__name__}: {exc}")

    output_dir = active_output_dir(ROOT)
    status = output_status(output_dir)
    data_meta = read_json(output_dir / "analysis_metadata.json")
    portfolio = read_json(output_dir / "portfolio_risk_summary.json")
    backtest = read_json(output_dir / "backtest_summary.json")
    validation = read_json(output_dir / "strategy_validation_metadata.json")
    walk = read_json(output_dir / "walk_forward_summary.json")
    memory = read_json(output_dir / "memory_allocation_summary.json")
    memory_positions = read_csv(output_dir / "memory_position_diagnostics.csv")
    memory_decisions = read_csv(output_dir / "memory_allocation_scenarios.csv")

    if data_meta:
        dates = data_meta.get("data_last_dates", {})
        latest_dates = "；".join(f"{ticker}: {last_date}" for ticker, last_date in dates.items()) if isinstance(dates, dict) else "暂无"
        with st.container(border=True):
            st.markdown("**行情数据更新时间**")
            st.write(f"分析时间（UTC）：{data_meta.get('run_at_utc', '未知')} · 数据来源：{data_meta.get('source', '未记录')}")
            st.caption(f"各标的最后数据日期：{latest_dates or '暂无记录'}")
    else:
        st.info("当前还没有研究结果。请在上方导入持仓或填写组合权重，然后点击“开始研究”。首次下载行情需要网络。")

    imported_positions = None
    imported_alerts = []
    if PRIVATE_HOLDINGS_PATH.is_file():
        try:
            imported_snapshot = load_holdings(PRIVATE_HOLDINGS_PATH)
            imported_positions, _imported_summary = summarize_holdings(imported_snapshot)
            imported_alerts = risk_guardrail_alerts(imported_positions)
        except Exception as exc:
            st.error(f"本地持仓文件无法解析：{exc}")

    overview_tab, risk_tab, backtest_tab, validation_tab, memory_tab, files_tab = st.tabs(
        ["账户概览", "组合风险", "策略回测", "稳健性检验", "存储主题", "研究报告与文件"]
    )

    with overview_tab:
        st.subheader("账户概览")
        if portfolio:
            metric_cols = st.columns(4)
            metric_cols[0].metric("区间累计收益", _pct(portfolio.get("total_return")), help="研究区间内组合净值的累计变化，不代表未来收益。")
            metric_cols[1].metric("年化收益率", _pct(portfolio.get("annualized_return")), help="将历史区间收益折算为年化口径，样本较短时参考价值有限。")
            metric_cols[2].metric("年化波动率", _pct(portfolio.get("annualized_volatility")), help="历史日收益波动的年化估计，不等同于最大可能亏损。")
            metric_cols[3].metric("最大回撤", _pct(portfolio.get("max_drawdown")), help="研究区间内从历史净值高点到随后低点的最大跌幅。")
            st.caption(
                f"统计区间：{portfolio.get('start_date', '暂无')} 至 {portfolio.get('end_date', '暂无')} · "
                f"有效收益观测：{portfolio.get('observations', '暂无')} 个交易日"
            )
        if imported_positions is not None and not imported_positions.empty:
            st.markdown("### 已导入持仓")
            st.caption("下表的价格来自导入文件中的截图参考价；研究运行成功后，可在估值文件中查看行情数据源的最近收盘价。")
            holdings_columns = [
                "ticker", "name", "shares", "average_cost", "snapshot_price",
                "market_value", "unrealized_pnl", "unrealized_return_pct", "weight_pct",
            ]
            shown = imported_positions[[col for col in holdings_columns if col in imported_positions.columns]]
            st.dataframe(_display_frame(shown), hide_index=True, use_container_width=True)
            for alert in imported_alerts:
                st.warning(alert["message"])
            reconciliation = read_json(output_dir / "holdings_reconciliation.json")
            if reconciliation and not reconciliation.get("within_tolerance", True):
                st.warning("持仓明细合计与券商汇总金额未能在设定误差范围内对齐。请核对股数、报价时间、遗漏持仓和费用后，再依赖组合风险结果。")
        latest_holdings = read_csv(output_dir / "holdings_valuation.csv")
        if latest_holdings is not None and not latest_holdings.empty:
            st.markdown("### 按最近收盘价估算的持仓市值")
            st.caption("这是历史收盘价估算，不是实时成交报价。")
            columns = [
                "ticker", "shares", "average_cost", "reference_price", "market_value",
                "unrealized_pnl", "unrealized_return_pct", "weight_pct", "risk_group", "subgroup",
            ]
            st.dataframe(_display_frame(latest_holdings[[col for col in columns if col in latest_holdings.columns]]), hide_index=True, use_container_width=True)
        if backtest:
            st.markdown("### 策略相对基准表现")
            strategy = backtest.get("strategy", {})
            benchmark_result = backtest.get("benchmark", {})
            left, right, extra = st.columns(3)
            left.metric("策略累计收益", _pct(strategy.get("total_return")))
            right.metric("对比基准累计收益", _pct(benchmark_result.get("total_return")))
            extra.metric("相对基准差值", _pct(backtest.get("excess_total_return")), help="策略累计收益减去基准累计收益；不是未来超额收益预测。")
        if not portfolio and imported_positions is None and not backtest:
            st.markdown("#### 第一次使用")
            st.write("1. 从券商导出持仓 CSV，或在上方填写组合权重。")
            st.write("2. 首次运行需要联网下载历史行情。")
            st.write("3. 运行完成后，查看账户概览、组合风险和策略回测。")

    with risk_tab:
        st.subheader("组合风险")
        st.caption("这一页回答三个问题：风险主要来自哪里、组合历史上经历过多大回撤、在假设冲击下可能受到多大影响。")
        if portfolio:
            risk_cols = st.columns(4)
            risk_cols[0].metric("年化波动率", _pct(portfolio.get("annualized_volatility")))
            risk_cols[1].metric("最大回撤", _pct(portfolio.get("max_drawdown")))
            risk_cols[2].metric("夏普比率", _number(portfolio.get("sharpe_ratio")), help="单位风险对应的历史超额收益表现，数值越高不代表未来一定更好。")
            risk_cols[3].metric("索提诺比率", _number(portfolio.get("sortino_ratio")), help="侧重下行波动的历史风险调整收益指标。")
            st.caption("VaR 是基于历史样本估算的单期损失分位数；CVaR 是落入尾部情形时的平均损失估计。二者都不是最大亏损保证。")
            risk_table = pd.DataFrame([
                {"指标": "累计收益率", "数值": _pct(portfolio.get("total_return"))},
                {"指标": "年化收益率", "数值": _pct(portfolio.get("annualized_return"))},
                {"指标": "年化波动率", "数值": _pct(portfolio.get("annualized_volatility"))},
                {"指标": "夏普比率", "数值": _number(portfolio.get("sharpe_ratio"))},
                {"指标": "索提诺比率", "数值": _number(portfolio.get("sortino_ratio"))},
                {"指标": "最大回撤", "数值": _pct(portfolio.get("max_drawdown"))},
                {"指标": "历史单期风险价值（VaR）", "数值": _pct(portfolio.get("historical_var_one_period"))},
                {"指标": "历史尾部平均损失（CVaR）", "数值": _pct(portfolio.get("historical_cvar_one_period"))},
            ])
            st.dataframe(risk_table, hide_index=True, use_container_width=True)
        contributions = read_csv(output_dir / "portfolio_risk_contributions.csv")
        if contributions is not None:
            st.markdown("### 各持仓对组合波动的贡献")
            st.dataframe(_display_frame(contributions), hide_index=True, use_container_width=True)
            if "percentage_of_portfolio_volatility" in contributions.columns:
                plot = contributions.set_index(contributions.columns[0])[["percentage_of_portfolio_volatility"]].rename(
                    columns={"percentage_of_portfolio_volatility": "组合波动贡献"}
                )
                st.bar_chart(plot)
        rolling = read_csv(output_dir / "portfolio_rolling_risk.csv", date_index=True)
        if rolling is not None and not rolling.empty:
            st.markdown("### 近期风险变化")
            for field, label in (
                ("rolling_annualized_volatility", "滚动年化波动率"),
                ("rolling_max_drawdown", "滚动最大回撤"),
            ):
                if field in rolling.columns:
                    view = rolling.pivot_table(index=rolling.index, columns="window_observations", values=field)
                    view.columns = [f"{int(col)} 日观察窗口" for col in view.columns]
                    st.markdown(f"**{label}**")
                    st.line_chart(view)
        drawdown = read_csv(output_dir / "portfolio_drawdown.csv", date_index=True)
        if drawdown is not None and "drawdown" in drawdown.columns:
            st.markdown("### 历史回撤曲线")
            st.area_chart(drawdown[["drawdown"]].rename(columns={"drawdown": "组合回撤"}))
        stress = read_csv(output_dir / "portfolio_stress_tests.csv")
        if stress is not None:
            st.markdown("### 假设情景压力测试")
            st.caption("这是用户设定价格冲击后的静态测算，不是行情预测。")
            st.dataframe(_display_frame(stress), hide_index=True, use_container_width=True)
        if contributions is None and rolling is None and drawdown is None and stress is None:
            st.info("目前还没有风险分析结果。完成一次研究后，这里会展示风险贡献、历史回撤和压力测试。")

    with backtest_tab:
        st.subheader("策略回测")
        st.caption("回测用于检查一套规则在历史数据中的表现。历史结果不代表未来收益，也不等于真实交易成交结果。")
        equity = read_csv(output_dir / "backtest_equity_curve.csv", date_index=True)
        if equity is not None and not equity.empty:
            st.markdown("### 策略与基准净值走势")
            renamed_equity = equity.rename(columns={"strategy": "策略组合", "benchmark": "对比基准"})
            st.line_chart(renamed_equity)
        daily = read_csv(output_dir / "backtest_daily_returns.csv", date_index=True)
        if daily is not None:
            st.markdown("### 最近交易日记录")
            st.caption("仅展示最近 60 条记录，完整数据可在研究文件中查看。")
            st.dataframe(_display_frame(daily.tail(60)), use_container_width=True)
        applied = read_csv(output_dir / "backtest_applied_weights.csv", date_index=True)
        if applied is not None:
            st.markdown("### 回测期间采用的组合权重")
            st.line_chart(applied)
        if equity is None and daily is None and applied is None:
            st.info("目前还没有回测结果。完成一次研究后，这里会展示策略净值、每日表现与实际使用的权重。")

    with validation_tab:
        st.subheader("稳健性检验")
        st.caption("这一页检查策略是否过度依赖某一组参数，以及换一段历史区间后结果是否仍然稳定。")
        if validation:
            x, y, z = st.columns(3)
            x.metric("尝试的参数组合", int(validation.get("requested_combinations", 0)))
            y.metric("可完成的组合", int(validation.get("successful_combinations", 0)))
            z.metric("因数据不足跳过", int(validation.get("skipped_combinations", 0)))
        grid = read_csv(output_dir / "strategy_parameter_sweep.csv")
        if grid is not None:
            st.markdown("### 不同参数下的样本内与样本外表现")
            st.dataframe(_display_frame(grid), use_container_width=True)
        yearly = read_csv(output_dir / "strategy_yearly_performance.csv")
        if yearly is not None:
            st.markdown("### 按年份拆分的策略表现")
            st.dataframe(_display_frame(yearly), use_container_width=True)
        if walk:
            st.markdown("### 滚动样本外验证结果")
            a, b, c = st.columns(3)
            a.metric("成功验证轮次", f"{walk.get('folds_successful', 0)} / {walk.get('folds_total', 0)}")
            b.metric("样本外交易日数", int(walk.get("oos_observations", 0)))
            c.metric("样本外相对基准差值", _pct(walk.get("out_of_sample_excess_total_return")))
            st.caption(
                f"训练窗口：{walk.get('train_window_observations', '暂无')} 个交易日 · "
                f"检验窗口：{walk.get('test_window_observations', '暂无')} 个交易日 · "
                f"检验区间：{walk.get('first_oos_date', '暂无')} 至 {walk.get('last_oos_date', '暂无')}"
            )
            if walk.get("selection_warning"):
                st.warning("参数只应使用各轮之前的训练数据选择。反复根据测试结果调参，会削弱样本外检验的独立性。")
        folds = read_csv(output_dir / "walk_forward_folds.csv")
        if folds is not None:
            st.markdown("### 每轮训练与检验明细")
            st.dataframe(_display_frame(folds), use_container_width=True)
        oos = read_csv(output_dir / "walk_forward_oos_returns.csv", date_index=True)
        if oos is not None and {"strategy_return", "benchmark_return"}.issubset(oos.columns):
            st.markdown("### 滚动检验期间的累计表现")
            equity_oos = (1.0 + oos[["strategy_return", "benchmark_return"]]).cumprod()
            equity_oos.columns = ["滚动检验策略", "对比基准"]
            st.line_chart(equity_oos)
        if grid is None and folds is None and validation is None and walk is None:
            st.info("目前还没有稳健性检验结果。完成一次研究后，这里会展示参数比较和滚动样本外结果。")

    with memory_tab:
        st.subheader("存储主题持仓分析")
        st.caption("检查 MU、SKHY 与 DRAM 之间的重叠敞口，避免直接持股和 ETF 间接持股叠加后过度集中。")
        if memory:
            a, b, c, d = st.columns(4)
            a.metric("存储主题当前占比", _pct(float(memory.get("memory_sleeve_weight_pct", 0.0)) / 100.0))
            b.metric("风险预算上限", _pct(float(memory.get("max_memory_weight_pct", 0.0)) / 100.0))
            c.metric("超过上限的金额", _number(memory.get("memory_value_above_limit")))
            d.metric("规则检查结果", VALUE_LABELS.get(str(memory.get("overall_action")), str(memory.get("overall_action", "暂无数据"))))
            st.write(_memory_overall_reason(memory))
            st.warning(_memory_denominator_note(str(memory.get("denominator_warning", ""))))
            st.caption(
                f"ETF 穿透持仓假设日期：{memory.get('lookthrough_asof', '暂无')}。ETF 每日持仓可能变化，交易前应重新核对。"
            )
            exposures = memory.get("lookthrough_issuer_weights_pct", {})
            if exposures:
                st.markdown("### DRAM 内部的发行人穿透敞口")
                exposure_frame = pd.DataFrame([
                    {"发行人": key, "穿透敞口占比": f"{float(value):.2f}%"}
                    for key, value in exposures.items()
                ])
                st.dataframe(exposure_frame, hide_index=True, use_container_width=True)
        if memory_positions is not None:
            st.markdown("### 最新价格、趋势与波动")
            st.dataframe(_display_frame(memory_positions), use_container_width=True)
        if memory_decisions is not None:
            st.markdown("### 平衡方案的持仓调整参考")
            balanced = memory_decisions[memory_decisions["scenario"] == "balanced"]
            if not balanced.empty:
                columns = [
                    "ticker", "current_shares", "latest_close", "target_weight_pct",
                    "target_shares", "share_delta", "action", "reason",
                ]
                st.dataframe(_display_frame(balanced[[name for name in columns if name in balanced.columns]]), hide_index=True, use_container_width=True)
            st.markdown("### 三种持仓结构对比")
            columns = [
                "scenario", "scenario_total_target_weight_pct", "ticker",
                "target_weight_pct", "target_shares", "share_delta", "action",
            ]
            st.dataframe(_display_frame(memory_decisions[[name for name in columns if name in memory_decisions.columns]]), hide_index=True, use_container_width=True)
        if memory is None:
            st.info("导入包含 MU、SKHY 或 DRAM 的持仓，并运行完整研究流程后，这里才会显示存储主题分析。")
        st.warning("目标仓位和风险上限是可修改的研究假设，不是自动买卖指令。交易前仍需核对公司基本面、估值、ETF 最新持仓和账户净资产。")

    with files_tab:
        st.subheader("研究报告与结果文件")
        st.caption("这里查看本次研究是否生成了关键文件，以及综合报告中的结论和限制。")
        output_table = pd.DataFrame([
            {"文件用途": FILE_LABELS.get(name, name), "生成状态": "已生成" if exists else "尚未生成"}
            for name, exists in status.items()
        ])
        st.dataframe(output_table, hide_index=True, use_container_width=True)
        report_path = output_dir / "research_report.md"
        if report_path.exists():
            st.markdown("### 综合研究报告")
            st.markdown(_localized_report(report_path.read_text(encoding="utf-8")))
        else:
            st.info("研究报告尚未生成。请先完成一次研究任务。")
        with st.expander("查看本地结果文件位置"):
            st.code(str(output_dir), language="text")
            st.caption("个人持仓原始文件保存在 data/private/，研究结果保存在 outputs/ 或对应的独立运行目录。")
        st.caption("此平台仅用于投资研究，不连接券商、不发送真实订单。公开部署前必须设置身份认证和访问控制。")


if __name__ == "__main__":
    main()
