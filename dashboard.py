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


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def render_task_monitor() -> None:
    """Render a lightweight, periodically refreshed status panel."""
    if st is None:
        return

    @st.fragment(run_every="3s")
    def _monitor() -> None:
        task = get_current_task(ROOT)
        if task is None:
            st.info("尚未运行研究任务。请从左侧导入持仓并启动分析。")
            return

        state = str(task.get("state", "unknown"))
        task_id = str(task.get("task_id", "unknown"))
        step = str(task.get("current_step") or "后台准备中")
        message = str(task.get("message") or "")
        task_label = "个股估值" if task.get("task_type") == "valuation" else "组合研究"
        if state in {"queued", "running"}:
            st.info(f"{task_label}任务运行中 · {step}。页面可以继续浏览，上一次成功结果不会被覆盖。")
        elif state == "succeeded":
            st.success(f"最近一次{task_label}任务已成功 · {task.get('finished_at', '')}")
            loaded = st.session_state.get("loaded_task_id")
            if loaded != task_id:
                st.session_state["loaded_task_id"] = task_id
                st.rerun()
        elif state == "failed":
            st.error(f"研究任务失败 · {message}")
        else:
            st.caption(f"最近任务状态：{state} · {message}")

        with st.expander("查看本次运行日志", expanded=state == "failed"):
            st.code(get_task_log(task_id, ROOT), language="text")

    _monitor()


def main() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install optional dashboard dependencies with "
            "'python -m pip install -r requirements-dashboard.txt'."
        )
    st.set_page_config(page_title="个人量化研究系统", page_icon="📈", layout="wide")
    st.title("个人量化研究系统")
    st.caption("个人投资研究工作台 · 持仓风险、策略验证与历史研究 · 不连接券商、不执行真实交易")

    output_dir = active_output_dir(ROOT)
    status = output_status(output_dir)
    valuation_output_dir = active_output_dir(ROOT, task_type="valuation")
    st.markdown("### 研究任务")
    render_task_monitor()
    with st.sidebar:
        st.header("运行研究")
        st.caption("提交后会实际运行数据更新和量化分析脚本。首次下载行情需要网络。持仓文件保存在本机 data/private/，该目录已加入 Git 忽略规则。")
        with st.form("research_run_form"):
            st.markdown("**第一步：选择你的持仓数据**")
            uploaded_holdings = st.file_uploader(
                "上传券商持仓 CSV",
                type=["csv"],
                help="必需列：ticker、shares、average_cost、snapshot_price。",
            )
            use_imported_holdings = st.checkbox(
                "根据导入持仓自动计算组合权重",
                value=PRIVATE_HOLDINGS_PATH.is_file(),
                help="股数和成本请与券商记录核对；现金和融资负债需另行计入。",
            )
            weights = st.text_input(
                "或者输入研究组合权重",
                value="",
                placeholder="GOOG=0.60,SPY=0.40",
                help="仅在不使用持仓 CSV 时填写，权重使用小数。",
            )
            with st.expander("高级研究设置（首次使用可保持默认）", expanded=False):
                benchmark = st.text_input("对比基准", value="SPY").strip().upper() or "SPY"
                test_start = st.date_input("样本外测试起始日期", value=date(2023, 1, 1))
                lookbacks = st.text_input("动量回看窗口（交易日）", value="63,126,252")
                skips = st.text_input("跳过最近窗口（交易日）", value="5,21")
                top_ns = st.text_input("策略持有资产数量", value="1,2,3")
                train_window = st.number_input("Walk-forward 训练窗口（交易日）", min_value=20, max_value=2000, value=252)
                test_window = st.number_input("Walk-forward 测试窗口（交易日）", min_value=10, max_value=500, value=63)
                cost_bps = st.number_input("交易成本（基点）", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
                slippage_bps = st.number_input("滑点（基点）", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
                risk_free_rate = st.number_input("年化无风险利率（小数）", min_value=-0.5, max_value=1.0, value=0.0, step=0.005, format="%.3f")
                refresh = st.checkbox("强制重新下载全部行情", value=False)
                allow_stale = st.checkbox(
                    "刷新失败时允许使用旧行情缓存",
                    value=False,
                    help="仅在明确接受数据过期风险时勾选。",
                )
                skip_walk_forward = st.checkbox(
                    "跳过 Walk-forward 验证",
                    value=False,
                    help="仅当历史长度不足时使用。",
                )
            submitted = st.form_submit_button("启动研究分析", type="primary", use_container_width=True)

    if submitted:
        imported_path = None
        if uploaded_holdings is not None:
            PRIVATE_HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PRIVATE_HOLDINGS_PATH.write_bytes(uploaded_holdings.getvalue())
            imported_path = PRIVATE_HOLDINGS_PATH
        elif use_imported_holdings and PRIVATE_HOLDINGS_PATH.is_file():
            imported_path = PRIVATE_HOLDINGS_PATH

        if use_imported_holdings and imported_path is None:
            st.sidebar.error("请先上传持仓 CSV，或将文件放在 data/private/holdings_snapshot.csv。")
        elif not use_imported_holdings and not weights.strip():
            st.sidebar.error("请填写手工组合权重，或勾选使用导入持仓自动计算权重。")
        else:
            command = make_pipeline_command(
                weights=None if use_imported_holdings else weights.strip(),
                holdings_csv=imported_path if use_imported_holdings else None,
                test_start=test_start, benchmark=benchmark,
                lookbacks=lookbacks.strip(), skip_recents=skips.strip(), top_ns=top_ns.strip(),
                train_window=int(train_window), test_window=int(test_window),
                transaction_cost_bps=float(cost_bps), slippage_bps=float(slippage_bps),
                risk_free_rate=float(risk_free_rate), refresh=refresh,
                allow_stale_cache=allow_stale, skip_walk_forward=skip_walk_forward,
            )
            try:
                task = start_task(command, repository_root=ROOT)
                st.sidebar.success(f"任务已提交：{task['task_id']}。你可以继续浏览上一次成功的分析结果。")
                st.rerun()
            except TaskAlreadyRunning as exc:
                st.sidebar.warning(str(exc))
            except Exception as exc:
                st.sidebar.error(f"无法启动研究任务：{type(exc).__name__}: {exc}")

    imported_positions = None
    imported_alerts = []
    if PRIVATE_HOLDINGS_PATH.is_file():
        try:
            imported_snapshot = load_holdings(PRIVATE_HOLDINGS_PATH)
            imported_positions, _imported_summary = summarize_holdings(imported_snapshot)
            imported_alerts = risk_guardrail_alerts(imported_positions)
        except Exception as exc:
            st.error(f"本地持仓文件无法解析：{exc}")

    data_meta = read_json(output_dir / "analysis_metadata.json")
    portfolio = read_json(output_dir / "portfolio_risk_summary.json")
    backtest = read_json(output_dir / "backtest_summary.json")
    validation = read_json(output_dir / "strategy_validation_metadata.json")
    walk = read_json(output_dir / "walk_forward_summary.json")
    memory = read_json(output_dir / "memory_allocation_summary.json")
    memory_positions = read_csv(output_dir / "memory_position_diagnostics.csv")
    memory_decisions = read_csv(output_dir / "memory_allocation_scenarios.csv")

    if data_meta:
        st.caption(
            f"上次数据分析时间（UTC）：{data_meta.get('run_at_utc', '未知')} · "
            f"数据截止日期：{data_meta.get('data_last_dates', {})}"
        )
    else:
        st.info("尚无分析结果。请在左侧填写组合目标权重并运行研究流程，或先在终端运行 run_research.py。")

    if imported_positions is not None:
        st.subheader("已导入的持仓（截图参考价格）")
        st.caption("这里先显示导入 CSV 中的截图参考价；运行完整研究流程后，可在最新估值结果中查看行情数据源的最近收盘价。")
        st.dataframe(
            imported_positions[["ticker", "name", "shares", "average_cost", "snapshot_price", "market_value", "unrealized_pnl", "unrealized_return_pct", "weight_pct"]],
            hide_index=True, use_container_width=True,
        )
        if imported_alerts:
            for alert in imported_alerts:
                st.warning(alert["message"])
        snapshot_reconciliation = read_json(output_dir / "holdings_reconciliation.json")
        if snapshot_reconciliation and not snapshot_reconciliation.get("within_tolerance", True):
            st.warning(
                "导入持仓汇总与券商总额未在容差内对齐。请核对股数、报价时间、截图遗漏持仓、费用和汇总口径后，再依赖组合风险结果。"
            )

    if portfolio:
        st.subheader("组合风险概览")
        a, b, c, d, e = st.columns(5)
        a.metric("累计收益", _pct(portfolio.get("total_return")))
        b.metric("年化收益", _pct(portfolio.get("annualized_return")))
        c.metric("年化波动", _pct(portfolio.get("annualized_volatility")))
        d.metric("最大回撤", _pct(portfolio.get("max_drawdown")))
        e.metric("Sharpe", "N/A" if portfolio.get("sharpe_ratio") is None else f"{portfolio['sharpe_ratio']:.2f}")
        st.caption(
            f"统计区间：{portfolio.get('start_date', 'N/A')} — {portfolio.get('end_date', 'N/A')}；"
            f"有效收益观测：{portfolio.get('observations', 'N/A')}"
        )

    overview_tab, risk_tab, backtest_tab, validation_tab, memory_tab, valuation_tab, files_tab = st.tabs(
        ["账户总览", "组合风险", "策略表现", "策略稳健性", "存储主题", "个股估值", "研究报告"]
    )
    with overview_tab:
        latest_holdings = read_csv(output_dir / "holdings_valuation.csv")
        latest_valuation_meta = read_csv(output_dir / "holdings_latest_price_weights.csv")
        if latest_holdings is not None and not latest_holdings.empty:
            st.markdown("### 最新可用收盘价估值（非实时成交价）")
            columns = [name for name in (
                "ticker", "shares", "average_cost", "reference_price", "market_value",
                "unrealized_pnl", "unrealized_return_pct", "weight_pct", "risk_group", "subgroup",
            ) if name in latest_holdings.columns]
            st.dataframe(latest_holdings[columns], hide_index=True, use_container_width=True)
            if latest_valuation_meta is not None and "price_date" in latest_valuation_meta.columns:
                st.caption("各持仓的价格日期见“结果文件”中的 holdings_latest_price_weights.csv。")
        if portfolio:
            st.markdown("### 组合风险指标")
            risk_table = pd.DataFrame([
                {"指标": "累计收益", "数值": _pct(portfolio.get("total_return"))},
                {"指标": "年化收益", "数值": _pct(portfolio.get("annualized_return"))},
                {"指标": "年化波动", "数值": _pct(portfolio.get("annualized_volatility"))},
                {"指标": "Sharpe", "数值": portfolio.get("sharpe_ratio")},
                {"指标": "Sortino", "数值": portfolio.get("sortino_ratio")},
                {"指标": "最大回撤", "数值": _pct(portfolio.get("max_drawdown"))},
                {"指标": "历史单期 VaR", "数值": _pct(portfolio.get("historical_var_one_period"))},
                {"指标": "历史单期 CVaR", "数值": _pct(portfolio.get("historical_cvar_one_period"))},
            ])
            st.dataframe(risk_table, hide_index=True, use_container_width=True)
        if backtest:
            s = backtest.get("strategy", {})
            bm = backtest.get("benchmark", {})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 动量基准策略")
                st.write(f"区间：{backtest.get('start_date', 'N/A')} — {backtest.get('end_date', 'N/A')}")
                st.write(f"策略累计收益：{_pct(s.get('total_return'))}")
                st.write(f"基准累计收益：{_pct(bm.get('total_return'))}")
                st.write(f"超额累计收益：{_pct(backtest.get('excess_total_return'))}")
            with col2:
                st.markdown("### 样本外验证")
                if walk:
                    st.write(f"成功滚动窗口：{walk.get('folds_successful', 0)} / {walk.get('folds_total', 0)}")
                    st.write(f"样本外观测数：{walk.get('oos_observations', 0)}")
                    st.write(f"样本外超额累计收益：{_pct(walk.get('out_of_sample_excess_total_return'))}")
                elif validation:
                    st.write(f"参数组合有效数：{validation.get('successful_combinations', 0)}")
                    st.write(f"参数组合跳过数：{validation.get('skipped_combinations', 0)}")
        if not portfolio and not backtest:
            st.info("请先运行一次研究流程，页面会自动读取已生成的输出。")

    with risk_tab:
        contributions = read_csv(output_dir / "portfolio_risk_contributions.csv")
        if contributions is not None:
            st.markdown("### 各资产对组合波动的贡献")
            st.dataframe(contributions, use_container_width=True)
            if "percentage_of_portfolio_volatility" in contributions.columns:
                plot = contributions.set_index(contributions.columns[0])[["percentage_of_portfolio_volatility"]]
                st.bar_chart(plot)
        rolling = read_csv(output_dir / "portfolio_rolling_risk.csv", date_index=True)
        if rolling is not None and not rolling.empty:
            st.markdown("### 滚动波动率与回撤")
            for field in ("rolling_annualized_volatility", "rolling_max_drawdown"):
                if field in rolling.columns:
                    view = rolling.pivot_table(index=rolling.index, columns="window_observations", values=field)
                    view.columns = [f"{int(col)} 日窗口" for col in view.columns]
                    st.write("年化波动率" if field == "rolling_annualized_volatility" else "滚动最大回撤")
                    st.line_chart(view)
        drawdown = read_csv(output_dir / "portfolio_drawdown.csv", date_index=True)
        if drawdown is not None and "drawdown" in drawdown.columns:
            st.markdown("### 组合历史回撤")
            st.area_chart(drawdown[["drawdown"]])
        stress = read_csv(output_dir / "portfolio_stress_tests.csv", date_index=False)
        if stress is not None:
            st.markdown("### 压力情景")
            st.dataframe(stress, use_container_width=True)
        if contributions is None and rolling is None and drawdown is None:
            st.info("组合风险输出尚不存在。运行研究流程后这里会显示风险贡献、滚动风险和回撤。")

    with backtest_tab:
        equity = read_csv(output_dir / "backtest_equity_curve.csv", date_index=True)
        if equity is not None and not equity.empty:
            st.markdown("### 策略与基准净值")
            st.line_chart(equity)
        daily = read_csv(output_dir / "backtest_daily_returns.csv", date_index=True)
        if daily is not None:
            st.markdown("### 每日策略收益与换手")
            st.dataframe(daily.tail(100), use_container_width=True)
            if "strategy_return" in daily.columns:
                st.line_chart(daily[["strategy_return", "benchmark_return"]].cumsum())
        applied = read_csv(output_dir / "backtest_applied_weights.csv", date_index=True)
        if applied is not None:
            st.markdown("### 各收益区间使用的目标权重")
            st.line_chart(applied)
        if equity is None and daily is None:
            st.info("回测输出尚不存在。运行研究流程后这里会显示策略净值和每日表现。")

    with validation_tab:
        if validation:
            st.markdown("### 参数验证概况")
            x, y, z = st.columns(3)
            x.metric("参数组合", int(validation.get("requested_combinations", 0)))
            y.metric("有效组合", int(validation.get("successful_combinations", 0)))
            z.metric("跳过组合", int(validation.get("skipped_combinations", 0)))
        grid = read_csv(output_dir / "strategy_parameter_sweep.csv")
        if grid is not None:
            st.markdown("### 样本内 / 样本外参数网格")
            st.dataframe(grid, use_container_width=True)
        yearly = read_csv(output_dir / "strategy_yearly_performance.csv")
        if yearly is not None:
            st.markdown("### 逐年表现")
            st.dataframe(yearly, use_container_width=True)
        if walk:
            st.markdown("### 滚动 Walk-forward 汇总")
            st.json(walk)
        folds = read_csv(output_dir / "walk_forward_folds.csv")
        if folds is not None:
            st.markdown("### 每轮参数选择与测试结果")
            st.dataframe(folds, use_container_width=True)
        oos = read_csv(output_dir / "walk_forward_oos_returns.csv", date_index=True)
        if oos is not None and {"strategy_return", "benchmark_return"}.issubset(oos.columns):
            st.markdown("### 样本外收益序列")
            equity_oos = (1.0 + oos[["strategy_return", "benchmark_return"]]).cumprod()
            equity_oos.columns = ["Walk-forward 策略", "基准"]
            st.line_chart(equity_oos)
        if grid is None and folds is None:
            st.info("稳健性验证输出尚不存在。运行研究流程后这里会显示参数网格和 walk-forward 结果。")

    with memory_tab:
        st.markdown("### MU / SKHY / DRAM 重叠敞口与持仓建议")
        st.caption("系统依据风险预算和最近可用收盘价生成可解释的调仓方案，不预测股价、不自动下单。")
        if memory:
            a, b, c1, d = st.columns(4)
            a.metric("当前存储主题占比", _pct(float(memory.get("memory_sleeve_weight_pct", 0.0)) / 100.0))
            b.metric("复核上限", _pct(float(memory.get("max_memory_weight_pct", 0.0)) / 100.0))
            c1.metric("超限金额估算", f"{memory.get('memory_value_above_limit', 0):,.2f}")
            d.metric("规则判断", memory.get("overall_action", "N/A"))
            st.write(memory.get("overall_reason", ""))
            st.caption(memory.get("denominator_warning", ""))
            st.caption(
                f"DRAM 持仓穿透比例假设日期：{memory.get('lookthrough_asof', 'N/A')}；"
                "ETF 持仓会变动，使用前应更新假设。"
            )
            exposures = memory.get("lookthrough_issuer_weights_pct", {})
            if exposures:
                st.markdown("#### DRAM 穿透后的单一发行人敞口")
                st.dataframe(
                    pd.DataFrame([{"发行人": key, "穿透敞口占比": f"{float(value):.2f}%"} for key, value in exposures.items()]),
                    hide_index=True, use_container_width=True,
                )
        if memory_positions is not None:
            st.markdown("#### 最新价格、趋势和波动")
            st.dataframe(memory_positions, use_container_width=True)
        if memory_decisions is not None:
            st.markdown("#### 默认平衡方案")
            balanced = memory_decisions[memory_decisions["scenario"] == "balanced"]
            if not balanced.empty:
                st.dataframe(
                    balanced[["ticker", "current_shares", "latest_close", "target_weight_pct", "target_shares", "share_delta", "action", "reason"]],
                    hide_index=True, use_container_width=True,
                )
            st.markdown("#### 三种持仓结构对比")
            st.dataframe(
                memory_decisions[["scenario", "scenario_total_target_weight_pct", "ticker", "target_weight_pct", "target_shares", "share_delta", "action"]],
                hide_index=True, use_container_width=True,
            )
        if memory is None:
            st.info("请先导入包含 MU、SKHY、DRAM 的持仓 CSV，并运行完整研究流程。")
        st.warning(
            "默认权重是可修改的风险预算假设，不是回测优化结果。交易前仍需核对基本面、估值、最新 ETF 持仓以及现金/融资情况。"
        )

    with valuation_tab:
        st.markdown("### 个股基本面估值")
        st.caption("DCF 与 PE 是独立的情景估值交叉检查。默认假设是可编辑的研究起点，不是分析师一致预期；系统不发送订单。")
        valuation_symbols = st.multiselect(
            "选择需要研究的股票",
            options=["GOOG", "NVDA", "MU", "SKHY"],
            default=["GOOG", "NVDA", "MU", "SKHY"],
            key="valuation_symbols",
        )
        force_valuation_refresh = st.checkbox(
            "强制刷新财务与行情数据",
            value=False,
            key="force_valuation_refresh",
            help="默认使用不超过 7 天的本地数据缓存。",
        )
        if st.button("运行估值分析", type="primary", disabled=not valuation_symbols, key="run_valuation"):
            command = [sys.executable, str(ROOT / "run_valuation.py"), "--tickers", ",".join(valuation_symbols)]
            if force_valuation_refresh:
                command.append("--refresh")
            try:
                task = start_task(command, repository_root=ROOT, task_type="valuation")
                st.success(f"估值任务已提交：{task['task_id']}。可以继续浏览旧估值结果。")
                st.rerun()
            except TaskAlreadyRunning as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"无法启动估值任务：{type(exc).__name__}: {exc}")

        valuation_status_names = [
            "valuation_summary.csv",
            "valuation_report.md",
            "valuation_fundamentals.json",
            "valuation_assumptions_used.json",
            "valuation_run_manifest.json",
        ]
        st.caption("估值结果文件状态")
        st.dataframe(
            pd.DataFrame([
                {"文件": name, "状态": "已生成" if (valuation_output_dir / name).is_file() else "未生成"}
                for name in valuation_status_names
            ]),
            hide_index=True,
            use_container_width=True,
        )
        valuation_summary = read_csv(valuation_output_dir / "valuation_summary.csv")
        if valuation_summary is not None and not valuation_summary.empty:
            st.markdown("#### 各股票估值对照")
            preferred = [
                "ticker", "current_price", "price_date", "dcf_bear", "dcf_base", "dcf_bull",
                "pe_bear", "pe_base", "pe_bull", "dcf_buy_below", "pe_buy_below",
                "dcf_sell_review_above", "pe_sell_review_above", "dcf_view", "pe_view",
                "overall_confidence", "data_quality",
            ]
            shown = [name for name in preferred if name in valuation_summary.columns]
            st.dataframe(valuation_summary[shown], hide_index=True, use_container_width=True)
            st.caption("价格单位沿用各证券交易币种；买入观察线与卖出复核线来自基准情景和安全边际/溢价假设，不是交易指令。")
            financial_columns = [
                "ticker", "revenue", "gross_profit", "operating_income", "net_income", "ebitda",
                "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin",
                "return_on_equity", "return_on_assets", "trailing_pe", "forward_pe",
            ]
            available_financial_columns = [name for name in financial_columns if name in valuation_summary.columns]
            if available_financial_columns:
                st.markdown("#### 经营与盈利质量")
                st.dataframe(
                    valuation_summary[available_financial_columns],
                    hide_index=True,
                    use_container_width=True,
                )

            selected_report = st.selectbox(
                "查看单只股票研究说明",
                options=valuation_summary["ticker"].astype(str).tolist(),
                key="valuation_report_ticker",
            )
            selected_row = valuation_summary[valuation_summary["ticker"].astype(str) == selected_report].iloc[0]
            st.markdown(f"#### {selected_report} 数据质量与估值限制")
            st.write(f"数据源：{selected_row.get('data_source', 'N/A')}；报价日期：{selected_row.get('price_date', 'N/A')}；财务期间：{selected_row.get('financial_period', 'N/A')}")
            st.write(f"置信度：{selected_row.get('overall_confidence', 'low')}")
            st.warning(str(selected_row.get("limitations", "估值依赖输入假设，请核验数据。")))
            st.markdown("#### 完整估值报告")
            valuation_report_path = valuation_output_dir / "valuation_report.md"
            if valuation_report_path.is_file():
                st.markdown(valuation_report_path.read_text(encoding="utf-8"))
        else:
            st.info("还没有估值结果。选择 GOOG、NVDA、MU 或 SKHY，点击「运行估值分析」开始。")

        st.markdown("#### 修改估值假设")
        st.code("config/valuation_assumptions.json", language="text")
        st.caption("熊/基准/牛市的 FCF 增长、股权资本成本、终值增长率、PE 倍数和安全边际都可编辑。修改后重新运行估值。")

    with files_tab:
        st.markdown("### 研究产物状态")
        output_table = pd.DataFrame(
            [{"文件": name, "存在": "是" if exists else "否"} for name, exists in status.items()]
        )
        st.dataframe(output_table, hide_index=True, use_container_width=True)
        report_path = output_dir / "research_report.md"
        if report_path.exists():
            st.markdown("### 综合研究报告")
            st.markdown(report_path.read_text(encoding="utf-8"))
        st.markdown("### 输出目录")
        st.code(str(OUTPUT_DIR), language="text")
        st.caption("此面板仅读取本地研究文件；它不连接券商、不发送真实订单。公开部署前应增加身份认证和访问控制。")


if __name__ == "__main__":
    main()
