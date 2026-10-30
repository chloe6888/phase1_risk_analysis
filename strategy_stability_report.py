#!/usr/bin/env python3
"""Reproducible daily/weekly/monthly cadence and parameter-stability review."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest_engine import (
    generate_cross_sectional_momentum_weights,
    load_risk_free_yield,
    parse_rebalance_freqs,
    run_backtest,
)
from portfolio_risk import DEFAULT_OUTPUT_DIR, DEFAULT_PRICE_PANEL, load_price_panel
from strategy_validation import parse_int_list, run_parameter_sweep
from walk_forward_validation import run_walk_forward

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "stability_review"


def _pct(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def _num(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def parameter_stability_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Count how often each cadence or momentum parameter was selected."""
    mapping = {
        "lookback": "selected_lookback",
        "skip_recent": "selected_skip_recent",
        "top_n": "selected_top_n",
        "rebalance_freq": "selected_rebalance_freq",
    }
    columns = ["parameter", "value", "folds_selected", "selection_pct"]
    if folds.empty or "status" not in folds:
        return pd.DataFrame(columns=columns)
    valid = folds.loc[folds["status"] == "OK"]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for parameter, column in mapping.items():
        if column not in valid:
            continue
        for value, count in valid[column].value_counts(dropna=False).items():
            rows.append({
                "parameter": parameter,
                "value": str(value),
                "folds_selected": int(count),
                "selection_pct": float(count / len(valid) * 100),
            })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["parameter", "folds_selected", "value"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)


def summarize_grid_by_frequency(grid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rebalance_freq", "combinations_total", "combinations_valid",
        "positive_test_excess_count", "positive_test_excess_pct",
        "median_test_sharpe", "median_test_excess_return",
    ]
    if grid.empty or "rebalance_freq" not in grid:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for frequency, group in grid.groupby("rebalance_freq", sort=False):
        valid = group.loc[group["status"] == "OK"]
        positive = int((valid["test_excess_total_return"] > 0).sum()) if not valid.empty else 0
        rows.append({
            "rebalance_freq": frequency,
            "combinations_total": int(len(group)),
            "combinations_valid": int(len(valid)),
            "positive_test_excess_count": positive,
            "positive_test_excess_pct": positive / len(valid) * 100 if len(valid) else None,
            "median_test_sharpe": float(valid["test_sharpe_ratio"].median()) if len(valid) else None,
            "median_test_excess_return": float(valid["test_excess_total_return"].median()) if len(valid) else None,
        })
    return pd.DataFrame(rows, columns=columns)


def run_stability_analysis(
    adjusted_close: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp,
    benchmark: str = "SPY",
    baseline_lookback: int = 126,
    baseline_skip_recent: int = 21,
    baseline_top_n: int = 2,
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_recents: Sequence[int] = (5, 21),
    top_ns: Sequence[int] = (1, 2, 3),
    rebalance_freqs: Sequence[str] = ("daily", "weekly", "monthly"),
    train_window: int = 252,
    test_window: int = 63,
    selection_metric: str = "sharpe_ratio",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield: pd.Series | None = None,
    skip_walk_forward: bool = False,
) -> dict[str, Any]:
    """Compare cadence on fixed signals, then test the full parameter/frequency grid."""
    if benchmark not in adjusted_close.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")
    allowed = {"daily", "weekly", "monthly"}
    if not rebalance_freqs or any(freq not in allowed for freq in rebalance_freqs):
        raise ValueError("rebalance_freqs must contain daily, weekly, and/or monthly.")
    if len(set(rebalance_freqs)) != len(rebalance_freqs):
        raise ValueError("rebalance_freqs must not contain duplicates.")
    if min(baseline_lookback, baseline_skip_recent, baseline_top_n) <= 0:
        raise ValueError("Baseline momentum parameters must be positive integers.")

    split = pd.Timestamp(test_start)
    if pd.isna(split):
        raise ValueError("test_start must be a valid date.")

    # One signal matrix is shared by all three cadence runs.
    target = generate_cross_sectional_momentum_weights(
        adjusted_close,
        lookback=baseline_lookback,
        skip_recent=baseline_skip_recent,
        top_n=baseline_top_n,
        benchmark=benchmark,
    )

    cadence_rows: list[dict[str, Any]] = []
    new_returns: dict[str, pd.Series] = {}
    old_returns: dict[str, pd.Series] = {}
    new_equity: dict[str, pd.Series] = {}
    old_equity: dict[str, pd.Series] = {}
    benchmark_returns: pd.Series | None = None

    for frequency in rebalance_freqs:
        summary, daily, _, equity = run_backtest(
            adjusted_close,
            target,
            benchmark=benchmark,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            risk_free_rate_annual=risk_free_rate_annual,
            risk_free_yield=risk_free_yield,
            rebalance_freq=frequency,
        )
        strategy = summary["strategy"]
        legacy = summary["legacy_strategy"]
        cadence_rows.append({
            "rebalance_freq": frequency,
            "observations": summary["observations"],
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
            "actual_rebalance_count": summary["actual_rebalance_count"],
            "new_total_return": strategy["total_return"],
            "new_annualized_return": strategy["annualized_return"],
            "new_annualized_volatility": strategy["annualized_volatility"],
            "new_sharpe": strategy["sharpe_ratio"],
            "new_sortino": strategy["sortino_ratio"],
            "new_max_drawdown": strategy["max_drawdown"],
            "legacy_total_return": legacy["total_return"],
            "legacy_sharpe": legacy["sharpe_ratio"],
            "legacy_max_drawdown": legacy["max_drawdown"],
            "new_minus_legacy_total_return": summary["legacy_comparison"]["total_return_delta_new_minus_legacy"],
            "new_minus_legacy_sharpe": summary["legacy_comparison"]["sharpe_delta_new_minus_legacy"],
            "average_daily_turnover": summary["average_daily_turnover"],
            "annualized_turnover": summary["annualized_turnover"],
            "rebalance_cost_drag": float(
                (daily["transaction_cost_return"] - daily["terminal_liquidation_cost_return"]).sum()
            ),
            "total_cost_drag": float(daily["transaction_cost_return"].sum()),
            "terminal_liquidation_cost": summary["terminal_liquidation_cost_return"],
            "terminal_liquidation_turnover": summary["terminal_liquidation_turnover"],
            "risk_free_rate_source": summary["risk_free_rate_source"],
        })
        new_returns[frequency] = daily["strategy_return"]
        old_returns[frequency] = daily["legacy_strategy_return"]
        new_equity[frequency] = equity["strategy_equity"]
        old_equity[frequency] = equity["legacy_strategy_equity"]
        if benchmark_returns is None:
            benchmark_returns = daily["benchmark_return"]

    cadence_metrics = pd.DataFrame(cadence_rows)

    grid, yearly = run_parameter_sweep(
        adjusted_close,
        test_start=split,
        benchmark=benchmark,
        lookbacks=lookbacks,
        skip_recents=skip_recents,
        top_ns=top_ns,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_free_rate_annual=risk_free_rate_annual,
        risk_free_yield=risk_free_yield,
        rebalance_freqs=rebalance_freqs,
    )
    grid_summary = summarize_grid_by_frequency(grid)

    if skip_walk_forward:
        folds = pd.DataFrame()
        oos = pd.DataFrame()
        walk_summary: dict[str, Any] = {
            "status": "SKIPPED_BY_USER",
            "requested_combinations": len(lookbacks) * len(skip_recents) * len(top_ns) * len(rebalance_freqs),
            "folds_total": 0,
            "folds_successful": 0,
            "folds_skipped": 0,
            "oos_observations": 0,
            "selection_warning": "Walk-forward explicitly skipped. Single-split grid is not a substitute.",
        }
    else:
        folds, oos, walk_summary = run_walk_forward(
            adjusted_close,
            benchmark=benchmark,
            lookbacks=lookbacks,
            skip_recents=skip_recents,
            top_ns=top_ns,
            rebalance_freqs=rebalance_freqs,
            train_window=train_window,
            test_window=test_window,
            selection_metric=selection_metric,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            risk_free_rate_annual=risk_free_rate_annual,
            risk_free_yield=risk_free_yield,
        )

    stability = parameter_stability_table(folds)
    daily = pd.DataFrame(new_returns)
    daily = daily.join(pd.DataFrame(old_returns).add_suffix("_legacy"), how="outer")
    if benchmark_returns is not None:
        daily = daily.join(benchmark_returns.rename("benchmark_return"), how="outer")
    equity = pd.DataFrame(new_equity)
    equity = equity.join(pd.DataFrame(old_equity).add_suffix("_legacy"), how="outer")

    return {
        "cadence_metrics": cadence_metrics,
        "cadence_grid": grid,
        "yearly_grid": yearly,
        "frequency_grid_summary": grid_summary,
        "walk_forward_folds": folds,
        "walk_forward_oos": oos,
        "walk_forward_summary": walk_summary,
        "parameter_stability": stability,
        "daily_returns": daily,
        "equity_curves": equity,
    }


def _build_markdown_report(
    cadence: pd.DataFrame,
    grid_summary: pd.DataFrame,
    stability: pd.DataFrame,
    walk: dict[str, Any],
    *,
    test_start: str,
    benchmark: str,
    price_start: str,
    price_end: str,
    row_count: int,
    ticker_count: int,
) -> str:
    lines = [
        "# 策略调仓频率与参数稳定性报告",
        "",
        f"- 行情区间：{price_start} 至 {price_end}；{row_count} 个交易日；{ticker_count} 个标的",
        f"- 基准：{benchmark}",
        f"- 样本外测试起始日（含）：{test_start}",
        "- 新口径：持仓权重漂移、按实际成交额计成本，并计入期末清仓。",
        "- 旧口径：仅保留为历史算法对照，不作为主要结果。",
        "",
        "## 一、日度 / 周度 / 月度调仓比较",
        "",
        "| 调仓频率 | 新口径总收益 | 年化收益 | Sharpe | Sortino | 最大回撤 | 年化换手 | 总成本拖累 | 期末清仓成本 | 旧口径总收益 | 新旧收益差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cadence.itertuples(index=False):
        lines.append(
            f"| {row.rebalance_freq} | {_pct(row.new_total_return)} | {_pct(row.new_annualized_return)} | "
            f"{_num(row.new_sharpe)} | {_num(row.new_sortino)} | {_pct(row.new_max_drawdown)} | "
            f"{_num(row.annualized_turnover, 2)} | {_pct(row.total_cost_drag)} | "
            f"{_pct(row.terminal_liquidation_cost)} | {_pct(row.legacy_total_return)} | "
            f"{_pct(row.new_minus_legacy_total_return)} |"
        )
    lines += [
        "",
        "三种频率共享相同信号参数、标的池、行情区间、基准、无风险利率和成本假设，只有调仓日程不同。新旧对照也使用相同信号与调仓日，只比较持仓核算方式。",
        "",
        "## 二、参数网格的样本外表现",
        "",
        "| 调仓频率 | 有效组合 / 总组合 | 测试期跑赢基准的组合 | 测试期 Sharpe 中位数 | 测试期超额收益中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grid_summary.itertuples(index=False):
        positive_pct = "N/A" if row.positive_test_excess_pct is None or pd.isna(row.positive_test_excess_pct) else f"{row.positive_test_excess_pct:.1f}%"
        lines.append(
            f"| {row.rebalance_freq} | {row.combinations_valid} / {row.combinations_total} | "
            f"{row.positive_test_excess_count} ({positive_pct}) | {_num(row.median_test_sharpe)} | "
            f"{_pct(row.median_test_excess_return)} |"
        )
    lines += [
        "",
        "该表汇总预先设定的所有参数组合，不会自动选出最优参数。不要反复使用测试期结果调参，否则测试期将失去独立性。",
        "",
        "## 三、Walk-forward 参数选择稳定性",
        "",
        f"- 请求参数/频率组合：{walk.get('requested_combinations', 'N/A')}",
        f"- 可用组合：{walk.get('usable_combinations', 'N/A')}",
        f"- 成功折数：{walk.get('folds_successful', 'N/A')} / {walk.get('folds_total', 'N/A')}",
        f"- 样本外观测数：{walk.get('oos_observations', 'N/A')}",
        f"- 样本外区间：{walk.get('first_oos_date', 'N/A')} 至 {walk.get('last_oos_date', 'N/A')}",
        "",
        "| 参数 | 被选取的值 | 选择折数 | 占成功折数比例 |",
        "|---|---:|---:|---:|",
    ]
    if stability.empty:
        lines.append("| N/A | N/A | 0 | N/A |")
    else:
        for row in stability.itertuples(index=False):
            lines.append(
                f"| {row.parameter} | {row.value} | {row.folds_selected} | {row.selection_pct:.1f}% |"
            )
    if "out_of_sample_strategy" in walk:
        strategy = walk["out_of_sample_strategy"]
        base = walk["out_of_sample_benchmark"]
        lines += [
            "",
            "### Walk-forward 汇总样本外结果",
            "",
            "| 指标 | 策略 | 基准 |",
            "|---|---:|---:|",
            f"| 总收益 | {_pct(strategy.get('total_return'))} | {_pct(base.get('total_return'))} |",
            f"| 年化收益 | {_pct(strategy.get('annualized_return'))} | {_pct(base.get('annualized_return'))} |",
            f"| 年化波动率 | {_pct(strategy.get('annualized_volatility'))} | {_pct(base.get('annualized_volatility'))} |",
            f"| Sharpe | {_num(strategy.get('sharpe_ratio'))} | {_num(base.get('sharpe_ratio'))} |",
            f"| 最大回撤 | {_pct(strategy.get('max_drawdown'))} | {_pct(base.get('max_drawdown'))} |",
            f"| 超额总收益 | {_pct(walk.get('out_of_sample_excess_total_return'))} | — |",
        ]
    lines += [
        "",
        "## 四、阅读结论时的限制",
        "",
        "- 不要只选全区间收益最高的频率；同时看测试期表现、最大回撤、换手率、成本和 walk-forward 选择稳定性。",
        "- 若不同折反复切换频率或动量参数，通常表示选择对市场阶段敏感或噪声较大，不能据此认定存在稳定优势。",
        "- 训练期经常被选中的参数仍可能在样本外落后。Walk-forward 是稳健性证据，不是未来收益预测。",
        "- 结论仍受标的池规模、幸存者偏差、公司行为、行情覆盖和成本假设限制。",
        "- 期末清仓按最后可用收盘价近似，不是订单级成交模拟。",
        "",
        "## 五、输出文件",
        "",
        "- cadence_comparison.csv：日/周/月指标及新旧算法比较。",
        "- cadence_daily_returns.csv、cadence_equity_curves.csv：收益与净值序列。",
        "- strategy_parameter_sweep.csv、strategy_yearly_performance.csv：参数×频率网格。",
        "- frequency_grid_summary.csv：各调仓频率的测试期汇总。",
        "- walk_forward_folds.csv、walk_forward_oos_returns.csv、walk_forward_summary.json：折级结果与样本外收益。",
        "- walk_forward_parameter_stability.csv：各参数在训练折中的入选频率。",
        "- stability_run_manifest.json：行情文件哈希、区间和运行假设。",
        "",
    ]
    return "\n".join(lines)


def run_and_save_stability_report(
    *,
    price_path: Path,
    output_dir: Path,
    test_start: str,
    benchmark: str = "SPY",
    baseline_lookback: int = 126,
    baseline_skip_recent: int = 21,
    baseline_top_n: int = 2,
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_recents: Sequence[int] = (5, 21),
    top_ns: Sequence[int] = (1, 2, 3),
    rebalance_freqs: Sequence[str] = ("daily", "weekly", "monthly"),
    train_window: int = 252,
    test_window: int = 63,
    selection_metric: str = "sharpe_ratio",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield_csv: Path | None = None,
    skip_walk_forward: bool = False,
) -> dict[str, Any]:
    prices = load_price_panel(price_path)
    risk_free_yield = load_risk_free_yield(risk_free_yield_csv) if risk_free_yield_csv else None
    results = run_stability_analysis(
        prices,
        test_start=test_start,
        benchmark=benchmark,
        baseline_lookback=baseline_lookback,
        baseline_skip_recent=baseline_skip_recent,
        baseline_top_n=baseline_top_n,
        lookbacks=lookbacks,
        skip_recents=skip_recents,
        top_ns=top_ns,
        rebalance_freqs=rebalance_freqs,
        train_window=train_window,
        test_window=test_window,
        selection_metric=selection_metric,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_free_rate_annual=risk_free_rate_annual,
        risk_free_yield=risk_free_yield,
        skip_walk_forward=skip_walk_forward,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results["cadence_metrics"].to_csv(output_dir / "cadence_comparison.csv", index=False, float_format="%.10f")
    results["cadence_grid"].to_csv(output_dir / "strategy_parameter_sweep.csv", index=False, float_format="%.10f")
    results["yearly_grid"].to_csv(output_dir / "strategy_yearly_performance.csv", index=False, float_format="%.10f")
    results["frequency_grid_summary"].to_csv(output_dir / "frequency_grid_summary.csv", index=False, float_format="%.10f")
    results["walk_forward_folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False, float_format="%.10f")
    results["walk_forward_oos"].to_csv(output_dir / "walk_forward_oos_returns.csv", index_label="Date", float_format="%.10f")
    results["parameter_stability"].to_csv(output_dir / "walk_forward_parameter_stability.csv", index=False, float_format="%.10f")
    results["daily_returns"].to_csv(output_dir / "cadence_daily_returns.csv", index_label="Date", float_format="%.10f")
    results["equity_curves"].to_csv(output_dir / "cadence_equity_curves.csv", index_label="Date", float_format="%.10f")
    with (output_dir / "walk_forward_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results["walk_forward_summary"], handle, ensure_ascii=False, indent=2, allow_nan=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_panel_path": str(price_path),
        "price_panel_sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
        "price_panel_rows": int(len(prices)),
        "price_panel_tickers": list(map(str, prices.columns)),
        "price_panel_start": prices.index.min().date().isoformat(),
        "price_panel_end": prices.index.max().date().isoformat(),
        "benchmark": benchmark,
        "test_start": pd.Timestamp(test_start).date().isoformat(),
        "baseline_parameters": {
            "lookback": baseline_lookback,
            "skip_recent": baseline_skip_recent,
            "top_n": baseline_top_n,
        },
        "grid_parameters": {
            "lookbacks": list(lookbacks),
            "skip_recents": list(skip_recents),
            "top_ns": list(top_ns),
            "rebalance_freqs": list(rebalance_freqs),
        },
        "train_window": train_window,
        "test_window": test_window,
        "selection_metric": selection_metric,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "risk_free_rate_source": "dated annualized yield series" if risk_free_yield_csv else "constant annual rate",
        "risk_free_yield_csv": str(risk_free_yield_csv) if risk_free_yield_csv else None,
        "risk_free_rate_annual_fallback": risk_free_rate_annual,
        "git_commit": commit,
        "walk_forward_skipped": skip_walk_forward,
    }
    with (output_dir / "stability_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)

    report = _build_markdown_report(
        results["cadence_metrics"],
        results["frequency_grid_summary"],
        results["parameter_stability"],
        results["walk_forward_summary"],
        test_start=manifest["test_start"],
        benchmark=benchmark,
        price_start=manifest["price_panel_start"],
        price_end=manifest["price_panel_end"],
        row_count=manifest["price_panel_rows"],
        ticker_count=len(prices.columns),
    )
    (output_dir / "stability_report.md").write_text(report, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare momentum rebalance cadence and parameter stability on a local adjusted-close panel."
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--test-start", required=True, help="Inclusive start date for the single-split test period")
    parser.add_argument("--baseline-lookback", type=int, default=126)
    parser.add_argument("--baseline-skip-recent", type=int, default=21)
    parser.add_argument("--baseline-top-n", type=int, default=2)
    parser.add_argument("--lookbacks", type=parse_int_list, default=[63, 126, 252])
    parser.add_argument("--skip-recents", type=parse_int_list, default=[5, 21])
    parser.add_argument("--top-ns", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--rebalance-freqs", type=parse_rebalance_freqs, default=["daily", "weekly", "monthly"])
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--selection-metric", choices=["sharpe_ratio", "annualized_return", "sortino_ratio"], default="sharpe_ratio")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--risk-free-yield-csv", type=Path)
    parser.add_argument("--skip-walk-forward", action="store_true", help="Run only the parameter grid if history is too short")
    args = parser.parse_args()

    if not args.prices.is_file():
        parser.error(f"Price panel does not exist: {args.prices}")
    if args.risk_free_yield_csv is not None and not args.risk_free_yield_csv.is_file():
        parser.error(f"Risk-free yield CSV does not exist: {args.risk_free_yield_csv}")
    if min(args.baseline_lookback, args.baseline_skip_recent, args.baseline_top_n) <= 0:
        parser.error("Baseline momentum parameters must be positive.")
    if args.train_window < 2 or args.test_window < 2:
        parser.error("Train and test windows must each be at least two observations.")
    if args.transaction_cost_bps < 0 or args.slippage_bps < 0:
        parser.error("Transaction costs and slippage must be non-negative.")

    results = run_and_save_stability_report(
        price_path=args.prices,
        output_dir=args.output_dir,
        test_start=args.test_start,
        benchmark=args.benchmark,
        baseline_lookback=args.baseline_lookback,
        baseline_skip_recent=args.baseline_skip_recent,
        baseline_top_n=args.baseline_top_n,
        lookbacks=args.lookbacks,
        skip_recents=args.skip_recents,
        top_ns=args.top_ns,
        rebalance_freqs=args.rebalance_freqs,
        train_window=args.train_window,
        test_window=args.test_window,
        selection_metric=args.selection_metric,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate,
        risk_free_yield_csv=args.risk_free_yield_csv,
        skip_walk_forward=args.skip_walk_forward,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "cadence_frequencies": results["cadence_metrics"]["rebalance_freq"].tolist(),
        "valid_grid_combinations": int((results["cadence_grid"]["status"] == "OK").sum()),
        "walk_forward_folds_successful": results["walk_forward_summary"].get("folds_successful", 0),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
