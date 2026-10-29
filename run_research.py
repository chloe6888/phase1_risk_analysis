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

from portfolio_risk import DEFAULT_OUTPUT_DIR, parse_scenario
from strategy_validation import parse_int_list

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
        ("total_return", "Total return", True),
        ("annualized_return", "Annualized return", True),
        ("annualized_volatility", "Annualized volatility", True),
        ("sharpe_ratio", "Sharpe ratio", False),
        ("sortino_ratio", "Sortino ratio", False),
        ("max_drawdown", "Max drawdown", True),
    ]
    rows = []
    for key, label, is_percent in specs:
        if key in metrics:
            value = fmt_pct(metrics[key]) if is_percent else fmt_num(metrics[key])
            rows.append(f"| {label} | {value} |")
    return rows


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
) -> str:
    lines = [
        "# Personal Quantitative Research Report",
        "",
        f"- Generated at (UTC): {generated_at}",
        f"- Benchmark: {benchmark}",
        f"- Portfolio weights: {weights}",
        "- Purpose: research and risk assessment; this report is not a buy/sell instruction.",
        "",
    ]
    if data:
        lines += [
            "## Data coverage", "",
            f"- Source: {data.get('source', 'N/A')}",
            f"- Requested dates: {data.get('requested_start', 'N/A')} to {data.get('requested_end_exclusive', 'N/A')} (end exclusive)",
            f"- Last date per ticker: {json.dumps(data.get('data_last_dates', {}), ensure_ascii=False)}", "",
        ]
    if portfolio:
        lines += [
            "## Portfolio risk", "",
            f"- Period: {portfolio.get('start_date', 'N/A')} to {portfolio.get('end_date', 'N/A')}",
            f"- Return observations: {portfolio.get('observations', 'N/A')}", "",
            "| Metric | Value |", "|---|---:|", *metrics_table(portfolio),
            f"| Historical one-period VaR | {fmt_pct(portfolio.get('historical_var_one_period'))} |",
            f"| Historical one-period CVaR | {fmt_pct(portfolio.get('historical_cvar_one_period'))} |", "",
            "VaR/CVaR are historical one-period estimates, not maximum-loss guarantees.", "",
        ]
    if backtest:
        strategy, base = backtest.get("strategy", {}), backtest.get("benchmark", {})
        lines += [
            "## Baseline momentum backtest", "",
            f"- Period: {backtest.get('start_date', 'N/A')} to {backtest.get('end_date', 'N/A')}",
            f"- Observations: {backtest.get('observations', 'N/A')}",
            f"- Cost assumptions: {backtest.get('transaction_cost_bps', 'N/A')} bps transaction costs and {backtest.get('slippage_bps', 'N/A')} bps slippage per traded dollar", "",
            "| Metric | Strategy | Benchmark |", "|---|---:|---:|",
        ]
        for key, label, percentage in [
            ("total_return", "Total return", True), ("annualized_return", "Annualized return", True),
            ("annualized_volatility", "Annualized volatility", True), ("sharpe_ratio", "Sharpe ratio", False),
            ("sortino_ratio", "Sortino ratio", False), ("max_drawdown", "Max drawdown", True),
        ]:
            left = fmt_pct(strategy.get(key)) if percentage else fmt_num(strategy.get(key))
            right = fmt_pct(base.get(key)) if percentage else fmt_num(base.get(key))
            lines.append(f"| {label} | {left} | {right} |")
        lines += [
            "", f"- Excess total return vs benchmark: {fmt_pct(backtest.get('excess_total_return'))}",
            f"- Annualized turnover estimate: {fmt_num(backtest.get('annualized_turnover'), 2)}", "",
            "The current backtest is a vectorized research approximation. It does not fully simulate share quantities, holdings-weight drift, actual fill prices, taxes, all market restrictions, or final liquidation costs.", "",
        ]
    if validation:
        lines += [
            "## Parameter sensitivity", "",
            f"- Requested combinations: {validation.get('requested_combinations', 'N/A')}",
            f"- Valid combinations: {validation.get('successful_combinations', 'N/A')}",
            f"- Skipped combinations: {validation.get('skipped_combinations', 'N/A')}", "",
            "Do not choose a strategy solely on the same test-period results used for tuning; repeatedly testing against a holdout removes its independence.", "",
        ]
    if walk_forward:
        lines += [
            "## Rolling walk-forward evaluation", "",
            f"- Training window: {walk_forward.get('train_window_observations', 'N/A')} return observations",
            f"- Test window: {walk_forward.get('test_window_observations', 'N/A')} return observations",
            f"- Successful folds: {walk_forward.get('folds_successful', 'N/A')} / {walk_forward.get('folds_total', 'N/A')}",
            f"- Out-of-sample observations: {walk_forward.get('oos_observations', 'N/A')}", "",
        ]
        if "out_of_sample_strategy" in walk_forward:
            strategy = walk_forward["out_of_sample_strategy"]
            base = walk_forward["out_of_sample_benchmark"]
            lines += ["| Metric | Walk-forward strategy | Benchmark |", "|---|---:|---:|"]
            for key, label, percentage in [
                ("total_return", "Total return", True), ("annualized_return", "Annualized return", True),
                ("annualized_volatility", "Annualized volatility", True), ("sharpe_ratio", "Sharpe ratio", False),
                ("max_drawdown", "Max drawdown", True),
            ]:
                left = fmt_pct(strategy.get(key)) if percentage else fmt_num(strategy.get(key))
                right = fmt_pct(base.get(key)) if percentage else fmt_num(base.get(key))
                lines.append(f"| {label} | {left} | {right} |")
            lines += ["", f"- Out-of-sample excess total return: {fmt_pct(walk_forward.get('out_of_sample_excess_total_return'))}", ""]
        lines += [
            "Walk-forward validation reduces dependence on a single split but cannot eliminate multiple-testing bias, survivorship bias, or market-regime changes. Keep a final untouched holdout.", "",
        ]
    lines += [
        "## Output files", "",
        "The outputs directory also contains underlying CSV/JSON artifacts: risk contributions, drawdowns, rolling risk, daily returns, applied weights, parameter-sweep rows, yearly results, and walk-forward folds.", "",
        "## Limitations", "",
        "This report summarizes historical observations and assumptions. It is not a forecast, does not place trades, and does not establish that a strategy has positive expected returns.", "",
    ]
    return "\n".join(lines)


def write_report(output_dir: Path, *, weights: str, benchmark: str) -> Path:
    data = read_json(output_dir / "analysis_metadata.json")
    portfolio = read_json(output_dir / "portfolio_risk_summary.json")
    backtest = read_json(output_dir / "backtest_summary.json")
    validation = read_json(output_dir / "strategy_validation_metadata.json")
    walk_forward = read_json(output_dir / "walk_forward_summary.json")
    report = render_report(
        generated_at=datetime.now(timezone.utc).isoformat(), weights=weights, benchmark=benchmark,
        data=data, portfolio=portfolio, backtest=backtest, validation=validation, walk_forward=walk_forward,
    )
    path = output_dir / "research_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_step(title: str, command: list[str]) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full offline/online personal quant research pipeline.")
    parser.add_argument("--weights", required=True, help="Target portfolio weights such as GOOGL=0.6,SPY=0.4")
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
    if args.max_cache_age_hours <= 0:
        parser.error("--max-cache-age-hours must be positive.")
    if args.transaction_cost_bps < 0 or args.slippage_bps < 0:
        parser.error("Transaction costs and slippage must be non-negative.")
    if min(args.backtest_lookback, args.backtest_skip_recent, args.backtest_top_n) <= 0:
        parser.error("Backtest lookback, skip-recent, and top-N must be positive.")
    if args.train_window < 2 or args.test_window < 2:
        parser.error("Train and test windows must be at least two observations.")

    py = sys.executable
    data_cmd = [py, str(ROOT / "phase1_risk_analysis.py"), "--max-cache-age-hours", str(args.max_cache_age_hours)]
    if args.refresh:
        data_cmd.append("--refresh")
    if args.allow_stale_cache:
        data_cmd.append("--allow-stale-cache")
    run_step("1/5 Data download, cache, and validation", data_cmd)

    risk_cmd = [py, str(ROOT / "portfolio_risk.py"), "--weights", args.weights, "--risk-free-rate", str(args.risk_free_rate)]
    if args.portfolio_value is not None:
        risk_cmd += ["--portfolio-value", str(args.portfolio_value)]
    for name, shocks in args.stress_scenario:
        packed = ",".join(f"{ticker}={shock}" for ticker, shock in shocks.items())
        risk_cmd += ["--scenario", f"{name}:{packed}"]
    run_step("2/5 Portfolio risk and stress cases", risk_cmd)

    backtest_cmd = [
        py, str(ROOT / "backtest_engine.py"), "--benchmark", args.benchmark,
        "--lookback", str(args.backtest_lookback), "--skip-recent", str(args.backtest_skip_recent),
        "--top-n", str(args.backtest_top_n), "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--slippage-bps", str(args.slippage_bps), "--risk-free-rate", str(args.risk_free_rate),
    ]
    run_step("3/5 Baseline backtest", backtest_cmd)

    grid_cmd = [
        py, str(ROOT / "strategy_validation.py"), "--benchmark", args.benchmark, "--test-start", args.test_start,
        "--lookbacks", ",".join(map(str, args.lookbacks)), "--skip-recents", ",".join(map(str, args.skip_recents)),
        "--top-ns", ",".join(map(str, args.top_ns)), "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--slippage-bps", str(args.slippage_bps), "--risk-free-rate", str(args.risk_free_rate),
    ]
    run_step("4/5 Parameter sensitivity and single split", grid_cmd)

    if not args.skip_walk_forward:
        walk_cmd = [
            py, str(ROOT / "walk_forward_validation.py"), "--benchmark", args.benchmark,
            "--lookbacks", ",".join(map(str, args.lookbacks)), "--skip-recents", ",".join(map(str, args.skip_recents)),
            "--top-ns", ",".join(map(str, args.top_ns)), "--train-window", str(args.train_window),
            "--test-window", str(args.test_window), "--selection-metric", args.selection_metric,
            "--transaction-cost-bps", str(args.transaction_cost_bps), "--slippage-bps", str(args.slippage_bps),
            "--risk-free-rate", str(args.risk_free_rate),
        ]
        run_step("5/5 Rolling walk-forward validation", walk_cmd)

    report_path = write_report(OUTPUT_DIR, weights=args.weights, benchmark=args.benchmark)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark, "portfolio_weights": args.weights, "single_split_test_start": args.test_start,
        "parameters": {
            "lookbacks": args.lookbacks, "skip_recents": args.skip_recents, "top_ns": args.top_ns,
            "backtest_lookback": args.backtest_lookback, "backtest_skip_recent": args.backtest_skip_recent,
            "backtest_top_n": args.backtest_top_n, "train_window": args.train_window, "test_window": args.test_window,
            "selection_metric": args.selection_metric, "transaction_cost_bps": args.transaction_cost_bps,
            "slippage_bps": args.slippage_bps, "risk_free_rate": args.risk_free_rate,
            "has_portfolio_value": args.portfolio_value is not None,
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
