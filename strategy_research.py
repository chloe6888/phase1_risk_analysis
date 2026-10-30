#!/usr/bin/env python3
"""Comparable multi-strategy research and a training-only overfitting audit.

All candidates use the same price panel, benchmark, split, rebalance schedule,
risk-free series, and transaction-cost assumptions. The selected strategy is
chosen using training data only; holdout data is reported only after selection.
This is an educational research tool, not a forecast or trading system.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd

from backtest_engine import (
    _metric_summary,
    generate_cross_sectional_momentum_weights,
    load_risk_free_yield,
    run_backtest,
)
from portfolio_risk import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRICE_PANEL,
    TRADING_DAYS_PER_YEAR,
    load_price_panel,
)

SUPPORTED_STRATEGIES = (
    "equal_weight",
    "cross_sectional_momentum",
    "time_series_momentum",
    "trend_following",
    "mean_reversion",
    "low_volatility",
)

STRATEGY_DESCRIPTIONS = {
    "equal_weight": "等权持有基准：作为简单配置对照",
    "cross_sectional_momentum": "横截面动量：持有过去表现排名靠前的资产",
    "time_series_momentum": "时间序列动量：持有过去区间收益为正的资产",
    "trend_following": "趋势跟踪：持有价格高于滚动均线且区间动量为正的资产",
    "mean_reversion": "均值回归：持有低于滚动均线幅度最大的资产",
    "low_volatility": "低波动：持有过去区间实现波动率最低的资产",
}


def _validated_prices(adjusted_close: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if not isinstance(adjusted_close, pd.DataFrame) or adjusted_close.empty:
        raise ValueError("Price panel must be a non-empty DataFrame.")
    prices = adjusted_close.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.isna().any() or prices.index.has_duplicates:
        raise ValueError("Price panel must have valid, unique dates.")
    if prices.columns.has_duplicates:
        raise ValueError("Price panel must have unique ticker columns.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' is absent from the price panel.")
    universe = [ticker for ticker in prices.columns if ticker != benchmark]
    if not universe:
        raise ValueError("At least one non-benchmark asset is required.")
    for ticker in prices.columns:
        observed = prices[ticker].dropna()
        if observed.empty:
            raise ValueError(f"{ticker}: no valid prices are available.")
        values = observed.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{ticker}: observed prices must be finite and positive.")
    if len(prices) < 3:
        raise ValueError("At least three price observations are required.")
    return prices


def _equalize(selected: pd.DataFrame) -> pd.DataFrame:
    counts = selected.sum(axis=1).replace(0, np.nan)
    return selected.astype(float).div(counts, axis=0).fillna(0.0)


def build_strategy_weights(
    adjusted_close: pd.DataFrame,
    *,
    strategy: str,
    benchmark: str = "SPY",
    lookback: int = 126,
    skip_recent: int = 21,
    top_n: int = 2,
) -> pd.DataFrame:
    """Build long-only target weights using information available at each close.

    Residual portfolio weight is cash. No strategy can allocate to the benchmark
    as a risky holding because the benchmark is used only for comparison.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strategy}'. Choose from {SUPPORTED_STRATEGIES}.")
    for name, value in (("lookback", lookback), ("top_n", top_n)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(skip_recent, bool) or not isinstance(skip_recent, int) or skip_recent < 0:
        raise ValueError("skip_recent must be a non-negative integer.")

    prices = _validated_prices(adjusted_close, benchmark)
    universe = [ticker for ticker in prices.columns if ticker != benchmark]
    assets = prices[universe]

    if strategy == "equal_weight":
        selected = pd.DataFrame(True, index=prices.index, columns=universe)
        weights = _equalize(selected)
    elif strategy == "cross_sectional_momentum":
        if skip_recent > 0:
            weights = generate_cross_sectional_momentum_weights(
                prices, lookback=lookback, skip_recent=skip_recent,
                top_n=top_n, benchmark=benchmark,
            )
        else:
            scores = assets.div(assets.shift(lookback)).sub(1.0)
            ranks = scores.rank(axis=1, method="first", ascending=False)
            selected = ranks.le(top_n) & scores.notna()
            weights = _equalize(selected)
    elif strategy == "time_series_momentum":
        trailing_return = assets.div(assets.shift(lookback)).sub(1.0)
        weights = _equalize(trailing_return.gt(0) & trailing_return.notna())
    elif strategy == "trend_following":
        moving_average = assets.rolling(lookback, min_periods=lookback).mean()
        trailing_return = assets.div(assets.shift(lookback)).sub(1.0)
        selected = assets.gt(moving_average) & trailing_return.gt(0)
        weights = _equalize(selected & selected.notna())
    elif strategy == "mean_reversion":
        moving_average = assets.rolling(lookback, min_periods=lookback).mean()
        deviation = assets.div(moving_average).sub(1.0)
        eligible = deviation.lt(0) & deviation.notna()
        ranks = deviation.where(eligible).rank(axis=1, method="first", ascending=True)
        weights = _equalize(ranks.le(top_n) & eligible)
    else:  # low_volatility
        returns = assets.pct_change(fill_method=None)
        realized_volatility = returns.rolling(lookback, min_periods=lookback).std()
        ranks = realized_volatility.rank(axis=1, method="first", ascending=True)
        selected = ranks.le(top_n) & realized_volatility.notna()
        weights = _equalize(selected)

    weights = weights.reindex(index=prices.index, columns=universe).fillna(0.0).astype(float)
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError(f"{strategy}: generated weights must be finite and non-negative.")
    if (weights.sum(axis=1) > 1.0 + 1e-9).any():
        raise ValueError(f"{strategy}: generated weights exceed 100%.")
    weights.index.name = "Date"
    weights.columns.name = "asset"
    return weights


def calculate_deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_returns: pd.Series | None = None,
) -> dict[str, float | int | None]:
    """Approximate the Deflated Sharpe Ratio (DSR) for one selected return series.

    The expected-maximum-Sharpe adjustment uses the number of tested candidates.
    It assumes independent trials; correlated strategies violate that assumption,
    so the result is a diagnostic rather than a formal significance guarantee.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if len(values) < 2:
        raise ValueError("At least two return observations are required for DSR.")
    if not np.isfinite(values.to_numpy()).all() or (values < -1).any():
        raise ValueError("Returns must be finite and no lower than -100%.")
    if risk_free_returns is None:
        excess = values
    else:
        rf = pd.to_numeric(risk_free_returns, errors="coerce").reindex(values.index)
        if rf.isna().any() or not np.isfinite(rf.to_numpy(dtype=float)).all():
            raise ValueError("Risk-free returns must cover all observations with finite values.")
        excess = values - rf

    daily_std = float(excess.std(ddof=1))
    if daily_std <= 0 or not math.isfinite(daily_std):
        return {
            "observations": int(len(excess)),
            "n_trials": n_trials,
            "daily_sharpe": None,
            "annualized_sharpe": None,
            "expected_max_daily_sharpe_under_null": None,
            "deflated_sharpe_ratio": None,
            "sample_skewness": None,
            "sample_kurtosis": None,
        }

    daily_sharpe = float(excess.mean() / daily_std)
    sample_skewness = float(excess.skew()) if len(excess) >= 3 else 0.0
    sample_kurtosis = float(excess.kurt() + 3.0) if len(excess) >= 4 else 3.0
    sample_skewness = sample_skewness if math.isfinite(sample_skewness) else 0.0
    sample_kurtosis = sample_kurtosis if math.isfinite(sample_kurtosis) else 3.0

    variance_term = (
        1.0
        - sample_skewness * daily_sharpe
        + ((sample_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    )
    sharpe_variance = variance_term / (len(excess) - 1)
    if not math.isfinite(sharpe_variance) or sharpe_variance <= 0:
        dsr = None
        expected_max = None
    else:
        if n_trials == 1:
            expected_max = 0.0
        else:
            normal = NormalDist()
            euler_gamma = 0.5772156649015329
            p1 = 1.0 - 1.0 / n_trials
            p2 = 1.0 - 1.0 / (n_trials * math.e)
            expected_max = math.sqrt(sharpe_variance) * (
                (1.0 - euler_gamma) * normal.inv_cdf(p1)
                + euler_gamma * normal.inv_cdf(p2)
            )
        z_score = (daily_sharpe - expected_max) / math.sqrt(sharpe_variance)
        dsr = float(NormalDist().cdf(z_score))

    return {
        "observations": int(len(excess)),
        "n_trials": n_trials,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": daily_sharpe * math.sqrt(periods_per_year),
        "expected_max_daily_sharpe_under_null": expected_max,
        "deflated_sharpe_ratio": dsr,
        "sample_skewness": sample_skewness,
        "sample_kurtosis": sample_kurtosis,
    }


def _metric_pair(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate_annual: float,
) -> tuple[dict[str, object], dict[str, object]]:
    strategy_metrics = _metric_summary(
        strategy_returns, periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=risk_free_returns,
    )
    benchmark_metrics = _metric_summary(
        benchmark_returns, periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=risk_free_returns,
    )
    return strategy_metrics, benchmark_metrics


def run_strategy_research(
    adjusted_close: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp,
    benchmark: str = "SPY",
    strategies: Sequence[str] = SUPPORTED_STRATEGIES,
    lookback: int = 126,
    skip_recent: int = 21,
    top_n: int = 2,
    rebalance_freq: str = "monthly",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
    risk_free_yield: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Compare strategy candidates on an identical window and select on train only."""
    if not strategies:
        raise ValueError("At least one strategy must be supplied.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("Strategy names must be unique.")
    unknown = sorted(set(strategies) - set(SUPPORTED_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported strategies: {unknown}")
    if rebalance_freq not in {"daily", "weekly", "monthly"}:
        raise ValueError("rebalance_freq must be daily, weekly, or monthly.")
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative.")
    if not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("slippage_bps must be finite and non-negative.")
    if not math.isfinite(risk_free_rate_annual) or risk_free_rate_annual <= -1:
        raise ValueError("risk_free_rate_annual must be finite and greater than -100%.")

    prices = _validated_prices(adjusted_close, benchmark)
    try:
        split = pd.Timestamp(test_start)
    except (TypeError, ValueError) as exc:
        raise ValueError("test_start must be a valid date.") from exc
    if pd.isna(split):
        raise ValueError("test_start must be a valid date.")

    runs: dict[str, pd.DataFrame] = {}
    first_active_dates: dict[str, str | None] = {}
    for strategy in strategies:
        target = build_strategy_weights(
            prices, strategy=strategy, benchmark=benchmark,
            lookback=lookback, skip_recent=skip_recent, top_n=top_n,
        )
        _, daily, applied_weights, _ = run_backtest(
            prices, target, benchmark=benchmark,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            risk_free_rate_annual=risk_free_rate_annual,
            risk_free_yield=risk_free_yield,
            rebalance_freq=rebalance_freq,
            include_cash_before_first_investment=True,
        )
        active = applied_weights.sum(axis=1).gt(0)
        first_active_dates[strategy] = (
            active[active].index[0].date().isoformat() if active.any() else None
        )
        runs[strategy] = daily

    common_index = runs[strategies[0]].index
    for daily in list(runs.values())[1:]:
        common_index = common_index.intersection(daily.index)
    common_index = common_index.sort_values()
    if len(common_index) < 4:
        raise ValueError("Too few common return observations to compare strategies.")

    benchmark_returns = runs[strategies[0]].reindex(common_index)["benchmark_return"]
    risk_free_returns = runs[strategies[0]].reindex(common_index)["risk_free_return"]
    if benchmark_returns.isna().any() or risk_free_returns.isna().any():
        raise ValueError("Benchmark or risk-free returns have gaps in the common evaluation window.")
    train_dates = common_index[common_index < split]
    test_dates = common_index[common_index >= split]
    if len(train_dates) < 2 or len(test_dates) < 2:
        raise ValueError(
            "test_start must leave at least two common return observations in both train and test periods."
        )

    rows: list[dict[str, object]] = []
    train_returns: dict[str, pd.Series] = {}
    test_return_frames: dict[str, pd.DataFrame] = {}
    train_benchmark = benchmark_returns.reindex(train_dates)
    test_benchmark = benchmark_returns.reindex(test_dates)
    train_rf = risk_free_returns.reindex(train_dates)
    test_rf = risk_free_returns.reindex(test_dates)
    benchmark_train_metrics = _metric_summary(
        train_benchmark, periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=train_rf,
    )
    benchmark_test_metrics = _metric_summary(
        test_benchmark, periods_per_year, risk_free_rate_annual,
        risk_free_daily_returns=test_rf,
    )

    for strategy in strategies:
        daily = runs[strategy].reindex(common_index)
        train = daily.loc[train_dates, "strategy_return"]
        test = daily.loc[test_dates, "strategy_return"]
        if train.isna().any() or test.isna().any():
            raise ValueError(f"{strategy}: strategy returns have gaps in the common evaluation window.")
        train_metrics, _ = _metric_pair(
            train, train_benchmark, train_rf, periods_per_year, risk_free_rate_annual,
        )
        test_metrics, test_benchmark_metrics = _metric_pair(
            test, test_benchmark, test_rf, periods_per_year, risk_free_rate_annual,
        )
        train_returns[strategy] = train
        test_return_frames[strategy] = daily.loc[test_dates, [
            "strategy_return", "benchmark_return", "risk_free_return",
        ]]
        benchmark_growth = float((1.0 + test_benchmark).prod())
        strategy_growth = float((1.0 + test).prod())
        relative_return = strategy_growth / benchmark_growth - 1.0 if benchmark_growth > 0 else None
        rows.append({
            "strategy": strategy,
            "description": STRATEGY_DESCRIPTIONS[strategy],
            "first_active_date": first_active_dates[strategy],
            "train_observations": int(len(train)),
            "test_observations": int(len(test)),
            "train_total_return": train_metrics["total_return"],
            "train_annualized_return": train_metrics["annualized_return"],
            "train_sharpe_ratio": train_metrics["sharpe_ratio"],
            "train_max_drawdown": train_metrics["max_drawdown"],
            "test_total_return": test_metrics["total_return"],
            "test_annualized_return": test_metrics["annualized_return"],
            "test_sharpe_ratio": test_metrics["sharpe_ratio"],
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_benchmark_total_return": test_benchmark_metrics["total_return"],
            "test_relative_return_vs_benchmark": relative_return,
            "selected_on_training_only": False,
        })

    def selection_key(row: dict[str, object]) -> tuple[float, float, float]:
        sharpe = row["train_sharpe_ratio"]
        annualized = row["train_annualized_return"]
        drawdown = row["train_max_drawdown"]
        return (
            float(sharpe) if sharpe is not None and math.isfinite(float(sharpe)) else -math.inf,
            float(annualized) if annualized is not None and math.isfinite(float(annualized)) else -math.inf,
            float(drawdown) if drawdown is not None and math.isfinite(float(drawdown)) else -math.inf,
        )

    winner = max(rows, key=selection_key)
    winner_name = str(winner["strategy"])
    winner["selected_on_training_only"] = True
    table = pd.DataFrame(rows).sort_values(
        ["selected_on_training_only", "train_sharpe_ratio"], ascending=[False, False],
        na_position="last", kind="stable",
    ).reset_index(drop=True)

    dsr = calculate_deflated_sharpe_ratio(
        train_returns[winner_name],
        n_trials=len(strategies),
        periods_per_year=periods_per_year,
        risk_free_returns=train_rf,
    )
    winner_test = test_return_frames[winner_name].copy()
    winner_test["excess_return_vs_benchmark"] = (
        winner_test["strategy_return"] - winner_test["benchmark_return"]
    )
    winner_test.index.name = "Date"

    summary: dict[str, object] = {
        "benchmark": benchmark,
        "train_start": train_dates.min().date().isoformat(),
        "train_end": train_dates.max().date().isoformat(),
        "test_start": test_dates.min().date().isoformat(),
        "test_end": test_dates.max().date().isoformat(),
        "train_observations": int(len(train_dates)),
        "test_observations": int(len(test_dates)),
        "candidate_count": int(len(strategies)),
        "candidate_strategies": list(strategies),
        "selection_metric": "training_period_sharpe_ratio",
        "selected_strategy": winner_name,
        "selected_strategy_description": STRATEGY_DESCRIPTIONS[winner_name],
        "selected_train_sharpe_ratio": winner["train_sharpe_ratio"],
        "selected_train_metrics": {
            "total_return": winner["train_total_return"],
            "annualized_return": winner["train_annualized_return"],
            "sharpe_ratio": winner["train_sharpe_ratio"],
            "max_drawdown": winner["train_max_drawdown"],
        },
        "selected_test_metrics": {
            "total_return": winner["test_total_return"],
            "annualized_return": winner["test_annualized_return"],
            "sharpe_ratio": winner["test_sharpe_ratio"],
            "max_drawdown": winner["test_max_drawdown"],
            "benchmark_total_return": winner["test_benchmark_total_return"],
            "relative_return_vs_benchmark": winner["test_relative_return_vs_benchmark"],
        },
        "benchmark_train_metrics": benchmark_train_metrics,
        "benchmark_test_metrics": benchmark_test_metrics,
        "deflated_sharpe_audit_training_only": dsr,
        "assumptions": {
            "rebalance_freq": rebalance_freq,
            "lookback": lookback,
            "skip_recent": skip_recent,
            "top_n": top_n,
            "transaction_cost_bps": float(transaction_cost_bps),
            "slippage_bps": float(slippage_bps),
            "risk_free_rate_annual": float(risk_free_rate_annual),
            "risk_free_yield_source": "dated series" if risk_free_yield is not None else "constant annual rate",
            "cash_earns_zero_return": True,
            "benchmark_is_excluded_from_strategy_holdings": True,
        },
        "limitations": [
            "策略仅使用价格数据；未包含财务质量、估值、行业和事件数据。",
            "所有策略均为简化的多头基线，不代表完整的机构策略实现。",
            "Deflated Sharpe Ratio 使用候选数量近似试验次数，并假设试验独立；策略相关性会影响修正准确性。",
            "单次训练/测试切分不能消除重复试验和测试集窥探；测试区间一旦用于调参，就不再是独立留出集。",
            "回测仍不模拟真实订单股数、盘中成交、市场冲击、税费、停牌和涨跌幅限制。",
            "研究结果不构成投资建议，不连接券商，也不执行真实交易。",
        ],
        "warnings": [
            "策略胜出者只根据训练期 Sharpe 选择；测试期结果没有参与选择。",
            "请保留未被反复查看的最终留出区间，并结合滚动 Walk-forward 结果审查策略。",
        ],
    }
    return table, summary, winner_test


def parse_strategy_list(value: str) -> list[str]:
    strategies = [part.strip() for part in value.split(",") if part.strip()]
    if not strategies:
        raise argparse.ArgumentTypeError("Provide at least one strategy.")
    if len(set(strategies)) != len(strategies):
        raise argparse.ArgumentTypeError("Strategy names must not contain duplicates.")
    unknown = sorted(set(strategies) - set(SUPPORTED_STRATEGIES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported strategies: {unknown}. Supported: {list(SUPPORTED_STRATEGIES)}"
        )
    return strategies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare simple strategy families and audit selection bias; no orders are placed."
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--test-start", required=True, help="Inclusive holdout test start date.")
    parser.add_argument("--strategies", type=parse_strategy_list, default=list(SUPPORTED_STRATEGIES))
    parser.add_argument("--lookback", type=int, default=126)
    parser.add_argument("--skip-recent", type=int, default=21)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--rebalance-freq", choices=["daily", "weekly", "monthly"], default="monthly")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--risk-free-yield-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    prices = load_price_panel(args.prices)
    risk_free_yield = (
        load_risk_free_yield(args.risk_free_yield_csv)
        if args.risk_free_yield_csv else None
    )
    comparison, summary, selected_oos = run_strategy_research(
        prices,
        test_start=args.test_start,
        benchmark=args.benchmark,
        strategies=args.strategies,
        lookback=args.lookback,
        skip_recent=args.skip_recent,
        top_n=args.top_n,
        rebalance_freq=args.rebalance_freq,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        risk_free_rate_annual=args.risk_free_rate,
        risk_free_yield=risk_free_yield,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        args.output_dir / "strategy_comparison.csv", index=False, float_format="%.10f"
    )
    selected_oos.to_csv(
        args.output_dir / "strategy_selected_oos_returns.csv", float_format="%.10f"
    )
    (args.output_dir / "strategy_research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 多策略比较与过拟合审计",
        "",
        f"- 基准：{summary['benchmark']}",
        f"- 训练区间：{summary['train_start']} 至 {summary['train_end']}",
        f"- 测试区间：{summary['test_start']} 至 {summary['test_end']}",
        f"- 候选策略数：{summary['candidate_count']}",
        f"- 仅按训练期 Sharpe 选择：{summary['selected_strategy']}",
        f"- 训练期 Sharpe：{summary['selected_train_sharpe_ratio']}",
        f"- 测试期 Sharpe：{summary['selected_test_metrics']['sharpe_ratio']}",
        f"- 测试期相对基准收益：{summary['selected_test_metrics']['relative_return_vs_benchmark']}",
        f"- 训练期 Deflated Sharpe Ratio：{summary['deflated_sharpe_audit_training_only']['deflated_sharpe_ratio']}",
        "",
        "## 候选策略比较",
        "",
        "~~~text",
        comparison.to_string(index=False),
        "~~~",
        "",
        "## 重要限制",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    (args.output_dir / "strategy_research_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
