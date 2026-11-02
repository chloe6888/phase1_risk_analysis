"""Compare current positions with balanced, conservative and minimum-variance allocations.

All inputs and reports are local. This is research tooling only: it does not send
orders, infer missing cash, or guarantee that historical risk estimates persist.
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
from portfolio_optimizer import optimize_portfolio
from portfolio_risk import calculate_portfolio_risk, load_price_panel
from run_personal_portfolio import (
    DEFAULT_CONFIG,
    DEFAULT_HOLDINGS,
    load_universe,
    validate_holdings_file,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CLOSE = ROOT / "data" / "raw" / "close_price_panel.csv"
DEFAULT_ADJUSTED = ROOT / "data" / "raw" / "adjusted_close_panel.csv"
DEFAULT_PROFILES = ROOT / "risk_profiles" / "personal_allocation_profiles.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "allocation_comparison"


def _read_profiles(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read allocation profiles: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Allocation profiles must be a schema_version=1 JSON object.")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not {"balanced", "conservative"}.issubset(profiles):
        raise ValueError("Allocation profiles must include balanced and conservative definitions.")
    minimum = int(payload.get("minimum_common_observations", 60))
    train = int(payload.get("walk_forward_train_observations", minimum))
    test = int(payload.get("walk_forward_test_observations", 21))
    if minimum < 2 or train < minimum or test < 2:
        raise ValueError("Profile history settings must satisfy 2 <= minimum <= train and test >= 2.")
    payload["minimum_common_observations"] = minimum
    payload["walk_forward_train_observations"] = train
    payload["walk_forward_test_observations"] = test
    return payload


def _profile_arguments(profile: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "risky_budget",
        "default_asset_max_weight",
        "group_max_weights",
        "subgroup_max_weights",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Allocation profile is missing fields: {missing}")
    for key in ("group_max_weights", "subgroup_max_weights"):
        if not isinstance(profile[key], dict):
            raise ValueError(f"Allocation profile {key} must be a JSON object.")
    return {
        "risky_budget": float(profile["risky_budget"]),
        "default_asset_max_weight": float(profile["default_asset_max_weight"]),
        "group_max_weights": {str(k): float(v) for k, v in profile["group_max_weights"].items()},
        "subgroup_max_weights": {str(k): float(v) for k, v in profile["subgroup_max_weights"].items()},
    }


def _current_market_values(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    cash: float | None,
    as_of_date: date | str | None,
    max_price_age_calendar_days: int = 7,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any]]:
    if cash is not None and (not math.isfinite(float(cash)) or float(cash) < 0):
        raise ValueError("cash must be finite and non-negative.")
    reference_date = date.today() if as_of_date is None else date.fromisoformat(str(as_of_date))
    prices = close_prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Raw close-price panel must have valid, unique dates.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")

    rows: list[dict[str, Any]] = []
    for row in holdings.itertuples(index=False):
        ticker = str(row.ticker).upper()
        if ticker not in prices.columns:
            raise ValueError(f"{ticker}: raw close-price panel does not contain this holding.")
        series = prices[ticker].dropna()
        if series.empty:
            raise ValueError(f"{ticker}: no valid raw close price.")
        last_date = pd.Timestamp(series.index[-1]).date()
        age = (reference_date - last_date).days
        if age < 0:
            raise ValueError(f"{ticker}: raw close date is later than the analysis date.")
        if age > max_price_age_calendar_days:
            raise ValueError(
                f"{ticker}: last close is {age} days old; maximum allowed age is {max_price_age_calendar_days}."
            )
        price = float(series.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"{ticker}: last raw close must be finite and positive.")
        market_value = float(row.shares) * price
        rows.append({
            "ticker": ticker,
            "shares": float(row.shares),
            "last_close": price,
            "price_date": last_date.isoformat(),
            "market_value": market_value,
        })

    positions = pd.DataFrame(rows)
    listed_value = float(positions["market_value"].sum())
    cash_value = 0.0 if cash is None else float(cash)
    denominator = listed_value + cash_value
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("Imported listed value plus cash must be finite and positive.")
    positions["weight"] = positions["market_value"] / denominator
    weights = dict(zip(positions["ticker"], positions["weight"].astype(float)))
    if cash_value > 0:
        weights["CASH"] = cash_value / denominator
    summary = {
        "as_of_date": reference_date.isoformat(),
        "latest_price_dates": dict(zip(positions["ticker"], positions["price_date"])),
        "listed_positions_market_value": listed_value,
        "cash_value": cash_value,
        "cash_source": "user_provided" if cash is not None else "not_provided_assumed_zero",
        "actual_weight_denominator": "listed positions plus explicitly supplied cash" if cash is not None else "listed positions only; cash not supplied",
        "portfolio_denominator": denominator,
    }
    return weights, positions, summary


def _make_target_weights(
    optimizer_output: pd.DataFrame,
    *,
    column: str,
    risky_budget: float,
) -> dict[str, float]:
    if column not in optimizer_output:
        raise ValueError(f"Optimizer output is missing {column!r}.")
    weights = {
        str(row.ticker): float(getattr(row, column))
        for row in optimizer_output.itertuples(index=False)
    }
    residual_cash = max(0.0, 1.0 - risky_budget)
    if residual_cash > 1e-10:
        weights["CASH"] = residual_cash
    total = float(sum(weights.values()))
    if not math.isclose(total, 1.0, abs_tol=1e-7):
        raise RuntimeError(f"Target allocation weights sum to {total:.10f}, not 1.0.")
    return weights


def _scenario_exposures(
    weights: Mapping[str, float],
    holdings: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    group: dict[str, float] = {}
    subgroup: dict[str, float] = {}
    for row in holdings.itertuples(index=False):
        ticker = str(row.ticker)
        weight = float(weights.get(ticker, 0.0))
        parent = str(row.risk_group)
        child = str(row.subgroup)
        group[parent] = group.get(parent, 0.0) + weight
        subgroup[child] = subgroup.get(child, 0.0) + weight
    return group, subgroup


def _metric_row(
    name: str,
    label: str,
    weights: Mapping[str, float],
    adjusted_prices: pd.DataFrame,
    *,
    turnover_vs_actual: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    panel = adjusted_prices.copy()
    panel["CASH"] = 1.0
    summary, contributions, drawdown, _ = calculate_portfolio_risk(panel, weights)
    row = {
        "scenario": name,
        "scenario_label": label,
        "risky_asset_weight": float(sum(value for key, value in weights.items() if key != "CASH")),
        "cash_weight": float(weights.get("CASH", 0.0)),
        "turnover_vs_actual_one_way": float(turnover_vs_actual),
        "observations": int(summary["observations"]),
        "start_date": summary["start_date"],
        "end_date": summary["end_date"],
        "annualized_return": summary["annualized_return"],
        "annualized_volatility": summary["annualized_volatility"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "sortino_ratio": summary["sortino_ratio"],
        "max_drawdown": summary["max_drawdown"],
        "historical_var_one_period": summary["historical_var_one_period"],
        "historical_cvar_one_period": summary["historical_cvar_one_period"],
        "risk_model_note": summary["risk_model_note"],
    }
    contrib = contributions.reset_index().rename(columns={"asset": "ticker"})
    contrib.insert(0, "scenario", name)
    drawdown_frame = drawdown.reset_index()
    drawdown_frame.insert(0, "scenario", name)
    return row, contrib, drawdown_frame


def walk_forward_minimum_variance(
    adjusted_prices: pd.DataFrame,
    *,
    profile: Mapping[str, Any],
    train_observations: int = 60,
    test_observations: int = 21,
    minimum_training_observations: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expanding time-ordered OOS test: fit before each test block, then hold."""
    prices = adjusted_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    tickers = [str(ticker) for ticker in prices.columns]
    expected_minimum = max(train_observations, minimum_training_observations)
    if train_observations < expected_minimum:
        raise ValueError("train_observations cannot be below minimum_training_observations.")
    if test_observations < 2:
        raise ValueError("test_observations must be at least 2.")

    args = _profile_arguments(profile)
    rows: list[dict[str, Any]] = []
    fold = 0
    anchor = train_observations
    while anchor < len(prices) - 1:
        train_start = max(0, anchor - train_observations)
        train_panel = prices.iloc[train_start:anchor + 1]
        train_returns_count = len(train_panel.pct_change(fill_method=None).dropna(how="any"))
        if train_returns_count < expected_minimum:
            anchor += test_observations
            continue
        test_end = min(anchor + test_observations, len(prices) - 1)
        test_panel = prices.iloc[anchor:test_end + 1]
        test_returns = test_panel.pct_change(fill_method=None).dropna(how="any")
        if len(test_returns) < 2:
            break

        fitted, fitted_summary = optimize_portfolio(
            train_panel,
            risky_budget=args["risky_budget"],
            default_asset_max_weight=args["default_asset_max_weight"],
            groups=profile["groups"],
            group_max_weights=args["group_max_weights"],
            subgroups=profile["subgroups"],
            subgroup_max_weights=args["subgroup_max_weights"],
            minimum_observations=expected_minimum,
        )
        optimized_weights = _make_target_weights(
            fitted, column="weight", risky_budget=args["risky_budget"]
        )
        balanced_weights = _make_target_weights(
            fitted, column="constrained_equal_weight", risky_budget=args["risky_budget"]
        )
        for strategy, weights in (
            ("minimum_variance", optimized_weights),
            ("balanced_capped_equal_weight", balanced_weights),
        ):
            risky_weights = pd.Series({ticker: weights.get(ticker, 0.0) for ticker in tickers})
            period_returns = test_returns[tickers].dot(risky_weights)
            wealth = (1.0 + period_returns).cumprod()
            running_peak = wealth.cummax().clip(lower=1.0)
            ann_vol = (
                float(period_returns.std(ddof=1) * math.sqrt(252))
                if len(period_returns) >= 2 else None
            )
            fold += 1
            rows.append({
                "fold": (fold + 1) // 2,
                "strategy": strategy,
                "train_start": pd.Timestamp(train_panel.index.min()).date().isoformat(),
                "train_end": pd.Timestamp(train_panel.index.max()).date().isoformat(),
                "test_start": pd.Timestamp(period_returns.index.min()).date().isoformat(),
                "test_end": pd.Timestamp(period_returns.index.max()).date().isoformat(),
                "test_observations": int(len(period_returns)),
                "total_return": float((1.0 + period_returns).prod() - 1.0),
                "annualized_volatility": ann_vol,
                "max_drawdown": float((wealth / running_peak - 1.0).min()),
                "train_observations": int(fitted_summary["observations"]),
            })
        anchor = test_end

    frame = pd.DataFrame(rows)
    unique_folds = int(frame["fold"].nunique()) if not frame.empty else 0
    total_test_obs = int(
        frame[frame["strategy"] == "minimum_variance"]["test_observations"].sum()
    ) if not frame.empty else 0
    status = (
        "insufficient_history" if unique_folds == 0
        else "limited_evidence" if unique_folds < 3 or total_test_obs < 60
        else "multiple_oos_folds"
    )
    summary = {
        "status": status,
        "fold_count": unique_folds,
        "total_out_of_sample_observations": total_test_obs,
        "train_observations_per_fold": int(train_observations),
        "test_observations_per_fold_target": int(test_observations),
        "no_lookahead": True,
        "note": (
            "Each optimizer fit uses only prices dated on or before the prior close to the test block. "
            "The fitted weights are held through the test block. Fewer than three folds or 60 OOS "
            "observations is labelled limited evidence; transaction costs and slippage are not included."
        ),
    }
    return frame, summary


def compare_allocations(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    adjusted_prices: pd.DataFrame,
    profiles: Mapping[str, Any],
    *,
    cash: float | None = None,
    as_of_date: date | str | None = None,
    max_price_age_calendar_days: int = 7,
) -> dict[str, Any]:
    if holdings.empty:
        raise ValueError("Holdings cannot be empty.")
    tickers = [str(item).strip().upper() for item in holdings["ticker"]]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Holdings must contain unique tickers.")
    missing_adjusted = sorted(set(tickers) - set(map(str, adjusted_prices.columns)))
    missing_close = sorted(set(tickers) - set(map(str, close_prices.columns)))
    if missing_adjusted or missing_close:
        raise ValueError(
            f"Price coverage is incomplete; missing adjusted={missing_adjusted}, raw={missing_close}."
        )

    minimum = int(profiles["minimum_common_observations"])
    adjusted = adjusted_prices.loc[:, tickers].copy()
    adjusted.index = pd.to_datetime(adjusted.index)
    daily_common = adjusted.pct_change(fill_method=None).dropna(how="any")
    if len(daily_common) < minimum:
        raise ValueError(
            f"Only {len(daily_common)} common return observations are available; at least {minimum} are required."
        )

    actual_weights, current_positions, current_summary = _current_market_values(
        holdings,
        close_prices,
        cash=cash,
        as_of_date=as_of_date,
        max_price_age_calendar_days=max_price_age_calendar_days,
    )

    balanced = _profile_arguments(profiles["profiles"]["balanced"])
    conservative = _profile_arguments(profiles["profiles"]["conservative"])
    parent_groups = {
        str(row.ticker): str(row.risk_group).strip()
        for row in holdings.itertuples(index=False)
    }
    child_groups = {
        str(row.ticker): str(row.subgroup).strip()
        for row in holdings.itertuples(index=False)
    }

    balanced_fit, balanced_fit_summary = optimize_portfolio(
        adjusted,
        risky_budget=balanced["risky_budget"],
        default_asset_max_weight=balanced["default_asset_max_weight"],
        groups=parent_groups,
        group_max_weights=balanced["group_max_weights"],
        subgroups=child_groups,
        subgroup_max_weights=balanced["subgroup_max_weights"],
        minimum_observations=minimum,
    )
    conservative_fit, conservative_fit_summary = optimize_portfolio(
        adjusted,
        risky_budget=conservative["risky_budget"],
        default_asset_max_weight=conservative["default_asset_max_weight"],
        groups=parent_groups,
        group_max_weights=conservative["group_max_weights"],
        subgroups=child_groups,
        subgroup_max_weights=conservative["subgroup_max_weights"],
        minimum_observations=minimum,
    )

    target_weights = {
        "actual_holdings": actual_weights,
        "balanced_capped_equal_weight": _make_target_weights(
            balanced_fit, column="constrained_equal_weight", risky_budget=balanced["risky_budget"]
        ),
        "balanced_minimum_variance": _make_target_weights(
            balanced_fit, column="weight", risky_budget=balanced["risky_budget"]
        ),
        "conservative_capped_equal_weight": _make_target_weights(
            conservative_fit, column="constrained_equal_weight", risky_budget=conservative["risky_budget"]
        ),
    }
    labels = {
        "actual_holdings": "当前实际持仓",
        "balanced_capped_equal_weight": "平衡型（受约束等权）",
        "balanced_minimum_variance": "平衡型（历史最小方差）",
        "conservative_capped_equal_weight": "保守型（受约束等权）",
    }

    metric_rows: list[dict[str, Any]] = []
    contribution_frames: list[pd.DataFrame] = []
    drawdown_frames: list[pd.DataFrame] = []
    weights_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for name, weights in target_weights.items():
        union = set(weights) | set(actual_weights)
        turnover = 0.5 * sum(
            abs(float(weights.get(ticker, 0.0)) - float(actual_weights.get(ticker, 0.0)))
            for ticker in union
        )
        metrics, contributions, drawdown = _metric_row(
            name, labels[name], weights, adjusted, turnover_vs_actual=turnover
        )
        groups, subgroups = _scenario_exposures(weights, holdings)
        metrics["group_exposures"] = groups
        metrics["subgroup_exposures"] = subgroups
        metrics["turnover_note"] = (
            "One-way turnover versus current holdings; actual cash is unknown and treated as zero when not supplied."
        )
        metric_rows.append(metrics)
        contribution_frames.append(contributions)
        drawdown_frames.append(drawdown)
        for ticker, weight in sorted(weights.items()):
            weights_rows.append({
                "scenario": name,
                "scenario_label": labels[name],
                "ticker": ticker,
                "target_weight": float(weight),
            })
        for group, weight in groups.items():
            exposure_rows.append({"scenario": name, "exposure_type": "group", "name": group, "weight": weight})
        for subgroup, weight in subgroups.items():
            exposure_rows.append({"scenario": name, "exposure_type": "subgroup", "name": subgroup, "weight": weight})

    walk_forward, walk_forward_summary = walk_forward_minimum_variance(
        adjusted,
        profile={
            **profiles["profiles"]["balanced"],
            "groups": parent_groups,
            "subgroups": child_groups,
        },
        train_observations=int(profiles["walk_forward_train_observations"]),
        test_observations=int(profiles["walk_forward_test_observations"]),
        minimum_training_observations=minimum,
    )

    return {
        "summary": {
            **current_summary,
            "position_count": int(len(holdings)),
            "common_return_observations": int(len(daily_common)),
            "common_start_date": pd.Timestamp(daily_common.index.min()).date().isoformat(),
            "common_end_date": pd.Timestamp(daily_common.index.max()).date().isoformat(),
            "balanced_profile_fit": balanced_fit_summary,
            "conservative_profile_fit": conservative_fit_summary,
            "walk_forward": walk_forward_summary,
            "profile_notes": profiles.get("notes", []),
            "methodology": [
                "Actual weights use current raw Close multiplied by confirmed shares; if cash is omitted, the actual denominator is only the imported securities.",
                "Balanced and conservative targets are constrained equal-weight allocations under asset, parent-group, subgroup and total-risk-budget caps.",
                "Minimum-variance uses historical sample covariance under the balanced profile; this is not an expected-return or valuation model.",
                "Target residual is labelled as cash and contributes zero return; no actual cash balance is inferred.",
                "Risk metrics use adjusted-price common complete-case daily returns with static target weights rebalanced each observation; this is not a trade-by-trade backtest.",
                "Walk-forward tests fit only on prior data and hold weights out of sample; few folds are explicitly labelled limited evidence.",
                "ETF look-through is not part of this weight optimizer; use the separate portfolio risk report to inspect underlying issuer overlap."
            ],
        },
        "current_positions": current_positions,
        "metrics": pd.DataFrame(metric_rows),
        "weights": pd.DataFrame(weights_rows),
        "exposures": pd.DataFrame(exposure_rows),
        "risk_contributions": pd.concat(contribution_frames, ignore_index=True),
        "drawdowns": pd.concat(drawdown_frames, ignore_index=True),
        "walk_forward": walk_forward,
    }


def write_allocation_report(result: Mapping[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = dict(result["summary"])
    (target / "allocation_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for key, filename in (
        ("current_positions", "allocation_current_positions.csv"),
        ("metrics", "allocation_scenario_metrics.csv"),
        ("weights", "allocation_scenario_weights.csv"),
        ("exposures", "allocation_scenario_exposures.csv"),
        ("risk_contributions", "allocation_risk_contributions.csv"),
        ("drawdowns", "allocation_drawdowns.csv"),
        ("walk_forward", "allocation_walk_forward.csv"),
    ):
        result[key].to_csv(target / filename, index=False, float_format="%.10f")

    metrics = result["metrics"]
    lines = [
        "# 个人组合资产配置比较",
        "",
        f"- 分析日期：{summary['as_of_date']}",
        f"- 持仓数量：{summary['position_count']}",
        f"- 当前现金口径：{summary['actual_weight_denominator']}",
        f"- 当前证券市值：{summary['listed_positions_market_value']:,.2f}",
        f"- 共同历史：{summary['common_start_date']} 至 {summary['common_end_date']}",
        f"- 共同日收益观测：{summary['common_return_observations']}",
        "",
        "## 方案比较",
        "",
        "| 方案 | 风险资产比例 | 现金比例 | 年化波动率 | 最大回撤 | 历史年化收益 | Sharpe | 对当前组合单边换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        def fmt(value: Any) -> str:
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.2%}"
        sharpe = "n/a" if row.sharpe_ratio is None or pd.isna(row.sharpe_ratio) else f"{float(row.sharpe_ratio):.2f}"
        lines.append(
            f"| {row.scenario_label} | {fmt(row.risky_asset_weight)} | {fmt(row.cash_weight)} | "
            f"{fmt(row.annualized_volatility)} | {fmt(row.max_drawdown)} | "
            f"{fmt(row.annualized_return)} | {sharpe} | {fmt(row.turnover_vs_actual_one_way)} |"
        )

    lines += [
        "",
        "## 目标配置权重",
        "",
    ]
    weights = result["weights"]
    for scenario in weights["scenario"].drop_duplicates():
        scenario_label = str(weights.loc[weights["scenario"] == scenario, "scenario_label"].iloc[0])
        lines += [f"### {scenario_label}", ""]
        for row in weights.loc[weights["scenario"] == scenario].sort_values("target_weight", ascending=False).itertuples(index=False):
            lines.append(f"- {row.ticker}: {row.target_weight:.2%}")
        lines.append("")

    wf = summary["walk_forward"]
    lines += [
        "## Walk-forward 样本外检验",
        "",
        f"- 状态：{wf['status']}",
        f"- 样本外折数：{wf['fold_count']}",
        f"- 样本外日收益观测：{wf['total_out_of_sample_observations']}",
        f"- 说明：{wf['note']}",
        "",
        "## 方法说明与限制",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["methodology"])
    lines.extend(f"- {item}" for item in summary.get("profile_notes", []))
    lines.append("")
    (target / "personal_allocation_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run_allocation_comparison(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    holdings_path: str | Path = DEFAULT_HOLDINGS,
    close_prices_path: str | Path = DEFAULT_CLOSE,
    adjusted_prices_path: str | Path = DEFAULT_ADJUSTED,
    profiles_path: str | Path = DEFAULT_PROFILES,
    output_dir: str | Path = DEFAULT_OUTPUT,
    cash: float | None = None,
    as_of_date: str | date | None = None,
) -> dict[str, Any]:
    universe = load_universe(config_path)
    holdings = validate_holdings_file(holdings_path, universe)
    close = load_price_panel(close_prices_path)
    adjusted = load_price_panel(adjusted_prices_path)
    profiles = _read_profiles(profiles_path)
    result = compare_allocations(
        holdings,
        close,
        adjusted,
        profiles,
        cash=cash,
        as_of_date=as_of_date,
    )
    write_allocation_report(result, output_dir)
    result["summary"]["report_path"] = str(Path(output_dir) / "personal_allocation_comparison.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare actual, balanced, conservative and minimum-variance portfolio allocations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS)
    parser.add_argument("--close-prices", type=Path, default=DEFAULT_CLOSE)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cash", type=float, default=None, help="Optional verified account cash in USD.")
    parser.add_argument("--as-of-date", help="Analysis date YYYY-MM-DD.")
    args = parser.parse_args()
    try:
        result = run_allocation_comparison(
            config_path=args.config,
            holdings_path=args.holdings_csv,
            close_prices_path=args.close_prices,
            adjusted_prices_path=args.adjusted_prices,
            profiles_path=args.profiles,
            output_dir=args.output_dir,
            cash=args.cash,
            as_of_date=args.as_of_date,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
