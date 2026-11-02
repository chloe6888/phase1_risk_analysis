"""One-command, local-only risk report for the configured personal portfolio.

This orchestrator reuses the tested risk monitor, rolling-risk calculator and
ETF look-through importer. It does not place orders or upload private files.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import risk_monitor
from portfolio_lookthrough import analyze_portfolio_lookthrough
from portfolio_risk import calculate_rolling_risk, load_price_panel, run_stress_scenarios
from run_personal_portfolio import (
    DEFAULT_ADJUSTED,
    DEFAULT_CLOSE,
    DEFAULT_CONFIG,
    DEFAULT_HOLDINGS,
    DEFAULT_REPORTS,
    load_universe,
    validate_holdings_file,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_GROUP_CAPS = ROOT / "risk_profiles" / "balanced_group_caps.json"
DEFAULT_SUBGROUP_CAPS = ROOT / "risk_profiles" / "balanced_subgroup_caps.json"
DEFAULT_STRESS_PROFILE = ROOT / "risk_profiles" / "balanced_stress_scenarios.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "personal_risk"


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON configuration {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON configuration {source} must contain an object.")
    return payload


def _pairwise_observation_counts(returns: pd.DataFrame) -> pd.DataFrame:
    tickers = list(returns.columns)
    counts = pd.DataFrame(0, index=tickers, columns=tickers, dtype="int64")
    for left in tickers:
        for right in tickers:
            counts.loc[left, right] = int((returns[left].notna() & returns[right].notna()).sum())
    counts.index.name = "ticker"
    counts.columns.name = "ticker"
    return counts.astype("int64")


def _scenario_shocks(
    holdings: pd.DataFrame,
    scenario: Mapping[str, Any],
    *,
    include_cash: bool,
) -> dict[str, float]:
    ticker_shocks = {
        str(k).strip().upper(): float(v)
        for k, v in dict(scenario.get("ticker_shocks", {})).items()
    }
    group_shocks = {
        str(k).strip(): float(v)
        for k, v in dict(scenario.get("group_shocks", {})).items()
    }
    subgroup_shocks = {
        str(k).strip(): float(v)
        for k, v in dict(scenario.get("subgroup_shocks", {})).items()
    }
    shocks: dict[str, float] = {}
    for row in holdings.itertuples(index=False):
        ticker = str(row.ticker).strip().upper()
        group = str(getattr(row, "risk_group", "ungrouped")).strip()
        subgroup = str(getattr(row, "subgroup", "ungrouped")).strip()
        if ticker in ticker_shocks:
            shock = ticker_shocks[ticker]
        elif subgroup in subgroup_shocks:
            shock = subgroup_shocks[subgroup]
        else:
            shock = group_shocks.get(group, 0.0)
        if not np.isfinite(shock) or shock < -1:
            raise ValueError(f"Scenario {scenario.get('name')}: invalid return shock for {ticker}.")
        shocks[ticker] = float(shock)
    if include_cash:
        shocks["CASH"] = 0.0
    return shocks


def analyze_personal_risk(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    adjusted_prices: pd.DataFrame,
    *,
    group_caps: Mapping[str, float],
    subgroup_caps: Mapping[str, float],
    stress_profile: Mapping[str, Any],
    cash: float | None = None,
    lookthrough_snapshots: Sequence[str | Path] = (),
    fund_tickers: Sequence[str] = ("DRAM", "QQQ"),
    as_of_date: str | date | None = None,
    minimum_correlation_observations: int = 60,
    single_name_limit: float = 0.15,
    volatility_limit: float = 0.30,
    drawdown_limit: float = 0.20,
    max_price_age_calendar_days: int = 5,
    max_lookthrough_age_calendar_days: int = 14,
) -> dict[str, Any]:
    """Calculate risk, rolling metrics, correlations, stress and issuer overlap."""
    if minimum_correlation_observations < 2:
        raise ValueError("minimum_correlation_observations must be at least 2.")
    if holdings.empty:
        raise ValueError("Holdings cannot be empty.")
    cash_value = 0.0 if cash is None else float(cash)
    if not np.isfinite(cash_value) or cash_value < 0:
        raise ValueError("cash must be finite and non-negative.")

    positions, alerts, summary = risk_monitor.analyze_risk_snapshot(
        holdings,
        close_prices,
        adjusted_prices=adjusted_prices,
        cash=cash_value,
        group_max_weights=group_caps,
        subgroup_max_weights=subgroup_caps,
        single_name_limit=single_name_limit,
        volatility_limit_annual=volatility_limit,
        drawdown_limit=drawdown_limit,
        volatility_window=60,
        minimum_observations=minimum_correlation_observations,
        max_data_age_calendar_days=max_price_age_calendar_days,
    )
    daily_path = summary.pop("_daily_path")
    daily_returns = pd.to_numeric(daily_path.get("daily_return"), errors="coerce").dropna()
    available_windows = tuple(window for window in (20, 60, 120) if len(daily_returns) >= window)
    if available_windows:
        rolling = calculate_rolling_risk(daily_returns, available_windows)
        rolling_note = (
            "Only windows supported by the available common-return history are included. "
            "A missing 120-day window is a history limitation, not zero risk."
        )
    else:
        rolling = pd.DataFrame(columns=[
            "window_observations",
            "rolling_annualized_volatility",
            "rolling_max_drawdown",
        ])
        rolling.index.name = "Date"
        rolling_note = (
            f"Only {len(daily_returns)} common portfolio-return observations are available; "
            "at least 20 are required for the first rolling window."
        )

    tickers = [str(item).strip().upper() for item in holdings["ticker"]]
    asset_returns = adjusted_prices.loc[:, tickers].apply(pd.to_numeric, errors="coerce").pct_change(fill_method=None)
    correlations = asset_returns.corr(min_periods=minimum_correlation_observations)
    observation_counts = _pairwise_observation_counts(asset_returns)

    weights = {
        str(row.ticker): float(row.portfolio_weight)
        for row in positions.itertuples(index=False)
    }
    if cash_value > 0:
        weights["CASH"] = float(summary["cash_weight"])
    raw_scenarios = stress_profile.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("Stress profile must contain a non-empty scenarios list.")
    stress_cases: dict[str, dict[str, float]] = {}
    stress_labels: dict[str, str] = {}
    for scenario in raw_scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Each stress scenario must be a JSON object.")
        name = str(scenario.get("name", "")).strip()
        if not name or name in stress_cases:
            raise ValueError("Stress scenario names must be non-empty and unique.")
        stress_cases[name] = _scenario_shocks(holdings, scenario, include_cash=cash_value > 0)
        stress_labels[name] = str(scenario.get("label", name))
    stress = run_stress_scenarios(
        weights,
        stress_cases,
        portfolio_value=float(summary["net_equity"]),
    ).reset_index()
    stress.insert(1, "scenario_label", stress["scenario"].map(stress_labels))

    lookthrough_exposures, lookthrough_funds, lookthrough_summary = analyze_portfolio_lookthrough(
        holdings,
        close_prices,
        lookthrough_snapshots,
        fund_tickers=fund_tickers,
        cash=cash if cash is not None else None,
        as_of_date=as_of_date,
        max_price_age_calendar_days=max_price_age_calendar_days,
        max_snapshot_age_calendar_days=max_lookthrough_age_calendar_days,
    )
    summary.update({
        "cash_source": "user_provided" if cash is not None else "not_provided_assumed_zero",
        "cash_note": (
            "Cash was not supplied; this run temporarily uses zero and the denominator covers the imported securities only."
            if cash is None else "Cash was supplied and included in the portfolio-equity denominator."
        ),
        "risk_profile": "balanced",
        "rolling_windows_used": list(available_windows),
        "rolling_risk_note": rolling_note,
        "correlation_minimum_pairwise_observations": int(minimum_correlation_observations),
        "correlation_observation_count_min": int(observation_counts.to_numpy().min()),
        "correlation_observation_count_max": int(observation_counts.to_numpy().max()),
        "stress_scenario_names": list(stress_cases),
        "stress_scenario_note": stress_profile.get(
            "notes", ["Scenario shocks are illustrative assumptions, not forecasts."]
        ),
        "lookthrough_status": (
            "complete" if not lookthrough_summary["fund_snapshots_missing"] and lookthrough_snapshots
            else "partial_or_missing"
        ),
        "lookthrough_funds_missing": lookthrough_summary["fund_snapshots_missing"],
        "lookthrough_warnings": lookthrough_summary["warnings"],
        "limitations": list(summary.get("limitations", [])) + [
            "Historical portfolio returns apply today's share quantities to earlier adjusted prices; past trades, deposits and withdrawals are not reconstructed.",
            "Daily VaR/CVaR are one-return-interval historical tail estimates, not maximum-loss limits.",
            "ETF issuer look-through is limited to matches in the supplied issuer maps; unrecognized underlying issuers remain unmapped.",
            "Stress scenarios are static one-period assumptions and do not model crisis-time correlation changes.",
            "If account cash was omitted, portfolio weights use the imported securities only; do not interpret them as a verified all-assets allocation.",
        ],
    })
    summary.pop("_daily_path", None)
    return {
        "summary": summary,
        "positions": positions,
        "alerts": alerts,
        "daily_path": daily_path,
        "rolling": rolling,
        "correlations": correlations,
        "correlation_observations": observation_counts,
        "stress": stress,
        "lookthrough_exposures": lookthrough_exposures,
        "lookthrough_funds": lookthrough_funds,
        "lookthrough_summary": lookthrough_summary,
    }


def _write_report(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(result["summary"])
    (output_dir / "personal_risk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for key, filename in (
        ("positions", "personal_risk_positions.csv"),
        ("alerts", "personal_risk_alerts.csv"),
        ("daily_path", "personal_risk_daily_path.csv"),
        ("rolling", "personal_risk_rolling.csv"),
        ("correlations", "personal_risk_asset_correlations.csv"),
        ("correlation_observations", "personal_risk_correlation_observations.csv"),
        ("stress", "personal_risk_stress_tests.csv"),
        ("lookthrough_exposures", "personal_portfolio_lookthrough.csv"),
        ("lookthrough_funds", "personal_portfolio_lookthrough_funds.csv"),
    ):
        frame = result[key]
        frame.to_csv(output_dir / filename, float_format="%.10f")

    lookthrough_summary = dict(result["lookthrough_summary"])
    (output_dir / "personal_portfolio_lookthrough_summary.json").write_text(
        json.dumps(lookthrough_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    positions = result["positions"]
    alerts = result["alerts"]
    stress = result["stress"]
    lines = [
        "# 个人组合风险分析报告",
        "",
        f"- 数据日期：{summary.get('latest_data_date', '暂无')}",
        f"- 持仓数量：{summary.get('position_count', 0)}",
        f"- 组合净资产估值：{summary.get('net_equity', 0):,.2f}",
        f"- 现金口径：{summary.get('cash_note', '')}",
        f"- 历史共同收益观测：{summary.get('portfolio_return_observations', 0)}",
        f"- 历史质量：{summary.get('history_quality', 'unknown')}",
        f"- 年化波动率：{summary.get('annualized_volatility')}",
        f"- 历史最大回撤：{summary.get('max_drawdown')}",
        f"- 单期历史 VaR 95%：{summary.get('historical_var_95_one_period')}",
        f"- 单期历史 CVaR 95%：{summary.get('historical_cvar_95_one_period')}",
        "",
        "## 风险主题敞口",
        "",
    ]
    for group, weight in summary.get("group_exposures", {}).items():
        cap = summary.get("group_max_weights", {}).get(group)
        lines.append(f"- {group}: {weight:.2%}" + (f"（复核上限 {cap:.0%}）" if cap is not None else ""))
    lines += ["", "## 子主题敞口", ""]
    for group, weight in summary.get("subgroup_exposures", {}).items():
        cap = summary.get("subgroup_max_weights", {}).get(group)
        lines.append(f"- {group}: {weight:.2%}" + (f"（复核上限 {cap:.0%}）" if cap is not None else ""))
    lines += ["", "## 各持仓风险贡献", ""]
    for row in positions.itertuples(index=False):
        contribution = getattr(row, "risk_contribution_pct", np.nan)
        lines.append(
            f"- {row.ticker}: 权重 {row.portfolio_weight:.2%}，"
            f"波动风险贡献 {contribution:.2f}%"
            if pd.notna(contribution) else
            f"- {row.ticker}: 权重 {row.portfolio_weight:.2%}，风险贡献暂不可计算"
        )
    lines += [
        "",
        "## 滚动风险",
        "",
        f"- 可用窗口：{summary.get('rolling_windows_used', [])}",
        f"- 说明：{summary.get('rolling_risk_note', '')}",
        "",
        "## 压力测试（静态假设）",
        "",
    ]
    if stress.empty:
        lines.append("没有可计算的压力情景。")
    else:
        for row in stress.itertuples(index=False):
            lines.append(
                f"- {row.scenario_label}: 组合假设收益 {row.portfolio_return:.2%}，"
                f"估算损益 {row.estimated_pnl:,.2f}"
            )
    lines += [
        "",
        "## ETF 穿透",
        "",
        f"- 状态：{summary.get('lookthrough_status', 'unknown')}",
    ]
    if result["lookthrough_exposures"].empty:
        lines.append("尚无可用的 ETF 发行人穿透结果。")
    else:
        for row in result["lookthrough_exposures"].itertuples(index=False):
            lines.append(
                f"- {row.issuer}: 直接 {row.direct_weight_pct:.2f}% / "
                f"含 ETF 间接敞口 {row.total_effective_exposure_pct:.2f}%"
            )
    lines += ["", "## 风险告警", ""]
    if alerts.empty:
        lines.append("当前阈值下未触发告警。")
    else:
        lines.extend(f"- [{row.level}] {row.message}" for row in alerts.itertuples(index=False))
    lines += ["", "## 重要限制", ""]
    lines.extend(f"- {item}" for item in summary.get("limitations", []))
    lines += ["", "## ETF 快照提示", ""]
    lines.extend(f"- {item}" for item in lookthrough_summary.get("warnings", []))
    lines.append("")
    (output_dir / "personal_portfolio_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_personal_risk_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    holdings_path: str | Path = DEFAULT_HOLDINGS,
    close_prices_path: str | Path = DEFAULT_CLOSE,
    adjusted_prices_path: str | Path = DEFAULT_ADJUSTED,
    group_caps_path: str | Path = DEFAULT_GROUP_CAPS,
    subgroup_caps_path: str | Path = DEFAULT_SUBGROUP_CAPS,
    stress_profile_path: str | Path = DEFAULT_STRESS_PROFILE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    cash: float | None = None,
    snapshot_paths: Sequence[str | Path] = (),
    as_of_date: str | date | None = None,
) -> dict[str, Any]:
    universe = load_universe(config_path)
    holdings = validate_holdings_file(holdings_path, universe)
    close_prices = load_price_panel(close_prices_path)
    adjusted_prices = load_price_panel(adjusted_prices_path)
    group_caps = _load_json_object(group_caps_path)
    subgroup_caps = _load_json_object(subgroup_caps_path)
    stress_profile = _load_json_object(stress_profile_path)
    fund_tickers = [
        ticker for ticker, metadata in universe["symbols"].items()
        if metadata.get("instrument_type") == "etf"
    ]
    result = analyze_personal_risk(
        holdings,
        close_prices,
        adjusted_prices,
        group_caps=group_caps,
        subgroup_caps=subgroup_caps,
        stress_profile=stress_profile,
        cash=cash,
        lookthrough_snapshots=snapshot_paths,
        fund_tickers=fund_tickers,
        as_of_date=as_of_date,
        minimum_correlation_observations=universe["minimum_common_observations"],
    )
    _write_report(result, Path(output_dir))
    result["summary"]["report_markdown"] = str(Path(output_dir) / "personal_portfolio_risk_report.md")
    result["summary"]["report_directory"] = str(Path(output_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local risk and ETF look-through report for the configured portfolio."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS)
    parser.add_argument("--close-prices", type=Path, default=DEFAULT_CLOSE)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--group-caps", type=Path, default=DEFAULT_GROUP_CAPS)
    parser.add_argument("--subgroup-caps", type=Path, default=DEFAULT_SUBGROUP_CAPS)
    parser.add_argument("--stress-profile", type=Path, default=DEFAULT_STRESS_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cash", type=float, default=None, help="Optional known cash balance in USD.")
    parser.add_argument(
        "--lookthrough-snapshot",
        type=Path,
        action="append",
        default=[],
        help="Dated ETF look-through JSON snapshot; repeat for DRAM and QQQ.",
    )
    parser.add_argument("--as-of-date", help="Analysis date YYYY-MM-DD; defaults to today's local date.")
    args = parser.parse_args()
    try:
        result = run_personal_risk_report(
            config_path=args.config,
            holdings_path=args.holdings_csv,
            close_prices_path=args.close_prices,
            adjusted_prices_path=args.adjusted_prices,
            group_caps_path=args.group_caps,
            subgroup_caps_path=args.subgroup_caps,
            stress_profile_path=args.stress_profile,
            output_dir=args.output_dir,
            cash=args.cash,
            snapshot_paths=args.lookthrough_snapshot,
            as_of_date=args.as_of_date,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
