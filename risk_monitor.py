#!/usr/bin/env python3
"""Read-only portfolio concentration and historical-risk monitor.

The historical risk path applies today's share quantities to adjusted historical
prices as a constant-share proxy. It does not reconstruct prior trades or account
flows. This module never connects to a broker or sends orders.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk import DEFAULT_OUTPUT_DIR, DEFAULT_PRICE_PANEL, ROOT, TRADING_DAYS_PER_YEAR, load_price_panel

DEFAULT_CLOSE_PRICES = ROOT / "data" / "raw" / "close_price_panel.csv"
UNGROUPED = "__ungrouped__"


def _normalise_panel(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError(f"{label} price panel must be non-empty.")
    frame = panel.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any() or frame.index.has_duplicates:
        raise ValueError(f"{label} price panel must have valid, unique dates.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{label} price panel must have unique tickers.")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    for ticker in frame.columns:
        observed = frame[ticker].dropna()
        values = observed.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{label} prices for {ticker} must be finite and positive.")
    return frame


def _normalise_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(holdings, pd.DataFrame) or holdings.empty:
        raise ValueError("Holdings must be a non-empty DataFrame.")
    if not {"ticker", "shares"}.issubset(holdings.columns):
        raise ValueError("Holdings must contain ticker and shares columns.")
    frame = holdings.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    if frame["ticker"].eq("").any() or frame["shares"].isna().any():
        raise ValueError("Holdings must have non-empty tickers and numeric shares.")
    if frame["ticker"].duplicated().any():
        raise ValueError("Holdings must not contain duplicate tickers.")
    quantities = frame["shares"].to_numpy(dtype=float)
    if not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("Holding shares must be finite and non-negative.")
    if "risk_group" not in frame.columns:
        frame["risk_group"] = UNGROUPED
    frame["risk_group"] = frame["risk_group"].fillna(UNGROUPED).astype(str).str.strip()
    frame.loc[frame["risk_group"].eq(""), "risk_group"] = UNGROUPED
    return frame.loc[frame["shares"] > 0].reset_index(drop=True)


def analyze_risk_snapshot(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    adjusted_prices: pd.DataFrame | None = None,
    cash: float = 0.0,
    groups: Mapping[str, str] | None = None,
    group_max_weights: Mapping[str, float] | None = None,
    single_name_limit: float = 0.15,
    volatility_limit_annual: float = 0.40,
    drawdown_limit: float = 0.25,
    volatility_window: int = 60,
    minimum_observations: int = 20,
    max_data_age_calendar_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build a current-weight snapshot, historical risk proxy, and alert list."""
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be finite and non-negative.")
    for name, value in (
        ("single_name_limit", single_name_limit),
        ("volatility_limit_annual", volatility_limit_annual),
        ("drawdown_limit", drawdown_limit),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if isinstance(volatility_window, bool) or not isinstance(volatility_window, int) or volatility_window < 2:
        raise ValueError("volatility_window must be an integer >= 2.")
    if isinstance(minimum_observations, bool) or not isinstance(minimum_observations, int) or minimum_observations < 2:
        raise ValueError("minimum_observations must be an integer >= 2.")
    if max_data_age_calendar_days is not None and (
        isinstance(max_data_age_calendar_days, bool)
        or not isinstance(max_data_age_calendar_days, int)
        or max_data_age_calendar_days < 0
    ):
        raise ValueError("max_data_age_calendar_days must be None or a non-negative integer.")

    positions = _normalise_holdings(holdings)
    close = _normalise_panel(close_prices, "close")
    adjusted = _normalise_panel(adjusted_prices if adjusted_prices is not None else close_prices, "adjusted")
    tickers = positions["ticker"].tolist()
    missing_close = sorted(set(tickers) - set(close.columns))
    missing_adjusted = sorted(set(tickers) - set(adjusted.columns))
    if missing_close:
        raise ValueError(f"Missing close-price history for holdings: {missing_close}")
    if missing_adjusted:
        raise ValueError(f"Missing adjusted-price history for holdings: {missing_adjusted}")

    overrides = {str(k).strip().upper(): str(v).strip() for k, v in dict(groups or {}).items()}
    unknown_tickers = sorted(set(overrides) - set(tickers))
    if unknown_tickers:
        raise ValueError(f"Group mapping references unknown holdings: {unknown_tickers}")
    for idx, row in positions.iterrows():
        if row["ticker"] in overrides:
            positions.at[idx, "risk_group"] = overrides[row["ticker"]] or UNGROUPED

    group_caps = {str(k): float(v) for k, v in dict(group_max_weights or {}).items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in group_caps.values()):
        raise ValueError("Every group cap must be finite and between 0 and 1.")
    known_groups = set(positions["risk_group"].astype(str))
    unknown_groups = sorted(set(group_caps) - known_groups)
    if unknown_groups:
        raise ValueError(f"Group caps reference unknown holdings groups: {unknown_groups}")

    latest_prices: dict[str, float] = {}
    latest_dates: dict[str, str] = {}
    for ticker in tickers:
        valid = close[ticker].dropna()
        if valid.empty:
            raise ValueError(f"{ticker}: no current close price is available.")
        latest_prices[ticker] = float(valid.iloc[-1])
        latest_dates[ticker] = pd.Timestamp(valid.index[-1]).date().isoformat()

    positions["latest_price"] = positions["ticker"].map(latest_prices)
    positions["market_value"] = positions["shares"] * positions["latest_price"]
    net_equity = float(cash + positions["market_value"].sum())
    if not math.isfinite(net_equity) or net_equity <= 0:
        raise ValueError("Net account equity must be positive.")
    positions["portfolio_weight"] = positions["market_value"] / net_equity
    group_series = positions.groupby("risk_group")["portfolio_weight"].sum().sort_values(ascending=False)
    group_exposure_map = {str(k): float(v) for k, v in group_series.items()}

    history = adjusted[tickers].dropna(how="any")
    shares = positions.set_index("ticker")["shares"].reindex(tickers)
    historical_equity = history.mul(shares, axis=1).sum(axis=1) + cash
    historical_equity = historical_equity[historical_equity > 0]
    daily_returns = historical_equity.pct_change().dropna()
    if daily_returns.empty:
        max_drawdown = None
        annualized_volatility = None
        var_95 = None
        cvar_95 = None
        drawdown_series = pd.Series(dtype=float, name="drawdown")
    else:
        wealth = (1.0 + daily_returns).cumprod()
        drawdown_series = wealth / wealth.cummax() - 1.0
        max_drawdown = float(drawdown_series.min())
        trailing = daily_returns.tail(volatility_window)
        annualized_volatility = (
            float(trailing.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
            if len(trailing) >= 2 else None
        )
        q05 = float(daily_returns.quantile(0.05))
        var_95 = -q05
        tail = daily_returns[daily_returns <= q05]
        cvar_95 = -float(tail.mean()) if not tail.empty else None

    alerts: list[dict[str, object]] = []
    for row in positions.itertuples(index=False):
        if float(row.portfolio_weight) > single_name_limit + 1e-9:
            alerts.append({
                "level": "warning", "kind": "single_name_concentration",
                "ticker": row.ticker, "group": row.risk_group,
                "observed": float(row.portfolio_weight), "limit": float(single_name_limit),
                "message": f"{row.ticker} weight {row.portfolio_weight:.1%} exceeds the {single_name_limit:.1%} single-name limit.",
            })
    for group, exposure in group_exposure_map.items():
        limit = group_caps.get(group)
        if limit is not None and exposure > limit + 1e-9:
            alerts.append({
                "level": "warning", "kind": "group_concentration",
                "ticker": None, "group": group, "observed": exposure, "limit": limit,
                "message": f"Group {group} weight {exposure:.1%} exceeds the {limit:.1%} group limit.",
            })
    if annualized_volatility is not None and annualized_volatility > volatility_limit_annual:
        alerts.append({
            "level": "warning", "kind": "portfolio_volatility",
            "ticker": None, "group": None, "observed": annualized_volatility,
            "limit": float(volatility_limit_annual),
            "message": f"Annualized portfolio volatility {annualized_volatility:.1%} exceeds the {volatility_limit_annual:.1%} limit.",
        })
    if max_drawdown is not None and max_drawdown <= -drawdown_limit:
        alerts.append({
            "level": "warning", "kind": "portfolio_drawdown",
            "ticker": None, "group": None, "observed": abs(max_drawdown),
            "limit": float(drawdown_limit),
            "message": f"Historical portfolio drawdown {abs(max_drawdown):.1%} reached the {drawdown_limit:.1%} alert threshold.",
        })
    if len(daily_returns) < minimum_observations:
        alerts.append({
            "level": "info", "kind": "insufficient_history",
            "ticker": None, "group": None, "observed": int(len(daily_returns)),
            "limit": int(minimum_observations),
            "message": f"Only {len(daily_returns)} portfolio return observations are available; risk estimates are limited.",
        })

    latest_data_date = max(latest_dates.values())
    if max_data_age_calendar_days is not None:
        age = (pd.Timestamp.now().normalize() - pd.Timestamp(latest_data_date)).days
        if age > max_data_age_calendar_days:
            alerts.append({
                "level": "warning", "kind": "stale_price_data",
                "ticker": None, "group": None, "observed": int(age),
                "limit": int(max_data_age_calendar_days),
                "message": f"Latest holdings price date is {age} calendar days old.",
            })

    positions = positions.sort_values("portfolio_weight", ascending=False, kind="stable").reset_index(drop=True)
    daily_path = pd.DataFrame({"portfolio_equity_proxy": historical_equity})
    daily_path["daily_return"] = historical_equity.pct_change()
    if not drawdown_series.empty:
        daily_path["drawdown"] = drawdown_series.reindex(daily_path.index)

    summary: dict[str, object] = {
        "net_equity": net_equity,
        "cash": float(cash),
        "cash_weight": float(cash / net_equity),
        "position_count": int(len(positions)),
        "latest_data_date": latest_data_date,
        "latest_data_dates_by_ticker": latest_dates,
        "historical_start": history.index.min().date().isoformat() if not history.empty else None,
        "historical_end": history.index.max().date().isoformat() if not history.empty else None,
        "portfolio_return_observations": int(len(daily_returns)),
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "historical_var_95_one_period": var_95,
        "historical_cvar_95_one_period": cvar_95,
        "single_name_limit": float(single_name_limit),
        "group_max_weights": group_caps,
        "group_exposures": group_exposure_map,
        "alert_count": int(len(alerts)),
        "methodology": {
            "market_value": "shares multiplied by each holding's latest available unadjusted close",
            "historical_risk_path": "today's share quantities applied to adjusted historical prices; cash held constant",
            "volatility": f"sample standard deviation of the latest {volatility_window} daily portfolio-proxy returns, annualized by sqrt(252)",
            "var_cvar": "historical one-period 95% loss-tail estimates; not maximum-loss limits",
            "no_broker_connection": True,
        },
        "limitations": [
            "The historical equity path applies current share quantities to earlier adjusted prices; it does not reconstruct prior trades, account flows, or historical share counts.",
            "Cash is assumed constant and earns zero return in the historical path.",
            "Concentration limits and alert thresholds are user-supplied risk-budget rules, not optimal limits.",
            "The monitor cannot predict losses or guarantee an alert before a drawdown.",
            "It reads files and writes reports only; it does not place orders or connect to a broker.",
        ],
        "_daily_path": daily_path,
    }
    return positions, pd.DataFrame(alerts), summary


def _load_json_mapping(path: Path | None, label: str) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only portfolio concentration and historical-risk monitor; no orders are placed."
    )
    parser.add_argument("--holdings-csv", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=DEFAULT_CLOSE_PRICES)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--single-name-limit", type=float, default=0.15)
    parser.add_argument("--group-caps-json", type=Path)
    parser.add_argument("--groups-json", type=Path)
    parser.add_argument("--volatility-limit", type=float, default=0.40)
    parser.add_argument("--drawdown-limit", type=float, default=0.25)
    parser.add_argument("--volatility-window", type=int, default=60)
    parser.add_argument("--max-data-age-days", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    holdings = pd.read_csv(args.holdings_csv)
    close_prices = load_price_panel(args.prices)
    adjusted_prices = load_price_panel(args.adjusted_prices)
    group_map = _load_json_mapping(args.groups_json, "groups")
    group_caps = _load_json_mapping(args.group_caps_json, "group caps")
    positions, alerts, summary = analyze_risk_snapshot(
        holdings, close_prices, adjusted_prices=adjusted_prices, cash=args.cash,
        groups={str(k): str(v) for k, v in group_map.items()},
        group_max_weights={str(k): float(v) for k, v in group_caps.items()},
        single_name_limit=args.single_name_limit,
        volatility_limit_annual=args.volatility_limit,
        drawdown_limit=args.drawdown_limit,
        volatility_window=args.volatility_window,
        max_data_age_calendar_days=args.max_data_age_days,
    )
    daily_path = summary.pop("_daily_path")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(args.output_dir / "risk_monitor_positions.csv", index=False, float_format="%.10f")
    alerts.to_csv(args.output_dir / "risk_monitor_alerts.csv", index=False)
    daily_path.to_csv(args.output_dir / "risk_monitor_daily_path.csv", float_format="%.10f")
    (args.output_dir / "risk_monitor_snapshot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 组合风险监控报告",
        "",
        f"- 账户净资产估值：{summary['net_equity']:.2f}",
        f"- 最新价格日期：{summary['latest_data_date']}",
        f"- 历史年化波动率：{summary['annualized_volatility']}",
        f"- 历史最大回撤：{summary['max_drawdown']}",
        f"- 告警数量：{summary['alert_count']}",
        "",
        "## 告警",
        "",
    ]
    if alerts.empty:
        lines.append("当前阈值下没有触发告警。")
    else:
        lines.extend(f"- [{row.level}] {row.message}" for row in alerts.itertuples(index=False))
    lines += ["", "## 限制", "", *[f"- {item}" for item in summary["limitations"]], ""]
    (args.output_dir / "risk_monitor_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
