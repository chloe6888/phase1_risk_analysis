#!/usr/bin/env python3
"""Portfolio-level risk analytics for a personal quantitative research project.

This module is research-only: it calculates historical statistics from adjusted
closing prices and does not generate buy/sell signals or place trades. Portfolio
weights are assumed to be rebalanced to target weights at each return interval.
Returns and risk measures are decimal values (0.10 = 10%).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_PRICE_PANEL = ROOT / "data" / "raw" / "adjusted_close_panel.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
TRADING_DAYS_PER_YEAR = 252


def load_price_panel(path: str | Path) -> pd.DataFrame:
    """Load a Date-indexed CSV price panel and run structural price checks."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price panel not found: {csv_path}. Run phase1_risk_analysis.py first.")
    frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Price panel is empty or has no ticker columns.")
    if frame.index.hasnans:
        raise ValueError("Price panel contains invalid dates.")
    if frame.index.has_duplicates:
        raise ValueError("Price panel contains duplicate dates.")
    if frame.columns.has_duplicates:
        raise ValueError("Price panel contains duplicate ticker columns.")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    for column in frame.columns:
        observed = frame[column].dropna()
        if observed.empty:
            raise ValueError(f"{column}: no valid prices in price panel.")
        if not np.isfinite(observed.to_numpy(dtype=float)).all() or (observed <= 0).any():
            raise ValueError(f"{column}: prices must be finite and positive where present.")
    return frame


def normalize_weights(weights: Mapping[str, float] | pd.Series,
                      available_assets: Sequence[str] | None = None) -> pd.Series:
    """Validate long-only weights and normalize them to sum to one."""
    series = pd.Series(dict(weights), dtype=float)
    if series.empty:
        raise ValueError("At least one portfolio weight is required.")
    if series.index.has_duplicates:
        raise ValueError("Portfolio weights contain duplicate asset names.")
    if any(not isinstance(name, str) or not name.strip() for name in series.index):
        raise ValueError("Portfolio asset names must be non-empty strings.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights must be finite.")
    if (series < 0).any():
        raise ValueError("Negative weights are not supported; this version is long-only.")
    total = float(series.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    if available_assets is not None:
        unknown = sorted(set(series.index) - set(available_assets))
        if unknown:
            raise ValueError(f"Weights refer to assets absent from the price panel: {unknown}")
    return series / total


def _portfolio_returns_from_panel(
    adjusted_close: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Return (portfolio returns, asset returns, normalized active weights)."""
    if adjusted_close.empty or len(adjusted_close.columns) == 0:
        raise ValueError("Adjusted-close price panel cannot be empty.")
    if adjusted_close.index.has_duplicates:
        raise ValueError("Adjusted-close panel contains duplicate dates.")
    prices = adjusted_close.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    active_weights = normalize_weights(weights, prices.columns)
    active_weights = active_weights[active_weights > 0]
    active_prices = prices.loc[:, active_weights.index]
    for column in active_prices.columns:
        vals = active_prices[column].dropna()
        if vals.empty or not np.isfinite(vals.to_numpy(dtype=float)).all() or (vals <= 0).any():
            raise ValueError(f"{column}: weighted asset prices must be finite and positive.")
    asset_returns = active_prices.pct_change(fill_method=None)
    # Only use dates with a valid return for every weighted asset. Never silently
    # convert missing prices into zero returns or renormalize portfolio weights.
    common_returns = asset_returns.dropna(how="any")
    if len(common_returns) < 2:
        raise ValueError("At least three common adjusted-close observations are required to estimate risk.")
    portfolio_returns = common_returns.dot(active_weights.reindex(common_returns.columns))
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns, common_returns, active_weights


def max_drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """Create wealth-index and drawdown series from simple periodic returns."""
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty or not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Returns must contain at least one finite observation.")
    if (clean < -1).any():
        raise ValueError("Simple returns cannot be below -100%.")
    wealth = (1.0 + clean).cumprod()
    # Include initial capital (wealth index 1.0) as a valid running peak.
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1.0
    return pd.DataFrame({"portfolio_wealth_index": wealth, "running_peak": running_peak, "drawdown": drawdown})


def calculate_portfolio_risk(
    adjusted_close: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    confidence_level: float = 0.95,
    risk_free_rate_annual: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Calculate portfolio metrics, risk contributions, returns and drawdowns.

    Historical VaR/CVaR are one-period estimates, not maximum-loss guarantees;
    these estimates can be unstable with short return histories.
    """
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if not np.isfinite(confidence_level) or not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0.5 and less than 1.")
    if not np.isfinite(risk_free_rate_annual) or risk_free_rate_annual <= -1:
        raise ValueError("risk_free_rate_annual must be finite and greater than -100%.")

    portfolio_returns, asset_returns, active_weights = _portfolio_returns_from_panel(adjusted_close, weights)
    drawdown = max_drawdown_series(portfolio_returns)
    observations = int(len(portfolio_returns))
    total_growth = float((1.0 + portfolio_returns).prod())
    annualized_return = total_growth ** (periods_per_year / observations) - 1.0 if total_growth > 0 else float("nan")
    daily_volatility = float(portfolio_returns.std(ddof=1))
    annualized_volatility = daily_volatility * math.sqrt(periods_per_year)
    daily_rf = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess = portfolio_returns - daily_rf
    sharpe = float(excess.mean() / daily_volatility * math.sqrt(periods_per_year)) if daily_volatility > 0 else float("nan")
    downside_deviation = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean()))
    sortino = float(excess.mean() / downside_deviation * math.sqrt(periods_per_year)) if downside_deviation > 0 else float("nan")
    max_drawdown = float(drawdown["drawdown"].min())
    alpha = 1.0 - confidence_level
    tail_threshold = float(portfolio_returns.quantile(alpha))
    tail_returns = portfolio_returns[portfolio_returns <= tail_threshold]
    historical_var = -tail_threshold
    historical_cvar = -float(tail_returns.mean())

    covariance = asset_returns.cov().reindex(index=active_weights.index, columns=active_weights.index)
    weight_vector = active_weights.to_numpy(dtype=float)
    covariance_array = covariance.to_numpy(dtype=float)
    daily_portfolio_vol = math.sqrt(max(float(weight_vector @ covariance_array @ weight_vector), 0.0))
    if daily_portfolio_vol > 0:
        marginal_daily = covariance_array @ weight_vector / daily_portfolio_vol
        component_daily = weight_vector * marginal_daily
        pct_risk = component_daily / daily_portfolio_vol
    else:
        marginal_daily = np.zeros_like(weight_vector)
        component_daily = np.zeros_like(weight_vector)
        pct_risk = np.zeros_like(weight_vector)
    contributions = pd.DataFrame({
        "asset": active_weights.index,
        "weight": weight_vector,
        "marginal_contribution_to_volatility_annualized": marginal_daily * math.sqrt(periods_per_year),
        "component_contribution_to_volatility_annualized": component_daily * math.sqrt(periods_per_year),
        "percentage_of_portfolio_volatility": pct_risk,
    }).set_index("asset")

    trough_date = drawdown["drawdown"].idxmin()
    wealth_before_trough = drawdown.loc[:trough_date, "portfolio_wealth_index"]
    peak_level = float(drawdown.loc[trough_date, "running_peak"])
    matching_peak_dates = wealth_before_trough[wealth_before_trough >= peak_level - 1e-12]
    peak_date = matching_peak_dates.index[-1] if not matching_peak_dates.empty else wealth_before_trough.idxmax()
    summary: dict[str, object] = {
        "start_date": portfolio_returns.index.min().date().isoformat(),
        "end_date": portfolio_returns.index.max().date().isoformat(),
        "observations": observations,
        "periods_per_year": periods_per_year,
        "confidence_level": confidence_level,
        "risk_free_rate_annual": risk_free_rate_annual,
        "normalized_weights": {str(key): float(value) for key, value in active_weights.items()},
        "total_return": total_growth - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_peak_date": pd.Timestamp(peak_date).date().isoformat(),
        "max_drawdown_trough_date": pd.Timestamp(trough_date).date().isoformat(),
        "historical_var_one_period": historical_var,
        "historical_cvar_one_period": historical_cvar,
        "historical_var_horizon": "one observed return interval",
        "risk_model_note": "Historical statistics; target weights are assumed to be rebalanced every observed return interval.",
    }
    daily_returns = pd.concat([asset_returns, portfolio_returns], axis=1)
    daily_returns.index.name = "Date"
    drawdown.index.name = "Date"
    return summary, contributions, drawdown, daily_returns


def calculate_rolling_risk(
    portfolio_returns: pd.Series,
    windows: Sequence[int] = (20, 60, 120),
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate rolling annualized volatility and rolling maximum drawdown."""
    if not windows or any(isinstance(window, bool) or not isinstance(window, int) or window < 2 for window in windows):
        raise ValueError("Every rolling window must be an integer of at least 2 observations.")
    if len(set(windows)) != len(windows):
        raise ValueError("Rolling windows must not contain duplicates.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    values = pd.to_numeric(portfolio_returns, errors="coerce").dropna().astype(float)
    if len(values) < min(windows):
        raise ValueError("Not enough return observations for the smallest requested rolling window.")
    if not np.isfinite(values.to_numpy()).all() or (values < -1).any():
        raise ValueError("Returns must be finite simple returns no lower than -100%.")
    rows: list[pd.DataFrame] = []
    for window in windows:
        rolling_vol = values.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(periods_per_year)

        def window_max_drawdown(raw_returns: np.ndarray) -> float:
            wealth = (1.0 + pd.Series(raw_returns)).cumprod()
            running_peak = wealth.cummax().clip(lower=1.0)
            return float((wealth / running_peak - 1.0).min())

        rolling_dd = values.rolling(window, min_periods=window).apply(window_max_drawdown, raw=True)
        result = pd.DataFrame({
            "window_observations": window,
            "rolling_annualized_volatility": rolling_vol,
            "rolling_max_drawdown": rolling_dd,
        }).dropna()
        result.index.name = "Date"
        rows.append(result)
    return pd.concat(rows).sort_index(kind="stable")


def run_stress_scenarios(
    weights: Mapping[str, float] | pd.Series,
    scenarios: Mapping[str, Mapping[str, float]],
    *,
    portfolio_value: float | None = None,
) -> pd.DataFrame:
    """Estimate portfolio impact for user-defined asset return shocks.

    Scenario shocks are decimal returns (-0.20 means a 20% fall). Assets omitted
    from a scenario are assumed to have zero return in that scenario.
    """
    normalized = normalize_weights(weights)
    if not scenarios:
        raise ValueError("At least one stress scenario is required.")
    if portfolio_value is not None and (not np.isfinite(portfolio_value) or portfolio_value < 0):
        raise ValueError("portfolio_value must be finite and non-negative.")
    rows = []
    for scenario_name, shocks in scenarios.items():
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            raise ValueError("Scenario names must be non-empty strings.")
        unknown = sorted(set(shocks) - set(normalized.index))
        if unknown:
            raise ValueError(f"Scenario '{scenario_name}' has assets absent from weights: {unknown}")
        if any(not np.isfinite(value) or value < -1 for value in shocks.values()):
            raise ValueError(f"Scenario '{scenario_name}' shocks must be finite and no lower than -100%.")
        portfolio_return = sum(float(normalized.get(asset, 0.0)) * float(shock) for asset, shock in shocks.items())
        row: dict[str, object] = {"scenario": scenario_name, "portfolio_return": portfolio_return,
                                  "portfolio_loss_pct": -portfolio_return}
        if portfolio_value is not None:
            row["estimated_pnl"] = portfolio_value * portfolio_return
            row["estimated_loss_amount"] = -portfolio_value * portfolio_return
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def parse_weights(value: str) -> dict[str, float]:
    """Parse command-line weights formatted as TICKER=weight,TICKER=weight."""
    result: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid weight item '{item}'; expected TICKER=weight.")
        ticker, raw_weight = item.split("=", 1)
        ticker = ticker.strip().upper()
        if not ticker or ticker in result:
            raise argparse.ArgumentTypeError(f"Ticker is empty or duplicated in weights: '{ticker}'.")
        try:
            result[ticker] = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid weight for {ticker}: '{raw_weight}'.") from exc
    try:
        normalize_weights(result)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return result


def parse_scenario(value: str) -> tuple[str, dict[str, float]]:
    """Parse a stress case such as name:TICKER=-0.1,OTHER=-0.2."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("Scenario must be NAME:TICKER=return[,TICKER=return...].")
    name, raw_shocks = value.split(":", 1)
    name = name.strip()
    if not name or not raw_shocks.strip():
        raise argparse.ArgumentTypeError("Scenario name and at least one ticker shock are required.")
    shocks: dict[str, float] = {}
    for item in raw_shocks.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid scenario item '{item}'; expected TICKER=return.")
        ticker, raw_value = item.split("=", 1)
        ticker = ticker.strip().upper()
        if not ticker or ticker in shocks:
            raise argparse.ArgumentTypeError(f"Ticker is empty or duplicated in scenario: '{ticker}'.")
        try:
            shocks[ticker] = float(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid shock for {ticker}: '{raw_value}'.") from exc
    if any(not np.isfinite(x) or x < -1 for x in shocks.values()):
        raise argparse.ArgumentTypeError("Stress returns must be finite and no lower than -100%.")
    return name, shocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate historical portfolio risk from an adjusted-close panel.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL, help="CSV from phase1_risk_analysis.py")
    parser.add_argument("--weights", required=True, type=parse_weights, help="Target weights, e.g. GOOGL=0.6,SPY=0.4")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Historical VaR/CVaR confidence level")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate as decimal, e.g. 0.04")
    parser.add_argument("--window", type=int, action="append", dest="windows",
                        help="Rolling window in observations; repeat to add windows (default: 20, 60, 120)")
    parser.add_argument("--portfolio-value", type=float, help="Optional portfolio market value for stress-test dollar impacts")
    parser.add_argument("--scenario", action="append", type=parse_scenario, default=[],
                        help="Repeatable scenario, e.g. market_down:SPY=-0.1,GOOGL=-0.2")
    args = parser.parse_args()

    prices = load_price_panel(args.prices)
    summary, contributions, drawdown, daily_returns = calculate_portfolio_risk(
        prices, args.weights, confidence_level=args.confidence_level,
        risk_free_rate_annual=args.risk_free_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "portfolio_risk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    contributions.to_csv(args.output_dir / "portfolio_risk_contributions.csv", float_format="%.10f")
    drawdown.to_csv(args.output_dir / "portfolio_drawdown.csv", float_format="%.10f")
    daily_returns.to_csv(args.output_dir / "portfolio_daily_returns.csv", float_format="%.10f")
    rolling = calculate_rolling_risk(daily_returns["portfolio_return"], args.windows or (20, 60, 120))
    rolling.to_csv(args.output_dir / "portfolio_rolling_risk.csv", float_format="%.10f")
    if args.scenario:
        scenario_dict: dict[str, dict[str, float]] = {}
        for name, shocks in args.scenario:
            if name in scenario_dict:
                parser.error(f"Duplicate scenario name: {name}")
            scenario_dict[name] = shocks
        stress = run_stress_scenarios(args.weights, scenario_dict, portfolio_value=args.portfolio_value)
        stress.to_csv(args.output_dir / "portfolio_stress_tests.csv", float_format="%.10f")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
