#!/usr/bin/env python3
"""Explain portfolio returns with user-supplied ETF or factor return proxies.

The model is a contemporaneous OLS decomposition, not a forecast. Factor proxies
are supplied by the user as adjusted-close price panels (for example SPY, SOXX,
QQQ, IWM, TLT, GLD); they are not represented as official academic factor data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk import DEFAULT_OUTPUT_DIR, DEFAULT_PRICE_PANEL, TRADING_DAYS_PER_YEAR, load_price_panel


def _normalise_price_panel(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError(f"{label} price panel must be a non-empty DataFrame.")
    frame = panel.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any() or frame.index.has_duplicates:
        raise ValueError(f"{label} price panel must have valid, unique dates.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{label} price panel must have unique ticker/factor columns.")
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError(f"{label} panel contains duplicate columns after normalization.")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    for column in frame.columns:
        observed = frame[column].dropna()
        if observed.empty:
            raise ValueError(f"{label} panel has no valid prices for {column}.")
        values = observed.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{label} prices for {column} must be finite and positive.")
    return frame


def normalize_portfolio_weights(
    weights: pd.DataFrame | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(weights, pd.DataFrame):
        if not {"ticker", "weight"}.issubset(weights.columns):
            raise ValueError("Weights must contain ticker and weight columns.")
        frame = weights[["ticker", "weight"]].copy()
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        if frame["ticker"].eq("").any() or frame["weight"].isna().any():
            raise ValueError("Weights must have non-empty tickers and numeric values.")
        if frame["ticker"].duplicated().any():
            raise ValueError("Portfolio weights must not contain duplicate tickers.")
        result = dict(zip(frame["ticker"], frame["weight"].astype(float)))
    else:
        result = {str(key).strip().upper(): float(value) for key, value in dict(weights).items()}
    if not result:
        raise ValueError("At least one portfolio weight is required.")
    values = np.asarray(list(result.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Portfolio weights must be finite and between 0 and 1.")
    if sum(result.values()) > 1.0 + 1e-9:
        raise ValueError("Portfolio risky weights must sum to no more than 100%; residual is cash.")
    return {key: value for key, value in result.items() if value > 1e-12}


def build_portfolio_returns(
    adjusted_close: pd.DataFrame,
    weights: pd.DataFrame | Mapping[str, float],
) -> tuple[pd.Series, dict[str, object]]:
    """Build a constant-weight daily-rebalanced proxy from a position snapshot.

    The unused weight is cash and earns zero. Every held security must have a
    return on a date for that date to be included; no missing return is filled.
    """
    prices = _normalise_price_panel(adjusted_close, "Asset")
    weight_map = normalize_portfolio_weights(weights)
    missing = sorted(set(weight_map) - set(prices.columns))
    if missing:
        raise ValueError(f"Weight file contains tickers missing from the asset price panel: {missing}")
    selected = prices[list(weight_map)]
    returns = selected.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No common daily return observations are available for the held assets.")
    portfolio = returns.mul(pd.Series(weight_map), axis=1).sum(axis=1)
    portfolio.name = "portfolio_return"
    portfolio.index.name = "Date"
    summary = {
        "tickers": list(weight_map),
        "weights": weight_map,
        "risky_weight": float(sum(weight_map.values())),
        "cash_weight": float(max(0.0, 1.0 - sum(weight_map.values()))),
        "observations_before_factor_alignment": int(len(portfolio)),
        "start_date_before_factor_alignment": portfolio.index.min().date().isoformat(),
        "end_date_before_factor_alignment": portfolio.index.max().date().isoformat(),
        "portfolio_return_method": "constant target weights rebalanced each observed close; residual cash earns zero",
    }
    return portfolio, summary


def build_factor_returns(factor_adjusted_close: pd.DataFrame) -> pd.DataFrame:
    """Convert user-supplied factor-proxy adjusted closes to daily simple returns."""
    prices = _normalise_price_panel(factor_adjusted_close, "Factor")
    if len(prices.columns) < 1:
        raise ValueError("At least one factor proxy is required.")
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns.index.name = "Date"
    returns.columns.name = "factor"
    return returns


def estimate_factor_risk(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    minimum_observations: int = 60,
    max_pairwise_correlation_warning: float = 0.90,
    condition_number_warning: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit OLS and decompose annualized portfolio variance by factor.

    Factor contribution is beta_i * Cov(factor_i, fitted factor return). It can
    be negative when a factor acts as a hedge. Residual risk is reported separately.
    """
    if not isinstance(portfolio_returns, pd.Series):
        raise ValueError("portfolio_returns must be a pandas Series.")
    if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
        raise ValueError("factor_returns must be a non-empty DataFrame.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if isinstance(minimum_observations, bool) or not isinstance(minimum_observations, int) or minimum_observations < 3:
        raise ValueError("minimum_observations must be an integer >= 3.")
    if (
        not math.isfinite(max_pairwise_correlation_warning)
        or not 0 <= max_pairwise_correlation_warning <= 1
    ):
        raise ValueError("max_pairwise_correlation_warning must be between 0 and 1.")
    if not math.isfinite(condition_number_warning) or condition_number_warning <= 0:
        raise ValueError("condition_number_warning must be finite and positive.")

    y = pd.to_numeric(portfolio_returns, errors="coerce").copy()
    y.index = pd.to_datetime(y.index, errors="coerce")
    if y.index.isna().any() or y.index.has_duplicates:
        raise ValueError("Portfolio return series must have valid, unique dates.")
    y = y.sort_index().rename("portfolio_return")

    factors = factor_returns.copy()
    factors.index = pd.to_datetime(factors.index, errors="coerce")
    if factors.index.isna().any() or factors.index.has_duplicates:
        raise ValueError("Factor returns must have valid, unique dates.")
    if factors.columns.has_duplicates:
        raise ValueError("Factor returns must have unique factor names.")
    factors.columns = [str(column).strip() for column in factors.columns]
    if any(not column for column in factors.columns) or len(set(factors.columns)) != len(factors.columns):
        raise ValueError("Factor names must be non-empty and unique.")
    factors = factors.sort_index().apply(pd.to_numeric, errors="coerce")
    if factors.empty or factors.shape[1] == 0:
        raise ValueError("At least one factor column is required.")

    combined = pd.concat([y, factors], axis=1, join="inner").dropna(how="any")
    if len(combined) < max(minimum_observations, factors.shape[1] + 3):
        raise ValueError(
            f"Only {len(combined)} aligned observations are available; "
            f"at least {max(minimum_observations, factors.shape[1] + 3)} are required."
        )
    values = combined.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Portfolio and factor returns must be finite.")
    if (values < -1.0).any():
        raise ValueError("Simple returns cannot be lower than -100%.")

    y_values = combined["portfolio_return"].to_numpy(dtype=float)
    factor_names = list(factors.columns)
    x_values = combined[factor_names].to_numpy(dtype=float)
    factor_std = x_values.std(axis=0, ddof=1)
    constant_factors = [
        factor_names[index] for index, value in enumerate(factor_std)
        if not math.isfinite(float(value)) or value <= 1e-14
    ]
    if constant_factors:
        raise ValueError(f"Factor returns have no variation: {constant_factors}")

    design = np.column_stack([np.ones(len(combined)), x_values])
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            "Factor matrix is rank-deficient due to perfectly collinear factors; remove or combine duplicate proxies."
        )
    coefficients, _, _, _ = np.linalg.lstsq(design, y_values, rcond=None)
    alpha_daily = float(coefficients[0])
    betas = coefficients[1:]
    fitted = design @ coefficients
    residuals = y_values - fitted

    total_variance_daily = float(np.var(y_values, ddof=1))
    if total_variance_daily <= 1e-18:
        raise ValueError("Portfolio return variance is too small for factor-risk decomposition.")
    residual_sum_squares = float(residuals @ residuals)
    total_sum_squares = float(np.square(y_values - y_values.mean()).sum())
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    r_squared = float(min(1.0, max(0.0, r_squared)))

    covariance_daily = np.cov(x_values, rowvar=False, ddof=1)
    covariance_daily = np.atleast_2d(np.asarray(covariance_daily, dtype=float))
    covariance_annual = covariance_daily * periods_per_year
    explained_variance_daily = float(betas @ covariance_daily @ betas)
    explained_variance_daily = min(total_variance_daily, max(0.0, explained_variance_daily))
    residual_variance_daily = max(0.0, total_variance_daily - explained_variance_daily)
    factor_cov_beta = covariance_daily @ betas
    contributions_daily = betas * factor_cov_beta
    scale = (
        explained_variance_daily / float(contributions_daily.sum())
        if abs(float(contributions_daily.sum())) > 1e-18 else 1.0
    )
    contributions_daily = contributions_daily * scale

    total_variance_annual = total_variance_daily * periods_per_year
    explained_variance_annual = explained_variance_daily * periods_per_year
    residual_variance_annual = residual_variance_daily * periods_per_year
    total_volatility_annual = math.sqrt(total_variance_annual)
    residual_volatility_annual = math.sqrt(residual_variance_annual)
    portfolio_factor_correlation = np.asarray([
        np.corrcoef(x_values[:, index], y_values)[0, 1]
        for index in range(x_values.shape[1])
    ], dtype=float)
    factor_correlations = pd.DataFrame(
        np.corrcoef(x_values, rowvar=False),
        index=factor_names,
        columns=factor_names,
    )
    if len(factor_names) == 1:
        factor_correlations = pd.DataFrame([[1.0]], index=factor_names, columns=factor_names)

    standardized_factors = (x_values - x_values.mean(axis=0)) / factor_std
    condition_number = float(np.linalg.cond(standardized_factors))
    max_pairwise_corr = 0.0
    high_corr_pairs: list[dict[str, object]] = []
    for left in range(len(factor_names)):
        for right in range(left + 1, len(factor_names)):
            correlation = float(factor_correlations.iloc[left, right])
            max_pairwise_corr = max(max_pairwise_corr, abs(correlation))
            if abs(correlation) >= max_pairwise_correlation_warning:
                high_corr_pairs.append({
                    "factor_1": factor_names[left],
                    "factor_2": factor_names[right],
                    "correlation": correlation,
                })

    exposure_rows: list[dict[str, object]] = []
    for index, factor in enumerate(factor_names):
        contribution = float(contributions_daily[index] * periods_per_year)
        contribution_pct = contribution / total_variance_annual if total_variance_annual > 0 else 0.0
        exposure_rows.append({
            "factor": factor,
            "beta": float(betas[index]),
            "factor_volatility_annualized": float(factor_std[index] * math.sqrt(periods_per_year)),
            "portfolio_factor_correlation": float(portfolio_factor_correlation[index]),
            "variance_contribution_annualized": contribution,
            "variance_contribution_pct": float(contribution_pct),
            "interpretation": (
                "risk_offset" if contribution_pct < -1e-8
                else "positive_risk_contribution" if contribution_pct > 1e-8
                else "near_zero_contribution"
            ),
        })

    residual_share = residual_variance_annual / total_variance_annual
    explained_share = explained_variance_annual / total_variance_annual
    alpha_annual_arithmetic = alpha_daily * periods_per_year
    factor_table = pd.DataFrame(exposure_rows).sort_values(
        "variance_contribution_pct", ascending=False, kind="stable"
    ).reset_index(drop=True)
    aligned = combined.copy()
    aligned["fitted_factor_return"] = fitted
    aligned["residual_return"] = residuals
    aligned.index.name = "Date"

    warnings: list[str] = []
    if high_corr_pairs:
        warnings.append(
            "Some factor proxies are highly correlated; individual betas and contributions may be unstable."
        )
    if condition_number >= condition_number_warning:
        warnings.append(
            f"Standardized factor design condition number {condition_number:.1f} exceeds the configured warning threshold."
        )
    if explained_share < 0.20:
        warnings.append("The selected factor proxies explain less than 20% of historical portfolio return variance.")
    if residual_share > 0.80:
        warnings.append("More than 80% of historical portfolio return variance remains in the residual component.")

    summary: dict[str, object] = {
        "start_date": combined.index.min().date().isoformat(),
        "end_date": combined.index.max().date().isoformat(),
        "observations": int(len(combined)),
        "periods_per_year": periods_per_year,
        "portfolio_annualized_volatility": total_volatility_annual,
        "factor_explained_annualized_volatility": math.sqrt(explained_variance_annual),
        "residual_annualized_volatility": residual_volatility_annual,
        "r_squared": r_squared,
        "adjusted_r_squared": (
            1.0 - (1.0 - r_squared) * (len(combined) - 1) / (len(combined) - design.shape[1])
            if len(combined) > design.shape[1] else None
        ),
        "alpha_daily": alpha_daily,
        "alpha_annualized_arithmetic": alpha_annual_arithmetic,
        "factor_names": factor_names,
        "factor_variance_share_sum": float(sum(row["variance_contribution_pct"] for row in exposure_rows)),
        "residual_variance_share": float(residual_share),
        "max_pairwise_factor_correlation": float(max_pairwise_corr),
        "high_correlation_pairs": high_corr_pairs,
        "standardized_factor_condition_number": condition_number,
        "warnings": warnings,
        "methodology": {
            "model": "ordinary least squares with intercept",
            "factor_source": "user-supplied adjusted-close proxy price panel converted to daily simple returns",
            "interpretation": "contemporaneous historical explanation, not a predictive or causal model",
            "variance_decomposition": "beta_i * Cov(factor_i, fitted factor return), plus residual variance",
            "cash": "residual portfolio cash earns zero in the constant-weight portfolio proxy",
        },
        "limitations": [
            "ETF proxies are not official Fama-French or other audited academic factor returns.",
            "Contemporaneous regression explains historical co-movement; it does not establish causation or forecast future returns.",
            "Highly correlated proxies can make individual factor betas and variance contributions unstable.",
            "Portfolio returns use constant target weights rebalanced each observed close; they do not reconstruct actual historical share-count drift, deposits, withdrawals, or execution costs.",
            "Residual risk includes omitted factors, asset-specific events, measurement error and model misspecification.",
        ],
    }
    return factor_table, summary, aligned


def _render_report(
    exposures: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# 组合因子风险分解报告",
        "",
        f"- 样本区间：{summary['start_date']} 至 {summary['end_date']}",
        f"- 共同观测数：{summary['observations']}",
        f"- 组合历史年化波动率：{summary['portfolio_annualized_volatility']:.2%}",
        f"- 因子解释波动率：{summary['factor_explained_annualized_volatility']:.2%}",
        f"- 残差波动率：{summary['residual_annualized_volatility']:.2%}",
        f"- 决定系数 R²：{summary['r_squared']:.3f}",
        f"- 年化截距（算术）：{summary['alpha_annualized_arithmetic']:.2%}",
        "",
        "## 因子暴露与方差贡献",
        "",
        "~~~text",
        exposures.to_string(index=False),
        "~~~",
        "",
        "## 模型警告",
        "",
    ]
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("当前阈值下未触发模型警告。")
    lines += [
        "",
        "## 使用限制",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose historical portfolio returns into user-supplied factor proxies; no orders are placed."
    )
    parser.add_argument("--asset-prices", type=Path, default=DEFAULT_PRICE_PANEL,
                        help="Adjusted-close panel containing portfolio tickers.")
    parser.add_argument("--factor-prices", type=Path, required=True,
                        help="Adjusted-close panel of user-selected factor ETFs/proxies.")
    parser.add_argument("--weights-csv", type=Path, required=True,
                        help="CSV with ticker,weight columns; optimizer output is accepted.")
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--max-factor-correlation-warning", type=float, default=0.90)
    parser.add_argument("--condition-number-warning", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    asset_prices = load_price_panel(args.asset_prices)
    factor_prices = load_price_panel(args.factor_prices)
    weights = pd.read_csv(args.weights_csv)
    portfolio_returns, portfolio_info = build_portfolio_returns(asset_prices, weights)
    factor_returns = build_factor_returns(factor_prices)
    exposures, summary, aligned = estimate_factor_risk(
        portfolio_returns,
        factor_returns,
        minimum_observations=args.minimum_observations,
        max_pairwise_correlation_warning=args.max_factor_correlation_warning,
        condition_number_warning=args.condition_number_warning,
    )
    summary["portfolio"] = portfolio_info
    summary["asset_price_source"] = str(args.asset_prices)
    summary["factor_price_source"] = str(args.factor_prices)
    summary["weights_source"] = str(args.weights_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(args.output_dir / "portfolio_factor_exposures.csv", index=False, float_format="%.10f")
    aligned.to_csv(args.output_dir / "portfolio_factor_daily_returns.csv", float_format="%.10f")
    (args.output_dir / "portfolio_factor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "portfolio_factor_report.md").write_text(
        _render_report(exposures, summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
