#!/usr/bin/env python3
"""Long-only portfolio optimization with per-asset and group exposure limits.

The default objective is historical minimum variance. All constraints are
expressed as fractions of total account equity; unused risky budget remains cash.
This is a historical risk-allocation research tool, not a return forecast.
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

UNGROUPED = "__ungrouped__"


def _normalise_prices(adjusted_close: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(adjusted_close, pd.DataFrame) or adjusted_close.empty:
        raise ValueError("Price panel must be a non-empty DataFrame.")
    prices = adjusted_close.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.isna().any() or prices.index.has_duplicates:
        raise ValueError("Price panel must have valid, unique dates.")
    if prices.columns.has_duplicates:
        raise ValueError("Price panel must have unique ticker columns.")
    if len(prices.columns) < 2:
        raise ValueError("At least two candidate assets are required for portfolio optimization.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    for ticker in prices.columns:
        observed = prices[ticker].dropna()
        if observed.empty:
            raise ValueError(f"{ticker}: no valid prices are available.")
        data = observed.to_numpy(dtype=float)
        if not np.isfinite(data).all() or (data <= 0).any():
            raise ValueError(f"{ticker}: observed prices must be finite and positive.")
    return prices


def _project_group_capped_budget(
    values: np.ndarray,
    *,
    asset_caps: np.ndarray,
    group_indices: Mapping[str, np.ndarray],
    group_caps: Mapping[str, float],
    budget: float,
    iterations: int = 100,
) -> np.ndarray:
    """Euclidean projection onto a capped simplex with disjoint group caps.

    For each Lagrange multiplier of the total-budget equality, group multipliers
    are solved by bisection when a group cap binds. An outer bisection solves the
    total risky budget. Groups are disjoint because each ticker has one group.
    """
    vector = np.asarray(values, dtype=float)
    caps = np.asarray(asset_caps, dtype=float)
    if vector.ndim != 1 or caps.shape != vector.shape:
        raise ValueError("Projection values and asset caps must be one-dimensional and aligned.")
    if not np.isfinite(vector).all() or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Projection values must be finite and caps finite/non-negative.")

    specs: list[tuple[np.ndarray, float]] = []
    total_capacity = 0.0
    for group, indices in group_indices.items():
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        available = float(caps[idx].sum())
        limit = min(float(group_caps.get(group, available)), available)
        if limit < 0:
            raise ValueError(f"Group cap for '{group}' must be non-negative.")
        specs.append((idx, limit))
        total_capacity += limit
    if total_capacity + 1e-10 < budget:
        raise ValueError(
            f"Constraints are infeasible: maximum allowed risky weight is {total_capacity:.6f}, "
            f"below requested budget {budget:.6f}."
        )

    def project_at(total_multiplier: float) -> np.ndarray:
        projected = np.zeros_like(vector)
        for idx, group_limit in specs:
            group_values = vector[idx] - total_multiplier
            group_caps_local = caps[idx]
            natural = np.clip(group_values, 0.0, group_caps_local)
            if float(natural.sum()) <= group_limit + 1e-12:
                projected[idx] = natural
                continue
            low = 0.0
            high = max(1.0, float(np.max(group_values)) + 1.0)
            for _ in range(iterations):
                middle = (low + high) / 2.0
                candidate = np.clip(group_values - middle, 0.0, group_caps_local)
                if float(candidate.sum()) > group_limit:
                    low = middle
                else:
                    high = middle
            projected[idx] = np.clip(group_values - high, 0.0, group_caps_local)
        return projected

    low = float(np.min(vector - caps)) - 1.0
    high = float(np.max(vector)) + 1.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = project_at(middle)
        if float(candidate.sum()) > budget:
            low = middle
        else:
            high = middle
    result = project_at((low + high) / 2.0)
    if abs(float(result.sum()) - budget) > 1e-7:
        raise RuntimeError("Capped-simplex projection did not satisfy the risky-budget equality.")
    return result


def _validate_constraints(
    tickers: list[str],
    *,
    default_asset_max_weight: float,
    asset_max_weights: Mapping[str, float] | None,
    groups: Mapping[str, str] | None,
    group_max_weights: Mapping[str, float] | None,
    risky_budget: float,
) -> tuple[np.ndarray, dict[str, str], dict[str, float], dict[str, np.ndarray]]:
    if not math.isfinite(risky_budget) or not 0 < risky_budget <= 1:
        raise ValueError("risky_budget must be greater than 0 and no more than 1.")
    if not math.isfinite(default_asset_max_weight) or not 0 < default_asset_max_weight <= 1:
        raise ValueError("default_asset_max_weight must be in (0, 1].")

    caps_map = dict(asset_max_weights or {})
    unknown_assets = sorted(set(caps_map) - set(tickers))
    if unknown_assets:
        raise ValueError(f"Asset caps reference unknown tickers: {unknown_assets}")
    asset_caps: list[float] = []
    for ticker in tickers:
        value = float(caps_map.get(ticker, default_asset_max_weight))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Asset cap for {ticker} must be between 0 and 1.")
        asset_caps.append(value)

    supplied_groups = dict(groups or {})
    unknown_group_assets = sorted(set(supplied_groups) - set(tickers))
    if unknown_group_assets:
        raise ValueError(f"Group mapping references unknown tickers: {unknown_group_assets}")
    group_by_ticker: dict[str, str] = {}
    for ticker in tickers:
        name = str(supplied_groups.get(ticker, UNGROUPED)).strip()
        if not name:
            raise ValueError(f"Group name for {ticker} must not be empty.")
        group_by_ticker[ticker] = name

    group_caps = {str(k): float(v) for k, v in dict(group_max_weights or {}).items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in group_caps.values()):
        raise ValueError("Every group cap must be finite and between 0 and 1.")
    known_groups = set(group_by_ticker.values())
    unknown_groups = sorted(set(group_caps) - known_groups)
    if unknown_groups:
        raise ValueError(f"Group caps reference unknown groups: {unknown_groups}")

    grouped_indices: dict[str, list[int]] = {}
    for position, ticker in enumerate(tickers):
        grouped_indices.setdefault(group_by_ticker[ticker], []).append(position)
    group_indices = {
        name: np.asarray(indices, dtype=int)
        for name, indices in grouped_indices.items()
    }

    capacity = 0.0
    cap_array = np.asarray(asset_caps, dtype=float)
    for name, indices in group_indices.items():
        available = float(cap_array[indices].sum())
        capacity += min(available, group_caps.get(name, available))
    if capacity + 1e-10 < risky_budget:
        raise ValueError(
            f"Constraints are infeasible: maximum allowed risky weight is {capacity:.6f}, "
            f"below requested budget {risky_budget:.6f}."
        )
    return cap_array, group_by_ticker, group_caps, group_indices


def _portfolio_stats(
    weights: np.ndarray,
    covariance: np.ndarray,
    expected_returns: np.ndarray,
) -> dict[str, float]:
    variance = max(0.0, float(weights @ covariance @ weights))
    volatility = math.sqrt(variance)
    expected_return = float(weights @ expected_returns)
    if variance > 0:
        contributions = weights * (covariance @ weights) / variance
    else:
        contributions = np.zeros_like(weights)
    return {
        "expected_return_annualized": expected_return,
        "volatility_annualized": volatility,
        "variance_annualized": variance,
        "risk_contribution_sum": float(contributions.sum()),
    }


def optimize_portfolio(
    adjusted_close: pd.DataFrame,
    *,
    risky_budget: float = 0.95,
    default_asset_max_weight: float = 0.35,
    asset_max_weights: Mapping[str, float] | None = None,
    groups: Mapping[str, str] | None = None,
    group_max_weights: Mapping[str, float] | None = None,
    objective: str = "minimum_variance",
    return_weight: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    minimum_observations: int = 60,
    max_iterations: int = 50000,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Optimize historical risk allocation subject to long-only exposure caps.

    objective='minimum_variance' minimizes annualized portfolio variance.
    objective='mean_variance' minimizes 0.5*variance - return_weight*expected_return;
    expected returns are annualized historical arithmetic means and are noisy.
    """
    if objective not in {"minimum_variance", "mean_variance"}:
        raise ValueError("objective must be 'minimum_variance' or 'mean_variance'.")
    if not math.isfinite(return_weight) or return_weight < 0:
        raise ValueError("return_weight must be finite and non-negative.")
    if objective == "minimum_variance" and return_weight != 0:
        raise ValueError("return_weight must be zero for the minimum_variance objective.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if isinstance(minimum_observations, bool) or not isinstance(minimum_observations, int) or minimum_observations < 2:
        raise ValueError("minimum_observations must be an integer >= 2.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    prices = _normalise_prices(adjusted_close)
    tickers = [str(ticker) for ticker in prices.columns]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ticker names must be unique after string conversion.")
    daily_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    daily_returns = daily_returns.dropna(how="any")
    if len(daily_returns) < minimum_observations:
        raise ValueError(
            f"Only {len(daily_returns)} common return observations are available; "
            f"at least {minimum_observations} are required."
        )
    returns_values = daily_returns.to_numpy(dtype=float)
    if not np.isfinite(returns_values).all() or (returns_values < -1.0).any():
        raise ValueError("Daily returns must be finite and no lower than -100%.")

    caps, group_by_ticker, group_caps, group_indices = _validate_constraints(
        tickers,
        default_asset_max_weight=default_asset_max_weight,
        asset_max_weights=asset_max_weights,
        groups=groups,
        group_max_weights=group_max_weights,
        risky_budget=risky_budget,
    )

    covariance = daily_returns.cov().to_numpy(dtype=float) * periods_per_year
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    ridge = max(1e-12, float(np.trace(covariance)) / len(tickers) * 1e-10)
    if float(eigenvalues.min()) < ridge:
        ridge += abs(min(0.0, float(eigenvalues.min())))
    covariance = covariance + np.eye(len(tickers)) * ridge
    expected_returns = daily_returns.mean().to_numpy(dtype=float) * periods_per_year

    def project(values: np.ndarray) -> np.ndarray:
        return _project_group_capped_budget(
            values,
            asset_caps=caps,
            group_indices=group_indices,
            group_caps=group_caps,
            budget=risky_budget,
        )

    def objective_value(weights: np.ndarray) -> float:
        return float(
            0.5 * (weights @ covariance @ weights)
            - return_weight * (weights @ expected_returns)
        )

    lipschitz = float(np.linalg.eigvalsh(covariance).max())
    if not math.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("Covariance matrix has no positive curvature.")
    step_size = 1.0 / lipschitz

    weights = project(np.full(len(tickers), risky_budget / len(tickers), dtype=float))
    extrapolated = weights.copy()
    acceleration = 1.0
    converged = False
    iteration_count = 0
    previous_objective = objective_value(weights)

    for iteration in range(1, max_iterations + 1):
        gradient = covariance @ extrapolated - return_weight * expected_returns
        candidate = project(extrapolated - step_size * gradient)
        candidate_objective = objective_value(candidate)

        # Monotone restart prevents an accelerated step from increasing the objective.
        if candidate_objective > previous_objective + 1e-12:
            extrapolated = weights.copy()
            acceleration = 1.0
            gradient = covariance @ extrapolated - return_weight * expected_returns
            candidate = project(extrapolated - step_size * gradient)
            candidate_objective = objective_value(candidate)

        previous_weights = weights
        delta = float(np.linalg.norm(candidate - weights))
        weights = candidate
        previous_objective = candidate_objective
        iteration_count = iteration

        if delta <= tolerance * max(1.0, float(np.linalg.norm(weights))):
            mapping = project(
                weights - step_size * (covariance @ weights - return_weight * expected_returns)
            )
            if float(np.linalg.norm(weights - mapping)) <= tolerance * max(
                1.0, float(np.linalg.norm(weights))
            ):
                converged = True
                break

        next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration**2)) / 2.0
        extrapolated = weights + (
            (acceleration - 1.0) / next_acceleration
        ) * (weights - previous_weights)
        acceleration = next_acceleration

    if not converged:
        raise RuntimeError(
            f"Portfolio optimizer did not converge within {max_iterations} iterations; "
            "increase max_iterations or relax tolerance."
        )

    equal_weights = project(np.full(len(tickers), risky_budget / len(tickers), dtype=float))
    optimized_stats = _portfolio_stats(weights, covariance, expected_returns)
    equal_stats = _portfolio_stats(equal_weights, covariance, expected_returns)
    marginal_risk = covariance @ weights
    variance = optimized_stats["variance_annualized"]
    risk_contributions = (
        weights * marginal_risk / variance if variance > 0 else np.zeros_like(weights)
    )

    output = pd.DataFrame({
        "ticker": tickers,
        "group": [group_by_ticker[ticker] for ticker in tickers],
        "weight": weights,
        "asset_max_weight": caps,
        "group_max_weight": [
            group_caps.get(group_by_ticker[ticker], float(caps[group_indices[group_by_ticker[ticker]]].sum()))
            for ticker in tickers
        ],
        "historical_expected_return_annualized": expected_returns,
        "historical_volatility_annualized": np.sqrt(np.diag(covariance)),
        "risk_contribution": risk_contributions,
        "constrained_equal_weight": equal_weights,
    }).sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)

    group_exposures = {
        group: float(sum(weights[i] for i in indices))
        for group, indices in group_indices.items()
    }
    summary: dict[str, object] = {
        "objective": objective,
        "return_weight": float(return_weight),
        "risky_budget": float(risky_budget),
        "cash_weight": float(1.0 - risky_budget),
        "observations": int(len(daily_returns)),
        "start_date": daily_returns.index.min().date().isoformat(),
        "end_date": daily_returns.index.max().date().isoformat(),
        "tickers": tickers,
        "group_exposures": group_exposures,
        "group_max_weights": group_caps,
        "convergence": {
            "converged": converged,
            "iterations": iteration_count,
            "tolerance": float(tolerance),
            "covariance_ridge": float(ridge),
        },
        "optimized_portfolio": optimized_stats,
        "constrained_equal_weight_baseline": equal_stats,
        "volatility_reduction_vs_constrained_equal_weight": (
            1.0 - optimized_stats["volatility_annualized"] / equal_stats["volatility_annualized"]
            if equal_stats["volatility_annualized"] > 0 else None
        ),
        "methodology": {
            "returns": "common complete-case daily adjusted-close simple returns",
            "covariance": "sample covariance annualized by periods_per_year",
            "expected_returns": "historical arithmetic mean annualized; used only for mean_variance objective",
            "constraints": "long-only, per-asset upper bounds, disjoint group upper bounds, fixed risky budget; residual is cash",
            "execution": "no trading orders or transaction costs are simulated by the optimizer",
        },
        "limitations": [
            "Historical covariance and expected returns are estimates and can change materially across market regimes.",
            "Minimum variance can favor assets that were historically quiet; it does not imply higher future returns.",
            "Mean-variance results are especially sensitive to noisy expected-return estimates and return_weight.",
            "Common complete-case returns can shorten the sample when assets have different listing histories.",
            "Tax, trading costs, liquidity, lot sizes, market impact, and broker execution are not optimized here.",
        ],
    }
    return output, summary


def _load_json_mapping(path: Path | None, description: str) -> dict[str, object]:
    if path is None:
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description} JSON file: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"{description} JSON must contain an object.")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize long-only portfolio weights under asset/group caps; no orders are placed."
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--risky-budget", type=float, default=0.95, help="Total risky allocation; residual is cash.")
    parser.add_argument("--asset-max-weight", type=float, default=0.35, help="Default maximum weight per asset.")
    parser.add_argument("--asset-caps-json", type=Path, help="Optional JSON object: ticker -> max weight.")
    parser.add_argument("--groups-json", type=Path, help="Optional JSON object: ticker -> group name.")
    parser.add_argument("--group-caps-json", type=Path, help="Optional JSON object: group name -> max weight.")
    parser.add_argument("--objective", choices=["minimum_variance", "mean_variance"], default="minimum_variance")
    parser.add_argument("--return-weight", type=float, default=0.0, help="Expected-return weight for mean_variance only.")
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    prices = load_price_panel(args.prices)
    asset_caps = _load_json_mapping(args.asset_caps_json, "asset caps")
    groups = _load_json_mapping(args.groups_json, "groups")
    group_caps = _load_json_mapping(args.group_caps_json, "group caps")
    weights, summary = optimize_portfolio(
        prices,
        risky_budget=args.risky_budget,
        default_asset_max_weight=args.asset_max_weight,
        asset_max_weights={str(k): float(v) for k, v in asset_caps.items()},
        groups={str(k): str(v) for k, v in groups.items()},
        group_max_weights={str(k): float(v) for k, v in group_caps.items()},
        objective=args.objective,
        return_weight=args.return_weight,
        minimum_observations=args.minimum_observations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir / "portfolio_optimized_weights.csv", index=False, float_format="%.10f")
    (args.output_dir / "portfolio_optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 约束组合优化报告",
        "",
        f"- 优化目标：{summary['objective']}",
        f"- 风险资产预算：{summary['risky_budget']:.2%}",
        f"- 现金比例：{summary['cash_weight']:.2%}",
        f"- 历史样本：{summary['start_date']} 至 {summary['end_date']}（{summary['observations']} 个共同收益观测）",
        f"- 优化组合年化波动率：{summary['optimized_portfolio']['volatility_annualized']:.2%}",
        f"- 约束等权组合年化波动率：{summary['constrained_equal_weight_baseline']['volatility_annualized']:.2%}",
        f"- 相对等权组合的历史波动率变化：{summary['volatility_reduction_vs_constrained_equal_weight']}",
        "",
        "## 目标权重",
        "",
        "~~~text",
        weights.to_string(index=False),
        "~~~",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    (args.output_dir / "portfolio_optimization_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
