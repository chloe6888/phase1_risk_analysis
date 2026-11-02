"""Offline tests for portfolio factor exposure and variance decomposition."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import factor_risk as app


def factor_returns(n: int = 420, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=n)
    market = rng.normal(0.0003, 0.008, n)
    semiconductor = rng.normal(0.0005, 0.012, n)
    rates = rng.normal(0.00005, 0.003, n)
    return pd.DataFrame(
        {"MKT": market, "SEMI": semiconductor, "BOND": rates},
        index=index,
    )


def prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    return start * (1.0 + returns).cumprod()


def test_factor_regression_recovers_known_exposures_and_variance_accounting() -> None:
    factors = factor_returns()
    rng = np.random.default_rng(2026)
    noise = rng.normal(0.0, 0.002, len(factors))
    portfolio = pd.Series(
        0.0001 + 1.15 * factors["MKT"] + 0.65 * factors["SEMI"] - 0.30 * factors["BOND"] + noise,
        index=factors.index,
        name="portfolio_return",
    )
    table, summary, aligned = app.estimate_factor_risk(portfolio, factors)
    betas = table.set_index("factor")["beta"]
    assert betas["MKT"] == pytest.approx(1.15, abs=0.08)
    assert betas["SEMI"] == pytest.approx(0.65, abs=0.08)
    assert betas["BOND"] == pytest.approx(-0.30, abs=0.08)
    assert summary["r_squared"] > 0.90
    assert summary["factor_variance_share_sum"] + summary["residual_variance_share"] == pytest.approx(1.0, abs=1e-7)
    assert table.loc[table["factor"] == "BOND", "interpretation"].iloc[0] in {
        "risk_offset", "near_zero_contribution"
    }
    assert len(aligned) == len(factors)


def test_factor_model_aligns_common_dates_and_drops_missing_rows() -> None:
    factors = factor_returns(200)
    portfolio = pd.Series(
        0.8 * factors["MKT"] + 0.1 * factors["SEMI"],
        index=factors.index,
    )
    factors.loc[factors.index[10], "MKT"] = np.nan
    portfolio.loc[portfolio.index[20]] = np.nan
    table, summary, aligned = app.estimate_factor_risk(
        portfolio, factors, minimum_observations=100,
    )
    assert summary["observations"] == 198
    assert aligned.index.is_monotonic_increasing
    assert not aligned.isna().any().any()


def test_factor_model_warns_on_correlated_proxies() -> None:
    factors = factor_returns(240)
    rng = np.random.default_rng(3)
    factors["MKT_COPY"] = factors["MKT"] * 0.8 + rng.normal(0, 0.00015, len(factors))
    portfolio = pd.Series(0.5 * factors["MKT"] + factors["SEMI"], index=factors.index)
    table, summary, aligned = app.estimate_factor_risk(
        portfolio, factors, minimum_observations=100,
        max_pairwise_correlation_warning=0.90,
    )
    assert summary["high_correlation_pairs"]
    assert any("highly correlated" in warning for warning in summary["warnings"])


def test_factor_model_rejects_perfect_collinearity_and_constant_factors() -> None:
    factors = factor_returns(200)
    portfolio = pd.Series(factors["MKT"] * 1.5, index=factors.index)
    duplicate = factors[["MKT"]].copy()
    duplicate["MKT_COPY"] = duplicate["MKT"]
    with pytest.raises(ValueError, match="rank-deficient"):
        app.estimate_factor_risk(portfolio, duplicate, minimum_observations=100)
    constant = pd.DataFrame({"ZERO": np.zeros(len(factors))}, index=factors.index)
    with pytest.raises(ValueError, match="no variation"):
        app.estimate_factor_risk(portfolio, constant, minimum_observations=100)


def test_weighted_portfolio_returns_use_cash_as_zero_return_and_reject_bad_weights() -> None:
    factors = factor_returns(100)
    prices = prices_from_returns(factors[["MKT", "SEMI"]].rename(columns={"MKT": "AAA", "SEMI": "BBB"}))
    returns, summary = app.build_portfolio_returns(prices, {"AAA": 0.6, "BBB": 0.2})
    expected = prices.pct_change(fill_method=None).dropna().mul(
        pd.Series({"AAA": 0.6, "BBB": 0.2}), axis=1
    ).sum(axis=1)
    pd.testing.assert_series_equal(returns, expected.rename("portfolio_return"))
    assert summary["cash_weight"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no more than 100"):
        app.normalize_portfolio_weights({"AAA": 0.8, "BBB": 0.3})


def test_weighted_portfolio_rejects_missing_tickers_and_duplicate_rows() -> None:
    prices = prices_from_returns(factor_returns(120)[["MKT"]].rename(columns={"MKT": "AAA"}))
    with pytest.raises(ValueError, match="missing from the asset price panel"):
        app.build_portfolio_returns(prices, {"NOPE": 0.5})
    duplicate = pd.DataFrame({"ticker": ["AAA", "AAA"], "weight": [0.2, 0.2]})
    with pytest.raises(ValueError, match="duplicate"):
        app.normalize_portfolio_weights(duplicate)


def test_factor_model_rejects_short_history_and_invalid_returns() -> None:
    factors = factor_returns(40)
    portfolio = pd.Series(factors["MKT"], index=factors.index)
    with pytest.raises(ValueError, match="at least"):
        app.estimate_factor_risk(portfolio, factors, minimum_observations=60)
    invalid = factors.copy()
    invalid.iloc[0, 0] = -1.1
    with pytest.raises(ValueError, match="lower than -100"):
        app.estimate_factor_risk(portfolio, invalid)


def test_cli_writes_factor_exposure_and_report_artifacts(tmp_path: Path) -> None:
    factors = factor_returns(260, seed=44)
    rng = np.random.default_rng(55)
    asset_returns = pd.DataFrame({
        "AAA": 0.0001 + 1.2 * factors["MKT"] + 0.6 * factors["SEMI"] + rng.normal(0, 0.002, len(factors)),
        "BBB": 0.00005 + 0.8 * factors["MKT"] - 0.25 * factors["BOND"] + rng.normal(0, 0.002, len(factors)),
    }, index=factors.index)
    asset_prices = prices_from_returns(asset_returns)
    factor_prices = prices_from_returns(factors)
    asset_path = tmp_path / "asset_prices.csv"
    factor_path = tmp_path / "factor_prices.csv"
    weights_path = tmp_path / "weights.csv"
    asset_prices.to_csv(asset_path, index_label="Date")
    factor_prices.to_csv(factor_path, index_label="Date")
    pd.DataFrame({"ticker": ["AAA", "BBB"], "weight": [0.6, 0.4]}).to_csv(weights_path, index=False)
    output_dir = tmp_path / "artifacts"
    script = Path(__file__).resolve().parents[1] / "factor_risk.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--asset-prices", str(asset_path),
            "--factor-prices", str(factor_path),
            "--weights-csv", str(weights_path),
            "--minimum-observations", "100",
            "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "portfolio_factor_summary.json").read_text())
    assert summary["observations"] >= 250
    assert len(summary["factor_names"]) == 3
    for filename in (
        "portfolio_factor_exposures.csv",
        "portfolio_factor_daily_returns.csv",
        "portfolio_factor_summary.json",
        "portfolio_factor_report.md",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert "r_squared" in result.stdout
