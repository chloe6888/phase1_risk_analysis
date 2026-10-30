"""Offline tests for portfolio-level risk analytics."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import portfolio_risk as app


def sample_prices() -> pd.DataFrame:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.03, -0.01, 0.02, 0.00],
            "BBB": [0.00, 0.01, -0.01, 0.02, -0.02, 0.01],
        },
        index=pd.bdate_range("2024-01-02", periods=6),
    )
    return 100 * (1 + returns).cumprod()


def test_normalize_weights_scales_to_one_and_checks_tickers() -> None:
    weights = app.normalize_weights({"AAA": 2, "BBB": 2}, ["AAA", "BBB"])
    assert weights.sum() == pytest.approx(1.0)
    assert weights["AAA"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="absent from the price panel"):
        app.normalize_weights({"CCC": 1}, ["AAA", "BBB"])


@pytest.mark.parametrize("weights", [{}, {"AAA": -0.1, "BBB": 1.1}, {"AAA": float("nan")}, {"AAA": 0}])
def test_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ValueError):
        app.normalize_weights(weights)


def test_calculate_portfolio_risk_metrics_and_contributions() -> None:
    prices = sample_prices()
    summary, contributions, drawdown, daily = app.calculate_portfolio_risk(prices, {"AAA": 0.6, "BBB": 0.4})
    assert summary["observations"] == 5
    assert summary["start_date"] == "2024-01-03"
    assert summary["normalized_weights"] == {"AAA": 0.6, "BBB": 0.4}
    assert summary["annualized_volatility"] > 0
    assert summary["max_drawdown"] <= 0
    assert summary["historical_var_one_period"] == pytest.approx(-daily["portfolio_return"].quantile(0.05))
    assert summary["historical_cvar_one_period"] >= summary["historical_var_one_period"] - 1e-12
    assert contributions["weight"].sum() == pytest.approx(1.0)
    assert contributions["component_contribution_to_volatility_annualized"].sum() == pytest.approx(
        summary["annualized_volatility"], rel=1e-8, abs=1e-10
    )
    assert "drawdown" in drawdown
    assert "portfolio_return" in daily


def test_calculate_risk_ignores_missing_unweighted_asset() -> None:
    prices = sample_prices()
    prices["UNUSED"] = np.nan
    summary, _, _, _ = app.calculate_portfolio_risk(prices, {"AAA": 1.0})
    assert summary["observations"] == len(prices) - 1
    assert summary["normalized_weights"] == {"AAA": 1.0}


def test_calculate_risk_uses_common_dates_for_weighted_assets() -> None:
    prices = sample_prices()
    prices.loc[prices.index[3], "BBB"] = np.nan
    summary, _, _, _ = app.calculate_portfolio_risk(prices, {"AAA": 0.5, "BBB": 0.5})
    assert summary["observations"] == 3


def test_reject_invalid_prices_and_insufficient_history() -> None:
    prices = sample_prices()
    prices.iloc[0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        app.calculate_portfolio_risk(prices, {"AAA": 1.0})
    with pytest.raises(ValueError, match="two common return observations"):
        app.calculate_portfolio_risk(sample_prices().iloc[:2], {"AAA": 1.0})


def test_rolling_risk_returns_each_requested_window() -> None:
    returns = pd.Series(
        np.tile([0.01, -0.005, 0.002, -0.003], 35),
        index=pd.bdate_range("2024-01-01", periods=140),
    )
    result = app.calculate_rolling_risk(returns, windows=(20, 60))
    assert set(result["window_observations"].unique()) == {20, 60}
    assert result["rolling_annualized_volatility"].notna().all()
    assert (result["rolling_max_drawdown"] <= 0).all()


def test_rolling_risk_rejects_invalid_windows_and_short_history() -> None:
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(ValueError, match="at least 2"):
        app.calculate_rolling_risk(returns, windows=(1,))
    with pytest.raises(ValueError, match="Not enough"):
        app.calculate_rolling_risk(returns, windows=(5,))


def test_stress_scenarios_weighted_return_and_dollar_impact() -> None:
    result = app.run_stress_scenarios(
        {"AAA": 0.6, "BBB": 0.4},
        {"selloff": {"AAA": -0.2, "BBB": -0.1}, "AAA_only": {"AAA": -0.1}},
        portfolio_value=10_000,
    )
    assert result.loc["selloff", "portfolio_return"] == pytest.approx(-0.16)
    assert result.loc["selloff", "estimated_pnl"] == pytest.approx(-1600)
    assert result.loc["AAA_only", "portfolio_return"] == pytest.approx(-0.06)


def test_stress_scenarios_reject_invalid_shock() -> None:
    with pytest.raises(ValueError, match="no lower than -100%"):
        app.run_stress_scenarios({"AAA": 1}, {"bad": {"AAA": -1.1}})


def test_load_price_panel_checks_duplicates_and_values(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame({"AAA": [100, 101]}, index=pd.to_datetime(["2024-01-01", "2024-01-01"])).to_csv(path)
    with pytest.raises(ValueError, match="duplicate dates"):
        app.load_price_panel(path)
    pd.DataFrame({"AAA": [100, -1]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_csv(path)
    with pytest.raises(ValueError, match="positive"):
        app.load_price_panel(path)


def test_cli_creates_portfolio_outputs(tmp_path) -> None:
    prices_path = tmp_path / "prices.csv"
    sample_prices().to_csv(prices_path, float_format="%.8f")
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable, str(app.ROOT / "portfolio_risk.py"),
            "--prices", str(prices_path), "--weights", "AAA=0.6,BBB=0.4",
            "--output-dir", str(output_dir), "--window", "2", "--window", "3",
            "--portfolio-value", "10000", "--scenario", "selloff:AAA=-0.2,BBB=-0.1",
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "portfolio_risk_summary.json").read_text())
    assert summary["observations"] == 5
    for filename in (
        "portfolio_risk_summary.json", "portfolio_risk_contributions.csv",
        "portfolio_drawdown.csv", "portfolio_daily_returns.csv",
        "portfolio_rolling_risk.csv", "portfolio_stress_tests.csv",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert '"max_drawdown"' in result.stdout


def test_drawdown_includes_initial_capital_and_peak_before_trough() -> None:
    returns = pd.Series(
        [-0.20, 0.10, 0.10, 0.40],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    result = app.max_drawdown_series(returns)
    assert result.loc[pd.Timestamp("2024-01-02"), "drawdown"] == pytest.approx(-0.20)
    prices = (1.0 + returns).cumprod().to_frame("AAA") * 100
    prices = pd.concat([
        pd.DataFrame({"AAA": [100.0]}, index=[pd.Timestamp("2024-01-01")]),
        prices,
    ])
    summary, _, _, _ = app.calculate_portfolio_risk(prices, {"AAA": 1.0})
    assert summary["max_drawdown"] == pytest.approx(-0.20)
    assert summary["max_drawdown_peak_date"] == "2024-01-02"
    assert summary["max_drawdown_trough_date"] == "2024-01-02"


def test_cli_serializes_constant_return_series_without_nan_json(tmp_path) -> None:
    prices_path = tmp_path / "flat_prices.csv"
    pd.DataFrame(
        {"AAA": [100.0, 100.0, 100.0, 100.0]},
        index=pd.bdate_range("2024-01-01", periods=4),
    ).to_csv(prices_path)
    output_dir = tmp_path / "flat_output"
    subprocess.run(
        [
            sys.executable, str(app.ROOT / "portfolio_risk.py"),
            "--prices", str(prices_path), "--weights", "AAA=1",
            "--output-dir", str(output_dir), "--window", "2",
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "portfolio_risk_summary.json").read_text())
    assert summary["annualized_return"] == pytest.approx(0.0)
    assert summary["sharpe_ratio"] is None
    assert summary["sortino_ratio"] is None


def test_cli_does_not_fail_when_default_rolling_windows_exceed_short_history(tmp_path) -> None:
    prices_path = tmp_path / "short_prices.csv"
    pd.DataFrame(
        {
            "AAA": [100, 101, 99, 102, 101, 103, 104],
            "BBB": [100, 99, 101, 102, 100, 101, 103],
        },
        index=pd.bdate_range("2026-07-01", periods=7),
    ).to_csv(prices_path)
    output_dir = tmp_path / "short_output"
    result = subprocess.run(
        [
            sys.executable, str(app.ROOT / "portfolio_risk.py"),
            "--prices", str(prices_path), "--weights", "AAA=0.5,BBB=0.5",
            "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((output_dir / "portfolio_risk_summary.json").read_text())
    assert summary["observations"] == 6
    assert summary["rolling_windows_used"] == []
    assert "Only 6 common-return observations" in summary["rolling_risk_note"]
    assert (output_dir / "portfolio_rolling_risk.csv").exists()
    assert "insufficient common-return history" in result.stdout.lower()
