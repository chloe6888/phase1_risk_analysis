"""Offline tests for read-only portfolio risk monitoring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import risk_monitor as app


def synthetic_prices(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    index = pd.bdate_range("2024-01-02", periods=n)
    a = 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))
    b = 80 * np.cumprod(1 + rng.normal(0.0004, 0.007, n))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=index)


def test_risk_monitor_generates_concentration_and_risk_alerts() -> None:
    close = synthetic_prices()
    holdings = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "shares": [90, 10],
        "risk_group": ["technology", "technology"],
    })
    positions, alerts, summary = app.analyze_risk_snapshot(
        holdings, close, adjusted_prices=close, cash=0,
        group_max_weights={"technology": 0.7},
        single_name_limit=0.6, volatility_limit_annual=0.05,
        drawdown_limit=0.05, volatility_window=60,
    )
    assert summary["net_equity"] > 0
    assert summary["portfolio_return_observations"] >= 200
    assert positions["portfolio_weight"].sum() == pytest.approx(1.0)
    kinds = set(alerts["kind"])
    assert {"single_name_concentration", "group_concentration", "portfolio_volatility", "portfolio_drawdown"}.issubset(kinds)


def test_risk_monitor_can_return_no_alerts_with_relaxed_limits() -> None:
    close = synthetic_prices()
    holdings = pd.DataFrame({"ticker": ["AAA", "BBB"], "shares": [1, 1], "risk_group": ["mixed", "mixed"]})
    positions, alerts, summary = app.analyze_risk_snapshot(
        holdings, close, cash=1000, group_max_weights={"mixed": 1.0},
        single_name_limit=1.0, volatility_limit_annual=1.0,
        drawdown_limit=1.0, max_data_age_calendar_days=None,
    )
    assert alerts.empty
    assert summary["alert_count"] == 0
    assert positions["portfolio_weight"].sum() < 1.0


def test_risk_monitor_flags_short_history_and_stale_data() -> None:
    close = synthetic_prices(10)
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [1]})
    _, alerts, summary = app.analyze_risk_snapshot(
        holdings, close, minimum_observations=20,
        max_data_age_calendar_days=1,
    )
    assert "insufficient_history" in set(alerts["kind"])
    assert summary["portfolio_return_observations"] == 9


def test_risk_monitor_rejects_missing_price_and_bad_limits() -> None:
    close = synthetic_prices()
    holdings = pd.DataFrame({"ticker": ["MISSING"], "shares": [1]})
    with pytest.raises(ValueError, match="Missing close-price history"):
        app.analyze_risk_snapshot(holdings, close)
    holdings = pd.DataFrame({"ticker": ["AAA"], "shares": [1]})
    with pytest.raises(ValueError, match="single_name_limit"):
        app.analyze_risk_snapshot(holdings, close, single_name_limit=1.5)


def test_risk_monitor_cli_writes_snapshot_and_alerts(tmp_path: Path) -> None:
    close = synthetic_prices()
    price_path = tmp_path / "close.csv"
    adjusted_path = tmp_path / "adjusted.csv"
    close.to_csv(price_path)
    close.to_csv(adjusted_path)
    holdings_path = tmp_path / "holdings.csv"
    pd.DataFrame({
        "ticker": ["AAA", "BBB"], "shares": [10, 1],
        "risk_group": ["tech", "tech"],
    }).to_csv(holdings_path, index=False)
    group_caps = tmp_path / "group_caps.json"
    group_caps.write_text(json.dumps({"tech": 0.5}), encoding="utf-8")
    output_dir = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "risk_monitor.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--holdings-csv", str(holdings_path),
            "--prices", str(price_path), "--adjusted-prices", str(adjusted_path),
            "--single-name-limit", "0.3", "--group-caps-json", str(group_caps),
            "--max-data-age-days", "9999", "--output-dir", str(output_dir),
        ],
        check=True, capture_output=True, text=True,
    )
    snapshot = json.loads((output_dir / "risk_monitor_snapshot.json").read_text())
    assert snapshot["position_count"] == 2
    for filename in (
        "risk_monitor_positions.csv", "risk_monitor_alerts.csv",
        "risk_monitor_daily_path.csv", "risk_monitor_snapshot.json",
        "risk_monitor_report.md",
    ):
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0
    assert "alert_count" in result.stdout
