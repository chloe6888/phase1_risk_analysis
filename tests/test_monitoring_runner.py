"""Offline tests for repeatable daily/weekly monitoring and alert transitions."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import monitoring_runner as app


def synthetic_prices(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20261102)
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    returns = rng.normal(0.0003, 0.018, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"AAA": prices}, index=index)


def holdings_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA"],
        "shares": [10],
        "risk_group": ["technology"],
    })


def test_alert_state_transitions_trigger_persist_and_resolve(tmp_path: Path) -> None:
    prices = synthetic_prices()
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    first = app.run_monitoring(
        holdings_frame(), prices, prices,
        output_dir=tmp_path,
        single_name_limit=0.5,
        volatility_limit_annual=1.0,
        drawdown_limit=1.0,
        max_data_age_calendar_days=7,
        now=now,
    )
    assert first["triggered_alert_count"] == 1
    assert first["active_alert_count"] == 1

    second = app.run_monitoring(
        holdings_frame(), prices, prices,
        output_dir=tmp_path,
        single_name_limit=0.5,
        volatility_limit_annual=1.0,
        drawdown_limit=1.0,
        max_data_age_calendar_days=7,
        now=now + timedelta(minutes=1),
    )
    assert second["triggered_alert_count"] == 0
    assert second["persisting_alert_count"] == 1
    assert second["active_alert_count"] == 1

    third = app.run_monitoring(
        holdings_frame(), prices, prices,
        output_dir=tmp_path,
        single_name_limit=1.0,
        volatility_limit_annual=1.0,
        drawdown_limit=1.0,
        max_data_age_calendar_days=7,
        now=now + timedelta(minutes=2),
    )
    assert third["resolved_alert_count"] == 1
    assert third["active_alert_count"] == 0
    events = json.loads((tmp_path / "latest_alert_events.json").read_text())
    assert events["events"][0]["event_type"] == "RESOLVED"


def test_alert_events_are_appended_to_durable_history(tmp_path: Path) -> None:
    prices = synthetic_prices()
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    for minute, single_limit in ((0, 0.5), (1, 0.5), (2, 1.0)):
        app.run_monitoring(
            holdings_frame(), prices, prices,
            output_dir=tmp_path,
            single_name_limit=single_limit,
            volatility_limit_annual=1.0,
            drawdown_limit=1.0,
            max_data_age_calendar_days=7,
            now=now + timedelta(minutes=minute),
        )
    history = tmp_path / "alert_event_history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["event_type"] for row in rows] == ["TRIGGERED", "PERSISTING", "RESOLVED"]
    assert all(row["snapshot_id"] for row in rows)


def test_old_lock_is_not_reclaimed_while_owner_process_is_alive(tmp_path: Path) -> None:
    import os
    import time

    lock = tmp_path / ".monitoring.lock"
    lock.write_text(str(os.getpid()), encoding="ascii")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    with pytest.raises(RuntimeError, match="另一个风险监控任务"):
        app._acquire_lock(tmp_path)


def test_dead_owner_stale_lock_can_be_reclaimed(tmp_path: Path) -> None:
    import os
    import time

    lock = tmp_path / ".monitoring.lock"
    lock.write_text("999999999", encoding="ascii")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    fd, path = app._acquire_lock(tmp_path)
    try:
        assert path == lock
        assert int(lock.read_text()) > 0
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def test_weekly_summary_uses_saved_snapshots_and_reports_observed_range(tmp_path: Path) -> None:
    prices = synthetic_prices()
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    app.run_monitoring(
        holdings_frame(), prices, prices,
        output_dir=tmp_path,
        single_name_limit=0.5,
        volatility_limit_annual=1.0,
        drawdown_limit=1.0,
        max_data_age_calendar_days=7,
        now=now - timedelta(days=2),
    )
    result = app.run_monitoring(
        holdings_frame(), prices, prices,
        output_dir=tmp_path,
        single_name_limit=0.5,
        volatility_limit_annual=1.0,
        drawdown_limit=1.0,
        max_data_age_calendar_days=7,
        mode="weekly",
        history_days=7,
        now=now,
    )
    weekly = result["weekly_summary"]
    assert weekly["snapshot_count"] == 2
    assert weekly["history_days"] == 7
    assert weekly["latest_active_alert_count"] == 1
    assert (tmp_path / "monitoring_weekly_report.md").is_file()
    assert (tmp_path / "monitoring_weekly_summary.json").is_file()


def test_runner_rejects_invalid_mode_and_history_window(tmp_path: Path) -> None:
    prices = synthetic_prices()
    with pytest.raises(ValueError, match="mode"):
        app.run_monitoring(holdings_frame(), prices, prices, output_dir=tmp_path, mode="monthly")
    with pytest.raises(ValueError, match="history_days"):
        app.run_monitoring(holdings_frame(), prices, prices, output_dir=tmp_path, history_days=0)


def test_monitor_lock_prevents_overlapping_runs(tmp_path: Path) -> None:
    lock = tmp_path / ".monitoring.lock"
    lock.write_text("999999", encoding="ascii")
    with pytest.raises(RuntimeError, match="另一个风险监控任务"):
        app._acquire_lock(tmp_path)


def test_cli_writes_snapshots_daily_and_weekly_reports(tmp_path: Path) -> None:
    prices = synthetic_prices(100)
    price_path = tmp_path / "prices.csv"
    adjusted_path = tmp_path / "adjusted.csv"
    holdings_path = tmp_path / "holdings.csv"
    prices.to_csv(price_path, index_label="Date")
    prices.to_csv(adjusted_path, index_label="Date")
    holdings_frame().to_csv(holdings_path, index=False)
    output_dir = tmp_path / "monitoring"
    script = Path(__file__).resolve().parents[1] / "monitoring_runner.py"
    base = [
        sys.executable, str(script),
        "--holdings-csv", str(holdings_path),
        "--prices", str(price_path),
        "--adjusted-prices", str(adjusted_path),
        "--output-dir", str(output_dir),
        "--max-data-age-days", "9999",
        "--single-name-limit", "0.5",
        "--volatility-limit", "1.0",
        "--drawdown-limit", "1.0",
    ]
    daily = subprocess.run(base, check=True, capture_output=True, text=True)
    first = json.loads(daily.stdout)
    assert first["mode"] == "daily"
    assert (output_dir / "latest_snapshot.json").is_file()
    assert (output_dir / "monitoring_daily_report.md").is_file()

    weekly = subprocess.run(base + ["--mode", "weekly"], check=True, capture_output=True, text=True)
    second = json.loads(weekly.stdout)
    assert second["mode"] == "weekly"
    assert second["weekly_summary"]["snapshot_count"] == 2
    assert (output_dir / "monitoring_weekly_report.md").is_file()
