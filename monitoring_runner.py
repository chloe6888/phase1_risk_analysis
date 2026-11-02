#!/usr/bin/env python3
"""Repeatable daily/weekly portfolio-risk monitoring with alert state transitions.

This runner is designed to be invoked by cron or Windows Task Scheduler.
It does not keep a background scheduler alive, send broker orders, or expose
private holdings over a network. Each run writes an immutable timestamped
snapshot and updates a small local state file for alert transition detection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from portfolio_risk import DEFAULT_OUTPUT_DIR, DEFAULT_PRICE_PANEL, ROOT, load_price_panel
from risk_monitor import DEFAULT_CLOSE_PRICES, analyze_risk_snapshot

DEFAULT_MONITOR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "monitoring"
DEFAULT_HOLDINGS_CSV = ROOT / "data" / "private" / "holdings_snapshot.csv"


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read monitoring state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Monitoring state {path} must contain a JSON object.")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _acquire_lock(output_dir: Path, *, stale_after_seconds: int = 3600) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".monitoring.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, str(os.getpid()).encode("ascii"))
        return fd, lock_path
    except FileExistsError as exc:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age > stale_after_seconds:
            try:
                lock_path.unlink()
            except OSError:
                pass
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(fd, str(os.getpid()).encode("ascii"))
                return fd, lock_path
            except FileExistsError:
                pass
        raise RuntimeError(
            "另一个风险监控任务正在运行；请勿重复启动。若确认进程已退出，可检查并清理过期 .monitoring.lock。"
        ) from exc


def _alert_key(alert: Mapping[str, Any]) -> str:
    return "|".join([
        str(alert.get("kind") or ""),
        str(alert.get("ticker") or ""),
        str(alert.get("group") or ""),
    ])


def _alert_events(
    current_alerts: list[dict[str, Any]],
    previous_alerts: list[dict[str, Any]],
    *,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = {_alert_key(item): item for item in current_alerts}
    previous = {_alert_key(item): item for item in previous_alerts}
    events: list[dict[str, Any]] = []
    for key, alert in current.items():
        event_type = "TRIGGERED" if key not in previous else "PERSISTING"
        events.append({
            "event_type": event_type,
            "alert_key": key,
            "created_at": created_at,
            "level": alert.get("level"),
            "kind": alert.get("kind"),
            "ticker": alert.get("ticker"),
            "group": alert.get("group"),
            "observed": alert.get("observed"),
            "limit": alert.get("limit"),
            "message": alert.get("message"),
        })
    for key, alert in previous.items():
        if key not in current:
            events.append({
                "event_type": "RESOLVED",
                "alert_key": key,
                "created_at": created_at,
                "level": alert.get("level"),
                "kind": alert.get("kind"),
                "ticker": alert.get("ticker"),
                "group": alert.get("group"),
                "observed": alert.get("observed"),
                "limit": alert.get("limit"),
                "message": f"告警已解除：{alert.get('message', key)}",
            })
    events.sort(key=lambda row: (str(row.get("event_type")), str(row.get("alert_key"))))
    return events, list(current.values())


def _write_daily_report(snapshot: Mapping[str, Any]) -> str:
    summary = snapshot["summary"]
    events = snapshot["alert_events"]
    lines = [
        "# 每日组合风险监控",
        "",
        f"- 运行时间（UTC）：{snapshot['created_at']}",
        f"- 最新行情日期：{summary.get('latest_data_date')}",
        f"- 账户净资产估值：{summary.get('net_equity')}",
        f"- 历史年化波动率：{summary.get('annualized_volatility')}",
        f"- 历史最大回撤：{summary.get('max_drawdown')}",
        f"- 当前活跃告警：{len(snapshot['active_alerts'])}",
        "",
        "## 告警状态变化",
        "",
    ]
    if not events:
        lines.append("本次运行没有新增、持续或解除的告警。")
    else:
        labels = {
            "TRIGGERED": "首次触发",
            "PERSISTING": "持续存在",
            "RESOLVED": "已经解除",
        }
        lines.extend(
            f"- **{labels.get(event['event_type'], event['event_type'])}**：{event.get('message')}"
            for event in events
        )
    lines += [
        "",
        "## 当前活跃告警",
        "",
    ]
    if snapshot["active_alerts"]:
        lines.extend(f"- {item.get('message')}" for item in snapshot["active_alerts"])
    else:
        lines.append("当前没有活跃告警。")
    lines += [
        "",
        "## 使用限制",
        "",
        *[f"- {item}" for item in summary.get("limitations", [])],
        "",
        "告警只写入本地事件文件，不会自动发送邮件、聊天消息或交易订单。",
        "",
    ]
    return "\n".join(lines)


def _weekly_summary(
    snapshots: list[dict[str, Any]],
    *,
    created_at: str,
    history_days: int,
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("No monitoring snapshots are available for the weekly summary.")
    volatilities = [
        float(row["summary"]["annualized_volatility"])
        for row in snapshots
        if row["summary"].get("annualized_volatility") is not None
    ]
    drawdowns = [
        abs(float(row["summary"]["max_drawdown"]))
        for row in snapshots
        if row["summary"].get("max_drawdown") is not None
    ]
    alert_events = [
        event
        for row in snapshots
        for event in row.get("alert_events", [])
    ]
    latest = max(snapshots, key=lambda row: str(row["created_at"]))
    triggered = sum(event.get("event_type") == "TRIGGERED" for event in alert_events)
    resolved = sum(event.get("event_type") == "RESOLVED" for event in alert_events)
    return {
        "created_at": created_at,
        "history_days": int(history_days),
        "snapshot_count": int(len(snapshots)),
        "period_start": min(str(row["created_at"]) for row in snapshots),
        "period_end": max(str(row["created_at"]) for row in snapshots),
        "latest_snapshot_id": latest["snapshot_id"],
        "latest_net_equity": latest["summary"].get("net_equity"),
        "latest_annualized_volatility": latest["summary"].get("annualized_volatility"),
        "latest_max_drawdown": latest["summary"].get("max_drawdown"),
        "minimum_observed_annualized_volatility": min(volatilities) if volatilities else None,
        "maximum_observed_annualized_volatility": max(volatilities) if volatilities else None,
        "maximum_observed_drawdown_abs": max(drawdowns) if drawdowns else None,
        "triggered_event_count": int(triggered),
        "resolved_event_count": int(resolved),
        "latest_active_alert_count": int(len(latest.get("active_alerts", []))),
        "latest_active_alerts": latest.get("active_alerts", []),
        "caveat": "Weekly extrema summarize available snapshots only; they are not intraday or continuously sampled risk limits.",
    }


def _render_weekly_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# 近 7 日组合风险周报",
        "",
        f"- 汇总时间（UTC）：{summary['created_at']}",
        f"- 汇总快照数：{summary['snapshot_count']}",
        f"- 实际覆盖：{summary['period_start']} 至 {summary['period_end']}",
        f"- 最新账户净资产估值：{summary.get('latest_net_equity')}",
        f"- 最新年化波动率：{summary.get('latest_annualized_volatility')}",
        f"- 最新历史最大回撤：{summary.get('latest_max_drawdown')}",
        f"- 本期首次触发告警数：{summary['triggered_event_count']}",
        f"- 本期解除告警数：{summary['resolved_event_count']}",
        f"- 最新活跃告警数：{summary['latest_active_alert_count']}",
        "",
        "## 周内风险区间（基于已保存快照）",
        "",
        f"- 年化波动率最小值：{summary.get('minimum_observed_annualized_volatility')}",
        f"- 年化波动率最大值：{summary.get('maximum_observed_annualized_volatility')}",
        f"- 最大历史回撤绝对值：{summary.get('maximum_observed_drawdown_abs')}",
        "",
        "## 当前活跃告警",
        "",
    ]
    alerts = summary.get("latest_active_alerts", [])
    if alerts:
        lines.extend(f"- {item.get('message')}" for item in alerts)
    else:
        lines.append("当前没有活跃告警。")
    lines += [
        "",
        f"**说明：** {summary['caveat']}",
        "",
    ]
    return "\n".join(lines)


def run_monitoring(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame,
    adjusted_prices: pd.DataFrame,
    *,
    output_dir: Path,
    cash: float = 0.0,
    groups: Mapping[str, str] | None = None,
    group_max_weights: Mapping[str, float] | None = None,
    single_name_limit: float = 0.15,
    volatility_limit_annual: float = 0.40,
    drawdown_limit: float = 0.25,
    volatility_window: int = 60,
    minimum_observations: int = 20,
    max_data_age_calendar_days: int | None = 5,
    mode: str = "daily",
    history_days: int = 7,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one monitor snapshot, alert transition check, and daily/weekly report."""
    if mode not in {"daily", "weekly"}:
        raise ValueError("mode must be 'daily' or 'weekly'.")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError("history_days must be a positive integer.")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    current_time = current_time.astimezone(timezone.utc)
    created_at = current_time.isoformat()
    snapshot_id = current_time.strftime("%Y%m%dT%H%M%S%fZ")

    target_dir = Path(output_dir)
    snapshots_dir = target_dir / "snapshots"
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    lock_fd, lock_path = _acquire_lock(target_dir)
    try:
        positions, alerts, monitor_summary = analyze_risk_snapshot(
            holdings,
            close_prices,
            adjusted_prices=adjusted_prices,
            cash=cash,
            groups=groups,
            group_max_weights=group_max_weights,
            single_name_limit=single_name_limit,
            volatility_limit_annual=volatility_limit_annual,
            drawdown_limit=drawdown_limit,
            volatility_window=volatility_window,
            minimum_observations=minimum_observations,
            max_data_age_calendar_days=max_data_age_calendar_days,
        )
        daily_path = monitor_summary.pop("_daily_path")
        alert_rows = _records(alerts)
        state_path = target_dir / "monitoring_state.json"
        previous_state = _read_json(state_path) or {}
        previous_alerts = previous_state.get("active_alerts", [])
        if not isinstance(previous_alerts, list):
            raise ValueError("monitoring_state.json active_alerts must be a list.")
        events, active_alerts = _alert_events(
            alert_rows, previous_alerts, created_at=created_at,
        )
        snapshot = {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "mode": mode,
            "summary": monitor_summary,
            "positions": _records(positions),
            "active_alerts": active_alerts,
            "alert_events": events,
        }
        snapshot_path = snapshots_dir / f"snapshot_{snapshot_id}.json"
        _atomic_write_json(snapshot_path, snapshot)
        _atomic_write_json(target_dir / "latest_snapshot.json", snapshot)
        _atomic_write_json(target_dir / "latest_alert_events.json", {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "events": events,
            "active_alerts": active_alerts,
        })
        _atomic_write_json(state_path, {
            "version": 1,
            "updated_at": created_at,
            "last_snapshot_id": snapshot_id,
            "active_alerts": active_alerts,
        })
        positions.to_csv(target_dir / "monitoring_positions.csv", index=False, float_format="%.10f")
        alerts.to_csv(target_dir / "monitoring_active_alerts.csv", index=False)
        daily_path.to_csv(target_dir / "monitoring_daily_path.csv", float_format="%.10f")

        daily_report = _write_daily_report(snapshot)
        (target_dir / "monitoring_daily_report.md").write_text(daily_report, encoding="utf-8")

        weekly_summary = None
        if mode == "weekly":
            earliest = current_time - timedelta(days=history_days)
            history_records: list[dict[str, Any]] = []
            for path in sorted(snapshots_dir.glob("snapshot_*.json")):
                record = _read_json(path)
                if not record:
                    continue
                try:
                    recorded_at = datetime.fromisoformat(str(record["created_at"]))
                except (KeyError, ValueError, TypeError):
                    continue
                if recorded_at.tzinfo is None:
                    recorded_at = recorded_at.replace(tzinfo=timezone.utc)
                if earliest <= recorded_at.astimezone(timezone.utc) <= current_time:
                    history_records.append(record)
            weekly_summary = _weekly_summary(
                history_records,
                created_at=created_at,
                history_days=history_days,
            )
            _atomic_write_json(target_dir / "monitoring_weekly_summary.json", weekly_summary)
            (target_dir / "monitoring_weekly_report.md").write_text(
                _render_weekly_report(weekly_summary), encoding="utf-8",
            )

        return {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "snapshot_path": str(snapshot_path),
            "active_alert_count": len(active_alerts),
            "triggered_alert_count": sum(event["event_type"] == "TRIGGERED" for event in events),
            "persisting_alert_count": sum(event["event_type"] == "PERSISTING" for event in events),
            "resolved_alert_count": sum(event["event_type"] == "RESOLVED" for event in events),
            "mode": mode,
            "weekly_summary": weekly_summary,
            "summary": monitor_summary,
        }
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_json_mapping(path: Path | None, label: str) -> dict[str, Any]:
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
        description="Run one persistent daily/weekly risk-monitor snapshot; suitable for cron or Task Scheduler."
    )
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS_CSV)
    parser.add_argument("--prices", type=Path, default=DEFAULT_CLOSE_PRICES)
    parser.add_argument("--adjusted-prices", type=Path, default=DEFAULT_PRICE_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MONITOR_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--groups-json", type=Path)
    parser.add_argument("--group-caps-json", type=Path)
    parser.add_argument("--single-name-limit", type=float, default=0.15)
    parser.add_argument("--volatility-limit", type=float, default=0.40)
    parser.add_argument("--drawdown-limit", type=float, default=0.25)
    parser.add_argument("--volatility-window", type=int, default=60)
    parser.add_argument("--minimum-observations", type=int, default=20)
    parser.add_argument("--max-data-age-days", type=int, default=5)
    parser.add_argument("--history-days", type=int, default=7)
    args = parser.parse_args()

    if not args.holdings_csv.is_file():
        parser.error(f"Holdings file not found: {args.holdings_csv}")
    holdings = pd.read_csv(args.holdings_csv)
    close_prices = load_price_panel(args.prices)
    adjusted_prices = load_price_panel(args.adjusted_prices)
    groups = _load_json_mapping(args.groups_json, "groups")
    group_caps = _load_json_mapping(args.group_caps_json, "group caps")
    result = run_monitoring(
        holdings,
        close_prices,
        adjusted_prices,
        output_dir=args.output_dir,
        cash=args.cash,
        groups={str(k): str(v) for k, v in groups.items()},
        group_max_weights={str(k): float(v) for k, v in group_caps.items()},
        single_name_limit=args.single_name_limit,
        volatility_limit_annual=args.volatility_limit,
        drawdown_limit=args.drawdown_limit,
        volatility_window=args.volatility_window,
        minimum_observations=args.minimum_observations,
        max_data_age_calendar_days=args.max_data_age_days,
        mode=args.mode,
        history_days=args.history_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
