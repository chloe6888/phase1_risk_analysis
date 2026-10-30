#!/usr/bin/env python3
"""Small file-backed task manager for long-running research jobs.

The dashboard starts a detached worker and returns immediately. The worker writes
status/log files and publishes a new result directory only after a complete run.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
RUNS_ROOT = OUTPUT_ROOT / "runs"
CURRENT_TASK_FILE = OUTPUT_ROOT / "current_task.json"
LATEST_SUCCESS_FILE = OUTPUT_ROOT / "latest_success.json"
LAUNCH_LOCK_FILE = OUTPUT_ROOT / ".task-launch.lock"
ACTIVE_STATES = {"queued", "running"}


class TaskAlreadyRunning(RuntimeError):
    """Raised when a research task is already active."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, object] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def task_directory(task_id: str, repository_root: Path = ROOT) -> Path:
    return repository_root / "outputs" / "runs" / task_id


def task_status_path(task_id: str, repository_root: Path = ROOT) -> Path:
    return task_directory(task_id, repository_root) / "status.json"


def get_current_task(repository_root: Path = ROOT) -> dict[str, object] | None:
    root = Path(repository_root)
    pointer = read_json(root / "outputs" / "current_task.json")
    if not pointer or not pointer.get("task_id"):
        return None
    status_path = task_status_path(str(pointer["task_id"]), root)
    status = read_json(status_path)
    if status is None:
        return None
    state = str(status.get("state", "unknown"))
    pid = status.get("pid")
    if not isinstance(pid, int):
        pid_file = status_path.parent / "worker.pid"
        try:
            pid = int(pid_file.read_text(encoding="ascii").strip())
            status["pid"] = pid
        except (OSError, ValueError):
            pid = None
    if state in ACTIVE_STATES and isinstance(pid, int) and not pid_is_alive(pid):
        status.update({
            "state": "failed",
            "finished_at": utc_now(),
            "message": "研究工作进程已退出，但未留下完成状态；旧的成功报告仍然保留。",
        })
        atomic_write_json(status_path, status)
        atomic_write_json(root / "outputs" / "current_task.json", {"task_id": pointer["task_id"]})
    return status


def get_task_log(task_id: str, repository_root: Path = ROOT, max_chars: int = 12000) -> str:
    path = task_directory(task_id, repository_root) / "worker.log"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars * 4), os.SEEK_SET)
            content = handle.read()
        return content.decode("utf-8", errors="replace")[-max_chars:]
    except OSError:
        return "暂无运行日志。"


def active_output_dir(repository_root: Path = ROOT) -> Path:
    """Return the last fully successful run; fall back to legacy outputs."""
    root = Path(repository_root)
    outputs = root / "outputs"
    pointer = read_json(outputs / "latest_success.json")
    relative = pointer.get("artifacts_dir") if pointer else None
    if isinstance(relative, str) and relative:
        candidate = (root / relative).resolve()
        runs_root = (outputs / "runs").resolve()
        try:
            candidate.relative_to(runs_root)
        except ValueError:
            candidate = None
        if candidate is not None and candidate.is_dir():
            return candidate
    return outputs


def _acquire_launch_lock(output_root: Path) -> tuple[int, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_file = output_root / ".task-launch.lock"
    try:
        return os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), lock_file
    except FileExistsError as exc:
        try:
            age = time.time() - lock_file.stat().st_mtime
        except OSError:
            age = 0
        if age > 120:
            try:
                lock_file.unlink()
            except OSError:
                pass
            try:
                return os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), lock_file
            except FileExistsError:
                pass
        raise TaskAlreadyRunning("另一个任务正在提交，请稍后再试。") from exc


def start_task(command: Sequence[str], repository_root: Path = ROOT) -> dict[str, object]:
    """Launch a detached worker. Returns immediately after the worker is spawned."""
    import subprocess
    import sys

    root = Path(repository_root).resolve()
    output_root = root / "outputs"
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_fd, lock_file = _acquire_launch_lock(output_root)
    try:
        current = get_current_task(root)
        if current and current.get("state") in ACTIVE_STATES:
            pid = current.get("pid")
            if isinstance(pid, int) and pid_is_alive(pid):
                raise TaskAlreadyRunning(
                    f"任务 {current.get('task_id')} 正在运行（{current.get('current_step') or '准备中'}），请勿重复提交。"
                )
            if pid is None:
                created = current.get("created_at")
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(created))).total_seconds()
                except (TypeError, ValueError):
                    age = 999
                if age < 60:
                    raise TaskAlreadyRunning("研究任务正在启动，请稍后再试。")

        task_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        task_dir = runs_root / task_id
        artifacts_dir = task_dir / "artifacts"
        task_dir.mkdir(parents=True, exist_ok=False)
        artifacts_dir.mkdir()
        status_path = task_dir / "status.json"
        log_path = task_dir / "worker.log"
        status: dict[str, object] = {
            "task_id": task_id,
            "state": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "current_step": "等待后台进程启动",
            "message": "任务已提交，上一份成功报告保持可用。",
            "exit_code": None,
        }
        atomic_write_json(status_path, status)
        atomic_write_json(output_root / "current_task.json", {"task_id": task_id})

        worker_command = [
            sys.executable, str(root / "task_worker.py"),
            "--task-id", task_id,
            "--run-dir", str(task_dir),
            "--",
            *[str(item) for item in command],
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                worker_command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        (task_dir / "worker.pid").write_text(str(process.pid), encoding="ascii")
        latest_status = read_json(status_path) or status
        latest_status["pid"] = int(process.pid)
        return latest_status
    except Exception:
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


def publish_success(task_id: str, task_dir: Path, repository_root: Path = ROOT) -> None:
    root = Path(repository_root)
    artifacts = task_dir / "artifacts"
    atomic_write_json(root / "outputs" / "latest_success.json", {
        "task_id": task_id,
        "completed_at": utc_now(),
        "artifacts_dir": str(artifacts.relative_to(root)),
    })
