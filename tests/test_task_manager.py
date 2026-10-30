"""Tests for the file-backed asynchronous research task manager."""
from __future__ import annotations

import json
from pathlib import Path

import task_manager as tasks


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_active_output_dir_uses_legacy_outputs_before_first_success(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    assert tasks.active_output_dir(tmp_path) == outputs


def test_active_output_dir_selects_last_successful_run(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    artifacts = outputs / "runs" / "run-123" / "artifacts"
    artifacts.mkdir(parents=True)
    _write_json(outputs / "latest_success.json", {
        "task_id": "run-123",
        "artifacts_dir": "outputs/runs/run-123/artifacts",
    })
    assert tasks.active_output_dir(tmp_path) == artifacts


def test_active_output_dir_rejects_pointer_outside_runs_directory(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _write_json(outputs / "latest_success.json", {
        "task_id": "bad",
        "artifacts_dir": "../private",
    })
    assert tasks.active_output_dir(tmp_path) == outputs


def test_get_current_task_marks_disappeared_worker_failed(
    tmp_path: Path, monkeypatch
) -> None:
    task_id = "run-dead"
    task_dir = tasks.task_directory(task_id, tmp_path)
    _write_json(task_dir / "status.json", {
        "task_id": task_id,
        "state": "running",
        "pid": 987654321,
        "current_step": "2/5 risk",
    })
    _write_json(tmp_path / "outputs" / "current_task.json", {"task_id": task_id})
    monkeypatch.setattr(tasks, "pid_is_alive", lambda _pid: False)

    result = tasks.get_current_task(tmp_path)

    assert result is not None
    assert result["state"] == "failed"
    assert "旧的成功报告仍然保留" in str(result["message"])
    persisted = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"


def test_get_task_log_returns_recent_text(tmp_path: Path) -> None:
    task_id = "run-log"
    task_dir = tasks.task_directory(task_id, tmp_path)
    task_dir.mkdir(parents=True)
    (task_dir / "worker.log").write_text("old\nnew output\n", encoding="utf-8")
    assert tasks.get_task_log(task_id, tmp_path).endswith("old\nnew output\n")


def test_detached_worker_publishes_only_complete_run(tmp_path: Path) -> None:
    import shutil
    import sys
    import time

    project = Path(tasks.__file__).resolve().parent
    shutil.copy2(project / "task_manager.py", tmp_path / "task_manager.py")
    shutil.copy2(project / "task_worker.py", tmp_path / "task_worker.py")

    writer = (
        "import os; from pathlib import Path; "
        "p=Path(os.environ['PHASE1_OUTPUT_DIR']); p.mkdir(parents=True, exist_ok=True); "
        "(p/'research_report.md').write_text('# complete', encoding='utf-8'); "
        "(p/'research_run_manifest.json').write_text('{}', encoding='utf-8')"
    )
    submitted = tasks.start_task([sys.executable, "-c", writer], repository_root=tmp_path)
    task_id = str(submitted["task_id"])

    deadline = time.monotonic() + 10
    current = tasks.get_current_task(tmp_path)
    while current is not None and current.get("state") in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.05)
        current = tasks.get_current_task(tmp_path)

    assert current is not None
    assert current["state"] == "succeeded"
    latest = tasks.active_output_dir(tmp_path)
    assert latest == tmp_path / "outputs" / "runs" / task_id / "artifacts"
    assert (latest / "research_report.md").read_text(encoding="utf-8") == "# complete"
