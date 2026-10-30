#!/usr/bin/env python3
"""Detached worker for a single quantitative research run."""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from task_manager import atomic_write_json, publish_success

ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_status(path: Path, status: dict[str, object], **changes: object) -> None:
    status.update(changes)
    atomic_write_json(path, status)


def main() -> int:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        print("Worker command separator '--' is missing.", file=sys.stderr)
        return 2

    args = sys.argv[1:separator]
    command = sys.argv[separator + 1:]
    try:
        task_index = args.index("--task-id")
        run_index = args.index("--run-dir")
        task_id = args[task_index + 1]
        task_dir = Path(args[run_index + 1]).resolve()
    except (ValueError, IndexError):
        print("Worker requires --task-id and --run-dir.", file=sys.stderr)
        return 2
    if not command:
        print("Worker received an empty research command.", file=sys.stderr)
        return 2

    status_path = task_dir / "status.json"
    log_path = task_dir / "worker.log"
    artifacts_dir = task_dir / "artifacts"
    status: dict[str, object] = {}
    try:
        import json
        with status_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            status = loaded
    except (OSError, ValueError):
        status = {"task_id": task_id}

    env = os.environ.copy()
    env["PHASE1_OUTPUT_DIR"] = str(artifacts_dir)
    env["PYTHONUNBUFFERED"] = "1"
    update_status(
        status_path, status,
        state="running",
        started_at=utc_now(),
        pid=os.getpid(),
        current_step="1/5 初始化研究流程",
        message="后台研究流程正在执行；页面可以继续浏览上一次成功结果。",
    )

    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{utc_now()}] START task={task_id}\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if line.startswith("=== "):
                    update_status(
                        status_path, status,
                        current_step=line.strip("= \r\n"),
                        message="研究阶段执行中。",
                    )
            return_code = process.wait()
            log.write(f"\n[{utc_now()}] EXIT code={return_code}\n")

        report = artifacts_dir / "research_report.md"
        manifest = artifacts_dir / "research_run_manifest.json"
        if return_code == 0 and report.is_file() and manifest.is_file():
            publish_success(task_id, task_dir, ROOT)
            update_status(
                status_path, status,
                state="succeeded",
                finished_at=utc_now(),
                exit_code=0,
                current_step="研究完成",
                message="完整研究报告已发布；仪表盘可加载最新成功结果。",
                report_path=str(report.relative_to(ROOT)),
            )
            return 0

        message = (
            f"研究命令退出码为 {return_code}。"
            if return_code != 0
            else "流程退出码为 0，但缺少完整报告或运行清单，结果未发布。"
        )
        update_status(
            status_path, status,
            state="failed",
            finished_at=utc_now(),
            exit_code=int(return_code),
            current_step="研究失败",
            message=message + " 上一次成功报告仍然保留。",
        )
        return int(return_code or 1)
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nWORKER EXCEPTION\n")
            log.write(traceback.format_exc())
        update_status(
            status_path, status,
            state="failed",
            finished_at=utc_now(),
            exit_code=1,
            current_step="后台工作进程异常",
            message=f"{type(exc).__name__}: {exc}。上一次成功报告仍然保留。",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
