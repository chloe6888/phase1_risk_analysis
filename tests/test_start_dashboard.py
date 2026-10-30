"""Regression tests for the dashboard shell launcher."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_fake_python(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-c', 'import streamlit']:\n"
        "    raise SystemExit(0)\n"
        "print(json.dumps(args))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(("port", "expected_port"), [(None, "8501"), ("8765", "8765")])
def test_launcher_expands_python_and_port_environment_variables(
    tmp_path: Path, port: str | None, expected_port: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    launcher = project / "start_dashboard.sh"
    shutil.copy2(PROJECT_ROOT / "start_dashboard.sh", launcher)
    fake_python = _make_fake_python(tmp_path / "fake-python")

    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    if port is None:
        env.pop("PORT", None)
    else:
        env["PORT"] = port

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    invocation = json.loads(completed.stdout.splitlines()[-1])
    assert invocation == [
        "-m", "streamlit", "run", "dashboard.py",
        "--server.address", "0.0.0.0",
        "--server.port", expected_port,
        "--server.headless", "true",
    ]
    assert (project / 'data' / 'private').is_dir()
