import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import run_state as rs

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def cli(state, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / "run_state.py"), "--state", str(state), *args],
                          capture_output=True, text=True)


def test_plan_skip_and_eta(tmp_path):
    state = tmp_path / "run-state.json"
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"permissions": {"builds": False, "download_gb": 0}}))
    assert cli(state, "init", "--answers", str(answers)).returncode == 0
    cli(state, "add", "--phase", "baseline", "--est", "5", "--desc", "measure", "--", "true")
    out = cli(state, "add", "--phase", "build", "--est", "20", "--needs", "builds", "--desc", "build HIP", "--", "make")
    assert "skipped" in out.stdout
    s = json.loads(state.read_text())
    sm = rs.summary(s, clock=dt.datetime(2026, 1, 1, 21, 0))
    assert sm["pending"] == 1 and sm["skipped"] == 1 and sm["minutes_left"] == 5 and sm["finish_at"] == "21:05"
    assert "needs permission 'builds'" in sm["skipped_needing_permission"][0]


def test_init_refuses_to_wipe_a_started_run(tmp_path):
    state = tmp_path / "run-state.json"
    answers = tmp_path / "a.json"
    answers.write_text("{}")
    cli(state, "init", "--answers", str(answers))
    cli(state, "add", "--phase", "p", "--est", "1", "--desc", "d", "--", "true")
    s = json.loads(state.read_text())
    s["steps"][0]["status"] = "done"
    state.write_text(json.dumps(s))
    assert cli(state, "init", "--answers", str(answers)).returncode != 0


def test_stale_running_step_is_reset():
    s = {"steps": [{"id": "s1", "status": "running", "pid": 2**22 + 12345, "phase": "p", "desc": "d"}]}
    assert rs.reset_stale(s) == 1 and s["steps"][0]["status"] == "pending"


def test_run_phase_resumes_and_records_failures(tmp_path):
    state = tmp_path / "run-state.json"
    answers = tmp_path / "a.json"
    answers.write_text("{}")
    cli(state, "init", "--answers", str(answers))
    cli(state, "add", "--phase", "p", "--est", "1", "--desc", "ok", "--", "true")
    cli(state, "add", "--phase", "p", "--est", "1", "--desc", "bad", "--", "false")
    cli(state, "add", "--phase", "q", "--est", "1", "--desc", "later", "--", "true")
    run = [sys.executable, str(SCRIPTS / "run_phase.py"), "p", "--state", str(state)]
    assert subprocess.run(run, capture_output=True).returncode == 0
    steps = json.loads(state.read_text())["steps"]
    assert [x["status"] for x in steps] == ["done", "failed", "pending"]
    out = subprocess.run(run, capture_output=True, text=True).stdout
    assert "0 step(s)" in out  # nothing rerun
