#!/usr/bin/env python3
"""Run the pending steps of one phase (or all phases) from run-state.json, unattended.

Usage: run_phase.py <phase | --all> [--state PATH] [--notify]

Start it as ONE background command and wait for it to exit (don't poll): each step runs in
turn with its output in <results>/logs/step-<id>.log, and its status, exit code and duration go
into run-state.json as it finishes. A failed step (out of memory, a config that doesn't load)
is recorded and the phase moves on: a failure is a result. Steps already done are never rerun,
so starting the same phase again after an interruption resumes it.

If the user allowed it (permissions.prevent_sleep), the computer is kept awake for the phase
with systemd-inhibit (Linux) or caffeinate (macOS). --notify sends a desktop notification when
the phase ends.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_state import default_state_path, granted, locked, notify, now, reset_stale, summary  # noqa: E402

INHIBITED = "LLAMACPP_SPEEDTUNE_INHIBITED"


def keep_awake(argv):
    """Re-run this script under a sleep inhibitor. Returns the exit code, or None if none is available."""
    if os.environ.get(INHIBITED):
        return None
    env = {**os.environ, INHIBITED: "1"}
    if shutil.which("systemd-inhibit"):
        cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=llamacpp-speedtune",
               "--why=benchmarking local models", sys.executable, *argv]
    elif shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", "-s", sys.executable, *argv]
    else:
        return None
    try:
        return subprocess.run(cmd, env=env).returncode
    except OSError:
        return None


def pick(state, phase):
    return [s for s in state["steps"] if s["status"] == "pending" and (phase is None or s["phase"] == phase)]


def run_step(path, step_id, logs):
    with locked(path) as box:
        s = next(x for x in box["state"]["steps"] if x["id"] == step_id)
        s.update(status="running", started=now(), pid=os.getpid(), log=str(logs / f"step-{step_id}.log"))
        cmd, log = s["cmd"], s["log"]
    t0 = time.time()
    with open(log, "w") as out:
        out.write(f"$ {cmd}\n")
        out.flush()
        code = subprocess.run(["bash", "-c", cmd], stdout=out, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    with locked(path) as box:
        s = next(x for x in box["state"]["steps"] if x["id"] == step_id)
        s.update(status="done" if code == 0 else "failed", ended=now(), exit=code, pid=None,
                 minutes=round((time.time() - t0) / 60, 1))
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--state")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args()
    if not a.phase and not a.all:
        sys.exit("name a phase, or --all")
    path = Path(a.state).expanduser() if a.state else default_state_path()
    if not path.exists():
        sys.exit(f"no run at {path}: run_state.py init first")

    with locked(path) as box:
        reset_stale(box["state"])
        awake = granted(box["state"], "prevent_sleep")
    if awake:
        code = keep_awake(sys.argv)
        if code is not None:
            sys.exit(code)

    logs = path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    phase = None if a.all else a.phase
    ran = failed = 0
    while True:
        with locked(path) as box:
            todo = pick(box["state"], phase)
        if not todo:
            break
        s = todo[0]
        print(f"[{now()}] {s['id']} [{s['phase']}] {s.get('model') or ''} {s['desc']} ...", flush=True)
        code = run_step(path, s["id"], logs)
        ran += 1
        failed += code != 0
        print(f"[{now()}] {s['id']} {'done' if code == 0 else f'FAILED (exit {code})'}", flush=True)

    with locked(path) as box:
        sm = summary(box["state"])
    msg = (f"phase {a.phase or 'all'} finished: {ran} step(s), {failed} failed. "
           f"Overall {sm['done']}/{sm['total']} done" + (f", ~{sm['minutes_left']:.0f} min left" if sm["minutes_left"] else ""))
    print(msg)
    if a.notify:
        notify(msg)


if __name__ == "__main__":
    main()
