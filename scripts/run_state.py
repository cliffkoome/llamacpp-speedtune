#!/usr/bin/env python3
"""The run plan and its progress, in <results_dir>/run-state.json, so a long unattended run can resume.

Usage:
  run_state.py init --answers answers.json [--force]   start a run from the user's answers (SKILL.md, step 2)
  run_state.py add --phase P --est MIN [--model M] [--needs PERM] --desc "..." -- <command ...>
  run_state.py skip <step-id> --note "why"
  run_state.py status [--json]                          done/total, what's next, minutes left, finish time
  run_state.py notify "message"                         desktop notification (falls back to printing)
  common option: --state PATH (default: <results_dir>/run-state.json from $TUNE_CONFIG or ./config.json)

answers.json holds the user's answers, including "permissions": {"builds", "install_engines",
"download_gb", "stop_live_server", "prevent_sleep"}. A step added with --needs <permission> that
the user didn't grant is recorded as skipped (it never blocks the run) and listed in the report.
Steps run with run_phase.py. A step left "running" by a run that died is reset to pending.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PERMISSIONS = ("builds", "install_engines", "download_gb", "stop_live_server", "prevent_sleep")


def default_state_path():
    cfg = Path(os.environ.get("TUNE_CONFIG", "config.json")).expanduser()
    results = "results"
    if cfg.exists():
        results = json.loads(cfg.read_text()).get("results_dir", "results")
    return Path(results).expanduser() / "run-state.json"


@contextlib.contextmanager
def locked(path):
    """Read-modify-write the state under a lock: run_phase.py and the agent may both write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else None
        box = {"state": state}
        yield box
        if box["state"] is not None:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(box["state"], indent=1))
            tmp.replace(path)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def granted(state, perm):
    v = (state.get("permissions") or {}).get(perm)
    return bool(v) and v != 0


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def reset_stale(state):
    """Steps marked running whose process is gone (agent or machine restarted) go back to pending."""
    n = 0
    for s in state["steps"]:
        if s["status"] == "running" and not _alive(s.get("pid")):
            s.update(status="pending", pid=None, note=(s.get("note") or "") + " [interrupted, will rerun]")
            n += 1
    return n


def summary(state, clock=None):
    clock = clock or dt.datetime.now()
    steps = state["steps"]
    by = {k: [s for s in steps if s["status"] == k] for k in ("pending", "running", "done", "failed", "skipped")}
    left = sum(s.get("est_min", 0) for s in by["pending"] + by["running"])
    nxt = (by["running"] or by["pending"] or [None])[0]
    return {"total": len(steps), **{k: len(v) for k, v in by.items()}, "minutes_left": left,
            "finish_at": (clock + dt.timedelta(minutes=left)).strftime("%H:%M") if left else None,
            "next": nxt and " ".join(x for x in (nxt["id"], f"[{nxt['phase']}]", nxt.get("model"), nxt["desc"]) if x),
            "skipped_needing_permission": [f"{s['id']}: {s['desc']} ({s.get('note')})" for s in by["skipped"]]}


def notify(msg):
    title = "llamacpp-speedtune"
    try:
        if platform.system() == "Darwin" and shutil.which("osascript"):
            subprocess.run(["osascript", "-e", f'display notification {json.dumps(msg)} with title "{title}"'],
                           timeout=10)
            return
        if shutil.which("notify-send") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            subprocess.run(["notify-send", title, msg], timeout=10)
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(f"\a[{title}] {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--answers", required=True)
    i.add_argument("--force", action="store_true")
    a_ = sub.add_parser("add")
    a_.add_argument("--phase", required=True)
    a_.add_argument("--est", type=float, required=True, help="estimated minutes")
    a_.add_argument("--model")
    a_.add_argument("--needs", choices=PERMISSIONS)
    a_.add_argument("--desc", required=True)
    a_.add_argument("command", nargs=argparse.REMAINDER)
    k = sub.add_parser("skip")
    k.add_argument("id")
    k.add_argument("--note", required=True)
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    n = sub.add_parser("notify")
    n.add_argument("message")
    a = ap.parse_args()
    path = Path(a.state).expanduser() if a.state else default_state_path()

    if a.cmd == "notify":
        notify(a.message)
        return
    with locked(path) as box:
        state = box["state"]
        if a.cmd == "init":
            if state and not a.force and any(s["status"] != "pending" for s in state["steps"]):
                sys.exit(f"{path} already has a run in progress: use status to resume it, or --force to start over")
            answers = json.loads(Path(a.answers).expanduser().read_text())
            box["state"] = {"created": now(), "answers": answers, "permissions": answers.get("permissions", {}),
                            "steps": []}
            print(f"started a run in {path}")
            return
        if state is None:
            sys.exit(f"no run at {path}: run_state.py init first")
        if a.cmd == "add":
            cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
            if not cmd:
                sys.exit("add needs a command after --")
            step = {"id": f"s{len(state['steps']) + 1}", "phase": a.phase, "model": a.model, "desc": a.desc,
                    "cmd": shlex.join(cmd) if len(cmd) > 1 else cmd[0], "est_min": a.est, "status": "pending",
                    "needs": a.needs}
            if a.needs and not granted(state, a.needs):
                step.update(status="skipped", note=f"needs permission '{a.needs}', not granted")
            state["steps"].append(step)
            print(f"{step['id']} {step['status']}: [{a.phase}] {a.desc}")
        elif a.cmd == "skip":
            s = next((s for s in state["steps"] if s["id"] == a.id), None)
            if not s:
                sys.exit(f"no step {a.id}")
            s.update(status="skipped", note=a.note)
        elif a.cmd == "status":
            reset_stale(state)
            sm = summary(state)
            if a.json:
                print(json.dumps(sm, indent=1))
                return
            print(f"{sm['done']}/{sm['total']} steps done, {sm['failed']} failed, {sm['skipped']} skipped; "
                  f"about {sm['minutes_left']:.0f} min left" + (f", finishing around {sm['finish_at']}"
                                                                 if sm["finish_at"] else ""))
            if sm["next"]:
                print(f"next: {sm['next']}")
            for x in sm["skipped_needing_permission"]:
                print(f"skipped: {x}")


if __name__ == "__main__":
    main()
