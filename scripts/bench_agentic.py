"""Run real coding-agent tasks end to end against one server config.

Usage: bench_agentic.py <model> <backend> <label> <reps> [server args...]
  env TASKS=t1_fix_test,t3_rename   run only some tasks
  env RECORD=1                      also save rendered prompts to results/prompts/<model>/ (for bench_replay.py)

For each task in agentic/tasks.json and each repetition: copy agentic/template
into a fresh, otherwise empty sandbox with its own git history, run the agent
headless against a test server, check the result with the task's check command,
and record wall time, pass/fail, tool calls and the server's prompt/generation
counters. Appends to results/agentic.csv; run folders go to results/runs/.

Adding another agent: write a function like run_opencode() below that runs the
agent non-interactively in `workdir` and returns its event log path, then add
it to ADAPTERS.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from server import CFG, PORT, RESULTS, ROOT, Server, vram_mib

TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 1800))
AGENTIC = ROOT / "agentic"


def opencode_config(alias, ctx, tmp):
    """A throwaway OpenCode config with one provider pointing at the test server."""
    path = tmp / "opencode-bench.json"
    key_env = CFG.get("api_key_env")
    path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "providers": {"bench": {
            "name": "llamacpp-speedtune test server",
            "package": "@opencode/ai/providers/openai-compatible",
            "settings": {"baseURL": f"http://127.0.0.1:{PORT}/v1",
                         "apiKey": f"{{env:{key_env}}}" if key_env else "none"},
            "models": {"m": {"modelID": alias, "name": alias,
                             "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                             "limit": {"context": ctx, "output": 8192},
                             "compatibility": {"maxTokensField": "max_tokens",
                                               "reasoningField": "reasoning_content"}}}}}}, indent=1))
    return path


def run_opencode(workdir, prompt, alias, ctx, env, tmp):
    """OpenCode adapter. Each fix here cost a broken benchmark run to discover:
    - stdin=DEVNULL: `opencode run` reads a non-TTY stdin as extra input and waits forever.
    - PWD: OpenCode takes its working directory from $PWD, not the process cwd.
    - XDG_*: keeps benchmark sessions out of the user's real OpenCode history.
    (The unique git root commit per run is made by the caller: OpenCode keys projects by it.)
    """
    xdg = tmp / "xdg"
    env = {**env, "OPENCODE_CONFIG": str(opencode_config(alias, ctx, tmp)), "OPENCODE_DISABLE_AUTOUPDATE": "1",
           "PWD": str(workdir), "XDG_DATA_HOME": str(xdg / "data"), "XDG_STATE_HOME": str(xdg / "state"),
           "XDG_CACHE_HOME": str(xdg / "cache")}
    log = workdir / "agent.jsonl"
    with open(log, "w") as out, open(workdir / "agent.err", "w") as err:
        try:
            subprocess.run([os.path.expanduser(CFG.get("agent", {}).get("bin", "opencode")), "run", "--standalone", "-m", "bench/m",
                            "--auto", "--format", "json", prompt],
                           cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
    return log


def summarize_opencode(log):
    events = [json.loads(l) for l in open(log) if l.strip()]
    return {"tool_calls": sum(e["type"] == "tool_use" for e in events),
            "steps": sum(e["type"] == "step_finish" for e in events),
            "compactions": sum("compact" in json.dumps(e).lower() and e["type"] != "tool_use" for e in events),
            "error": "; ".join(e.get("error", {}).get("message", "")[:80] for e in events if e["type"] == "error")}


ADAPTERS = {"opencode": (run_opencode, summarize_opencode)}


def run_task(task, alias, ctx, keep_dir):
    run, summarize = ADAPTERS[CFG.get("agent", {}).get("type", "opencode")]
    # Agents search outside their folder when asked to change things "everywhere", so the
    # project lives alone in a fresh temp dir, away from the template and earlier runs.
    tmp = Path(tempfile.mkdtemp(prefix="llamacpp-speedtune-"))
    workdir = tmp / "shop-project"
    shutil.copytree(AGENTIC / "template", workdir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    subprocess.run(["chmod", "-R", "u+w", workdir], check=True)
    git = ["git", "-c", "user.email=bench@example.com", "-c", "user.name=bench"]
    subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    # A unique root commit per run: OpenCode treats repos with the same root commit as one project.
    subprocess.run([*git, "commit", "-qm", f"start {keep_dir.name} {time.time()}"], cwd=workdir, check=True)
    py_bin = CFG.get("agent", {}).get("python_bin_dir")
    env = {**os.environ, "CHECKS": str(AGENTIC / "checks")}
    if py_bin:
        env["PATH"] = f"{Path(py_bin).expanduser()}:{env['PATH']}"
    try:
        t0 = time.time()
        log = run(workdir, task["prompt"], alias, ctx, env, tmp)
        wall = time.time() - t0
        check = subprocess.run(["bash", "-c", task["check"]], cwd=workdir, env=env, capture_output=True)
        res = {"wall_s": round(wall, 1), "passed": check.returncode == 0, **summarize(log)}
        shutil.rmtree(keep_dir, ignore_errors=True)
        keep_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(workdir, keep_dir)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    mdl, be, label, reps, *args = sys.argv[1:]
    tasks = json.load(open(AGENTIC / "tasks.json"))
    if os.environ.get("TASKS"):
        tasks = [t for t in tasks if t["id"] in os.environ["TASKS"].split(",")]
    m = CFG["models"][mdl]
    ctx = m.get("ctx", 32768)
    prompts = RESULTS / "prompts" / mdl if os.environ.get("RECORD") else None
    rows = []
    with Server(mdl, args, be, log=RESULTS / f"logs/agentic-{mdl}-{label}.log", prompts_dir=prompts) as srv:
        vram = vram_mib()
        for rep in range(int(reps)):
            for task in tasks:
                before = srv.metrics()
                res = run_task(task, m["alias"], ctx, RESULTS / f"runs/{mdl}/{label}/{task['id']}-{rep}")
                d = {k: v - before.get(k, 0) for k, v in srv.metrics().items()}
                pp_n, pp_s = d.get("llamacpp:prompt_tokens_total", 0), d.get("llamacpp:prompt_seconds_total", 0)
                tg_n, tg_s = d.get("llamacpp:tokens_predicted_total", 0), d.get("llamacpp:tokens_predicted_seconds_total", 0)
                row = {"model": mdl, "backend": be, "label": label, "args": " ".join(args), "task": task["id"],
                       "rep": rep, **res, "prompt_tokens": int(pp_n), "pp_ts": round(pp_n / pp_s, 1) if pp_s else 0,
                       "gen_tokens": int(tg_n), "tg_ts": round(tg_n / tg_s, 2) if tg_s else 0, "vram_mib": vram}
                rows.append(row)
                print(f"{mdl:8} {label:24} {task['id']:16} rep{rep} {'PASS' if row['passed'] else 'fail'} "
                      f"{row['wall_s']:6.1f}s tools={row['tool_calls']:2} pp={row['pp_ts']:7.1f} "
                      f"tg={row['tg_ts']:6.2f} ptok={row['prompt_tokens']} compact={row['compactions']} {row['error']}",
                      flush=True)
    out = RESULTS / "agentic.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"== {mdl} {label}: {sum(r['passed'] for r in rows)}/{n} passed, "
          f"mean wall {sum(r['wall_s'] for r in rows) / n:.1f}s, total {sum(r['wall_s'] for r in rows):.0f}s")


if __name__ == "__main__":
    main()
