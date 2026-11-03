#!/usr/bin/env python3
"""Each model's best config, stored once and turned into the config that serves it.

Usage:
  profiles.py save <model> <backend-or-engine> [--label after] [--note "..."]
  profiles.py show
  profiles.py serve-config --out DIR [--models a,b]

save    reads the chosen config's flags and measurements (label "after", and "before" for
        comparison) from the results folder and writes <profiles_dir>/<model>.json. The previous
        profile moves to <profiles_dir>/history/, so later runs can compare against it.
serve-config writes what serves all profiled models behind one address:
        - every model on llama.cpp with one binary: presets.ini for llama-server's router mode
        - otherwise (several engines or binaries): llama-swap.yaml, which starts the right engine
          for each model on request (https://github.com/mostlygeek/llama-swap)
        Both files are only written to DIR; installing them is a separate, approved step.
profiles_dir comes from the config (default ~/.config/local-llm-speedtune/profiles).
"""
import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw  # noqa: E402
from quick_check import parse_cmdline, to_cli  # noqa: E402

DEFAULT_DIR = "~/.config/local-llm-speedtune/profiles"


def load_config():
    cfg = Path(os.environ.get("TUNE_CONFIG", "config.json")).expanduser()
    if not cfg.exists():
        sys.exit(f"no config at {cfg}: set TUNE_CONFIG")
    return json.loads(cfg.read_text())


def profiles_dir(cfg):
    return Path(cfg.get("profiles_dir") or DEFAULT_DIR).expanduser()


def _rows(path, model, label):
    if not path.exists():
        return []
    return [r for r in csv.DictReader(open(path)) if r.get("model") == model and r.get("label") == label]


def _speeds(rows):
    return [{"prompt_tokens": int(float(r["prompt_n"])), "pp_ts": float(r["pp_ts"]), "tg_ts": float(r["tg_ts"]),
             "vram_mib": int(float(r.get("vram_mib") or -1))} for r in rows]


def _agent(rows):
    if not rows:
        return None
    passed = sum(r["passed"] in ("True", "true", "1") for r in rows)
    return {"passed": f"{passed}/{len(rows)}", "mean_wall_s": round(sum(float(r["wall_s"]) for r in rows) / len(rows), 1)}


def build_profile(cfg, model, name, label, results, note=None, version=None, hardware=None):
    m = cfg["models"][model]
    engines = cfg.get("engines") or {}
    measured = _rows(results / "validate.csv", model, label) or _rows(results / "openai.csv", model, label)
    if not measured:
        raise SystemExit(f"no measurements for {model!r} with label {label!r} in {results}: measure it first")
    args = measured[-1]["args"]
    before = _rows(results / "validate.csv", model, "before") or _rows(results / "openai.csv", model, "before")
    if name in engines:
        e = engines[name]
        serve = {"engine": name, "start": e["start"], "env": e.get("env") or {},
                 "model_file": (m.get("engines") or {}).get(name, m["path"])}
    else:
        b = cfg["backends"][name]
        serve = {"engine": "llama.cpp", "backend": name, "server": b["server"], "env": b.get("env") or {},
                 "model_file": m["path"], "mmproj": m.get("mmproj")}
    return {
        "model": model, "alias": m["alias"], "ctx": m.get("ctx"), **serve,
        "args": to_cli(parse_cmdline(args)),
        "measured": {"label": label, "at_depth": _speeds(measured),
                     "agent_tasks": _agent(_rows(results / "agentic.csv", model, label))},
        "before": {"at_depth": _speeds(before), "agent_tasks": _agent(_rows(results / "agentic.csv", model, "before"))},
        "engine_version": version, "hardware": hardware, "saved": dt.datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }


def engine_version(server):
    try:
        out = subprocess.run([os.path.expanduser(server), "--version"], capture_output=True, text=True, timeout=30)
        return next((l.strip() for l in (out.stdout + out.stderr).splitlines() if "version" in l), None)
    except (OSError, subprocess.TimeoutExpired):
        return None


def save(cfg, profile):
    d = profiles_dir(cfg)
    (d / "history").mkdir(parents=True, exist_ok=True)
    path = d / f"{profile['model']}.json"
    if path.exists():
        old = json.loads(path.read_text())
        stamp = (old.get("saved") or "old").replace(":", "")
        path.replace(d / "history" / f"{profile['model']}-{stamp}.json")
    path.write_text(json.dumps(profile, indent=1) + "\n")
    return path


def load_all(cfg, only=None):
    d = profiles_dir(cfg)
    out = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
    return [p for p in out if not only or p["model"] in only]


def _pairs(args):
    """'--flag value --bool' -> [(flag, value), (bool, 'true')]."""
    toks, out, i = shlex.split(args), [], 0
    while i < len(toks):
        k = toks[i].lstrip("-")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out.append((k, toks[i + 1]))
            i += 2
        else:
            out.append((k, "true"))
            i += 1
    return out


def presets_ini(profiles):
    lines = ["version = 1", "; Generated by local-llm-speedtune profiles.py from the saved profiles.", ""]
    for p in profiles:
        lines += [f"[{p['alias'].replace(':', '-')}]", f"model = {Path(p['model_file']).expanduser()}"]
        if p.get("mmproj") and "mmproj" not in p["args"]:
            lines.append(f"mmproj = {Path(p['mmproj']).expanduser()}")
        if p.get("ctx") and "ctx-size" not in p["args"]:
            lines.append(f"ctx-size = {p['ctx']}")
        lines += [f"{k} = {v}" for k, v in _pairs(p["args"])]
        lines.append("")
    return "\n".join(lines)


def llama_swap_yaml(profiles):
    """llama-swap config: one entry per model; ${PORT} is filled in by llama-swap."""
    lines = ["# Generated by local-llm-speedtune profiles.py. llama-swap starts the right engine per model.",
             "# Check the format against your llama-swap version's README before installing.", "models:"]
    for p in profiles:
        if p["engine"] == "llama.cpp":
            cmd = (f"{Path(p['server']).expanduser()} --host 127.0.0.1 --port ${{PORT}} -m {Path(p['model_file']).expanduser()}"
                   f" --alias {p['alias']}" + (f" --mmproj {Path(p['mmproj']).expanduser()}"
                                               if p.get("mmproj") and "mmproj" not in p["args"] else "")
                   + (f" --ctx-size {p['ctx']}" if p.get("ctx") and "ctx-size" not in p["args"] else "")
                   + f" {p['args']}")
        else:
            cmd = os.path.expanduser(p["start"]).format(model=Path(p["model_file"]).expanduser(), port="${PORT}",
                                                       ctx=p.get("ctx") or 32768, alias=p["alias"], args=p["args"])
            if "{args}" not in p["start"]:
                cmd += f" {p['args']}"
        lines += [f'  "{p["alias"]}":', f"    cmd: {json.dumps(cmd.strip())}"]
        if p.get("env"):
            lines.append("    env:")
            lines += [f"      - {json.dumps(f'{k}={v}')}" for k, v in p["env"].items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("model")
    s.add_argument("name", help="llama.cpp backend or engine from the config")
    s.add_argument("--label", default="after")
    s.add_argument("--note")
    sub.add_parser("show")
    c = sub.add_parser("serve-config")
    c.add_argument("--out", required=True)
    c.add_argument("--models")
    a = ap.parse_args()
    cfg = load_config()

    if a.cmd == "save":
        results = Path(cfg.get("results_dir", "results")).expanduser()
        server = (cfg.get("backends") or {}).get(a.name, {}).get("server")
        prof = build_profile(cfg, a.model, a.name, a.label, results, a.note,
                             engine_version(server) if server else None, hw.summary())
        print(f"saved {save(cfg, prof)}")
    elif a.cmd == "show":
        for p in load_all(cfg):
            best = max(p["measured"]["at_depth"], key=lambda r: r["prompt_tokens"], default={})
            print(f"{p['model']:28} {p['engine']:10} {p.get('backend') or '':12} pp {best.get('pp_ts', 0):7.1f} "
                  f"tg {best.get('tg_ts', 0):6.2f} @ {best.get('prompt_tokens', 0)} tok  saved {p['saved']}")
    else:
        profs = load_all(cfg, a.models.split(",") if a.models else None)
        if not profs:
            sys.exit("no saved profiles")
        out = Path(a.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        one_binary = {p.get("server") for p in profs if p["engine"] == "llama.cpp"}
        if all(p["engine"] == "llama.cpp" for p in profs) and len(one_binary) == 1:
            f = out / "presets.ini"
            f.write_text(presets_ini(profs))
            env = " ".join(f"{k}={v}" for k, v in (profs[0].get("env") or {}).items())
            print(f"wrote {f}\nstart: {env + ' ' if env else ''}{Path(profs[0]['server']).expanduser()} "
                  f"--models-preset {f} --models-max 1 --host 127.0.0.1 --port 8080")
        else:
            f = out / "llama-swap.yaml"
            f.write_text(llama_swap_yaml(profs))
            print(f"wrote {f}\nstart: llama-swap --config {f} --listen 127.0.0.1:8080")


if __name__ == "__main__":
    main()
