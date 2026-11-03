#!/usr/bin/env python3
"""Before/after table for the results report, built from the measurements (no retyping numbers).

Usage: before_after.py <model> [--before LABEL] [--after LABEL] [--results DIR]
  Labels default to "before" and "after": the labels SKILL.md uses for the baseline
  (the user's current config) and the chosen config.

Reads whichever of these exist in the results folder and prints Markdown:
  validate.csv   real-server speed at each depth (the headline table)
  micro.csv      llama-bench speed at each depth (rows whose label is exactly LABEL)
  agentic.csv    agent tasks: passed, wall time, speed during the tasks
The results folder is the config's results_dir ($TUNE_CONFIG or ./config.json), else ./results.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def load_config():
    cfg = Path(os.environ.get("TUNE_CONFIG", "config.json")).expanduser()
    return json.loads(cfg.read_text()) if cfg.exists() else {}


def results_dir(arg, cfg):
    if arg:
        return Path(arg).expanduser()
    return Path(cfg.get("results_dir", "results")).expanduser()


def _rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def change(before, after):
    if not before or after is None:
        return "–"
    pct = (after / before - 1) * 100
    if after / before >= 2:
        return f"**{after / before:.1f}×**"
    return f"**{pct:+.0f}%**" if abs(pct) >= 5 else f"{pct:+.0f}%"


def fmt(v, digits=1):
    return "–" if v is None else f"{v:,.{digits}f}"


def _depth(n):
    n = int(float(n))
    return f"{n / 1000:.0f}K" if n >= 1000 else str(n)


def validate_table(rows, model, before, after):
    by = defaultdict(dict)  # depth -> label -> row
    for r in rows:
        if r["model"] == model and r["label"] in (before, after):
            # Same target depth gives a slightly different token count per run: bucket to 500 tokens.
            by[round(int(r["prompt_n"]) / 500) * 500][r["label"]] = r
    if not by:
        return None
    out = ["| Context depth | Prompt before | Prompt after | Change | Generation before | Generation after | Change | VRAM after |",
           "|---|---|---|---|---|---|---|---|"]
    for d in sorted(by):
        b, a = by[d].get(before, {}), by[d].get(after, {})
        pb, pa, tb, ta = _f(b.get("pp_ts")), _f(a.get("pp_ts")), _f(b.get("tg_ts")), _f(a.get("tg_ts"))
        vram = _f(a.get("vram_mib"))
        out.append(f"| {_depth(d)} | {fmt(pb, 0)} | {fmt(pa, 0)} | {change(pb, pa)} | {fmt(tb)} | {fmt(ta)} | "
                   f"{change(tb, ta)} | {'–' if vram is None else f'{vram / 1024:.1f} GB'} |")
    return out


def micro_table(rows, model_file, before, after):
    """micro.csv has no model column: when the model's file is known, keep only its rows."""
    by = defaultdict(lambda: defaultdict(dict))  # depth -> label -> pp/tg
    for r in rows:
        if model_file and Path(r.get("model_filename", "")).name != model_file:
            continue
        if r["label"] in (before, after):
            kind = "pp" if int(r["n_prompt"]) else "tg"
            by[int(r["n_depth"])][r["label"]][kind] = _f(r["avg_ts"])
    if not by:
        return None
    out = ["| Depth (llama-bench) | Prompt before | Prompt after | Change | Generation before | Generation after | Change |",
           "|---|---|---|---|---|---|---|"]
    for d in sorted(by):
        b, a = by[d][before], by[d][after]
        out.append(f"| {_depth(d)} | {fmt(b.get('pp'), 0)} | {fmt(a.get('pp'), 0)} | {change(b.get('pp'), a.get('pp'))} | "
                   f"{fmt(b.get('tg'))} | {fmt(a.get('tg'))} | {change(b.get('tg'), a.get('tg'))} |")
    return out


def agentic_table(rows, model, before, after):
    stats = {}
    for label in (before, after):
        rs = [r for r in rows if r["model"] == model and r["label"] == label]
        if rs:
            n = len(rs)
            passed = sum(r["passed"] in ("True", "true", "1") for r in rs)
            mean = lambda k: sum(_f(r[k]) or 0 for r in rs) / n  # noqa: E731
            stats[label] = {"passed": f"{passed}/{n}", "wall": mean("wall_s"), "pp": mean("pp_ts"), "tg": mean("tg_ts")}
    if not stats:
        return None
    b, a = stats.get(before, {}), stats.get(after, {})
    return ["| Agent tasks | Before | After | Change |", "|---|---|---|---|",
            f"| Passed | {b.get('passed', '–')} | {a.get('passed', '–')} | |",
            f"| Mean wall time per task | {fmt(b.get('wall'), 0)} s | {fmt(a.get('wall'), 0)} s | "
            f"{change(a.get('wall'), b.get('wall')) if b and a else '–'} faster |",
            f"| Prompt speed during tasks | {fmt(b.get('pp'), 0)} | {fmt(a.get('pp'), 0)} | {change(b.get('pp'), a.get('pp'))} |",
            f"| Generation during tasks | {fmt(b.get('tg'))} | {fmt(a.get('tg'))} | {change(b.get('tg'), a.get('tg'))} |"]


def report(results, model, before, after, model_file=None):
    parts = []
    t = validate_table(_rows(results / "validate.csv"), model, before, after)
    if t:
        parts += ["Real server, tokens/s at the agent's context depths:", "", *t, ""]
    t = micro_table(_rows(results / "micro.csv"), model_file, before, after)
    if t:
        parts += ["llama-bench, tokens/s:", "", *t, ""]
    t = agentic_table(_rows(results / "agentic.csv"), model, before, after)
    if t:
        parts += [*t, ""]
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--before", default="before")
    ap.add_argument("--after", default="after")
    ap.add_argument("--results")
    a = ap.parse_args()
    cfg = load_config()
    res = results_dir(a.results, cfg)
    path = (cfg.get("models", {}).get(a.model) or {}).get("path")
    parts = report(res, a.model, a.before, a.after, Path(path).name if path else None)
    if not parts:
        raise SystemExit(f"no rows for model {a.model!r} with labels {a.before!r}/{a.after!r} in {res}")
    print(f"### {a.model}: before ({a.before}) vs after ({a.after})\n")
    print("\n".join(parts).rstrip())


if __name__ == "__main__":
    main()
