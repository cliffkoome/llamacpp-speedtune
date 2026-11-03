#!/usr/bin/env python3
"""Measure any OpenAI-compatible server at the agent's context depths, timed from the client.

Usage: bench_openai.py <model> <backend-or-engine> <label> <depths> [server args...]
       bench_openai.py --url http://127.0.0.1:8080 --served-name <id> <model> <name> <label> <depths>

Use this to compare **engines** (llama.cpp vs ExLlama, MLX, ik_llama.cpp, ...): every engine is
measured the same way, by streaming a completion and timing it from outside. Prompt speed =
prompt tokens / time to first token; generation speed = tokens after the first / time after it.
Server-side numbers (validate_fill.py) are more precise for llama.cpp alone, but other engines
don't report them, so engine comparisons must use this script for every contender, llama.cpp too.

<backend-or-engine> is a llama.cpp backend or an entry under "engines" in the config; the
server is started and stopped like validate_fill.py does. With --url, an already running
server is measured instead (nothing is started). Each depth uses a fresh prefix, so no prompt
cache is reused. Appends to results/openai.csv (same columns as validate.csv).
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

from server import KEY, RESULTS, ROOT, URL, make_server, vram_mib

FILLER = (ROOT / "references/methodology.md").read_text() + (ROOT / "references/gotchas.md").read_text()
CHARS_PER_TOKEN = 3.6  # first guess for sizing the prompt; the server's own count is reported when it gives one


def prompt_of(tokens):
    text, need = "", int(tokens * CHARS_PER_TOKEN)
    while len(text) < need:
        text += FILLER
    return f"Session {uuid.uuid4()}.\n" + text[:need]


def _post_stream(url, path, body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=3600)


def parse_stream(lines, t0, clock=time.time):
    """SSE lines -> (t_first, t_last, n_chunks, usage): times of the first and last token. Pure apart from the clock."""
    t_first, t_last, n, usage = None, None, 0, None
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            piece = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
            if piece:
                n += 1
                t_last = clock()
                if t_first is None:
                    t_first = t_last
    return t_first, t_last, n, usage


def measure(url, served_name, depth, n_predict=128):
    """One streamed completion. Returns {'prompt_n', 'pp_ts', 'tg_ts', 'wall_s', 'endpoint'}."""
    prompt = prompt_of(depth) + "\n\nSummarise the above in one paragraph."
    body = {"model": served_name, "prompt": prompt, "max_tokens": n_predict, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_prompt": False}
    endpoint = "/v1/completions"
    t0 = time.time()
    try:
        resp = _post_stream(url, endpoint, body)
    except urllib.error.HTTPError as e:
        if e.code not in (400, 404, 405):
            raise
        # Chat-only engine: same prompt as one user message, thinking off where the engine supports it.
        endpoint = "/v1/chat/completions"
        body = {k: v for k, v in body.items() if k != "prompt"}
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        resp = _post_stream(url, endpoint, body)
    with resp:
        t_first, t_end, chunks, usage = parse_stream(resp, t0)
    if t_first is None:
        raise RuntimeError("the server streamed no tokens")
    p_n = (usage or {}).get("prompt_tokens") or int(len(prompt) / CHARS_PER_TOKEN)
    g_n = (usage or {}).get("completion_tokens") or chunks
    ttft = t_first - t0
    return {"prompt_n": p_n, "pp_ts": round(p_n / ttft, 1) if ttft > 0 else 0,
            "tg_ts": round((g_n - 1) / (t_end - t_first), 2) if g_n > 1 and t_end > t_first else 0,
            "wall_s": round(t_end - t0, 1), "endpoint": endpoint, "counted": "server" if usage else "estimated"}


def run(url, served_name, mdl, name, label, depths, args):
    rows = []
    loaded = vram_mib()
    for d in depths:
        r = measure(url, served_name, d)
        rows.append({"model": mdl, "backend": name, "label": label, "args": " ".join(args), "prompt_n": r["prompt_n"],
                     "pp_ts": r["pp_ts"], "tg_ts": r["tg_ts"], "wall_s": r["wall_s"], "vram_loaded_mib": loaded,
                     "vram_mib": vram_mib()})
        print(f"{name:10} {label:24} {r['prompt_n']:6} tok: pp {r['pp_ts']:7.1f} t/s, tg {r['tg_ts']:6.2f} t/s "
              f"({r['endpoint']}, tokens {r['counted']})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="measure this running server instead of starting one")
    ap.add_argument("--served-name", help="model id on that server (default: the model's alias)")
    ap.add_argument("model")
    ap.add_argument("name", help="llama.cpp backend or engine name from the config")
    ap.add_argument("label")
    ap.add_argument("depths", help="comma-separated token counts, e.g. 12000,25000,49000")
    a, args = ap.parse_known_args()
    depths = [int(x) for x in a.depths.split(",")]
    try:
        if a.url:
            from server import model
            rows = run(a.url.rstrip("/"), a.served_name or model(a.model)["alias"], a.model, a.name, a.label,
                       depths, args)
        else:
            with make_server(a.model, args, a.name, log=RESULTS / f"logs/openai-{a.model}-{a.label}.log") as s:
                rows = run(URL, s.alias, a.model, a.name, a.label, depths, args)
    except Exception as e:  # report and move on: a failed config is a result too
        print(f"{a.name:10} {a.label:24} FAILED: {e}")
        sys.exit(1)
    out = RESULTS / "openai.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
