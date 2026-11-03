#!/usr/bin/env python3
"""2-minute check: read your current llama-server settings and point out likely speed left on the table.

Usage:
  quick_check.py --router http://127.0.0.1:8080      # every model a llama-server router knows
  quick_check.py --preset ~/models/presets.ini        # a router presets file
  quick_check.py --args "-m model.gguf -c 65536 ..."  # one llama-server command line
  quick_check.py --ollama [http://127.0.0.1:11434]    # Ollama: its models, settings and what's loaded
Options:
  --key-env NAME   env var holding the router's API key (default LLAMA_API_KEY)
  --json           machine-readable output

No config needed, no benchmarks, nothing is changed. Hardware is read from this machine,
so run it on the machine that serves the models. Findings are rules of thumb from
references/flags.md; a quick tune measures what they're really worth on your hardware.
"""
import argparse
import configparser
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_meta  # noqa: E402
import hw  # noqa: E402

# llama-server flag spellings (CLI short, CLI long, preset key) -> one canonical name
ALIASES = {
    "flash_attn": ["fa", "flash-attn"],
    "ngl": ["ngl", "n-gpu-layers", "gpu-layers"],
    "threads": ["t", "threads"],
    "ubatch": ["ub", "ubatch-size"],
    "batch": ["b", "batch-size"],
    "ctk": ["ctk", "cache-type-k"],
    "ctv": ["ctv", "cache-type-v"],
    "parallel": ["np", "parallel"],
    "ctx": ["c", "ctx-size"],
    "n_cpu_moe": ["ncmoe", "n-cpu-moe"],
    "cpu_moe": ["cmoe", "cpu-moe"],
    "override_tensor": ["ot", "override-tensor"],
    "jinja": ["jinja"],
    "no_jinja": ["no-jinja"],
    "model": ["m", "model"],
    "mmproj": ["mm", "mmproj"],
    "device": ["dev", "device"],
}
CANON = {spelling: name for name, spellings in ALIASES.items() for spelling in spellings}
BOOL_FLAGS = {"jinja", "no-jinja", "cpu-moe", "cmoe", "no-mmproj-offload"}
LOW_CACHE = {"q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
# Not tuning flags: the benchmark scripts set these themselves (model, port, ...) or they only affect logging.
NOT_TUNING = {"model", "alias", "host", "port", "api-key", "api-key-file", "metrics", "log-colors", "log-file",
              "log-prompts-dir", "models-preset", "models-max", "models-dir", "verbose", "v"}


def normalize(pairs):
    """[(key, value)] with any spelling -> {canonical: value}. Unknown keys are kept as-is."""
    out = {}
    for k, v in pairs:
        k = k.lstrip("-")
        out[CANON.get(k, k)] = v
    return out


def parse_cmdline(argv):
    """llama-server argv (list or string) -> normalized dict."""
    toks = shlex.split(argv) if isinstance(argv, str) else list(argv)
    pairs, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("-"):
            name = t.lstrip("-")
            if name in BOOL_FLAGS or i + 1 >= len(toks) or toks[i + 1].startswith("--") \
                    or (toks[i + 1].startswith("-") and not toks[i + 1][1:2].isdigit()):
                pairs.append((name, "true"))
                i += 1
            else:
                pairs.append((name, toks[i + 1]))
                i += 2
        else:
            i += 1  # the binary itself, or a stray value
    return normalize(pairs)


def to_cli(cfg):
    """Normalized config -> llama-server arguments, minus the ones the benchmark scripts set themselves.

    These are llama-server flags: paste them after the fixed arguments of validate_fill.py or
    bench_agentic.py to measure the user's current config as the baseline. (llama-bench, used by
    micro.py, spells several flags differently.)
    """
    out = []
    for k, v in cfg.items():
        flag = ALIASES[k][-1] if k in ALIASES else k
        if flag in NOT_TUNING or k in NOT_TUNING:
            continue
        v = str(v).strip()
        if v.lower() in ("true", "on") and (flag in BOOL_FLAGS or k in ("jinja", "cpu_moe")):
            out.append(f"--{flag}")
        elif v.lower() == "false" and (flag in BOOL_FLAGS or k in ("jinja", "cpu_moe")):
            continue
        else:
            out += [f"--{flag}", v]
    return shlex.join(out)


def parse_preset(path):
    """presets.ini -> {section: normalized dict}, with [*] merged into every section."""
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))
    cp.optionxform = str
    text = Path(path).expanduser().read_text()
    cp.read_string("[__top__]\n" + text)  # allow keys (version = 1) before the first section
    base = dict(cp["*"]) if cp.has_section("*") else {}
    return {s: normalize({**base, **dict(cp[s])}.items()) for s in cp.sections() if s not in ("*", "__top__")}


def from_router(url, key):
    req = urllib.request.Request(url.rstrip("/") + "/v1/models",
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)["data"]
    out, binary = {}, None
    for m in data:
        args = (m.get("status") or {}).get("args")
        if args:
            binary = binary or args[0]
            out[m["id"]] = parse_cmdline(args[1:])
    return out, binary


def list_devices(binary):
    if not binary:
        return []
    try:
        txt = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.split(":")[0].strip() for line in txt.splitlines() if line.startswith("  ") and ":" in line]


def _int(v, default=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _off(v):
    return str(v).strip().lower() in ("off", "0", "false", "no")


def check(cfg, meta, hwi):
    """Rules from references/flags.md. Returns [{'level', 'title', 'why', 'fix'}]; level is big/medium/small."""
    f = []

    def add(level, title, why, fix):
        f.append({"level": level, "title": title, "why": why, "fix": fix})

    gpu = hwi.get("gpu") or {}
    vram = gpu.get("total_mib", 0)
    cores = hwi.get("physical_cores") or 0
    ram = hwi.get("ram_mib", 0)
    size = meta["size_mib"] if meta else 0
    moe = bool(meta and meta["moe"])
    layers = meta["block_count"] if meta else 0
    experts_on_cpu = "n_cpu_moe" in cfg or "cpu_moe" in cfg or "override_tensor" in cfg
    ngl = _int(cfg.get("ngl"))
    ub = _int(cfg.get("ubatch"), 512)
    ctx = _int(cfg.get("ctx"))
    fits = size and vram and size < vram * 0.8  # weights alone; leaves room for KV cache and buffers

    if "flash_attn" in cfg and _off(cfg["flash_attn"]):
        add("big", "Flash attention is off",
            "At 16K context it measured +55% prompt and +23% generation speed (Vulkan), and it's required "
            "for a quantized V cache.", "flash-attn = on")
    if "no_jinja" in cfg:
        add("big", "Chat template disabled (--no-jinja)",
            "Without the model's own template, tool calls usually break, so coding agents fail.", "jinja = true")
    if size and vram and ram and size > vram + ram * 0.8:
        add("big", "Model is bigger than VRAM + RAM",
            f"{size / 1024:.1f} GB of weights vs {vram / 1024:.1f} GB VRAM + {ram / 1024:.1f} GB RAM: it will "
            "swap from disk or fail to load.", "use a smaller quant or a smaller model")
    if moe and size and vram and size > vram * 0.9 and not experts_on_cpu:
        add("big", "MoE model bigger than VRAM without expert offload",
            "Whole layers end up on the CPU. --n-cpu-moe keeps attention and the KV cache on the GPU and "
            "moves only the expert weights to RAM, which is usually much faster.",
            "ngl = 99 plus n-cpu-moe = N (lowest N that fits; a quick tune finds it)")
    if experts_on_cpu and ub <= 512:
        add("big", "Small batch size with experts in RAM",
            "With CPU experts, ubatch 512 -> 2048 nearly doubled prompt speed in the worked example "
            "(172 -> 526 t/s at 49K with the other changes). Agents re-read long prompts every turn.",
            "ubatch-size = 2048 and batch-size = 2048, if VRAM allows")
    if fits:
        if ngl is None and not experts_on_cpu:
            add("medium", "GPU layers left to --fit",
                "--fit can keep layers on the CPU even when the whole model fits: Gemma 12B went from 16 to "
                "29 t/s with ngl 99 in the worked example.", "n-gpu-layers = 99, then confirm it fits at full context")
        elif ngl is not None and layers and ngl < layers and not experts_on_cpu:
            add("big", f"Only {ngl} of {layers} layers on the GPU",
                "The weights look small enough to fit in VRAM. Every layer on the CPU slows generation a lot.",
                "n-gpu-layers = 99, then confirm it fits at full context")
    threads = _int(cfg.get("threads"))
    if threads and cores and threads > cores:
        add("medium", f"{threads} threads on {cores} physical cores",
            "SMT threads compete for the same cores; with CPU experts this can halve generation speed.",
            f"threads = {cores}")
    for k in ("ctk", "ctv"):
        if str(cfg.get(k, "")).lower() in LOW_CACHE:
            add("medium", f"KV cache {cfg[k]} ({'K' if k == 'ctk' else 'V'})",
                "Below q8_0 the cache hurts tool-call accuracy in agents.", f"cache-type-{k[-1]} = q8_0")
    if str(cfg.get("ctk", "")).lower() == "q8_0" and fits and size < vram * 0.6:
        add("small", "q8_0 KV cache on a model with VRAM to spare",
            "f16 cache is often 10-30% faster generation (measured +30% on HIP). It needs twice the cache memory.",
            "try cache-type-k/v = f16 and check it still fits at full context")
    par = _int(cfg.get("parallel"))
    if par and par > 1:
        add("medium", f"{par} parallel slots",
            f"The context is split between slots, so one agent request gets only part of it"
            f"{f' (~{ctx // par} tokens)' if ctx else ''}.", "parallel = 1 for a single user")
    if ctx and meta and meta["context_length"] and ctx > meta["context_length"]:
        add("small", "Context above what the model was trained for",
            f"{ctx} > {meta['context_length']}: quality drops past the trained length unless RoPE scaling is set.",
            f"ctx-size <= {meta['context_length']}")
    return f


def check_backends(devices, used):
    """Machine-wide: a build with two GPU backends where every model uses the same one."""
    gpu_devs = [d for d in devices if not d.upper().startswith("CPU")]
    if len({d.rstrip("0123456789") for d in gpu_devs}) > 1 and len(set(used)) <= 1:
        return [{"level": "medium", "title": f"Your build has {len(gpu_devs)} GPU backends ({', '.join(gpu_devs)}) "
                 "but every model uses the same one",
                 "why": "Backends differed by 20-50% per model in the worked example (HIP for the MoE model with "
                 "CPU experts, Vulkan for small dense models).",
                 "fix": "compare backends per model, then set device = per preset"}]
    return []


def _ollama_get(url, path, body=None):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def ollama_env():
    """OLLAMA_* settings of the Ollama service (systemd on Linux), then of this shell."""
    env = {}
    try:
        out = subprocess.run(["systemctl", "show", "ollama", "-p", "Environment"], capture_output=True,
                             text=True, timeout=5).stdout
        for tok in shlex.split(out.partition("=")[2]):
            k, _, v = tok.partition("=")
            if k.startswith("OLLAMA_"):
                env[k] = v
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    env.update({k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")})
    return env


def from_ollama(url):
    """[{'name', 'size_mib', 'params', 'meta', 'loaded'}] from a running Ollama."""
    tags = _ollama_get(url, "/api/tags").get("models", [])
    loaded = {m["name"]: m for m in _ollama_get(url, "/api/ps").get("models", [])}
    out = []
    for m in tags:
        show = _ollama_get(url, "/api/show", {"model": m["name"]})
        params = {}
        for line in (show.get("parameters") or "").splitlines():
            k, _, v = line.strip().partition(" ")
            params[k] = v.strip().strip('"')
        mi = show.get("model_info") or {}
        arch = mi.get("general.architecture", "")
        experts = int(mi.get(f"{arch}.expert_count") or 0)
        meta = {"arch": arch, "block_count": int(mi.get(f"{arch}.block_count") or 0), "expert_count": experts,
                "expert_used": int(mi.get(f"{arch}.expert_used_count") or 0),
                "context_length": int(mi.get(f"{arch}.context_length") or 0),
                "size_mib": int(m.get("size", 0)) // 2**20, "moe": experts > 0}
        out.append({"name": m["name"], "params": params, "meta": meta, "loaded": loaded.get(m["name"])})
    return out


def check_ollama(model, env, hwi, min_ctx=16384):
    """Rules for one Ollama model. Same finding format as check()."""
    f = []

    def add(level, title, why, fix):
        f.append({"level": level, "title": title, "why": why, "fix": fix})

    meta, params, loaded = model["meta"], model["params"], model.get("loaded")
    vram = (hwi.get("gpu") or {}).get("total_mib", 0)
    ram = hwi.get("ram_mib", 0)
    size = meta["size_mib"]
    ctx = _int(params.get("num_ctx")) or _int(env.get("OLLAMA_CONTEXT_LENGTH"))
    if loaded and loaded.get("context_length"):
        ctx = int(loaded["context_length"])
    if ctx is None:
        add("big", "Context size left at Ollama's default",
            "The default depends on the Ollama version and your VRAM, and on GPUs under 24 GB it's usually about 4K "
            "tokens. Coding agents send 12K+ tokens every turn, and Ollama cuts longer prompts off without an error.",
            f"PARAMETER num_ctx 32768 in a Modelfile (or OLLAMA_CONTEXT_LENGTH), at least {min_ctx} for agents")
    elif ctx < min_ctx:
        add("big", f"Context of {ctx} tokens",
            "Coding agents send 12K+ tokens every turn; anything past the context is cut off without an error.",
            f"num_ctx {max(min_ctx, 32768)} if it fits")
    if env.get("OLLAMA_FLASH_ATTENTION", "").lower() not in ("1", "true"):
        add("medium", "Flash attention not switched on",
            "It measured +55% prompt and +23% generation speed at 16K context (llama.cpp, Vulkan), and Ollama needs "
            "it for a quantized KV cache. Recent Ollama versions turn it on by themselves for some models.",
            "OLLAMA_FLASH_ATTENTION=1 in the service environment")
    if env.get("OLLAMA_KV_CACHE_TYPE", "").lower() == "q4_0":
        add("medium", "KV cache q4_0", "Below q8_0 the cache hurts tool-call accuracy in agents.",
            "OLLAMA_KV_CACHE_TYPE=q8_0 (or f16 if it fits)")
    par = _int(env.get("OLLAMA_NUM_PARALLEL"))
    if par and par > 1:
        add("medium", f"OLLAMA_NUM_PARALLEL={par}",
            "Each parallel slot needs its own context memory, which pushes layers off the GPU.",
            "OLLAMA_NUM_PARALLEL=1 for a single user")
    if size and vram and ram and size > vram + ram * 0.8:
        add("big", "Model is bigger than VRAM + RAM",
            f"{size / 1024:.1f} GB of weights vs {vram / 1024:.1f} GB VRAM + {ram / 1024:.1f} GB RAM.",
            "use a smaller quant or a smaller model")
    if loaded and loaded.get("size") and loaded.get("size_vram") is not None:
        on_cpu = 1 - loaded["size_vram"] / loaded["size"]
        if on_cpu > 0.02:
            pct = f"{on_cpu:.0%} of the loaded model is in system RAM"
            if meta["moe"]:
                add("big", f"MoE model split by whole layers ({pct})",
                    "Ollama moves whole layers to the CPU. llama-server's --n-cpu-moe keeps attention and the KV cache "
                    "on the GPU and moves only the experts, which is usually much faster (3x faster prompts for "
                    "Qwen3.6-35B-A3B on 8 GB in the worked example).",
                    "run this model with llama-server; a quick tune finds the --n-cpu-moe value")
            elif vram and size < vram * 0.8:
                add("medium", f"Partly on the CPU although the weights fit ({pct})",
                    "Usually the context is too big for what's left of VRAM, or other apps are using it.",
                    "a smaller num_ctx, OLLAMA_KV_CACHE_TYPE=q8_0, or close other GPU apps")
            else:
                add("small", pct, "Expected for a model this size; generation will be slower than fully on the GPU.",
                    "a smaller quant if it's too slow")
    return f


def describe(meta):
    if not meta:
        return "model file not found on this machine"
    kind = f"MoE {meta['expert_used']}/{meta['expert_count']} experts" if meta["moe"] else "dense"
    return f"{meta['arch']}, {kind}, {meta['block_count']} layers, {meta['size_mib'] / 1024:.1f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--router")
    src.add_argument("--preset")
    src.add_argument("--args")
    src.add_argument("--ollama", nargs="?", const="http://127.0.0.1:11434", metavar="URL")
    ap.add_argument("--key-env", default="LLAMA_API_KEY")
    ap.add_argument("--server-bin", help="llama-server binary, to list its GPU backends")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.ollama:
        return main_ollama(a)
    binary = a.server_bin
    if a.router:
        models, router_bin = from_router(a.router, os.environ.get(a.key_env, ""))
        binary = binary or router_bin
    elif a.preset:
        models = parse_preset(a.preset)
    else:
        cfg = parse_cmdline(a.args)
        models = {Path(cfg.get("model", "model")).stem: cfg}
    if not models:
        sys.exit("no models found (a router only reports launch args for models it has started at least once "
                 "or has presets for)")

    hwi = hw.summary()
    devices = list_devices(binary)
    used = [c.get("device", "") for c in models.values()]
    report = []
    for name, cfg in models.items():
        meta = None
        if cfg.get("model") and Path(cfg["model"]).expanduser().exists():
            try:
                meta = gguf_meta.meta(cfg["model"])
            except (OSError, ValueError):
                meta = None
        report.append({"model": name, "device": cfg.get("device", ""), "about": describe(meta),
                       "server_args": to_cli(cfg), "findings": check(cfg, meta, hwi)})
    machine = check_backends(devices, used)

    if a.json:
        print(json.dumps({"hardware": hwi, "devices": devices, "machine": machine, "models": report}, indent=1))
        return
    print_report(hwi, devices, report, machine)


def print_report(hwi, devices, report, machine):
    g = hwi.get("gpu") or {}
    print(f"Hardware: {g.get('name', 'no GPU found')} {g.get('total_mib', 0) / 1024:.1f} GB VRAM, "
          f"{hwi['physical_cores']} physical cores, {hwi['ram_mib'] / 1024:.0f} GB RAM"
          + (f"; backends: {', '.join(devices)}" if devices else ""))
    order = {"big": 0, "medium": 1, "small": 2}
    flagged = 0
    for r in report:
        print(f"\n== {r['model']}  ({r['about']}){'  device ' + r['device'] if r['device'] else ''}")
        if r.get("server_args"):
            print(f"   current flags: {r['server_args']}")
        if not r["findings"]:
            print("   looks tuned: no rule-of-thumb problems found")
            continue
        flagged += any(x["level"] != "small" for x in r["findings"])
        for x in sorted(r["findings"], key=lambda x: order[x["level"]]):
            print(f"   [{x['level']:6}] {x['title']}\n            {x['why']}\n            fix: {x['fix']}")
    for x in machine:
        print(f"\n== all models\n   [{x['level']:6}] {x['title']}\n            {x['why']}\n            fix: {x['fix']}")
    print(f"\n{flagged} of {len(report)} model(s) likely have speed left on the table."
          + (" A quick tune (20-40 min per model) measures and fixes it." if flagged else ""))


def main_ollama(a):
    try:
        models = from_ollama(a.ollama)
    except OSError as e:
        sys.exit(f"can't reach Ollama at {a.ollama}: {e}")
    if not models:
        sys.exit("Ollama has no models")
    hwi, env = hw.summary(), ollama_env()
    report = []
    for m in models:
        about = describe(m["meta"])
        if m["loaded"]:
            about += ", loaded now"
        report.append({"model": m["name"], "device": "", "about": about, "findings": check_ollama(m, env, hwi)})
    if a.json:
        print(json.dumps({"hardware": hwi, "ollama_env": env, "machine": [], "models": report}, indent=1))
        return
    print_report(hwi, [], report, [])
    print("Ollama settings: " + (", ".join(f"{k}={v}" for k, v in sorted(env.items())) or "none set") +
          ". Models that aren't loaded right now are checked on their settings only; load one (run a prompt) "
          "and check again to see its GPU/CPU split.")


if __name__ == "__main__":
    main()
