#!/usr/bin/env python3
"""Create config.json from what's on this machine: llama.cpp builds, their GPU backends, and GGUF models.

Usage: init_config.py [--out config.json] [--force] [model-dir ...]
  default model dirs: ~/models ~/.cache/llama.cpp ~/.cache/huggingface

Read-only apart from writing the config file (never overwritten without --force).
Review the result: set ctx to what your agent needs, router_url if you run a
llama-server router, and agent.self_model if the agent running this skill uses one
of these models (see SKILL.md, Safety rules).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_meta  # noqa: E402

DEFAULT_DIRS = ["~/models", "~/.cache/llama.cpp", "~/.cache/huggingface"]
# Consumer AMD chips run ROCm by posing as their family's supported chip (references/backends.md).
GFX_OVERRIDE = [(re.compile(r"gfx103[1-6]"), "10.3.0"), (re.compile(r"gfx110[1-3]"), "11.0.0")]


def find_servers():
    found = set()
    on_path = shutil.which("llama-server")
    if on_path:
        found.add(str(Path(on_path).resolve()))
    home = Path.home()
    for root, dirs, files in os.walk(home):
        depth = Path(root).relative_to(home).parts
        if len(depth) >= 8 or any(p.startswith(".") and p not in (".local", ".cache") for p in depth[:1]):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "proc")]
        if "llama-server" in files:
            p = Path(root) / "llama-server"
            if os.access(p, os.X_OK):
                found.add(str(p.resolve()))
    return sorted(found)


def amd_override():
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return os.environ["HSA_OVERRIDE_GFX_VERSION"]
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for pat, ver in GFX_OVERRIDE:
        if pat.search(out):
            return ver
    return None


def devices(binary, env):
    try:
        out = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=60,
                             env={**os.environ, **env}).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.split(":")[0].strip() for line in out.splitlines() if line.startswith("  ") and ":" in line]


def backend_name(devs):
    kinds = []
    for d in devs:
        k = {"ROCm": "hip", "CUDA": "cuda", "Vulkan": "vulkan", "MTL": "metal", "Metal": "metal",
             "SYCL": "sycl"}.get(d.rstrip("0123456789"))
        if k and k not in kinds:
            kinds.append(k)
    return "-".join(kinds) or "cpu"


def find_models(dirs):
    models = {}
    files = []
    for d in dirs:
        d = Path(d).expanduser()
        if d.is_dir():
            files += [p for p in d.rglob("*.gguf") if p.stat().st_size > 50 * 2**20]
    metas = {}
    for p in files:
        try:
            metas[p] = gguf_meta.meta(p)
        except (OSError, ValueError):
            pass
    for p, m in sorted(metas.items()):
        # Skip vision projectors, draft/assistant models and non-text models (no transformer layers).
        if m["arch"] == "clip" or "assistant" in m["arch"] or "draft" in p.stem.lower() or m["block_count"] <= 4:
            continue
        key = re.sub(r"[^a-z0-9.]+", "-", p.stem.lower()).strip("-")
        key = re.sub(r"-(ud-)?(i?q\d.*|f16|bf16|f32)$", "", key) or key
        entry = {"alias": key, "path": str(p).replace(str(Path.home()), "~", 1),
                 "ctx": min(m["context_length"] or 32768, 32768)}
        siblings = [q for q in metas if q.parent == p.parent and q != p]
        mm = [q for q in siblings if metas[q]["arch"] == "clip"]
        dr = [q for q in siblings if "assistant" in metas[q]["arch"] or "draft" in q.stem.lower()]
        if mm:
            entry["mmproj"] = str(mm[0]).replace(str(Path.home()), "~", 1)
        if dr:
            entry["draft"] = str(dr[0]).replace(str(Path.home()), "~", 1)
        models[key] = entry
    return models


def build_config(servers, models, override):
    backends = {}
    for s in servers:
        env = {}
        devs = devices(s, {"HSA_OVERRIDE_GFX_VERSION": override} if override else {})
        if override and any(d.startswith("ROCm") for d in devs):
            env["HSA_OVERRIDE_GFX_VERSION"] = override
        name, n = backend_name(devs), 2
        base = name
        while name in backends:
            name, n = f"{base}{n}", n + 1
        bench = Path(s).with_name("llama-bench")
        backends[name] = {"server": s.replace(str(Path.home()), "~", 1),
                          "bench": str(bench).replace(str(Path.home()), "~", 1) if bench.exists() else "",
                          "env": env, "_devices": devs}
    pytest = shutil.which("pytest")
    return {
        "_comment": "Generated by init_config.py. Check ctx (your agent's real need), router_url, agent.self_model.",
        "port": 8090,
        "results_dir": "results",
        "api_key_env": "LLAMA_API_KEY",
        "router_url": None,
        "profiles_dir": "~/.config/llamacpp-speedtune/profiles",
        "backends": backends,
        "engines": {},
        "models": models,
        "agent": {"type": "opencode", "bin": "opencode",
                  "python_bin_dir": str(Path(pytest).parent) if pytest else None, "self_model": None},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="config.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and not a.force:
        sys.exit(f"{out} already exists; use --force to replace it")
    cfg = build_config(find_servers(), find_models(a.dirs or DEFAULT_DIRS), amd_override())
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {out}")
    for name, b in cfg["backends"].items():
        print(f"  backend {name:12} {b['server']}  devices: {', '.join(b['_devices']) or 'CPU only'}")
    for name, m in cfg["models"].items():
        print(f"  model   {name:28} ctx {m['ctx']}{'  +mmproj' if 'mmproj' in m else ''}{'  +draft' if 'draft' in m else ''}")
    if not cfg["backends"]:
        print("  no llama-server found: see references/backends.md to build one")
    if not cfg["agent"]["python_bin_dir"]:
        print("  pytest not found: set agent.python_bin_dir to a venv's bin/ with pytest before the agent tasks")
    print(f"Next: review it, then export TUNE_CONFIG={out.resolve()}")


if __name__ == "__main__":
    main()
