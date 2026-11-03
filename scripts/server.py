"""Shared helpers: config, a throwaway test llama-server, HTTP requests.

The config is read from $TUNE_CONFIG or ./config.json (see config.example.json).
Every benchmark talks to its own server on a spare port; nothing here touches the
user's live server except optionally asking a llama-server router to unload
models so the GPU is free.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import hw

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("TUNE_CONFIG", "config.json")).expanduser()


def load_config():
    if not CONFIG_PATH.exists():
        # Show usage instead of a config error when the user only wants to know how to call a script.
        if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
            raise SystemExit((getattr(sys.modules.get("__main__"), "__doc__", None) or "").strip()
                             + f"\n\n(No config yet at {CONFIG_PATH}: run ./init_config.py to create one.)")
        raise SystemExit(f"no config at {CONFIG_PATH}: run ./init_config.py (or copy config.example.json "
                         "and edit it), or point TUNE_CONFIG at an existing one")
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg.setdefault("port", 8090)
    cfg.setdefault("results_dir", "results")
    cfg["results_dir"] = Path(cfg["results_dir"]).expanduser()
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


CFG = load_config()
PORT = CFG["port"]
URL = f"http://127.0.0.1:{PORT}"
KEY = os.environ.get(CFG.get("api_key_env") or "", "")
RESULTS = CFG["results_dir"]


def model(name):
    """Model entry from the config: alias, path and optional mmproj/draft."""
    try:
        m = CFG["models"][name]
    except KeyError:
        raise SystemExit(f"model {name!r} is not in the config (have: {', '.join(CFG['models'])})")
    return {**m, "path": str(Path(m["path"]).expanduser())}


def backend(name):
    """Binaries and environment for one backend, e.g. vulkan / hip / cuda / metal / both."""
    try:
        b = CFG["backends"][name]
    except KeyError:
        raise SystemExit(f"backend {name!r} is not in the config (have: {', '.join(CFG['backends'])})")
    return {"server": str(Path(b["server"]).expanduser()), "bench": str(Path(b.get("bench", "")).expanduser()),
            "env": b.get("env", {})}


def request(path, body=None, timeout=1800, url=URL):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data) if data[:1] in (b"{", b"[") else data.decode()


SELF_UNLOAD_HELP = """\
The agent running this skill uses model {model!r}, which is served by the router being tuned
({url}). Unloading the router's models would stop the agent itself, and the GPU can't run the
agent's model and a benchmark at the same time. Options:
  - run the tuning from an agent on a cloud model (Claude Code, Codex, or OpenCode with a cloud provider);
  - or run the scripts yourself in a terminal (see README, "Running scripts yourself");
  - if the agent's model is NOT on this machine's GPU after all, set agent.self_model to null in the config."""


def free_router(cfg=None):
    """Ask the user's llama-server router (if configured) to unload models so VRAM is free.

    Refuses when the agent running the skill is served by that router (config agent.self_model):
    unloading would kill the agent mid-run.
    """
    cfg = CFG if cfg is None else cfg
    url = cfg.get("router_url")
    if not url:
        return []
    try:
        models = request("/v1/models", url=url, timeout=5)["data"]
    except OSError:
        return []
    me = (cfg.get("agent") or {}).get("self_model")
    if me and any(me in (m["id"], *m.get("aliases", [])) for m in models):
        raise SystemExit(SELF_UNLOAD_HELP.format(model=me, url=url))
    loaded = [m["id"] for m in models if m.get("status", {}).get("value") not in (None, "unloaded")]
    if loaded:
        print(f"unloading from the live router {url}: {', '.join(loaded)}", flush=True)
    for mid in loaded:
        request("/models/unload", {"model": mid}, url=url, timeout=60)
    if loaded:
        time.sleep(3)
    return loaded


def vram_mib():
    """Used VRAM in MiB on the main GPU (the one with the most VRAM), or -1 if unknown."""
    return hw.vram_used_mib()


class Server:
    """A test llama-server on CFG['port'] for one model; use as a context manager."""

    def __init__(self, model_name, args, backend_name, log, prompts_dir=None):
        m, b = model(model_name), backend(backend_name)
        self.alias = m["alias"]
        self.cmd = [b["server"], "-m", m["path"], "--alias", m["alias"], "--host", "127.0.0.1",
                    "--port", str(PORT), "--jinja", "--metrics", *args]
        if prompts_dir:
            self.cmd += ["--log-prompts-dir", str(prompts_dir)]
        self.env = {**os.environ, **b["env"]}
        self.log = Path(log)
        self.proc = None

    def __enter__(self):
        try:
            request("/health", timeout=2)
            raise RuntimeError(f"port {PORT} is already in use; stop that server first")
        except OSError:
            pass
        free_router()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        # Own process group, so stopping it never needs pkill -f (which can match the caller's shell).
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited (out of memory or bad flag?), see {self.log}")
            try:
                if "ok" in json.dumps(request("/health", timeout=2)):
                    return self
            except OSError:
                pass
            time.sleep(1)
        raise RuntimeError("server did not become healthy in 10 minutes")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        time.sleep(2)

    def metrics(self):
        """Prometheus counters, e.g. llamacpp:prompt_tokens_total."""
        out = {}
        for line in request("/metrics").splitlines():
            if line and not line.startswith("#"):
                k, v = line.rsplit(" ", 1)
                out[k] = float(v)
        return out
