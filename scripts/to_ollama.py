#!/usr/bin/env python3
"""Turn a tuned llama-server config into Ollama settings (Modelfile + environment).

Usage:
  to_ollama.py --preset ~/models/presets.ini --model qwen3.6-35b-a3b [--name qwen36-tuned]
  to_ollama.py --args "-m model.gguf -ngl 99 -c 32768 -fa on -ub 1024 -ctk q8_0 -ctv q8_0 -t 8"

Prints a Modelfile, the environment variables to set for the Ollama service, how to
apply them, and what can't be carried over. Nothing is written or changed.

Ollama exposes fewer knobs than llama-server, so the result is a starting point:
measure it (README: Ollama) and compare with the llama-server numbers.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_meta  # noqa: E402
from quick_check import parse_cmdline, parse_preset  # noqa: E402


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def convert(cfg, meta=None):
    """Normalized llama-server config -> {'modelfile': [lines], 'env': {k: v}, 'lost': [str], 'notes': [str]}."""
    lines, env, lost, notes = [], {}, [], []
    model = cfg.get("model")
    lines.append(f"FROM {Path(model).expanduser() if model else '<path to model.gguf>'}")

    ctx = _int(cfg.get("ctx"))
    if ctx:
        lines.append(f"PARAMETER num_ctx {ctx}")
    else:
        notes.append("No context size set: Ollama's default is small, so set num_ctx to what your agent needs.")

    ngl = _int(cfg.get("ngl"))
    layers = meta["block_count"] if meta else 0
    n_cpu_moe = _int(cfg.get("n_cpu_moe"))
    if "cpu_moe" in cfg:
        n_cpu_moe = layers or None
    if n_cpu_moe is not None:
        if layers:
            approx = max(layers - n_cpu_moe, 0)
            lines.append(f"# approximate: llama-server kept attention on the GPU for all {layers} layers and moved only "
                         f"the experts of {n_cpu_moe} to RAM; Ollama can only offload whole layers")
            lines.append(f"PARAMETER num_gpu {approx}")
        lost.append(f"--n-cpu-moe {n_cpu_moe}: Ollama moves whole layers, not just experts, so expect slower prompt "
                    "and generation than llama-server. Try num_gpu a few layers higher or lower, or remove it and "
                    "let Ollama decide; measure each.")
    elif ngl is not None:
        lines.append(f"PARAMETER num_gpu {999 if (ngl >= 99 or (layers and ngl >= layers)) else ngl}")

    threads = _int(cfg.get("threads"))
    if threads:
        lines.append(f"PARAMETER num_thread {threads}")
    ub, b = _int(cfg.get("ubatch")), _int(cfg.get("batch"))
    if ub or b:
        lines.append(f"PARAMETER num_batch {ub or b}")
        if ub and b and ub != b:
            lost.append(f"Separate batch ({b}) and ubatch ({ub}) sizes: Ollama has one num_batch; it's set to the "
                        "ubatch size, which is the one that drives prompt speed.")

    fa = str(cfg.get("flash_attn", "")).lower()
    if fa in ("on", "1", "true", "auto"):
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    ctk, ctv = cfg.get("ctk"), cfg.get("ctv")
    if ctk or ctv:
        kv = (ctk or ctv).lower()
        if kv in ("f16", "q8_0", "q4_0"):
            env["OLLAMA_KV_CACHE_TYPE"] = kv
            env.setdefault("OLLAMA_FLASH_ATTENTION", "1")  # a quantized cache needs it
        else:
            lost.append(f"KV cache type {kv}: Ollama supports f16, q8_0 and q4_0 only.")
        if ctk and ctv and ctk.lower() != ctv.lower():
            lost.append(f"Different K/V cache types ({ctk}/{ctv}): Ollama uses one type for both ({kv}).")
        notes.append("OLLAMA_KV_CACHE_TYPE applies to every model the Ollama service runs, not just this one.")
    par = _int(cfg.get("parallel"))
    env["OLLAMA_NUM_PARALLEL"] = str(par or 1)

    if cfg.get("device"):
        lost.append(f"device = {cfg['device']}: Ollama picks the GPU library itself, so you can't choose HIP vs "
                    "Vulkan per model.")
    for key, why in (("model-draft", "speculative decoding (draft model)"), ("md", "speculative decoding (draft model)"),
                     ("spec-type", "speculative decoding"), ("cache-ram", "the RAM prompt cache"),
                     ("ctx-checkpoints", "context checkpoints"), ("no-mmproj-offload", "keeping the vision encoder on the CPU")):
        if key in cfg:
            lost.append(f"{key}: Ollama has no setting for {why}.")
    if cfg.get("mmproj"):
        notes.append("Vision: Ollama may not pick up a separate mmproj file from a Modelfile; if images don't work, "
                     "use the model from Ollama's own library and apply the PARAMETER lines to it.")
    notes.append("Ollama reads the chat template from the GGUF for most models; if tool calls fail, add a TEMPLATE.")
    return {"modelfile": lines, "env": env, "lost": lost, "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset")
    src.add_argument("--args")
    ap.add_argument("--model", help="preset section to convert (with --preset)")
    ap.add_argument("--name", help="name for the Ollama model (default: the preset section or file name)")
    a = ap.parse_args()

    if a.preset:
        presets = parse_preset(a.preset)
        if a.model not in presets:
            sys.exit(f"--model must be one of: {', '.join(presets)}")
        cfg, name = presets[a.model], a.model
    else:
        cfg = parse_cmdline(a.args)
        name = Path(cfg.get("model", "model")).stem
    name = (a.name or name).replace(":", "-").lower()
    meta = None
    if cfg.get("model") and Path(cfg["model"]).expanduser().exists():
        meta = gguf_meta.meta(cfg["model"])
    out = convert(cfg, meta)

    print("# ---- Modelfile ----")
    print("\n".join(out["modelfile"]))
    print("\n# ---- Ollama service environment ----")
    for k, v in out["env"].items():
        print(f"{k}={v}")
    print(f"""
# ---- Apply ----
# 1. Save the Modelfile above as Modelfile, then:   ollama create {name} -f Modelfile
# 2. Set the environment for the service. Linux (systemd): sudo systemctl edit ollama, add under [Service]:
{chr(10).join(f'#      Environment="{k}={v}"' for k, v in out['env'].items())}
#    then: sudo systemctl restart ollama.   macOS app: launchctl setenv NAME value, then restart Ollama.
# 3. Measure (generation t/s = eval_count / eval_duration * 1e9):
#    curl -s localhost:11434/api/generate -d '{{"model":"{name}","prompt":"Write a long story","stream":false}}'""")
    if out["lost"]:
        print("\n# ---- Can't be carried over ----")
        for x in out["lost"]:
            print(f"# - {x}")
    if out["notes"]:
        print("\n# ---- Notes ----")
        for x in out["notes"]:
            print(f"# - {x}")


if __name__ == "__main__":
    main()
