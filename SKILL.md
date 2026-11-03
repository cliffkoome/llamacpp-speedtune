---
name: llamacpp-speedtune
description: Make llama.cpp faster for the user's coding agent. Benchmark and tune local LLMs served with llama.cpp (llama-server) for the machine they run on, with coding agents like OpenCode as the main workload. Finds the best backend per model (CUDA, HIP/ROCm, Vulkan, Metal), batch sizes, KV cache type, MoE expert offload, threads and speculative decoding, measured at the context depths a coding agent really reaches, confirmed in a real server and on real agent tasks, then applied and documented. Also has a 2-minute no-benchmark check of current settings, and converts a tuned config to Ollama settings. Use when the user wants to speed up, benchmark, tune or optimise local models, check whether their llama-server or Ollama settings are any good, compare backends, pick llama-server flags, or set up a multi-model llama-server router.
---

# llamacpp-speedtune

You're tuning the user's local models on their own hardware. The goal is the fastest **working** config for each model under the user's real workload, usually a coding agent. Every claim must come from a measurement on this machine. Today's defaults and other people's numbers are starting points only.

Read before starting:
- `references/methodology.md`: why to measure at depth, agent compaction maths, the test ladder.
- `references/gotchas.md`: pitfalls that silently corrupt results. Read it fully before running anything.
- Then, as needed: `references/flags.md` (what to sweep and what's settled), `references/backends.md` (build recipes), `references/example-rx6600xt.md` (a complete worked example).

## Safety rules
- **Ask before** using sudo, changing a running service, editing the user's agent or server config, installing packages or building large projects. Prepare the change, show it, and apply it once approved.
- **Never kill processes by pattern** (`pkill -f`, `killall`): the pattern can match your own shell. Kill by PID.
- **Never unload the model you're running on.** Before any benchmark, find out which model *you* (the agent running this skill) use. If it's served by the user's local server or GPU, a tune would unload it and stop you mid-run, and the GPU can't serve you and a benchmark at once. Stop and tell the user: run the tune from an agent on a cloud model, or run the scripts themselves (README, "Running scripts yourself"). The 2-minute check is fine either way: it changes nothing. Record your model as `agent.self_model` in the config; `free_router()` refuses to unload it.
- **One test server at a time**, on a spare port (config `port`, default 8090). The user's live server stays untouched, except that you unload its models (`router_url`) so the GPU is free. Tell the user the GPU will be busy.
- **Don't trust a run you haven't checked.** After the first agentic task, inspect the run folder: did the agent edit the right files? Did the check really fail or pass? See gotchas.
- Keep the user posted: one line per finished experiment with the key numbers.

## Start with the 2-minute check
Unless the user asked straight away for a tune, start here. It needs no config, runs no benchmarks and changes nothing, so the GPU and the user's server stay free:
- `scripts/quick_check.py --router <url>` for a llama-server router (API key from `$LLAMA_API_KEY`, or `--key-env`), `--preset <presets.ini>`, or `--args "<llama-server command line>"`. Add `--json` to parse it.
- For Ollama: `scripts/quick_check.py --ollama [url]` reads its models, their settings, the service environment and, for loaded models, how much sits in system RAM. Models that aren't loaded are checked on their settings only.
- Report the findings in plain words, biggest first, and say whether a quick tune looks worth it. The rules come from flags.md; a tune measures what they're really worth.

## Choose a mode and give a time estimate first
Before any benchmark, run step 1 (inventory, ~2 minutes), then tell the user **which mode, what it will do, how long it should take, and that the GPU (and any live model server) will be busy the whole time**. Wait for a yes. If the user already named a mode, still state the estimate before starting.

| | **Quick tune** (default) | **Comprehensive tune** |
|---|---|---|
| Goal | A big, safe win today | The best config this hardware can do, with evidence |
| Builds | None: uses the llama.cpp builds already installed | Builds every possible backend from the same commit (asks first) |
| Per model | Rule-of-thumb start (flags.md), one sweep of `-ub` × cache type at the max depth, MoE: lowest `--n-cpu-moe` that fits, confirmed in a real server | Steps 3–6 in full: backend comparison, all sweeps at 3–4 depths, drift checks, speculative decoding, replay |
| Agent tasks | 2 short tasks (t1, t4) per agent model, tuned config only: catches models that can't use tools | All 6 tasks, old vs tuned config, for every agent model |
| Before/after | Current config saved, backed up and measured in a real server first (step 2b); the report opens with the before/after table | Same, plus all 6 agent tasks on the old config |
| Typical time | **25–45 min per model**; 50–100 min for 2–3 models | **1.5–3 h per model**; 5–10 h for 3–4 models: run it overnight |

Work out the estimate from this machine's inventory with these per-step costs (measured on an 8 GB GPU; faster GPUs are quicker, MoE models with CPU experts are the slowest):
- Building a backend: 10–30 min each (HIP/CUDA are the slowest).
- `micro.py` at 4 depths: 3–5 min per config for a dense ≤8B model, 10–15 min for a large MoE.
- `validate_fill.py` up to full context: 3–6 min per config.
- `bench_agentic.py`, all 6 tasks: 20–35 min per config for a capable model (set `TASK_TIMEOUT`; a weak model can loop until it). Tasks t1 + t4 only: 3–8 min.
- `bench_replay.py`: ~15 min per config. `bench_general.py`: 2–5 min per config.
- Baseline (step 2b, both modes): 3–6 min per model (one `validate_fill.py` run); comprehensive mode adds the old config's agent tasks.

Say it plainly, e.g. *"Quick tune of 3 models: about 1 hour 30 minutes, GPU busy throughout. Comprehensive: about 7 hours, best run overnight."* Report progress against the estimate as you go. A quick tune can be upgraded later: comprehensive mode reuses its results.

**Autonomy.** Once the user agrees to the mode, run the whole thing without stopping for routine choices. Still stop for the safety rules above (sudo, builds, live services, the user's configs), and at the end present the prepared config and results for approval.

## Workflow
Steps 1, 2, 2b, 7 and 8 apply to both modes. Quick mode does the reduced versions of steps 3–6 described in the table; comprehensive mode does them all.

### 1. Inventory
- Confirm you aren't running on one of the models being tuned (Safety rules). If you are, stop here.
- Run `scripts/detect_hw.sh [model dirs]`. Note the GPU, VRAM, physical cores, RAM (size, speed, module count), the PCIe link, available llama.cpp builds and their devices, GGUF files (flagged if too big for VRAM + RAM), and Ollama models.
- Tell the user about anything that caps performance before tuning starts: one RAM module or RAM below its rated speed, a narrow PCIe link, a power-saver profile, VRAM already used by other apps. These can matter more than any flag.
- Ask the user, if not already clear:
  - their main workload (which coding agent? chat?)
  - which models matter
  - the context size their agent needs
  - whether a live llama-server/router or Ollama is running
- Create the config with `scripts/init_config.py --out <working folder>/config.json [model dirs]`. It finds the llama.cpp builds, their GPU backends (with the AMD override where needed) and the GGUF models with their mmproj/draft files. Then fix what it can't know: each model's `ctx` (the agent's real need), `router_url` if they run a llama-server router, `agent.self_model` (see Safety rules) and `agent.python_bin_dir` (a venv with `pytest` for the agentic checks). Set `TUNE_CONFIG` to point at it. Results go to `results_dir`. (`config.example.json` documents every field.)

### 2. Agent profile
Work out the agent's context range: base prompt, compaction trigger, what's kept (see methodology.md; for OpenCode with 64K, ~12K → ~49.5K). Pick 3–4 depths inside that range. **All later measurements use these depths, not depth 0 alone.**

### 2b. Baseline: save and measure what the user runs today (both modes, never skip)
The report's "before" column comes from here, measured the same way as "after". Do it before changing anything.
- **Save the current settings:** `scripts/quick_check.py <--router url | --preset file | --ollama> --json > <results_dir>/baseline-settings.json`. Each model's `server_args` are its current llama-server flags.
- **Back up every file you might change** (router presets, agent config such as `opencode.json`, service files) into `<results_dir>/backup/`, keeping their names. Note the original paths in the report for undo.
- **Measure each model's current config** in a real server at the agent's depths, label `before`:
  `scripts/validate_fill.py <model> <backend> before <depths> <server_args from baseline-settings.json>`.
  Use the backend that matches its current device (a combined build accepts `--device`). For Ollama users, measure their Ollama model with the curl method in backends.md at the same depths and record it.
- If a model has no settings yet (new model, nothing configured), the baseline is llama-server's defaults: `-c <ctx>` only. Say so in the report.
- If the current config doesn't load or doesn't fit, record that: it's a finding, and "before" is "fails at full context".
- The chosen config gets the label `after` everywhere (`validate_fill.py`, `bench_agentic.py`), so `scripts/before_after.py <model>` can build the table.

### 3. Backend check
- If more than one GPU backend is possible, build them **from the same commit** (backends.md) after asking. A combined build (e.g. `-DGGML_HIP=ON -DGGML_VULKAN=ON`) lets one router serve different models on different backends.
- For each model: `scripts/micro.py <backend> C1-<model> <model> -ngl 99 [-ncmoe N] -fa 1 -p 4096 -n 128 -d <depths> -r 1`.
- Decide **per model**. MoE models with CPU experts and dense models can prefer different backends.

### 4. Parameter sweeps (micro.py, then confirm)
For each model on its chosen backend, at a mid and a max depth:
1. `-ub 512,1024,2048` (`-b 2048`). Prompt speed vs VRAM.
2. `-ctk f16,q8_0 -ctv f16,q8_0`. Generation speed vs VRAM.
3. MoE only: `--n-cpu-moe` from tight to loose, **together with** the best `-ub` and cache type, at the max depth.
4. MoE / CPU-heavy: `-t` = 4, 6, physical cores, logical cores.
5. Note any out-of-memory failures; they mark the boundary.
6. End each sweep by rerunning its first config (drift check, methodology.md).

Then `scripts/table.py <label-prefix>` to compare, and confirm the top 2–3 configs with `scripts/validate_fill.py <model> <backend> <label> <depths> <server args>` at the full `-c`. Pick the fastest that fits with **a few hundred MB of VRAM headroom**.

### 5. Speculative decoding
- Check GGUF metadata for MTP/nextn tensors or a matching assistant/draft model (flags.md).
- Try what's available: `draft-mtp` with an assistant model; `ngram-mod` for every model.
- Judge with `scripts/bench_agentic.py` and `scripts/bench_general.py`, not micro-benchmarks. Keep it only if wall time drops and the pass rate holds.

### 6. End-to-end check
- Run at least the first two tasks for **every** model the user wants to use with their agent before spending long on tuning it: small or older models (e.g. Qwen2.5-Coder-3B) may never make a tool call, and no flag fixes that.
- `scripts/bench_agentic.py <model> <backend> before 1 <server_args from the baseline>` (comprehensive mode) and `... after 1 <new flags>`, with `RECORD=1` once per model to capture prompts. Raise `TASK_TIMEOUT` (seconds) for slow models.
- `scripts/bench_replay.py` to compare finalists on identical recorded traffic and see prompt-cache reuse.
- `scripts/bench_general.py` for chat use (set `IMAGE=` for vision models).
- Compare pass rate, wall time, pp and tg. A config that's faster but fails more tasks is not a win.

### 7. Apply (with approval)
- Write the router presets (flags.md has a template: per-model `device =`, no `:` in names). Show the service or startup command, and the agent config entries (context limits must match `c`).
- Prepare the files, show the diff, and let the user install anything that needs sudo.
- Restart, then verify every model answers through the user's real endpoint at the expected speed, at a long prompt, with mmproj loaded for vision models (send a real image).
- **Ollama users:** run `scripts/to_ollama.py --preset <presets.ini> --model <name>` (or `--args "..."`) on the winning config. Show the Modelfile and environment, and say plainly what didn't carry over (it lists it: expert-only MoE offload, per-model backend, separate `-ub`, speculative decoding). Apply with approval (`ollama create`, the service environment needs sudo on Linux), then measure with `eval_count / eval_duration` (backends.md) and compare with the llama-server numbers. If Ollama is much slower, say so and offer `llama-server` instead.

### 8. Document
Write a results file for the user, modelled on `references/example-rx6600xt.md`. **Start it with the before/after table for each model**, generated by `scripts/before_after.py <model>` from the measurements (don't retype numbers). Then:
- the hardware
- the agent profile
- each experiment, with what was tested, a table and the verdict
- the final config, before vs after
- what changed where, and how to undo it: the backups in `<results_dir>/backup/` and the command to put each one back

Then offer, once, to share the result with other users: fill in the fields of the repo's "Share your results" issue form (`.github/ISSUE_TEMPLATE/share-results.yml`: GPU/VRAM, CPU/RAM/OS, llama.cpp version and backend, model and quant, agent and context, flags before/after, pp/tg at depth, tasks passed) and give the user the text and the link `https://github.com/cliffkoome/llamacpp-speedtune/issues/new?template=share-results.yml`. Never post it yourself without a clear yes, and leave out anything personal (usernames, paths under home, hostnames, keys).

## Decision rules of thumb (confirm by measuring)
- Agents re-read long prompts every turn, so **prompt speed at depth matters as much as generation speed**.
- Bigger `-ub` mostly helps MoE models with CPU experts; dense models fully on the GPU barely change.
- f16 cache beats q8_0 for speed when it fits; q8_0 is the floor for agents.
- Fewer CPU-offloaded MoE layers is faster only until VRAM runs out; then everything collapses.
- If the GPU drives the display, keep headroom for it.
