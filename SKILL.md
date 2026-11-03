---
name: local-llm-speedtune
description: Make the user's local LLMs run as fast as their hardware allows, measured the way their coding agent uses them. Picks the best engine (llama.cpp by default; challengers such as ExLlama, MLX, ik_llama.cpp or OpenVINO only where the hardware suits them), then the best backend (CUDA, HIP/ROCm, Vulkan, Metal), then the best flags (batch sizes, KV cache, MoE expert offload, threads, speculative decoding), at the context depths agents really reach, confirmed in a real server and on real agent tasks. Asks all its questions up front, then runs unattended with a finish time, resumes if interrupted, and saves each model's best config as a profile behind one endpoint. Also has a 2-minute no-benchmark check (llama-server or Ollama) and Ollama export. Use when the user wants to speed up, benchmark, tune or optimise local models, pick an engine or backend, check their llama-server or Ollama settings, or set up one endpoint for several local models.
---

# local-llm-speedtune

You're tuning the user's local models on their own hardware: **engine first, then backend, then flags**. The goal is the fastest **working** config for each model under the user's real workload, usually a coding agent. Every claim must come from a measurement on this machine. Defaults, online advice and other people's numbers are starting points only.

Read before starting:
- `references/methodology.md`: why to measure at depth, agent compaction maths, the test ladder.
- `references/gotchas.md`: pitfalls that silently corrupt results. Read it fully before running anything.
- `references/intake.md`: the questions, the plan message and the closing message, word for word.
- Then, as needed: `references/engines.md` (challenger engines), `references/flags.md` (what to sweep), `references/backends.md` (build recipes, Ollama), `references/example-rx6600xt.md` (a complete worked example).

## Safety rules
- **Permissions are collected once, up front** (intake question 7), because the user won't be there to answer later. Within them, never stop to ask; outside them, skip the step (`run_state.py add --needs ...` records it) and list it in the report. **Never use sudo, and never change the user's live configs (server presets, agent config, services) while they're away**: prepare the change and apply it only when they're back and say yes.
- **Never unload the model you're running on.** Find out which model *you* use. If it's served by the user's local server or GPU, a tune would stop you mid-run. Stop and tell the user: run the tune from an agent on a cloud model, or run the scripts themselves. The 2-minute check is fine either way. Record your model as `agent.self_model` in the config; `free_router()` refuses to unload it.
- **Never kill processes by pattern** (`pkill -f`, `killall`): the pattern can match your own shell. Kill by PID.
- **One test server at a time**, on a spare port (config `port`, default 8090). The user's live server is left alone, except that its models are unloaded (`router_url`) or it's stopped, if they allowed it.
- **Don't trust a run you haven't checked.** After the first agentic task, inspect the run folder: did the agent edit the right files? Did the check really fail or pass?
- **Downloads and installs** go into their own folder or virtual environment, within the size the user allowed, from the model's official repo or a well-known quantizer.

## The 2-minute check
If the user only wants to know whether tuning is worth it, or before a tune, run the check. No config, no benchmarks, nothing changed:
- `scripts/quick_check.py --router <url>` (API key from `$LLAMA_API_KEY` or `--key-env`), `--preset <presets.ini>`, `--args "<llama-server command line>"`, or `--ollama [url]`. Add `--json` to parse it.
- Report the findings in plain words, biggest first, and say whether a tune looks worth it.

## The flow
**Inventory → one round of questions → plan with a finish time → the user walks away → unattended run in phases → results and profiles → apply when they're back → closing message.**

### 1. Inventory (~2 min, read-only)
- Confirm you aren't running on one of the models being tuned (Safety rules). If you are, stop here.
- `scripts/detect_hw.sh [model dirs]`: GPU, VRAM, cores, RAM size/speed/channels, PCIe link, WSL2, llama.cpp builds and devices, GGUF files (and whether they fit), Ollama models, other engines already installed.
- Read the user's live setup: `quick_check.py` on their router, presets file or Ollama.
- `scripts/init_config.py --out <results_dir>/config.json [model dirs]` drafts the config; set `TUNE_CONFIG` to it.
- Work out the agent's context range (methodology.md; OpenCode at 64K lives between ~12K and ~49.5K tokens) and pick 3–4 depths in it. **All measurements use these depths.**

### 2. Questions (once, one message)
Ask the questions in `references/intake.md` with what you found filled in: use, models (confirm the list; up to 3 suggestions that suit the hardware, downloaded only if picked), context, users (warn in numbers if several users won't fit well), priority, mode and time away, **permissions**, where to save, and whether they must stay on their current engine. Wait for the answers. Save them as `answers.json`, then `scripts/run_state.py init --answers answers.json`, and fix up the config (models, `ctx`, `router_url`, `agent.self_model`, `agent.python_bin_dir`, `profiles_dir`).

### 3. Plan and walk away
- **Research candidates** for this exact hardware (engines.md): current engine and backend advice, known bugs, newer forks. Write each claim down with source, date and the hardware it was measured on. These are things to try, not results.
- Add every step to the run state with an estimate: `scripts/run_state.py add --phase <phase> --model <m> --est <min> [--needs <permission>] --desc "..." -- <command>`. Per-step costs (8 GB GPU; faster GPUs are quicker, MoE models with CPU experts the slowest):
  - building a llama.cpp backend: 10–30 min (HIP/CUDA slowest); installing an engine: 5–20 min; downloads: size ÷ the connection's speed
  - baseline (`validate_fill.py` or `bench_openai.py`): 3–6 min per model
  - `micro.py` at 4 depths: 3–5 min per config (dense ≤8B), 10–15 min (large MoE)
  - `validate_fill.py` / `bench_openai.py` to full context: 3–6 min per config
  - `bench_agentic.py`: all 6 tasks 20–35 min per config; t1 + t4 only 3–8 min
  - `bench_replay.py` ~15 min, `bench_general.py` 2–5 min per config
- If the total doesn't fit the time they can be away, propose what to cut before starting.
- Send the plan message from intake.md: per-model times, **finish clock time** (`run_state.py status`), what's busy, what could make it longer, how to stop it, what was skipped for lack of permission. Then they can leave.

| | **Quick tune** (default) | **Comprehensive tune** |
|---|---|---|
| Engines | llama.cpp; a challenger only if it **and** its model file are already installed | Full challenger round within the permissions (installs, downloads) |
| Backends | The llama.cpp builds already installed | Builds every possible backend from the same commit (if allowed) |
| Flags | One sweep of `-ub` × cache type at the max depth; MoE: lowest `--n-cpu-moe` that fits; confirmed in a real server | All sweeps at 3–4 depths, drift checks, speculative decoding, replay |
| Agent tasks | t1 + t4 (`TASKS=t1_fix_test,t4_stacktrace`) per agent model, on the chosen config | All 6 tasks, before and after, for every agent model |
| Before/after | Always (phase 4a) | Always, plus the old config's agent tasks |
| Typical time | **25–45 min per model** | **1.5–3 h per model**: run it overnight |

### 4. The unattended run, in phases
Run each phase as **one background command**, `scripts/run_phase.py <phase> --notify`, and wait for it to exit (don't poll: that costs the user money and keeps the session busy). When it exits, read the results, decide the next phase, add its steps, and start it. Everything is in `run-state.json`: if the session or machine restarts, `run_state.py status` shows where it stopped and `run_phase.py` resumes; finished steps are never rerun. With `prevent_sleep` allowed, run_phase.py keeps the computer awake.

**4a. Baseline: what the user runs today (never skip).** Do it before changing anything; it's the report's "before".
- `quick_check.py <source> --json > <results_dir>/baseline-settings.json` (each model's `server_args` are its current flags).
- Back up every file that may change (presets, agent config, service files) into `<results_dir>/backup/`, noting their original paths.
- Measure each model's current config at the agent's depths, label `before`: `validate_fill.py <model> <backend> before <depths> <server_args>`. Ollama or another engine: `bench_openai.py --url <its endpoint> --served-name <id> <model> <name> before <depths>`.
- No settings yet → the baseline is llama-server's defaults (`-c <ctx>`); a config that doesn't load → "fails at full context". Both are findings.

**4b. Engine.** Only for models where a challenger fits the hardware (engines.md) and the user is open to switching (or wants to know what switching would gain).
- Check the hard requirements first (OpenAI API, tool calls, context, install without sudo); skip with the reason otherwise.
- Use a model file of the same or higher bits per weight; give the challenger its recommended settings plus a sweep of its 1–2 key settings.
- Measure **every** contender with `bench_openai.py` (llama.cpp too, with its best config so far), then `TASKS=t1_fix_test,t4_stacktrace bench_agentic.py <model> <engine> <label> 1 [flags]`.
- A challenger **wins only if** it's ≥10% faster on prompt or generation at the agent's depths, not slower on the other, passes the agent tasks and fits at full context. Otherwise llama.cpp stays; report the margin either way.

**4c. Backend (llama.cpp).** Build the possible backends from the same commit if allowed (backends.md); a combined build (`-DGGML_HIP=ON -DGGML_VULKAN=ON`) lets one server put different models on different backends. For each model: `micro.py <backend> C1-<model> <model> -ngl 99 [-ncmoe N] -fa 1 -p 4096 -n 128 -d <depths> -r 1`. Decide **per model**.

**4d. Flags** (on the chosen engine and backend; flags.md), at a mid and the max depth:
1. `-ub 512,1024,2048` (`-b 2048`): prompt speed vs VRAM.
2. `-ctk f16,q8_0 -ctv f16,q8_0`: generation speed vs VRAM.
3. MoE only: `--n-cpu-moe` from tight to loose, **with** the best `-ub` and cache type, at the max depth.
4. MoE / CPU-heavy: `-t` = 4, 6, physical cores, logical cores.
5. Out-of-memory failures mark the boundary; end each sweep by rerunning its first config (drift check).
Then `table.py <prefix>`, and confirm the top 2–3 with `validate_fill.py` at the full `-c`. Pick the fastest that fits with a few hundred MB of VRAM headroom. Measure the winner with label `after`.

**4e. Speculative decoding** (comprehensive): MTP/assistant drafts and `ngram-mod` (flags.md), judged with `bench_agentic.py` and `bench_general.py`. Keep only if wall time drops and the pass rate holds.

**4f. End-to-end.** Run t1 + t4 for **every** agent model early: small or older models may never make a tool call, and no flag fixes that. `bench_agentic.py <model> <name> after 1 <flags>` (comprehensive: also `before`, all 6 tasks, `RECORD=1` once for `bench_replay.py`). A config that's faster but fails more tasks is not a win.

### 5. Results and profiles (still unattended)
- `scripts/profiles.py save <model> <backend-or-engine>` for every tuned model: engine, backend, flags, model file, measured speeds before and after, agent tasks, engine version, hardware, date. The previous profile moves to `history/`.
- `scripts/profiles.py serve-config --out <results_dir>/serve`: router presets when everything is on one llama.cpp build, otherwise a llama-swap config. Either way the agent talks to **one address**, and switching models switches the whole config.
- Users who stay on Ollama: `to_ollama.py` on the winner, plus the measured cost of staying (`bench_openai.py` against Ollama vs the llama.cpp winner).
- The report (`<results_dir>/report.md`), modelled on `references/example-rx6600xt.md`, **opens with `before_after.py <model>` for each model**, then: hardware, agent profile, research claims vs what was measured, each experiment with its verdict, what was skipped and why, the prepared changes (diffs), and how to undo each one from `backup/`.
- `run_state.py notify "Tuning finished: <one-line result>"`, or your own notification tool if you have one.

### 6. Apply (when the user is back, with their yes)
Show the before/after and the prepared changes. On a yes: install the serving config, update the agent config (context limits must match), restart, and verify every model answers through the real endpoint at the expected speed with a long prompt (send a real image to vision models). Anything needing sudo, the user runs.

### 7. Closing message
Once, after the results: the neutral invitation in `references/intake.md` (GitHub results form, which you fill in for them to check; a DEV comment in their own words; a ⭐ if it helped). Never post, comment, react or star anything yourself, and don't make it depend on the results being good.

## Decision rules of thumb (confirm by measuring)
- Agents re-read long prompts every turn, so **prompt speed at depth matters as much as generation speed**.
- Bigger `-ub` mostly helps MoE models with CPU experts; dense models fully on the GPU barely change.
- f16 cache beats q8_0 for speed when it fits; q8_0 is the floor for agents.
- Fewer CPU-offloaded MoE layers is faster only until VRAM runs out; then everything collapses.
- If the GPU drives the display, keep headroom for it.
- Wrappers (Ollama, LM Studio) can't beat a tuned llama-server running the same code; only real challengers (engines.md) can.
