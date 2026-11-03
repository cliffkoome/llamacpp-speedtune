# Methodology

## Measure where the agent actually works

A coding agent doesn't run on an empty context. Every turn resends the system prompt, the tool definitions and the whole conversation, so the model spends most of its time **deep in the context**. Speeds measured at depth 0, the default for `llama-bench`, flatter every config and can rank them in the wrong order.

So first work out the agent's **context range**, then measure at points inside it.

### OpenCode (v2.x)
From its source (`session-compaction`):
- Usable input before auto-compaction is `ctx − max(floor(0.15 × ctx), ctx ≥ 32000 ? 16000 : 0)`.
  - 64K context: 65,536 − 16,000 = **49,536 tokens**.
  - 32K context: 32,768 − 16,000 = 16,768 tokens.
  - 128K context: 131,072 − 19,660 = 111,412 tokens.
- After compaction it keeps the most recent `compaction.keep.tokens` (default **15,000**) and adds a summary.
- The system prompt plus tool definitions are about **12K tokens** before the user says anything.
- Configurable in `opencode.json`: `compaction.auto`, `compaction.buffer` (alias `reserved`), `compaction.keep.tokens` (alias `preserve_recent_tokens`).

So with a 64K context a session lives between **~12K and ~49.5K tokens**, and each compaction re-reads up to ~49K tokens.

For another agent, find the same three numbers: base prompt size, compaction trigger, and what's kept after. Read its docs or source, or record real prompts with `llama-server --log-prompts-dir` and look at their sizes.

### Depth points
- `-d 0`: sanity check, comparable with published numbers.
- `-d` ≈ 1/3 and 2/3 of the agent's range: typical turns.
- `-d` ≈ the compaction trigger: the worst case, and where VRAM limits show up.

## What to measure
| Metric | Meaning | Why it matters for agents |
|---|---|---|
| pp (prompt processing, t/s) | Reading input | Every turn and every compaction re-reads a long prompt; this decides time to first token |
| tg (token generation, t/s) | Writing output | Code, tool-call arguments, explanations |
| VRAM at max context | Does it fit | Out of VRAM means a crash (ROCm) or a 10–30× slowdown (Vulkan/CUDA spilling to shared memory) |
| Pass rate on real tasks | Does it still work | Cache quantization, speculative decoding and templates can break tool calls while looking fast |
| Wall time per task | What the user feels | Combines everything, including how many turns the model needs |

## Test ladder
Cheap and many first, expensive and few last:

1. **`micro.py` (llama-bench)** for broad sweeps. Seconds to minutes per config. Use `-r 1` for depth runs (each rep re-fills the depth), `-r 2–3` for short ones. Single-rep numbers are noisy, around ±5%; re-run anything that looks surprising.
2. **`validate_fill.py` (real server, full `-c`)** for the top 2–3 configs. llama-bench allocates only prompt + depth, while the server allocates the whole context and its own buffers, so fit and speed can differ. In our runs llama-bench said `--n-cpu-moe 33` didn't fit but the server ran it fine, and the reverse happens too.
3. **`bench_agentic.py` (real agent, real tasks)** for the finalists, baseline vs tuned. This catches anything that broke tool calling, and shows real wall time.
4. **`bench_replay.py`** to compare configs on identical recorded agent traffic, including prompt-cache reuse between turns.
5. **`bench_general.py`** for non-coding use: chat, summaries, JSON, images.

## Fair backend comparisons
- Build every backend **from the same llama.cpp commit** with the same CPU flags (`-DGGML_NATIVE=ON`). With MoE models running experts on the CPU, the CPU build matters as much as the GPU backend.
- Confirm a packaged build against a self-built one before blaming the backend (our AUR Vulkan build matched a same-source build within 1%).
- Decide **per model**. In our case one backend won by 50% on an MoE model and lost by 28% on a dense one.

## Hygiene
- One test server at a time. VRAM holds one model, and a second server on the same port silently answers for the first. `server.py` refuses to start if the port is taken.
- Unload the user's live server first (`router_url` in the config), and check that VRAM is near zero. Never when the agent doing the tuning runs on that server (`agent.self_model`, see SKILL.md Safety rules).
- Keep the CPU otherwise idle during runs. A compile in the background skews MoE results.
- **Drift check:** end every sweep by rerunning its first config. More than ~5% difference means heat, power limits or background load changed the results; fix that and re-measure.
- Other GPU users (browser, games, a compositor on a second monitor) take VRAM. Note `detect_hw.sh`'s VRAM "used" figure before starting, and keep it the same for the final check.
- Write down every config and result as you go (`results/*.csv`); summarise at the end.
