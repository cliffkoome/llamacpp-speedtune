# Engines: when to try something other than llama.cpp

llama.cpp is the default: it runs on every GPU and CPU, splits a model between GPU and RAM, and reads GGUF, the most common local format. A **challenger** engine is tried only when the hardware gives it a real chance to win, and it has to win by measurement (rules below).

This file is the fallback list. Before the run, **research what's current** for the user's exact hardware (engine releases, new forks, known bugs) and write each claim down with its source, date and the hardware it was measured on. Research proposes candidates; only measurements on this machine decide. Online advice is often for other hardware, out of date, or measured on an empty prompt: "Vulkan is best on AMD" was wrong for an MoE model on an RX 6600 XT (HIP was 47–55% faster at reading).

## Which challengers to try

| Hardware / situation | Challenger | Why it can win | Why it may not |
|---|---|---|---|
| NVIDIA, model **fits in VRAM**, one user | **ExLlamaV2/V3** (served by TabbyAPI) | NVIDIA-tuned kernels, EXL2/EXL3 quants | Can't offload to RAM: a model that doesn't fit doesn't run |
| MoE model with experts in RAM, any GPU | **ik_llama.cpp** (llama.cpp fork) | Faster CPU and hybrid GPU/CPU paths for MoE | Fewer backends than llama.cpp (check yours is supported); some newer models arrive later |
| Apple Silicon | **MLX** (`mlx-lm`) | Apple's own framework; often faster prompt processing | Different model files; tool-call support depends on the model and version |
| Intel CPU, Arc GPU or NPU | **OpenVINO** (OpenVINO Model Server) | Intel's own kernels | Model conversion; fewer ready-made models |
| NVIDIA/AMD datacenter cards, or **several users at once** | **vLLM**, **SGLang** | Batching many requests | Usually slower for one user; needs the model in VRAM; AWQ/GPTQ/FP8 files |

Not challengers: **Ollama, LM Studio, KoboldCpp, Jan** run llama.cpp's own code (Ollama partly its own engine on the same ggml library). At best they match a tuned llama-server; usually they lose to their defaults. Detect them, measure them as the baseline when the user uses them, and use `to_ollama.py` if the user must stay on Ollama.

## Requirements before a challenger is even measured
1. An **OpenAI-compatible API** (`/v1/chat/completions`, ideally `/v1/completions`) the user's agent can use.
2. **Working tool calls** with this model (chat template, tool-call parser). Checked by the agent tasks.
3. The **context the agent needs**, and image input for vision models.
4. Installable **without sudo**, in its own folder or virtual environment, within the user's permissions (`install_engines`, `download_gb`).

An engine that fails one of these is skipped and named in the report with the reason.

## Fair comparison
- **Same size class of model file.** Pick the challenger's quant at the same or higher bits per weight as the GGUF (Q4_K_M ≈ 4.8 bpw; EXL2/EXL3 4.0–5.0 bpw; MLX 4-bit ≈ 4.5 bpw with group scales). The report states each file's bpw.
- **Same measuring method for every contender**: `scripts/bench_openai.py`, llama.cpp included (client-side timing; it matched llama-server's own numbers within 4–8% in our tests).
- **Tune the challenger a little**: its documented recommended settings plus a sweep of its 1–2 key settings (e.g. ExLlama cache type and max sequence length; MLX prefill step size).
- **It wins only if** it is **≥10% faster on prompt or generation at the agent's depths, not slower on the other, passes the same agent tasks, and fits at full context.** Otherwise llama.cpp stays, and the report says by how much the challenger lost or how little it won.

## Config entries
Add each installed challenger under `engines` in the config, and the model file for it under the model:

```json
"engines": {
  "exllama": {
    "start": "~/engines/tabbyAPI/start.sh --host 127.0.0.1 --port {port} --model-name {model} --max-seq-len {ctx}",
    "ready_path": "/v1/models"
  },
  "mlx": {
    "start": "~/engines/mlx/bin/mlx_lm.server --host 127.0.0.1 --port {port} --model {model}",
    "served_name": "{model}"
  },
  "ik": {
    "start": "~/src/ik_llama.cpp/build/bin/llama-server -m {model} --host 127.0.0.1 --port {port} -c {ctx} --alias {alias} {args}"
  }
},
"models": {
  "qwen7b": {"alias": "qwen2.5-7b-instruct", "path": "~/models/qwen2.5-7b-instruct-Q4_K_M.gguf", "ctx": 32768,
             "engines": {"exllama": "Qwen2.5-7B-Instruct-exl2-4.65bpw", "mlx": "~/models/Qwen2.5-7B-Instruct-4bit-mlx"}}
}
```
Placeholders: `{model}` (the model file or name for this engine), `{port}`, `{ctx}`, `{alias}`, `{args}` (extra flags; appended when missing). These start commands are **examples to check against each engine's current docs**: flags change between releases.

Then measure exactly like llama.cpp backends: `bench_openai.py <model> <engine> <label> <depths> [engine flags]`, and `bench_agentic.py <model> <engine> <label> 1 [engine flags]`.

## Installing (with the user's permission only)
- Everything goes into its own folder (`~/engines/<name>/`) or virtual environment; nothing system-wide, no sudo.
- Note the version installed, and the download size of each model file before downloading (counts against `download_gb`).
- Model files come from the model's official repo or a well-known quantizer. Name the source in the report.

## Status
Measured end to end so far: llama.cpp (HIP, Vulkan, CPU) on Linux/AMD. The challenger paths above follow each project's documentation and **haven't been run end to end yet**: expect rough edges, and record what you hit.
