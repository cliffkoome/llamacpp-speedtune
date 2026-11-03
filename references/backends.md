# Backends: building llama.cpp for a fair comparison

Build every backend from **the same commit** into separate build dirs. If a distro package exists, check out its commit (`llama-server --version`, or the package version) so packaged and self-built binaries are comparable.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp && cd ~/src/llama.cpp
git checkout <commit>
COMMON="-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON"
T="--target llama-bench llama-server -j$(nproc)"
```

| Backend | Hardware | Build | Run-time notes |
|---|---|---|---|
| CUDA | NVIDIA | `cmake -S . -B build-cuda $COMMON -DGGML_CUDA=ON && cmake --build build-cuda $T` | `-DCMAKE_CUDA_ARCHITECTURES=native` builds for the installed card only |
| HIP (ROCm) | AMD | `HIPCXX="$(hipconfig -l)/clang" HIP_PATH=/opt/rocm cmake -S . -B build-hip $COMMON -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx>` | Unsupported consumer cards need `HSA_OVERRIDE_GFX_VERSION` (table below). ROCm can't spill to RAM |
| Vulkan | AMD, NVIDIA, Intel | `cmake -S . -B build-vk $COMMON -DGGML_VULKAN=ON` (needs `glslc` and Vulkan headers) | Spills to shared memory when VRAM runs out (very slow); watch for sudden collapses |
| Metal | Apple Silicon | `cmake -S . -B build-metal $COMMON` (Metal is on by default on macOS) | Unified memory; the "VRAM" limit is the GPU working-set limit (`sudo sysctl iogpu.wired_limit_mb=...` raises it) |
| CPU | any | `cmake -S . -B build-cpu $COMMON` | Baseline only |
| HIP + Vulkan | AMD | `... -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx> -DGGML_VULKAN=ON` | One binary with devices `ROCm0` and `Vulkan0`; pick one per model with `-dev` / preset `device =` |
| CUDA + Vulkan | NVIDIA | `... -DGGML_CUDA=ON -DGGML_VULKAN=ON` | Same idea |

Check each build: `./build-x/bin/llama-server --list-devices`.

## AMD consumer GPUs and `HSA_OVERRIDE_GFX_VERSION`
ROCm officially supports only some chips. Consumer cards usually run by pretending to be the supported chip of their family:

| Family | Cards (examples) | gfx | Build target | Override |
|---|---|---|---|---|
| RDNA2 | RX 6800/6900 | gfx1030 | gfx1030 | none |
| RDNA2 | RX 6700 XT, 6750 XT | gfx1031 | gfx1030 | `10.3.0` |
| RDNA2 | RX 6600, 6600 XT, 6650 XT | gfx1032 | gfx1030 | `10.3.0` |
| RDNA3 | RX 7900 XT/XTX | gfx1100 | gfx1100 | none |
| RDNA3 | RX 7800 XT, 7700 XT | gfx1101 | gfx1100 | `11.0.0` |
| RDNA3 | RX 7600 | gfx1102 | gfx1100 | `11.0.0` |

`rocminfo | grep gfx` shows the real chip.

## What we saw on an RX 6600 XT (8 GB)
- HIP beat Vulkan by 47–55% pp and 22–35% tg on **Qwen3.6-35B-A3B (MoE, experts partly on CPU)**.
- Vulkan beat HIP by 5–28% tg on **dense 3B/7B** models, while HIP read prompts about 20% faster.
- **Gemma-4 12B** only fit on Vulkan, because HIP needed 1.1 GB more compute buffer.

So decide per model, and serve mixed backends from one combined build. See `example-rx6600xt.md`.

## Ollama
Ollama bundles its own llama.cpp-based runners (CUDA/ROCm/Vulkan), and it uses ROCm on AMD when available. It's fine for projects built around its API, but its settings are limited: one batch size, whole-layer GPU offload only (no `--n-cpu-moe`), no per-model backend, and the KV cache type and flash attention are set for the whole service.

So tune with `llama-server` (Ollama's blobs are plain GGUF files, see gotchas.md), then carry the winner over:

```bash
scripts/to_ollama.py --preset presets.ini --model <name>     # or --args "<llama-server flags>"
```

It prints a Modelfile (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`), the service environment (`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`) and what didn't carry over. For MoE models with CPU experts, `num_gpu` is only an approximation of `--n-cpu-moe`: try a few values.

Measure Ollama with `curl localhost:11434/api/generate -d '{"model":"...","prompt":"...","stream":false}'` and compute `eval_count / eval_duration * 1e9` (generation t/s) and `prompt_eval_count / prompt_eval_duration * 1e9` (prompt t/s). Use a long prompt near your agent's real depth, not a one-liner.
