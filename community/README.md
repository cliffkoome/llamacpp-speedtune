# Community results

Tuned configs from real machines. Find hardware like yours and start from its flags, then confirm on your machine with a quick tune: drivers, RAM speed and the PCIe link change the numbers.

**Add yours:** [open a "Share your results" issue](https://github.com/cliffkoome/llamacpp-speedtune/issues/new?template=share-results.yml). If an agent ran the tune, ask it to fill the form in from your results file. Rows are added from those issues.

Speeds are tokens/s, before → after, measured at the depth shown.

| GPU (VRAM) | CPU / RAM | Backend | Model | Agent, context | Key flags (after) | Prompt (pp) | Generation (tg) | Tasks | Details |
|---|---|---|---|---|---|---|---|---|---|
| RX 6600 XT (8 GB) | Ryzen 7 5700X / 32 GB DDR4-3200 | HIP (ROCm 7.2), llama.cpp b11935 | Qwen3.6-35B-A3B UD-Q4_K_M | OpenCode, 64K | `-ngl 99 --n-cpu-moe 34 -ub 2048 -b 2048 -ctk f16 -ctv f16 -t 8` | 172 → **526** @ 49K | 19.2 → **25.3** @ 49K | 5/6 → 6/6 | [example](../references/example-rx6600xt.md) |
| RX 6600 XT (8 GB) | Ryzen 7 5700X / 32 GB DDR4-3200 | Vulkan (Mesa 26.2), llama.cpp b11935 | Gemma-4 12B Q4_K_M + vision | chat, 32K | `-ngl 99 -ub 1024 -ctk q8_0 -ctv q8_0 --no-mmproj-offload` | 121 → **212** @ 28K | **24.3** @ 28K | – | [example](../references/example-rx6600xt.md) |
