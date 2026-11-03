# Prompts

Copy one into your coding agent (Claude Code, OpenCode, Codex, …) with this skill installed. Replace the parts in `<angle brackets>`. Without skill support, start with: *"Read SKILL.md in <path to llamacpp-speedtune> and follow it."*

## Install and run everything (one prompt)
```text
Install the agent skill from https://github.com/cliffkoome/llamacpp-speedtune (clone it into your skills folder, or into
~/llamacpp-speedtune and read its SKILL.md if you don't support skills). Then use it to tune my local models.
Start with the hardware inventory, then tell me what you found, the quick and comprehensive options,
and how long each would take on my machine. After I pick one, run it autonomously to the end.
Only stop to ask before using sudo, building software, or changing my live server or agent config.
```

## 2-minute check (no benchmarks, changes nothing)
```text
Use llamacpp-speedtune to do the 2-minute check of my current setup: <my llama-server router at http://127.0.0.1:8080 |
my presets file ~/models/presets.ini | my Ollama models>. Tell me what's likely slowing my models down and whether a quick tune is worth it.
```

## Quick tune (about an hour)
```text
Use llamacpp-speedtune to do a quick tune of my local models for <OpenCode>. Tell me the time estimate first.
```

## Comprehensive tune (several hours; best overnight)
```text
Use the llamacpp-speedtune skill to do a comprehensive tune of my local models for <OpenCode / Claude Code / my agent> on this machine.
- Models: <all GGUFs in ~/models | just these: ...>
- Main use: <agentic coding at 64K context>. Secondary: <chat, long documents, vision>.
- Priority: <fastest at full context | most context that still fits | balance>.
- My live server: <llama-server router on port 8080 | Ollama | nothing running>.
Ask me before using sudo, restarting services or changing my agent config. Report the results as a table with before vs after, and write them to <~/Documents/llm-tuning.md>.
```

## Quick re-check (after a llama.cpp, driver or kernel update)
```text
Use llamacpp-speedtune to re-check my current config after an update. Don't redo every sweep:
rebuild/compare the backends from the same commit, rerun the winning config and the runner-up for each model at my agent's depths,
and tell me if anything got slower, faster, or stopped fitting. Compare against <path to my last results file>.
```

## Add one new model
```text
Use llamacpp-speedtune to add <path/to/model.gguf> to my setup. Check it fits, compare backends, sweep its batch size, cache type
(and --n-cpu-moe if it's MoE) at <64K> context, then prepare a preset for my router and an entry for my agent config.
Show me the changes before applying them.
```

## Diagnose a slow model
```text
<Model> got slow / is slower than expected on my machine. Use llamacpp-speedtune to find out why: check RAM speed and channels,
PCIe link, VRAM use by other apps, spills to system RAM, thread count, and whether its config still fits at full context.
Tell me the cause before changing anything.
```

## Only measure, change nothing
```text
Use llamacpp-speedtune to benchmark my models at my agent's real context depths and give me a report.
Don't change any config, service or file outside the results folder.
```
