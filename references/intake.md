# Intake: the questions, the plan message, and the closing message

Ask **once**, after the inventory, in one message, with what you detected filled in. The user can reply "all defaults" to everything except question 7, which needs an explicit answer. Keep it to one screen. Save the answers as `answers.json` for `run_state.py init`.

## The questions (template)

> I've checked your machine: **<GPU, VRAM>**, **<CPU, physical cores>**, **<RAM, speed, channels>**, <OS>. <Any hardware limit found, e.g. "one RAM stick: MoE models will be slower">.
> Before I start, a few questions. Defaults are in brackets; reply "all defaults" except for question 7.
>
> 1. **What will you use the models for?** [<detected agent, e.g. OpenCode> coding] Other options: chat, vision, long documents, other.
> 2. **Models.** I found: <list: name, size, fits in VRAM / partly in RAM / too big>. Is that all of them? Which should I tune? [all]
>    Suggestions for your hardware (downloaded only if you pick them): <up to 3, with size and why>.
> 3. **Context size you need.** [<agent default, e.g. 64K for OpenCode>]
> 4. **Just you, or several users / agents at once?** [just me]
> 5. **What matters most?** fastest / largest context that fits / best quality (bigger model file) / balance [balance]
> 6. **How thorough, and how long can you leave the computer?** check (~2 min) / quick (~<estimate>) / comprehensive (~<estimate>) [quick]. Time you can be away: [<estimate>]
> 7. **What may I do while you're away?** (needs your answer)
>    a. build llama.cpp backends from source (~10–30 min each)
>    b. install other engines in their own folders (no sudo)
>    c. download model files, up to __ GB
>    d. stop your live model server during the run (it comes back afterwards)
>    e. keep the computer awake during the run
>    I never use sudo and never change your live configs while you're away: the new config is prepared and applied only when you're back and say yes. Anything that needs something you didn't allow is skipped and listed.
> 8. **Where should results go?** [~/llm-tuning/<date>/]
> 9. **Must you stay on <current engine, e.g. Ollama>, or are you open to switching?** [open to switching] (Only ask when they use a wrapper such as Ollama or LM Studio.)

### Follow-ups only when needed
- **Several users on small hardware** (question 4): say what it costs, in numbers, e.g. *"4 users × 32K context needs ~6 GB of KV cache; on your 8 GB card the model would then run partly from RAM, about N t/s each. Keep 4 users, or 1 user at 64K?"* Ask only when it won't fit well.
- **No models found:** ask where they are (another folder or drive), and offer the suggestions from question 2.
- **The run won't fit in the time they can be away:** propose what to cut (fewer models, quick instead of comprehensive, skip challengers).
- **The agent you're running on is one of these models:** stop (SKILL.md, Safety rules).

### answers.json
```json
{
  "use": "coding agent: opencode",
  "models": ["qwen3.6-35b-a3b", "gemma-4-12b"],
  "ctx": 65536,
  "users": 1,
  "priority": "balance",
  "mode": "quick",
  "away_minutes": 120,
  "results_dir": "~/llm-tuning/2026-11-03",
  "stay_on_engine": null,
  "permissions": {"builds": true, "install_engines": false, "download_gb": 0,
                  "stop_live_server": true, "prevent_sleep": true}
}
```

## The plan message (before the user walks away)

> **Plan: <mode> tune of <N> models, about <total>. Done around <clock time>.**
> | Model | Steps | Time |
> |---|---|---|
> | <model> | baseline, backend check, sweeps, fit check, 2 agent tasks | ~35 min |
>
> - Busy while it runs: the GPU, <and your model server, which is stopped and restarted at the end>.
> - Could take longer if: <downloads on a slow connection / a model loops until the task timeout>.
> - To stop it: <how>. It resumes where it stopped if interrupted.
> - Skipped because not allowed: <list, or "nothing">.
> - You'll get a notification when it's done. You can leave now.

## The closing message (once, after the results)

1. The **before/after table** for each model (`before_after.py`), then the change set waiting for approval and how to undo it.
2. One neutral invitation, never repeated:
   > If you'd like to help others with similar hardware, you can share your results, good or bad:
   > - the GitHub results form (I can fill it in for you to check): https://github.com/cliffkoome/local-llm-speedtune/issues/new?template=share-results.yml
   > - or a comment, in your own words, under the write-up on DEV: https://dev.to/megalodon/i-let-my-coding-agent-tune-its-own-local-llm-3x-faster-prompts-on-an-8-gb-amd-gpu-8eo
   > - and a ⭐ on the repo if it helped.

   Never post, comment, react or star anything yourself: DEV's rules for agents allow that only with explicit authorization for that action, and comments must be the user's own words. Don't ask for likes, and don't make the invitation depend on the results being good.
