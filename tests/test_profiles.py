import csv

import profiles as pr

CFG = {"models": {"q": {"alias": "qwen-x", "path": "/m/q.gguf", "ctx": 65536, "mmproj": "/m/mm.gguf",
                        "engines": {"ex": "Qwen-exl2"}}},
       "backends": {"hip": {"server": "/opt/llama/llama-server", "env": {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}}},
       "engines": {"ex": {"start": "/e/start.sh --port {port} --model-name {model} --max-seq-len {ctx}"}}}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def results(tmp_path):
    row = {"model": "q", "args": "-ngl 99 --n-cpu-moe 34 -ub 2048 -fa on --device ROCm0", "prompt_n": 49000,
           "tg_ts": 25.3, "vram_mib": 7475}
    write(tmp_path / "validate.csv", [{**row, "label": "after", "pp_ts": 526},
                                      {**row, "label": "before", "pp_ts": 172, "tg_ts": 19.2}])
    write(tmp_path / "openai.csv", [{**row, "label": "after", "pp_ts": 600}])
    return tmp_path


def test_llamacpp_profile_and_presets(tmp_path):
    p = pr.build_profile(CFG, "q", "hip", "after", results(tmp_path))
    assert p["engine"] == "llama.cpp" and p["backend"] == "hip"
    assert p["args"] == "--gpu-layers 99 --n-cpu-moe 34 --ubatch-size 2048 --flash-attn on --device ROCm0"
    assert p["measured"]["at_depth"][0]["pp_ts"] == 526 and p["before"]["at_depth"][0]["pp_ts"] == 172
    ini = pr.presets_ini([p])
    assert "[qwen-x]" in ini and "mmproj = /m/mm.gguf" in ini and "ctx-size = 65536" in ini
    assert "n-cpu-moe = 34" in ini and "device = ROCm0" in ini


def test_mixed_engines_give_llama_swap(tmp_path):
    res = results(tmp_path)
    a = pr.build_profile(CFG, "q", "hip", "after", res)
    b = pr.build_profile(CFG, "q", "ex", "after", res)
    assert b["engine"] == "ex" and b["model_file"] == "Qwen-exl2"
    y = pr.llama_swap_yaml([a, b])
    assert "--port ${PORT}" in y and "--model-name Qwen-exl2 --max-seq-len 65536" in y
    assert '"HSA_OVERRIDE_GFX_VERSION=10.3.0"' in y


def test_save_keeps_history(tmp_path):
    cfg = {**CFG, "profiles_dir": str(tmp_path / "profiles")}
    p = pr.build_profile(CFG, "q", "hip", "after", results(tmp_path))
    pr.save(cfg, p)
    pr.save(cfg, {**p, "saved": "2026-11-04T10:00:00"})
    assert len(list((tmp_path / "profiles" / "history").glob("q-*.json"))) == 1
    assert [x["model"] for x in pr.load_all(cfg)] == ["q"]
