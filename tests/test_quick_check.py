import quick_check as qc

HW = {"gpu": {"name": "test", "total_mib": 8192}, "physical_cores": 8, "ram_mib": 32768}
DENSE_7B = {"arch": "qwen2", "block_count": 28, "expert_count": 0, "expert_used": 0, "context_length": 32768,
            "size_mib": 4400, "moe": False}
MOE_35B = {"arch": "qwen35moe", "block_count": 40, "expert_count": 256, "expert_used": 8, "context_length": 262144,
           "size_mib": 21000, "moe": True}


def titles(findings):
    return " | ".join(f["title"] for f in findings)


def test_parse_cmdline_spellings():
    c = qc.parse_cmdline("llama-server -m /m.gguf -ngl 99 --flash-attn on --jinja -ub 2048 --n-cpu-moe 34 -c 65536")
    assert c == {"model": "/m.gguf", "ngl": "99", "flash_attn": "on", "jinja": "true", "ubatch": "2048",
                 "n_cpu_moe": "34", "ctx": "65536"}


def test_parse_cmdline_negative_value():
    assert qc.parse_cmdline("-ngl -1 -t 8")["ngl"] == "-1"


def test_parse_preset_merges_defaults(tmp_path):
    p = tmp_path / "presets.ini"
    p.write_text("version = 1\n[*]\nfa = on\nthreads = 8\n\n[a]\nmodel = /a.gguf ; comment\nthreads = 6\n[b]\nmodel = /b.gguf\n")
    out = qc.parse_preset(p)
    assert out["a"] == {"flash_attn": "on", "threads": "6", "model": "/a.gguf"}
    assert out["b"]["threads"] == "8"


def test_untuned_dense_model():
    cfg = qc.parse_cmdline("-m x -ngl 20 -t 16 -fa off -np 4 -c 32768 -ctk q4_0")
    t = titles(qc.check(cfg, DENSE_7B, HW))
    for expected in ("Flash attention is off", "Only 20 of 28 layers", "16 threads", "KV cache q4_0", "4 parallel slots"):
        assert expected in t


def test_tuned_moe_model_is_clean():
    cfg = qc.parse_cmdline("-m x -ngl 99 --n-cpu-moe 34 -ub 2048 -b 2048 -fa on -t 8 -c 65536 -ctk f16 -ctv f16")
    assert qc.check(cfg, MOE_35B, HW) == []


def test_moe_small_batch_and_no_offload():
    assert "Small batch size with experts in RAM" in titles(qc.check({"n_cpu_moe": "33"}, MOE_35B, HW))
    assert "without expert offload" in titles(qc.check({"ngl": "20"}, MOE_35B, HW))


def test_ngl_left_to_fit():
    assert "left to --fit" in titles(qc.check({}, DENSE_7B, HW))


def test_backends_machine_wide():
    assert qc.check_backends(["ROCm0", "Vulkan0"], ["", ""])
    assert not qc.check_backends(["ROCm0", "Vulkan0"], ["ROCm0", "Vulkan0"])
    assert not qc.check_backends(["Vulkan0"], [""])


def test_to_cli_round_trips_and_drops_launcher_flags():
    cfg = qc.parse_cmdline("llama-server --host 127.0.0.1 --port 0 --alias q --model /m.gguf --jinja -fa on "
                           "-ngl 99 --n-cpu-moe 34 -ub 2048 --no-mmproj-offload --log-colors off")
    args = qc.to_cli(cfg)
    assert args == "--jinja --flash-attn on --gpu-layers 99 --n-cpu-moe 34 --ubatch-size 2048 --no-mmproj-offload"
    assert qc.parse_cmdline(args) == {k: v for k, v in cfg.items() if k not in ("host", "port", "alias", "model", "log-colors")}


def test_to_cli_from_preset_booleans(tmp_path):
    p = tmp_path / "presets.ini"
    p.write_text("[*]\njinja = true\n[m]\nmodel = /m.gguf\nno-mmproj-offload = true\nctx-size = 8192\n")
    assert qc.to_cli(qc.parse_preset(p)["m"]) == "--jinja --no-mmproj-offload --ctx-size 8192"


def _ollama_model(moe=False, size=4400, params=None, loaded=None):
    return {"name": "m", "params": params or {}, "loaded": loaded,
            "meta": {"arch": "x", "block_count": 28, "expert_count": 8 if moe else 0, "expert_used": 2,
                     "context_length": 32768, "size_mib": size, "moe": moe}}


def test_ollama_defaults_flagged():
    t = titles(qc.check_ollama(_ollama_model(), {}, HW))
    assert "Context size left at Ollama's default" in t and "Flash attention not switched on" in t


def test_ollama_tuned_is_clean():
    env = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    loaded = {"size": 5000 * 2**20, "size_vram": 5000 * 2**20, "context_length": 32768}
    assert qc.check_ollama(_ollama_model(params={"num_ctx": "32768"}, loaded=loaded), env, HW) == []


def test_ollama_moe_split_and_small_context():
    loaded = {"size": 21000 * 2**20, "size_vram": 6000 * 2**20, "context_length": 4096}
    t = titles(qc.check_ollama(_ollama_model(moe=True, size=21000, loaded=loaded), {"OLLAMA_NUM_PARALLEL": "4"}, HW))
    assert "MoE model split by whole layers" in t and "Context of 4096 tokens" in t and "OLLAMA_NUM_PARALLEL=4" in t
