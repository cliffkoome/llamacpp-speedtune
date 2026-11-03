from quick_check import parse_cmdline
from to_ollama import convert

MOE = {"block_count": 40}


def test_dense_full_offload():
    out = convert(parse_cmdline("-m /m.gguf -ngl 99 -c 32768 -fa on -ub 1024 -ctk q8_0 -ctv q8_0 -t 8"))
    assert out["modelfile"] == ["FROM /m.gguf", "PARAMETER num_ctx 32768", "PARAMETER num_gpu 999",
                                "PARAMETER num_thread 8", "PARAMETER num_batch 1024"]
    assert out["env"] == {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_NUM_PARALLEL": "1"}
    assert out["lost"] == []


def test_moe_offload_is_approximate():
    out = convert(parse_cmdline("-m /m.gguf -ngl 99 --n-cpu-moe 34 -c 65536 -ub 2048 -b 4096 --device ROCm0"), MOE)
    assert "PARAMETER num_gpu 6" in out["modelfile"]
    lost = " ".join(out["lost"])
    assert "--n-cpu-moe 34" in lost and "device = ROCm0" in lost and "ubatch" in lost


def test_unsupported_cache_and_mixed_types():
    out = convert(parse_cmdline("-m /m.gguf -ctk q5_1 -ctv q8_0"))
    assert "OLLAMA_KV_CACHE_TYPE" not in out["env"]
    assert any("q5_1" in x for x in out["lost"]) and any("Different K/V" in x for x in out["lost"])
