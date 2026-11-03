import struct

import gguf_meta


def _s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv):
    """kv: [(key, type, value)] with type 'str', 'u32' or 'strarr'/'f32arr'."""
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, typ, val in kv:
        out += _s(key)
        if typ == "str":
            out += struct.pack("<I", 8) + _s(val)
        elif typ == "u32":
            out += struct.pack("<II", 4, val)
        elif typ == "strarr":
            out += struct.pack("<IIQ", 9, 8, len(val)) + b"".join(_s(v) for v in val)
        elif typ == "f32arr":
            out += struct.pack("<IIQ", 9, 6, len(val)) + struct.pack(f"<{len(val)}f", *val)
    path.write_bytes(out + b"\0" * 64)


def test_moe_model(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("general.architecture", "str", "qwen3moe"), ("general.name", "str", "Test MoE"),
                   ("tokenizer.ggml.tokens", "strarr", ["a", "b", "c"]), ("tokenizer.ggml.scores", "f32arr", [0.1, 0.2]),
                   ("qwen3moe.block_count", "u32", 40), ("qwen3moe.expert_count", "u32", 128),
                   ("qwen3moe.expert_used_count", "u32", 8), ("qwen3moe.context_length", "u32", 262144)])
    m = gguf_meta.meta(p)
    assert m["arch"] == "qwen3moe" and m["block_count"] == 40 and m["moe"]
    assert m["expert_count"] == 128 and m["expert_used"] == 8 and m["context_length"] == 262144
    assert m["name"] == "Test MoE"


def test_dense_model(tmp_path):
    p = tmp_path / "d.gguf"
    write_gguf(p, [("general.architecture", "str", "llama"), ("llama.block_count", "u32", 32)])
    m = gguf_meta.meta(p)
    assert not m["moe"] and m["block_count"] == 32 and m["context_length"] == 0


def test_not_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"nope" * 10)
    try:
        gguf_meta.meta(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
