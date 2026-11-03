#!/usr/bin/env python3
"""Read the few GGUF metadata fields tuning needs, with no dependencies.

Usage: gguf_meta.py <file.gguf> [...]

Returns arch, layer count (block_count), expert count (MoE if > 0), trained
context length and file size. Only the header is read; tensor data is skipped.
"""
import json
import struct
import sys
from pathlib import Path

# GGUF value types -> struct format for the fixed-size ones
_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack("<" + fmt, data)[0]

    def string(self):
        n = self.read("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vtype, keep=True):
        if vtype in _FIXED:
            return self.read(_FIXED[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.read("I"), self.read("Q")
            if itype in _FIXED:  # skip fixed-size arrays without decoding them
                self.f.seek(count * struct.calcsize("<" + _FIXED[itype]), 1)
            else:
                for _ in range(count):
                    self.value(itype, keep=False)
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_kv(path):
    """All scalar metadata as a dict (arrays such as the vocabulary are skipped)."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = r.read("I")
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} is too old")
        r.read("Q")  # tensor count
        kv = {}
        for _ in range(r.read("Q")):
            key, vtype = r.string(), r.read("I")
            val = r.value(vtype)
            if val is not None:
                kv[key] = val
        return kv


def meta(path):
    """{'arch', 'block_count', 'expert_count', 'expert_used', 'context_length', 'size_mib', 'moe', 'name'}."""
    path = Path(path).expanduser()
    kv = read_kv(path)
    arch = kv.get("general.architecture", "")

    def a(key, default=0):
        return kv.get(f"{arch}.{key}", default)

    experts = int(a("expert_count") or 0)
    return {"name": kv.get("general.name", path.stem), "arch": arch, "block_count": int(a("block_count")),
            "expert_count": experts, "expert_used": int(a("expert_used_count") or 0),
            "context_length": int(a("context_length")), "size_mib": path.stat().st_size // 2**20,
            "moe": experts > 0}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(json.dumps({"path": p, **meta(p)}))
