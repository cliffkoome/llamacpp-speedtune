"""Hardware facts that need no config: GPUs and their VRAM, physical cores, RAM.

Used by server.py, quick_check.py and init_config.py. Read-only.
"""
import os
import platform
import re
import subprocess
from pathlib import Path

DRM = Path("/sys/class/drm")


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def amd_gpus(drm=DRM):
    """AMD cards from sysfs: [{'name', 'total_mib', 'used_mib', 'id'}]."""
    out = []
    for total in sorted(drm.glob("card*/device/mem_info_vram_total")):
        dev = total.parent
        try:
            t = int(total.read_text()) // 2**20
            used_f = dev / "mem_info_vram_used"
            u = int(used_f.read_text()) // 2**20 if used_f.exists() else -1
        except (OSError, ValueError):
            continue
        name_f = dev / "product_name"
        name = name_f.read_text().strip() if name_f.exists() else ""
        out.append({"id": total.parts[-3], "name": name or f"AMD GPU ({total.parts[-3]})",
                    "total_mib": t, "used_mib": u, "vendor": "amd"})
    return out


def nvidia_gpus():
    out = []
    for line in _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                      "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                out.append({"id": f"nvidia{parts[0]}", "name": parts[1], "total_mib": int(parts[2]),
                            "used_mib": int(parts[3]), "vendor": "nvidia"})
            except ValueError:
                pass
    return out


def gpus(drm=DRM):
    """All GPUs with dedicated memory. On Apple Silicon: one entry for unified memory."""
    if platform.system() == "Darwin":
        mem = ram_mib()
        return [{"id": "metal0", "name": "Apple GPU (unified memory)", "total_mib": mem, "used_mib": -1,
                 "vendor": "apple", "unified": True}] if mem else []
    return amd_gpus(drm) + nvidia_gpus()


def main_gpu(drm=DRM, found=None):
    """The GPU with the most VRAM. An integrated GPU (small carve-out) never wins over a real card."""
    found = gpus(drm) if found is None else found
    return max(found, key=lambda g: g["total_mib"]) if found else None


def vram_used_mib(drm=DRM):
    g = main_gpu(drm)
    return g["used_mib"] if g else -1


def physical_cores():
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"]).strip()
        return int(out) if out.isdigit() else os.cpu_count()
    out = _run(["lscpu", "-p=CORE,SOCKET"])
    cores = {line for line in out.splitlines() if line and not line.startswith("#")}
    return len(cores) or os.cpu_count()


def ram_mib():
    """Total RAM in MiB."""
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) // 2**20 if out.isdigit() else 0
    try:
        m = re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())
        return int(m.group(1)) // 1024 if m else 0
    except OSError:
        return 0


def summary():
    g = main_gpu()
    return {"gpu": g, "physical_cores": physical_cores(), "ram_mib": ram_mib()}


if __name__ == "__main__":
    import json
    print(json.dumps({"gpus": gpus(), **summary()}, indent=1))
