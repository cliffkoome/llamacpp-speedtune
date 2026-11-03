#!/usr/bin/env bash
# Report what the tuning workflow needs to know about this machine.
# Read-only: prints GPU, VRAM, CPU, RAM, llama.cpp builds and their devices, GGUF files and Ollama.
# Usage: detect_hw.sh [model-dir ...]   (default model dirs: ~/models ~/.cache/llama.cpp ~/.cache/huggingface)

section() { printf '\n== %s\n' "$1"; }

section "OS"
uname -srm
# shellcheck source=/dev/null
[[ -f /etc/os-release ]] && . /etc/os-release && echo "${PRETTY_NAME:-}"

section "CPU"
if command -v lscpu >/dev/null; then
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|L3'
elif [[ $(uname) == Darwin ]]; then
  sysctl -n machdep.cpu.brand_string; echo "physical: $(sysctl -n hw.physicalcpu)  logical: $(sysctl -n hw.logicalcpu)"
fi
echo "Use physical cores for -t; logical (SMT) threads can halve MoE generation speed."

section "RAM"
if command -v free >/dev/null; then free -h | sed -n 1,2p; else echo "$(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"; fi
# Speed and populated slots matter for MoE models and partial offload: CPU-side weights are read
# from RAM for every token, so generation speed scales with memory bandwidth.
if command -v udevadm >/dev/null; then
  udevadm info -e 2>/dev/null | awk -F= '
    /MEMORY_DEVICE_[0-9]+_SIZE=/ {split($1,a,"_"); size[a[3]]=$2}
    /MEMORY_DEVICE_[0-9]+_CONFIGURED_SPEED_MTS=/ {split($1,a,"_"); spd[a[3]]=$2}
    /MEMORY_DEVICE_[0-9]+_TYPE=/ {split($1,a,"_"); typ[a[3]]=$2}
    END {n=0; for (i in size) if (size[i]>0) {n++; printf "DIMM: %d GiB %s @ %s MT/s\n", size[i]/1073741824, typ[i], spd[i]; s=spd[i]}
         if (n) printf "%d module(s) populated. Rough peak bandwidth if dual channel: %.0f GB/s (1 module = single channel, half that)\n", n, (n>=2?2:1)*s*8/1000}'
fi
echo "Note: XMP/EXPO off (RAM at 2133/2400 instead of its rated speed) or a single module can cut MoE generation speed by 30-50%."

section "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
fi
if command -v lspci >/dev/null; then lspci | grep -iE 'vga|3d|display'; fi
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  [[ -f $f ]] && echo "AMD VRAM total: $(( $(cat "$f") / 1048576 )) MiB, used: $(( $(cat "${f%total}used") / 1048576 )) MiB"
done
if command -v rocminfo >/dev/null; then
  gfx=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]{3,4}[0-9a-f]?\b' | sort -u | tr '\n' ' ')
  echo "ROCm agents: $gfx"
  # Consumer chips run ROCm by posing as the supported chip of their family (see references/backends.md).
  case "$gfx" in
    *gfx103[1-6]*) suggested=10.3.0 ;;
    *gfx110[1-3]*) suggested=11.0.0 ;;
    *) suggested= ;;
  esac
  if [[ -n $suggested ]]; then
    echo "This chip needs HSA_OVERRIDE_GFX_VERSION=$suggested for ROCm/HIP builds (used below to list devices)."
    : "${HSA_OVERRIDE_GFX_VERSION:=$suggested}"
  fi
fi
# PCIe link: matters when weights or experts live in RAM (large -ub streams them to the GPU).
# AMD cards contain an internal switch that reports x16; the narrowest link up the chain is the real one.
for card in /sys/class/drm/card[0-9]; do
  [[ -e $card/device ]] || continue
  d=$(readlink -f "$card/device"); narrow=""
  while [[ $d == /sys/devices/* ]]; do
    if [[ -f $d/current_link_width ]]; then
      w=$(cat "$d/current_link_width"); sp=$(cat "$d/current_link_speed")
      [[ -z $narrow || $w -lt ${narrow%%|*} ]] && narrow="$w|$sp|$(cat "$d/max_link_width")"
    fi
    d=$(dirname "$d")
  done
  [[ -n $narrow ]] && IFS='|' read -r w sp mw <<<"$narrow" && echo "$(basename "$card") PCIe link: x$w @ $sp (narrowest in chain; that link supports up to x$mw). If the GPU sits in a chipset slot or a riser, this can be far below the card's own spec."
  pl=$card/device/power_dpm_force_performance_level
  [[ -f $pl ]] && echo "$(basename "$card") power level: $(cat "$pl") (auto is fine; 'low' or a laptop power-saver profile skews results)"
done
[[ $(uname) == Darwin ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset|Metal|Memory'

section "llama.cpp builds"
mapfile -t bins < <( { command -v llama-server; find ~ -maxdepth 8 -type f -name llama-server -perm -u+x 2>/dev/null; } | sort -u )
for b in "${bins[@]}"; do
  [[ -x $b ]] || continue
  echo "-- $b"
  "$b" --version 2>&1 | grep -E 'version|built' | head -2
  env ${HSA_OVERRIDE_GFX_VERSION:+HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION} "$b" --list-devices 2>&1 | grep -E '^\s+\S+: ' || echo "   (CPU only: no GPU devices)"
done
[[ ${#bins[@]} -eq 0 ]] && echo "none found: see references/backends.md to build one"

section "GGUF models (size vs memory)"
vram_mib=0
# Use the card with the most VRAM: an integrated GPU's small carve-out must not hide the real card.
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  [[ -f $f ]] || continue
  v=$(( $(cat "$f") / 1048576 )); (( v > vram_mib )) && vram_mib=$v
done
command -v nvidia-smi >/dev/null && vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
avail_mib=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
echo "VRAM ${vram_mib} MiB, RAM available ${avail_mib:-?} MiB. Models bigger than VRAM + ~80% of available RAM will swap or fail; flagged below."
dirs=("$@"); [[ ${#dirs[@]} -eq 0 ]] && dirs=(~/models ~/.cache/llama.cpp ~/.cache/huggingface)
for d in "${dirs[@]}"; do
  [[ -d $d ]] && find "$d" -type f -name '*.gguf' -size +50M -printf '%10s  %p\n' 2>/dev/null \
    | awk -v lim="$(( vram_mib + ${avail_mib:-0} * 8 / 10 ))" '{f=($1/1048576 > lim && lim > 0) ? "  <-- too big for VRAM+RAM" : ""; printf "%7.2f GB  %s%s\n", $1/1e9, $2, f}'
done

section "Ollama"
if command -v ollama >/dev/null; then
  ollama list 2>/dev/null
  echo "Ollama blobs are plain GGUF files; on btrfs/XFS/APFS they can be reflink-copied to readable names without using extra space."
else
  echo "not installed"
fi
