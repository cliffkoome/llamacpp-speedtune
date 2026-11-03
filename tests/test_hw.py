import hw


def fake_card(drm, name, total_mib, used_mib):
    d = drm / name / "device"
    d.mkdir(parents=True)
    (d / "mem_info_vram_total").write_text(str(total_mib * 2**20))
    (d / "mem_info_vram_used").write_text(str(used_mib * 2**20))


def test_main_gpu_prefers_largest_card(tmp_path):
    # card0 is an integrated GPU with a 512 MB carve-out; card1 the real 8 GB card.
    fake_card(tmp_path, "card0", 512, 100)
    fake_card(tmp_path, "card1", 8176, 7000)
    found = hw.amd_gpus(tmp_path)
    assert [g["id"] for g in found] == ["card0", "card1"]
    g = hw.main_gpu(found=found)
    assert g["id"] == "card1" and g["total_mib"] == 8176 and g["used_mib"] == 7000


def test_no_gpu():
    assert hw.main_gpu(found=[]) is None
