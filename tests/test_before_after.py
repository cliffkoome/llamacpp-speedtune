import csv

import before_after as ba


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def test_validate_and_agentic_tables(tmp_path):
    write_csv(tmp_path / "validate.csv", [
        {"model": "q", "label": "before", "prompt_n": 49010, "pp_ts": 172, "tg_ts": 19.2, "vram_mib": 7000},
        {"model": "q", "label": "after", "prompt_n": 48990, "pp_ts": 526, "tg_ts": 25.3, "vram_mib": 7475},
        {"model": "other", "label": "after", "prompt_n": 49000, "pp_ts": 1, "tg_ts": 1, "vram_mib": 1},
    ])
    write_csv(tmp_path / "agentic.csv", [
        {"model": "q", "label": "before", "passed": "True", "wall_s": 300, "pp_ts": 150, "tg_ts": 20},
        {"model": "q", "label": "before", "passed": "False", "wall_s": 300, "pp_ts": 150, "tg_ts": 20},
        {"model": "q", "label": "after", "passed": "True", "wall_s": 150, "pp_ts": 450, "tg_ts": 25},
        {"model": "q", "label": "after", "passed": "True", "wall_s": 150, "pp_ts": 450, "tg_ts": 25},
    ])
    out = "\n".join(ba.report(tmp_path, "q", "before", "after"))
    assert "| 49K | 172 | 526 | **3.1×** | 19.2 | 25.3 | **+32%** | 7.3 GB |" in out
    assert "| Passed | 1/2 | 2/2 | |" in out
    assert "**3.0×**" in out  # prompt speed during tasks


def test_micro_filtered_by_model_file(tmp_path):
    base = {"n_gen": 0, "n_depth": 16384}
    write_csv(tmp_path / "micro.csv", [
        {**base, "label": "before", "model_filename": "/m/a.gguf", "n_prompt": 4096, "avg_ts": 100},
        {**base, "label": "after", "model_filename": "/m/a.gguf", "n_prompt": 4096, "avg_ts": 150},
        {**base, "label": "after", "model_filename": "/m/b.gguf", "n_prompt": 4096, "avg_ts": 999},
    ])
    out = "\n".join(ba.report(tmp_path, "a", "before", "after", model_file="a.gguf"))
    assert "| 16K | 100 | 150 | **+50%** |" in out and "999" not in out


def test_nothing_found(tmp_path):
    assert ba.report(tmp_path, "q", "before", "after") == []
