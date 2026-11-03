import json
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

# server.py reads its config at import time: give it a throwaway one.
_cfg = Path(__file__).resolve().parent / ".test-config.json"
_cfg.write_text(json.dumps({"results_dir": str(_cfg.parent / ".test-results"), "backends": {}, "models": {}}))
os.environ["TUNE_CONFIG"] = str(_cfg)
