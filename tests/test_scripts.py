"""Seam: scripts/run_toy.py — the command-line sweep that the README's toy table comes from."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_toy_writes_a_table_per_regime(tmp_path):
    out = tmp_path / "toy.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_toy.py"), "--config", str(ROOT / "configs" / "toy.toml"),
           "--users", "4", "--seeds", "1", "--out", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert set(result["regimes"]) == {"in-order", "shuffled"}
    assert set(result["regimes"]["in-order"]["mean"]) == {"ledger", "wti", "wti+modality"}
    assert "overall" in done.stdout
