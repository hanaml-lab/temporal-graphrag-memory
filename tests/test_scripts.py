"""Seam: scripts/run_toy.py, the wrapper around python -m ledger.demo that the README's toy run comes from."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_toy_prints_both_parts():
    cmd = [sys.executable, str(ROOT / "scripts" / "run_toy.py"), "--config", str(ROOT / "configs" / "toy.yaml"), "--users", "4"]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr
    assert "distinct final timelines" in done.stdout
    assert "in-order ingest" in done.stdout and "shuffled ingest" in done.stdout
    assert "overall" in done.stdout
