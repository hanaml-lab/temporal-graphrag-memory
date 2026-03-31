"""Sweep the CPU toy setting over seeds and ingest regimes and print one table per regime.

    python scripts/run_toy.py --config configs/toy.toml [--users N] [--seeds K] [--out results/toy.json]
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ledger import Params
from ledger.toy import QUESTION_TYPES, SYSTEMS, run_toy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy.toml")
    ap.add_argument("--users", type=int, default=None, help="override the number of synthetic users")
    ap.add_argument("--seeds", type=int, default=None, help="use the first K seeds of the config")
    ap.add_argument("--out", default="results/toy.json")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    users = args.users or cfg["users"]
    seeds = cfg["seeds"][: args.seeds] if args.seeds else cfg["seeds"]
    params = Params(**cfg.get("params", {}))
    cols = list(QUESTION_TYPES) + ["overall"]
    out = {"users": users, "seeds": seeds, "regimes": {}}
    for regime in cfg["regimes"]:
        runs = [run_toy(n_users=users, seed=s, regime=regime, p_late=cfg["p_late"], modality_noise=cfg["modality_noise"], params=params) for s in seeds]
        mean = {sys_: {c: sum(r["accuracy"][sys_][c] for r in runs) / len(runs) for c in cols} for sys_ in SYSTEMS}
        out["regimes"][regime] = {"mean": mean, "questions": runs[0]["questions"]}
        print(f"\n{regime} ({users} users, seeds {seeds})")
        print("system         " + " ".join(f"{c:>8}" for c in cols))
        for sys_ in SYSTEMS:
            print(f"{sys_:<14} " + " ".join(f"{100 * mean[sys_][c]:8.1f}" for c in cols))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
