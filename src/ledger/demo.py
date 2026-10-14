"""LEDGER on the CPU toy setting.

    python -m ledger.demo [--config configs/toy.yaml] [--users N] [--seeds K]

Part 1 ingests one history of 20 assertions about two slots in many random arrival orders and counts the distinct
final timelines each store ends with. Part 2 runs the BiTempQA-style toy (synthetic life scripts, five question
types) with evidence ingested in order and shuffled.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import yaml

from .baseline import WriteTimeInvalidation
from .reconcile import Params, Timeline
from .store import LedgerStore
from .toy import QUESTION_TYPES, SYSTEMS, run_toy
from .types import INF, Assertion, FuzzyInterval, Modality

C, R, P, REP, X = (Modality.CURRENT, Modality.RETROSPECTIVE, Modality.PROSPECTIVE, Modality.REPORTED,
                   Modality.RETRACTION)
LIVES, WORKS = ("user", "lives_in"), ("user", "works_at")


def _say(slot, value, modality, day, valid=None) -> Assertion:
    return Assertion(slot, value, modality, float(day), float(day), valid)


# What a user said over 20 months, in days. They lived in Berlin, moved to Lisbon around day 124 and to Porto around
# day 390; they worked at Quill until day 300, then at Harbor. Plans, an undated reminiscence and other people's
# claims carry no valid-time evidence for a change, and two moves are disclosed late with a stated date.
HISTORY = [
    _say(LIVES, "Berlin", C, 10), _say(LIVES, "Berlin", C, 40),
    _say(LIVES, "Lisbon", P, 95),                                           # "I might move to Lisbon"
    _say(LIVES, "Lisbon", C, 160, FuzzyInterval(118, 130, 160, 160)),       # "I moved to Lisbon in May"
    _say(LIVES, "Lisbon", C, 200),
    _say(LIVES, "Berlin", R, 230),                                          # "back when I lived in Berlin"
    _say(LIVES, "Porto", REP, 260),                                         # "my sister says I'm moving to Porto"
    _say(LIVES, "Lisbon", C, 300),
    _say(LIVES, "Porto", C, 420, FuzzyInterval(385, 395, 420, 420)),        # "in Porto since early April"
    _say(LIVES, "Porto", C, 470),
    _say(LIVES, "Lisbon", X, 520),                                          # "I don't live in Lisbon any more"
    _say(WORKS, "Quill", C, 15), _say(WORKS, "Quill", C, 120),
    _say(WORKS, "Harbor", P, 210),                                          # "I'm interviewing at Harbor"
    _say(WORKS, "Harbor", C, 330, FuzzyInterval(298, 302, 330, 330)),       # "I joined Harbor last month"
    _say(WORKS, "Quill", R, 360, FuzzyInterval(-10, 0, 295, 305)),          # "I was at Quill until then"
    _say(WORKS, "Harbor", C, 400),
    _say(WORKS, "Quill", REP, 450),                                         # "my old manager says I'm going back"
    _say(WORKS, "Harbor", C, 500), _say(WORKS, "Harbor", C, 540),
]
TRUTH = {LIVES: [(0.0, 124.0, "Berlin"), (124.0, 390.0, "Lisbon"), (390.0, INF, "Porto")],
         WORKS: [(0.0, 300.0, "Quill"), (300.0, INF, "Harbor")]}
PROBES = [float(t) for t in range(20, 600, 10) if all(abs(t - b) > 15 for b in (124, 300, 390))]


def _true(slot, t: float) -> str:
    return next(v for start, end, v in TRUTH[slot] if start <= t < end)


def _shape(tl: Timeline) -> tuple:
    return tuple((round(s.start, 1), round(s.end, 1), s.value) for s in tl.segments)


def order_invariance(n_orders: int, seed: int = 0, params: Params = Params()) -> dict:
    """Ingest HISTORY in n_orders arrival orders (the first in chronological order) and compare the final timelines."""
    rng = random.Random(seed)
    shapes = {s: set() for s in SYSTEMS}
    correct = {s: [] for s in SYSTEMS}
    for i in range(n_orders):
        order = HISTORY[:]
        if i:
            rng.shuffle(order)
        ledger = LedgerStore(params)
        wti, wtim = WriteTimeInvalidation(), WriteTimeInvalidation(use_modality=True)
        for j, a in enumerate(order):  # every assertion arrives after the fact, in this order
            b = Assertion(a.slot, a.value, a.modality, a.stmt_time, 600.0 + j, a.valid)
            ledger.append(b)
            wti.ingest(b)
            wtim.ingest(b)
        final = {"ledger": ledger.current, "wti": wti.timeline, "wti+modality": wtim.timeline}
        for name, timeline in final.items():
            shapes[name].add(tuple(_shape(timeline(slot)) for slot in TRUTH))
            hits = [timeline(slot).value_at(t) == _true(slot, t) for slot in TRUTH for t in PROBES]
            correct[name].append(sum(hits) / len(hits))
    return {name: {"timelines": len(shapes[name]), "mean": sum(correct[name]) / n_orders,
                   "worst": min(correct[name]), "in_order": correct[name][0]} for name in SYSTEMS}


def sweep(cfg: dict, users: int, seeds: list[int]) -> dict:
    params = Params(**cfg.get("params", {}))
    cols = list(QUESTION_TYPES) + ["overall"]
    out = {}
    for regime in cfg["regimes"]:
        runs = [run_toy(n_users=users, seed=s, regime=regime, p_late=cfg["p_late"], modality_noise=cfg["modality_noise"],
                        params=params) for s in seeds]
        out[regime] = {name: {c: sum(r["accuracy"][name][c] for r in runs) / len(runs) for c in cols} for name in SYSTEMS}
        out[regime]["questions"] = runs[0]["questions"]["overall"]
    return out


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    ap.add_argument("--users", type=int, default=None, help="override the number of synthetic users")
    ap.add_argument("--seeds", type=int, default=None, help="use the first K seeds of the config")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    users = args.users or cfg["users"]
    seeds = cfg["seeds"][: args.seeds] if args.seeds else cfg["seeds"]
    t0 = time.time()
    inv = order_invariance(cfg["orders"], params=Params(**cfg.get("params", {})))
    print(f"one history of {len(HISTORY)} assertions about 2 slots, ingested in {cfg['orders']} random orders")
    print(f"{'store':<14}{'distinct final timelines':>26}{'correct, in order':>19}{'mean':>7}{'worst':>7}")
    for name, r in inv.items():
        print(f"{name:<14}{r['timelines']:>26}{r['in_order']:>19.3f}{r['mean']:>7.3f}{r['worst']:>7.3f}")
    res = sweep(cfg, users, seeds)
    cols = list(QUESTION_TYPES) + ["overall"]
    for regime, table in res.items():
        print(f"\nBiTempQA-style toy, {regime} ingest ({users} users, seeds {seeds}, {table['questions']} questions), accuracy %")
        print(f"{'store':<14}" + "".join(f"{c:>9}" for c in cols))
        for name in SYSTEMS:
            print(f"{name:<14}" + "".join(f"{100 * table[name][c]:>9.1f}" for c in cols))
    print(f"\ndone in {time.time() - t0:.1f}s")
    return {"order_invariance": inv, "sweep": res}


if __name__ == "__main__":
    main()
