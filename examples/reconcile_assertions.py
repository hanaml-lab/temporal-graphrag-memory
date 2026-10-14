"""Reconcile your own assertions into bitemporal timelines with LEDGER.

    python examples/reconcile_assertions.py examples/data/account_assertions.jsonl --as-of 120

Each line of the input is one assertion, in the order the store received it: ``slot`` ([subject, predicate]),
``value``, ``modality`` (current, retrospective, prospective, reported or retraction), ``said`` (the day it was said)
and optionally ``learned`` (the day the store ingested it; default: the day it was said) and ``valid``
([start_lo, start_hi, end_lo, end_hi], a stated interval with uncertain ends). Days are numbers on any fixed origin.

For each slot the script prints LEDGER's timeline, recomputed from all the evidence whatever its arrival order, the
timeline write-time invalidation builds from the same arrivals, and what LEDGER believed at each --as-of day.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ledger import Assertion, FuzzyInterval, LedgerStore, Modality, WriteTimeInvalidation


def load(path: str) -> list[Assertion]:
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        valid = FuzzyInterval(*[math.inf if x is None else float(x) for x in d["valid"]]) if d.get("valid") else None
        out.append(Assertion(tuple(d["slot"]), d["value"], Modality(d["modality"]), float(d["said"]),
                             float(d.get("learned", d["said"])), valid, source=f"line {i + 1}"))
    return out


def show(timeline) -> str:
    parts = [f"{s.value} [{s.start:g}, {'...' if math.isinf(s.end) else f'{s.end:g}'})" for s in timeline.segments if s.value]
    return " -> ".join(parts) or "(nothing)"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("assertions", help="JSON Lines file, in arrival order")
    ap.add_argument("--as-of", type=float, nargs="*", default=[], help="days at which to ask what the store believed")
    args = ap.parse_args(argv)
    assertions = load(args.assertions)
    ledger, wti = LedgerStore(), WriteTimeInvalidation(use_modality=True)
    for a in assertions:
        ledger.append(a)
        wti.ingest(a)
    out = {}
    for slot in dict.fromkeys(a.slot for a in assertions):
        current = ledger.current(slot)
        print(f"{slot[0]}.{slot[1]}")
        print(f"  {'LEDGER':<25}{show(current)}")
        print(f"  {'write-time invalidation':<25}{show(wti.timeline(slot))}")
        for day in args.as_of:
            print(f"  as of day {day:g}, LEDGER held {show(ledger.as_of(slot, day))}")
        out[slot] = current
    return out


if __name__ == "__main__":
    main()
