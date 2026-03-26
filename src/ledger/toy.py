"""A CPU toy setting: synthetic life scripts, three ingest regimes and five bitemporal question types.

Each user has six functional slots whose true values change a few times over two years. The script discloses
those changes the way people do in conversation: promptly or months late, with or without a date, sometimes with
a wrong date that is corrected later. It also produces restatements, undated reminiscences, plans and
hypotheticals, and claims reported by other people. Assertions are generated directly (the extraction step is
replaced by the script), optionally with modality-tagger noise.

Questions mirror BiTempQA: current state, value at a point in valid time, what the store believed at a
transaction time (as-of), when a change happened, and when the store first learned a value. Ground truth comes
from the script alone, never from a system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .baseline import WriteTimeInvalidation
from .reconcile import Params, Timeline
from .store import LedgerStore
from .types import Assertion, FuzzyInterval, Modality

QUESTION_TYPES = ("current", "pit", "asof", "change", "learned")
SYSTEMS = ("ledger", "wti", "wti+modality")
T_END = 720.0
PREDICATES: dict[str, list[str]] = {
    "lives_in": ["Berlin", "Lisbon", "Porto", "Oslo", "Austin", "Leeds", "Lyon", "Turin"],
    "works_at": ["Brightline", "Corvel", "Harbor", "Northwind", "Quill", "Tessera", "Umber", "Vantage"],
    "role": ["analyst", "engineer", "lead", "manager", "designer", "researcher", "consultant", "director"],
    "gym": ["Atlas", "Forge", "Summit", "Pulse", "Core", "Peak", "Iron", "Flow"],
    "team": ["payments", "search", "growth", "infra", "data", "mobile", "risk", "platform"],
    "manager": ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana"],
}


@dataclass
class SlotScript:
    slot: tuple[str, str]
    changes: list[tuple[float, str]]  # (valid start, value); the first starts at 0
    statements: list[tuple[Assertion, bool, int]] = field(default_factory=list)  # (assertion, correct disclosure, segment)

    def value_at(self, t: float) -> str:
        v = self.changes[0][1]
        for start, value in self.changes:
            if start <= t:
                v = value
        return v

    def segment_at(self, t: float) -> int:
        return max(i for i, (start, _) in enumerate(self.changes) if start <= t)


def _sessions(rng: random.Random) -> list[float]:
    days, t = [], rng.uniform(0, 6)
    while t < T_END:
        days.append(round(t, 1))
        t += rng.uniform(5, 15)
    return days


def _after(sessions: list[float], t: float) -> float:
    for s in sessions:
        if s >= t:
            return s
    return sessions[-1]


def _script(rng: random.Random, user: str, predicate: str, sessions: list[float], p_late: float) -> SlotScript:
    values = PREDICATES[predicate]
    v = rng.choice(values)
    changes = [(0.0, v)]
    t = 0.0
    while True:
        t += rng.expovariate(1 / 240)
        if t > T_END - 160:
            break
        v = rng.choice([x for x in values if x != v])
        changes.append((round(t, 1), v))
    sc = SlotScript((user, predicate), changes)
    n = 0

    def say(value, stmt, modality, valid=None, correct=False, seg=-1):
        nonlocal n
        n += 1
        a = Assertion(sc.slot, value, modality, stmt, stmt, valid, 1.0, f"{user}/{predicate}/{n}", f"{user}/{predicate}/{n}")
        sc.statements.append((a, correct, seg))

    say(changes[0][1], _after(sessions, rng.uniform(0, 40)), Modality.CURRENT, correct=True, seg=0)
    for k, (c, value) in enumerate(changes[1:], start=1):
        late = rng.random() < p_late
        stmt = _after(sessions, c + (rng.uniform(30, 150) if late else rng.uniform(0, 12)))
        half = 12.0 if late else 2.0
        nxt = changes[k + 1][0] if k + 1 < len(changes) else math.inf
        if stmt >= nxt:  # the change was over before it was mentioned: a dated retrospective
            w = min(10.0, (nxt - c) / 4)
            say(value, stmt, Modality.RETROSPECTIVE, FuzzyInterval(c - w, c + w, nxt - w, nxt + w), correct=True, seg=k)
            continue
        if rng.random() < (0.85 if late else 0.6):
            claimed = c
            wrong = late and rng.random() < 0.3
            if wrong:
                lower = changes[k - 1][0] + 5
                claimed = min(stmt - 5, max(lower, c + rng.choice([-1, 1]) * rng.uniform(30, 60)))
            valid = FuzzyInterval(claimed - half, min(claimed + half, stmt), stmt, stmt)
            say(value, stmt, Modality.CURRENT, valid, correct=not wrong, seg=k)
            if wrong:
                fix = _after(sessions, stmt + rng.uniform(20, 90))
                lo, hi = sorted((claimed, c))
                if claimed < c:
                    say(value, fix, Modality.RETRACTION, FuzzyInterval.exact(lo, hi), seg=k)
                say(value, fix, Modality.CURRENT, FuzzyInterval(c - 3, min(c + 3, fix), fix, fix), correct=True, seg=k)
        else:
            say(value, stmt, Modality.CURRENT, correct=True, seg=k)
        if rng.random() < 0.3:
            say(changes[k - 1][1], stmt, Modality.RETRACTION, seg=k - 1)
    for s in sessions:
        r = rng.random()
        if r < 0.11:  # restatement of the current value
            seg = sc.segment_at(s)
            say(sc.value_at(s), s, Modality.CURRENT, correct=True, seg=seg)
        elif r < 0.135 and len(changes) > 1 and s > changes[1][0] + 30:  # reminiscence about an ended segment
            ended = [k for k in range(len(changes) - 1) if changes[k + 1][0] <= s]
            k = rng.choice(ended)
            if rng.random() < 0.3:
                lo, hi = changes[k][0], changes[k + 1][0]
                w = min(10.0, (hi - lo) / 4)
                valid = FuzzyInterval(lo - w, lo + w, hi - w, hi + w)
                say(changes[k][1], s, Modality.RETROSPECTIVE, valid, correct=True, seg=k)
            else:
                say(changes[k][1], s, Modality.RETROSPECTIVE, seg=k)
        elif r < 0.155:  # plan or hypothetical, half the time about the next true value
            nxt = [v for start, v in changes if start > s]
            guess = nxt[0] if nxt and rng.random() < 0.5 else rng.choice([x for x in values if x != sc.value_at(s)])
            say(guess, s, Modality.PROSPECTIVE)
        elif r < 0.17:  # someone else's claim, right 60% of the time
            right = rng.random() < 0.6
            value = sc.value_at(s) if right else rng.choice([x for x in values if x != sc.value_at(s)])
            say(value, s, Modality.REPORTED, correct=right, seg=sc.segment_at(s) if right else -1)
    return sc


def _ingest_times(rng: random.Random, sessions: list[float], regime: str, delayed: float) -> dict[float, float]:
    if regime == "in-order":
        return {s: s + 0.5 for s in sessions}
    if regime == "shuffled":
        order = sessions[:]
        rng.shuffle(order)
        return {s: T_END + 1 + i * 60.0 / len(order) for i, s in enumerate(order)}
    if regime == "partial":
        return {s: s + 0.5 + (rng.uniform(30, 300) if rng.random() < delayed else 0.0) for s in sessions}
    raise ValueError(regime)


def _noisy(rng: random.Random, a: Assertion, noise: float) -> Assertion:
    if noise <= 0 or rng.random() >= noise:
        return a
    other = rng.choice([m for m in Modality if m is not a.modality])
    return Assertion(a.slot, a.value, other, a.stmt_time, a.tx_time, a.valid, a.confidence, a.source, a.id)


def _discloses(a: Assertion, seg: int) -> bool:
    """Whether an assertion tells the store a segment's value (a wrong start date still discloses the value)."""
    if seg < 0:
        return False
    return a.modality in (Modality.CURRENT, Modality.REPORTED) or (a.modality is Modality.RETROSPECTIVE and a.valid is not None)


def _answer(tl: Timeline, t: float) -> str | None:
    v = tl.value_at(t)
    return v if v is not None else tl.last_known(t)[0]


def run_toy(
    n_users: int = 40,
    seed: int = 0,
    regime: str = "in-order",
    delayed: float = 0.3,
    modality_noise: float = 0.0,
    p_late: float = 0.35,
    params: Params = Params(),
) -> dict:
    """Generate scripts, ingest them under a regime into each system and score the five question types."""
    correct = {s: {q: 0 for q in QUESTION_TYPES} for s in SYSTEMS}
    asked = {q: 0 for q in QUESTION_TYPES}
    for u in range(n_users):
        user = f"user{u}"
        srng = random.Random(f"script/{seed}/{u}")
        sessions = _sessions(srng)
        scripts = [_script(srng, user, p, sessions, p_late) for p in PREDICATES]
        tx_of = _ingest_times(random.Random(f"ingest/{seed}/{u}/{regime}/{delayed}"), sessions, regime, delayed)
        nrng = random.Random(f"noise/{seed}/{u}/{modality_noise}")
        rng = random.Random(f"questions/{seed}/{u}")
        stream = []
        for sc in scripts:
            for a, ok, seg in sc.statements:
                tx = tx_of[a.stmt_time]
                b = Assertion(a.slot, a.value, a.modality, a.stmt_time, tx, a.valid, a.confidence, a.source, a.id)
                stream.append((b, ok, seg))
        stream.sort(key=lambda item: item[0].tx_time)
        ledger = LedgerStore(params)
        wti, wtim = WriteTimeInvalidation(), WriteTimeInvalidation(use_modality=True)
        for a, _, _ in stream:
            tagged = _noisy(nrng, a, modality_noise)
            ledger.append(tagged)
            wti.ingest(a)
            wtim.ingest(tagged)
        tx_lo, tx_hi = stream[0][0].tx_time, stream[-1][0].tx_time
        for sc in scripts:
            slot = sc.slot
            final = {"ledger": ledger.current(slot), "wti": wti.timeline(slot), "wti+modality": wtim.timeline(slot)}
            disclosures = [(a.tx_time, seg) for a, _, seg in stream if a.slot == slot and _discloses(a, seg)]
            # current state and point-in-time values
            asked["current"] += 1
            for s in SYSTEMS:
                correct[s]["current"] += _answer(final[s], T_END - 1) == sc.value_at(T_END - 1)
            for _ in range(2):
                t = rng.uniform(5, T_END - 5)
                asked["pit"] += 1
                for s in SYSTEMS:
                    correct[s]["pit"] += final[s].value_at(t) == sc.value_at(t)
            # as-of: what the store held at transaction time tx about the then-current value
            for _ in range(2):
                tx = rng.uniform(tx_lo, tx_hi)
                tv = min(tx, T_END - 1)
                known = {seg for t_in, seg in disclosures if t_in <= tx}
                cands = [k for k in known if sc.changes[k][0] <= tv]
                ideal = sc.changes[max(cands)][1] if cands else None
                views = {"ledger": ledger.as_of(slot, tx), "wti": wti.timeline(slot, as_of=tx), "wti+modality": wtim.timeline(slot, as_of=tx)}
                asked["asof"] += 1
                for s in SYSTEMS:
                    correct[s]["asof"] += _answer(views[s], tv) == ideal
            # when did each change happen
            for k in range(1, len(sc.changes)):
                c, value = sc.changes[k]
                prev = sc.changes[k - 1][1]
                asked["change"] += 1
                for s in SYSTEMS:
                    named = [g for g in final[s].segments if g.value is not None]
                    hits = [b.start for a, b in zip(named, named[1:]) if a.value == prev and b.value == value]
                    correct[s]["change"] += any(abs(h - c) <= 31 for h in hits)
            # when did the store first learn each value (values that occur once only)
            counts: dict[str, int] = {}
            for _, v in sc.changes:
                counts[v] = counts.get(v, 0) + 1
            for k, (_, value) in enumerate(sc.changes):
                if counts[value] > 1:
                    continue
                times = [a.tx_time for a, _, seg in stream if a.slot == slot and seg == k and a.modality is not Modality.RETRACTION]
                if not times:
                    continue
                ideal_t = min(times)
                asked["learned"] += 1
                got = {
                    "ledger": ledger.learned_at(slot, value),
                    "wti": min((e.created_at for e in wti.edges if e.slot == slot and e.value == value), default=None),
                    "wti+modality": min((e.created_at for e in wtim.edges if e.slot == slot and e.value == value), default=None),
                }
                for s in SYSTEMS:
                    correct[s]["learned"] += got[s] is not None and abs(got[s] - ideal_t) <= 1.0
    total = sum(asked.values())
    accuracy = {
        s: {**{q: correct[s][q] / max(asked[q], 1) for q in QUESTION_TYPES}, "overall": sum(correct[s].values()) / total}
        for s in SYSTEMS
    }
    return {"accuracy": accuracy, "questions": {**asked, "overall": total}, "regime": regime, "seed": seed}


def format_table(result: dict) -> str:
    """A plain-text table of one run."""
    cols = list(QUESTION_TYPES) + ["overall"]
    lines = ["system         " + " ".join(f"{c:>8}" for c in cols)]
    for s, acc in result["accuracy"].items():
        lines.append(f"{s:<14} " + " ".join(f"{100 * acc[c]:8.1f}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    for regime in ("in-order", "shuffled"):
        print(regime)
        print(format_table(run_toy(regime=regime)))
