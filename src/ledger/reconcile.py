"""Exact reconciliation of one slot's valid-time timeline from a set of assertions.

Every assertion contributes a piecewise-constant evidence density over valid time for its value. The timeline
is the piecewise-constant function f maximising

    S(f) = sum_t e_{f(t)}(t) dt  -  lambda * #switches(f),

solved exactly by Viterbi over the elementary segments between evidence breakpoints. Because the evidence is
constant inside a segment, an optimal switch can always be placed on a breakpoint, so the restriction is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Assertion


@dataclass(frozen=True)
class Params:
    switch_penalty: float = 6.0  # lambda, in weight-days


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    value: str | None
    confidence: float = 1.0


@dataclass
class Timeline:
    segments: list[Segment] = field(default_factory=list)

    def _find(self, t: float) -> Segment | None:
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return None

    def value_at(self, t: float) -> str | None:
        s = self._find(t)
        return s.value if s else None

    def change_points(self) -> list[float]:
        return [b.start for a, b in zip(self.segments, self.segments[1:]) if a.value != b.value and a.value is not None and b.value is not None]


def _pieces(a: Assertion, params: Params) -> list[tuple[float, float, str, float]]:
    """(start, end, value, density) evidence pieces for one assertion."""
    if a.valid is None:
        return []
    return [(lo, hi, a.value, a.confidence * m) for lo, hi, m in a.valid.pieces()]


def _evidence(assertions: list[Assertion], params: Params) -> tuple[list[float], list[str | None], np.ndarray]:
    """Breakpoints, states (None first, then values in sorted order) and the segment-by-state evidence matrix."""
    ordered = sorted(assertions, key=Assertion.canonical_key)
    pieces = [p for a in ordered for p in _pieces(a, params)]
    cuts = sorted({x for lo, hi, _, _ in pieces for x in (lo, hi)})
    states: list[str | None] = [None, *sorted({v for _, _, v, _ in pieces})]
    index = {v: i for i, v in enumerate(states)}
    lengths = np.diff(np.array(cuts, dtype=float))
    evidence = np.zeros((max(len(cuts) - 1, 0), len(states)))
    for lo, hi, v, d in pieces:
        i0, i1 = np.searchsorted(cuts, lo), np.searchsorted(cuts, hi)
        evidence[i0:i1, index[v]] += d * lengths[i0:i1]
    return cuts, states, evidence


def reconcile(assertions: list[Assertion], params: Params = Params()) -> Timeline:
    """The timeline maximising the objective, as a function of the assertion set alone."""
    cuts, states, evidence = _evidence(assertions, params)
    if len(cuts) < 2:
        return Timeline([])
    path = _viterbi(evidence, params.switch_penalty)
    return Timeline(_merge(cuts, [states[i] for i in path]))


def objective(assertions: list[Assertion], timeline: Timeline, params: Params = Params()) -> float:
    """S(f): evidence collected by the timeline minus lambda per switch between consecutive segments."""
    cuts, states, evidence = _evidence(assertions, params)
    total = 0.0
    for j in range(len(cuts) - 1):
        mid = 0.5 * (cuts[j] + cuts[j + 1])
        v = timeline.value_at(mid)
        if v in states:
            total += evidence[j, states.index(v)]
    switches = sum(1 for a, b in zip(timeline.segments, timeline.segments[1:]) if a.value != b.value)
    return total - params.switch_penalty * switches


def _viterbi(evidence: np.ndarray, lam: float) -> list[int]:
    n, k = evidence.shape
    score = evidence[0].copy()
    back = np.zeros((n, k), dtype=int)
    for j in range(1, n):
        best = int(np.argmax(score))
        switch = score[best] - lam
        stay = score >= switch
        back[j] = np.where(stay, np.arange(k), best)
        score = np.where(stay, score, switch) + evidence[j]
    path = [int(np.argmax(score))]
    for j in range(n - 1, 0, -1):
        path.append(int(back[j, path[-1]]))
    return path[::-1]


def _merge(cuts: list[float], labels: list[str | None]) -> list[Segment]:
    out: list[Segment] = []
    for j, v in enumerate(labels):
        if out and out[-1].value == v:
            out[-1] = Segment(out[-1].start, cuts[j + 1], v)
        else:
            out.append(Segment(cuts[j], cuts[j + 1], v))
    return out
