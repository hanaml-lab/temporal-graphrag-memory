"""Exact reconciliation of one slot's valid-time timeline from a set of assertions.

Every assertion contributes a piecewise-constant evidence density over valid time for its value. Explicit
evidence (stated intervals, the moment a current statement is made, retractions) is summed across assertions;
default persistence after a current statement is combined by a maximum, so restating an old value many times
does not outvote one explicit change. The timeline is the piecewise-constant function f maximising

    S(f) = integral of e_{f(t)}(t) dt  -  lambda * #switches(f),

solved exactly by Viterbi over the elementary segments between evidence breakpoints. Because the evidence is
constant inside a segment, an optimal switch can always be placed on a breakpoint, so the restriction is exact.
The evidence never reads transaction time, and assertions are combined in a canonical order, so the result is a
function of the assertion set alone (order invariance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import Assertion, Modality

PERSIST = "persist"
EXPLICIT = "explicit"


@dataclass(frozen=True)
class Params:
    switch_penalty: float = 6.0  # lambda, in weight-days
    persistence_weight: float = 0.5  # rho: weight of default persistence relative to explicit evidence
    persistence_days: float = 365.0  # H_p: how long an open-ended current statement is presumed to hold
    lookback_days: float = 30.0  # half-membership support before an undated current statement
    reported_weight: float = 0.6  # pi(reported)
    recency_per_year: float = 0.3  # eta: statement-time recency (order-invariant: statement time is evidence)
    temperature: float = 2.0  # T: posterior temperature for segment confidences


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

    def confidence_at(self, t: float) -> float:
        s = self._find(t)
        return s.confidence if s else 0.0

    def last_known(self, t: float) -> tuple[str | None, float | None]:
        """The latest non-empty value at or before t, and the end of its segment (None, None if there is none)."""
        best: tuple[str | None, float | None] = (None, None)
        for s in self.segments:
            if s.start > t:
                break
            if s.value is not None:
                best = (s.value, s.end)
        return best

    def change_points(self) -> list[float]:
        """Valid times at which the slot switches from one value to another."""
        named = [s for s in self.segments if s.value is not None]
        return [b.start for a, b in zip(named, named[1:]) if a.value != b.value]


def weight(a: Assertion, params: Params, t_ref: float) -> float:
    """w(a) = confidence * pi(modality) * exp(eta * years since the slot's first statement)."""
    pi = {Modality.REPORTED: params.reported_weight, Modality.PROSPECTIVE: 0.0}.get(a.modality, 1.0)
    return a.confidence * pi * math.exp(params.recency_per_year * max(0.0, a.stmt_time - t_ref) / 365.0)


def _pieces(a: Assertion, params: Params, t_ref: float) -> list[tuple[float, float, str, float, str]]:
    """(start, end, value, density, kind) evidence pieces for one assertion."""
    w = weight(a, params, t_ref)
    if w == 0.0:
        return []
    sign = -1.0 if a.modality is Modality.RETRACTION else 1.0
    out: list[tuple[float, float, str, float, str]] = []
    if a.valid is not None:
        out += [(lo, hi, a.value, sign * w * m, EXPLICIT) for lo, hi, m in a.valid.pieces()]
        open_ended = a.valid.end_hi <= a.stmt_time + 1.0
    elif a.modality is Modality.RETROSPECTIVE:
        return []  # ended before the statement, at an unknown time: provenance only, no valid-time evidence
    elif a.modality is Modality.RETRACTION:
        return [(a.stmt_time, a.stmt_time + params.persistence_days, a.value, -w, EXPLICIT)]
    else:
        out += [
            (a.stmt_time - params.lookback_days, a.stmt_time, a.value, 0.5 * w, EXPLICIT),
            (a.stmt_time, a.stmt_time + 1.0, a.value, w, EXPLICIT),
        ]
        open_ended = True
    if open_ended and sign > 0 and a.modality in (Modality.CURRENT, Modality.REPORTED):
        start = max(a.stmt_time + 1.0, a.valid.end_hi if a.valid else a.stmt_time + 1.0)
        out.append((start, a.stmt_time + params.persistence_days, a.value, params.persistence_weight * w, PERSIST))
    return [p for p in out if p[1] > p[0]]


def _evidence(assertions: list[Assertion], params: Params) -> tuple[list[float], list[str | None], np.ndarray]:
    """Breakpoints, states (None first, then values in sorted order) and the segment-by-state evidence matrix."""
    ordered = sorted(assertions, key=Assertion.canonical_key)
    t_ref = min((a.stmt_time for a in ordered), default=0.0)
    pieces = [p for a in ordered for p in _pieces(a, params, t_ref)]
    cuts = sorted({x for lo, hi, *_ in pieces for x in (lo, hi)})
    states: list[str | None] = [None, *sorted({v for _, _, v, _, _ in pieces})]
    index = {v: i for i, v in enumerate(states)}
    lengths = np.diff(np.array(cuts, dtype=float))
    n, k = max(len(cuts) - 1, 0), len(states)
    explicit, persist = np.zeros((n, k)), np.zeros((n, k))
    for lo, hi, v, d, kind in pieces:
        i0, i1 = np.searchsorted(cuts, lo), np.searchsorted(cuts, hi)
        if kind == PERSIST:
            persist[i0:i1, index[v]] = np.maximum(persist[i0:i1, index[v]], d)
        else:
            explicit[i0:i1, index[v]] += d
    return cuts, states, (explicit + persist) * lengths[:, None]


def reconcile(assertions: list[Assertion], params: Params = Params()) -> Timeline:
    """The timeline maximising the objective, with posterior confidences, as a function of the assertion set."""
    cuts, states, evidence = _evidence(assertions, params)
    if len(cuts) < 2:
        return Timeline([])
    path = _viterbi(evidence, params.switch_penalty)
    marg = _marginals(evidence, params.switch_penalty, params.temperature)
    conf = [float(marg[j, s]) for j, s in enumerate(path)]
    return Timeline(_merge(cuts, [states[i] for i in path], conf, np.diff(cuts)))


def objective(assertions: list[Assertion], timeline: Timeline, params: Params = Params()) -> float:
    """S(f): evidence collected by the timeline minus lambda per switch between consecutive segments."""
    cuts, states, evidence = _evidence(assertions, params)
    total = 0.0
    for j in range(len(cuts) - 1):
        v = timeline.value_at(0.5 * (cuts[j] + cuts[j + 1]))
        if v in states:
            total += evidence[j, states.index(v)]
    switches = sum(1 for a, b in zip(timeline.segments, timeline.segments[1:]) if a.value != b.value)
    return total - params.switch_penalty * switches


def _viterbi(evidence: np.ndarray, lam: float) -> list[int]:
    """MAP state path; O(n k) because every switch costs the same lambda."""
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


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def _marginals(evidence: np.ndarray, lam: float, temperature: float) -> np.ndarray:
    """Per-segment state marginals of the Gibbs distribution exp(S(f) / T) (forward-backward)."""
    n, k = evidence.shape
    phi = evidence / temperature
    trans = np.full((k, k), -lam / temperature)
    np.fill_diagonal(trans, 0.0)
    alpha = np.zeros((n, k))
    beta = np.zeros((n, k))
    alpha[0] = phi[0]
    for j in range(1, n):
        alpha[j] = phi[j] + _logsumexp(alpha[j - 1][:, None] + trans, axis=0)
    for j in range(n - 2, -1, -1):
        beta[j] = _logsumexp(trans + (phi[j + 1] + beta[j + 1])[None, :], axis=1)
    post = alpha + beta
    return np.exp(post - _logsumexp(post, axis=1)[:, None])


def _merge(cuts: list[float], labels: list[str | None], conf: list[float], lengths: np.ndarray) -> list[Segment]:
    """Merge runs of equal labels; a merged segment's confidence is the length-weighted mean of its pieces."""
    out: list[Segment] = []
    mass: list[float] = []
    for j, v in enumerate(labels):
        if out and out[-1].value == v:
            prev = out[-1]
            total = mass[-1] + float(lengths[j])
            c = (prev.confidence * mass[-1] + conf[j] * float(lengths[j])) / total
            out[-1], mass[-1] = Segment(prev.start, cuts[j + 1], v, c), total
        else:
            out.append(Segment(cuts[j], cuts[j + 1], v, conf[j]))
            mass.append(float(lengths[j]))
    return out
