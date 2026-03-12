"""Write-time invalidation: the arrival-ordered baseline that LEDGER replaces.

This follows the edge-resolution rules of temporal knowledge-graph memories such as Zep/Graphiti. Each incoming
assertion is resolved once, against the edges that are still open, at the moment it arrives:

1. Deduplicate: if an open edge already holds the same value, the assertion is merged into it as one more source.
2. Otherwise a new edge [start, end) is created, with start = the stated start or the statement time.
3. Invalidate: an open edge with a different value that starts earlier is closed at the new edge's start; one
   that starts later instead bounds the new edge. Closing is recorded in transaction time and never revisited.

Every step reads only the state left by earlier arrivals, so the result depends on arrival order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .reconcile import Segment, Timeline
from .types import Assertion, Modality, Slot

INF = math.inf


@dataclass
class Edge:
    slot: Slot
    value: str
    valid_from: float
    valid_to: float
    created_at: float  # transaction time the edge was written
    closed_at: float = INF  # transaction time an invalidation closed it
    closed_to: float = INF  # the valid_to it was given when closed
    sources: list[str] = field(default_factory=list)

    def valid_to_as_of(self, tx: float) -> float:
        return self.closed_to if self.closed_at <= tx else self.valid_to


class WriteTimeInvalidation:
    """Bitemporal edge store maintained by arrival-ordered, irreversible invalidation."""

    def __init__(self, use_modality: bool = False) -> None:
        self.use_modality = use_modality
        self.edges: list[Edge] = []

    def _open(self, slot: Slot) -> list[Edge]:
        return [e for e in self.edges if e.slot == slot and e.closed_at == INF]

    def ingest(self, a: Assertion) -> None:
        if self.use_modality and (a.modality is Modality.PROSPECTIVE or (a.modality is Modality.RETROSPECTIVE and a.valid is None)):
            return
        if a.modality is Modality.RETRACTION:
            for e in self._open(a.slot):
                if e.value == a.value and e.valid_from < a.stmt_time:
                    e.closed_at, e.closed_to = a.tx_time, a.stmt_time
            return
        start = a.valid.start_hi if a.valid else a.stmt_time
        end = a.valid.end_lo if a.valid and a.valid.end_lo > a.stmt_time + 1.0 else INF
        if a.valid and a.modality is Modality.RETROSPECTIVE:
            end = a.valid.end_lo
        open_edges = self._open(a.slot)
        for e in open_edges:
            if e.value == a.value:
                e.sources.append(a.source)
                return
        new = Edge(a.slot, a.value, start, end, created_at=a.tx_time, sources=[a.source])
        for e in open_edges:
            if e.valid_from < new.valid_from < e.valid_to:
                e.closed_at, e.closed_to = a.tx_time, new.valid_from
            elif new.valid_from <= e.valid_from < new.valid_to:
                new.valid_to = e.valid_from
        self.edges.append(new)

    def timeline(self, slot: Slot, as_of: float = INF) -> Timeline:
        """The slot's valid-time timeline as recorded at transaction time `as_of` (later edges paint over earlier)."""
        known = [e for e in self.edges if e.slot == slot and e.created_at <= as_of]
        cuts = sorted({x for e in known for x in (e.valid_from, e.valid_to_as_of(as_of)) if x != INF})
        if not known:
            return Timeline([])
        hi = max(cuts) + 1.0 if cuts else 1.0
        cuts = sorted(set(cuts) | {hi})
        segs: list[Segment] = []
        for a, b in zip(cuts, cuts[1:]):
            mid = 0.5 * (a + b)
            v = None
            for e in sorted(known, key=lambda e: e.created_at):
                if e.valid_from <= mid < e.valid_to_as_of(as_of):
                    v = e.value
            if segs and segs[-1].value == v:
                segs[-1] = Segment(segs[-1].start, b, v)
            else:
                segs.append(Segment(a, b, v))
        if segs and segs[-1].value is not None:  # an open edge extends to the end of time
            last_open = any(e.valid_to_as_of(as_of) == INF and e.value == segs[-1].value for e in known)
            if last_open:
                segs[-1] = Segment(segs[-1].start, INF, segs[-1].value)
        return Timeline(segs)
