"""Assertions: the unit of evidence the ledger stores.

Times are floats in days (any fixed origin). Valid time is when a fact held in the world; statement time is
when the evidence was produced (the conversation turn's date); transaction time is when the store ingested it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

INF = math.inf
Slot = tuple[str, str]


class Modality(str, Enum):
    """How an assertion relates its value to valid time."""

    CURRENT = "current"  # holds at statement time ("I live in Lisbon")
    RETROSPECTIVE = "retrospective"  # held before statement time and has ended ("back when I lived in Berlin")
    PROSPECTIVE = "prospective"  # a plan or hypothetical: no valid-time evidence ("I might move to Porto")
    REPORTED = "reported"  # attributed to someone else ("my sister says I'm moving")
    RETRACTION = "retraction"  # the value does not (or no longer) hold ("I don't work there any more")


@dataclass(frozen=True)
class FuzzyInterval:
    """A valid-time interval with uncertain endpoints.

    Membership is 1 on the core [start_hi, end_lo), 1/2 on the uncertainty windows [start_lo, start_hi) and
    [end_lo, end_hi), and 0 elsewhere. Exact endpoints have zero-width windows.
    """

    start_lo: float
    start_hi: float
    end_lo: float
    end_hi: float

    def __post_init__(self) -> None:
        if not (self.start_lo <= self.start_hi <= self.end_lo <= self.end_hi):
            raise ValueError(f"interval endpoints out of order: {self}")

    @classmethod
    def exact(cls, start: float, end: float) -> "FuzzyInterval":
        return cls(start, start, end, end)

    def pieces(self) -> list[tuple[float, float, float]]:
        """(a, b, membership) pieces with a < b."""
        out = [(self.start_lo, self.start_hi, 0.5), (self.start_hi, self.end_lo, 1.0), (self.end_lo, self.end_hi, 0.5)]
        return [(a, b, m) for a, b, m in out if b > a]


@dataclass(frozen=True)
class Assertion:
    """One modality-tagged claim about a slot (subject, predicate), extracted from one episode."""

    slot: Slot
    value: str
    modality: Modality
    stmt_time: float
    tx_time: float
    valid: FuzzyInterval | None = None
    confidence: float = 1.0
    source: str = ""
    id: str = ""

    @classmethod
    def explicit(cls, slot: Slot, value: str, start: float, end: float, *, stmt_time: float, tx_time: float, **kw) -> "Assertion":
        """An assertion with an explicitly stated valid interval [start, end)."""
        modality = kw.pop("modality", Modality.CURRENT if end > stmt_time else Modality.RETROSPECTIVE)
        return cls(slot, value, modality, stmt_time, tx_time, FuzzyInterval.exact(start, end), **kw)

    def canonical_key(self) -> tuple:
        """A sort key that ignores transaction time, so evidence is combined in an arrival-independent order."""
        v = self.valid
        iv = (v.start_lo, v.start_hi, v.end_lo, v.end_hi) if v else ()
        return (self.stmt_time, self.value, self.modality.value, iv, self.confidence, self.source, self.id)
