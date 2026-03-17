"""The append-only, versioned assertion ledger.

Assertions are appended in transaction-time order and never edited. After each append the affected slot is
reconciled and the new timeline is recorded as a version stamped with the transaction time, so an as-of query
reads the version that was current then; by construction it equals reconciling the assertions known at that
time. Redaction (a deletion request) is the one operation that rewrites history: it removes every assertion from
a source, rebuilds every version of the affected slots and advances their epoch, and any derived artefact that
carries an older epoch is refused.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

from .reconcile import Params, Timeline, reconcile
from .types import Assertion, Modality, Slot


class StaleArtefact(RuntimeError):
    """A derived artefact predates a redaction that touched its slot."""


class LedgerStore:
    def __init__(self, params: Params = Params()) -> None:
        self.params = params
        self._assertions: dict[Slot, list[Assertion]] = defaultdict(list)
        self._versions: dict[Slot, list[tuple[float, Timeline]]] = defaultdict(list)
        self._epoch: dict[Slot, int] = defaultdict(int)
        self._clock = float("-inf")
        self.redactions: list[tuple[str, list[Slot]]] = []

    # writes
    def append(self, a: Assertion) -> Timeline:
        if a.tx_time < self._clock:
            raise ValueError(f"transaction time {a.tx_time} precedes the ledger clock {self._clock}")
        self._clock = a.tx_time
        self._assertions[a.slot].append(a)
        timeline = reconcile(self._assertions[a.slot], self.params)
        versions = self._versions[a.slot]
        if versions and versions[-1][0] == a.tx_time:
            versions[-1] = (a.tx_time, timeline)
        else:
            versions.append((a.tx_time, timeline))
        return timeline

    def redact(self, source: str) -> list[Slot]:
        """Remove every assertion from `source`, rebuild the history of each affected slot and bump its epoch."""
        touched = [s for s, items in self._assertions.items() if any(a.source == source for a in items)]
        for slot in touched:
            kept = [a for a in self._assertions[slot] if a.source != source]
            self._assertions[slot] = []
            self._versions[slot] = []
            for a in kept:
                self._assertions[slot].append(a)
                tl = reconcile(self._assertions[slot], self.params)
                if self._versions[slot] and self._versions[slot][-1][0] == a.tx_time:
                    self._versions[slot][-1] = (a.tx_time, tl)
                else:
                    self._versions[slot].append((a.tx_time, tl))
            self._epoch[slot] += 1
        self.redactions.append((source, touched))
        return touched

    # reads
    def slots(self) -> list[Slot]:
        return list(self._assertions)

    def assertions(self, slot: Slot) -> list[Assertion]:
        return list(self._assertions.get(slot, []))

    def current(self, slot: Slot) -> Timeline:
        versions = self._versions.get(slot)
        return versions[-1][1] if versions else Timeline([])

    def as_of(self, slot: Slot, tx: float) -> Timeline:
        """The timeline the store held at transaction time tx."""
        versions = self._versions.get(slot, [])
        i = bisect.bisect_right([v[0] for v in versions], tx)
        return versions[i - 1][1] if i else Timeline([])

    def learned_at(self, slot: Slot, value: str) -> float | None:
        """The first transaction time at which the value was asserted (plans and retractions do not count)."""
        times = [
            a.tx_time
            for a in self._assertions.get(slot, [])
            if a.value == value and a.modality not in (Modality.PROSPECTIVE, Modality.RETRACTION)
        ]
        return min(times) if times else None

    def changes(self, slot: Slot, t0: float, t1: float, as_of: float = float("inf")) -> list[tuple[float, str, str]]:
        """(valid time, from, to) for every switch between two values inside [t0, t1)."""
        named = [s for s in self.as_of(slot, as_of).segments if s.value is not None]
        return [(b.start, a.value, b.value) for a, b in zip(named, named[1:]) if a.value != b.value and t0 <= b.start < t1]

    # epochs
    def epoch(self, slot: Slot) -> int:
        return self._epoch[slot]

    def check(self, slot: Slot, epoch: int) -> None:
        if epoch < self._epoch[slot]:
            raise StaleArtefact(f"artefact for {slot} built at epoch {epoch}; the slot is at epoch {self._epoch[slot]}")
