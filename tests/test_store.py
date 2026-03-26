"""Seam: ledger.LedgerStore — the append-only, versioned store and its bitemporal queries."""

import random

import pytest

from ledger import Assertion, FuzzyInterval, LedgerStore, Modality, Params, StaleArtefact, reconcile

SLOT = ("user", "lives_in")


def said(value, stmt, tx, valid=None, modality=Modality.CURRENT, source=""):
    return Assertion(SLOT, value, modality, stmt_time=stmt, tx_time=tx, valid=valid, source=source or f"ep{tx}")


def test_as_of_returns_what_was_believed_at_a_transaction_time():
    store = LedgerStore()
    store.append(said("Berlin", 30, 30))
    store.append(said("Lisbon", 150, 150, valid=FuzzyInterval.exact(60, 150)))
    assert store.as_of(SLOT, 100).value_at(90) == "Berlin"
    assert store.as_of(SLOT, 200).value_at(90) == "Lisbon"
    assert store.as_of(SLOT, 10).value_at(5) is None


def random_assertions(rng, n=12):
    values = ["Berlin", "Lisbon", "Porto", "Oslo"]
    out = []
    for i in range(n):
        stmt = rng.uniform(0, 600)
        if rng.random() < 0.5:
            start = stmt - rng.uniform(0, 200)
            valid = FuzzyInterval(start - 5, start, stmt, stmt)
        else:
            valid = None
        modality = rng.choice([Modality.CURRENT] * 4 + [Modality.RETROSPECTIVE, Modality.PROSPECTIVE, Modality.REPORTED])
        out.append(Assertion(SLOT, rng.choice(values), modality, stmt_time=stmt, tx_time=0.0, valid=valid, source=f"s{i}"))
    return out


def ingest(assertions, tx_times):
    store = LedgerStore()
    for a, tx in sorted(zip(assertions, tx_times), key=lambda p: p[1]):
        store.append(Assertion(a.slot, a.value, a.modality, a.stmt_time, tx, a.valid, a.confidence, a.source))
    return store


def test_the_final_state_does_not_depend_on_ingest_order():
    rng = random.Random(3)
    for _ in range(10):
        evidence = random_assertions(rng)
        a = ingest(evidence, [a.stmt_time for a in evidence])
        b = ingest(evidence, rng.sample(range(1000, 2000), len(evidence)))
        assert a.current(SLOT).segments == b.current(SLOT).segments


def test_as_of_equals_reconciling_what_was_known_then():
    rng = random.Random(5)
    evidence = random_assertions(rng, 15)
    store = ingest(evidence, [a.stmt_time for a in evidence])
    known = sorted(store.assertions(SLOT), key=lambda a: a.tx_time)
    for t in [0, 50, 200, 333, 450, 599, 10_000]:
        expected = reconcile([a for a in known if a.tx_time <= t], Params())
        assert store.as_of(SLOT, t).segments == expected.segments


def test_ingest_must_follow_the_transaction_clock():
    store = LedgerStore()
    store.append(said("Berlin", 30, 30))
    with pytest.raises(ValueError):
        store.append(said("Lisbon", 20, 20))


def test_learned_at_is_the_first_transaction_that_asserted_the_value():
    store = LedgerStore()
    store.append(said("Porto", 10, 10, modality=Modality.PROSPECTIVE))
    store.append(said("Berlin", 30, 30))
    store.append(said("Porto", 160, 170, valid=FuzzyInterval.exact(140, 160)))
    assert store.learned_at(SLOT, "Porto") == 170
    assert store.learned_at(SLOT, "Berlin") == 30
    assert store.learned_at(SLOT, "Oslo") is None


def test_changes_lists_the_switches_inside_a_window():
    store = LedgerStore()
    store.append(said("Berlin", 30, 30, valid=FuzzyInterval.exact(0, 30)))
    store.append(said("Lisbon", 150, 150, valid=FuzzyInterval.exact(60, 150)))
    assert store.changes(SLOT, 0, 300) == [(60.0, "Berlin", "Lisbon")]
    assert store.changes(SLOT, 100, 300) == []


def test_redaction_removes_a_source_from_every_version_and_refuses_stale_artefacts():
    store = LedgerStore()
    store.append(said("Berlin", 30, 30, source="ep1"))
    store.append(said("Lisbon", 150, 150, valid=FuzzyInterval.exact(60, 150), source="ep2"))
    cached_epoch = store.epoch(SLOT)
    store.check(SLOT, cached_epoch)
    assert store.redact("ep2") == [SLOT]
    for t in [0, 100, 150, 200, 10_000]:
        assert all(s.value != "Lisbon" for s in store.as_of(SLOT, t).segments)
    with pytest.raises(StaleArtefact):
        store.check(SLOT, cached_epoch)
    assert store.learned_at(SLOT, "Lisbon") is None


def test_first_believed_ignores_claims_that_never_won():
    store = LedgerStore()
    store.append(said("Berlin", 30, 30))
    store.append(said("Porto", 40, 40, modality=Modality.REPORTED))  # outvoted by the user's own statement
    store.append(said("Porto", 200, 200, valid=FuzzyInterval.exact(180, 200)))
    assert store.learned_at(SLOT, "Porto") == 40
    assert store.first_believed(SLOT, "Porto") == 200
