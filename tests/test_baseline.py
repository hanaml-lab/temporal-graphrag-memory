"""Seam: ledger.WriteTimeInvalidation, the arrival-ordered baseline (Zep/Graphiti-style edge invalidation)."""

import math
import random

from ledger import Assertion, FuzzyInterval, Modality, Params, WriteTimeInvalidation, reconcile

SLOT = ("user", "lives_in")
BERLIN = Assertion(SLOT, "Berlin", Modality.CURRENT, stmt_time=30, tx_time=30, valid=FuzzyInterval.exact(0, 30), source="A")
LISBON = Assertion(SLOT, "Lisbon", Modality.CURRENT, stmt_time=150, tx_time=150, valid=FuzzyInterval.exact(60, 150), source="B")
REMINISCENCE = Assertion(SLOT, "Berlin", Modality.RETROSPECTIVE, stmt_time=210, tx_time=210, source="C")


def current(order):
    store = WriteTimeInvalidation()
    for a in order:
        store.ingest(a)
    return store.timeline(SLOT).value_at(220)


def test_write_time_invalidation_depends_on_arrival_order():
    assert current([BERLIN, LISBON, REMINISCENCE]) == "Berlin"  # the reminiscence reopens Berlin
    assert current([BERLIN, REMINISCENCE, LISBON]) == "Lisbon"  # the reminiscence is absorbed as a duplicate


def test_reconciliation_gives_one_answer_for_every_order():
    for order in ([BERLIN, LISBON, REMINISCENCE], [BERLIN, REMINISCENCE, LISBON]):
        assert reconcile(order, Params()).value_at(220) == "Lisbon"


def test_closing_an_edge_is_recorded_in_transaction_time():
    store = WriteTimeInvalidation()
    store.ingest(BERLIN)
    store.ingest(LISBON)
    assert store.timeline(SLOT, as_of=100).value_at(90) == "Berlin"  # before Lisbon was learned
    assert store.timeline(SLOT, as_of=200).value_at(90) == "Lisbon"


def test_latest_start_wins_is_the_infinite_weight_limit_of_reconciliation():
    rng = random.Random(7)
    horizon = 1000.0
    for _ in range(20):
        starts = sorted(rng.sample(range(0, 500), 5))
        values = [f"v{i}" for i in range(5)]
        rng.shuffle(values)
        claims = [(s, v) for s, v in zip(starts, values)]
        beta = 5.0  # weight exp(beta * rank of start): a later start outweighs all earlier claims combined
        ledger_in = [
            Assertion.explicit(SLOT, v, s, horizon, stmt_time=s, tx_time=s, confidence=math.exp(beta * (rank - 4)))
            for rank, (s, v) in enumerate(claims)
        ]
        tl = reconcile(ledger_in, Params(switch_penalty=0.0, recency_per_year=0.0))
        wti = WriteTimeInvalidation()
        for s, v in claims:
            wti.ingest(Assertion.explicit(SLOT, v, s, horizon, stmt_time=s, tx_time=s))
        for t in range(0, 1000, 7):
            assert tl.value_at(t) == wti.timeline(SLOT).value_at(t)
