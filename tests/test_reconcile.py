"""Seam: ledger.reconcile(assertions, params) -> Timeline."""

from ledger import Assertion, Params, reconcile

SLOT = ("user", "lives_in")


def explicit(value, start, end, stmt, tx=None, **kw):
    return Assertion.explicit(SLOT, value, start, end, stmt_time=stmt, tx_time=stmt if tx is None else tx, **kw)


def test_two_adjacent_explicit_claims_give_two_segments():
    tl = reconcile([explicit("Berlin", 0, 100, stmt=50), explicit("Lisbon", 100, 200, stmt=150)], Params())
    assert tl.value_at(10) == "Berlin"
    assert tl.value_at(150) == "Lisbon"
    assert tl.change_points() == [100.0]


def test_the_timeline_does_not_depend_on_arrival_order():
    import itertools

    evidence = [
        explicit("Berlin", 0, 120, stmt=30, tx=400),
        explicit("Lisbon", 90, 300, stmt=200, tx=10),
        explicit("Berlin", 280, 330, stmt=320, tx=50, confidence=0.4),
        explicit("Porto", 300, 420, stmt=410, tx=5),
    ]
    first = reconcile(evidence, Params()).segments
    for perm in itertools.permutations(evidence):
        assert reconcile(list(perm), Params()).segments == first


def test_reconcile_is_the_exact_maximiser_of_the_objective():
    import itertools

    from ledger import Segment, Timeline, objective

    evidence = [
        explicit("A", 0, 10, stmt=5),
        explicit("B", 4, 14, stmt=12, confidence=0.7),
        explicit("A", 12, 20, stmt=18, confidence=0.5),
    ]
    params = Params(switch_penalty=3.0)
    cuts = sorted({x for a in evidence for x in (a.valid.start_lo, a.valid.end_hi)})
    best = max(
        objective(evidence, Timeline([Segment(cuts[j], cuts[j + 1], v) for j, v in enumerate(labels)]), params)
        for labels in itertools.product([None, "A", "B"], repeat=len(cuts) - 1)
    )
    assert abs(objective(evidence, reconcile(evidence, params), params) - best) < 1e-9


def test_the_switch_penalty_suppresses_short_unsupported_flicker():
    evidence = [explicit("Berlin", 0, 100, stmt=50, confidence=0.8), explicit("Paris", 40, 41, stmt=40)]
    assert reconcile(evidence, Params(switch_penalty=0.0)).value_at(40.5) == "Paris"
    assert reconcile(evidence, Params(switch_penalty=6.0)).value_at(40.5) == "Berlin"
