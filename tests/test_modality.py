"""Seam: ledger.reconcile — how each modality turns into valid-time evidence."""

from ledger import Assertion, FuzzyInterval, Modality, Params, reconcile

SLOT = ("user", "lives_in")


def said(value, stmt, modality=Modality.CURRENT, valid=None, **kw):
    return Assertion(SLOT, value, modality, stmt_time=stmt, tx_time=stmt, valid=valid, **kw)


def test_an_undated_current_statement_holds_at_its_statement_time_and_persists_for_a_while():
    tl = reconcile([said("Berlin", 100)], Params(persistence_days=365))
    assert tl.value_at(100) == "Berlin"
    assert tl.value_at(300) == "Berlin"
    assert tl.value_at(600) is None


def test_restating_an_old_value_does_not_outvote_a_later_explicit_change():
    since_60 = FuzzyInterval.exact(60, 120)
    tl = reconcile([said("Berlin", 0), said("Berlin", 30), said("Lisbon", 120, valid=since_60)], Params())
    assert tl.value_at(10) == "Berlin"
    assert tl.value_at(90) == "Lisbon"
    assert tl.value_at(200) == "Lisbon"


def test_a_plan_or_hypothetical_contributes_no_valid_time_evidence():
    tl = reconcile([said("Berlin", 0), said("Porto", 50, Modality.PROSPECTIVE)], Params())
    assert tl.value_at(60) == "Berlin"


def test_an_undated_reminiscence_does_not_reopen_an_ended_state():
    evidence = [
        said("Berlin", 30, valid=FuzzyInterval.exact(0, 30)),
        said("Lisbon", 100, valid=FuzzyInterval.exact(60, 100)),
        said("Berlin", 150, Modality.RETROSPECTIVE),
    ]
    assert reconcile(evidence, Params()).value_at(160) == "Lisbon"


def test_reported_evidence_counts_for_less_than_the_users_own_statement():
    tl = reconcile([said("Berlin", 100), said("Porto", 100, Modality.REPORTED)], Params())
    assert tl.value_at(120) == "Berlin"


def test_a_retraction_ends_a_state():
    tl = reconcile([said("Berlin", 0), said("Berlin", 100, Modality.RETRACTION)], Params())
    assert tl.value_at(50) == "Berlin"
    assert tl.value_at(150) is None


def test_confidence_is_high_when_uncontested_and_lower_when_contested():
    alone = reconcile([said("Berlin", 100)], Params())
    contested = reconcile([said("Berlin", 100), said("Porto", 100, Modality.REPORTED)], Params())
    assert alone.confidence_at(150) > 0.9
    assert 0.5 < contested.confidence_at(150) < alone.confidence_at(150)
