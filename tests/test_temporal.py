"""Seams: ledger.temporal.resolve / since / during (time expressions) and ledger.tagging.tag_modality (cues)."""

from datetime import date

from ledger import Modality
from ledger.tagging import tag_modality
from ledger.temporal import day, during, resolve, since

STMT = date(2025, 6, 10)


def test_a_month_name_resolves_to_its_most_recent_past_occurrence():
    assert resolve("March", STMT) == (day(date(2025, 3, 1)), day(date(2025, 4, 1)))
    assert resolve("in September", STMT) == (day(date(2024, 9, 1)), day(date(2024, 10, 1)))


def test_a_month_name_in_a_plan_resolves_forward():
    assert resolve("September", STMT, future=True) == (day(date(2025, 9, 1)), day(date(2025, 10, 1)))


def test_relative_expressions_resolve_against_the_statement_date_not_the_processing_date():
    assert resolve("yesterday", STMT) == (day(date(2025, 6, 9)), day(date(2025, 6, 10)))
    assert resolve("last month", STMT) == (day(date(2025, 5, 1)), day(date(2025, 6, 1)))
    lo, hi = resolve("two weeks ago", STMT)
    assert lo < day(date(2025, 5, 27)) < hi and hi - lo <= 7
    assert resolve("in 2023", STMT) == (day(date(2023, 1, 1)), day(date(2024, 1, 1)))


def test_unresolvable_expressions_return_none():
    assert resolve("a while back, I think", STMT) is None


def test_since_makes_a_fuzzy_start_and_ends_at_the_statement():
    iv = since("since March", STMT)
    assert (iv.start_lo, iv.start_hi, iv.end_lo, iv.end_hi) == (day(date(2025, 3, 1)), day(date(2025, 4, 1)), day(STMT), day(STMT))


def test_during_covers_the_stated_period():
    iv = during("from 2019 to 2023", STMT)
    assert (iv.start_lo, iv.start_hi) == (day(date(2019, 1, 1)), day(date(2020, 1, 1)))
    assert (iv.end_lo, iv.end_hi) == (day(date(2023, 1, 1)), day(date(2024, 1, 1)))


def test_modality_cues():
    assert tag_modality("I live in Lisbon now") is Modality.CURRENT
    assert tag_modality("I might move to Porto next year") is Modality.PROSPECTIVE
    assert tag_modality("If I took the job in Oslo, we'd need a flat") is Modality.PROSPECTIVE
    assert tag_modality("Back when I lived in Berlin we cycled everywhere") is Modality.RETROSPECTIVE
    assert tag_modality("I used to work at Brightline") is Modality.RETROSPECTIVE
    assert tag_modality("I don't work at Brightline anymore") is Modality.RETRACTION
    assert tag_modality("My sister says I'm moving to Oslo") is Modality.REPORTED
