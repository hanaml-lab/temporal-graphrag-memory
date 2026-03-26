"""Seam: ledger.toy.run_toy — the CPU toy setting end to end (life scripts, ingest regimes, five question types)."""

import time

from ledger.toy import QUESTION_TYPES, run_toy


def test_the_toy_runs_end_to_end_on_cpu_in_well_under_a_minute():
    t0 = time.time()
    result = run_toy(n_users=12, seed=0)
    assert time.time() - t0 < 60
    assert set(result["accuracy"]) == {"ledger", "wti", "wti+modality"}
    assert set(result["accuracy"]["ledger"]) == set(QUESTION_TYPES) | {"overall"}
    assert result["questions"]["overall"] > 300


def test_the_ledgers_final_state_is_unaffected_by_ingest_order_and_write_time_invalidations_is_not():
    in_order = run_toy(n_users=20, seed=1, regime="in-order")["accuracy"]
    shuffled = run_toy(n_users=20, seed=1, regime="shuffled")["accuracy"]
    for q in ("current", "pit", "change"):  # questions about the final state, not about transaction time
        assert in_order["ledger"][q] == shuffled["ledger"][q]
    assert in_order["wti"]["pit"] - shuffled["wti"]["pit"] > 0.05
    assert in_order["wti+modality"]["change"] - shuffled["wti+modality"]["change"] > 0.05


def test_ledger_answers_as_of_questions_better_than_write_time_invalidation():
    acc = run_toy(n_users=20, seed=2)["accuracy"]
    assert acc["ledger"]["asof"] > acc["wti+modality"]["asof"] > 0
    assert acc["ledger"]["overall"] > acc["wti+modality"]["overall"] > acc["wti"]["overall"]
