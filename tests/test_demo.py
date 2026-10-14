from ledger.demo import HISTORY, order_invariance


def test_one_history_gives_one_timeline_in_every_arrival_order():
    out = order_invariance(30)
    assert len(HISTORY) == 20
    assert out["ledger"]["timelines"] == 1
    assert out["ledger"]["worst"] == out["ledger"]["mean"] == 1.0
    # write-time invalidation ends with a different timeline for many orders, and is wrong more often
    assert out["wti"]["timelines"] > 10 and out["wti"]["mean"] < 0.9
