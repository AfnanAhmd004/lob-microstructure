import numpy as np
import pytest

from lob import Market, OrderBook, almgren_chriss_schedule, collect, replay, twap_schedule, vwap_schedule


def test_price_time_priority_and_partial_fills():
    b = OrderBook()
    b.add_limit(1, "S", 101, 5)
    b.add_limit(2, "S", 101, 5)
    b.add_limit(3, "S", 102, 5)
    tr = b.market("B", 7)
    assert [(t.maker_id, t.qty, t.price) for t in tr] == [(1, 5, 101), (2, 2, 101)]
    assert b.level_qty("S", 101) == 3 and b.best_ask() == 101


def test_marketable_limit_rests_remainder():
    b = OrderBook()
    b.add_limit(1, "S", 100, 3)
    trades = b.add_limit(2, "B", 101, 5)
    assert sum(t.qty for t in trades) == 3 and b.best_bid() == 101 and b.level_qty("B", 101) == 2


def test_cancel():
    b = OrderBook()
    b.add_limit(1, "B", 99, 10)
    assert b.cancel(1, 4) == 4 and b.level_qty("B", 99) == 6
    assert b.cancel(1) == 6 and b.best_bid() is None


def test_replay_reconstructs_live_book():
    m = Market(seed=3)
    m.run_until(120)
    r = replay(m.messages)
    assert r.depth("B", 20) == m.book.depth("B", 20) and r.depth("S", 20) == m.book.depth("S", 20)


def test_ofi_explains_price_moves():
    df = collect(Market(seed=4), duration=900)
    d = df.dropna()
    assert np.corrcoef(d["ofi"], d["dmid"])[0, 1] > 0.3


def test_schedules_sum_to_total():
    assert twap_schedule(101, 10).sum() == 101
    assert vwap_schedule(100, [1, 2, 3, 4]).sum() == 100
    ac = almgren_chriss_schedule(500, 10, 1e-2, 3.0, 0.05)
    assert ac.sum() == 500 and ac[0] > ac[-1]  # risk-averse: front-loaded
