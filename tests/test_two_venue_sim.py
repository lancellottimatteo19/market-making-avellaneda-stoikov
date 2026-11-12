from dataclasses import replace

import numpy as np
import pytest

import two_venue_sim as tv

MARKET = tv.Market(horizon=300.0)
MM = tv.MarketMaker()


@pytest.fixture(scope="module")
def price():
    return tv.efficient_price(MARKET, 60, np.random.default_rng(0))


def run(price, market=MARKET, mm=MM, **kw):
    return tv.simulate(market, mm, n_paths=price.shape[1], seed=1, price=price, **kw)


def test_pnl_splits_into_edge_and_inventory(price):
    r = run(price)
    np.testing.assert_allclose(r["pnl"], r["edge"] + r["inventory_pnl"], atol=1e-9)


def test_efficient_price_variance_matches_the_model():
    m = tv.Market(horizon=600.0)
    p = tv.efficient_price(m, 2000, np.random.default_rng(3))
    var = np.var(p[-1] - p[0]) / m.horizon
    assert var == pytest.approx(m.total_variance, rel=0.1)


def test_textbook_case_has_no_arbitrage_and_earns_the_spread(price):
    r = run(price, market=replace(MARKET, arbitrage=False), mm=replace(MM, latency=0.0))
    assert r["stats"]["arbitrage"]["fills"].sum() == 0
    assert r["pnl"].mean() > 0
    assert tv.markout_bp(r, "noise", 10.0) > 0


def test_arbitrageurs_are_adverse_and_latency_makes_them_costly(price):
    slow = run(price, mm=replace(MM, latency=1.0))
    fast = run(price, mm=replace(MM, latency=0.1))
    assert tv.markout_bp(slow, "arbitrage", 10.0) < 0
    assert slow["stats"]["arbitrage"]["fills"].sum() > fast["stats"]["arbitrage"]["fills"].sum()
    assert slow["pnl"].mean() < fast["pnl"].mean()


def test_guard_withdraws_the_exposed_side(price):
    plain = run(price)
    guarded = run(price, mm=replace(MM, guard_bps=3.0, guard_pause=3.0))
    assert guarded["stats"]["arbitrage"]["fills"].sum() < plain["stats"]["arbitrage"]["fills"].sum()


def test_inventory_stays_within_the_limit(price):
    r = run(price, mm=replace(MM, max_inventory=3))
    assert np.abs(r["final_q"]).max() <= 3


def test_trace_records_the_first_path(price):
    r = run(price, trace=True)
    tr = r["trace"]
    assert len(tr["seen"]) == MARKET.steps
    n_fills = r["stats"]["noise"]["fills"][0] + r["stats"]["arbitrage"]["fills"][0]
    assert len(tr["fills"]) == n_fills
