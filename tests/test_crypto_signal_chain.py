import numpy as np

from src.crypto.signal_chain import beta_neutral_pair, buy_hold, hysteresis, ledger


def test_hysteresis_requires_stronger_entry_than_exit():
    assert hysteresis(.7, 0, .5, .2) == 1
    assert hysteresis(.3, 1, .5, .2) == 1
    assert hysteresis(.1, 1, .5, .2) == 0
    assert hysteresis(-.7, 1, .5, .2) == -1


def test_pair_has_zero_estimated_market_beta_and_unit_gross():
    beta = np.array([.8, 1.4, .6])
    weight = beta_neutral_pair(0, 1, beta)
    assert np.isclose(weight @ beta, 0)
    assert np.isclose(np.abs(weight).sum(), 1)
    assert weight[0] > 0 > weight[1]


def test_ledger_charges_for_drifted_rebalance_and_buyhold_does_not_rebalance():
    price = np.log(np.array([[100., 100.], [110., 100.], [121., 100.]]))
    weight = np.array([[.5, .5], [.5, .5]])
    book = ledger(weight, price, fee_bp_side=10)
    assert np.isclose(book["turnover"][0], 1)
    assert book["turnover"][1] > 0  # prior weights drift toward the winner
    assert np.isclose(book["gross_return"][0], .05)
    assert np.isclose(buy_hold(price)[-1], (1.21 + 1) / 2)
    assert book["net_wealth"][-1] < book["gross_wealth"][-1]
