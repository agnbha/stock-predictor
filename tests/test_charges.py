from __future__ import annotations

from stock_predictor.eval.charges import (
    ChargeRates,
    net_gain_pct,
    round_trip_charges,
    round_trip_cost_pct,
)


def test_round_trip_cost_is_in_the_expected_band():
    # design section 6: "A round trip costs roughly 0.05-0.1%"
    cost = round_trip_cost_pct(entry_price=1000.0, exit_price=1005.0, notional=100_000.0)
    assert 0.04 <= cost <= 0.12


def test_stt_only_on_the_sell_leg():
    c = round_trip_charges(100.0, 101.0, quantity=1000, rates=ChargeRates())
    # one leg's worth of STT at 0.025% of ~101_000 turnover
    assert 20.0 < c.stt < 30.0
    # stamp duty only on the buy leg
    assert 2.5 < c.stamp_duty < 3.5


def test_net_gain_is_gross_minus_cost():
    gross = 0.5
    net = net_gain_pct(gross, entry_price=1000.0, exit_price=1005.0)
    assert net < gross
    assert abs((gross - net) - round_trip_cost_pct(1000.0, 1005.0)) < 1e-9


def test_costs_erase_a_small_move():
    # a sub-0.1% window is a loss after costs
    assert net_gain_pct(0.08, entry_price=500.0) < 0
