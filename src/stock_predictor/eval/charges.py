"""What a round trip costs beyond its price.

A port of stock-analyzer's ``GrowwChargeModel`` / ``ChargeRates`` with the same
default schedule. On Indian equity intraday these fees are a large fraction of
a small move, so **every metric in this project is computed net of them**
(design section 6). A round trip costs roughly 0.05-0.1% of turnover, which
erases most windows under half a percent.

All percentages here are *percent*, not fractions: ``0.05`` means 0.05%.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChargeRates:
    brokerage_intraday_pct: float = 0.05
    brokerage_intraday_max: float = 20.0     # rupees per fill
    stt_intraday_sell_pct: float = 0.025
    exchange_txn_pct: float = 0.00297
    sebi_pct: float = 0.0001
    stamp_duty_buy_pct: float = 0.003
    gst_pct: float = 18.0


@dataclass(frozen=True)
class Charges:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp_duty: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange_txn + self.sebi + self.stamp_duty + self.gst


def _pct(base: float, percent: float) -> float:
    return base * percent / 100.0


def _leg(turnover: float, *, selling: bool, rates: ChargeRates) -> Charges:
    brokerage = min(_pct(turnover, rates.brokerage_intraday_pct), rates.brokerage_intraday_max)
    stt = _pct(turnover, rates.stt_intraday_sell_pct) if selling else 0.0
    exchange_txn = _pct(turnover, rates.exchange_txn_pct)
    sebi = _pct(turnover, rates.sebi_pct)
    stamp_duty = 0.0 if selling else _pct(turnover, rates.stamp_duty_buy_pct)
    gst = _pct(brokerage + exchange_txn + sebi, rates.gst_pct)
    return Charges(brokerage, stt, exchange_txn, sebi, stamp_duty, gst)


def round_trip_charges(
    entry_price: float, exit_price: float, quantity: float, rates: ChargeRates | None = None
) -> Charges:
    r = rates or ChargeRates()
    buy = _leg(entry_price * quantity, selling=False, rates=r)
    sell = _leg(exit_price * quantity, selling=True, rates=r)
    return Charges(
        brokerage=buy.brokerage + sell.brokerage,
        stt=buy.stt + sell.stt,
        exchange_txn=buy.exchange_txn + sell.exchange_txn,
        sebi=buy.sebi + sell.sebi,
        stamp_duty=buy.stamp_duty + sell.stamp_duty,
        gst=buy.gst + sell.gst,
    )


def round_trip_cost_pct(
    entry_price: float,
    exit_price: float,
    notional: float = 100_000.0,
    rates: ChargeRates | None = None,
) -> float:
    """Round-trip cost as a percentage of the entry notional.

    ``notional`` sets the position size, which matters only through the flat
    brokerage cap; the default (Rs 1,00,000) is a realistic single-name intraday
    position and yields ~0.075%.
    """
    if entry_price <= 0:
        return 0.0
    quantity = max(notional / entry_price, 1e-9)
    charges = round_trip_charges(entry_price, exit_price, quantity, rates)
    return charges.total / (entry_price * quantity) * 100.0


def net_gain_pct(
    gross_gain_pct: float,
    entry_price: float,
    exit_price: float | None = None,
    notional: float = 100_000.0,
    rates: ChargeRates | None = None,
) -> float:
    """Gross window gain %, less the round-trip cost %."""
    if exit_price is None:
        exit_price = entry_price * (1 + gross_gain_pct / 100.0)
    return gross_gain_pct - round_trip_cost_pct(entry_price, exit_price, notional, rates)
