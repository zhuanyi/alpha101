"""Trading-cost model for the retail broker (docs/design.md section 6).

All parameters live in one frozen dataclass so a run can print and store them. Costs are expressed
in basis points of the traded value and charged on the return series of a book whose returns are
per unit of capital per side (a long-short book is long 1 and short 1).

FX (fx_mode="desk_tiered"): a USD trade in a CAD account converts at the desk's rate plus a tiered spread
(1.26% below $25k, 0.78% to $99,999, 0.46% to $249,999, 0.27% to $999,999, 0.19% from $1M), plus
a surcharge of one third of the spread (desk FX pricing page, verified 2026-10-18). One conversion per trade: a buy converts CAD->USD, a sell USD->CAD. The broker's posted rate also
carries a margin over the interbank mid that is not modelled here (see tests/test_costs.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import pandas as pd

# (upper bound of the trade value, exclusive; spread in percent)
FX_DESK_TIERS_PCT = ((25_000.0, 1.26), (100_000.0, 0.78), (250_000.0, 0.46), (1_000_000.0, 0.27),
               (float("inf"), 0.19))
FX_DESK_SURCHARGE = 1.0 / 3.0
TRADING_DAYS = 252


def _check_trade_value(trade_value: float) -> None:
    if not (math.isfinite(trade_value) and trade_value > 0):
        raise ValueError(f"trade value must be a positive finite amount, got {trade_value!r}")


@dataclass(frozen=True)
class CostModel:
    commission: float = 9.99           # per trade, CAD; 7.00 for a high-volume tier
    notional_per_name: float = 5000.0  # assumed ticket size for commission; charged pro rata per unit traded
    half_spread_bps: dict = field(default_factory=lambda: {"us": 3.0, "ca": 8.0})
    borrow_bps_per_year: float = 50.0
    fx_mode: str = "none"              # "none" | "desk_tiered" (US legs in a CAD account)

    def __post_init__(self):
        if not (math.isfinite(self.notional_per_name) and self.notional_per_name > 0):
            raise ValueError(f"notional_per_name must be positive and finite, got {self.notional_per_name!r}")
        if not (math.isfinite(self.commission) and self.commission >= 0):
            raise ValueError(f"commission must be non-negative and finite, got {self.commission!r}")
        for market, bps in self.half_spread_bps.items():
            if not (math.isfinite(bps) and bps >= 0):
                raise ValueError(f"half_spread_bps[{market!r}] must be non-negative and finite, got {bps!r}")
        if self.fx_mode not in ("none", "desk_tiered"):
            raise ValueError("fx_mode must be 'none' or 'desk_tiered'")

    def fx_cost_bps(self, trade_value: float) -> float:
        """desk tiered FX spread plus the one-third surcharge, in bps of `trade_value`, per conversion."""
        _check_trade_value(trade_value)
        pct = next(p for upper, p in FX_DESK_TIERS_PCT if trade_value < upper)
        return pct * 100.0 * (1.0 + FX_DESK_SURCHARGE)

    def trade_cost_bps(self, market: str, trade_value: float) -> float:
        """One-way cost of one trade of `trade_value` dollars: commission + half-spread (+ FX on US)."""
        _check_trade_value(trade_value)
        if market not in self.half_spread_bps:
            raise ValueError(f"unknown market {market!r}; valid markets: {sorted(self.half_spread_bps)}")
        bps = self.commission / trade_value * 1e4 + self.half_spread_bps[market]
        if self.fx_mode == "desk_tiered" and market == "us":
            bps += self.fx_cost_bps(trade_value)
        return bps


def apply_costs(gross_daily: pd.Series, trades: pd.DataFrame, market: str, cm: CostModel) -> pd.Series:
    """Net daily returns = gross - trading costs - short borrow.

    `trades` is indexed by date (the signal date whose book the trade opens, so the cost is charged on
    that book's first daily return) with columns:
      traded  sum |w_new - w_drifted| over names, in units of per-side capital (a full flip of one
              side is 2.0: sell 1, buy 1)
      short   short weight held that day, in units of per-side capital (1.0 for a long-short book)
    Every trade is priced as a ticket of `cm.notional_per_name` dollars, so the commission is charged
    pro rata to the traded value. Missing dates cost nothing; a NaN gross return stays NaN.
    """
    t = trades.reindex(gross_daily.index).fillna(0.0)
    cost = t["traded"] * cm.trade_cost_bps(market, cm.notional_per_name) / 1e4 \
        + t["short"] * cm.borrow_bps_per_year / 1e4 / TRADING_DAYS
    return gross_daily - cost
