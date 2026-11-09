import numpy as np
import pandas as pd
import pytest

from alpha101.costs import CostModel, apply_costs

NO_SPREAD = {"us": 0.0, "ca": 0.0}


def test_commission_on_5000_is_19_98_bps():
    cm = CostModel(half_spread_bps=NO_SPREAD)
    assert cm.trade_cost_bps("ca", 5000.0) == pytest.approx(19.98)
    assert cm.trade_cost_bps("us", 5000.0) == pytest.approx(19.98)       # fx_mode="none"
    # high-volume-tier commission and the default half-spreads
    assert CostModel(commission=7.0, half_spread_bps=NO_SPREAD).trade_cost_bps("ca", 5000.0) \
        == pytest.approx(14.0)
    assert CostModel().trade_cost_bps("ca", 5000.0) == pytest.approx(19.98 + 8.0)
    assert CostModel().trade_cost_bps("us", 5000.0) == pytest.approx(19.98 + 3.0)


def test_fx_tier_at_75k_is_104_bps():
    # desk tier 0.78% for $25k..$99,999, plus the one-third surcharge: 0.78% * 4/3 = 1.04% per conversion.
    # the broker's own example quotes 145 bps all-in on a $75k trade: that figure also includes the margin of
    # the broker's posted rate over the interbank mid, which this model does not include (104 bps here).
    cm = CostModel()
    assert cm.fx_cost_bps(75_000.0) == pytest.approx(104.0)
    assert cm.fx_cost_bps(24_999.0) == pytest.approx(1.26 * 100 * 4 / 3)
    assert cm.fx_cost_bps(25_000.0) == pytest.approx(104.0)
    assert cm.fx_cost_bps(100_000.0) == pytest.approx(0.46 * 100 * 4 / 3)
    assert cm.fx_cost_bps(250_000.0) == pytest.approx(0.27 * 100 * 4 / 3)
    assert cm.fx_cost_bps(1_000_000.0) == pytest.approx(0.19 * 100 * 4 / 3)


def test_fx_only_on_us_legs_in_desk_tiered_mode():
    cm = CostModel(half_spread_bps=NO_SPREAD, fx_mode="desk_tiered")
    assert cm.trade_cost_bps("us", 75_000.0) == pytest.approx(9.99 / 75_000 * 1e4 + 104.0)
    assert cm.trade_cost_bps("ca", 75_000.0) == pytest.approx(9.99 / 75_000 * 1e4)
    with pytest.raises(ValueError):
        CostModel(fx_mode="flat")


def test_apply_costs_hand_computed():
    d = pd.bdate_range("2015-01-05", periods=3)
    gross = pd.Series([0.01, 0.0, np.nan], index=d)
    trades = pd.DataFrame({"traded": [2.0, 0.0], "short": [1.0, 1.0]}, index=d[:2])
    net = apply_costs(gross, trades, "ca", CostModel())
    # day 0: full flip of one side (sell 1, buy 1) at 19.98 + 8 bps, plus one day of 50 bps/yr borrow
    assert net.iloc[0] == pytest.approx(0.01 - 2.0 * 27.98e-4 - 50e-4 / 252)
    assert net.iloc[1] == pytest.approx(-50e-4 / 252)
    assert np.isnan(net.iloc[2])


def test_invalid_inputs_raise():
    cm = CostModel()
    for bad in (0.0, -5000.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            cm.trade_cost_bps("ca", bad)
        with pytest.raises(ValueError):
            cm.fx_cost_bps(bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CostModel(notional_per_name=bad)
    for bad in (-0.01, -9.99):
        with pytest.raises(ValueError, match="commission"):
            CostModel(commission=bad)
        with pytest.raises(ValueError, match="half_spread_bps"):
            CostModel(half_spread_bps={"us": bad, "ca": 8.0})
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="commission"):
            CostModel(commission=bad)
        with pytest.raises(ValueError, match="half_spread_bps"):
            CostModel(half_spread_bps={"us": bad, "ca": 8.0})
        with pytest.raises(ValueError, match="half_spread_bps"):
            CostModel(half_spread_bps={"us": 3.0, "ca": bad})
    assert CostModel(commission=0.0, half_spread_bps={"us": 0.0, "ca": 0.0}).trade_cost_bps("us", 1.0) == 0.0
    with pytest.raises(ValueError, match="'ca', 'us'|valid markets"):
        cm.trade_cost_bps("US", 5000.0)
