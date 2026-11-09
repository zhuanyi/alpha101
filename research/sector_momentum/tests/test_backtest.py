"""Self-tests for the sector-momentum backtest (synthetic data only; no network needed).

Run:  pytest research/sector_momentum/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_sector_momentum as bt  # noqa: E402

TICKERS = ["SPY", "XLE", "XLK", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM", "USO", "XLV", "XLF"]


def make_market(n=700, seed=0, drifts=None) -> bt.Market:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n)
    k = len(TICKERS)
    mu = rng.normal(0.0003, 0.0004, k) if drifts is None else np.asarray(drifts)
    ret = rng.normal(mu, 0.012, size=(n, k))
    P = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=dates, columns=TICKERS)
    O = P.shift(1) * (1 + rng.normal(0, 0.002, size=(n, k)))     # open ~ previous close
    fx = pd.Series(1.30 + 0.05 * np.sin(np.arange(n) / 50), index=dates)
    mkt = bt.Market(P, O, fx)
    for name in ("P", "O", "Pv"):                    # pandas 3 hands out read-only views
        setattr(mkt, name, getattr(mkt, name).copy())
    return mkt


def run(mkt, **kw):
    kw.setdefault("start_idx", 150)
    kw.setdefault("end_idx", len(mkt.dates) - 1)
    return bt.simulate(mkt, 250_000, **kw)


def test_signal_date_precedes_trade_date():
    r = run(make_market(), record=True, solver="fast")
    orders = r["orders"]
    assert len(orders) > 10
    assert (orders["signal_date"] < orders["trade_date"]).all()


def test_future_prices_cannot_change_past_nav():
    """Causality: perturbing every price/open after day T must leave NAV up to and including T unchanged."""
    base = make_market(seed=3)
    T = 450
    alt = make_market(seed=3)
    rng = np.random.default_rng(99)
    for arr in (alt.P, alt.O, alt.Pv):
        arr[T + 1:] = arr[T + 1:] * rng.uniform(0.5, 1.5, size=arr[T + 1:].shape)
    alt._mom = {}
    r0, r1 = run(base, solver="fast"), run(alt, solver="fast")
    d = T - 150
    np.testing.assert_allclose(r0["nav_usd"][:d + 1], r1["nav_usd"][:d + 1], rtol=0, atol=1e-9)
    assert not np.allclose(r0["nav_usd"][d + 1:], r1["nav_usd"][d + 1:])


def test_shifting_signal_prices_one_day_changes_results():
    mkt = make_market(seed=5)
    base = run(mkt, solver="fast")
    shifted = np.vstack([np.full((1, mkt.P.shape[1]), np.nan), mkt.P[:-1]])   # signal sees yesterday's prices
    lagged = run(mkt, solver="fast", signal_prices=shifted)
    assert not np.allclose(base["nav_usd"], lagged["nav_usd"])


def test_known_breakout_is_bought_only_after_it_happens():
    mkt = make_market(seed=7, drifts=np.zeros(len(TICKERS)))
    j, jump_day = TICKERS.index("GLD"), 400
    path = 100 * 0.998 ** np.arange(len(mkt.dates))          # steady decline: never a top-5 momentum name
    path[jump_day:] *= 4.0                                   # +300% overnight shock on day 400
    for name in ("P", "Pv"):
        getattr(mkt, name)[:, j] = path
    mkt.O[:, j] = path
    mkt._mom = {}
    r = run(mkt, record=True, solver="fast", weighting="ew")
    gld = r["orders"][(r["orders"].ticker == "GLD") & (r["orders"].shares > 0)]
    assert len(gld) > 0                                      # it is eventually bought ...
    assert gld["trade_date"].min() > mkt.dates[jump_day]     # ... but only after the shock is visible at a close


def test_costs_accounting_and_pnl_reconcile():
    mkt = make_market(seed=11)
    net = run(mkt, solver="fast", record=True)
    gross = run(mkt, solver="fast", costs=False)
    exact = (net["norders"] * bt.COMMISSION_CAD / mkt.fx[150:]).sum()       # each order at its own day's USDCAD
    assert net["comm"].sum() == pytest.approx(exact, rel=1e-12)
    assert net["norders"].sum() > 20
    assert net["nav_usd"][-1] < gross["nav_usd"][-1]
    pnl = net["pnl"].sum()
    assert pnl - net["comm"].sum() == pytest.approx(net["nav_usd"][-1] - net["nav_usd"][0], abs=1e-6)


def test_spy_buy_hold_matches_direct_series_when_costless():
    mkt = make_market(seed=2)
    r = bt.simulate(mkt, 250_000, start_idx=100, end_idx=len(mkt.dates) - 1, mode="fixed", fixed_ticker="SPY",
                    costs=False, holdings_trigger=False, drift_trigger=False, monthly_trigger=False)
    j = mkt.tickers.index("SPY")
    sh = np.floor(250_000 / mkt.fx[100] / mkt.Pv[100, j]).astype(int)   # first trade fills day 101 open
    nav = r["nav_usd"]
    fill = mkt.O[101, j]
    shares = int(250_000 / mkt.fx[100] / mkt.Pv[100, j])
    cash = 250_000 / mkt.fx[100] - shares * fill
    assert nav[-1] == pytest.approx(cash + shares * mkt.Pv[-1, j], rel=1e-9)
    assert sh >= shares


def test_active_set_solver_matches_slsqp():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.choice([3, 5, 8]))
        A = rng.normal(size=(n, n + 20)) * rng.uniform(0.005, 0.02, size=(n, 1))
        cov = np.cov(A)
        w = bt.qp_active_set(cov)
        ref = bt.qp_slsqp(cov)
        assert w is not None
        assert w @ cov @ w <= ref @ cov @ ref * (1 + 1e-6)
        assert w.sum() == pytest.approx(1) and w.min() >= bt.W_MIN - 1e-9 and w.max() <= bt.W_MAX + 1e-9


def test_cvxpy_reference_qp_respects_bounds():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(4)
    cov = np.cov(rng.normal(size=(5, 200)) * 0.01)
    w = bt.reference_qp_cvxpy(cov, 5)
    assert w.sum() == pytest.approx(1) and w.min() >= 0.05 - 1e-4 and w.max() <= 0.40 + 1e-4


def test_momentum_lag_matches_iloc():
    """Original rule: prices.iloc[-1] / prices.iloc[-126] - 1, i.e. 125 rows back."""
    mkt = make_market(n=300)
    M = mkt.momentum(126)
    i = 200
    assert M[i, 0] == pytest.approx(mkt.P[i, 0] / mkt.P[i - 125, 0] - 1)
    assert np.isnan(M[124, 0]) and np.isfinite(M[125, 0])


def test_newey_west_t_sane():
    rng = np.random.default_rng(0)
    m, t = bt.newey_west_t(rng.normal(0.01, 0.02, 400))
    assert 0.007 < m < 0.013 and t > 5
    _, t0 = bt.newey_west_t(rng.normal(0, 0.02, 400))
    assert abs(t0) < 3


def test_min_trade_usd_override_blocks_all_target_orders_and_scaled_default_trades():
    mkt = make_market(seed=13)
    scaled = run(mkt, solver="fast")
    huge = run(mkt, solver="fast", min_trade_usd=1e12)       # absolute floor above any possible trade
    assert scaled["norders"].sum() > 10
    assert huge["norders"].sum() == 0                        # nothing is ever bought, so nothing is held
    small = run(mkt, solver="fast", min_trade_usd=0.0)
    assert small["norders"].sum() >= scaled["norders"].sum() # a lower floor never trades less


def test_signal_prices_shape_is_validated():
    mkt = make_market(seed=5)
    with pytest.raises(AssertionError):
        run(mkt, solver="fast", signal_prices=mkt.P[:-1])


def test_placebo_gross_is_cost_free_and_reproducible():
    mkt = make_market(seed=21)
    a = bt.simulate(mkt, 250_000, start_idx=150, end_idx=len(mkt.dates) - 1, mode="random", seed=3, solver="fast", costs=False)
    b = bt.simulate(mkt, 250_000, start_idx=150, end_idx=len(mkt.dates) - 1, mode="random", seed=3, solver="fast", costs=False)
    assert a["comm"].sum() == 0 and a["spread"].sum() == 0
    np.testing.assert_array_equal(a["nav_usd"], b["nav_usd"])
