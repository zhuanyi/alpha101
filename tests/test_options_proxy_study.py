import importlib.util
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

_spec = importlib.util.spec_from_file_location(
    "ops", Path(__file__).resolve().parents[1] / "scripts" / "options_proxy_study.py")
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)


def _levels(rets, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(rets), freq="ME")
    return pd.Series(100 * np.cumprod(1 + np.asarray(rets)), index=idx)


def test_perf_constant_return():
    p = ops.perf(_levels([0.01] * 24))
    assert p["cagr"] == pytest.approx(1.01 ** 12 - 1)
    assert p["vol"] == pytest.approx(0, abs=1e-12)
    assert p["maxdd"] == 0


def test_perf_maxdd():
    p = ops.perf(_levels([0.1, -0.5, 0.1]))
    assert p["maxdd"] == pytest.approx(-0.5)


def test_newey_west_iid_matches_plain_t_roughly():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.02, 2000)
    t_plain = x.mean() / (x.std(ddof=0) / sqrt(len(x)))
    assert ops.newey_west_t(x) == pytest.approx(t_plain, rel=0.2)


def test_compare_recovers_beta_alpha():
    rng = np.random.default_rng(1)
    rb = rng.normal(0.008, 0.04, 240)
    ri = 0.5 * rb + 0.002
    r = ops.compare(_levels(ri), _levels(rb))
    assert r["beta"] == pytest.approx(0.5, abs=1e-6)
    assert r["alpha_ann"] == pytest.approx(0.024, abs=1e-6)


def test_month_levels_gap_gives_nan_and_no_cross_gap_return():
    s = pd.Series([100, 110, 130], index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-04-30"]))
    m = ops.month_levels(s)
    assert len(m) == 4 and np.isnan(m.iloc[2])
    assert len(m.pct_change(fill_method=None).dropna()) == 1


def test_cost_hand_computed():
    # 100k at C$50: n=20 contracts; monthly = 9.99 + 1.25*20 + 100*20*0.03 = 94.99; x12 / 100k = 113.99 bp
    assert ops.covered_call_cost_bp(100_000, 50, 0.03) == pytest.approx(12 * 94.99 / 100_000 * 1e4)
    assert ops.covered_call_cost_bp(100_000, 50, 0.03, close_frac=1.0) == pytest.approx(
        2 * ops.covered_call_cost_bp(100_000, 50, 0.03))


def test_breakeven_consistent_with_cost():
    s = ops.breakeven_size(200, 50)
    assert ops.covered_call_cost_bp(s, 50) == pytest.approx(200)
    assert ops.breakeven_size(50, 50) is None  # floor is 12*(1.25+3)/50*... = 102 bp > 50


def test_put_strike_has_target_delta():
    sig, T = 0.25, 21 / 252
    X = ops.put_strike_drop(sig)
    K = 1 - X
    d1 = (-np.log(K) + 0.5 * sig ** 2 * T) / (sig * sqrt(T))
    assert norm.cdf(d1) - 1 == pytest.approx(-0.10, abs=1e-6)


def test_episode_starts_nonoverlap():
    c = pd.Series([True] * 50)
    k = ops.episode_starts(c)
    assert list(np.where(k)[0]) == [0, 21, 42]


def test_drawdown_events_flags_and_forward():
    rng = np.random.default_rng(2)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))),
                   index=pd.bdate_range("2020-01-01", periods=600))
    d = ops.drawdown_events(px)
    assert (d[d.cond].dd <= -0.05).all() and (d[d.cond].dd > -0.10).all()
    i = d.index[10]
    assert d.fwd.loc[i] == pytest.approx(px.shift(-21).loc[i] / px.loc[i] - 1)
    assert (d.hit == (d.fwd < -d.X)).all()
