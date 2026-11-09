import numpy as np
import pandas as pd
import pytest

from alpha101 import Evaluator, Panel, forward_returns, get_alpha
from alpha101.costs import CostModel, apply_costs
from alpha101.evaluate import (LAG, RunConfig, censor_holdout, composite_backtest, evaluate_alpha,
                               ic_correlation, multiple_testing, turnover)
from alpha101.evaluate import _resolve
from alpha101.stats import newey_west_t, rank_ic
from alpha101.synthetic import random_panel

CFG = RunConfig(markets=("us",))
IS_PERIODS = ("is", "2005-2012", "2013-2018")


def planted_panel(T=700, N=60, seed=0, beta=0.02, noise=0.01):
    """random_panel whose 5-day forward return is beta * z + noise, with the score z stored as `cap`.

    Daily log returns are l[u] = beta * z[u-2] + e[u] with z, e iid normal. forward_returns(h=5, lag=1)
    at t spans l[t+2] .. l[t+6], so log(1 + fwd5[t]) = beta * z[t] + (beta * (z[t+1] + .. + z[t+4])
    + e[t+2] + .. + e[t+6]); the bracket is noise independent of z[t].
    """
    p = random_panel(T=T, N=N, seed=seed)
    rng = np.random.default_rng(seed + 100)
    z = rng.standard_normal((T, N))
    ell = rng.normal(0.0, noise, (T, N))
    ell[2:] += beta * z[:-2]
    fields = dict(p.fields)
    fields["cap"] = z
    fields["adj_close"] = 50.0 * np.exp(np.cumsum(ell, axis=0))
    return Panel(dates=p.dates, symbols=p.symbols, fields=fields, groups=p.groups)


def _values(d: dict) -> np.ndarray:
    return np.array([d[k] for k in sorted(d)], dtype=float)


# ---------------------------------------------------------------------- 1. planted signal
def test_planted_signal_ic_and_decile_spread():
    beta, noise, N = 0.02, 0.01, 60
    res = evaluate_alpha(planted_panel(N=N, beta=beta, noise=noise), "rank(cap)", CFG, None)
    # z[t] and log(1 + fwd5[t]) are jointly normal with rho = beta / sqrt(5 beta^2 + 5 noise^2) = 0.4.
    # Rank IC is invariant to the monotone exp, so its expectation is the Spearman correlation of a
    # bivariate normal, (6 / pi) * arcsin(rho / 2) = 0.385 (Moran's finite-n correction at n = 60 is
    # -0.006, well inside the tolerance).
    rho = beta / np.sqrt(5 * beta ** 2 + 5 * noise ** 2)
    expected = 6 / np.pi * np.arcsin(rho / 2)
    s = res.summary[5]["is"]
    assert s["n_days"] > 600
    assert abs(s["mean_ic"] - expected) < 0.03, (s["mean_ic"], expected)
    ls = res.deciles["long_short"]
    assert ls.mean() > 0
    assert newey_west_t(ls, lags=5) > 5
    # monotone in the bin: top decile beats bottom decile on average
    assert res.deciles["bin_10"].mean() > res.deciles["bin_1"].mean()


# ---------------------------------------------------------------------- 2. null
def test_null_alpha_is_rarely_significant():
    cfg = RunConfig(markets=("us",), horizons=(5,))
    ts = [evaluate_alpha(random_panel(T=500, N=40, seed=s), 101, cfg, None).summary[5]["is"]["t_nw"]
          for s in range(20)]
    assert np.isfinite(ts).all()
    assert sum(abs(t) < 3 for t in ts) >= 19, ts


# ---------------------------------------------------------------------- 3. holdout censoring
def test_censor_holdout_alpha_by_date_and_forward_returns_by_exit():
    d = pd.bdate_range("2018-12-24", periods=10)       # row 6 = 2019-01-01
    df = pd.DataFrame(np.arange(20.0).reshape(10, 2), index=d)
    a = censor_holdout(df, CFG)
    assert a.iloc[:6].notna().all().all() and a.iloc[6:].isna().all().all()
    f = censor_holdout(df, CFG, horizon=2)              # lag 1: row t exits at row t + 3
    assert f.iloc[:3].notna().all().all() and f.iloc[3:].isna().all().all()
    assert d[3 + 3] >= pd.Timestamp(CFG.holdout_start) > d[2 + 3]
    unlocked = RunConfig(unlock_holdout=True)
    assert censor_holdout(df, unlocked) is df           # alpha values: unchanged
    u = censor_holdout(df, unlocked, horizon=2)         # straddle band: rows [6 - 1 - 2, 6) = 3..5
    assert u.iloc[:3].notna().all().all() and u.iloc[3:6].isna().all().all()
    assert u.iloc[6:].equals(df.iloc[6:])
    assert df.notna().all().all()                       # the input is not modified


def _perturbed_after(p: Panel, when: str, seed: int = 9) -> Panel:
    rng = np.random.default_rng(seed)
    rows = np.asarray(p.dates >= pd.Timestamp(when))
    fields = dict(p.fields)
    for k in ("open", "high", "low", "close", "adj_close", "volume"):
        f = np.array(fields[k], copy=True)
        f[rows] *= rng.uniform(0.5, 2.0, size=(rows.sum(), f.shape[1]))
        fields[k] = f
    return Panel(dates=p.dates, symbols=p.symbols, fields=fields, groups=p.groups)


def test_holdout_statistics_do_not_see_post_holdout_prices(tmp_path):
    """Every returned in-sample statistic is bit-identical when post-holdout prices change.

    Run A: prices dated >= holdout_start are perturbed, which is what exit-date censoring must protect:
    a 21-day return from mid-December 2018 exits in 2019 and would leak under signal-date censoring
    (checked below). Perturbing from earlier (holdout_start - 30 days) cannot leave in-sample statistics
    unchanged, since in-sample signals and returns legitimately use those prices. The spec's tightness
    clause (perturb from P = holdout_start - h - lag; the first changed IS IC row is P - lag - h =
    cut - 2 (lag + h)) is not asserted here.
    """
    base = random_panel(T=1100, N=40, seed=3)
    assert base.dates[-1] > pd.Timestamp("2019-03-01")
    pert = _perturbed_after(base, CFG.holdout_start)
    hold = pd.Timestamp(CFG.holdout_start)
    for alpha in (1, 6, 12, 101):
        r0 = evaluate_alpha(base, alpha, CFG, tmp_path / "base")
        r1 = evaluate_alpha(pert, alpha, CFG, tmp_path / "pert")
        a_scored = Evaluator(base, CFG.options).evaluate(_resolve(alpha)[1])
        a_scored.loc[(a_scored.index < pd.Timestamp(CFG.start)) | (a_scored.index >= hold)] = np.nan
        for h in CFG.horizons:
            for per in IS_PERIODS:
                assert np.array_equal(_values(r0.summary[h][per]), _values(r1.summary[h][per]),
                                      equal_nan=True), (alpha, h, per)
            assert r0.summary[h]["2019-2026"]["n_days"] == 0
            assert r0.summary[h]["2013-2018"]["n_days"] > 500
            # every row with an in-sample exit and a finite UNCENSORED IC is kept, nothing else
            ic_h = r0.ic[h]["ic"]
            exit_ok = np.arange(len(base.dates)) + LAG + h < base.dates.searchsorted(hold)
            n_exit_ok = int(rank_ic(a_scored, forward_returns(base, h, execution=CFG.execution, lag=LAG),
                                    min_names=CFG.min_names["us"])["ic"].notna()[
                exit_ok & np.asarray(base.dates >= pd.Timestamp(CFG.start))].sum())
            assert n_exit_ok > 0
            assert r0.summary[h]["is"]["n_days"] == n_exit_ok == int(ic_h.notna().sum()), (alpha, h)
            assert np.array_equal(r0.ic[h].to_numpy(), r1.ic[h].to_numpy(), equal_nan=True)
            assert r0.ic[h]["ic"].loc[hold:].isna().all()
        for x0, x1 in ((r0.deciles, r1.deciles), (r0.turnover, r1.turnover), (r0.trades, r1.trades)):
            assert np.array_equal(x0.to_numpy(), x1.to_numpy(), equal_nan=True), alpha
        assert r0.deciles["long_short"].loc[:"2018-12-24"].notna().sum() > 500
        assert r0.deciles.loc[hold:].isna().all().all()
        v0 = pd.read_parquet(r0.values_path)
        assert (v0.dtypes == np.float32).all()
        assert v0.loc[hold:].isna().all().all() and v0.loc[:"2018-12-31"].notna().any().any()
        assert pd.read_parquet(r1.values_path).equals(v0)

    # power: unlocked, the holdout era sees the perturbation ...
    cfg_u = RunConfig(markets=("us",), unlock_holdout=True)
    u0, u1 = (evaluate_alpha(p, 101, cfg_u, None).summary[5]["2019-2026"] for p in (base, pert))
    assert u0["n_days"] > 30 and u0["mean_ic"] != u1["mean_ic"]
    # ... and censoring forward returns by signal date instead of exit date would leak 2019 prices
    naive = []
    for p in (base, pert):
        a = Evaluator(p).evaluate(get_alpha(101).expression)
        a.loc[a.index >= hold] = np.nan
        f = forward_returns(p, 21)
        f.loc[f.index >= hold] = np.nan
        naive.append(rank_ic(a, f, min_names=30)["ic"].to_numpy())
    cut = int(base.dates.searchsorted(hold))
    early = slice(0, cut - 1 - 21)                      # exits before the holdout: unaffected
    assert np.array_equal(naive[0][early], naive[1][early], equal_nan=True)
    late = slice(cut - 1 - 21, cut)                     # in-sample signal dates whose exit is in 2019
    assert not np.array_equal(naive[0][late], naive[1][late], equal_nan=True)


# ---------------------------------------------------------------------- 4. turnover
def test_turnover_constant_is_zero_and_full_flip_is_one():
    d = pd.bdate_range("2015-01-02", periods=3)
    cols = [f"S{i}" for i in range(20)]
    const = pd.DataFrame(0.0, index=d, columns=cols)
    const.iloc[:, :10] = 0.1
    t = turnover(const)
    assert np.isnan(t.iloc[0]) and (t.iloc[1:] == 0).all()
    assert (turnover(const, drifted=const).iloc[1:] == 0).all()
    flip = const.copy()
    flip.iloc[1, :10] = 0.0
    flip.iloc[1, 10:] = 0.1
    assert turnover(flip).iloc[1] == pytest.approx(1.0)
    # drift: the old book grew to 0.6/0.4 relative weights, the new target is 0.5/0.5 -> 0.1 one-way
    w = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=d[:2])
    drifted = pd.DataFrame([[0.0, 0.0], [0.6, 0.4]], index=d[:2])
    assert turnover(w, drifted).iloc[1] == pytest.approx(0.1)


def test_decile_turnover_and_trades_are_consistent():
    res = evaluate_alpha(planted_panel(T=400), "rank(cap)", CFG, None)
    t = res.turnover.dropna()
    assert len(t) > 300 and (t >= 0).all() and (t <= 2 + 1e-12).all()
    # z is redrawn every day, so the book is almost entirely replaced at every rebalance
    assert t.mean() > 1.6
    assert (res.trades["traded"] >= 0).all()


# ---------------------------------------------------------------------- multiple testing
def test_multiple_testing_counts():
    summary = pd.DataFrame({"t_nw": [0.5, -2.6, 3.1, 4.0]}, index=["a", "b", "c", "d"])
    mt = multiple_testing(summary, pd.DataFrame(np.eye(4), index=summary.index, columns=summary.index))
    assert mt["n_eff"] == pytest.approx(4.0)
    assert mt["thresholds"]["bonferroni_101"] == pytest.approx(3.4834, abs=1e-4)
    assert mt["passing"] == {"raw": 3, "sidak_neff": 3, "bonferroni_101": 1, "hlz": 2}
    # two identical series count once
    d = pd.bdate_range("2018-09-03", periods=120)          # 85 days before 2019-01-01, 35 after
    x = pd.Series(np.random.default_rng(0).standard_normal(120), index=d)
    c = ic_correlation({"a": x, "b": x, "c": -x.shift(1)})
    assert c.loc["a", "b"] == pytest.approx(1.0)
    c2 = ic_correlation({"a": x, "b": x.where(d < "2019-01-01", -x)}, before="2019-01-01")
    assert c2.loc["a", "b"] == pytest.approx(1.0)
    assert ic_correlation({"a": x, "b": x.where(d < "2019-01-01", -x)}).loc["a", "b"] < 0.5   # `before` bites


def test_ic_correlation_min_periods():
    d = pd.bdate_range("2018-01-01", periods=100)
    x = pd.Series(np.random.default_rng(1).standard_normal(100), index=d)
    short = x.where(d >= d[-2])                           # overlaps x on 2 days only
    c = ic_correlation({"a": x, "b": short})
    assert np.isnan(c.loc["a", "b"]) and c.loc["a", "a"] == pytest.approx(1.0)
    assert ic_correlation({"a": x, "b": short}, min_periods=2).loc["a", "b"] == pytest.approx(1.0)
    for n_overlap, finite in ((59, False), (60, True)):    # the default boundary is exactly 60 days
        b = x.where(d >= d[-n_overlap])
        assert np.isfinite(ic_correlation({"a": x, "b": b}).loc["a", "b"]) == finite, n_overlap
    mt = multiple_testing(pd.DataFrame({"t_nw": [1.0, 2.0]}, index=["a", "b"]), c)
    assert mt["n_eff"] == pytest.approx(2.0)              # the NaN pair counts as uncorrelated


# ---------------------------------------------------------------------- H3 composite
def test_composite_backtest_trades_only_membership_changes():
    p = random_panel(T=300, N=40, seed=4)
    cm = CostModel()
    out = composite_backtest(p, [12, 101], "equal", CFG, cm)
    reb = out["turnover"].notna()
    assert reb.sum() > 40
    assert (out.index[reb].dayofweek == 4).all()                    # W-FRI on a bdate calendar
    n = out["n_trades"]
    assert (out.index[n > 0].dayofweek == 4).all()                  # trades only on rebalances
    assert n[n > 0].iloc[0] == 20 and n.max() <= 40                # first build: 10 + 10 tickets
    assert (n[reb] % 2 == 0).all()                                   # every exit has an entry
    g = out["gross"].notna()
    assert (out.loc[g, "net"] <= out.loc[g, "gross"]).all()
    assert out.loc[g, "cost"].min() == pytest.approx(50e-4 / 252)   # borrow on a no-trade day
    signed = composite_backtest(p, [12, 101], {12: 1.0, 101: -1.0}, CFG, cm)
    assert not np.allclose(signed["gross"].dropna(), out["gross"].dropna())
    lo = composite_backtest(p, [12], "equal", CFG, cm, book="long_only", n_long=20)
    assert lo["short"].isna().all() and "active" in lo
    np.testing.assert_allclose(lo["active"], lo["net"] - lo["benchmark"])


# ---------------------------------------------------------------------- unlocked runs keep the boundary cut
def _same(x, y) -> bool:
    return np.array_equal(np.asarray(x, dtype=float), np.asarray(y, dtype=float), equal_nan=True)


def test_unlock_keeps_is_identical():
    """Locked and unlocked runs agree on every pre-holdout row (rows <= cut - 1) of every series:
    IC, summaries, deciles, turnover and `trades` (`traded`, `short`, `n_trades`), so the
    liquidation forced by censoring the alphas at the holdout is not charged to any in-sample row."""
    p = planted_panel(seed=0)                             # 700 business days from 2015-01-02
    hs = "2016-06-01"
    eras = (("e1", "2015-01-01", "2015-12-31"), ("e2", "2016-01-01", "2016-05-31"),
            ("hold", "2016-06-01", "2026-12-31"))
    locked = RunConfig(markets=("us",), holdout_start=hs, eras=eras)
    unlocked = RunConfig(markets=("us",), holdout_start=hs, eras=eras, unlock_holdout=True)
    hold = pd.Timestamp(hs)
    cut = int(p.dates.searchsorted(hold))
    assert 0 < cut < len(p.dates) - 30
    r0 = evaluate_alpha(p, "rank(cap)", locked, None)
    r1 = evaluate_alpha(p, "rank(cap)", unlocked, None)
    for h in locked.horizons:
        # 1. every IS and era summary (all fields) is identical; the holdout era only exists unlocked
        for per in ("is", "e1", "e2"):
            assert _same(_values(r0.summary[h][per]), _values(r1.summary[h][per])), (h, per)
        assert r0.summary[h]["hold"]["n_days"] == 0 and r1.summary[h]["hold"]["n_days"] > 0
        # 2. IC rows before holdout_start match; unlocked has values from holdout_start on
        a, b = r0.ic[h], r1.ic[h]
        assert _same(a.loc[:hold - pd.Timedelta(days=1)], b.loc[:hold - pd.Timedelta(days=1)]), h
        assert a["ic"].loc[hold:].isna().all() and b["ic"].loc[hold:].notna().any(), h
        # 4. band rows of the forward-return frame
        fwd = forward_returns(p, h, lag=LAG)
        u = censor_holdout(fwd, unlocked, horizon=h)
        band = slice(cut - LAG - h, cut)
        assert u.iloc[band].isna().all().all()
        assert u.iloc[:cut - LAG - h].equals(fwd.iloc[:cut - LAG - h])
        assert u.iloc[cut].notna().any() and u.iloc[cut:].equals(fwd.iloc[cut:])
        lk = censor_holdout(fwd, locked, horizon=h)
        assert lk.iloc[cut - LAG - h:].isna().all().all()
    # 3. decile book, turnover and trades are identical on every row up to cut - 1 (the end-of-IS
    # liquidation caused by the censoring is not charged to any in-sample row)
    before = p.dates[cut - 1]
    assert _same(r0.deciles.loc[:before], r1.deciles.loc[:before])
    assert _same(r0.turnover.loc[:before], r1.turnover.loc[:before])
    assert _same(r0.trades.loc[:before], r1.trades.loc[:before])
    assert r1.deciles["long_short"].loc[hold:].notna().any()


# ---------------------------------------------------------------------- exits, guards, misc
def _dying_book(flip: bool):
    """h = 2, one name per bin: the top and bottom names go to 0 at row 10. flip=True swaps the sides
    at every rebalance (the new target differs); flip=False re-picks the same dead names."""
    import warnings
    from alpha101.evaluate import _decile_book
    T, N = 40, 12
    d = pd.bdate_range("2015-01-02", periods=T)
    syms = [f"S{i}" for i in range(N)]
    rng = np.random.default_rng(0)
    close = 50.0 + rng.normal(0, 0.1, (T, N)).cumsum(axis=0)
    score = np.tile(np.arange(N, dtype=float), (T, 1))     # S0 lowest .. S11 highest, every day
    if flip:
        swap = (np.arange(T) // 2) % 2 == 1                 # each sub-book (h = 2) flips at every rebalance
        score[swap] = score[swap][:, ::-1]
    close[10:, 0] = 0.0                                     # the bottom-bin names go to 0 at row 10 ...
    close[10:, N - 1] = 0.0                                 # ... and so do the top-bin names
    close[:10, 0] = close[:10, N - 1]
    fields = {k: close.copy() for k in ("open", "high", "low", "close", "adj_close")}
    fields["volume"] = np.full((T, N), 1e6)
    panel = Panel(dates=d, symbols=syms, fields=fields, groups={})
    cfg = RunConfig(markets=("us",), start="2015-01-02", horizons=(2,), primary_horizon=2, n_bins=12,
                    min_names={"us": 12})
    alpha = pd.DataFrame(score, index=d, columns=syms)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no RuntimeWarning
        return _decile_book(alpha, panel, 2, 12, cfg, 12), d


def test_zero_growth_side_is_worth_zero_and_flip_trades_the_new_weight():
    (deciles, turn, trades), d = _dying_book(flip=True)
    live = trades["traded"][trades["traded"] > 0]
    assert len(live) > 10 and np.isfinite(trades["traded"]).all()
    assert turn.notna().sum() > 10 and np.isfinite(turn.dropna()).all()
    assert (trades.loc[d[11:30], "traded"] > 0).all()
    # 2 builds x 1, then 2 per row: nothing is sold from a dead side, only the new weight is bought
    # (turnover is NaN only on the two rebuild rows: a re-entry has no previous book)
    assert trades["traded"].sum() == pytest.approx(76.0)


def test_zero_growth_side_repicked_trades_its_full_new_weight():
    (_, _, trades), _ = _dying_book(flip=False)
    assert np.isfinite(trades["traded"]).all()
    # builds on rows 0 and 1: 2 sub-books x 2 sides x 1 / h = 2.0; the rebuild after the names died
    # trades the full new weight once more (a worthless position is neither sold nor kept): 2.0
    assert trades["traded"].sum() == pytest.approx(4.0)


def test_flat_book_charges_liquidation_and_reentry():
    p = planted_panel(T=260, N=60, seed=1)
    cfg = RunConfig(markets=("us",), start="2005-01-03")
    a = Evaluator(p).evaluate("rank(cap)")
    a.iloc[150:160] = np.nan
    from alpha101.evaluate import _decile_book
    _, _, trades = _decile_book(a, p, 5, 10, cfg, 30)
    t = trades["traded"]
    # sub-book j goes flat on row 150 + j and is charged on its last held row 149 + j; n_trades stays
    # on the flat rebalance row
    assert (trades["n_trades"].iloc[150:155] == 12).all()
    assert np.allclose(t.iloc[150:154], 2.0 / 5)      # rows carrying only a liquidation
    assert t.iloc[149] > 2.0 / 5                                   # + the rebalance of a live sub-book
    assert t.iloc[154:160].eq(0).all()                             # the last one was charged on row 153
    assert (t.iloc[160:165] > 0).all()                             # re-entry still charged
    assert t.iloc[:200][trades["short"].iloc[:200] == 0].sum() == 0  # none left on a NaN-return row
    # all 5 liquidations (5 x 0.4 = 2.0 per-side capital, 45.96 bps at 22.98 bps/unit) reach `net`
    cm = CostModel(borrow_bps_per_year=0.0)
    gross = pd.Series(0.0, index=trades.index).where(trades["short"] > 0)
    cost = gross - apply_costs(gross, trades, "us", cm)
    bps = cm.trade_cost_bps("us", cm.notional_per_name)
    assert bps == pytest.approx(22.98)
    assert cost.iloc[:200].sum() == pytest.approx(t.iloc[:200].sum() * bps / 1e4)
    assert cost.iloc[150:154].sum() == pytest.approx(4 * 2.0 / 5 * bps / 1e4)


def test_composite_flat_book_charges_liquidation():
    p = random_panel(T=300, N=40, seed=4)
    cfg = RunConfig(markets=("us",))
    full = composite_backtest(p, [12], "equal", cfg, CostModel(half_spread_bps={"us": 0.0, "ca": 0.0}))
    assert full["n_trades"].max() > 0
    # n_long = n_short = 20 on 40 names needs every name; a thin universe forces flat rebalances
    uni = np.ones(p.shape, dtype=bool)
    uni[100:140, 10:] = False
    thin = Panel(dates=p.dates, symbols=p.symbols, fields=p.fields, groups=p.groups, universe=uni)
    cm = CostModel()
    out = composite_backtest(thin, [12], "equal", cfg, cm, n_long=20, n_short=20)
    flat = out["gross"].isna() & (out["n_trades"] > 0)
    assert list(out.index[flat]) == [pd.Timestamp("2015-05-22")]
    assert out.loc["2015-05-22", "n_trades"] == 40          # counted on the flat rebalance row
    # charged on the closing book's last held row: 2 sides x 1 x 22.98 bps on top of the borrow
    liq = 2 * cm.trade_cost_bps("us", cm.notional_per_name) / 1e4
    assert liq == pytest.approx(2 * 22.98e-4)
    prev = out.loc["2015-05-21"]
    assert np.isfinite(prev["net"]) and prev["n_trades"] == 0
    assert prev["cost"] == pytest.approx(cm.borrow_bps_per_year / 1e4 / 252 + liq)
    assert prev["net"] == pytest.approx(prev["gross"] - prev["cost"])


def test_locked_holdout_flat_is_not_charged_but_in_sample_flat_is():
    from alpha101.evaluate import _decile_book
    p = random_panel(T=300, N=40, seed=4)
    hs = str(p.dates[200].date())
    cut = int(p.dates.searchsorted(pd.Timestamp(hs)))
    assert cut == 200
    locked = RunConfig(markets=("us",), holdout_start=hs)
    unlocked = RunConfig(markets=("us",), holdout_start=hs, unlock_holdout=True)
    uni = np.ones(p.shape, dtype=bool)
    uni[100:140, 10:] = False                                  # a genuine in-sample flat
    thin = Panel(dates=p.dates, symbols=p.symbols, fields=p.fields, groups=p.groups, universe=uni)
    cm = CostModel()
    kw = dict(n_long=20, n_short=20)
    c0 = composite_backtest(thin, [12], "equal", locked, cm, **kw)
    c1 = composite_backtest(thin, [12], "equal", unlocked, cm, **kw)
    pre = c0.index < p.dates[cut]
    assert _same(c0[pre], c1[pre])                             # every series, incl. cost and net
    assert c0["n_trades"].loc[p.dates[cut:]].sum() > 0         # holdout flat: tickets still counted
    flat = c0["gross"].isna() & (c0["n_trades"] > 0) & pre
    assert list(c0.index[flat]) == [pd.Timestamp("2015-05-22")]
    assert c0.loc["2015-05-21", "cost"] > cm.borrow_bps_per_year / 1e4 / 252 + 1e-4   # genuine flat charged on s-1
    held = c0["gross"].loc[pre].dropna().index[-1]             # last held in-sample row: no holdout charge
    assert c0.loc[held, "cost"] == pytest.approx(c1.loc[held, "cost"])
    # the unlocked book has real rebalances from the holdout on; the locked one is flat there
    assert c1["gross"].loc[p.dates[cut]:].notna().any() and c0["gross"].loc[p.dates[cut]:].isna().all()

    a = Evaluator(thin).evaluate("rank(cap)")
    a_locked = a.copy()
    a_locked.iloc[cut:] = np.nan                               # what censoring does to the alpha
    a_locked.iloc[150:160] = np.nan                            # a genuine in-sample flat
    a.iloc[150:160] = np.nan
    t0 = _decile_book(a_locked, p, 5, 10, locked, 30)[2]
    t1 = _decile_book(a, p, 5, 10, unlocked, 30)[2]
    assert _same(t0.iloc[:cut], t1.iloc[:cut])
    assert t0["n_trades"].iloc[cut:cut + 5].sum() > 0 and t0["traded"].iloc[cut - 1] == t1["traded"].iloc[cut - 1]
    assert t0["traded"].iloc[149] > 2.0 / 5                    # in-sample flat still charged on s-1


def test_unknown_market_names_valid_markets():
    with pytest.raises(ValueError, match="us"):
        evaluate_alpha(random_panel(T=60, N=10, seed=0), 101, CFG, None, market="US")


def test_alpha_id_for_expression_is_sha1_of_canonical_form():
    import hashlib
    from alpha101.expr.parser import canonical
    expr = "rank(cap)"
    res = evaluate_alpha(planted_panel(T=120), expr, RunConfig(markets=("us",), horizons=(5,)), None)
    assert res.alpha_id == "expr_" + hashlib.sha1(canonical(expr).encode()).hexdigest()[:10]
    assert evaluate_alpha(planted_panel(T=120), 12, RunConfig(markets=("us",), horizons=(5,)),
                          None).alpha_id == "alpha012"


def test_decile_gate_and_tie_break():
    from alpha101.evaluate import _bins
    A = np.array([[1.0, np.nan, 2.0, 3.0, 4.0, 5.0],                  # 5 valid names
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],                    # all tied
                  [2.0, 1.0, np.nan, np.nan, np.nan, np.nan]])        # 2 valid names
    b = _bins(A, n_bins=3, min_names=3)
    assert (b[2] == 0).all()                                          # below max(min_names, n_bins): nothing
    assert (b[0] > 0).sum() == 5
    assert list(b[1]) == [1, 1, 2, 2, 3, 3]                           # ties: symbol order, earlier = lower
    # end to end: a day below min_names gives NaN decile rows
    p = planted_panel(T=200, N=60, seed=1)
    a = Evaluator(p).evaluate("rank(cap)")
    a.iloc[100:110, 5:] = np.nan                                      # 5 valid names < min_names 30
    from alpha101.evaluate import _decile_book
    out = _decile_book(a, p, 1, 10, RunConfig(markets=("us",), start="2005-01-03"), 30)[0]
    assert out.iloc[100:110].isna().all().all() and out.iloc[10:90].notna().all().all()


# ---------------------------------------------------------------------- liquidation edge cases
def _recorded_charges(monkeypatch):
    """Wrap `_charge_liquidations`; each call appends [(row, amount added to traded)] to the returned list."""
    import alpha101.evaluate as ev
    calls, orig = [], ev._charge_liquidations

    def wrap(traded, gross, liquidations, n_trades, *a, **k):
        before = traded.copy()
        orig(traded, gross, liquidations, n_trades, *a, **k)
        d = traded - before
        calls.append([(int(r), round(float(d[r]), 9)) for r in np.flatnonzero(d)])
    monkeypatch.setattr(ev, "_charge_liquidations", wrap)
    return calls


def _decile_flat_runs(nan_rows, monkeypatch):
    """Locked and unlocked decile book (h = 5, cut = row 200) for an alpha NaN on `nan_rows`; the locked
    alpha is also NaN from the cut on (what censoring does). Returns (locked trades, charges, unlocked ...)."""
    from alpha101.evaluate import _decile_book
    p = random_panel(T=300, N=40, seed=4)
    hs = str(p.dates[200].date())
    base = Evaluator(p).evaluate("rank(cap)")
    a_unlocked = base.copy()
    a_unlocked.iloc[nan_rows] = np.nan
    a_locked = a_unlocked.copy()
    a_locked.iloc[200:] = np.nan
    calls = _recorded_charges(monkeypatch)
    t0 = _decile_book(a_locked, p, 5, 10, RunConfig(markets=("us",), holdout_start=hs), 30)[2]
    t1 = _decile_book(a_unlocked, p, 5, 10, RunConfig(markets=("us",), holdout_start=hs,
                                                      unlock_holdout=True), 30)[2]
    return t0, calls[0], t1, calls[1]


def test_decile_flat_at_cut_minus_one_is_charged_and_locked_equals_unlocked(monkeypatch):
    # sub-book 4 goes flat at s = 199 = cut - 1 (genuine: the alpha is back on row 200). Its charge row is
    # 197, not 198: the straddle band 198-199 is NaN. Charged in both runs; rows 200-203 are censored flats.
    t0, c0, t1, c1 = _decile_flat_runs(slice(199, 200), monkeypatch)
    assert c0 == [(197, pytest.approx(0.4))] and c1 == c0
    assert _same(t0.iloc[:200], t1.iloc[:200])                 # traded, short, n_trades, every pre-cut row
    assert t0["n_trades"].iloc[199] == 8                       # a genuine flat's tickets sit on row s


def test_decile_flats_straddling_the_cut(monkeypatch):
    # flats at s = 199 (= cut - 1, charged on 197) and s = 200 (= cut, genuine: charged unlocked on 197
    # as well, skipped locked): the one documented place where locked and unlocked differ before the cut
    t0, c0, t1, c1 = _decile_flat_runs(slice(199, 201), monkeypatch)
    assert c0 == [(197, pytest.approx(0.4))]
    assert c1 == [(197, pytest.approx(0.8))]
    assert t1["traded"].iloc[197] - t0["traded"].iloc[197] == pytest.approx(0.4)
    assert _same(t0.iloc[:197], t1.iloc[:197])
    assert (t0["n_trades"].iloc[199:201] > 0).all()            # tickets still counted on both flat rows


def test_composite_flat_at_cut_minus_one_is_charged_and_straddle(monkeypatch):
    p = random_panel(T=300, N=40, seed=4)
    fri = [i for i in range(300) if p.dates[i].dayofweek == 4 and i > 190][0]
    assert fri == 195
    hs = str(p.dates[fri + 1].date())                          # cut = 196: the Friday flat is cut - 1
    cm = CostModel()
    calls = _recorded_charges(monkeypatch)

    def run(masks, unlock):
        uni = np.ones(p.shape, dtype=bool)
        for lo, hi in masks:
            uni[lo:hi, 10:] = False
        thin = Panel(dates=p.dates, symbols=p.symbols, fields=p.fields, groups=p.groups, universe=uni)
        cfg = RunConfig(markets=("us",), holdout_start=hs, unlock_holdout=unlock)
        return composite_backtest(thin, [12], "equal", cfg, cm, n_long=20, n_short=20)
    # (a) genuine flat at s = cut - 1: charged on cut - 3 = 193 (band 194-195 is NaN); locked == unlocked
    l0, u0 = run([(fri, fri + 3)], False), run([(fri, fri + 3)], True)
    assert [r for r, _ in calls[0]] == [193] and calls[1] == calls[0]
    assert _same(l0.iloc[:196], u0.iloc[:196])                 # incl. traded-driven cost and net
    assert l0["cost"].iloc[193] > cm.borrow_bps_per_year / 1e4 / 252 + 1e-4
    # (b) straddle: that flat plus a second one at s = 205 >= cut after a rebuild on row 200
    l1, u1 = run([(fri, fri + 3), (205, 300)], False), run([(fri, fri + 3), (205, 300)], True)
    assert [r for r, _ in calls[2]] == [193]                   # locked: the later flat is skipped
    assert [r for r, _ in calls[3]][0] == 193 and len(calls[3]) == 2   # unlocked: charged too
    assert _same(l1.iloc[:196], u1.iloc[:196])


def test_decile_liquidation_before_start_is_free_and_not_counted(monkeypatch):
    from alpha101.evaluate import _decile_book
    p = random_panel(T=300, N=40, seed=4)
    a = Evaluator(p).evaluate("rank(cap)")
    a.iloc[50:60] = np.nan
    cfg = RunConfig(markets=("us",), start=str(p.dates[50].date()), holdout_start=str(p.dates[250].date()))
    calls = _recorded_charges(monkeypatch)
    _, _, tr = _decile_book(a, p, 5, 10, cfg, 30)
    # sub-book 0 is flat at s = 50 = start: its charge row 49 is before start, so it is neither in traded
    # nor in n_trades; sub-books 1-4 (flats at 51-54) are charged on rows 50-53
    assert calls[0] == [(r, pytest.approx(0.4)) for r in range(50, 54)]
    assert np.allclose(tr["traded"].iloc[0:4], 0.4) and tr["traded"].iloc[4] == 0   # rows 50-53 only
    assert tr["n_trades"].iloc[0] == 0                         # row 50: the dropped liquidation (8 tickets)
    assert (tr["n_trades"].iloc[1:5] == 8).all()               # rows 51-54: charged liquidations
    assert tr["n_trades"].iloc[:5].sum() == 4 * 8
    # traded and n_trades agree: every liquidation that is in one is in the other
    assert (tr["traded"].iloc[:5] > 0).sum() == 4 and (tr["n_trades"].iloc[:5] > 0).sum() == 4
