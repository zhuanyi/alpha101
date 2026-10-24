import datetime as dt
import pathlib
import random
import re

import numpy as np
import pandas as pd
import pytest

import prices as pr
import rebound as rb
from synth import make_world, mini_snap, sym

D = dt.date
YEARS = (2012, 2014, 2016, 2018, 2020)           # not adjacent: each event owns its D_prev file
TAXLOSS = pathlib.Path(__file__).resolve().parents[2] / "research" / "taxloss"


@pytest.fixture(scope="module")
def world(generic_cal):
    snap, yd = make_world(generic_cal, years=YEARS)
    return snap, yd, pr.PricePanel(snap, generic_cal, yd)


def test_rank_ytd_orders_worst_first_and_adjusts_splits(generic_cal, world):
    snap, yd, panel = world
    e = generic_cal.event_row(2014)
    r = rb.rank_ytd(snap, panel.hits, e["D_prev"], e["R"])
    assert list(r["ysym"][:3]) == [sym(0), sym(1), sym(2)] and len(r) == 40
    assert r["ytd"].iloc[0] == pytest.approx(-0.50) and r["ytd"].is_monotonic_increasing
    # a 1:10 consolidation of S00 inside the year must not turn the worst loser into the best winner
    s2 = snap.copy()
    m = (s2["ysym"] == sym(0)) & (s2["as_of"] >= e["R"] - dt.timedelta(days=3)) & (s2["as_of"] <= e["R"])
    s2.loc[m, "price"] *= 10
    s2.loc[m, "shares"] /= 10
    h = pr.detect_splits(s2)
    assert (h["ysym"] == sym(0)).any()
    r2 = rb.rank_ytd(s2, h, e["D_prev"], e["R"])
    assert r2["ysym"].iloc[0] == sym(0) and r2["ytd"].iloc[0] == pytest.approx(-0.50, abs=1e-6)
    assert rb.rank_ytd(s2, h.iloc[0:0], e["D_prev"], e["R"])["ysym"].iloc[-1] == sym(0)   # unadjusted: looks like the best name


def test_lookahead_truncation_at_five_events_for_every_variant(generic_cal, world):
    """Rank output is unchanged when everything after the rank day is truncated or poisoned (P and each variant's own
    rank day: V-E0 ranks at L-6)."""
    snap, yd, panel = world
    rng = random.Random(7)
    years = rng.sample(list(YEARS), 5)
    for y in years:
        e = rb.event_dates(generic_cal, y)
        poisoned_for = {}
        for cell in rb.CELLS:
            rank_day = e[cell.rank_role]
            full = rb.rank_ytd(snap, panel.hits, e["D_prev"], rank_day, cell.floor)
            cut = rb.rank_ytd(snap[snap["as_of"] <= rank_day], panel.hits[panel.hits["d1"] <= rank_day], e["D_prev"], rank_day, cell.floor)
            if rank_day not in poisoned_for:
                poisoned = snap.copy()
                after = poisoned["as_of"] > rank_day
                poisoned.loc[after, "price"] = np.random.default_rng(0).uniform(0.1, 500, after.sum())
                poisoned.loc[after, "shares"] = np.random.default_rng(1).uniform(1, 5e4, after.sum())
                poisoned_for[rank_day] = (poisoned, pr.detect_splits(poisoned))
            poisoned, ph = poisoned_for[rank_day]
            pois = rb.rank_ytd(poisoned, ph, e["D_prev"], rank_day, cell.floor)
            pd.testing.assert_frame_equal(full, cut)
            pd.testing.assert_frame_equal(full[["ysym", "ytd", "price_R"]], pois[["ysym", "ytd", "price_R"]])
    assert rb.CELLS[1].rank_role == "L-6"


def test_no_forward_shift_in_the_code():
    pat = re.compile(r"\.shift\(\s*-")
    for f in TAXLOSS.glob("*.py"):
        assert not pat.search(f.read_text()), f"{f.name} uses a forward pandas shift"


def test_halted_names_use_volume_where_yahoo_exists(generic_cal, world):
    snap, yd, panel = world
    e = rb.event_dates(generic_cal, 2014)
    f = yd.frames[sym(3)].copy()
    f.loc[pd.Timestamp(e["L"]), "volume"] = 0.0
    yd2 = pr.YahooData({**yd.frames, sym(3): f}, yd.firsts)
    p2 = pr.PricePanel(snap, generic_cal, yd2)
    h = rb.halted_names(p2, [sym(3), sym(4)], e["R"], e["L"], e["D_prev"], e["X3"])
    assert h == {sym(3)}
    p3 = pr.PricePanel(snap, generic_cal, pr.YahooData({}))               # no Yahoo at all: nothing is halted
    assert rb.halted_names(p3, [sym(3)], e["R"], e["L"], e["D_prev"], e["X3"]) == set()
    ranked = rb.rank_ytd(snap, panel.hits, e["D_prev"], e["R"])
    assert len(rb.basket(ranked, 20, {sym(0)})) == 20 and sym(0) not in set(rb.basket(ranked, 20, {sym(0)})["ysym"])


def test_spread_estimators():
    n = 60
    flat = np.full(n, 10.0)
    assert rb.abdi_ranaldo(flat * 1.0, flat * 1.0, flat) == 0.0
    rng = np.random.default_rng(3)
    mid = 10 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    half = 0.01
    close = mid * (1 + half * rng.choice([-1, 1], n))        # bid/ask bounce of a 2% spread
    hi, lo = mid * 1.004 + 0.0, mid * 0.996
    s = rb.abdi_ranaldo(np.maximum(hi, close), np.minimum(lo, close), close)
    assert 0.0 < s < 0.05
    assert rb.abdi_ranaldo(hi[:10], lo[:10], close[:10]) != rb.abdi_ranaldo(hi[:10], lo[:10], close[:10])   # NaN: too short
    assert rb.corwin_schultz(hi, lo) >= 0.0


def test_half_spread_buckets_and_fallback(generic_cal, world):
    snap, yd, panel = world
    e = rb.event_dates(generic_cal, 2014)
    uni = rb.rank_ytd(snap, panel.hits, e["D_prev"], e["R"])
    hs = rb.half_spreads(panel, uni, e["R"], e["D_prev"], e["X3"])
    assert set(hs["h_source"]) <= {"own", "bucket", "fallback"} and (hs["h"] > 0).all()
    # without Yahoo every name uses the fallback table
    p3 = pr.PricePanel(snap, generic_cal, pr.YahooData({}))
    h3 = rb.half_spreads(p3, uni, e["R"], e["D_prev"], e["X3"])
    assert set(h3["h_source"]) == {"fallback"}
    row = h3.iloc[0]
    assert row["h"] == rb.FALLBACK_H_BP[(row["bucket_etf"], row["band"])] / 1e4


def test_costs_commission_and_e2h():
    c100 = rb.per_side_cost(np.array([0.0]), rb.BASE, 20, True)[0]
    assert c100 == pytest.approx(20e-4 * 0.999 + 10e-4, abs=2e-5)                       # 19.98 bp + 10 bp
    c250 = rb.per_side_cost(np.array([0.0]), rb.Scenario(book=250_000.0), 20, True)[0]
    assert c250 == pytest.approx(7.992e-4 + 10e-4, abs=1e-6)
    h = np.array([0.003])
    e2h = rb.Scenario(e2h=True)
    assert rb.per_side_cost(h, e2h, 20, True)[0] - rb.per_side_cost(h, rb.BASE, 20, True)[0] == pytest.approx(0.003)
    assert rb.per_side_cost(h, e2h, 20, False)[0] == rb.per_side_cost(h, rb.BASE, 20, False)[0]
    tbl = pd.DataFrame({"ret": [0.10, 0.0], "exit_class": ["holdings", "unresolved"], "h": [0.003, 0.003], "ok": [True, True]})
    t = rb.apply_scenario(tbl, rb.BASE, 20)
    cin = rb.per_side_cost(0.003, rb.BASE, 20, True)
    assert t["net_s"].iloc[0] == pytest.approx(1.1 * (1 - cin) / (1 + cin) - 1)
    assert rb.apply_scenario(tbl, rb.Scenario(unresolved="wipe"), 20)["ret_s"].iloc[1] == -1.0


def test_cell_event_matches_the_hand_computation(generic_cal, world):
    snap, yd, panel = world
    st = rb.Study(panel, generic_cal, years=[2014])
    r = st.cell_event(2014, rb.CELLS[0])
    assert r["n_universe"] == 40 and r["n_basket"] == 20
    assert r["basket_gross"] == pytest.approx(0.05) and r["ew_gross"] == pytest.approx(0.025) and r["g"] == pytest.approx(0.025)
    assert r["g_net"] < r["g"] and r["g_net"] > 0
    assert r["share_holdings"] == 1.0 and not r["partial_exit"]
    assert r["xic_tr"] == pytest.approx(0.0)
    # variants: other exit dates are on the same path
    vx1 = st.cell_event(2014, rb.CELLS[3])
    assert 0 < vx1["basket_gross"] < 0.05
    n10 = st.cell_event(2014, rb.CELLS[5])
    assert n10["n_basket"] == 10
    assert st.cell_event(2014, rb.CELLS[0], rb.Scenario(unresolved="wipe"))["g"] == pytest.approx(0.025)


def test_big_moves_found_and_cleared_names_excluded(generic_cal):
    snap, yd = make_world(generic_cal, years=(2014,), post=lambda i: 1.5 if i == 5 else 0.0)
    panel = pr.PricePanel(snap, generic_cal, yd)
    st = rb.Study(panel, generic_cal, years=[2014])
    bm = rb.big_moves(st)
    assert list(bm["ysym"]) == [sym(5)] and bm.iloc[0]["scope"] == "basket" and "P" in bm.iloc[0]["cells"]
    st2 = rb.Study(panel, generic_cal, years=[2013], excluded={(2014, sym(5))})
    assert rb.big_moves(st2).empty and st2.cell_event(2014, rb.CELLS[0])["n_universe"] == 39


def test_random_placebo_is_seeded_and_percentile_sane():
    rng = np.random.default_rng(0)
    py = {2012: (rng.normal(0, .05, 60), rng.normal(0, .05, 60)), 2013: (rng.normal(0, .05, 80), rng.normal(0, .05, 80))}
    a, b = rb.random_placebo(py, draws=1000, seed=rb.SEED), rb.random_placebo(py, draws=1000, seed=rb.SEED)
    np.testing.assert_array_equal(a["g_net"], b["g_net"])
    assert a["g_net"].shape == (1000, 2)
    c = rb.random_placebo(py, draws=1000, seed=1)
    assert not np.array_equal(a["g_net"], c["g_net"])
    d = a["g_net"].mean(axis=1)
    assert rb.percentile(d, d.max() + 1) == 100 and rb.percentile(d, d.min() - 1) == 0
    assert 45 < rb.percentile(d, float(np.median(d))) < 55


def test_t_dec_is_the_smaller_of_plain_and_nw():
    x = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.03, 0.05, 0.04, 0.06, 0.03, 0.05, 0.04])
    assert rb.t_dec(x) == min(rb.t_plain(x), rb.t_nw(x, 1))


# --- decide() ---------------------------------------------------------------------------------------------------------
def ev(g, g_net, n, pct=99.0, first=True, m_oos=0.01, years=None):
    k = len(g)
    return rb.Evidence(years=years or list(range(2012, 2012 + k)), g=np.array(g, float), g_net=np.array(g_net, float),
                       n=np.array(n, float), pct_net=pct, dec_first=first, m_oos=m_oos)


# 14 events, tight spread so t is large
GOOD = [0.06, 0.05, 0.07, 0.06, 0.05, 0.06, 0.07, 0.05, 0.06, 0.05, 0.06, 0.07, 0.05, 0.06]
NET = [x - 0.01 for x in GOOD]
POS = [0.03] * 14


def test_decide_four_outcomes():
    assert rb.decide(ev(GOOD, NET, POS))["verdict"] == "ADVANCE"
    # REJECT: mean gross <= 0
    assert rb.decide(ev([-x for x in GOOD], [-x for x in NET], POS))["verdict"] == "REJECT"
    # REJECT by the placebo percentile < 75
    assert rb.decide(ev(GOOD, NET, POS, pct=60.0))["verdict"] == "REJECT"
    # WEAK: conditions 2 and 4 hold, gross t >= 1.771, but net t < 2 (costs eat it)
    g = [0.012, 0.03, -0.01, 0.04, 0.0, 0.03, 0.02, 0.01, -0.01, 0.04, 0.02, 0.0, 0.03, 0.02]
    net = [x - 0.012 for x in g]
    r = rb.decide(ev(g, net, POS))
    assert rb.t_dec(g) >= 1.771 and rb.t_dec(net) < 2.0 and r["verdict"] == "WEAK" and not r["conditions"]["c1"]
    # gross t >= 1.771 but net t_dec < 2.0 must never be ADVANCE
    assert r["verdict"] != "ADVANCE"
    # UNDERPOWERED: positive but noisy, percentile fine, December not first
    assert rb.decide(ev(g, net, POS, first=False))["verdict"] == "UNDERPOWERED"


def test_decide_order_reject_before_advance():
    # all ADVANCE conditions hold except the percentile gate: REJECT wins only when a reject clause fires
    assert rb.decide(ev(GOOD, NET, POS, pct=90.0))["verdict"] in ("UNDERPOWERED", "WEAK")
    assert rb.decide(ev(GOOD, NET, POS, pct=74.0))["verdict"] == "REJECT"


def test_decide_pending_m_and_condition_six():
    assert rb.decide(ev(GOOD, NET, POS, m_oos=None))["verdict"] == "ADVANCE_PENDING_M"
    assert rb.decide(ev(GOOD, NET, POS, m_oos=-0.01))["verdict"] != "ADVANCE"
    base = ev(GOOD, NET, POS)
    bad = ev(GOOD, [x - 0.07 for x in GOOD], POS)                  # e.g. the E2h scenario kills the net edge
    r = rb.decide(base, {"e2h": bad})
    assert not r["conditions"]["c6"] and r["verdict"] != "ADVANCE"
    # without 2016: an outlier year that carries the result flips the verdict
    years = list(range(2012, 2026))
    g = [0.0] * 14
    g[4] = 0.5
    carried = ev(g, g, [0.01] * 14, years=years)
    assert rb.decide(carried)["conditions"]["c6"] in (True, False)


def test_decide_weak_upgrade_at_250k():
    g = [0.012, 0.03, -0.01, 0.04, 0.0, 0.03, 0.02, 0.01, -0.01, 0.04, 0.02, 0.0, 0.03, 0.02]
    r = rb.decide(ev(g, [x - 0.012 for x in g], POS), ev250=ev(g, [x + 0.05 for x in g], POS))
    assert r["verdict"] == "WEAK" and r["treated_as"] == "ADVANCE_C250K_ONLY"


def test_decide_refuses_with_uncleared_big_moves(tmp_path):
    p = tmp_path / "big_moves.csv"
    row = {c: "" for c in pr.BIG_COLUMNS} | {"year": 2020, "ysym": "A.TO"}
    pd.DataFrame([row]).to_csv(p, index=False)
    with pytest.raises(pr.GateError):
        rb.decide(ev(GOOD, NET, POS), big_moves_path=p)
    row.update(clearance="keep", reason="real move: verified against a news item")
    pd.DataFrame([row]).to_csv(p, index=False)
    assert rb.decide(ev(GOOD, NET, POS), big_moves_path=p)["verdict"] == "ADVANCE"
    pd.DataFrame(columns=pr.BIG_COLUMNS).to_csv(p, index=False)               # empty file = nothing to clear
    assert rb.decide(ev(GOOD, NET, POS), big_moves_path=p)["verdict"] == "ADVANCE"


def test_december_first_is_strict():
    s = pd.DataFrame({"month": [2, 3, 12], "mean_excess": [0.01, 0.02, 0.02]})
    assert not rb.december_first(s)                                           # a tie is not first
    s.loc[2, "mean_excess"] = 0.03
    assert rb.december_first(s)


def test_neff_and_stats_block(generic_cal, world):
    snap, yd, panel = world
    st = rb.Study(panel, generic_cal, years=list(YEARS))
    tabs = {c.id: st.cell_table(c) for c in rb.CELLS}
    n = rb.neff(tabs)
    assert 1 <= n["n_eff"] <= 9 and n["student_t_df13_equivalent"] > n["sidak_two_sided_normal_bar"] > 1.9
    sb = rb.stats_block(tabs["P"])
    assert sb["g"]["mean"] == pytest.approx(0.025) and "capm" not in sb      # constant benchmark: no regression
