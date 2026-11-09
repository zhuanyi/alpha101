import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "cost_edge_duallisting", Path(__file__).resolve().parents[1] / "scripts" / "cost_edge_duallisting.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

FX, K = 1.4, 9.99


needs_norbert_snapshot = pytest.mark.skipif(
    not dl.NORBERT_SNAPSHOT.exists(),
    reason="needs a local Norbert snapshot (vendor price data is not redistributed); run scripts/cost_edge_norbert.py")


def test_route_costs_hand_computed_10k():
    # V = 10,000; h_tsx = 10 bp, h_us = 8 bp, NG half-spread sum = 10 bp
    r = dl.route_costs(10_000, FX, 0.001, 0.0008, 0.001)
    assert r["a"] == pytest.approx(2 * 0.001 * 10_000 + 2 * 9.99)               # 39.98
    assert r["b"] == pytest.approx(2 * 0.0008 * 10_000 + 2 * 9.99 * 1.4)        # 16 + 27.972: US$ commission in C$
    assert r["c_fx_desk"] == pytest.approx(0.0126 * 4 / 3)                           # tier of US$7,142.86
    assert r["c"] == pytest.approx(r["b"] + 2 * 0.0168 * 10_000)                # 43.972 + 336
    assert r["c_ng"] == pytest.approx(2 * 9.99 / 10_000 + 0.001)                # 0.002998
    assert r["d"] == pytest.approx(r["b"] + 2 * 0.002998 * 10_000)              # 43.972 + 59.96
    assert bool(r["ng_wins"])
    assert r["e"] == pytest.approx(r["a"] + 2 * 0.002998 * 10_000)              # cheaper FX leg is NG


def test_us_commission_converted_to_cad():
    zero = dl.route_costs(10_000, FX, 0.0, 0.0, 0.0)
    assert zero["a"] == pytest.approx(19.98)
    assert zero["b"] == pytest.approx(27.972)


def test_in_trade_floor_and_tier_basis():
    # V = 100,000 C$: USD amount 71,428.57 -> tier 0.78% listed (1.04% all-in); in-trade >= 60k keeps the 0.78% tier
    usd = dl.route_costs(100_000, FX, 0, 0, 0.001)
    assert usd["c_fx_desk"] == pytest.approx(0.0104)
    assert usd["c_in"] - usd["b"] == pytest.approx(2 * 0.0104 * 100_000)
    # CAD basis reads the tier off the C$ amount (100,000 -> 0.46% listed, 0.6133% all-in)
    cad = dl.route_costs(100_000, FX, 0, 0, 0.001, basis="cad")
    assert cad["c_fx_desk"] == pytest.approx(0.0046 * 4 / 3)


def test_ng_not_used_when_desk_cheaper():
    # huge NG spread: NG one-way 2*9.99/250000 + 0.02 = 2.008% > desk 0.6133% at 250k (USD amount 178,571)
    r = dl.route_costs(250_000, FX, 0.001, 0.001, 0.02)
    assert not r["ng_wins"]
    assert r["e"] == pytest.approx(r["a"] + 2 * 0.0046 * 4 / 3 * 250_000)


def test_pick_winner_margin_and_tie():
    w, m = dl.pick({"TSX": 100.0, "US": 130.0}, 10_000)
    assert w == "TSX" and m == pytest.approx(30.0)            # 30 / 10,000 = 30 bp
    w, m = dl.pick({"TSX": 100.0, "US": 100.5}, 10_000)       # 0.5 bp < 1 bp tie threshold
    assert w == "tie" and m == pytest.approx(0.5)


def test_parity_gap_formula():
    idx = pd.date_range("2025-01-01", periods=3)
    tsx = pd.Series([101.0, 100.0, 99.0], index=idx)
    us = pd.Series([50.0, 50.0, 50.0], index=idx)
    boc = pd.Series([2.0, 2.0, 2.0], index=idx)
    g = dl.parity_gap_bp(tsx, us, boc)
    assert list(g.round(6)) == [100.0, 0.0, -100.0]
    # dates missing in any series are dropped
    g2 = dl.parity_gap_bp(tsx, us, boc.iloc[:2])
    assert len(g2) == 2


def test_required_spread_edge():
    v, h_ng = 10_000, 0.001
    r = dl.required_spread_edge_bp(v, FX, h_ng)
    # default route: gambit one way = (C$ order + US$ order converted) / V + half-spread sum, the same DLR-U parameter as route_costs
    ng_default = (K + dl.DLRU_COMMISSION_USD * FX) / v + h_ng
    ng_cad_k = 2 * K / v + h_ng                                             # the old (inconsistent) reading: DLR-U leg at C$9.99
    desk = dl.route_costs(v, FX, 0.0, 0.0, h_ng)["c_fx_desk"]
    assert ng_default < desk                                                 # NG is the cheaper FX leg here
    assert r["fx_per_side_bp"] == pytest.approx(ng_default * 1e4)
    assert abs(r["fx_per_side_bp"] - ng_cad_k * 1e4) > 3.0                   # and differs from the C$9.99 reading by ~4 bp at C$10k
    assert r["commission_gap_bp_per_side"] == pytest.approx((2 * 9.99 * 1.4 - 2 * 9.99) / 10_000 * 1e4 / 2)  # 3.996
    r0 = dl.required_spread_edge_bp(v, FX, h_ng, k_dlru_usd=0.0)
    assert r0["fx_per_side_bp"] == pytest.approx((K / v + h_ng) * 1e4)


def _line(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"High": c * 1.01 * scale, "Low": c * 0.99 * scale, "Close": c * scale,
                         "Volume": 1_000_000.0}, index=idx)


def test_pair_sanity_filter_flags_wrong_mapping_and_keeps_good_pair():
    us = _line(600, 1)
    boc = pd.Series(1.35, index=us.index)
    good = {"us": us, "tsx": us * np.array([1.35, 1.35, 1.35, 1.35])}   # same price in C$ at BoC 1.35
    good["tsx"]["Volume"] = 1_000_000.0
    ok = dl.analyse_pair("G.TO|G", good, boc, 1.35, 0.001)
    assert ok["drop"] is None and ok["h_tsx"] > 0 and abs(ok["gap_stats"]["median"]) < 1e-6
    bad = {"us": us, "tsx": us * 2.7}                                   # 2:1 line: parity gap +100%
    bad["tsx"]["Volume"] = 1_000_000.0
    res = dl.analyse_pair("B.TO|B", bad, boc, 1.35, 0.001)
    assert res["drop"] and "parity gap" in res["drop"]
    short = {"us": us.iloc[:100], "tsx": us.iloc[:100] * 1.35}
    assert "common sessions" in dl.analyse_pair("S.TO|S", short, boc, 1.35, 0.001)["drop"]
    miss = dl.analyse_pair("X.TO|X", {"us": us}, boc, 1.35, 0.001)["drop"]
    assert "no price history" in miss and "X.TO" in miss and "not downloaded" not in miss


def test_select_universe_ranks_by_tsx_adv_and_reports_cut():
    rs = [{"pair": f"P{i}", "drop": None, "adv_tsx_cad": float(i)} for i in range(5)]
    rs.append({"pair": "Z", "drop": "bad"})
    kept, dropped = dl.select_universe(rs, top_n=3)
    assert [r["pair"] for r in kept] == ["P4", "P3", "P2"]
    assert {r["pair"] for r in dropped} == {"Z", "P1", "P0"}
    assert "ranked 4" in [r["drop"] for r in dropped if r["pair"] == "P1"][0]


def test_verdicts_cad_prefers_tsx_usd_prefers_us_and_tie():
    def name(a, b, c, d, e, ng_wins=True):
        r = {"a": a, "b": b, "c": c, "d": d, "e": e, "c_in": c, "e_desk": e, "ng_wins": ng_wins}
        return {"routes": {v: r for v in dl.SIZES}}
    vd = dl.verdicts([name(40.0, 44.0, 380.0, 104.0, 100.0)] * 2)
    x = vd[("CAD", 10_000)]
    assert x["winner"] == "TSX" and x["us_route"] == "d" and x["us_cost"] == 104.0 and x["tsx_wins"] == 2
    assert vd[("USD", 10_000)]["winner"] == "US"
    # NG not winning -> US route from CAD is the FX desk
    vd2 = dl.verdicts([name(40.0, 44.0, 380.0, 104.0, 100.0, ng_wins=False)])
    assert vd2[("CAD", 25_000)]["us_route"] == "c"
    # a 0.5 bp difference at V=10,000 is a tie
    assert vd[("USD", 10_000)]["us_wins"] == 2                                        # b = 44 < e = 100 for both names
    eq = dl.verdicts([name(40.0, 40.0, 40.0, 40.0, 40.0, ng_wins=False)])             # equal costs are not wins for either line
    assert eq[("CAD", 10_000)]["tsx_wins"] == 0 and eq[("USD", 10_000)]["us_wins"] == 0
    vd3 = dl.verdicts([name(40.0, 40.0, 44.0, 44.0, 40.5)])
    assert vd3[("USD", 10_000)]["winner"] == "tie"
    assert vd3[("USD", 10_000)]["margin_bp"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- review-round tests
def _ohlc(rows, start="2026-01-05"):
    return pd.DataFrame(rows, columns=["High", "Low", "Close"], index=pd.bdate_range(start, periods=len(rows)))


def test_ar_negative_mean_is_floored_at_zero_not_reflected():
    # close fixed at 100, mid alternates 99.9 / 100.1: (c - eta_t)(c - eta_t+1) < 0 every day, so s^2 < 0 -> 0 (abs() would give > 0)
    rows = [((100.0 + (0.1 if i % 2 else -0.1)) + 0.05, (100.0 + (0.1 if i % 2 else -0.1)) - 0.05, 100.0) for i in range(12)]
    df = _ohlc(rows)
    assert dl.ce.ar_half_spread_month(df).iloc[0] == 0.0
    assert dl.ce.spread_window(df)["ar_half"] == 0.0


def test_cs_negative_spread_is_clamped_to_zero():
    # a 10% jump between two tight days: raw CS spread < 0; without the clamp the half spread would be negative
    df = _ohlc([(101.0, 100.0, 100.5), (111.0, 110.0, 110.5)])
    assert dl.ce.cs_half_spread(df).iloc[0] == 0.0
    assert dl.ce.spread_window(df)["cs_half"] == 0.0


def _full_pair(n=700, tsx_only_holidays=(), us_only_holidays=()):
    """US line on a weekday calendar; TSX line = US price x 1.35 on the same calendar, with optional one-sided holidays."""
    us = _line(n, 7)
    ts = us * np.array([1.35, 1.35, 1.35, 1.35])
    ts["Volume"] = 1_000_000.0
    us = us.drop(us.index[list(us_only_holidays)]) if us_only_holidays else us
    ts = ts.drop(ts.index[list(tsx_only_holidays)]) if tsx_only_holidays else ts
    return ts, us


def test_window_lines_uses_the_last_common_sessions_not_the_first():
    ts, us = _full_pair(700)
    t_own, u_own, w = dl.window_lines(ts, us, 252)
    assert len(w) == 252 and w[-1] == us.index[-1] and w[0] == us.index[-252]
    assert t_own.index[0] == us.index[-252] and t_own.index[-1] == us.index[-1]


def test_window_selection_drives_spread_and_adv():
    # first 400 sessions have wide bars and big volume; the last 300 are tight with small volume
    us = _line(700, 3)
    us.loc[us.index[:400], "High"] = us["Close"][:400] * 1.08
    us.loc[us.index[:400], "Low"] = us["Close"][:400] * 0.92
    us.loc[us.index[:400], "Volume"] = 9_000_000.0
    us.loc[us.index[400:], "Volume"] = 1_000_000.0
    ts = us * np.array([1.35, 1.35, 1.35, 1.35])
    ts["Volume"] = us["Volume"]
    boc = pd.Series(1.35, index=us.index)
    res = dl.analyse_pair("W.TO|W", {"tsx": ts, "us": us}, boc, 1.35, 0.001)
    tail = dl.ce.spread_window(us.tail(252))["used"]
    head = dl.ce.spread_window(us.head(252))["used"]
    assert head > 3 * tail                                            # the fixture distinguishes the windows
    assert res["h_us"] == pytest.approx(tail) and res["h_us"] != pytest.approx(head)
    assert res["adv_tsx_cad"] == pytest.approx(float((ts["Close"] * ts["Volume"]).tail(252).mean()))
    assert res["adv_tsx_cad"] < 2e8                                   # the 9M-share early period is not in the ADV window


def test_spreads_are_estimated_on_each_lines_own_calendar():
    # three sessions inside the window are US holidays (TSX trades): the TSX line must keep them, the common series must not
    ts, us = _full_pair(700, us_only_holidays=(500, 520, 540))
    ts.loc[ts.index[[500, 520, 540]], "High"] *= 1.05                 # wide bars on the TSX-only sessions
    t_own, u_own, w = dl.window_lines(ts, us, 252)
    assert len(w) == 252 and len(t_own) == 255 and len(u_own) == 252
    boc = pd.Series(1.35, index=ts.index)
    res = dl.analyse_pair("O.TO|O", {"tsx": ts, "us": us}, boc, 1.35, 0.001)
    assert res["h_tsx"] == pytest.approx(dl.ce.spread_window(t_own)["used"])
    assert res["h_tsx_common"] == pytest.approx(dl.ce.spread_window(ts.loc[w])["used"])
    assert res["h_tsx"] != pytest.approx(res["h_tsx_common"])
    assert res["h_us"] == pytest.approx(dl.ce.spread_window(u_own)["used"])


def test_stale_pair_is_dropped_but_current_pair_is_kept():
    ts, us = _full_pair(700)
    boc = pd.Series(1.35, index=us.index)
    last = us.index[-1]
    ok = dl.analyse_pair("S.TO|S", {"tsx": ts, "us": us}, boc, 1.35, 0.001, asof=last + pd.Timedelta(days=dl.STALE_DAYS))
    assert ok["drop"] is None
    stale = dl.analyse_pair("S.TO|S", {"tsx": ts, "us": us}, boc, 1.35, 0.001, asof=last + pd.Timedelta(days=dl.STALE_DAYS + 1))
    assert stale["drop"] and "stale" in stale["drop"]
    # a delisted line: US series ends 30 sessions earlier than the newest session in the snapshot
    short_us = us.iloc[:-30]
    gone = dl.analyse_pair("D.TO|D", {"tsx": ts, "us": short_us}, boc, 1.35, 0.001, asof=last)
    assert gone["drop"] and "stale" in gone["drop"]


def test_cad_basis_tier_boundary_at_25k():
    # CAD basis reads the tier off V itself: C$25,000 is the first dollar of the 0.78% tier, C$24,999.99 is still 1.26%
    at = dl.route_costs(25_000, FX, 0, 0, 0.001, basis="cad")["c_fx_desk"]
    below = dl.route_costs(24_999.99, FX, 0, 0, 0.001, basis="cad")["c_fx_desk"]
    assert at == pytest.approx(0.0078 * 4 / 3) and below == pytest.approx(0.0126 * 4 / 3)
    # USD basis at the same C$25k is US$17,857 -> still the 1.26% tier; the boundary sits at C$35,000 (= US$25,000 x 1.4)
    assert dl.route_costs(25_000, FX, 0, 0, 0.001)["c_fx_desk"] == pytest.approx(0.0126 * 4 / 3)
    assert dl.route_costs(35_000, FX, 0, 0, 0.001)["c_fx_desk"] == pytest.approx(0.0078 * 4 / 3)
    assert dl.route_costs(34_999, FX, 0, 0, 0.001)["c_fx_desk"] == pytest.approx(0.0126 * 4 / 3)


def test_dlru_commission_parameter_and_use_ng_flag():
    d9 = dl.route_costs(10_000, FX, 0.001, 0.001, 0.001, k_dlru_usd=9.99)
    d0 = dl.route_costs(10_000, FX, 0.001, 0.001, 0.001, k_dlru_usd=0.0)
    assert d9["c_ng"] == pytest.approx((9.99 + 9.99 * FX) / 10_000 + 0.001)       # DLR.TO C$9.99 + DLR-U.TO US$9.99 in C$
    assert d0["c_ng"] == pytest.approx(9.99 / 10_000 + 0.001)                       # DLR-U.TO leg free
    assert d9["d"] - d0["d"] == pytest.approx(2 * 9.99 * FX)                        # two DLR-U orders per round trip
    desk = dl.route_costs(10_000, FX, 0.001, 0.001, 0.001, use_ng=False)
    assert not desk["ng_wins"] and desk["e"] == pytest.approx(desk["e_desk"]) and desk["e"] > d9["e"]


def _routes_dict(a, b, c, d, e, ng_wins=True):
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "c_in": c, "e_desk": e, "ng_wins": ng_wins}


def test_saving_is_median_of_per_name_savings_not_difference_of_medians():
    # per-name (a, c): (10, 20), (20, 22), (30, 100): medians a = 20, c = 22 -> difference of medians 2; per-name savings 10, 2, 70 -> median 10
    rows = [(10.0, 20.0), (20.0, 22.0), (30.0, 100.0)]
    names = [{"routes": {v: _routes_dict(a, a + 5, c, c, a + 50, ng_wins=False) for v in dl.SIZES}} for a, c in rows]
    x = dl.verdicts(names)[("CAD", 10_000)]
    assert x["winner"] == "TSX" and x["margin_bp"] == pytest.approx(10.0)          # 10 C$ on V = 10,000 -> 10 bp
    assert x["tsx_cost"] == 20.0 and x["us_cost"] == 22.0                            # displayed costs stay medians


def test_alt_check_can_fail_when_a_reading_flips_the_verdict():
    same = {v: _routes_dict(40.0, 44.0, 380.0, 104.0, 100.0) for v in dl.SIZES}
    flip = {v: _routes_dict(40.0, 44.0, 20.0, 104.0, 100.0, ng_wins=False) for v in dl.SIZES}   # US via a cheap desk beats the TSX line
    names = [{"routes": same, **{k: same for k in dl.ALT_READINGS}, "routes_cadbasis": flip}]
    vd = dl.verdicts(names)
    chk = dl.alt_check(vd, names)
    assert vd[("CAD", 10_000)]["winner"] == "TSX"
    assert chk[("CAD", 10_000)] == [dl.ALT_READINGS["routes_cadbasis"]]
    assert chk[("USD", 10_000)] == []
    unchanged = dl.alt_check(vd, [{k: same for k in names[0]}])
    assert all(v == [] for v in unchanged.values())


def test_sign_test_p_exact_values():
    assert dl.sign_test_p(0, 4) == pytest.approx(2 / 16)
    assert dl.sign_test_p(1, 10) == pytest.approx(2 * 11 / 1024)
    assert dl.sign_test_p(5, 10) == 1.0
    assert dl.sign_test_p(9, 40) == dl.sign_test_p(31, 40)


def test_spread_findings_counts():
    def nm(ht, hu, gap):
        return {"h_tsx": ht, "h_us": hu, "gap_stats": {"median": gap}}
    f = dl.spread_findings([nm(0.0012, 0.0010, 1.0), nm(0.0010, 0.0012, 2.0), nm(0.0013, 0.0010, -1.0), nm(0.0010, 0.0010, 0.5)])
    assert (f["n_tsx_wider"], f["n_tsx_narrower"], f["n_gap_pos"]) == (2, 1, 3)
    assert f["median_diff_bp"] == pytest.approx(1.0)                                  # diffs (bp): 2, -2, 3, 0 -> median 1.0
    assert f["p_spread"] == pytest.approx(dl.sign_test_p(2, 3))
    assert f["n"] == 4 and f["median_gap_bp"] == pytest.approx(0.75)
    z = dl.spread_findings([nm(0.0012, 0.0010, 1.0), nm(0.0012, 0.0010, 0.0), nm(0.0012, 0.0010, 2.0), nm(0.0012, 0.0010, 3.0)])
    assert z["n_gap_pos"] == 3 and z["p_gap"] == pytest.approx(dl.sign_test_p(3, 3))      # a zero gap is neither positive nor counted in n


def test_one_leg_check_counts_cases_where_one_fx_leg_flips_the_ordering():
    # CAD account: TSX a = 100 vs US b = 40 before FX (advantage 60); the cheapest US route from CAD c = 160 -> one leg 60 -> b + 60 = 100 = a: not strictly cheaper
    n = {"pair": "X", "routes": {v: _routes_dict(100.0, 40.0, 160.0, 200.0, 190.0, ng_wins=False) for v in dl.SIZES}}
    ol = dl.one_leg_check([n])
    # CAD: own 100 < 40 + 60 = 100? no (equal) -> fails; USD: b = 40 vs a = 100 + half_usd 45 -> holds. 4 sizes x 2 accounts
    assert ol["total"] == 8 and ol["holds"] == 4 and len(ol["fails"]) == 4 and all(f[2] == "CAD" for f in ol["fails"])
    assert ol["min"][0] == pytest.approx(60 / 60) and ol["min"][1] == "X"
    n2 = {"pair": "Y", "routes": {v: _routes_dict(100.0, 40.0, 400.0, 500.0, 500.0, ng_wins=False) for v in dl.SIZES}}
    assert dl.one_leg_check([n2])["holds"] == 8


def test_stress_ordering_reports_sizes_where_the_winner_changes():
    vd = {(acct, v): {"winner": "TSX" if acct == "CAD" else "US"} for v in dl.SIZES for acct in ("CAD", "USD")}
    assert dl.stress_ordering(FX, 0.001, 9.99, vd) == {"CAD": [], "USD": []}
    wrong = {k: {"winner": "US" if k[0] == "CAD" else "TSX"} for k in vd}
    ch = dl.stress_ordering(FX, 0.001, 9.99, wrong)
    assert ch["CAD"] == dl.SIZES and ch["USD"] == dl.SIZES


def _write_snapshot(tmp_path, specs, n=700):
    """specs: (symbol, tsx_bar_scale, us_trim): the TSX bars are widened around the close by tsx_bar_scale, the US line loses its last us_trim sessions."""
    import json
    frames = []
    for i, (sym, bar_scale, us_trim) in enumerate(specs):
        us = _line(n, 10 + i)
        ts = us * np.array([1.35, 1.35, 1.35, 1.35])
        ts["High"] = ts["Close"] + (ts["High"] - ts["Close"]) * bar_scale
        ts["Low"] = ts["Close"] - (ts["Close"] - ts["Low"]) * bar_scale
        ts["Volume"] = 1_000_000.0 * (i + 1)
        if us_trim:
            us = us.iloc[:-us_trim]
        for side, df, name in (("tsx", ts, f"{sym}.TO"), ("us", us, sym)):
            o = df[["High", "Low", "Close", "Volume"]].reset_index().rename(columns={"index": "date"})
            o.insert(0, "side", side)
            o.insert(0, "pair", f"{sym}.TO|{sym}")
            frames.append(o)
    pd.concat(frames).to_csv(tmp_path / "prices.csv.gz", index=False)
    pd.DataFrame({"date": _line(n, 10).index, "FXUSDCAD": 1.35}).to_csv(tmp_path / "boc_fxusdcad.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({"fetched": "2026-10-23", "source": "synthetic", "candidates": len(specs), "failed_downloads": []}))


def _run(tmp_path, monkeypatch, specs, candidates=None):
    _write_snapshot(tmp_path, specs)
    monkeypatch.setattr(dl, "CANDIDATES", candidates or [(f"{s}.TO", s) for s, _, _ in specs])
    return dl.run(tmp_path, tmp_path / "note.md")


@needs_norbert_snapshot
def test_run_renders_generated_prose_from_a_synthetic_snapshot(tmp_path, monkeypatch):
    doc = _run(tmp_path, monkeypatch, [("AAA", 1.0, 0), ("BBB", 1.0, 0), ("CCC", 1.0, 0)])
    assert (tmp_path / "note.md").exists() and (tmp_path / "route_costs.csv").exists()
    assert "Plain words (generated from the table above)" in doc
    assert "UNVERIFIED. Not reviewed by a tax adviser" in doc and "per secondary sources" in doc
    assert "several times" not in doc and "data do not support" not in doc and "NG wins at every modelled size on the USD-basis" not in doc
    assert "two-sided exact sign test" in doc and "median of per-name savings" in doc
    assert "with CAD cash the cheaper line is the TSX line at every size modelled" in doc
    assert "with USD cash it is the US line at every size modelled" in doc
    # d = 0 wording: no unqualified "lower bound" on the gambit cost; the net direction is called unknown
    assert "net direction of the error in the NG numbers is unknown" in doc and "is a **lower bound** on the gambit" not in doc
    assert "every NG number is a lower bound" not in doc
    assert "treats the 3 names as independent" in doc and "After removing the daily cross-name mean" in doc
    assert "The sign test is fragile; the direction is more robust." in doc
    assert "Survivor-only" in doc and "1 of 1 candidates" not in doc and "Not in the final list: 0 of 3 candidates (0 failed" in doc


@needs_norbert_snapshot
def test_run_direction_sentence_follows_the_data(tmp_path, monkeypatch):
    wide = _run(tmp_path, monkeypatch, [("AAA", 3.0, 0), ("BBB", 3.0, 0), ("CCC", 3.0, 0)])
    assert "The TSX half-spread is wider than the US one for 3 of 3 names (wider for 3, narrower for 0" in wide
    narrow = _run(tmp_path, monkeypatch, [("AAA", 0.3, 0), ("BBB", 0.3, 0), ("CCC", 3.0, 0)])
    assert "The TSX half-spread is narrower than the US one for 2 of 3 names (wider for 1, narrower for 2" in narrow


@needs_norbert_snapshot
def test_run_drops_stale_pairs_relative_to_the_newest_session_and_ignores_non_candidates(tmp_path, monkeypatch):
    # BBB's US line ends 30 sessions before the others; ZZZ is in the snapshot but not in CANDIDATES
    specs = [("AAA", 1.0, 0), ("BBB", 1.0, 30), ("ZZZ", 1.0, 0)]
    doc = _run(tmp_path, monkeypatch, specs, candidates=[("AAA.TO", "AAA"), ("BBB.TO", "BBB")])
    assert "`BBB.TO|BBB`: stale: last common session" in doc
    assert "ZZZ" not in doc and "AAA.TO|AAA" in doc
    routes = pd.read_csv(tmp_path / "route_costs.csv")
    assert set(routes["pair"]) == {"AAA.TO|AAA"}


def test_clean_drops_rows_with_high_below_low():
    df = pd.DataFrame({"High": [10.0, 9.0], "Low": [9.0, 10.0], "Close": [9.5, 9.5], "Volume": [1.0, 1.0]})
    assert len(dl.clean(df)) == 1


def test_window_lines_cut_both_lines_at_the_common_window_end():
    ts, us = _full_pair(700)
    us = us.iloc[:-3]                                                 # the US line has no last three sessions
    t_own, u_own, w = dl.window_lines(ts, us, 252)
    assert w[-1] == us.index[-1] and t_own.index[-1] == us.index[-1] and ts.index[-1] > t_own.index[-1]


def _tail_head_pair(n=700, early_gap=1.2):
    """TSX = US x 1.35 (zero parity gap) on the last 300 sessions, but x 1.35 x early_gap before: only the tail window is clean."""
    us = _line(n, 5)
    scale = np.where(np.arange(n) < n - 300, 1.35 * early_gap, 1.35)[:, None]
    ts = us * scale
    ts["Volume"] = 1_000_000.0
    return ts, us, pd.Series(1.35, index=us.index)


def test_parity_statistics_and_filter_use_the_last_252_sessions_only():
    ts, us, boc = _tail_head_pair()
    ok = dl.analyse_pair("T.TO|T", {"tsx": ts, "us": us}, boc, 1.35, 0.001)
    assert ok["drop"] is None                                          # full-history median |gap| is 2,000 bp, the last 252 sessions are clean
    assert len(ok["gap_win"]) == 252 and ok["gap_win"].index[-1] == us.index[-1]
    assert ok["gap_stats"]["abs_median"] < 1e-6 and ok["gap_stats"]["median"] < 1e-6
    # reverse: clean history, bad last 252 sessions -> dropped
    ts2 = ts.copy()
    ts2.iloc[-252:, :4] = (us.iloc[-252:] * 2.7).values[:, :4]
    bad = dl.analyse_pair("T.TO|T", {"tsx": ts2, "us": us}, boc, 1.35, 0.001)
    assert bad["drop"] and "parity gap" in bad["drop"]


def test_us_adv_is_converted_to_cad_at_the_given_fx():
    ts, us, boc = _tail_head_pair()
    res = dl.analyse_pair("T.TO|T", {"tsx": ts, "us": us}, boc, 1.35, 0.001, asof=None)
    tail_usd = float((us["Close"] * us["Volume"]).tail(252).mean())
    for fx in (1.35, 1.6):
        assert dl.analyse_pair("T.TO|T", {"tsx": ts, "us": us}, boc, fx, 0.001)["adv_us_cad"] == pytest.approx(tail_usd * fx)
    assert res["adv_us_cad"] == pytest.approx(tail_usd * 1.35)


def _wide_tsx_pair():
    us = _line(700, 21)
    ts = us * np.array([1.35, 1.35, 1.35, 1.35])
    ts["High"] = ts["Close"] + (ts["High"] - ts["Close"]) * 2.0
    ts["Low"] = ts["Close"] - (ts["Close"] - ts["Low"]) * 2.0
    ts["Volume"] = 1_000_000.0
    return ts, us, pd.Series(1.35, index=us.index)


@pytest.mark.parametrize("key,kwargs", [
    ("routes", {"k_dlru_usd": 9.99}),
    ("routes_dlru0", {"k_dlru_usd": 0.0}),
    ("routes_cadbasis", {"k_dlru_usd": 9.99, "basis": "cad"}),
    ("routes_desk_usd", {"k_dlru_usd": 9.99, "use_ng": False}),
    ("routes_desk_cad", {"k_dlru_usd": 9.99, "use_ng": False, "basis": "cad"}),
])
def test_route_variants_are_wired_with_the_documented_parameters(key, kwargs):
    ts, us, boc = _wide_tsx_pair()
    res = dl.analyse_pair("T.TO|T", {"tsx": ts, "us": us}, boc, 1.4, 0.001)
    for v in dl.SIZES:
        assert res[key][v] == dl.route_costs(v, 1.4, res["h_tsx"], res["h_us"], 0.001, **kwargs)
    # the variants must actually differ from the primary reading somewhere, otherwise the comparison above would be vacuous
    if key != "routes":
        assert any(res[key][v] != res["routes"][v] for v in dl.SIZES)


def test_spread_estimator_variants_are_wired_and_differ():
    ts, us, boc = _wide_tsx_pair()
    res = dl.analyse_pair("T.TO|T", {"tsx": ts, "us": us}, boc, 1.4, 0.001)
    st, su = res["st"], res["su"]
    assert st["cs_half"] != pytest.approx(st["ar_half"]) and st["used"] == pytest.approx(max(st["cs_half"], st["ar_half"], st["half_tick"]))
    for v in dl.SIZES:
        assert res["routes_cs"][v] == dl.route_costs(v, 1.4, max(st["cs_half"], st["half_tick"]), max(su["cs_half"], su["half_tick"]), 0.001, k_dlru_usd=9.99)
        assert res["routes_ar"][v] == dl.route_costs(v, 1.4, max(st["ar_half"], st["half_tick"]), max(su["ar_half"], su["half_tick"]), 0.001, k_dlru_usd=9.99)
        assert res["routes_halfspread"][v] == dl.route_costs(v, 1.4, st["used"] / 2, su["used"] / 2, 0.0005, k_dlru_usd=9.99)
    assert res["routes_halfspread"][10_000]["a"] < res["routes"][10_000]["a"]


def test_alt_check_flags_a_spread_estimator_reading_that_flips_a_verdict():
    # primary: TSX a = 40 < US-from-CAD c = 380 -> TSX. Under the half-spread reading the US line from CAD is cheaper (c = 20)
    same = {v: _routes_dict(40.0, 44.0, 380.0, 104.0, 100.0) for v in dl.SIZES}
    flip = {v: _routes_dict(40.0, 44.0, 20.0, 104.0, 100.0, ng_wins=False) for v in dl.SIZES}
    base = {"routes": same, **{k: same for k in dl.ALT_READINGS}}
    for key in ("routes_cs", "routes_ar", "routes_halfspread"):
        names = [{**base, key: flip}]
        chk = dl.alt_check(dl.verdicts(names), names)
        assert chk[("CAD", 10_000)] == [dl.ALT_READINGS[key]]
    # none of the spread readings is silently dropped from the check
    assert {"routes_cs", "routes_ar", "routes_halfspread"} <= set(dl.ALT_READINGS)


def test_spread_by_year_uses_each_lines_own_calendar():
    ts, us = _full_pair(700, us_only_holidays=(500, 520, 540))
    ts.loc[ts.index[[500, 520, 540]], "High"] *= 1.05
    boc = pd.Series(1.35, index=ts.index)
    res = dl.analyse_pair("O.TO|O", {"tsx": ts, "us": us}, boc, 1.35, 0.001)
    c0, c1 = res["common_start"], res["common_end"]
    pd.testing.assert_frame_equal(res["spread_by_year"][0], dl.ce.spread_table(ts.loc[c0:c1]))
    common_ts = ts.loc[ts.index.intersection(us.index)]
    assert not res["spread_by_year"][0].equals(dl.ce.spread_table(common_ts))


def test_gap_common_factor_removes_the_daily_cross_name_mean():
    idx = pd.bdate_range("2026-01-05", periods=6)
    common = pd.Series([10.0, -10.0, 10.0, -10.0, 10.0, 10.0], index=idx)
    # every name = common + its own small offset; raw medians are all positive, demeaned medians split by the offsets
    names = [{"pair": f"P{i}", "gap_win": common + off} for i, off in enumerate((-1.0, 0.0, 1.0, 2.0, 3.0))]
    raw_pos = sum(float(n["gap_win"].median()) > 0 for n in names)
    f = dl.gap_common_factor(names)
    assert raw_pos == 5 and f["n"] == 5
    assert f["n_pos"] == 2                                              # offsets minus their mean 1.0: -2, -1, 0, 1, 2 -> two positive
    assert f["p"] == pytest.approx(dl.sign_test_p(2, 4))                # the zero median is not counted in n
    assert f["common_share"] == pytest.approx(1.0 - 2.0 / (100.0 + 2.0), abs=0.02)   # var of offsets 2 against var(common)+2 (approx)


def test_spread_sensitivity_counts_and_one_name_move():
    def nm(pair, ht, hu, adv=1.0):
        return {"pair": pair, "h_tsx": ht * 1e-4, "h_us": hu * 1e-4, "h_tsx_common": ht * 1e-4, "h_us_common": hu * 1e-4, "adv_us_cad": adv}
    names = [nm("A", 10, 13), nm("B", 10, 11.5), nm("C", 10, 10.5), nm("D", 12, 10), nm("E", 10, 14, adv=9.0)]
    extra = [nm("F", 12, 10, adv=0.5)]
    sens = dl.spread_sensitivity(names, names + extra)
    assert sens["one_name_moves_p"] == pytest.approx(dl.sign_test_p(2, 5))        # 1 wider / 4 narrower -> 2 / 3
    assert sens["min_1bp"][:2] == (1, 3) and sens["min_2bp"][:2] == (1, 2)
    assert sens["all_passing"][:3] == (6, 2, 4) and sens["without_remapped"][:2] == (1, 4)
    assert sens["top_by_us_adv"][:2] == (1, 4) or sens["top_by_us_adv"][:2] == (2, 4)
    nm_r = [nm("ABX.TO|B", 10, 14), nm("X", 12, 10)]
    assert dl.spread_sensitivity(nm_r)["without_remapped"][:2] == (1, 0)


def test_tax_text_flags_only_the_curated_us_domiciled_names():
    t = dl.tax_text([{"pair": "OVV.TO|OVV"}, {"pair": "RY.TO|RY"}])
    assert "Flagged names in the final list: OVV.TO|OVV." in t and "The 1 names in the final list" in t
    assert "None of the flagged names is in the final list" in dl.tax_text([{"pair": "RY.TO|RY"}])
    assert "pro-rata" in t or "pro-rata share" in t or "bears to its total" in t


@needs_norbert_snapshot
def test_ng_inputs_commission_and_fx_reach_the_breakeven_and_range():
    a = dl.ng_inputs(1.4)
    assert a["bt"][dl.DLRU_COMMISSION_USD][1] != a["bt"][dl.DLRU_ALT_USD][1]            # US$9.99 vs US$0 on the DLR-U leg
    assert dl.ng_inputs(1.0)["bt"][dl.DLRU_COMMISSION_USD][1] != a["bt"][dl.DLRU_COMMISSION_USD][1]
    # the break-even must be computed at the same fx as the commission conversion (not at fx = 1)
    h_sum = a["h_sum"]
    k_eff = (dl.ce.COMMISSION + dl.DLRU_COMMISSION_USD * 1.4) / 2.0
    v_star, _ = dl.ce.breakeven(k_eff, h_sum, 0.0, fx=1.4, v_max=max(dl.SIZES))
    assert a["bt"][dl.DLRU_COMMISSION_USD][0].startswith(f"C${v_star:,.0f}")

