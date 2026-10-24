import datetime as dt

import numpy as np
import pandas as pd
import pytest

import prices as pr
from synth import mini_snap

D = dt.date
D0, D1, D2, D3 = D(2020, 1, 31), D(2020, 2, 28), D(2020, 3, 31), D(2020, 4, 30)


def two_files(prices0, prices1, shares0, shares1):
    rows = []
    for t in prices0:
        rows.append(("XIC", D0, t, prices0[t], shares0[t]))
        rows.append(("XIC", D1, t, prices1[t], shares1[t]))
    return mini_snap(rows)


def base(n=20):
    names = [f"N{i}" for i in range(n)]
    return names, {t: 10.0 for t in names}, {t: 1000.0 for t in names}


def test_consolidation_and_split_caught_genuine_drop_not():
    names, p0, s0 = base()
    p1, s1 = dict(p0), dict(s0)
    p1["N1"], s1["N1"] = 100.0, 100.0           # 1:10 consolidation
    p1["N2"], s1["N2"] = 5.0, 2000.0            # 2:1 split
    p1["N3"] = 5.5                              # genuine -45%, shares unchanged
    h = pr.detect_splits(two_files(p0, p1, s0, s1))
    got = {r.ysym: r.q_applied for r in h.itertuples()}
    assert got == pytest.approx({"N1.TO": 0.1, "N2.TO": 2.0})
    assert h["applied"].all()


def test_fund_wide_creation_does_not_trigger_and_is_normalised_out():
    names, p0, s0 = base()
    p1 = dict(p0)
    s1 = {t: v * 1.05 for t, v in s0.items()}   # 5% creation moves every holding
    assert pr.detect_splits(two_files(p0, p1, s0, s1)).empty
    p1["N1"], s1["N1"] = 100.0, 100.0 * 1.05    # consolidation during the same creation
    h = pr.detect_splits(two_files(p0, p1, s0, s1))
    assert list(h["ysym"]) == ["N1.TO"] and h.iloc[0]["q_applied"] == pytest.approx(0.1)


def test_nonround_holdings_only_hit_with_small_move_not_applied():
    names, p0, s0 = base()
    p1, s1 = dict(p0), dict(s0)
    p1["N1"], s1["N1"] = 15.0, 1000.0 / 1.5     # index share-cap change: +50%, shares /1.5 (not a round ratio)
    h = pr.detect_splits(two_files(p0, p1, s0, s1))
    assert len(h) == 1 and not h.iloc[0]["applied"]


def test_yahoo_split_field_applied_when_consistent_and_ignored_when_not():
    names, p0, s0 = base()
    p1 = dict(p0)
    p1["N1"] = 5.2                              # holdings price halved but fund share count lagged
    ys = {"N1.TO": pd.Series({pd.Timestamp("2020-02-10"): 2.0}),
          "N2.TO": pd.Series({pd.Timestamp("2020-02-10"): 2.0})}     # N2: price unchanged -> inconsistent
    h = pr.detect_splits(two_files(p0, p1, s0, s0), ys).set_index("ysym")
    assert h.loc["N1.TO", "applied"] and h.loc["N1.TO", "q_applied"] == 2.0 and h.loc["N1.TO", "trigger"] == "yahoo"
    assert not h.loc["N2.TO", "applied"]


def test_tiny_yahoo_only_ratio_not_applied():
    names, p0, s0 = base()
    ys = {"N1.TO": pd.Series({pd.Timestamp("2020-02-10"): 1.03})}
    h = pr.detect_splits(two_files(p0, dict(p0), s0, s0), ys)
    assert len(h) == 1 and not h.iloc[0]["applied"]


def test_holdout_files_refused():
    snap = mini_snap([("XIC", D(2010, 12, 31), "A", 10.0, 100.0)])
    with pytest.raises(ValueError, match="sealed"):
        pr.PricePanel(snap, None, pr.YahooData({}))


# --- exit hierarchy ---------------------------------------------------------------------------------------------------
E, X = D(2020, 12, 29), D(2021, 1, 29)
PREV = D(2019, 12, 31)


def yahoo(frames):
    out = {}
    for s, (dates, close) in frames.items():
        c = pd.Series(close, index=pd.DatetimeIndex(dates), dtype=float)
        out[s] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c, "volume": 100.0,
                               "dividend": 0.0, "split": 0.0})
    return pr.YahooData(out, {s: pd.Timestamp("2005-01-03") for s in out})


def panel_for_exit_tests(takeovers=None):
    days = [PREV, D(2020, 12, 24), E, X]
    rows = [("XIC", d, "HLD", 10.0 if d != X else 11.0, 100.0) for d in days]                 # in XIC at X
    rows += [("XCS", d, "OTH", 10.0 if d != X else 12.0, 100.0) for d in days[:3]] + [("XIC", d, "OTH", 10.0, 100.0) for d in days[:3]]
    rows += [("XIC", d, "YAH", 10.0, 100.0) for d in days[:3]]                                  # gone at X, Yahoo has it
    rows += [("XIC", d, "UNR", 10.0, 100.0) for d in days[:3]]                                  # gone at X, nothing else
    rows += [("XIC", d, "TKO", 10.0, 100.0) for d in days[:3]]
    # OTH: exit file X holds it only in XCS
    rows += [("XCS", X, "OTH", 12.0, 100.0)]
    snap = mini_snap(rows)
    d24 = D(2020, 12, 24)
    yd = {"YAH.TO": ([PREV, d24, E, X], [10.0, 10.0, 10.0, 13.0]), "HLD.TO": ([PREV, d24, E, X], [10.0, 10.0, 10.0, 11.0])}
    return pr.PricePanel(snap, None, yahoo(yd), takeovers=takeovers)


def trade(p, s, **kw):
    return p.trade(2020, s, "XIC", E, X, E, 10.0, PREV, X, **kw)


def test_each_exit_class_is_reached():
    tk = pd.DataFrame([{"year": 2020, "ysym": "TKO.TO", "cash_per_share": 14.0, "source_url": "https://example.test/deal"}])
    p = panel_for_exit_tests(tk)
    r = {s: trade(p, s) for s in ("HLD.TO", "OTH.TO", "YAH.TO", "TKO.TO", "UNR.TO")}
    assert {k: v["exit_class"] for k, v in r.items()} == {"HLD.TO": "holdings", "OTH.TO": "other_etf", "YAH.TO": "yahoo",
                                                         "TKO.TO": "takeover", "UNR.TO": "unresolved"}
    assert r["HLD.TO"]["ret"] == pytest.approx(0.10) and r["OTH.TO"]["ret"] == pytest.approx(0.20)
    assert r["YAH.TO"]["ret"] == pytest.approx(0.30) and r["TKO.TO"]["ret"] == pytest.approx(0.40)
    assert r["UNR.TO"]["ret"] == pytest.approx(-0.30)                      # last holdings price x 0.70
    assert trade(p, "UNR.TO", unresolved="zero")["ret"] == 0.0 and trade(p, "UNR.TO", unresolved="wipe")["ret"] == -1.0


def test_takeover_needs_a_logged_source_url():
    tk = pd.DataFrame([{"year": 2020, "ysym": "TKO.TO", "cash_per_share": 14.0, "source_url": ""}])
    assert trade(panel_for_exit_tests(tk), "TKO.TO")["exit_class"] == "unresolved"


def test_yahoo_exit_with_later_consolidation_is_as_traded_not_ten_times():
    """Holdings are as-traded (price ~10 throughout); Yahoo is adjusted to a later 1:10 consolidation (close ~100)."""
    days = [PREV, E, D(2021, 1, 15)]
    snap = mini_snap([("XIC", PREV, "CON", 10.0, 100.0), ("XIC", E, "CON", 10.0, 100.0), ("XIC", D(2021, 1, 15), "CON", 11.0, 100.0)])
    c = pd.Series([100.0, 100.0, 110.0, 110.0, 110.0], index=pd.DatetimeIndex([PREV, E, D(2021, 1, 15), X, D(2021, 6, 1)]))
    f = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c, "volume": 1.0, "dividend": 0.0,
                      "split": [0.0, 0.0, 0.0, 0.0, 0.1]}, index=c.index)
    p = pr.PricePanel(snap, None, pr.YahooData({"CON.TO": f}, {"CON.TO": pd.Timestamp("2005-01-03")}))
    assert p.identity("CON.TO", PREV, X)                                     # passes after un-adjusting with the split field
    r = p.trade(2020, "CON.TO", "XIC", E, X, E, 10.0, PREV, X)               # X file is missing -> Yahoo exit
    assert r["exit_class"] == "yahoo" and r["ret"] == pytest.approx(0.10)    # not +1000%
    naive = 110.0 / 10.0 - 1
    assert naive > 9 and r["ret"] < 1


def test_split_between_entry_and_exit_adjusts_holdings_return():
    snap = mini_snap([("XIC", PREV, "SPL", 10.0, 100.0), ("XIC", E, "SPL", 10.0, 100.0), ("XIC", X, "SPL", 5.1, 200.0)]
                     + [("XIC", PREV, f"F{i}", 10.0, 100.0) for i in range(5)] + [("XIC", E, f"F{i}", 10.0, 100.0) for i in range(5)]
                     + [("XIC", X, f"F{i}", 10.0, 100.0) for i in range(5)])
    p = pr.PricePanel(snap, None, pr.YahooData({}))
    r = p.trade(2020, "SPL.TO", "XIC", E, X, E, 10.0, PREV, X)
    assert r["split_factor"] == pytest.approx(2.0) and r["ret"] == pytest.approx(0.02)   # 2 x 5.1 / 10 - 1


def test_dividend_yield_on_yahoo_basis():
    p = panel_for_exit_tests()
    f = p.yahoo.frames["HLD.TO"].copy()
    f.loc[pd.Timestamp(X), "dividend"] = 0.5
    p.yahoo.frames["HLD.TO"] = f
    assert p.dividend_yield("HLD.TO", E, X) == pytest.approx(0.5 / 10.0)


# --- gate -------------------------------------------------------------------------------------------------------------
def test_uncleared_merge_and_exclusions(tmp_path):
    new = pd.DataFrame([{c: "" for c in pr.BIG_COLUMNS} | {"year": 2020, "ysym": "A.TO"},
                        {c: "" for c in pr.BIG_COLUMNS} | {"year": 2021, "ysym": "B.TO"}])
    assert len(pr.uncleared(new)) == 2
    old = new.copy()
    old.loc[0, ["clearance", "reason"]] = ["exclude", "reverse split missed by the detector"]
    old.loc[1, "clearance"] = "keep"                                       # keep without a reason is still uncleared
    merged = pr.merge_big_moves(new, old)
    assert list(pr.uncleared(merged)["ysym"]) == ["B.TO"]
    assert pr.excluded_names(merged) == {(2020, "A.TO")}
    assert pr.uncleared(pr.read_big_moves(tmp_path / "missing.csv")).empty


# --- window trigger: holdings vs Yahoo return mismatch at a round factor (CHR 1-for-7, Feb 2025) ----------------------
W0, W1, W2 = D(2020, 1, 31), D(2020, 2, 5), D(2020, 2, 10)


def window_snap(sym, p0, p1, s0, s1, gap_file=True):
    """XIC files D0, [D1 without `sym`], D2; 20 other names constant. Returns (snap, D_start, D_end)."""
    names, q, sh = base()
    rows = []
    dates = [W0, W1, W2] if gap_file else [W0, W2]
    for d in dates:
        for t in names:
            rows.append(("XIC", d, t, q[t], sh[t]))
    rows.append(("XIC", W0, sym, p0, s0))
    rows.append(("XIC", W2, sym, p1, s1))
    return mini_snap(rows)


def yclose(sym, c0, c1):
    s = pd.Series([c0, c1], index=[pd.Timestamp(W0), pd.Timestamp(W2)])
    return lambda t: s if t == f"{sym}.TO" else None


def test_window_trigger_catches_consolidation_missing_from_the_file_between():
    # CHR: holdings 2.89 -> 19.36 (shares 99750 -> 14059), Yahoo 20.23 -> 19.36; the name is absent from the file between
    h = pr.detect_splits(window_snap("CHR", 2.89, 19.36, 99750.0, 14059.0), {}, yclose("CHR", 20.23, 19.36))
    h = h[h["ysym"] == "CHR.TO"]
    assert len(h) == 1
    r = h.iloc[0]
    assert r["trigger"] == "window" and bool(r["applied"]) and (r["d0"], r["d1"]) == (W0, W2)
    assert r["q_applied"] == pytest.approx(1 / 7)
    assert pr.is_nice(r["q_applied"])


def test_window_trigger_leaves_genuine_move_matching_yahoo_alone():
    # BB: 8.44 -> 17.96 in both sources
    h = pr.detect_splits(window_snap("BB", 8.44, 17.96, 1000.0, 1000.0), {}, yclose("BB", 8.44, 17.96))
    assert h.empty


def test_window_trigger_leaves_later_consolidation_multiple_alone():
    # WPRT / HUT: Yahoo closes are 10x / 5x the as-traded prices (later consolidation) but the return is identical
    assert pr.detect_splits(window_snap("WPRT", 6.49, 14.69, 1000.0, 1000.0), {}, yclose("WPRT", 64.9, 146.9)).empty
    assert pr.detect_splits(window_snap("HUT", 1.10, 2.58, 1000.0, 1000.0), {}, yclose("HUT", 5.5, 12.9)).empty


def test_window_trigger_ignores_mismatch_when_share_count_does_not_follow():
    # ticker re-assignment (IAG 2012): returns disagree by about 2x but the ETF share count moved 5x the other way
    h = pr.detect_splits(window_snap("IAG", 15.96, 25.83, 413262.0, 81138.0), {}, yclose("IAG", 31.5, 25.83))
    assert h.empty


def test_window_trigger_off_without_yahoo_closes():
    assert pr.detect_splits(window_snap("CHR", 2.89, 19.36, 99750.0, 14059.0), {}).empty


def test_window_split_adjusts_trade_return_through_panel():
    snap = window_snap("CHR", 3.01, 19.42, 99750.0, 14303.0)
    h = pr.detect_splits(snap, {}, yclose("CHR", 21.07, 19.42))
    panel = pr.PricePanel.__new__(pr.PricePanel)
    panel.hits = h
    panel._applied = {r.ysym: [(r.d1, float(r.q_applied))] for r in h[h["applied"]].itertuples()}
    q = pr.PricePanel.split_factor(panel, "CHR.TO", W0, W2)
    assert q * 19.42 / 3.01 - 1 == pytest.approx(-0.0783, abs=1e-3)


def test_window_trigger_not_applied_twice_when_both_etfs_see_the_same_consolidation():
    # XIC pair (W0, W2) and XCS pair (X0, X1) overlap with different boundaries: the factor 1/7 must enter once
    X0, X1 = D(2020, 2, 3), D(2020, 2, 12)
    snap = window_snap("CHR", 2.89, 19.36, 99750.0, 14059.0)
    xcs = mini_snap([("XCS", X0, "CHR", 2.80, 100000.0), ("XCS", X1, "CHR", 19.40, 14000.0)])
    snap = pd.concat([snap, xcs], ignore_index=True)
    ys = pd.Series([20.23, 20.0, 19.40, 19.36], index=[pd.Timestamp(d) for d in (W0, X0, X1, W2)])
    h = pr.detect_splits(snap, {}, lambda t: ys if t == "CHR.TO" else None)
    h = h[(h["ysym"] == "CHR.TO") & h["applied"]]
    assert len(h) == 1 and h.iloc[0]["trigger"] == "window"

