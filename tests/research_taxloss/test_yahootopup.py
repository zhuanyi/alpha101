import datetime as dt

import numpy as np
import pandas as pd
import pytest

import yahootopup as yt

D = dt.date


def hold(prices, start=D(2018, 12, 20)):
    d = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"as_of": [x.date() for x in d], "price": prices})


def yclose(prices, start=D(2018, 12, 20)):
    return pd.Series(prices, index=pd.bdate_range(start, periods=len(prices)), dtype=float)


FIRST = pd.Timestamp("2005-01-03")


def test_identity_plain_and_mismatch():
    h = hold([10.0, 9.8, 9.9, 10.1])
    ok = yt.identity_ok("A.TO", h, yclose([10.05, 9.8, 9.95, 10.0]), None, FIRST, D(2018, 12, 14))
    assert ok["ok"] and not ok["after_unadjust"]
    reuse = yt.identity_ok("A.TO", h, yclose([55.0, 54.0, 56.0, 57.0]), None, FIRST, D(2018, 12, 14))
    assert not reuse["ok"] and reuse["reason"] == "price_mismatch"


def test_identity_after_one_for_ten_consolidation():
    h = hold([2.0, 2.1, 2.05, 2.0])                       # as traded, before a later 1:10 consolidation
    y = yclose([20.0, 21.0, 20.5, 20.0])                 # Yahoo is adjusted to the post-consolidation basis
    splits = pd.Series({pd.Timestamp("2020-03-02"): 0.1})
    res = yt.identity_ok("A.TO", h, y, splits, FIRST, D(2018, 12, 14))
    assert res["ok"] and res["after_unadjust"]
    assert not yt.identity_ok("A.TO", h, y, None, FIRST, D(2018, 12, 14))["ok"]   # without the split field: fails


def test_identity_two_for_one_and_split_before_date_is_ignored():
    h = hold([100.0, 101.0, 99.0])
    y = yclose([50.0, 50.5, 49.5])
    assert yt.identity_ok("A.TO", h, y, pd.Series({pd.Timestamp("2019-06-03"): 2.0}), FIRST, D(2018, 12, 14))["ok"]
    # a split dated BEFORE the holdings dates is already inside both series: no adjustment
    assert not yt.identity_ok("A.TO", h, y, pd.Series({pd.Timestamp("2018-01-02"): 2.0}), FIRST, D(2018, 12, 14))["ok"]


def test_identity_start_and_overlap_rules():
    h = hold([10.0, 10.0, 10.0])
    y = yclose([10.0, 10.0, 10.0])
    assert yt.identity_ok("A.TO", h, y, None, pd.Timestamp("2019-01-02"), D(2018, 12, 14))["reason"] == "starts_after_D_prev"
    assert yt.identity_ok("A.TO", h, yclose([10.0, 10.0], D(2018, 12, 20)), None, FIRST, D(2018, 12, 14))["reason"] == "insufficient_overlap"
    assert yt.identity_ok("A.TO", h, None, None, None, D(2018, 12, 14))["reason"] == "no_yahoo"


def snap_rows(etf, d, items):
    return [{"etf": etf, "as_of": d, "ticker": t, "ysym": t.replace(".", "-") + ".TO", "name": n, "sector": "x",
             "price": p, "shares": sh, "market_value": p * sh, "weight": 1.0} for t, n, p, sh in items]


def test_eligible_universe_rules():
    d0, r = D(2018, 12, 31), D(2019, 12, 20)
    rows = (snap_rows("XIC", d0, [("AAA", "A", 10.0, 1000.0), ("BBD.A", "B A", 20.0, 100.0), ("BBD.B", "B B", 20.0, 500.0),
                                    ("CRE", "CREATION", 10.0, 1000.0), ("PEN", "PENNY", 3.0, 100.0)])
            + snap_rows("XCS", d0, [("SMALL", "S", 5.0, 100.0)])
            + snap_rows("XIC", r, [("AAA", "A", 5.0, 1000.0), ("BBD.A", "B A", 20.0, 100.0), ("BBD.B", "B B", 20.0, 500.0),
                                   ("CRE", "CREATION", 8.0, 2500.0), ("PEN", "PENNY", 1.5, 100.0), ("NEW", "JOINED", 9.0, 1.0)])
            + snap_rows("XCS", r, [("SMALL", "S", 4.0, 100.0)]))
    elig = yt.eligible_universe(pd.DataFrame(rows), d0, r, 2.0).set_index("ysym")
    assert set(elig.index) == {"AAA.TO", "BBD-B.TO", "CRE.TO", "SMALL.TO"}    # BBD.A deduped, PEN < 2, NEW joined late
    assert not elig.loc["CRE.TO", "guard_ok"] and elig.loc["AAA.TO", "guard_ok"]    # share count x2.5 = fund creation
    assert elig.loc["SMALL.TO", "etf"] == "XCS"
    assert yt.bottom20(elig.reset_index(), n=2) == ["AAA.TO", "SMALL.TO"]           # CRE excluded by the guard; names only


class FakeConn:
    def ping(self, reconnect=False):
        pass


def test_topup_paths_and_resume(tmp_path, monkeypatch):
    from alpha101.data import ingest as ing
    monkeypatch.setattr(yt, "STATE", tmp_path / "state.csv")
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "close": [1.0]})
    calls, written = [], []

    def fake_fetch(sym, pause=1.0):
        calls.append(sym)
        if sym == "BAD.TO":
            raise ValueError("invalid")
        if sym == "DEAD.TO":
            return None
        return frame

    monkeypatch.setattr(ing, "fetch_history", fake_fetch)
    monkeypatch.setattr(ing, "upsert_ohlcv", lambda c, s, df, t: written.append(s) or len(df))
    monkeypatch.setattr(yt.time, "sleep", lambda s: None)
    res = yt.topup(FakeConn(), ["HAVE.TO", "GOOD.TO", "DEAD.TO", "BAD.TO"], {"HAVE.TO"})
    assert calls == ["GOOD.TO", "DEAD.TO", "BAD.TO"] and written == ["GOOD.TO"]
    assert res["no_data"] == ["DEAD.TO"] and res["invalid"] == ["BAD.TO"] and res["written"] == 1 and res["rows"] == 1
    calls.clear()
    yt.topup(FakeConn(), ["HAVE.TO", "GOOD.TO", "DEAD.TO", "BAD.TO"], {"HAVE.TO"})
    assert calls == []                                     # resumable: nothing fetched twice


def test_topup_stops_after_repeated_rate_limits(tmp_path, monkeypatch):
    from alpha101.data import ingest as ing
    monkeypatch.setattr(yt, "STATE", tmp_path / "state.csv")

    def limited(sym, pause=1.0):
        raise ing.RateLimited(sym)

    monkeypatch.setattr(ing, "fetch_history", limited)
    monkeypatch.setattr(yt.time, "sleep", lambda s: None)
    syms = [f"S{i}.TO" for i in range(40)]
    res = yt.topup(FakeConn(), syms, set())
    assert len(res["rate_limited"]) == yt.MAX_CONSECUTIVE_LIMITED


def test_bottom20_on_an_empty_universe():
    elig = yt.eligible_universe(pd.DataFrame(snap_rows("XIC", D(2018, 12, 31), [("AAA", "A", 10.0, 1.0)])),
                                D(2018, 12, 31), D(2019, 12, 20))          # no R file at all
    assert elig.empty and yt.bottom20(elig) == []


def test_guard_normalises_by_fund_wide_creations():
    d0, r = D(2018, 12, 31), D(2019, 12, 20)
    names = [(f"N{i}", f"N{i}", 10.0, 1000.0) for i in range(6)] + [("SPL", "SPLIT", 10.0, 1000.0)]
    after = [(t, n, p, sh * 2.0) for t, n, p, sh in names]          # fund-wide 2x creation moved every holding
    after[-1] = ("SPL", "SPLIT", 10.0, 1000.0 * 2.0 * 0.1)          # ... and one name also had a 1:10 consolidation
    elig = yt.eligible_universe(pd.DataFrame(snap_rows("XIC", d0, names) + snap_rows("XIC", r, after)), d0, r).set_index("ysym")
    assert elig.loc["N0.TO", "guard_ok"] and not elig.loc["SPL.TO", "guard_ok"]


def test_resolve_R_uses_the_latest_file_within_5_days_and_feeds_the_fetch_universe():
    snap = pd.DataFrame({"as_of": [D(2021, 12, 17), D(2021, 12, 17), D(2021, 12, 22)], "etf": ["XIC"] * 3,
                         "ysym": ["A.TO", "B.TO", "A.TO"]})
    assert yt.resolve_R(snap, D(2021, 12, 24)) == D(2021, 12, 22)             # 2 days earlier
    assert yt.resolve_R(snap, D(2021, 12, 29)) == D(2021, 12, 29)             # nothing within 5 days: R itself
    assert yt.resolve_R(snap, D(2021, 12, 20)) == D(2021, 12, 17)
