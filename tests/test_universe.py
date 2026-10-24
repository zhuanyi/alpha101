"""Offline tests for data/universe.py: synthetic panels, fake DB connections, no network."""
import contextlib
import datetime as dt
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha101.data import cache, universe
from alpha101.data.universe import (UNIVERSE_MEMBERSHIP_DDL, UPSERT_SQL, ar_spread, eligible_symbols,
                                    index_membership_events, liquidity_source, liquidity_universe, load_universe,
                                    membership_events, resolve_source, store_events, versioned)
from alpha101.panel import Panel
from alpha101.synthetic import random_panel


def raw_panel(T=500, N=40, seed=0) -> Panel:
    p = random_panel(T=T, N=N, seed=seed)
    fields = dict(p.fields)
    fields["split"] = np.zeros((T, N))
    return Panel(dates=p.dates, symbols=p.symbols, fields=fields).to_raw_basis()


def small_panel(close, volume, start="2015-01-02", symbols=None) -> Panel:
    close = np.asarray(close, dtype=float)
    T, N = close.shape
    dates = pd.bdate_range(start, periods=T)
    syms = pd.Index(symbols or [f"S{i}" for i in range(N)])
    f = {"close": close, "volume": np.asarray(volume, dtype=float), "adj_close": close.copy(),
         "split": np.zeros((T, N))}
    return Panel(dates=dates, symbols=syms, fields=f).to_raw_basis()


def month_end_row(p, d):
    """Last row of the calendar month containing row d."""
    m = p.dates.to_period("M")
    rows = np.flatnonzero(m == m[d])
    return int(rows[-1])


# ---------------------------------------------------------------------------------------- 1. look-ahead
# 301, 324, 345 are last trading days of a month (the decision day itself); 302, 325, 346 are first trading days.
@pytest.mark.parametrize("d", [300, 301, 302, 305, 321, 322, 324, 325, 340, 345, 346, 400])
def test_no_lookahead(d):
    p0 = raw_panel()
    mo = p0.dates.to_period("M")
    if d in (301, 324, 345):
        assert mo[d] != mo[d + 1]      # last trading day of its month
    if d in (302, 325, 346):
        assert mo[d] != mo[d - 1]      # first trading day of its month
    p = raw_panel()
    base = liquidity_universe(p, top_n=10, min_price=5.0)
    rng = np.random.default_rng(d)
    T, N = p.shape
    fields = {}
    for k, v in p.fields.items():
        v = np.array(v, copy=True)
        if k in ("open", "high", "low", "close", "adj_close"):
            v[d + 1:] *= rng.uniform(0.1, 10.0, size=(T - d - 1, N))
            v[d + 1:, ::2] *= 1e-3     # half the names fall far below min_price from d+1 on
        elif k == "volume":
            v[d + 1:] = rng.permutation(v[d + 1:].ravel()).reshape(T - d - 1, N)
        fields[k] = v
    q = Panel(dates=p.dates, symbols=p.symbols, fields=fields)
    pert = liquidity_universe(q, top_n=10, min_price=5.0)
    cut = month_end_row(p, d + 1) + 1
    assert base[:cut].any() and np.array_equal(base[:cut], pert[:cut])


# ---------------------------------------------------------------------------------------- 2. counts
def test_member_count_and_monthly_constancy():
    p = raw_panel()
    mask = liquidity_universe(p, top_n=10, min_price=5.0)
    cnt = mask.sum(axis=1)
    assert cnt.max() <= 10
    months = p.dates.to_period("M")
    for m in months.unique():
        rows = np.flatnonzero(months == m)
        assert (mask[rows] == mask[rows[0]]).all()
    assert (cnt[300:] == 10).all()           # plenty of eligible names after 252 days of history
    assert cnt[:5].sum() == 0 and not mask[:20].any()   # first panel month and warm-up: nobody
    assert liquidity_universe(p, top_n=1000, min_price=0.0)[300:].sum(axis=1).max() <= 40


def test_requires_raw_basis():
    p = random_panel(T=60, N=5)
    with pytest.raises(ValueError, match="raw"):
        liquidity_universe(p, 3, 1.0)


# ---------------------------------------------------------------------------------------- 3. round trip
class EventDB:
    """Fake connection serving universe_membership rows, recording statements."""

    def __init__(self, events=None, existing=None):
        self.events = events
        self.existing = existing or {}
        self.executed, self.many, self.commits = [], [], 0

    def cursor(self):
        return EventCur(self)

    def commit(self):
        self.commits += 1


class EventCur:
    def __init__(self, db):
        self.db, self.rows = db, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.db.executed.append((sql, params))
        if sql.startswith("SELECT symbol, as_of, in_universe"):
            e = self.db.events[self.db.events["source"] == params[0]].sort_values(["symbol", "as_of"])
            self.rows = [(r.symbol, r.as_of, r.in_universe) for r in e.itertuples()]
        elif sql.startswith("SELECT symbol, as_of FROM"):
            self.rows = [(s, d) for (s, d) in self.db.existing.get(params[0], [])]

    def executemany(self, sql, data):
        self.db.many.append((sql, list(data)))

    def fetchall(self):
        return self.rows


def test_roundtrip_events_to_mask():
    p = raw_panel()
    mask = liquidity_universe(p, top_n=10, min_price=5.0)
    ev = membership_events(mask, p.dates, p.symbols, "us", "liquidity_top500_us")
    assert list(ev.columns) == ["symbol", "market", "source", "as_of", "in_universe"]
    assert set(ev["market"]) == {"US"} and isinstance(ev["as_of"].iloc[0], dt.date)
    assert set(ev["in_universe"]) <= {0, 1}
    assert ev.groupby("symbol").size().min() >= 1 and len(ev) < mask.size
    back = load_universe(EventDB(ev), p, "liquidity_top500_us")
    assert back.dtype == bool and np.array_equal(back, mask)


def test_load_universe_edge_cases():
    p = small_panel(np.ones((6, 3)), np.ones((6, 3)))
    d = p.dates
    ev = pd.DataFrame([("S0", "US", "x", d[2].date(), 1), ("S0", "US", "x", d[4].date(), 0),
                       ("ZZZ", "US", "x", d[0].date(), 1)],
                      columns=["symbol", "market", "source", "as_of", "in_universe"])
    m = load_universe(EventDB(ev), p, "x")
    assert m[:, 0].tolist() == [False, False, True, True, False, False]   # before first event: False
    assert not m[:, 1:].any()                                              # no events / not in panel
    assert not load_universe(EventDB(ev), p, "other").any()


# ---------------------------------------------------------------------------------------- 4. walk-back
def _ev(df):
    return [(r.symbol, r.as_of.isoformat(), r.in_universe) for r in df.itertuples()]


def test_index_walkback_toy():
    current = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "BRK.B"]})
    changes = pd.DataFrame({"date": pd.to_datetime(["2020-03-01", "2019-06-15", "2010-01-04", "2004-01-01"]),
                            "added": ["CCC", "XXX", "BRK.B", "OLD"],
                            "removed": ["XXX", None, "DDD", "ZZZ"]})
    # since 2005-01-01: the 2004 row is ignored. Walk back: 2020-03-01 undoes CCC added / XXX removed
    # (XXX joins), 2019-06-15 undoes XXX added (XXX leaves again), 2010-01-04 undoes BRK-B added / DDD removed.
    ev = index_membership_events(current, changes, "us", "sp500_wiki_changes", "2005-01-01")
    assert _ev(ev) == [
        ("AAA", "2005-01-01", 1),
        ("BBB", "2005-01-01", 1),
        ("BRK-B", "2005-01-01", 0), ("BRK-B", "2010-01-04", 1),
        ("CCC", "2005-01-01", 0), ("CCC", "2020-03-01", 1),
        ("DDD", "2005-01-01", 1), ("DDD", "2010-01-04", 0),
        ("XXX", "2005-01-01", 0), ("XXX", "2019-06-15", 1), ("XXX", "2020-03-01", 0),
    ]
    assert set(ev["market"]) == {"US"} and (ev["source"] == "sp500_wiki_changes").all()
    assert "OLD" not in set(ev["symbol"])


def test_index_walkback_anomaly_warns(caplog):
    current = pd.DataFrame({"symbol": ["AAA"]})
    changes = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "added": ["NOPE"], "removed": ["AAA"]})
    with caplog.at_level(logging.WARNING, logger="alpha101.universe"):
        ev = index_membership_events(current, changes, "us", "s", "2005-01-01")
    assert any("2 walk-back anomalies" in r.getMessage() for r in caplog.records)
    # anomalies are counted but emit no event and leave the set unchanged
    assert _ev(ev) == [("AAA", "2005-01-01", 1)]


def _state_on(ev, sym, date):
    g = ev[(ev["symbol"] == sym) & (ev["as_of"] <= pd.Timestamp(date).date())].sort_values("as_of")
    return bool(len(g) and g["in_universe"].iloc[-1])


def test_index_walkback_reused_ticker_att_pattern():
    # Today's list holds T (AT&T Inc.); the 2005-11-18 row removes the old T (AT&T Corp.) that SBC then reused.
    current = pd.DataFrame({"symbol": ["T", "AAA"]})
    changes = pd.DataFrame({"date": pd.to_datetime(["2005-11-18", "2010-05-03"]),
                            "added": ["NEW", "AAA"], "removed": ["T", "OLD"]})
    ev = index_membership_events(current, changes, "us", "sp500_wiki_changes", "2005-01-01")
    for d in pd.bdate_range("2005-01-03", "2026-01-02", freq="90D"):
        assert _state_on(ev, "T", d), d
    assert _state_on(ev, "T", "2005-01-01") and _state_on(ev, "T", "2005-11-18")
    assert (ev[ev["symbol"] == "T"]["in_universe"] == 1).all()


def _replay_final(ev):
    last = ev.sort_values(["symbol", "as_of"]).groupby("symbol")["in_universe"].last()
    return set(last.index[last == 1])


@pytest.mark.parametrize("consistent", [True, False])
def test_index_walkback_forward_replay_ends_at_today(consistent):
    rng = np.random.default_rng(7 if consistent else 11)
    tickers = list("ABCDEFGH")
    for trial in range(200):
        n = int(rng.integers(0, 9))
        dates = pd.to_datetime("2006-01-01") + pd.to_timedelta(rng.integers(0, 12, size=n) * 30, unit="D")
        if consistent:   # simulate forward from a start set; today's list is the end state
            state = set(rng.choice(tickers, size=int(rng.integers(0, 6)), replace=False))
            rows = []
            for d in sorted(dates):
                out_ = [t for t in tickers if t in state]
                inn = [t for t in tickers if t not in state]
                a = str(rng.choice(inn)) if inn and rng.random() < 0.8 else None
                r = str(rng.choice(out_)) if out_ and rng.random() < 0.8 else None
                if a == r and a is not None:
                    a = None
                if a:
                    state.add(a)
                if r:
                    state.discard(r)
                rows.append((d, a, r))
            current_set = state
        else:            # random rows, unrelated to today's list
            rows = [(d, rng.choice(tickers + [None]), rng.choice(tickers + [None])) for d in dates]
            rows = [(d, None if a is None else str(a), None if r is None else str(r)) for d, a, r in rows]
            current_set = set(rng.choice(tickers, size=int(rng.integers(0, 7)), replace=False))
        current = pd.DataFrame({"symbol": sorted(current_set)}, dtype=object)
        changes = pd.DataFrame(rows, columns=["date", "added", "removed"])
        changes["date"] = pd.to_datetime(changes["date"])
        ev = index_membership_events(current, changes, "us", "s", "2005-01-01")
        assert _replay_final(ev) == set(current_set), (trial, rows, sorted(current_set))


def test_index_current_only_and_ca_mapping(caplog):
    cur = pd.DataFrame({"symbol": ["RY", "BBD.B", "bad sym", "NA"]})
    empty = pd.DataFrame(columns=["date", "added", "removed"])
    with caplog.at_level(logging.WARNING, logger="alpha101.universe"):
        ev = index_membership_events(cur, empty, "ca", "tsx_wiki_current", "2005-01-01")
    assert _ev(ev) == [("BBD-B.TO", "2005-01-01", 1), ("NA.TO", "2005-01-01", 1), ("RY.TO", "2005-01-01", 1)]
    assert set(ev["market"]) == {"CA"}
    assert any("dropping index ticker" in r.getMessage() for r in caplog.records)


def test_index_same_date_forward_order_and_since_date():
    cur = pd.DataFrame({"symbol": ["AAA"]})
    # Same date, table newest first: the "added AAA" row is listed first, hence last in forward order,
    # so AAA ends that day as a member.
    ch = pd.DataFrame({"date": pd.to_datetime(["2006-02-01", "2006-02-01"]),
                       "added": ["AAA", None], "removed": [None, "AAA"]})
    ev = index_membership_events(cur, ch, "us", "s", "2006-02-01")
    assert _ev(ev) == [("AAA", "2006-02-01", 1)]


# ---------------------------------------------------------------------------------------- 5. eligibility
def _dates_flat(T):
    return pd.bdate_range("2015-01-02", periods=T)


def _mask_last_month(p, **kw):
    m = liquidity_universe(p, **kw)
    return m[-1]


def test_min_history_boundary():
    T = 330
    dates = _dates_flat(T)
    close = np.full((T, 2), 10.0)
    vol = np.full((T, 2), 1000.0)
    months = dates.to_period("M")
    last_start = max(i for i in range(1, T) if months[i] != months[i - 1])
    e = last_start - 1
    close[:e + 1 - 251, 0] = np.nan       # exactly 251 finite closes in rows 0..e for S0
    close[:e + 1 - 252, 1] = np.nan       # exactly 252 finite closes for S1
    p = small_panel(close, vol)
    assert int(np.isfinite(p.get("close")[:e + 1, 0]).sum()) == 251
    assert int(np.isfinite(p.get("close")[:e + 1, 1]).sum()) == 252
    m = liquidity_universe(p, top_n=5, min_price=1.0)[-1]
    assert m.tolist() == [False, True]


def test_min_price_at_decision_day():
    T = 300
    close = np.full((T, 3), 10.0)
    vol = np.full((T, 3), 1000.0)
    dates = _dates_flat(T)
    months = dates.to_period("M")
    last_start = max(i for i in range(1, T) if months[i] != months[i - 1])
    e = last_start - 1
    close[e, 0] = 4.99          # below the floor on the decision day only
    close[e + 1:, 1] = 1.0      # cheap only AFTER the decision day: irrelevant
    close[e, 2] = np.nan        # no close on the decision day
    p = small_panel(close, vol)
    m = liquidity_universe(p, top_n=5, min_price=5.0, min_history=100)
    assert m[-1].tolist() == [False, True, False]


def test_eligible_flag_ties_nan_volume_and_first_month():
    T = 300
    dates = _dates_flat(T)
    close = np.full((T, 5), 10.0)
    vol = np.full((T, 5), 1000.0)
    vol[:, 0] = 1e9                        # best score but not eligible
    vol[:, 4] = np.nan                     # all-NaN volume window
    p = small_panel(close, vol, symbols=["ZED", "BBB", "AAA", "CCC", "NANV"])
    elig = np.array([False, True, True, True, True])
    m = liquidity_universe(p, top_n=2, min_price=1.0, min_history=100, eligible=elig)
    assert not m[:, 0].any() and not m[:, 4].any()
    last = m[-1]
    assert last.sum() == 2
    # BBB, AAA, CCC tie on score: the two smallest symbols win
    assert sorted(s for s, f in zip(p.symbols, last) if f) == ["AAA", "BBB"]
    months = dates.to_period("M")
    first_month_rows = np.flatnonzero(months == months[0])
    assert not m[first_month_rows].any()
    assert m[first_month_rows[-1] + 1:].any()


# ---------------------------------------------------------------------------------------- 6. eligible_symbols
class PoolDB:
    def __init__(self, counts, master):
        self.counts, self.master = counts, master

    def cursor(self):
        return PoolCur(self)


class PoolCur:
    def __init__(self, db):
        self.db, self.rows = db, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        assert "%s" in sql and ("FROM yahoo_ohlcv" in sql or "FROM security_master" in sql)
        if "FROM yahoo_ohlcv" in sql:
            self.rows = [(s, *self.db.counts[s]) for s in params if s in self.db.counts]
        else:
            self.rows = [(s, *self.db.master[s]) for s in params if s in self.db.master]

    def fetchall(self):
        return self.rows


def test_eligible_symbols_rules():
    d = pd.Timestamp("2010-06-01")
    pool = pd.DataFrame({
        "symbol": ["OK", "NODATA", "ETF", "NOMASTER", "REUSED", "REUSEDOK", "SHORT19", "SHORT20", "CAOK"],
        "market": ["US"] * 8 + ["CA"],
        "source": ["sp500_current", "sp500_current", "sp500_current", "sp500_current", "sp500_changes",
                   "sp500_changes", "sp500_current", "sp500_current", "tsx_current"],
        "change_date": [pd.NaT] * 4 + [d, d, pd.NaT, pd.NaT, pd.NaT],
        "wiki_names": [()] * 4 + [("Reused Corp",), ("Reused Corp",), (), (), ()],
    })
    counts = {"OK": (500, pd.Timestamp("2005-01-03")), "ETF": (500, d), "NOMASTER": (500, d),
              "REUSED": (500, pd.Timestamp("2015-01-02")), "REUSEDOK": (500, pd.Timestamp("2005-01-03")),
              "SHORT19": (19, d), "SHORT20": (20, d), "CAOK": (300, d)}
    master = {"OK": ("Ok Inc", "EQUITY"), "ETF": ("An ETF", "ETF"), "REUSED": ("Reused Corp", "EQUITY"),
              "REUSEDOK": ("Reused Corp", "EQUITY"), "SHORT19": ("S", "EQUITY"), "SHORT20": ("S", "EQUITY"),
              "CAOK": ("Ca", "EQUITY")}
    elig, info = eligible_symbols(PoolDB(counts, master), pool, "us")
    assert elig == ["OK", "REUSEDOK", "SHORT20"]
    assert info["no_data"] == ["NODATA"]
    assert info["not_equity"] == ["ETF", "NOMASTER"]
    assert list(info["reused_ticker"]) == ["REUSED"] and "after change date" in info["reused_ticker"]["REUSED"]
    assert info["short_history"] == ["SHORT19"]
    assert info["pool"] == 8 and info["eligible"] == 3
    elig_ca, info_ca = eligible_symbols(PoolDB(counts, master), pool, "ca")
    assert elig_ca == ["CAOK"] and info_ca["pool"] == 1


# ---------------------------------------------------------------------------------------- 7. store_events
def test_upsert_sql_pure_placeholders():
    assert UPSERT_SQL.count("%s") == 5
    m = re.search(r"VALUES\s*\(([^)]*)\)", UPSERT_SQL)
    assert [t.strip() for t in m.group(1).split(",")] == ["%s"] * 5
    assert len(re.findall(r"VALUES\s*\(%s", UPSERT_SQL)) == 1
    assert "NOW(" not in UPSERT_SQL.upper() and "DELETE" not in UPSERT_SQL.upper()
    assert "ON DUPLICATE KEY UPDATE" in UPSERT_SQL


def test_ddl_matches_design_md():
    design = (Path(__file__).resolve().parents[1] / "docs" / "design.md").read_text()
    m = re.search(r"CREATE TABLE IF NOT EXISTS universe_membership \(.*?;", design, re.S)
    assert m and UNIVERSE_MEMBERSHIP_DDL == m.group(0)


def _events(n, source="s"):
    return pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "market": "US", "source": source,
                         "as_of": dt.date(2010, 1, 4), "in_universe": 1})


def test_store_chunking():
    db = EventDB(existing={})
    n = store_events(db, _events(12000))
    assert n == 12000
    assert [len(d) for _, d in db.many] == [5000, 5000, 2000] and db.commits == 3
    assert all(sql == UPSERT_SQL for sql, _ in db.many)
    assert db.executed[0][0] == UNIVERSE_MEMBERSHIP_DDL
    s, mk, src, asof, st = db.many[0][1][0]
    assert (type(s), type(mk), type(src), type(asof), type(st)) == (str, str, str, dt.date, int)


def test_store_stale_key_guard():
    ev = _events(3)
    db = EventDB(existing={"s": [("S0", dt.date(2010, 1, 4)), ("GONE", dt.date(2010, 1, 4))]})
    with pytest.raises(RuntimeError, match="source 's': 1"):
        store_events(db, ev)
    assert db.many == [] and db.commits == 0
    ok = EventDB(existing={"s": [("S0", dt.date(2010, 1, 4))]})   # existing subset of new: fine
    assert store_events(ok, ev) == 3 and len(ok.many) == 1


def test_store_stale_asof_for_known_symbol():
    ev = _events(3)      # S0..S2 at 2010-01-04
    db = EventDB(existing={"s": [("S0", dt.date(2010, 1, 4)), ("S1", dt.date(2010, 2, 1))]})
    with pytest.raises(RuntimeError, match="source 's': 1"):    # S1 is known, but not at 2010-02-01
        store_events(db, ev)
    assert db.many == [] and db.commits == 0


@pytest.mark.parametrize("make", [
    lambda d: pd.Timestamp(d),
    lambda d: np.datetime64(d),
    lambda d: dt.datetime(d.year, d.month, d.day),
    lambda d: d,
])
def test_store_normalises_asof_to_date(make):
    base = _events(3)
    ev = base.copy()
    ev["as_of"] = pd.Series([make(dt.date(2010, 1, 4)) for _ in range(3)], index=ev.index)
    db = EventDB(existing={"s": [("S0", dt.date(2010, 1, 4))]})
    assert store_events(db, ev) == 3
    assert {type(r[3]) for r in db.many[0][1]} == {dt.date}
    assert {r[3] for r in db.many[0][1]} == {dt.date(2010, 1, 4)}


def test_store_datetime64_column():
    ev = _events(3)
    ev["as_of"] = pd.to_datetime(ev["as_of"])
    assert str(ev["as_of"].dtype).startswith("datetime64")
    db = EventDB(existing={"s": [("S0", dt.date(2010, 1, 4))]})
    assert store_events(db, ev) == 3
    assert all(type(r[3]) is dt.date for r in db.many[0][1])


# ---------------------------------------------------------------------------------------- 8. load_panel
def test_liquidity_source_names():
    assert liquidity_source("us") == "liquidity_top500_us" and liquidity_source("CA") == "liquidity_top200_ca"
    assert liquidity_source("us", 100) == "liquidity_top100_us"
    with pytest.raises(ValueError):
        liquidity_source("uk")


def test_load_panel_liquidity(monkeypatch, tmp_path):
    import test_cache as tc
    fake = tc.Fake()
    fake.install(monkeypatch)
    start = tc.START
    got = {}

    def fake_load(conn, p, source):
        got["source"] = source
        return np.ones(p.shape, dtype=bool)
    monkeypatch.setattr(universe, "load_universe", fake_load)
    monkeypatch.setattr(universe, "resolve_source", lambda conn, s: s if "@" in s else s + "@20261019")
    p = cache.load_panel("us", start=start, cache_dir=tmp_path, universe_source="liquidity")
    assert got["source"] == "liquidity_top500_us@20261019"
    k = int((p.dates < pd.Timestamp(start)).sum())
    assert not p.universe[:k].any() and p.universe[k:].all()
    cache.load_panel("us", start=start, cache_dir=tmp_path, universe_source="my_source")
    assert got["source"] == "my_source@20261019"
    cache.load_panel("us", start=start, cache_dir=tmp_path, universe_source="my_source@20250101")
    assert got["source"] == "my_source@20250101"
    monkeypatch.setattr(universe, "load_universe", lambda conn, p, s: np.zeros(p.shape, dtype=bool))
    with pytest.raises(ValueError, match="no members"):
        cache.load_panel("us", start=start, cache_dir=tmp_path, universe_source="liquidity")


# ---------------------------------------------------------------------------------------- 9. spread screen (amend-1)
def _roll_sim(spread, D=3000, steps=390, seed=2, sigma=0.02):   # seed 2: the s=0.005 estimate is noisy (sd ~ 0.001)
    """Daily high/low/close from an intraday efficient log price traded at bid or ask with prob 1/2."""
    rng = np.random.default_rng(seed)
    eff = np.cumsum(rng.normal(0.0, sigma / np.sqrt(steps), D * steps)).reshape(D, steps)
    obs = eff + spread / 2.0 * rng.choice([-1.0, 1.0], size=(D, steps))
    return (np.exp(obs.max(axis=1))[:, None], np.exp(obs.min(axis=1))[:, None], np.exp(obs[:, -1])[:, None])


@pytest.mark.parametrize("s", [0.005, 0.01, 0.02])
def test_ar_spread_recovers_roll_spread(s):
    h, l, c = _roll_sim(s)
    est = ar_spread(h, l, c, window=len(c) - 1)[-1, 0]
    assert est == pytest.approx(s, rel=0.10)


def test_ar_spread_zero_spread():
    h, l, c = _roll_sim(0.0)
    assert ar_spread(h, l, c, window=len(c) - 1)[-1, 0] < 0.002


def _hlc(T=40, N=3, seed=3):
    rng = np.random.default_rng(seed)
    mid = 50 * np.exp(np.cumsum(rng.normal(0, 0.002, (T, N)), axis=0))
    hi, lo = mid * 1.02, mid * 0.98
    return hi, lo, lo + rng.uniform(0.0, 1.0, (T, N)) * (hi - lo)


def test_ar_spread_split_pair_excluded_and_hand_value():
    h, l, c = _hlc()
    T = len(c)
    age = np.full(c.shape, 5.0)
    age[20] = 0.0                                    # the pair (19, 20) straddles a split
    lc, eta = np.log(c), (np.log(h) + np.log(l)) / 2
    pair = np.full(c.shape, np.nan)
    pair[1:] = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    got = ar_spread(h, l, c, window=T, split_age=age, min_frac=0.5)[-1]
    want = np.sqrt(np.maximum(4 * np.nanmean(np.where(np.arange(T)[:, None] == 20, np.nan, pair), axis=0), 0))
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert not np.allclose(got, ar_spread(h, l, c, window=T, min_frac=0.5)[-1])


def test_ar_spread_no_lookahead_and_min_frac():
    h, l, c = _hlc()
    base = ar_spread(h, l, c, window=10)
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    for a in (h2, l2, c2):
        a[25:] *= 3.7
    np.testing.assert_array_equal(base[:25], ar_spread(h2, l2, c2, window=10)[:25])
    assert np.isnan(base[:8]).all() and np.isfinite(base[8:]).all()     # needs ceil(.8*10)=8 finite pairs (rows 1..8)
    c3 = c.copy()
    c3[12:30:2] = np.nan                                                 # half of the pairs in the window go
    assert np.isnan(ar_spread(h, l, c3, window=10)[25]).all()
    assert np.isfinite(ar_spread(h, l, c3, window=10, min_frac=0.3)[25]).all()


def _spread_panel(T=300):
    """S0 wide spread, highest dollar volume; S1 NaN highs; S2..S4 zero spread, volumes 3, 2, 1."""
    mid = np.full((T, 5), 50.0)
    w = 0.05
    close = mid.copy()
    close[::2, 0] = 50 * (1 + w)
    close[1::2, 0] = 50 * (1 - w)
    high, low = mid * (1 + w), mid * (1 - w)
    high[:, 1] = np.nan
    high[:, 2:], low[:, 2:], close[:, 2:] = mid[:, 2:] * 1.001, mid[:, 2:] * 0.999, mid[:, 2:]
    close[:, 1] = 50.0
    vol = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]) * 1e5, (T, 1))
    f = {"open": mid, "high": high, "low": low, "close": close, "volume": vol, "adj_close": close.copy(),
         "split": np.zeros((T, 5))}
    return Panel(dates=pd.bdate_range("2015-01-02", periods=T), symbols=pd.Index([f"S{i}" for i in range(5)]),
                 fields=f).to_raw_basis()


def test_spread_screen_replaces_wide_and_nan_names():
    p = _spread_panel()
    plain = liquidity_universe(p, top_n=2, min_price=5.0, min_history=50)
    assert plain[-1].tolist() == [True, True, False, False, False]
    scr = liquidity_universe(p, top_n=2, min_price=5.0, min_history=50, max_spread=0.02)
    assert scr[-1].tolist() == [False, False, True, True, False]      # S0 wide, S1 NaN spread: next names enter


def test_spread_screen_no_lookahead():
    p = raw_panel()
    ms = 0.2
    base = liquidity_universe(p, top_n=10, min_price=5.0, max_spread=ms)
    d = 340
    rng = np.random.default_rng(1)
    T, N = p.shape
    fields = {}
    for k, v in p.fields.items():
        v = np.array(v, copy=True)
        if k in ("open", "high", "low", "close", "adj_close"):
            v[d + 1:] *= rng.uniform(0.5, 2.0, size=(T - d - 1, N))
        fields[k] = v
    pert = liquidity_universe(Panel(dates=p.dates, symbols=p.symbols, fields=fields), top_n=10, min_price=5.0,
                              max_spread=ms)
    cut = month_end_row(p, d + 1) + 1
    assert base[:cut].any() and np.array_equal(base[:cut], pert[:cut])


def test_top_n_by_month():
    p = raw_panel()
    months = p.dates.to_period("M")
    uniq = list(months.unique())
    K = {m: (i * 3) % 14 for i, m in enumerate(uniq)}
    absent = uniq[-3]
    del K[absent]
    mask = liquidity_universe(p, top_n=10, min_price=5.0, top_n_by_month=K)
    for m in uniq:
        rows = np.flatnonzero(months == m)
        if rows[0] < 300:
            continue
        want = min(10, K.get(m, 0))
        assert (mask[rows].sum(axis=1) == want).all(), m
    assert mask[np.flatnonzero(months == absent)].sum() == 0


# ---------------------------------------------------------------------------------------- 10. versioned sources
class SrcCur:
    def __init__(self, sources):
        self.sources, self.rows = sources, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "LIKE" in sql:
            like = params[0]
            rx = "^" + "".join("." if ch == "_" else ".*" if ch == "%" else re.escape(ch) for ch in like) + "$"
            self.rows = [(x,) for x in sorted(set(self.sources)) if re.match(rx, x)]
        else:
            self.rows = [(1,)] if params[0] in self.sources else []

    def fetchall(self):
        return self.rows


class SrcDB:
    def __init__(self, sources):
        self.sources = sources

    def cursor(self):
        return SrcCur(self.sources)


def test_versioned_names():
    assert versioned("liquidity_top500_us", "20261019") == "liquidity_top500_us@20261019"
    assert versioned("x", "20261019b") == "x@20261019b"
    for bad in ("2026-10-19", "2026101", "20261019bb", "", "20261019B"):
        with pytest.raises(ValueError):
            versioned("x", bad)
    with pytest.raises(ValueError):
        versioned("x@20261019", "20261020")
    with pytest.raises(ValueError):
        versioned("y" * 32, "20261019")      # 32 + 9 = 41 > 40
    assert len(versioned("y" * 31, "20261019")) == 40


def test_resolve_source():
    db = SrcDB({"liquidity_top500_us", "liquidity_top500_us@20260101", "liquidity_top500_us@20261019",
                "liquidity_top500_us@20261019a", "liquidity_top5X0_us@20300101", "liquidity_top500_usX@20300101",
                "liquidity_top200_ca"})
    assert resolve_source(db, "liquidity_top500_us") == "liquidity_top500_us@20261019a"   # latest; wildcard rows ignored
    assert resolve_source(db, "liquidity_top500_us@20260101") == "liquidity_top500_us@20260101"
    assert resolve_source(db, "liquidity_top200_ca") == "liquidity_top200_ca"            # legacy bare name
    with pytest.raises(ValueError, match="no stored source"):
        resolve_source(db, "nope")
    with pytest.raises(ValueError):
        resolve_source(db, "liquidity_top500_us@20990101")
    assert resolve_source(SrcDB({"s@bad", "s"}), "s") == "s"                             # malformed version ignored


def test_store_rejects_long_source():
    db = EventDB(existing={})
    with pytest.raises(ValueError, match="longer than 40"):
        store_events(db, _events(2, source="x" * 41))
    assert db.executed == [] and db.many == []
    assert store_events(db, _events(2, source="x" * 40)) == 2


# ---------------------------------------------------------------------------------------- 11. build_universe
def test_ar_spread_zero_volume_pairs_removed():
    h, l, c = _hlc()
    T = len(c)
    vol = np.full(c.shape, 1e6)
    stale = slice(10, 15)
    vol[stale] = 0.0
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[stale], l2[stale], c2[stale] = c[10], c[10], c[10]     # stale O=H=L=C block
    got = ar_spread(h2, l2, c2, window=T, volume=vol, min_frac=0.5)[-1]
    # reference: same data with the pairs touching rows 10..14 (pairs dated 10..15) removed via NaN prices
    h3, l3, c3 = h.copy(), l.copy(), c.copy()
    h3[stale] = l3[stale] = c3[stale] = np.nan
    ref = ar_spread(h3, l3, c3, window=T, min_frac=0.5)[-1]
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    assert not np.allclose(got, ar_spread(h2, l2, c2, window=T, min_frac=0.5)[-1])


@pytest.mark.parametrize("bad", [0, -0.01, float("nan"), float("inf"), "x", True])
def test_build_universe_rejects_bad_max_spread(bad):
    with pytest.raises(ValueError, match="max_spread"):
        universe.build_universe("us", max_spread=bad)


def test_build_universe_versioned_sources_and_cacap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    us, ca = raw_panel(T=500, N=12, seed=0), raw_panel(T=500, N=12, seed=1)
    since = str(us.dates[0].date())
    syms = list(us.symbols)

    @contextlib.contextmanager
    def rd():
        yield object()
    stored = {}
    monkeypatch.setattr(universe, "reader", rd)
    monkeypatch.setattr(universe, "writer", lambda: type("C", (), {"close": lambda self: None})())
    monkeypatch.setattr(universe, "store_events", lambda conn, ev: stored.setdefault("ev", ev) is ev and len(ev))
    monkeypatch.setattr(universe, "candidate_pool", lambda html, since=None: pd.DataFrame())
    monkeypatch.setattr(universe, "eligible_symbols",
                        lambda conn, pool, market: (syms, {"pool": 12, "eligible": 12}))
    monkeypatch.setattr(cache, "load_panel", lambda m, **kw: us if m == "us" else ca)

    def res_src(conn, s):
        if "@" in s:
            raise ValueError("none")
        return s
    monkeypatch.setattr(universe, "resolve_source", res_src)
    camask = np.zeros(ca.shape, dtype=bool)
    camask[:, :3] = True
    monkeypatch.setattr(universe, "load_universe", lambda conn, p, s: camask)
    monkeypatch.setattr(universe.constituents, "fetch_sp500", lambda html: (
        pd.DataFrame({"symbol": syms[:6]}), pd.DataFrame(columns=["date", "added", "removed"])))
    res = universe.build_universe("us", since=since, top_n=5, min_price=1.0, version="20261019", max_spread=0.05)
    names = set(stored["ev"]["source"])
    assert names == {"liquidity_top5_us@20261019", "liquidity_cacap_us@20261019",
                     "sp500_wiki_current@20261019", "sp500_wiki_changes@20261019"}
    assert res["cacap"]["ca_source"] == "liquidity_top200_ca" and res["cacap"]["member_count_range"][1] == 3
    assert res["screen"] == {"estimator": "abdi_ranaldo_2017", "max_spread": 0.05, "spread_window": 63}
    assert set(res["screen_removed_by_year"]) == set(int(y) for y in us.dates.year.unique())
    assert all(v["mean_daily_removed"] >= 0 and v["symbols_removed"] >= 0 for v in res["screen_removed_by_year"].values())
    # a tight bar removes names; no screen removes none
    tight = universe.build_universe("us", since=since, top_n=5, min_price=1.0, version="20261019", max_spread=1e-9)
    assert any(v["symbols_removed"] > 0 for v in tight["screen_removed_by_year"].values())
    none = universe.build_universe("us", since=since, top_n=5, min_price=1.0, version="20261019", max_spread=None)
    assert all(v["symbols_removed"] == 0 for v in none["screen_removed_by_year"].values())
    monkeypatch.setattr(universe, "resolve_source", lambda conn, s: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(RuntimeError, match="build the CA universe first"):
        universe.build_universe("us", since=since, top_n=5, min_price=1.0, version="20261019")
