"""Offline tests for data/cache.py: fake DB layer, no network."""
import contextlib
import pathlib

import numpy as np
import pandas as pd
import pytest

from alpha101.data import cache
from alpha101.data.cache import WARMUP_ROWS, load_long, load_panel
from alpha101.data.yahoo import COLUMNS

DATES = pd.bdate_range("2012-01-02", periods=700)
START = str(DATES[520].date())
SPLIT_I = 600


def _rows(sym, dates, base, split_at=None):
    out = []
    for i, d in enumerate(dates):
        c = base + 0.01 * i
        out.append((sym, d, c, c + 1, c - 1, c, c, 1000.0 + i, 0.0, 2.0 if i == split_at else 0.0))
    return out


def _long():
    rows = _rows("^GSPC", DATES, 100.0)
    rows += _rows("AAA", DATES, 10.0, split_at=SPLIT_I)
    rows += _rows("BBB", DATES, 20.0)
    rows += _rows("CCC", DATES, 30.0)
    rows += _rows("ETF1", DATES, 40.0)
    rows += _rows("ZZZ", DATES, 50.0)  # not in master at all
    return pd.DataFrame(rows, columns=COLUMNS)


def _master():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "ETF1"],
        "quote_type": ["EQUITY", "EQUITY", "EQUITY", "ETF"],
        "sector": ["Tech", "Tech", None, "Fund"],
        "industry": ["Soft", "Hard", None, "Fund"],
        "shares_outstanding": [1000.0, 2000.0, np.nan, 5.0]})


class Fake:
    def __init__(self, long=None):
        self.long = _long() if long is None else long
        self.reads = 0
        self.fp = {"rows": 1}

    def install(self, mp):
        @contextlib.contextmanager
        def rd():
            yield object()
        mp.setattr(cache, "reader", rd)
        mp.setattr(cache, "db_fingerprint", lambda conn, m: dict(self.fp))
        mp.setattr(cache, "_read_master", lambda conn, m: _master())

        def rl(conn, symbols=None, start=None, end=None):
            self.reads += 1
            d = self.long
            if symbols:
                d = d[d["symbol"].isin(symbols)]
            if start:
                d = d[d["date"] >= pd.Timestamp(start)]
            return d.sort_values(["symbol", "date"]).reset_index(drop=True)
        mp.setattr(cache, "read_long", rl)


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    f.install(monkeypatch)
    return f


def panel(tmp_path, **kw):
    kw.setdefault("universe_source", None)
    return load_panel("us", start=START, cache_dir=tmp_path, **kw)


def test_hit_and_files(fake, tmp_path):
    a = load_long("us", "2012-01-01", cache_dir=tmp_path)
    assert fake.reads == 1
    assert (tmp_path / "yahoo_us.parquet").exists() and (tmp_path / "yahoo_us.json").exists()
    b = load_long("us", "2012-01-01", cache_dir=tmp_path)
    assert fake.reads == 1
    pd.testing.assert_frame_equal(a, b)
    assert set(a["symbol"]) == {"AAA", "BBB", "CCC", "^GSPC"}


def test_invalidation(fake, tmp_path):
    load_long("us", "2013-01-01", cache_dir=tmp_path)
    fake.fp = {"rows": 2}
    load_long("us", "2013-01-01", cache_dir=tmp_path)
    assert fake.reads == 2
    load_long("us", "2013-01-01", refresh=True, cache_dir=tmp_path)
    assert fake.reads == 3
    load_long("us", "2012-06-01", cache_dir=tmp_path)  # earlier than cached
    assert fake.reads == 4
    late = load_long("us", "2013-06-03", cache_dir=tmp_path)  # later: hit, filtered
    assert fake.reads == 4 and late["date"].min() >= pd.Timestamp("2013-06-03")


def test_cache_equals_db(fake, tmp_path):
    a = panel(tmp_path, refresh=True)
    r = fake.reads
    b = panel(tmp_path)
    assert fake.reads == r
    assert a.dates.equals(b.dates) and a.symbols.equals(b.symbols)
    assert a.fields.keys() == b.fields.keys()
    for k in a.fields:
        assert np.array_equal(a.fields[k], b.fields[k], equal_nan=True), k
    assert np.array_equal(a.universe, b.universe)
    assert a.groups.keys() == b.groups.keys()
    for k in a.groups:
        assert np.array_equal(a.groups[k], b.groups[k])


def test_warmup_and_universe(fake, tmp_path):
    long = pd.concat([fake.long, pd.DataFrame(_rows("AAA", [pd.Timestamp("2014-05-24")], 9.0), columns=COLUMNS)])
    fake.long = long  # a Saturday: not a calendar date
    p = panel(tmp_path)
    i0 = int(DATES.searchsorted(pd.Timestamp(START)))
    assert p.dates[0] == DATES[i0 - WARMUP_ROWS]
    assert p.dates.equals(DATES[i0 - WARMUP_ROWS:])
    k = WARMUP_ROWS
    assert not p.universe[:k].any() and p.universe[k:].all()
    assert pd.Timestamp("2014-05-24") not in p.dates


def test_equities_only(fake, tmp_path):
    p = panel(tmp_path)
    assert list(p.symbols) == ["AAA", "BBB", "CCC"]


def test_split_guard(fake, tmp_path):
    base = fake.long

    def with_split(date, close=None, sym="BBB"):
        d = base.copy()
        if close is None:
            extra = pd.DataFrame(_rows(sym, [pd.Timestamp(date)], 5.0), columns=COLUMNS)
            extra["split"] = 2.0
            return pd.concat([d, extra])
        m = (d["symbol"] == sym) & (d["date"] == pd.Timestamp(date))
        d.loc[m, "split"] = 2.0
        d.loc[m, "close"] = close
        return d
    fake.long = with_split("2014-05-24")  # Saturday
    with pytest.raises(ValueError, match="BBB"):
        panel(tmp_path, refresh=True)
    fake.long = with_split(DATES[SPLIT_I + 5], close=np.nan)
    with pytest.raises(ValueError, match="1 split"):
        panel(tmp_path, refresh=True)
    fake.long = with_split(DATES[3])  # before the first panel date (120 < start-400)
    assert len(panel(tmp_path, refresh=True).dates) > 0


def test_raw_basis(fake, tmp_path):
    adj = panel(tmp_path, basis="adjusted")
    raw = panel(tmp_path, basis="raw")
    assert adj.basis == "adjusted" and raw.basis == "raw" and "split_age" in raw.fields
    j = list(raw.symbols).index("AAA")
    t = list(raw.dates).index(DATES[SPLIT_I - 1])
    assert raw.fields["close"][t, j] == pytest.approx(adj.fields["close"][t, j] * 2)
    assert raw.fields["shares_outstanding"][t, j] == pytest.approx(500.0)
    assert raw.fields["shares_outstanding"][t + 1, j] == pytest.approx(1000.0)
    cap = raw.fields["close"] * raw.fields["shares_outstanding"]
    assert cap[t, j] == pytest.approx(adj.fields["close"][t, j] * 1000.0)
    assert np.isnan(raw.fields["shares_outstanding"][:, list(raw.symbols).index("CCC")]).all()


def test_groups(fake, tmp_path):
    p = panel(tmp_path)
    assert set(p.groups) == {"sector", "industry"}
    assert list(p.groups["sector"]) == [0, 0, -1]
    assert list(p.groups["industry"]) == [0, 1, -1]
    assert panel(tmp_path, with_groups=False).groups == {}


def test_default_universe_source_is_liquidity(fake, tmp_path):
    from alpha101.data import universe as uni
    seen = {}

    def fake_load(conn, p, source):
        seen["source"] = source
        return np.ones(p.shape, dtype=bool)
    with pytest.MonkeyPatch.context() as mp:  # the default source is now "liquidity" 
        mp.setattr(uni, "load_universe", fake_load)
        mp.setattr(uni, "resolve_source", lambda conn, s: s + "@20261019")
        p = load_panel("us", start=START, cache_dir=tmp_path)
    assert seen["source"] == "liquidity_top500_us@20261019"
    assert not p.universe[:WARMUP_ROWS].any() and p.universe[WARMUP_ROWS:].all()


def test_bad_args(fake, tmp_path):
    with pytest.raises(ValueError):
        panel(tmp_path, basis="weird")
    with pytest.raises(ValueError):
        load_panel("uk", start=START, universe_source=None, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        cache._market("xx")


class Cur:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchone(self):
        return (0, None, 0) if "DISTINCT" in self.log[-1][0] else (0, None)

    def fetchall(self):
        return []


class Conn:
    def __init__(self):
        self.log = []

    def cursor(self):
        return Cur(self.log)


def test_sql_shape():
    import re
    c = Conn()
    cache.db_fingerprint(c, "us")
    cache._read_master(c, "ca")
    assert len(c.log) == 4
    for sql, params in c.log:
        assert "%s" in sql and params
        assert set(re.findall(r"'([^']*)'", sql)) <= {"EQUITY"}


def test_atomic_write(fake, tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(OSError):
            load_long("us", "2013-01-01", cache_dir=tmp_path)
    assert not (tmp_path / "yahoo_us.json").exists()
    assert not (tmp_path / "yahoo_us.parquet").exists()
    n = fake.reads
    load_long("us", "2013-01-01", cache_dir=tmp_path)
    assert fake.reads == n + 1


def test_partial_publish_is_a_miss(fake, tmp_path, monkeypatch):
    """F1: a crash after the parquet write but before the sidecar must not leave a hit for a different start."""
    full = load_long("us", "2012-01-01", cache_dir=tmp_path)
    orig = cache._atomic

    def crash(path, write):
        if path.suffix == ".json":
            raise OSError("crash before sidecar")
        return orig(path, write)
    with monkeypatch.context() as m:
        m.setattr(cache, "_atomic", crash)
        with pytest.raises(OSError, match="crash before sidecar"):
            load_long("us", "2013-06-03", refresh=True, cache_dir=tmp_path)
    assert not (tmp_path / "yahoo_us.json").exists()
    n = fake.reads
    res = load_long("us", "2012-01-01", cache_dir=tmp_path)
    assert fake.reads == n + 1  # MISS: re-read
    assert len(res) == len(full) and res["date"].min() == DATES[0]
    pd.testing.assert_frame_equal(res, full)
    assert not list(tmp_path.glob("*.tmp"))


def test_tmp_cleanup_and_unique_names(fake, tmp_path, monkeypatch):
    names = []
    orig = cache._atomic

    def spy(path, write):
        return orig(path, lambda tmp: (names.append(tmp.name), write(tmp)))
    with monkeypatch.context() as m:
        m.setattr(cache, "_atomic", spy)
        load_long("us", "2013-01-01", cache_dir=tmp_path)
        load_long("us", "2013-01-01", refresh=True, cache_dir=tmp_path)
    assert len(names) == 6 and len(set(names)) == 6 and all(n.endswith(".tmp") for n in names)

    def boom(self, path, *a, **k):
        pathlib.Path(path).write_bytes(b"partial")  # a half-written temp file
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(OSError):
            load_long("us", "2013-01-01", refresh=True, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_hit_boundary_equal_start(fake, tmp_path):
    """Mutant o: cached start == requested start is a hit, and the row dated exactly `start` is kept."""
    s = str(DATES[100].date())
    load_long("us", s, cache_dir=tmp_path)
    assert fake.reads == 1
    res = load_long("us", s, cache_dir=tmp_path)
    assert fake.reads == 1
    assert res["date"].min() == DATES[100]
    assert (res["date"] == DATES[100]).sum() == 4  # ^GSPC, AAA, BBB, CCC


def test_ca_panel_uses_tsx_calendar(fake, tmp_path):
    """Mutant p: market 'ca' takes its dates from ^GSPTSE, not ^GSPC."""
    keep = [i for i in range(len(DATES)) if i % 5 != 3]
    ca_dates = DATES[keep]
    fake.long = pd.concat([fake.long, pd.DataFrame(_rows("^GSPTSE", ca_dates, 200.0), columns=COLUMNS)])
    p = load_panel("ca", start=START, universe_source=None, cache_dir=tmp_path)
    i0 = int(ca_dates.searchsorted(pd.Timestamp(START)))
    assert p.dates.equals(ca_dates[i0 - WARMUP_ROWS:])
    assert len(p.dates) != len(DATES[int(DATES.searchsorted(pd.Timestamp(START))) - WARMUP_ROWS:])


def test_first_panel_row_has_data(fake, tmp_path):
    """Mutant q: the first panel date holds that date's equity data."""
    p = panel(tmp_path)
    i0 = int(DATES.searchsorted(pd.Timestamp(START)))
    assert p.dates[0] == DATES[i0 - WARMUP_ROWS]
    assert np.isfinite(p.fields["close"][0]).all()
    assert np.isfinite(p.fields["volume"][0]).all()


def test_short_history_starts_at_first_row(fake, tmp_path, caplog):
    """Mutant r: fewer than WARMUP_ROWS calendar rows before start -> start at row 0, warn."""
    s = str(DATES[100].date())
    with caplog.at_level("WARNING", logger=cache.log.name):
        p = load_panel("us", start=s, universe_source=None, cache_dir=tmp_path)
    assert p.dates[0] == DATES[0] and len(p.dates) == len(DATES)
    assert any("only 100 calendar rows" in r.getMessage() for r in caplog.records)


def test_split_guard_zero_close(fake, tmp_path):
    """Mutant j: a split row with close == 0 raises."""
    d = fake.long.copy()
    m = (d["symbol"] == "BBB") & (d["date"] == DATES[SPLIT_I + 5])
    d.loc[m, "split"] = 2.0
    d.loc[m, "close"] = 0.0
    fake.long = d
    with pytest.raises(ValueError, match="1 split"):
        panel(tmp_path, refresh=True)


class FpCur:
    def __init__(self, log, answers):
        self.log, self.answers = log, answers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchone(self):
        return self.answers[len(self.log) - 1]


class FpConn:
    def __init__(self, answers):
        self.log, self.answers = [], answers

    def cursor(self):
        return FpCur(self.log, self.answers)


def test_db_fingerprint_content():
    """Mutants s/t: the real db_fingerprint issues 3 SELECTs and builds the dict from the tuples."""
    import datetime as dt
    answers = [(5, dt.datetime(2024, 1, 2, 3, 4, 5), 3), (7, None), (9, dt.datetime(2024, 2, 3))]
    for mk, cal in (("us", "^GSPC"), ("ca", "^GSPTSE")):
        c = FpConn(answers)
        fp = cache.db_fingerprint(c, mk)
        assert [p for _, p in c.log] == [[mk.upper()], [cal], [mk.upper()]]
        assert all(sql.lstrip().upper().startswith("SELECT") for sql, _ in c.log)
        assert fp == {"rows": 5, "max_fetched_at": "2024-01-02T03:04:05", "symbols": 3,
                      "calendar_rows": 7, "calendar_max_fetched_at": None,
                      "master_rows": 9, "master_max_fetched_at": "2024-02-03T00:00:00"}


def test_fingerprint_taken_before_read(fake, tmp_path, monkeypatch):
    """Mutant n: on a miss db_fingerprint is called before read_long (a hit calls only the fingerprint)."""
    calls = []
    fp, rl = cache.db_fingerprint, cache.read_long
    monkeypatch.setattr(cache, "db_fingerprint", lambda *a, **k: (calls.append("fingerprint"), fp(*a, **k))[1])
    monkeypatch.setattr(cache, "read_long", lambda *a, **k: (calls.append("read_long"), rl(*a, **k))[1])
    load_long("us", "2013-01-01", cache_dir=tmp_path)
    assert calls == ["fingerprint", "read_long"]
    calls.clear()
    load_long("us", "2013-01-01", cache_dir=tmp_path)
    assert calls == ["fingerprint"]
