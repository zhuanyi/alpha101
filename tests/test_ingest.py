import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha101.data import constituents, ingest
from alpha101.data.constituents import (candidate_pool, parse_sp500_changes, parse_sp500_current,
                                        check_identity, names_match, parse_tsx, valid_symbol, yahoo_symbol)
from alpha101.data.ingest import (MASTER_UPSERT_SQL, OHLCV_UPSERT_SQL, SECURITY_MASTER_DDL,
                                  needs_full_refetch, upsert_ohlcv, upsert_security_master)


def test_yahoo_symbol():
    assert yahoo_symbol("BRK.B", "us") == "BRK-B"
    assert yahoo_symbol(" AAPL ", "US") == "AAPL"
    assert yahoo_symbol("BBD.B", "ca") == "BBD-B.TO"
    assert yahoo_symbol("RY", "ca") == "RY.TO"
    assert yahoo_symbol("RCI.B[1]", "ca") == "RCI-B.TO"
    assert yahoo_symbol("JCP |", "us") == "JCP"
    assert yahoo_symbol("NA", "ca") == "NA.TO"
    with pytest.raises(ValueError):
        yahoo_symbol("X", "uk")


def test_yahoo_symbol_strict():
    assert yahoo_symbol("", "ca") == "" and yahoo_symbol("  ", "us") == ""   # F10: no bare ".TO"
    assert yahoo_symbol("IIP.UN", "ca") == "IIP-UN.TO"
    for bad in ("A B", "AB$", "A,B", "a b c"):
        with pytest.raises(ValueError):
            yahoo_symbol(bad, "us")
    assert valid_symbol("BRK-B") and valid_symbol("RCI-B.TO")
    assert not valid_symbol("ALLE |") and not valid_symbol("") and not valid_symbol("A B") and not valid_symbol(None)


def test_fetch_history_refuses_bad_symbol():
    for bad in ("ALLE |", "A B", "", "A|B"):
        with pytest.raises(ValueError):
            ingest.fetch_history(bad)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self._sql = sql

    def fetchone(self):
        assert "MAX(date)" in self._sql
        return (self.conn.stored_max,)

    def fetchall(self):
        assert "ORDER BY date DESC LIMIT 10" in self._sql and "%s" in self._sql
        return list(self.conn.rows)

    def executemany(self, sql, rows):
        self.conn.batches.append((sql, list(rows)))


class FakeConn:
    """`rows`: stored rows (date, close, adj_close, dividend, split), newest first; by default one row on
    stored_max. stored_adj="same" is the adj_close on the basis `_frame` gives (0.9 x close)."""

    def __init__(self, stored_max=None, stored_close=None, stored_adj="same", stored_div=None, stored_split=None,
                 rows=None):
        self.stored_max = stored_max
        if rows is None:
            adj = (stored_close * 0.9 if stored_close is not None else None) if stored_adj == "same" else stored_adj
            rows = [] if stored_close is None else [(stored_max, stored_close, adj, stored_div, stored_split)]
        self.rows = rows
        self.executed, self.batches, self.commits = [], [], 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def _frame(closes, splits=None, start="2024-01-02", divs=None, adj_scale=1.0):
    """OHLC, adj_close and volume all differ so a swapped column in the row tuple is caught."""
    c = pd.Series(closes, dtype=float)
    n = len(c)
    d = pd.DataFrame({"date": pd.bdate_range(start, periods=n), "open": c * 0.98, "high": c * 1.03,
                      "low": c * 0.95, "close": c, "adj_close": c * adj_scale * 0.9,
                      "volume": 100.0 + np.arange(n), "dividend": 0.0, "split": 0.0})
    for i, v in (splits or {}).items():
        d.loc[i, "split"] = v
    for i, v in (divs or {}).items():
        d.loc[i, "dividend"] = v
    return d


def test_refetch_new_symbol_is_false():
    assert needs_full_refetch(FakeConn(stored_max=None), "X", _frame([10.0, 11.0])) is False


def test_refetch_split_after_stored_max():
    new = _frame([10.0, 11.0, 12.0], splits={2: 2.0})
    conn = FakeConn(stored_max=dt.date(2024, 1, 3), stored_close=11.0)
    assert needs_full_refetch(conn, "X", new) is True


def test_refetch_split_not_newer_than_stored_is_ignored():
    new = _frame([10.0, 11.0, 12.0], splits={0: 2.0})
    conn = FakeConn(stored_max=dt.date(2024, 1, 3), stored_close=11.0)  # stored max = day 2; split on day 1
    assert needs_full_refetch(conn, "X", new) is False


def test_refetch_split_on_stored_max_depends_on_stored_row():
    new = _frame([10.0, 11.0, 12.0], splits={1: 2.0})                   # split dated exactly on the stored max
    d = dt.date(2024, 1, 3)
    assert needs_full_refetch(FakeConn(d, 11.0, stored_split=0.0), "X", new) is True
    assert needs_full_refetch(FakeConn(d, 11.0, stored_split=None), "X", new) is True
    assert needs_full_refetch(FakeConn(d, 11.0, stored_split=2.0), "X", new) is False   # already carried


def test_refetch_close_readjusted():
    new = _frame([10.0, 11.0, 12.0])
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 3), 11.0 * 1.006), "X", new) is True
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 3), 11.0 * 1.004), "X", new) is False


def test_refetch_uses_latest_overlap_when_stored_date_missing_in_new():
    new = _frame([10.0, 11.0, 12.0])  # 01-02, 01-03, 01-04
    rows = [(dt.date(2024, 1, 7), 99.0, 89.1, 0.0, 0.0), (dt.date(2024, 1, 4), 12.0, 10.8, 0.0, 0.0)]
    # stored max is a weekend row Yahoo does not have: the overlap is 01-04
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 7), rows=rows), "X", new) is False
    rows[1] = (dt.date(2024, 1, 4), 12.0 * 1.01, 10.8, 0.0, 0.0)
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 7), rows=rows), "X", new) is True


def _placeholders(sql: str) -> list[str]:
    m = re.search(r"VALUES\s*\(([^)]*)\)", sql)
    return [t.strip() for t in m.group(1).split(",")]


@pytest.mark.parametrize("sql", [OHLCV_UPSERT_SQL, MASTER_UPSERT_SQL])
def test_upsert_sql_is_pure_placeholders(sql):
    assert sql.count("%s") == 12
    assert _placeholders(sql) == ["%s"] * 12     # the VALUES list has no literal and no NOW()
    assert "NOW(" not in sql.upper() and "DELETE" not in sql.upper() and "DROP" not in sql.upper()
    assert "ON DUPLICATE KEY UPDATE" in sql


def test_ddl_matches_design_md():
    design = (Path(__file__).resolve().parents[1] / "docs" / "design.md").read_text()
    m = re.search(r"(CREATE TABLE IF NOT EXISTS security_master \(.*?\) ENGINE=InnoDB[^;]*;)", design, re.S)
    assert m and m.group(1) == SECURITY_MASTER_DDL


def test_upsert_ohlcv_rows_and_chunks(monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK", 3)
    df = _frame([10.0, np.nan, 12.0, 13.0, 14.0])
    df.loc[1, "volume"] = np.nan
    conn = FakeConn()
    at = dt.datetime(2026, 10, 18, 3, 0, 0)
    assert upsert_ohlcv(conn, "BRK-B", df, at) == 5
    assert [len(b[1]) for b in conn.batches] == [3, 2] and conn.commits == 2
    row = conn.batches[0][1][1]
    assert row[0] == "BRK-B" and row[1] == dt.date(2024, 1, 3) and row[5] is None and row[7] is None
    assert len(row) == 12 and row[-2:] == ("yfinance", at)
    assert isinstance(conn.batches[0][1][0][7], int)


def test_upsert_security_master_row():
    conn = FakeConn()
    rows = [{"symbol": "RY.TO", "market": "CA", "name": "Royal Bank", "sector": "Financial Services",
             "market_cap_snapshot": float("nan"), "currency": "CAD"}]
    assert upsert_security_master(conn, rows, dt.datetime(2026, 10, 18)) == 1
    r = conn.batches[0][1][0]
    assert len(r) == 12 and r[0] == "RY.TO" and r[2] == "CA" and r[7] is None and r[9] == "CAD"


def test_parse_sp500_changes_multiindex_and_header_repeat():
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"),
                                      ("Added", "Security"), ("Removed", "Ticker"),
                                      ("Removed", "Security"), ("Reason", "Reason")])
    t = pd.DataFrame([["Effective Date", "Ticker", "Security", "Ticker", "Security", "Reason"],
                      ["October 6, 2026", "TWLO", "Twilio", "WBD", "WBD Inc", "x"],
                      ["October 1, 2026", "VYLR", "Vylor", np.nan, np.nan, "y"]], columns=cols)
    out = parse_sp500_changes([pd.DataFrame({"a": [1]}), t])
    assert list(out.columns) == ["date", "added", "added_name", "removed", "removed_name", "reason"]
    assert len(out) == 2
    assert out.loc[0, "added"] == "TWLO" and out.loc[0, "removed"] == "WBD"
    assert out.loc[0, "removed_name"] == "WBD Inc" and out.loc[0, "reason"] == "x"
    assert pd.isna(out.loc[1, "removed"]) and out.loc[1, "date"] == pd.Timestamp("2026-10-01")
    assert parse_sp500_changes([pd.DataFrame({"a": [1]})]) is None


def _toy_tables():
    cur = pd.DataFrame({"Symbol": ["AAPL", "BRK.B"], "Security": ["Apple", "Berkshire"],
                        "GICS Sector": ["IT", "Fin"], "GICS Sub-Industry": ["HW", "Ins"]})
    ch = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2003-01-01"]),
                       "added": ["TSLA", "OLD"], "removed": [None, "GONE"]})
    tsx = pd.DataFrame({"Ticker": ["RY", "BBD.B"], "Company": ["Royal", "Bombardier"],
                        "Sector [10]": ["Fin", "Ind"], "Industry [10]": ["a", "b"]})
    return cur, ch, tsx


def test_parse_current_tables():
    cur, _, tsx = _toy_tables()
    assert list(parse_sp500_current([cur])["symbol"]) == ["AAPL", "BRK.B"]
    p = parse_tsx([tsx])
    assert list(p.columns) == ["symbol", "name", "sector"] and list(p["symbol"]) == ["RY", "BBD.B"]


def test_candidate_pool_union_and_since(monkeypatch, tmp_path):
    cur, ch, tsx = _toy_tables()
    monkeypatch.setattr(constituents, "fetch_sp500",
                        lambda cache_dir, raw_dir=None: (parse_sp500_current([cur]), ch))
    monkeypatch.setattr(constituents, "fetch_tsx_composite",
                        lambda cache_dir, raw_dir=None: parse_tsx([tsx]))
    pool = candidate_pool(tmp_path, since="2005-01-01")
    got = dict(zip(pool["symbol"], zip(pool["market"], pool["source"])))
    assert got == {"AAPL": ("US", "sp500_current"), "BRK-B": ("US", "sp500_current"),
                   "TSLA": ("US", "sp500_changes"), "RY.TO": ("CA", "tsx_current"),
                   "BBD-B.TO": ("CA", "tsx_current")}     # OLD / GONE predate `since`
    assert "GONE" in set(candidate_pool(tmp_path, since="2000-01-01")["symbol"])


def test_read_html_keeps_na_ticker():
    html = ("<table><tr><th>Ticker</th><th>Company</th><th>Sector [1]</th></tr>"
            "<tr><td>NA</td><td>National Bank</td><td>Financials</td></tr></table>")
    assert list(parse_tsx(constituents._tables(html))["symbol"]) == ["NA"]


# ---- F3: dividend / adj_close re-adjustment ------------------------------------------------------

def test_refetch_dividend_after_stored_max():
    new = _frame([10.0, 11.0, 12.0], divs={2: 0.1})
    conn = FakeConn(stored_max=dt.date(2024, 1, 3), stored_close=11.0)
    assert needs_full_refetch(conn, "X", new) is True


def test_refetch_dividend_not_newer_than_stored_is_ignored():
    new = _frame([10.0, 11.0, 12.0], divs={0: 0.1})
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 3), 11.0), "X", new) is False


def test_refetch_dividend_on_stored_max_depends_on_stored_row():
    new = _frame([10.0, 11.0, 12.0], divs={1: 0.1})                     # dividend dated exactly on the stored max
    d = dt.date(2024, 1, 3)
    assert needs_full_refetch(FakeConn(d, 11.0, stored_div=0.0), "X", new) is True
    assert needs_full_refetch(FakeConn(d, 11.0, stored_div=None), "X", new) is True
    assert needs_full_refetch(FakeConn(d, 11.0, stored_div=0.1), "X", new) is False     # already carried


def test_refetch_special_dividend_5pct():
    # history adj_close rescaled by 0.95, close unchanged, dividend on a later day
    new = _frame([100.0, 101.0, 102.0, 99.0], divs={3: 5.2}, adj_scale=0.95)
    conn = FakeConn(stored_max=dt.date(2024, 1, 4), stored_close=102.0)
    assert needs_full_refetch(conn, "X", new) is True


def test_refetch_junction_day_ratio_drift_without_dividend_row():
    # extra_checks 2c: Yahoo rescaled adj_close by 0.99 but the dividend row is not in the frame
    new = _frame([100.0, 101.0, 102.0, 103.0], adj_scale=0.99)
    conn = FakeConn(stored_max=dt.date(2024, 1, 4), stored_close=102.0)   # stored adj = 0.9 x close
    assert needs_full_refetch(conn, "X", new) is True
    # a 5e-5 drift is float noise, not a re-adjustment
    assert needs_full_refetch(conn, "X", _frame([100.0, 101.0, 102.0, 103.0], adj_scale=1.00005)) is False


def test_refetch_junction_day_with_dividend_after_stored_max():
    f = 1 - 1.04 / 104.0
    new = _frame([100.0, 101.0, 102.0, 103.0, 104.0, 103.5], divs={5: 1.04}, adj_scale=f)
    conn = FakeConn(stored_max=dt.date(2024, 1, 8), stored_close=104.0)
    assert needs_full_refetch(conn, "X", new) is True


def test_refetch_nan_overlap_close_uses_previous_overlap_day():
    new = _frame([10.0, 11.0, np.nan])
    def rows(c):
        return [(dt.date(2024, 1, 4), 12.0, 10.8, 0.0, 0.0), (dt.date(2024, 1, 3), c, c * 0.9, 0.0, 0.0)]
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 4), rows=rows(11.0)), "X", new) is False
    assert needs_full_refetch(FakeConn(dt.date(2024, 1, 4), rows=rows(11.0 * 1.01)), "X", new) is True


def test_refetch_overlap_falls_back_when_latest_stored_row_has_no_close():
    new = _frame([10.0, 11.0, 12.0])
    d4, d3 = dt.date(2024, 1, 4), dt.date(2024, 1, 3)
    for c3, expect in ((11.0 * 1.01, True), (11.0, False)):
        rows = [(d4, None, None, 0.0, 0.0), (d3, c3, c3 * 0.9, 0.0, 0.0)]    # latest stored row has NULL close
        assert needs_full_refetch(FakeConn(d4, rows=rows), "X", new) is expect


def test_refetch_ratio_falls_back_when_latest_stored_adj_close_is_null():
    new = _frame([10.0, 11.0, 12.0])
    d4, d3 = dt.date(2024, 1, 4), dt.date(2024, 1, 3)
    for adj3, expect in ((11.0 * 0.9 * 1.01, True), (11.0 * 0.9, False)):
        rows = [(d4, 12.0, None, 0.0, 0.0), (d3, 11.0, adj3, 0.0, 0.0)]      # close agrees; adj_close NULL on 01-04
        assert needs_full_refetch(FakeConn(d4, rows=rows), "X", new) is expect


def test_refetch_ratio_drift_tolerance_is_tight():
    # a 5e-3 ratio drift (close unchanged) is a re-adjustment; the tolerance is 1e-4, not 1e-2
    conn = FakeConn(stored_max=dt.date(2024, 1, 4), stored_close=102.0)
    assert needs_full_refetch(conn, "X", _frame([100.0, 101.0, 102.0, 103.0], adj_scale=1.005)) is True


# ---- F1: identity of change-only tickers ----------------------------------------------------------

def test_names_match():
    assert names_match("TechnipFMC plc", "FMC Technologies")
    assert names_match("V.F. Corporation", "VF Corporation")
    assert names_match("ATI Inc.", "Allegheny Technologies")
    assert names_match("O-I Glass, Inc.", "Owens-Illinois")
    assert names_match("CNX Resources Corporation", "CNX Resources")
    assert not names_match("Sea Limited", "Spectra Energy")
    assert not names_match("SentinelOne, Inc.", "Sprint Nextel")
    assert not names_match(None, "Sprint Nextel") and not names_match("Weibo Corporation", "")


def test_check_identity():
    d = pd.Timestamp("2012-01-03")
    assert check_identity(pd.Timestamp("1990-01-02"), "Sprint Corp", d, ("Sprint Nextel",)) is None
    assert "after change date" in check_identity(pd.Timestamp("2017-10-20"), "Sea Limited", d, ("Spectra Energy",))
    assert "name" in check_identity(pd.Timestamp("2006-12-04"), "Ocean Thermal Energy", d, ("Compuware",))
    # unknown Yahoo name: only the date rule can speak
    assert check_identity(pd.Timestamp("1990-01-02"), None, d, ("Compuware",)) is None
    # first trade a few days after the change date (new listing / spin-off) is tolerated
    assert check_identity(pd.Timestamp("2012-01-06"), "X Corp", d, ("X Corp",)) is None


def test_check_identity_name_override():
    d = pd.Timestamp("2020-12-21")
    old = pd.Timestamp("1994-07-22")
    # mismatched name: fails without the symbol, and for a non-whitelisted symbol
    assert "name" in check_identity(old, "Apartment Investment and Management Company", d, ("Zzz Holdings",))
    assert "name" in check_identity(old, "Apartment Investment and Management Company", d, ("Zzz Holdings",), symbol="XYZ")
    # whitelisted symbols skip the name test
    for sym in ("AIV", "CNX"):
        assert check_identity(old, "Some Other Name", d, ("Zzz Holdings",), symbol=sym) is None
    # the first-trade-date test still applies
    assert "after change date" in check_identity(pd.Timestamp("2021-06-01"), "Some Other Name", d, ("Zzz Holdings",), symbol="AIV")


def test_candidate_pool_carries_identity_inputs(monkeypatch, tmp_path):
    cur, _, tsx = _toy_tables()
    ch = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2010-05-05", "2012-01-01"]),
                       "added": ["TSLA", None, "ZZZ"], "added_name": ["Tesla", None, "Zed"],
                       "removed": [None, "ZZZ", "A B"], "removed_name": [None, "Zed Corp", "bad"]})
    monkeypatch.setattr(constituents, "fetch_sp500", lambda cache_dir, raw_dir=None: (parse_sp500_current([cur]), ch))
    monkeypatch.setattr(constituents, "fetch_tsx_composite", lambda cache_dir, raw_dir=None: parse_tsx([tsx]))
    pool = candidate_pool(tmp_path).set_index("symbol")
    assert "A B" not in pool.index and "AB" not in pool.index            # F2: bad cell dropped
    assert pool.loc["ZZZ", "change_date"] == pd.Timestamp("2010-05-05")  # earliest row
    assert set(pool.loc["ZZZ", "wiki_names"]) == {"Zed", "Zed Corp"}
    assert pool.loc["AAPL", "wiki_names"] == () and pd.isna(pool.loc["AAPL", "change_date"])


# ---- F6, F10, F11, F13: Wikipedia parsing, download, raw files ------------------------------------

def _changes_table(rows):
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"),
                                      ("Added", "Security"), ("Removed", "Ticker"), ("Removed", "Security")])
    return pd.DataFrame(rows, columns=cols)


def test_parse_changes_cleans_footnote_and_nbsp_dates():
    t = _changes_table([["October 6, 2026[4]", "AAA", "a", "BBB", "b"],
                        ["October\xa01, 2026", "CCC", "c", "DDD", "d"]])
    out = parse_sp500_changes([t])
    assert list(out["added"]) == ["AAA", "CCC"] and out.loc[1, "date"] == pd.Timestamp("2026-10-01")


def test_parse_changes_fails_loud_when_dates_unparsable():
    with pytest.raises(ValueError):
        parse_sp500_changes([_changes_table([["2026-10-06", "AAA", "a", "BBB", "b"]])])
    many = [["October 6, 2026", "T%d" % i, "n", "U%d" % i, "n"] for i in range(19)] + [["06/10/2026", "X", "x", "Y", "y"]]
    assert len(parse_sp500_changes([_changes_table(many)])) == 19        # 5% of 20 rows lost: tolerated, warned
    many.append(["07/10/2026", "X2", "x", "Y2", "y"])
    with pytest.raises(ValueError):
        parse_sp500_changes([_changes_table(many)])                      # 2 of 21 > 5%


def test_fetch_sp500_falls_back_to_changes_url_when_list_table_unusable(monkeypatch, tmp_path):
    cur, _, _ = _toy_tables()
    bad = _changes_table([["2026-10-06", "AAA", "a", "BBB", "b"]])
    good = _changes_table([["October 6, 2026", "AAA", "a", "BBB", "b"]])
    urls = []
    monkeypatch.setattr(constituents, "_download", lambda url, cd, name: urls.append(url) or url)
    monkeypatch.setattr(constituents, "_tables", lambda html: [cur, bad] if html == constituents.SP500_URL else [good])
    _, ch = constituents.fetch_sp500(tmp_path, tmp_path / "raw")
    assert urls == [constituents.SP500_URL, constituents.SP500_CHANGES_URL] and len(ch) == 1


def test_save_raw_keeps_all_columns_and_stamps_utc_time(tmp_path):
    ch = parse_sp500_changes([_changes_table([["October 6, 2026", "AAA", "Aaa Inc", "BBB", "Bbb Inc"]])])
    constituents._save_raw(ch, tmp_path, "sp500_changes")
    f = next(tmp_path.glob("sp500_changes_*.csv"))
    assert re.fullmatch(r"sp500_changes_\d{8}T\d{6}Z\.csv", f.name)
    saved = pd.read_csv(f)
    assert {"added_name", "removed_name", "reason"} <= set(saved.columns) and saved.loc[0, "added_name"] == "Aaa Inc"


def test_download_retries_5xx_then_succeeds(monkeypatch, tmp_path):
    import requests
    seq = [500, 429, 200]
    sleeps = []

    class R:
        def __init__(self, code):
            self.status_code, self.text = code, "<html>ok</html>"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    monkeypatch.setattr(requests, "get", lambda *a, **k: R(seq.pop(0)))
    monkeypatch.setattr(constituents.time, "sleep", sleeps.append)
    assert constituents._download("http://x", tmp_path, "p") == "<html>ok</html>"
    assert len(sleeps) == 2 and seq == []
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(503))
    with pytest.raises(requests.HTTPError):
        constituents._download("http://x", tmp_path, "p")


# ---- fetch_history: column handling, retries, rate limit -------------------------------------------

def _yahoo_raw(n=3, multi=True, drop=()):
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n, tz="America/New_York"))
    data = {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.4, "Volume": 10, "Dividends": 0.0,
            "Stock Splits": 0.0}
    df = pd.DataFrame({k: [v + i for i in range(n)] for k, v in data.items() if k not in drop}, index=idx)
    if multi:
        df.columns = pd.MultiIndex.from_product([df.columns, ["X"]], names=["Price", "Ticker"])
    return df


@pytest.fixture
def yf_stub(monkeypatch):
    import yfinance
    from yfinance import shared
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    state = {"calls": 0, "frames": [], "errors": []}

    def download(symbol, **kw):
        i = state["calls"]
        state["calls"] += 1
        if i < len(state["errors"]) and state["errors"][i]:
            shared._ERRORS[symbol.upper()] = state["errors"][i]
        return state["frames"][min(i, len(state["frames"]) - 1)]

    monkeypatch.setattr(yfinance, "download", download)
    return state


def test_fetch_history_flattens_multiindex_strips_tz_and_maps_columns(yf_stub):
    yf_stub["frames"] = [_yahoo_raw()]
    out = ingest.fetch_history("X")
    assert list(out.columns) == ["date"] + ingest.OHLCV_COLUMNS
    assert out["date"].dt.tz is None and out.loc[0, "date"] == pd.Timestamp("2024-01-02")
    r = out.iloc[1]
    assert (r.open, r.high, r.low, r.close, r.adj_close, r.volume) == (2.0, 3.0, 1.5, 2.5, 2.4, 11)


def test_fetch_history_retries_empty_then_returns_none(yf_stub):
    yf_stub["frames"] = [pd.DataFrame()]
    assert ingest.fetch_history("X", retries=3) is None and yf_stub["calls"] == 3


def test_fetch_history_warns_on_missing_columns(yf_stub, caplog):
    yf_stub["frames"] = [_yahoo_raw(multi=False, drop=("Adj Close", "Dividends"))]
    with caplog.at_level("WARNING", logger="alpha101.ingest"):
        out = ingest.fetch_history("X")
    assert out["adj_close"].isna().all() and (out["dividend"] == 0).all()
    assert any("lacks columns" in m and "adj_close" in m for m in caplog.messages)


def test_fetch_history_rate_limit_from_recorded_errors(yf_stub):
    yf_stub["frames"] = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    yf_stub["errors"] = ["YFRateLimitError('Too Many Requests')"] * 3
    with pytest.raises(ingest.RateLimited):
        ingest.fetch_history("X")
    # a delisting message is not a rate limit
    yf_stub.update(calls=0, errors=["YFPricesMissingError('possibly delisted')"] * 3)
    assert ingest.fetch_history("X") is None


def test_fetch_history_rate_limit_backs_off_longer(monkeypatch, yf_stub):
    sleeps = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    yf_stub["frames"] = [pd.DataFrame(), pd.DataFrame(), _yahoo_raw()]
    yf_stub["errors"] = ["YFRateLimitError()", "YFRateLimitError()"]
    assert ingest.fetch_history("X", pause=1.0) is not None
    assert sleeps == [30.0, 60.0]                                       # no sleep after the last attempt


# ---- ingest() harness --------------------------------------------------------------------------------

class FakeDB:
    """In-memory yahoo_ohlcv with ON DUPLICATE KEY semantics; answers the SQL ingest() issues."""

    def __init__(self):
        self.rows = {}                  # (symbol, date) -> dict
        self.master = {}
        self.deleted = []               # (symbol, [dates])
        self.executed = []

    def load(self, sym, df, fetched_at=dt.datetime(2026, 10, 17)):
        for r in df.itertuples(index=False):
            self.rows[(sym, r.date.date())] = dict(open=r.open, high=r.high, low=r.low, close=r.close,
                                                   adj_close=r.adj_close, volume=int(r.volume), fetched_at=fetched_at,
                                                   dividend=r.dividend, split=r.split)

    def dates(self, sym):
        return sorted(d for (s, d) in self.rows if s == sym)


class DBCursor:
    def __init__(self, db):
        self.db, self.res = db, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, p=None):
        self.db.executed.append(sql)
        if sql.startswith("SELECT MAX(date)"):
            ds = self.db.dates(p[0])
            self.res = [(ds[-1] if ds else None,)]
        elif sql.startswith("SELECT date, close, adj_close, dividend, split"):
            assert "ORDER BY date DESC LIMIT 10" in sql
            ds = sorted(self.db.dates(p[0]), reverse=True)[:10]
            self.res = [(d, *(self.db.rows[(p[0], d)][k] for k in ("close", "adj_close", "dividend", "split")))
                        for d in ds]
        elif sql.startswith("SELECT date"):
            self.res = [(d,) for d in self.db.dates(p[0])]
        elif sql.startswith("DELETE FROM yahoo_ohlcv WHERE symbol=%s AND date IN"):
            assert sql.count("%s") == 1 + len(p) - 1 and p[0] not in ("", None)
            for d in p[1:]:
                del self.db.rows[(p[0], d)]
            self.db.deleted.append((p[0], list(p[1:])))
        else:
            self.res = []

    def fetchone(self):
        return self.res[0] if self.res else None

    def fetchall(self):
        return self.res

    def executemany(self, sql, rows):
        if sql == ingest.OHLCV_UPSERT_SQL:
            for r in rows:
                self.db.rows[(r[0], r[1])] = dict(open=r[2], high=r[3], low=r[4], close=r[5], adj_close=r[6],
                                                  volume=r[7], fetched_at=r[11], dividend=r[8], split=r[9])
        else:
            assert sql == ingest.MASTER_UPSERT_SQL
            for r in rows:
                self.db.master[r[0]] = dict(name=r[1], fetched_at=r[11])


class DBConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        return DBCursor(self.db)

    def commit(self):
        pass

    def ping(self, reconnect=True):
        pass

    def close(self):
        pass


def _pool(*rows):
    """rows: (symbol, source[, change_date, wiki_names])"""
    out = []
    for r in rows:
        r = list(r) + [pd.NaT, ()][len(r) - 2:]
        out.append((r[0], "US", r[1], r[2], r[3]))
    return pd.DataFrame(out, columns=["symbol", "market", "source", "change_date", "wiki_names"])


@pytest.fixture
def run(monkeypatch, tmp_path):
    """run(db, pool, histories, infos=None) -> ingest('us') result with everything external faked."""
    import alpha101.db as D
    monkeypatch.chdir(tmp_path)                                         # ingest writes logs/ and data/raw/
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)

    def go(db, pool, histories, infos=None, **kw):
        monkeypatch.setattr(ingest, "candidate_pool", lambda *a, **k: pool)
        monkeypatch.setattr(D, "writer", lambda: DBConn(db))

        def fh(sym, **k):
            h = histories[sym]
            if isinstance(h, Exception):
                raise h
            return None if h is None else h.copy()

        def fm(sym, **k):
            i = (infos or {}).get(sym, {"name": sym})
            if isinstance(i, Exception):
                raise i
            return dict(i, symbol=sym)

        monkeypatch.setattr(ingest, "fetch_history", fh)
        monkeypatch.setattr(ingest, "fetch_security_master", fm)
        return ingest.ingest("us", **kw)

    return go


D5 = pd.bdate_range("2026-10-12", periods=5)          # stored: 10-12 .. 10-16
D7 = pd.bdate_range("2026-10-12", periods=7)          # Yahoo now: + 10-19, 10-20


def _assert_partition(res):
    """G3: every attempted symbol is in exactly one of the six outcomes."""
    assert res["attempted"] == (res["with_data"] + len(res["no_data"]) + len(res["rate_limited"])
                                + len(res["reused_ticker"]) + len(res["identity_unverified"])
                                + len(res["write_failed"]))


def test_ingest_incremental_slice_no_skipped_day_no_duplicates(run):
    db = FakeDB()
    db.load("XYZ", _frame([100, 101, 102, 103, 104.0], start="2026-10-12"))
    new = _frame([100, 101, 102, 103, 104.3, 105, 106.0], start="2026-10-12")      # last stored close revised +0.29%
    res = run(db, _pool(("XYZ", "sp500_current")), {"XYZ": new})
    assert db.dates("XYZ") == [d.date() for d in D7]                    # nothing skipped, one row per date
    assert db.rows[("XYZ", dt.date(2026, 10, 16))]["close"] == pytest.approx(104.3)    # `>=`: stored max refreshed
    assert db.rows[("XYZ", dt.date(2026, 10, 15))]["fetched_at"] == dt.datetime(2026, 10, 17)   # older rows untouched
    assert res["rows"] == 3 and res["refetched"] == [] and res["with_data"] == 1
    assert db.deleted == [] and not any(s.startswith("DELETE") for s in db.executed)


def test_ingest_new_symbol_writes_full_history_and_master(run):
    db = FakeDB()
    res = run(db, _pool(("NEW", "sp500_current")), {"NEW": _frame([10, 11, 12.0])}, infos={"NEW": {"name": "New Co"}})
    assert len(db.dates("NEW")) == 3 and db.master["NEW"]["name"] == "New Co"
    assert (res["attempted"], res["with_data"], res["rows"], res["master_rows"]) == (1, 1, 3, 1)


def test_ingest_row_tuple_has_distinct_columns_in_order(run):
    db = FakeDB()
    run(db, _pool(("NEW", "sp500_current")), {"NEW": _frame([10, 11.0])})
    r = db.rows[("NEW", dt.date(2024, 1, 3))]
    assert (r["open"], r["high"], r["low"], r["close"], r["adj_close"], r["volume"]) == (
        pytest.approx(10.78), pytest.approx(11.33), pytest.approx(10.45), 11.0, pytest.approx(9.9), 101)


def test_ingest_refetch_on_split_rewrites_history_and_deletes_stale_dates(run, caplog):
    db = FakeDB()
    db.load("XYZ", _frame([100, 101, 102, 103, 104.0], start="2026-10-12"))
    db.load("XYZ", _frame([99.0], start="2026-10-09"))                  # a date Yahoo no longer returns
    db.load("OTHER", _frame([5.0], start="2026-10-09"))                 # other symbols are never touched
    half = _frame([50, 50.5, 51, 51.5, 52, 52.5, 53.0], splits={5: 2.0}, start="2026-10-12")
    with caplog.at_level("INFO", logger="alpha101.ingest"):
        res = run(db, _pool(("XYZ", "sp500_current")), {"XYZ": half})
    assert res["refetched"] == ["XYZ"] and res["rows"] == 7
    assert db.rows[("XYZ", dt.date(2026, 10, 12))]["close"] == 50       # whole history on the new basis
    assert db.dates("XYZ") == [d.date() for d in D7]                    # the 10-09 row is gone
    assert db.deleted == [("XYZ", [dt.date(2026, 10, 9)])] and res["stale_deleted"] == {"XYZ": 1}
    assert ("OTHER", dt.date(2026, 10, 9)) in db.rows
    assert any("deleted 1 stored dates" in m for m in caplog.messages)


def test_ingest_refetch_on_dividend_only(run):
    db = FakeDB()
    db.load("XYZ", _frame([100, 101, 102, 103, 104.0], start="2026-10-12"))
    new = _frame([100, 101, 102, 103, 104, 103.5, 104.5], start="2026-10-12", divs={5: 1.04})
    new.loc[:4, "adj_close"] *= 0.99                                    # Yahoo rescaled the earlier adj_close
    res = run(db, _pool(("XYZ", "sp500_current")), {"XYZ": new})
    assert res["refetched"] == ["XYZ"]
    assert db.rows[("XYZ", dt.date(2026, 10, 12))]["adj_close"] == pytest.approx(100 * 0.9 * 0.99)


def test_delete_stale_dates_refuses_mass_delete():
    db = FakeDB()
    db.load("XYZ", _frame(list(range(10, 30)), start="2026-01-02"))
    new = _frame([1.0, 2.0], start="2026-01-02")
    assert ingest.delete_stale_dates(DBConn(db), "XYZ", new["date"]) == 0
    assert len(db.dates("XYZ")) == 20 and db.deleted == []


def test_ingest_nan_close_never_overwrites_stored_close(run):
    db = FakeDB()
    db.load("XYZ", _frame([100, 101, 102, 103, 104.0], start="2026-10-12"))
    new = _frame([100, 101, 102, 103, np.nan, 105, 106.0], start="2026-10-12")
    run(db, _pool(("XYZ", "sp500_current")), {"XYZ": new})
    assert db.rows[("XYZ", dt.date(2026, 10, 16))]["close"] == 104.0
    assert db.rows[("XYZ", dt.date(2026, 10, 20))]["close"] == 106.0


def test_ingest_return_dict_splits_failures(run):
    db = FakeDB()
    hist = {"GOOD": _frame([10, 11.0]), "GONE": None, "BAD": ValueError("malformed"),
            "SLOW": ingest.RateLimited("SLOW"), "OK2": _frame([10, 11.0])}
    res = run(db, _pool(*[(s, "sp500_current") for s in hist]), hist,
              infos={"GOOD": RuntimeError("no info")})
    assert res["with_data"] == 2 and res["attempted"] == 5
    assert sorted(res["no_data"]) == ["BAD", "GONE"] and res["failed"] == res["no_data"]
    assert res["rate_limited"] == ["SLOW"] and res["write_failed"] == []
    assert res["master_failed"] == ["GOOD"] and res["master_rows"] == 1      # "GOOD" info fails twice
    _assert_partition(res)


def test_ingest_write_failure_is_not_no_data(run, monkeypatch):
    db = FakeDB()
    real = ingest.upsert_ohlcv

    def flaky(conn, sym, df, at):
        if sym == "WF":
            raise RuntimeError("db down")
        return real(conn, sym, df, at)

    monkeypatch.setattr(ingest, "upsert_ohlcv", flaky)
    hist = {"OK": _frame([10, 11.0]), "WF": _frame([10, 11.0])}
    res = run(db, _pool(*[(s, "sp500_current") for s in hist]), hist)
    assert res["write_failed"] == ["WF"] and res["no_data"] == [] and res["failed"] == [] and res["with_data"] == 1
    assert db.dates("WF") == []


def test_ingest_master_retry_pass_uses_fresh_fetched_at(run, monkeypatch):
    db = FakeDB()
    calls = {"n": 0}
    real_dt = dt.datetime

    class Clock(real_dt):
        ticks = iter(real_dt(2026, 10, 19, 12, 0, s) for s in range(0, 59))

        @classmethod
        def now(cls, tz=None):
            return next(cls.ticks).replace(tzinfo=tz)

    monkeypatch.setattr(ingest.dt, "datetime", Clock)

    def fm(sym, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return {"symbol": sym, "name": "Late"}

    hist = {"A": _frame([10, 11.0]), "B": _frame([10, 11.0])}
    monkeypatch.setattr(ingest, "candidate_pool", lambda *a, **k: _pool(("A", "sp500_current"), ("B", "sp500_current")))
    import alpha101.db as D
    monkeypatch.setattr(D, "writer", lambda: DBConn(db))
    monkeypatch.setattr(ingest, "fetch_history", lambda s, **k: hist[s].copy())
    monkeypatch.setattr(ingest, "fetch_security_master", fm)
    res = ingest.ingest("us")
    assert res["master_failed"] == [] and res["master_rows"] == 2
    assert db.master["A"]["fetched_at"] > db.rows[("B", dt.date(2024, 1, 3))]["fetched_at"]   # stamped in the retry pass


def test_ingest_reused_ticker_is_rejected_and_not_ingested(run):
    db = FakeDB()
    old = pd.Timestamp("2012-01-03")
    pool = _pool(("SE", "sp500_changes", old, ("Spectra Energy",)),        # date rule: Yahoo starts 2024
                 ("CPWR", "sp500_changes", pd.Timestamp("2030-01-03"), ("Compuware",)),   # name rule only
                 ("FTI", "sp500_changes", pd.Timestamp("2030-01-03"), ("FMC Technologies",)),    # renamed: passes
                 ("AAPL", "sp500_current"))
    hist = {s: _frame([10, 11.0]) for s in ("SE", "CPWR", "FTI", "AAPL")}
    res = run(db, pool, hist, infos={"SE": {"name": "Spectra Energy"}, "CPWR": {"name": "Ocean Thermal Energy"},
                                     "FTI": {"name": "TechnipFMC plc"}})
    assert sorted(res["reused_ticker"]) == ["CPWR", "SE"]
    assert "after change date" in res["reused_reason"]["SE"] and "name" in res["reused_reason"]["CPWR"]
    assert db.dates("SE") == [] and db.dates("CPWR") == [] and "SE" not in db.master and "CPWR" not in db.master
    assert len(db.dates("FTI")) == 2 and len(db.dates("AAPL")) == 2
    assert res["with_data"] == 2 and res["no_data"] == []
    _assert_partition(res)


def test_ingest_unverifiable_identity_writes_nothing(run, monkeypatch):
    db = FakeDB()
    ohlcv, master = [], []
    real_o, real_m = ingest.upsert_ohlcv, ingest.upsert_security_master
    monkeypatch.setattr(ingest, "upsert_ohlcv", lambda c, sym, *a: ohlcv.append(sym) or real_o(c, sym, *a))
    monkeypatch.setattr(ingest, "upsert_security_master",
                        lambda c, rows, at: master.extend(r["symbol"] for r in rows) or real_m(c, rows, at))
    pool = _pool(("CHG", "sp500_changes", pd.Timestamp("2030-01-03"), ("Some Co",)), ("CUR", "sp500_current"))
    hist = {"CHG": _frame([10, 11.0]), "CUR": _frame([10, 11.0])}
    res = run(db, pool, hist, infos={"CHG": RuntimeError("no info"), "CUR": RuntimeError("no info")})
    # change-only symbol: identity cannot be verified -> nothing written, not even by the retry pass
    assert res["identity_unverified"] == ["CHG"] and "CHG" not in ohlcv and "CHG" not in master
    assert db.dates("CHG") == [] and "CHG" not in db.master
    assert "CHG" not in res["master_failed"] and res["reused_ticker"] == []
    # current constituent with failing info: still ingested, master retried (and still failing here)
    assert ohlcv == ["CUR"] and len(db.dates("CUR")) == 2 and res["master_failed"] == ["CUR"]
    assert res["with_data"] == 1 and master == []
    _assert_partition(res)


def test_ingest_nameless_info_on_change_only_symbol_writes_nothing(run, monkeypatch):
    db = FakeDB()
    ohlcv, master = [], []
    real_o, real_m = ingest.upsert_ohlcv, ingest.upsert_security_master
    monkeypatch.setattr(ingest, "upsert_ohlcv", lambda c, sym, *a: ohlcv.append(sym) or real_o(c, sym, *a))
    monkeypatch.setattr(ingest, "upsert_security_master",
                        lambda c, rows, at: master.extend(r["symbol"] for r in rows) or real_m(c, rows, at))
    pool = _pool(("CHG", "sp500_changes", pd.Timestamp("2030-01-03"), ("Some Co",)), ("CUR", "sp500_current"))
    hist = {"CHG": _frame([10, 11.0]), "CUR": _frame([10, 11.0])}
    res = run(db, pool, hist, infos={"CHG": {"name": None}, "CUR": {"name": "Current Co"}})
    # Yahoo returns an info dict with no name: identity still unverifiable -> nothing written, no retry
    assert res["identity_unverified"] == ["CHG"] and "CHG" not in ohlcv and "CHG" not in master
    assert db.dates("CHG") == [] and "CHG" not in db.master
    assert "CHG" not in res["master_failed"] and res["reused_ticker"] == []
    assert ohlcv == ["CUR"] and master == ["CUR"] and res["with_data"] == 1
    _assert_partition(res)


def test_ingest_master_write_failure_is_only_write_failed(run, monkeypatch):
    db = FakeDB()

    def boom(conn, rows, at):
        raise RuntimeError("master down")

    monkeypatch.setattr(ingest, "upsert_security_master", boom)
    res = run(db, _pool(("MF", "sp500_current")), {"MF": _frame([10, 11.0])})
    assert res["write_failed"] == ["MF"] and res["with_data"] == 0 and res["rows"] == 0
    assert res["master_failed"] == [] and res["master_rows"] == 0 and res["no_data"] == []
    _assert_partition(res)


def test_ingest_refetch_deletes_old_basis_row_of_dropped_nan_overlap_bar(run):
    db = FakeDB()
    db.load("XYZ", _frame([100.0 + i for i in range(10)], start="2026-10-05"))      # 10-05 .. 10-16
    new = _frame([50.0 + i / 2 for i in range(12)], splits={10: 2.0}, start="2026-10-05")
    new.loc[7, "close"] = np.nan                                                    # 10-14: NaN-close overlap bar
    res = run(db, _pool(("XYZ", "sp500_current")), {"XYZ": new})
    assert res["refetched"] == ["XYZ"]
    assert ("XYZ", dt.date(2026, 10, 14)) not in db.rows                            # old-basis row is gone
    assert db.deleted == [("XYZ", [dt.date(2026, 10, 14)])]
    assert db.rows[("XYZ", dt.date(2026, 10, 15))]["close"] == 50.0 + 8 / 2        # rest is on the new basis


def test_delete_stale_dates_keeps_dates_newer_than_last_yahoo_bar(caplog):
    db = FakeDB()
    db.load("XYZ", _frame(list(range(10, 30)), start="2026-01-02"))
    all_dates = db.dates("XYZ")
    new = [d for d in all_dates[:17] if d != all_dates[5]]                           # Yahoo ends 3 bars early
    with caplog.at_level("WARNING", logger="alpha101.ingest"):
        assert ingest.delete_stale_dates(DBConn(db), "XYZ", pd.Series(pd.to_datetime(new))) == 1
    assert db.deleted == [("XYZ", [all_dates[5]])]
    assert all(("XYZ", d) in db.rows for d in all_dates[17:])                        # newer stored dates survive
    assert any("newer than Yahoo's last bar" in m for m in caplog.messages)


def test_delete_stale_dates_guard_boundary_is_exactly_20_percent():
    db = FakeDB()
    db.load("XYZ", _frame(list(range(10, 20)), start="2026-01-02"))                  # 10 stored dates
    d = db.dates("XYZ")
    keep = [x for x in d if x not in (d[2], d[4])]                                   # 2 stale = exactly 20%: deleted
    assert ingest.delete_stale_dates(DBConn(db), "XYZ", pd.Series(pd.to_datetime(keep))) == 2
    db2 = FakeDB()
    db2.load("XYZ", _frame(list(range(10, 20)), start="2026-01-02"))
    keep3 = [x for x in d if x not in (d[2], d[4], d[6])]                            # 3 stale = 30%: refused
    assert ingest.delete_stale_dates(DBConn(db2), "XYZ", pd.Series(pd.to_datetime(keep3))) == 0
    assert len(db2.dates("XYZ")) == 10


def test_is_rate_limit_matches_429_as_a_whole_token():
    assert ingest._is_rate_limit(RuntimeError("HTTP 429")) and ingest._is_rate_limit(RuntimeError("Too Many Requests"))
    assert not ingest._is_rate_limit(RuntimeError("took 14290 ms")) and not ingest._is_rate_limit(ValueError("x4291"))


def test_download_does_not_retry_4xx_other_than_429(monkeypatch, tmp_path):
    import requests
    calls, sleeps = [], []

    class R:
        status_code, text = 404, "nope"

        def raise_for_status(self):
            raise requests.HTTPError("404")

    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(1) or R())
    monkeypatch.setattr(constituents.time, "sleep", sleeps.append)
    with pytest.raises(requests.HTTPError):
        constituents._download("http://x", tmp_path, "p")
    assert len(calls) == 1 and sleeps == []


def test_ingest_only_restricts_the_run(run):
    db = FakeDB()
    pool = _pool(("A", "sp500_current"), ("B", "sp500_current"))
    res = run(db, pool, {"A": _frame([1, 2.0]), "B": _frame([1, 2.0])}, only=["B"])
    assert res["attempted"] == 1 and db.dates("A") == [] and len(db.dates("B")) == 2
