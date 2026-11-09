import datetime as dt

import pandas as pd
import pytest

import ishares

D = dt.date
HEAD = 'Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency'


def csvtext(rows, as_of='"Dec 15, 2025"'):
    return f"﻿Fund Holdings as of,{as_of}\n \n{HEAD}\n" + "\n".join(rows) + "\n \n"


def row(t, name, ac="Equity", mv="1,000.00", sh="100.00", px="10.00", cur="CAD", sector="Financials"):
    return f'"{t}","{name}","{sector}","{ac}","{mv}","1.00","{mv}","{sh}","{px}","Canada","TSX","{cur}","1.00","{cur}"'


def parse_text(tmp_path, text, ref=None):
    p = tmp_path / "f.csv"
    p.write_text(text, encoding="utf-8")
    return ishares.parse(p, "XIC", ref)


def test_parser_traps(tmp_path):
    text = csvtext([
        row("NA", "NATIONAL BANK OF CANADA", mv="2,000.00", sh="1,000.00", px="2.00"),
        row("ABC", "NO PRICE INC", mv="1,500.00", sh="300.00", px="-"),
        row("MLPFT", "CASH COLLATERAL", ac="Cash Collateral and Margins"),
        row("PTZ5", "S&P/TSE 60 INDEX DEC 25", ac="Futures"),
        row("AAPL", "APPLE", cur="USD"),
        row("GIB/A", "CGI GROUP A"),
        row("FET UN", "WHITESPACE TRUST"),
        row("ZERO", "ZERO PRICE", mv="0.61", sh="60,915.00", px="0.00"),
    ])
    df, ex = parse_text(tmp_path, text)
    assert set(df["ticker"]) == {"NA", "ABC", "GIB/A"}
    assert df.set_index("ticker").loc["NA", "ysym"] == "NA.TO"
    assert df.set_index("ticker").loc["ABC", "price"] == pytest.approx(5.0)       # MV / Shares
    assert df.set_index("ticker").loc["GIB/A", "ysym"] == "GIB-A.TO"              # crosswalk
    assert df["as_of"].iloc[0] == D(2025, 12, 15)
    reasons = {e["ticker"]: e["reason"] for e in ex}
    assert "currency" in reasons["AAPL"] and "unclean" in reasons["FET UN"] and "non-positive" in reasons["ZERO"]
    assert "MLPFT" not in reasons and "PTZ5" not in reasons                        # non-equity rows silently skipped


def test_dual_scheme_dedupe_by_name(tmp_path):
    text = csvtext([row("GOLD", "BARRICK GOLD CORP", mv="5,000.00"), row("ABX", "BARRICK GOLD CORP", mv="5,000.00"),
                    row("945", "MANULIFE", mv="900.00"), row("MFC", "MANULIFE", mv="900.00"),
                    row("RY", "ROYAL BANK", mv="7,000.00")], as_of='"Dec 30, 2011"')
    df, ex = parse_text(tmp_path, text, ref={"ABX", "MFC", "RY"})
    assert sorted(df["ticker"]) == ["ABX", "MFC", "RY"]
    assert {e["ticker"] for e in ex} == {"GOLD", "945"}
    df2, _ = parse_text(tmp_path, text)                    # no reference: TSX-style ticker beats digit-style
    assert "945" not in set(df2["ticker"]) and df2["name"].is_unique


def test_stub_file(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text('﻿Fund Holdings as of,"-"\n', encoding="utf-8")
    df, ex = ishares.parse(p, "XIC")
    assert df.empty and ishares.classify(p.read_text(encoding="utf-8-sig"), D(2017, 1, 3)) == "stub"


def test_classify():
    assert ishares.classify(csvtext([]), D(2025, 12, 15)) == "present"
    assert ishares.classify(csvtext([]), D(2025, 12, 16)) == "mismatch"
    assert ishares.classify("<html>blocked</html>", D(2025, 12, 15)) == "bad"


class FakeResp:
    def __init__(self, code, text="", ctype="text/csv", headers=None):
        self.status_code, self.content, self.headers = code, text.encode(), {"content-type": ctype, **(headers or {})}


class FakeSession:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return r(url) if callable(r) else r


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(ishares, "MIN_GAP", 0.0)
    monkeypatch.setattr(ishares, "BACKOFF", (0, 0))
    monkeypatch.setattr(ishares.time, "sleep", lambda s: None)


def test_fetch_caches_and_resumes(tmp_path, quick):
    s = FakeSession([FakeResp(200, csvtext([row("RY", "ROYAL BANK")]))])
    r1 = ishares.fetch("XIC", D(2025, 12, 15), s, tmp_path)
    assert r1["status"] == "present" and not r1["from_cache"] and (tmp_path / "XIC" / "20251215.csv").exists()
    r2 = ishares.fetch("XIC", D(2025, 12, 15), s, tmp_path)
    assert r2["from_cache"] and len(s.calls) == 1                      # re-run: no request
    s404 = FakeSession([FakeResp(404)])
    assert ishares.fetch("XCS", D(2020, 1, 2), s404, tmp_path)["status"] == "missing"
    ishares.fetch("XCS", D(2020, 1, 2), s404, tmp_path)
    assert len(s404.calls) == 1 and (tmp_path / "XCS" / "20200102.missing").exists()


def test_retry_429_then_ok_and_blocked(tmp_path, quick):
    s = FakeSession([FakeResp(429), FakeResp(503), FakeResp(200, csvtext([row("RY", "ROYAL BANK")]))])
    assert ishares.fetch("XIC", D(2025, 12, 15), s, tmp_path)["status"] == "present" and len(s.calls) == 3
    with pytest.raises(ishares.Blocked):
        ishares.fetch("XIC", D(2025, 12, 16), FakeSession([FakeResp(403)]), tmp_path)
    with pytest.raises(ishares.Blocked):
        ishares.fetch("XIC", D(2025, 12, 17), FakeSession([FakeResp(429)]), tmp_path)        # never recovers
    with pytest.raises(ishares.Blocked):
        ishares.fetch("XIC", D(2025, 12, 18), FakeSession([FakeResp(200, "<html/>", "text/html")]), tmp_path)


def test_politeness_gap(monkeypatch):
    sleeps, t = [], [100.0]
    monkeypatch.setattr(ishares, "MIN_GAP", 1.1)
    ishares._last_request[0] = 0.0
    s = FakeSession([FakeResp(200, "x")])
    for _ in range(2):
        ishares._http_get("u", s, sleep=sleeps.append, clock=lambda: t[0])
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(1.1)          # second call waits the full gap


def test_month_end_fallback_only_when_allowed(tmp_path, quick):
    ok = csvtext([row("RY", "ROYAL BANK")], '"Jan 30, 2015"')
    stub = 'Fund Holdings as of,"-"\n'
    s = FakeSession([lambda u: FakeResp(200, stub) if "20150131" in u else FakeResp(200, ok)])
    r = ishares.fetch("XIC", D(2015, 1, 30), s, tmp_path, fallback_days=3)
    assert r["status"] == "present"
    s = FakeSession([FakeResp(200, 'Fund Holdings as of,"-"\n')])
    assert ishares.fetch("XIC", D(2015, 1, 29), s, tmp_path, fallback_days=0)["status"] == "stub" and len(s.calls) == 1
    s = FakeSession([lambda u: FakeResp(200, stub) if "20150227" in u else FakeResp(200, csvtext([row("RY", "R")], '"Feb 26, 2015"'))])
    r = ishares.fetch("XIC", D(2015, 2, 27), s, tmp_path, fallback_days=3)
    assert r["as_of_used"] == D(2015, 2, 26) and r["fallback_steps"] == 1


def test_plan_defers_holdout_and_keeps_dec30_2011(generic_cal):
    plan = ishares.request_plan(generic_cal)
    fetch_rows = plan[~plan["holdout"]]
    assert D(2011, 12, 30) in set(fetch_rows["date"])                     # the 2012 event's D_{y-1}
    assert fetch_rows["date"].min() == D(2011, 2, 28)                     # first 2011 placebo base
    assert (fetch_rows["date"] < D(2012, 1, 1)).sum() == 2 * 10           # 9 placebo bases + Dec 30, two ETFs
    held = plan[plan["holdout"]]
    assert len(held) > 0 and set(held["roles"].str.split("|").sum()) == {"M_holdout"}
    assert held["date"].min() < D(2008, 1, 1)
    p1 = plan[plan["priority"] == 1]
    assert {"D_prev", "pre_review", "R", "L", "L-6", "L-5", "first_jan", "X"} <= set(p1["roles"].str.split("|").sum())


def test_2011_placebo_bases_are_fetched_and_only_m_series_is_sealed(generic_cal):
    n = ishares.needed_dates(generic_cal)
    pb = n[n["role"].str.startswith("placebo_base") & (n["date"].map(lambda d: d.year) == 2011)]
    assert len(pb) == 9 and not pb["holdout"].any()
    assert n.loc[n["role"] == "M_holdout", "holdout"].all()
    assert not set(pb["date"]) & set(n.loc[n["role"] == "M_holdout", "date"])
    fetched = ishares.request_plan(generic_cal).query("not holdout")["date"]
    assert not fetched.map(lambda d: 2007 <= d.year <= 2010).any()
    assert set(fetched[fetched.map(lambda d: d.year == 2011)]) == set(pb["date"]) | {D(2011, 12, 30)}


def test_coverage_excludes_2017_gap(generic_cal):
    rows = []
    for etf in ("XIC", "XCS"):
        rows += [{"etf": etf, "date": D(2016, 12, 30), "status": "present"},
                 {"etf": etf, "date": D(2017, 1, 3), "status": "stub"},
                 {"etf": etf, "date": D(2017, 1, 4), "status": "present"},
                 {"etf": etf, "date": D(2018, 1, 3), "status": "stub"},
                 {"etf": etf, "date": D(2007, 12, 31), "status": "deferred (holdout)"}]
    c = ishares.coverage(pd.DataFrame(rows)).set_index(["etf", "year"])
    assert c.loc[("XIC", "ALL"), "requested"] == 4 and c.loc[("XIC", "ALL"), "present"] == 2
    assert c.loc[("XIC", "ALL_ex_2017_gap"), "requested"] == 3            # the 2017-01-03 stub is the known gap
    assert c.loc[("XIC", "ALL"), "deferred"] == 1


def test_same_name_is_kept_after_the_scheme_change(tmp_path):
    text = csvtext([row("LAAC", "LITHIUM AMERICAS", mv="500.00"), row("LAC", "LITHIUM AMERICAS", mv="400.00")])
    df, ex = parse_text(tmp_path, text)                       # as of Dec 2025: two listings are two securities
    assert sorted(df["ticker"]) == ["LAAC", "LAC"] and not ex


def test_duplicate_ticker_aggregation_survives_missing_market_value(tmp_path):
    text = csvtext([row("RY", "ROYAL BANK", mv="1,000.00", sh="100.00", px="10.00"),
                    row("RY", "ROYAL BANK", mv="-", sh="50.00", px="12.00")])
    df, _ = parse_text(tmp_path, text)
    r = df.iloc[0]
    assert r["shares"] == 150.0 and r["price"] == pytest.approx((10 * 100 + 12 * 50) / 150)     # not MV/shares = 6.67 or 0


def test_is_cached_counts_hard_miss_sentinel(tmp_path):
    assert not ishares.is_cached("XIC", D(2020, 1, 2), tmp_path)
    (tmp_path / "XIC").mkdir()
    (tmp_path / "XIC" / "20200102.missing").write_text("")
    (tmp_path / "XIC" / "20200103.csv").write_text("x")
    assert ishares.is_cached("XIC", D(2020, 1, 2), tmp_path) and ishares.is_cached("XIC", D(2020, 1, 3), tmp_path)


def test_load_to_db_closes_its_connection_when_parse_fails(tmp_path, monkeypatch):
    import alpha101.db as db

    class Conn:
        closed = False
        def cursor(self): raise RuntimeError("no db in tests")
        def close(self): self.closed = True

    c = Conn()
    monkeypatch.setattr(db, "writer", lambda: c)
    with pytest.raises(RuntimeError):
        ishares.load_to_db(pd.DataFrame({"etf": ["XIC"], "date": [D(2020, 1, 2)], "status": ["present"],
                                         "as_of_used": [D(2020, 1, 2)]}), tmp_path)
    assert c.closed
