"""iShares (BlackRock CA) daily holdings files for XIC (S&P/TSX Capped Composite) and XCS (SmallCap).

Raw files are cached under data/raw/ishares/<ETF>/<YYYYMMDD>.csv (git-ignored; BlackRock licence: personal,
non-commercial use, never committed). A hard miss is cached as <YYYYMMDD>.missing so reruns skip it.
Parsed rows go to a private `etf_holdings_snapshot` table.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("taxloss.ishares")

ETFS = {
    "XIC": ("239837", "ishares-sptsx-capped-composite-index-etf"),
    "XCS": ("239852", "ishares-sptsx-small-cap-index-etf"),
}
URL = ("https://www.blackrock.com/ca/investors/en/products/{id}/{slug}/1464253357814.ajax"
       "?fileType=csv&fileName={etf}_holdings&dataType=fund&asOfDate={d:%Y%m%d}")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
MIN_GAP = 1.1                   # seconds between requests (politeness: >= 1 s)
BACKOFF = (5, 15, 45, 120)      # waits after 429 / 5xx / network errors
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "ishares"

HOLDINGS_DDL = """CREATE TABLE IF NOT EXISTS etf_holdings_snapshot (
  etf VARCHAR(10) NOT NULL, as_of DATE NOT NULL, ticker VARCHAR(20) NOT NULL, ysym VARCHAR(20) NULL,
  name VARCHAR(200) NULL, sector VARCHAR(80) NULL, price DOUBLE NULL, shares DOUBLE NULL,
  market_value DOUBLE NULL, weight DOUBLE NULL, currency CHAR(3) NULL, fetched_at DATETIME NOT NULL,
  PRIMARY KEY (etf, as_of, ticker)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
HOLDINGS_UPSERT_SQL = (
    "INSERT INTO etf_holdings_snapshot (etf,as_of,ticker,ysym,name,sector,price,shares,market_value,weight,"
    "currency,fetched_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    " ON DUPLICATE KEY UPDATE ysym=VALUES(ysym),name=VALUES(name),sector=VALUES(sector),price=VALUES(price),"
    "shares=VALUES(shares),market_value=VALUES(market_value),weight=VALUES(weight),currency=VALUES(currency),"
    "fetched_at=VALUES(fetched_at)")


DUAL_SCHEME_END = dt.date(2012, 6, 1)      # identifier scheme changed 2012-05-31: name dedupe only before this

# Non-standard iShares tickers with a known Yahoo form; everything else unclean is excluded and listed.
CROSSWALK = {"GIB/A": "GIB.A"}


class Blocked(RuntimeError):
    """BlackRock refused us (persistent 403/429/5xx or an HTML block page); stop and report."""


# --- needed dates ------------------------------------------------------------------------------------------------
def needed_dates(cal, years=range(2012, 2026), holdout_years=range(2007, 2012)) -> pd.DataFrame:
    """One row per (date, role): columns date, role, year, priority (1 = event-critical), fallback_ok, holdout.

    Event y (2012-2025): D_{y-1}, pre-review day, R, L, L-6, L-5, first Jan day, exit X (priority 1); every trading day
    L..X3 (priority 2); placebo months Feb..Oct (month-M-12 end, L_M-1, L_M, end of M+1; priority 2); M-series
    Nov 30 / Dec 31 / Jan 31 / prior Dec 31 (priority 2). Month-end roles may fall back up to 3 business days.
    Only the 2007-2011 M-series dates (role M_holdout) are holdout=True and must not be fetched; the 2011 month-end
    placebo bases (bases of the 2012 placebo months) are ordinary dates.
    """
    rows = []

    def add(d, role, y, prio, fb=False):
        rows.append({"date": d, "role": role, "year": y, "priority": prio, "fallback_ok": fb,
                     "holdout": role == "M_holdout"})

    for y in years:
        e = cal.event_row(y)
        for role, key in (("D_prev", "D_prev"), ("pre_review", "pre_review_day"), ("R", "R"), ("L", "L"),
                          ("L-6", "VE0_rank_day"), ("L-5", "VE0_entry"), ("first_jan", "first_jan_day"),
                          ("X", "jan_exit_X")):
            add(e[key], role, y, 1)
        x3 = e["VX3_on_or_after_feb15"]
        d = e["L"]
        while d <= x3:
            add(d, "window", y, 2)
            d = cal.shift(d, 1)
        add(cal.last_trading_day_of_month(y, 11), "M_nov", y, 2, True)
        add(cal.last_trading_day_of_month(y, 12), "M_dec", y, 2, True)
        for m in range(2, 11):
            add(cal.last_trading_day_of_month(y - 1, m), f"placebo_base_{m}", y, 2, True)
            add(cal.placebo_rank_day(y, m), f"placebo_rank_{m}", y, 2)
            add(cal.placebo_L(y, m), f"placebo_L_{m}", y, 2)
            add(cal.last_trading_day_of_month(y, m + 1), f"placebo_exit_{m}", y, 2, True)
    for y in holdout_years:      # M-series 2007-2011: Dec(y-1), Nov y, Dec y, Jan y+1 month-ends; never fetched
        for yy, mm in ((y - 1, 12), (y, 11), (y, 12), (y + 1, 1)):
            add(cal.last_trading_day_of_month(yy, mm), "M_holdout", y, 2, True)
    return pd.DataFrame(rows)


# --- fetch -------------------------------------------------------------------------------------------------------
_last_request = [0.0]


def _path(etf: str, d: dt.date, raw_dir: Path = RAW_DIR, suffix=".csv") -> Path:
    return Path(raw_dir) / etf / f"{d:%Y%m%d}{suffix}"


def is_cached(etf: str, d: dt.date, raw_dir: Path = RAW_DIR) -> bool:
    """A holdings file or a cached hard miss exists for (etf, d): fetching it costs no request."""
    return _path(etf, d, raw_dir).exists() or _path(etf, d, raw_dir, ".missing").exists()


def header_date(text: str):
    """The 'Fund Holdings as of' date of a file, or None for the stub ('-')."""
    first = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
    m = re.match(r'\s*Fund Holdings as of\s*,\s*"?([^"]*)"?\s*$', first)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1).strip(), "%b %d, %Y").date()
    except ValueError:
        return None


def classify(text: str, requested: dt.date) -> str:
    """present | stub | mismatch (file for another date) | bad (not a holdings file)."""
    first = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
    if not first.startswith("Fund Holdings as of"):
        return "bad"
    a = header_date(text)
    if a is None:
        return "stub"
    return "present" if a == requested else "mismatch"


def _http_get(url: str, session, sleep=time.sleep, clock=time.monotonic) -> str | None:
    """GET with the politeness gap and 429/5xx backoff. Returns text, None for a hard 404; raises Blocked."""
    last = ""
    for attempt in range(len(BACKOFF) + 1):
        wait = MIN_GAP - (clock() - _last_request[0])
        if wait > 0:
            sleep(wait)
        _last_request[0] = clock()
        try:
            r = session.get(url, headers=UA, timeout=60)
            code = r.status_code
        except Exception as e:  # noqa: BLE001 - network errors are retried
            code, r, last = None, None, type(e).__name__
        if code == 200:
            ctype = r.headers.get("content-type", "")
            if "html" in ctype.lower():
                raise Blocked("HTML response (block page?) from BlackRock")
            return r.content.decode("utf-8-sig", errors="replace")
        if code == 404:
            return None
        if code is not None:
            last = f"HTTP {code}"
        if code is not None and code != 429 and code < 500:
            raise Blocked(f"HTTP {code}")
        if attempt < len(BACKOFF):
            pause = BACKOFF[attempt]
            if r is not None and r.headers.get("Retry-After", "").isdigit():
                pause = max(pause, int(r.headers["Retry-After"]))
            log.warning("%s; retry in %ds", last, pause)
            sleep(pause)
    raise Blocked(f"gave up after {len(BACKOFF) + 1} attempts ({last})")


def fetch(etf: str, d: dt.date, session=None, raw_dir: Path = RAW_DIR, fallback_days: int = 0) -> dict:
    """Fetch (or reuse the cache of) one holdings file. Returns {etf, requested, status, as_of_used, path,
    from_cache}; status is present | stub | missing | mismatch. With fallback_days > 0 (month-end roles only) a
    stub/missing date falls back up to that many BUSINESS days earlier; as_of_used says which day answered."""
    import requests
    session = session or requests.Session()
    pid, slug = ETFS[etf]
    tried = []
    day = d
    for step in range(fallback_days + 1):
        if step:
            day -= dt.timedelta(days=1)
            while day.weekday() >= 5:
                day -= dt.timedelta(days=1)
        p, miss = _path(etf, day, raw_dir), _path(etf, day, raw_dir, ".missing")
        cached = p.exists() or miss.exists()
        if p.exists():
            text = p.read_text(encoding="utf-8-sig", errors="replace")
        elif miss.exists():
            text = None
        else:
            text = _http_get(URL.format(id=pid, slug=slug, etf=etf, d=day), session)
            p.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                miss.write_text("")
            else:
                if classify(text, day) == "bad":
                    raise Blocked("response is not a holdings file")
                p.write_text(text, encoding="utf-8")
        status = "missing" if text is None else classify(text, day)
        tried.append(status)
        if status in ("present", "mismatch") or step == fallback_days:
            return {"etf": etf, "requested": d, "status": status,
                    "as_of_used": day if status == "present" else None, "path": p if text is not None else None,
                    "from_cache": cached, "fallback_steps": step}
    raise AssertionError("unreachable")


# --- parse -------------------------------------------------------------------------------------------------------
PARSED_COLUMNS = ["etf", "as_of", "ticker", "ysym", "name", "sector", "price", "shares", "market_value", "weight",
                  "currency"]


def _num(s):
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse(path, etf: str, ref_tickers: set[str] | None = None):
    """Parse one cached holdings file -> (DataFrame[PARSED_COLUMNS], exclusions list of dicts).

    Traps handled: no NA-to-NaN conversion (ticker "NA" stays), Equity asset class only, CAD lines only, `Price` "-"
    -> market value / shares, thousands separators, unclean tickers (yahoo_symbol ValueError / empty) go to the
    exclusion list with a reason, non-positive price or shares excluded, and a file holding both identifier schemes
    (before 2012-06) is deduped by Name (keeping the ticker in `ref_tickers` if given, else the larger market value).
    """
    from alpha101.data.constituents import yahoo_symbol
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    as_of = header_date(text)
    if as_of is None:
        return pd.DataFrame(columns=PARSED_COLUMNS), [{"ticker": "", "reason": "stub file"}]
    rows = list(csv.reader(io.StringIO(text)))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Ticker")
    head = [h.strip() for h in rows[hi]]
    col = {h: i for i, h in enumerate(head)}
    shares_col = "Shares" if "Shares" in col else "Par Value"

    def g(r, name):
        i = col.get(name)
        return r[i].strip() if i is not None and i < len(r) else ""

    out, excl = [], []
    for r in rows[hi + 1:]:
        if len(r) < len(head) - 2 or not r[0].strip() and not g(r, "Name"):
            continue                                    # footer / blank lines
        if g(r, "Asset Class") != "Equity":
            continue
        ticker = g(r, "Ticker")
        if g(r, "Currency") != "CAD":
            excl.append({"ticker": ticker, "name": g(r, "Name"), "reason": f"currency {g(r, 'Currency')!r}"})
            continue
        mv, sh, px = _num(g(r, "Market Value")), _num(g(r, shares_col)), _num(g(r, "Price"))
        if px is None and mv is not None and sh:
            px = mv / sh
        if not px or px <= 0 or not sh or sh <= 0:
            excl.append({"ticker": ticker, "name": g(r, "Name"), "reason": "non-positive price or shares"})
            continue
        try:
            ysym = yahoo_symbol(CROSSWALK.get(ticker, ticker), "ca")
        except ValueError:
            ysym = ""
        if not ysym:
            excl.append({"ticker": ticker, "name": g(r, "Name"), "reason": "unclean ticker (no Yahoo symbol)"})
            continue
        out.append({"etf": etf, "as_of": as_of, "ticker": ticker, "ysym": ysym, "name": g(r, "Name"),
                    "sector": g(r, "Sector"), "price": px, "shares": sh, "market_value": mv,
                    "weight": _num(g(r, "Weight (%)")), "currency": "CAD"})
    df = pd.DataFrame(out, columns=PARSED_COLUMNS)
    if len(df):
        dup = df.duplicated("name", keep=False) & (as_of < DUAL_SCHEME_END)
        if dup.any():
            pref = df["ticker"].isin(ref_tickers) if ref_tickers else pd.Series(False, index=df.index)
            tsx_style = df["ticker"].str.fullmatch(r"[A-Z]+([.-][A-Z]+)*")      # old scheme has 0P6E, 945, ...
            order = df.assign(_p=pref, _t=tsx_style, _mv=df["market_value"].fillna(0)).sort_values(
                ["name", "_p", "_t", "_mv", "ticker"], ascending=[True, False, False, False, True])
            keep = order.drop_duplicates("name").index
            for i in df.index[dup & ~df.index.isin(keep)]:
                excl.append({"ticker": df.at[i, "ticker"], "name": df.at[i, "name"], "reason": "duplicate Name (dual identifier scheme)"})
            df = df.loc[df.index.isin(keep) | ~dup].reset_index(drop=True)
        # same ticker twice (several lots): the DB key is (etf, as_of, ticker); aggregate
        if df.duplicated("ticker").any():
            agg = df.groupby("ticker", as_index=False).agg(
                {"etf": "first", "as_of": "first", "ysym": "first", "name": "first", "sector": "first",
                 "shares": "sum", "market_value": "sum", "weight": "sum", "currency": "first"})
            agg["price"] = df.assign(_v=df["price"] * df["shares"]).groupby("ticker")["_v"].sum().to_numpy() / agg["shares"]
            df = agg[PARSED_COLUMNS]
    return df, excl


# --- DB ----------------------------------------------------------------------------------------------------------
def create_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(HOLDINGS_DDL)
    conn.commit()


def upsert_holdings(conn, df: pd.DataFrame, fetched_at: dt.datetime) -> int:
    rows = [(r.etf, r.as_of, r.ticker[:20], r.ysym[:20], r.name[:200], r.sector[:80], r.price, r.shares,
             None if pd.isna(r.market_value) else r.market_value, None if pd.isna(r.weight) else r.weight,
             r.currency, fetched_at) for r in df.itertuples(index=False)]
    with conn.cursor() as cur:
        for i in range(0, len(rows), 2000):
            cur.executemany(HOLDINGS_UPSERT_SQL, rows[i:i + 2000])
            conn.commit()
    return len(rows)


# --- orchestration -----------------------------------------------------------------------------------------------
GAP_2017 = (dt.date(2017, 1, 1), dt.date(2017, 6, 30))
TIME_CAP_MIN = 90


def request_plan(cal) -> pd.DataFrame:
    """One row per (etf, date) to request: priority, roles, fallback_days, deferred (holdout)."""
    n = needed_dates(cal)
    g = n.groupby("date").agg(priority=("priority", "min"), roles=("role", lambda r: "|".join(sorted(set(r)))),
                              fb=("fallback_ok", "min"), holdout=("holdout", "min")).reset_index()
    g["fallback_days"] = g["fb"].map(lambda b: 3 if b else 0)
    g["holdout"] = g["holdout"].astype(bool)
    plan = pd.concat([g.assign(etf=e) for e in ETFS], ignore_index=True)
    return plan.drop(columns="fb").sort_values(["priority", "date", "etf"]).reset_index(drop=True)


def fetch_all(plan: pd.DataFrame, raw_dir: Path = RAW_DIR, cap_min: float = TIME_CAP_MIN, session=None,
              clock=time.monotonic) -> pd.DataFrame:
    """Fetch every non-holdout row of `plan`, priority 1 first. Resumable (cache hits cost no request). If the
    projection from the first 30 live requests exceeds cap_min, priority-2 rows stay 'pending'. On Blocked the
    remaining rows stay 'pending' and `blocked` (in df.attrs) carries the reason."""
    import requests
    session = session or requests.Session()
    res = plan.copy()
    for c in ("status", "as_of_used", "fallback_steps"):
        res[c] = None
    res.loc[res["holdout"], "status"] = "deferred (holdout)"
    todo = [i for i in res.index if not res.at[i, "holdout"]]
    live = [0, 0.0]
    start = None
    skip_p2 = False
    res.attrs["blocked"] = None
    try:
        for i in todo:
            row = res.loc[i]
            if row["priority"] == 2 and skip_p2:
                res.at[i, "status"] = "pending"
                continue
            was_cached = is_cached(row["etf"], row["date"], raw_dir)
            t0 = clock()
            if start is None and not was_cached:
                start = t0
            r = fetch(row["etf"], row["date"], session, raw_dir, int(row["fallback_days"]))
            res.at[i, "status"], res.at[i, "as_of_used"], res.at[i, "fallback_steps"] = (
                r["status"], r["as_of_used"], r["fallback_steps"])
            if not was_cached:
                live[0] += 1
                live[1] += clock() - t0
                if live[0] == 30 and not skip_p2:
                    uncached = sum(1 for j in todo if res.at[j, "status"] is None
                                   and not is_cached(res.at[j, "etf"], res.at[j, "date"], raw_dir))
                    proj = (live[1] / 30) * (uncached + 30) / 60
                    log.info("projected fetch time %.0f min for %d requests", proj, uncached + 30)
                    if proj > cap_min:
                        skip_p2 = True
                        log.warning("projection %.0f min > cap %.0f: fetching event-critical dates only", proj, cap_min)
            if live[0] and live[0] % 100 == 0:
                log.info("%d live requests, %.2fs/request", live[0], live[1] / live[0])
    except Blocked as e:
        res.attrs["blocked"] = str(e)
        log.error("BLOCKED: %s", e)
    res["status"] = res["status"].fillna("pending")
    res.attrs["live_requests"] = live[0]
    return res


def load_to_db(status: pd.DataFrame, raw_dir: Path = RAW_DIR, conn=None) -> dict:
    """Parse every present file (the file that actually answered) and upsert into etf_holdings_snapshot.
    Returns {rows, files, exclusions: DataFrame}."""
    from alpha101.db import writer
    own = conn is None
    conn = conn or writer()
    fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
    ok = status[status["status"] == "present"].copy()
    ok["file_date"] = ok["as_of_used"]
    ok = ok.drop_duplicates(["etf", "file_date"]).sort_values(["etf", "file_date"])
    rows = files = 0
    excl = []
    try:
        create_table(conn)
        ref = {}
        for etf, grp in ok.groupby("etf"):
            late = grp[grp["file_date"] >= dt.date(2012, 6, 1)]
            if len(late):
                ref[etf] = set(parse(_path(etf, late.iloc[0]["file_date"], raw_dir), etf)[0]["ticker"])
        for r in ok.itertuples(index=False):
            df, ex = parse(_path(r.etf, r.file_date, raw_dir), r.etf, ref.get(r.etf) if r.file_date < dt.date(2012, 6, 1) else None)
            conn.ping(reconnect=True)
            rows += upsert_holdings(conn, df, fetched_at)
            files += 1
            excl += [{"etf": r.etf, "as_of": r.file_date, **e} for e in ex]
    finally:
        if own:
            conn.close()
    return {"rows": rows, "files": files, "exclusions": pd.DataFrame(excl, columns=["etf", "as_of", "ticker", "name", "reason"])}


def coverage(status: pd.DataFrame) -> pd.DataFrame:
    """holdings_coverage.csv: per ETF and calendar year of the requested date, plus ALL and ALL_ex_2017_gap rows.
    requested = non-holdout dates (present+stub+missing+mismatch+pending); the 2017 gap = Jan-Jun 2017 dates that
    are not present."""
    s = status.copy()
    s["year"] = s["date"].map(lambda d: d.year)
    s["in_2017_gap"] = s["date"].map(lambda d: GAP_2017[0] <= d <= GAP_2017[1]) & (s["status"] != "present")
    cols = ["present", "stub", "missing", "mismatch", "pending", "deferred (holdout)"]

    def agg(g, label, etf):
        c = g["status"].value_counts()
        row = {"etf": etf, "year": label}
        row.update({k.split(" ")[0]: int(c.get(k, 0)) for k in cols})
        row["requested"] = len(g) - row["deferred"]
        row["present_pct"] = round(100 * row["present"] / row["requested"], 2) if row["requested"] else None
        return row

    out = []
    for etf, g in s.groupby("etf"):
        for y, gy in g.groupby("year"):
            out.append(agg(gy, y, etf))
        out.append(agg(g, "ALL", etf))
        out.append(agg(g[~g["in_2017_gap"]], "ALL_ex_2017_gap", etf))
    return pd.DataFrame(out)


# --- read back / deleted-before-R ---------------------------------------------------------------------------------
def read_snapshots(conn, start=None, end=None) -> pd.DataFrame:
    """etf_holdings_snapshot rows as a DataFrame (pure %s placeholders)."""
    where, params = [], []
    if start:
        where.append("as_of >= %s")
        params.append(start)
    if end:
        where.append("as_of <= %s")
        params.append(end)
    sql = "SELECT etf,as_of,ticker,ysym,name,sector,price,shares,market_value,weight FROM etf_holdings_snapshot"
    if where:
        sql += " WHERE " + " AND ".join(where)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    df = pd.DataFrame(list(rows), columns=["etf", "as_of", "ticker", "ysym", "name", "sector", "price", "shares",
                                           "market_value", "weight"])
    for c in ("price", "shares", "market_value", "weight"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df


def deleted_before_R(snap: pd.DataFrame, cal, years=range(2012, 2026)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Names in the XIC/XCS file of the last trading day before the December review effective date but absent from
    both ETFs' R files (descriptive artifact only; the would-be YTD rank is NOT computed here, the ranking step adds it).
    Also returns a per-event summary (names dropped / added between the two dates, and whether a change is visible)."""
    rows, summ = [], []
    for y in years:
        e = cal.event_row(y)
        pre, R = e["pre_review_day"], e["R"]
        a, b = snap[snap["as_of"] == pre], snap[snap["as_of"] == R]
        s = {"year": y, "pre_review_day": pre, "dec_review_effective": e["dec_review_effective"], "R": R,
             "pre_file_rows": len(a), "R_file_rows": len(b),
             "pre_equals_R": pre == R, "effective_after_R": e["dec_review_effective"] > R}
        if a.empty or b.empty:
            s.update(dropped=None, added=None)
        else:
            gone = a[~a["ysym"].isin(set(b["ysym"]))]
            s.update(dropped=gone["ysym"].nunique(), added=len(set(b["ysym"]) - set(a["ysym"])))
            for r in gone.sort_values(["etf", "ysym"]).itertuples(index=False):
                rows.append({"year": y, "etf": r.etf, "ysym": r.ysym, "name": r.name, "pre_review_day": pre,
                             "R": R, "pre_price_fund_mv": r.market_value})
        summ.append(s)
    cols = ["year", "etf", "ysym", "name", "pre_review_day", "R", "pre_price_fund_mv"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(summ)
