"""Yahoo ingestion into `alpha101.yahoo_ohlcv` and `alpha101.security_master`.

History is always fetched with period="max": Yahoo re-adjusts the whole history after every split,
so one symbol must stay on one share basis (design.md section 4.2). A symbol that already has rows
is normally topped up from its stored max(date) onward; `needs_full_refetch` detects a re-adjustment
and then the whole history is upserted. All SQL uses pure `%s` placeholders in VALUES, writes are
upserts (no DROP/TRUNCATE), and connections come from `alpha101.db`. The only DELETE is
`delete_stale_dates`: after a FULL refetch of one symbol it removes that symbol's stored dates that
Yahoo no longer returns (they would keep the old share basis).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import re
import time
from pathlib import Path

import pandas as pd

from .constituents import candidate_pool, check_identity, valid_symbol

PAUSE = 1.0                  # seconds between symbols
CLOSE_TOLERANCE = 0.005      # re-adjustment threshold on the overlap-day close
RATIO_TOLERANCE = 1e-4       # relative change of the overlap-day adj_close/close ratio (dividend re-adjustment)
MAX_STALE_FRACTION = 0.2     # refuse to delete more than this share of a symbol's stored dates
CHUNK = 2000

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]

# NOTE: every VALUES element is a %s placeholder, otherwise pymysql cannot rewrite executemany as one
# multi-row INSERT and falls back to one round trip per row.
OHLCV_UPSERT_SQL = (
    "INSERT INTO yahoo_ohlcv (symbol,date,open,high,low,close,adj_close,volume,dividend,split,source,fetched_at)"
    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    " ON DUPLICATE KEY UPDATE open=VALUES(open),high=VALUES(high),low=VALUES(low),close=VALUES(close),"
    " adj_close=VALUES(adj_close),volume=VALUES(volume),dividend=VALUES(dividend),split=VALUES(split),"
    " fetched_at=VALUES(fetched_at)"
)

MASTER_UPSERT_SQL = (
    "INSERT INTO security_master (symbol,name,market,exchange,quote_type,sector,industry,"
    "market_cap_snapshot,shares_outstanding,currency,source,fetched_at)"
    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    " ON DUPLICATE KEY UPDATE name=VALUES(name),market=VALUES(market),exchange=VALUES(exchange),"
    "quote_type=VALUES(quote_type),sector=VALUES(sector),industry=VALUES(industry),"
    "market_cap_snapshot=VALUES(market_cap_snapshot),shares_outstanding=VALUES(shares_outstanding),"
    "currency=VALUES(currency),source=VALUES(source),fetched_at=VALUES(fetched_at)"
)

# design.md section 8, verbatim.
SECURITY_MASTER_DDL = """CREATE TABLE IF NOT EXISTS security_master (
  symbol              VARCHAR(20)  NOT NULL,           -- Yahoo symbol (BRK-B, RCI-B.TO)
  name                VARCHAR(200) NULL,
  market              CHAR(2)      NOT NULL,           -- US | CA
  exchange            VARCHAR(20)  NULL,
  quote_type          VARCHAR(20)  NULL,               -- EQUITY | ETF | INDEX
  sector              VARCHAR(80)  NULL,
  industry            VARCHAR(120) NULL,
  market_cap_snapshot DOUBLE       NULL,               -- yfinance marketCap on fetched_at (NOT PIT)
  shares_outstanding  DOUBLE       NULL,               -- yfinance sharesOutstanding on fetched_at
  currency            CHAR(3)      NULL,
  source              VARCHAR(30)  NOT NULL DEFAULT 'yfinance',
  fetched_at          DATETIME     NOT NULL,
  PRIMARY KEY (symbol),
  KEY idx_market (market)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

log = logging.getLogger("alpha101.ingest")


def _clean(x):
    """NaN/None -> None, numpy scalars -> Python floats."""
    if x is None:
        return None
    try:
        if math.isnan(x):
            return None
    except TypeError:
        return x
    return float(x)


def _is_rate_limit(exc) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in s or "too many requests" in s or re.search(r"\b429\b", s) is not None


class RateLimited(RuntimeError):
    """Yahoo rate-limited every attempt for a symbol; the symbol says nothing about delisting."""


def fetch_history(symbol: str, retries: int = 3, pause: float = 1.0) -> pd.DataFrame | None:
    """Full Yahoo history, columns: date, open, high, low, close, adj_close, volume, dividend, split.

    Returns None after `retries` failed or empty attempts (exponential backoff; rate limits wait
    much longer). Delisted names normally come back empty. Raises ValueError for a symbol that is
    not a clean Yahoo symbol, and RateLimited if the last attempt was rate-limited.
    """
    import yfinance as yf
    from yfinance import shared
    if not valid_symbol(symbol):
        raise ValueError(f"refusing to fetch invalid symbol {symbol!r}")
    limited = False
    for attempt in range(retries):
        getattr(shared, "_ERRORS", {}).pop(symbol.upper(), None)
        df, err = None, None
        try:
            df = yf.download(symbol, period="max", auto_adjust=False, actions=True,
                             progress=False, threads=False)
        except Exception as e:  # noqa: BLE001 - network/Yahoo errors are data, not bugs
            err = e
        # yf.download swallows per-ticker exceptions and records them here instead of raising
        err = err or getattr(shared, "_ERRORS", {}).get(symbol.upper())
        if df is not None and not df.empty:
            break
        limited = err is not None and _is_rate_limit(err)
        if err is not None:
            log.warning("%s: download error %s (attempt %d/%d)", symbol,
                        type(err).__name__ if isinstance(err, Exception) else str(err)[:80], attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(pause * (30 if limited else 1) * 2 ** attempt)
    else:
        if limited:
            raise RateLimited(symbol)
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "adj_close", "Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume", "Dividends": "dividend",
                            "Stock Splits": "split"})
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        log.warning("%s: Yahoo frame lacks columns %s (filled with %s)", symbol, missing,
                    "0/NaN")
    for c in missing:
        df[c] = 0.0 if c in ("dividend", "split") else float("nan")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = df[OHLCV_COLUMNS].copy()
    out.insert(0, "date", idx.normalize())
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def needs_full_refetch(conn, symbol: str, new: pd.DataFrame) -> bool:
    """True if Yahoo re-adjusted the history, i.e. one of
      - `new` holds a split newer than the stored max(date);
      - `new` holds a dividend newer than the stored max(date) (adj_close of all earlier days is rescaled);
      - `new` holds a split or dividend dated exactly on the stored max(date) while the stored row of that
        date carries 0/NULL for it (the stored row predates the event; no refetch if it already has it);
      - the stored close on the latest overlapping date differs from `new` by more than 0.5%;
      - the overlap-day adj_close/close ratio differs from the stored ratio by more than 1e-4 (relative).
    The overlap day is the latest date that is a usable new row (close > 0) and a stored row with
    close > 0, looking back at most 10 stored rows; the ratio test uses the latest such day whose stored
    adj_close is also non-NULL. A symbol with no stored rows has nothing to be inconsistent with: False.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM yahoo_ohlcv WHERE symbol=%s", (symbol,))
        stored_max = cur.fetchone()[0]
        if stored_max is None:
            return False
        stored_max = pd.Timestamp(stored_max)
        dates = pd.to_datetime(new["date"])
        after = dates > stored_max
        if bool(((new["split"].fillna(0.0) > 0) & after).any()):
            return True
        if bool(((new["dividend"].fillna(0.0) > 0) & after).any()):
            return True
        cur.execute("SELECT date, close, adj_close, dividend, split FROM yahoo_ohlcv WHERE symbol=%s"
                    " ORDER BY date DESC LIMIT 10", (symbol,))
        stored = list(cur.fetchall())
    on_max = dates == stored_max
    max_row = next((r for r in stored if pd.Timestamp(r[0]) == stored_max), None)
    if max_row is not None:
        for col, k in (("split", 4), ("dividend", 3)):
            if bool(((new[col].fillna(0.0) > 0) & on_max).any()) and (max_row[k] is None or float(max_row[k]) == 0.0):
                return True
    usable = (dates <= stored_max) & new["close"].notna() & (new["close"] > 0)
    if not usable.any():
        return False
    by_date = {}
    for d, c, a in zip(dates[usable], new.loc[usable, "close"], new.loc[usable, "adj_close"]):
        by_date[pd.Timestamp(d).date()] = (float(c), a)      # a duplicate date: the last row wins
    overlap = []                                             # newest first
    for r in stored:
        if r[1] is None or not (float(r[1]) > 0):
            continue
        hit = by_date.get(pd.Timestamp(r[0]).date())
        if hit is not None:
            overlap.append((float(r[1]), r[2], hit[0], hit[1]))
    if not overlap:
        return False
    old, _, new_close, _ = overlap[0]
    if abs(new_close / old - 1.0) > CLOSE_TOLERANCE:
        return True
    for old, old_adj, new_close, new_adj in overlap:
        if old_adj is None or pd.isna(new_adj) or not (float(old_adj) > 0):
            continue
        old_ratio, new_ratio = float(old_adj) / old, float(new_adj) / new_close
        return abs(new_ratio / old_ratio - 1.0) > RATIO_TOLERANCE
    return False


def delete_stale_dates(conn, symbol: str, new_dates) -> int:
    """After a FULL refetch: delete this symbol's stored dates that Yahoo no longer returns (they keep
    the old share basis). The only DELETE in this module; scoped to one symbol, chunked, and skipped
    (with a warning) if it would remove more than MAX_STALE_FRACTION of the stored dates. Only stored
    dates not newer than the last date in `new_dates` are candidates: rows newer than Yahoo's last bar
    are kept (and logged). `new_dates` should be the dates actually upserted."""
    keep = {pd.Timestamp(d).date() for d in new_dates}
    if not keep:
        return 0
    last = max(keep)
    with conn.cursor() as cur:
        cur.execute("SELECT date FROM yahoo_ohlcv WHERE symbol=%s", (symbol,))
        stored = [pd.Timestamp(r[0]).date() for r in cur.fetchall()]
        newer = [d for d in stored if d > last]
        if newer:
            log.warning("%s: %d stored dates are newer than Yahoo's last bar %s; keeping them",
                        symbol, len(newer), last)
        stale = sorted(d for d in stored if d <= last and d not in keep)
        if not stale:
            return 0
        if len(stale) > MAX_STALE_FRACTION * len(stored):
            log.warning("%s: %d of %d stored dates are absent from the refetch; NOT deleting",
                        symbol, len(stale), len(stored))
            return 0
        for i in range(0, len(stale), CHUNK):
            part = stale[i:i + CHUNK]
            cur.execute("DELETE FROM yahoo_ohlcv WHERE symbol=%s AND date IN (" + ",".join(["%s"] * len(part)) + ")",
                        (symbol, *part))
            conn.commit()
    log.info("%s: deleted %d stored dates absent from the full refetch", symbol, len(stale))
    return len(stale)


def upsert_ohlcv(conn, symbol: str, df: pd.DataFrame, fetched_at: dt.datetime) -> int:
    """Upsert `df` (fetch_history layout) for one symbol, chunked and committed per chunk."""
    rows = []
    for r in df.itertuples(index=False):
        vol = _clean(r.volume)
        rows.append((symbol, pd.Timestamp(r.date).date(), _clean(r.open), _clean(r.high), _clean(r.low),
                     _clean(r.close), _clean(r.adj_close), None if vol is None else int(vol),
                     _clean(r.dividend), _clean(r.split), "yfinance", fetched_at))
    with conn.cursor() as cur:
        for i in range(0, len(rows), CHUNK):
            cur.executemany(OHLCV_UPSERT_SQL, rows[i:i + CHUNK])
            conn.commit()
    return len(rows)


def fetch_security_master(symbol: str, retries: int = 3, pause: float = 1.0) -> dict:
    """yf.Ticker(symbol).info -> name, exchange, quote_type, sector, industry, market_cap_snapshot,
    shares_outstanding, currency. Raises the last error if every attempt fails."""
    import yfinance as yf
    last: Exception | None = None
    for attempt in range(retries):
        try:
            info = yf.Ticker(symbol).info or {}
            if info:
                return {
                    "symbol": symbol,
                    "name": info.get("longName") or info.get("shortName"),
                    "exchange": info.get("exchange"),
                    "quote_type": info.get("quoteType"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap_snapshot": _clean(info.get("marketCap")),
                    "shares_outstanding": _clean(info.get("sharesOutstanding")),
                    "currency": info.get("currency"),
                }
            last = RuntimeError("empty info")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(pause * (30 if last is not None and _is_rate_limit(last) else 1) * 2 ** attempt)
    raise last  # type: ignore[misc]


def upsert_security_master(conn, rows: list[dict], fetched_at: dt.datetime) -> int:
    data = [(r["symbol"], (r.get("name") or None) and str(r["name"])[:200], r["market"],
             r.get("exchange"), r.get("quote_type"), r.get("sector"), r.get("industry"),
             _clean(r.get("market_cap_snapshot")), _clean(r.get("shares_outstanding")),
             (r.get("currency") or None) and str(r["currency"])[:3], "yfinance", fetched_at)
            for r in rows]
    with conn.cursor() as cur:
        for i in range(0, len(data), CHUNK):
            cur.executemany(MASTER_UPSERT_SQL, data[i:i + CHUNK])
            conn.commit()
    return len(data)


def _setup_logging(market: str) -> Path:
    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / f"ingest_{market}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.log"
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return path


def _drop_null_overlap(part: pd.DataFrame, stored_max) -> pd.DataFrame:
    """Never let a NaN-close bar on an already stored date overwrite the stored close with NULL."""
    if stored_max is None:
        return part
    bad = part["close"].isna() & (part["date"] <= pd.Timestamp(stored_max))
    return part[~bad]


def ingest(market: str, limit: int | None = None, since: str = "2005-01-01",
           only: list[str] | None = None) -> dict:
    """Ingest one market ("us" or "ca"). `only` restricts the run to those pool symbols. Returns
    {"attempted": n, "with_data": k, "failed": [...], "rows": m, "refetched": [...]} plus
      "no_data"        symbols Yahoo returned nothing for ("failed" is the same list: no data only),
      "write_failed"   symbols whose DB write raised,
      "rate_limited"   symbols Yahoo rate-limited on every attempt (not a delisting signal),
      "reused_ticker"  change-only symbols that failed the identity check (not ingested; reasons in
                       "reused_reason"),
      "identity_unverified"  change-only symbols whose Yahoo name could not be fetched: nothing is
                       written (no OHLCV, no master row) and the next run retries them,
      "stale_deleted"  {symbol: n} stored dates removed after a full refetch,
      "master_failed"  with_data symbols whose security_master info could not be fetched (an annotation,
                       not a partition member), "master_rows".

    Invariant: every attempted symbol is in exactly one of with_data (a count; its rows are in "rows"),
    no_data, rate_limited, reused_ticker, identity_unverified, write_failed, i.e.
    attempted == with_data + len(no_data) + len(rate_limited) + len(reused_ticker)
                 + len(identity_unverified) + len(write_failed).
    with_data and rows are counted only after the whole per-symbol block (OHLCV upsert, stale delete,
    master step) succeeded."""
    from ..db import writer
    market = market.lower()
    if market not in ("us", "ca"):
        raise ValueError("market must be 'us' or 'ca'")
    logfile = _setup_logging(market)
    stamp = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    pool = candidate_pool(Path("data/raw") / f"html_{stamp}", since=since)
    pool = pool[pool["market"] == market.upper()]
    if only is not None:
        pool = pool[pool["symbol"].isin(only)]
    symbols = list(pool["symbol"])[:limit] if limit else list(pool["symbol"])
    meta = pool.set_index("symbol")
    log.info("market=%s pool=%d attempting=%d log=%s", market, len(pool), len(symbols), logfile)

    conn = writer()
    with conn.cursor() as cur:
        cur.execute(SECURITY_MASTER_DDL)
    conn.commit()

    def now():
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)

    res = {"attempted": len(symbols), "with_data": 0, "failed": [], "no_data": [], "write_failed": [],
           "rate_limited": [], "reused_ticker": [], "identity_unverified": [], "reused_reason": {}, "rows": 0, "refetched": [],
           "stale_deleted": {}, "master_failed": [], "master_rows": 0}

    def get_info(sym):
        try:
            info = fetch_security_master(sym, pause=PAUSE)
            info["market"] = market.upper()
            return info
        except Exception as e:  # noqa: BLE001
            log.warning("%s: security_master info failed (%s)", sym, type(e).__name__)
            return None

    try:
        for i, sym in enumerate(symbols, 1):
            fetched_at = now()
            try:
                df = fetch_history(sym, pause=PAUSE)
            except RateLimited:
                res["rate_limited"].append(sym)
                log.warning("[%d/%d] %s: rate limited on every attempt", i, len(symbols), sym)
                time.sleep(PAUSE)
                continue
            except Exception as e:  # noqa: BLE001 - one malformed frame must not stop the market
                df = None
                log.warning("[%d/%d] %s: fetch failed (%s)", i, len(symbols), sym, type(e).__name__)
            if df is None:
                res["no_data"].append(sym)
                res["failed"].append(sym)
                log.warning("[%d/%d] %s: no data", i, len(symbols), sym)
                time.sleep(PAUSE)
                continue
            try:
                info = None
                if meta.loc[sym, "source"] == "sp500_changes":
                    # Yahoo maps a removed ticker to whoever holds it today: check before writing
                    info = get_info(sym)
                    if info is None or not info.get("name"):
                        res["identity_unverified"].append(sym)
                        log.warning("[%d/%d] %s: identity unverifiable (no Yahoo name), not ingested",
                                    i, len(symbols), sym)
                        time.sleep(PAUSE)
                        continue
                    reason = check_identity(df["date"].min(), info.get("name"),
                                            meta.loc[sym, "change_date"], meta.loc[sym, "wiki_names"], symbol=sym)
                    if reason:
                        res["reused_ticker"].append(sym)
                        res["reused_reason"][sym] = reason
                        log.warning("[%d/%d] %s: reused ticker, not ingested (%s)", i, len(symbols), sym, reason)
                        time.sleep(PAUSE)
                        continue
                conn.ping(reconnect=True)  # a market takes tens of minutes; the link may drop
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(date) FROM yahoo_ohlcv WHERE symbol=%s", (sym,))
                    stored_max = cur.fetchone()[0]
                if stored_max is None:
                    part = df
                elif needs_full_refetch(conn, sym, df):
                    part = _drop_null_overlap(df, stored_max)
                    res["refetched"].append(sym)
                    log.info("%s: re-adjusted, new split or new dividend, upserting full history", sym)
                else:
                    part = _drop_null_overlap(df[df["date"] >= pd.Timestamp(stored_max)], stored_max)
                n = upsert_ohlcv(conn, sym, part, fetched_at)
                if sym in res["refetched"]:
                    k = delete_stale_dates(conn, sym, part["date"])
                    if k:
                        res["stale_deleted"][sym] = k
                if info is None:
                    info = get_info(sym)
                master_rows = 0
                if info is not None:
                    master_rows = upsert_security_master(conn, [info], fetched_at)
                log.info("[%d/%d] %s: %d rows (history %s..%s)", i, len(symbols), sym, n,
                         df["date"].min().date(), df["date"].max().date())
                # the whole block succeeded: only now does the symbol count as with_data
                res["rows"] += n
                res["with_data"] += 1
                res["master_rows"] += master_rows
                if info is None:
                    res["master_failed"].append(sym)
            except Exception as e:  # noqa: BLE001 - one symbol must not stop the market
                # DB errors can echo host/user; log only the class name.
                res["write_failed"].append(sym)
                log.error("[%d/%d] %s: write failed (%s)", i, len(symbols), sym, type(e).__name__)
            time.sleep(PAUSE)
        # one more pass for symbols whose info call failed (rate limits are often transient)
        if res["master_failed"]:
            time.sleep(30 * PAUSE)
            for sym in list(res["master_failed"]):
                try:
                    info = fetch_security_master(sym, retries=4, pause=PAUSE)
                    info["market"] = market.upper()
                    res["master_rows"] += upsert_security_master(conn, [info], now())
                    res["master_failed"].remove(sym)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: security_master retry failed (%s)", sym, type(e).__name__)
                time.sleep(PAUSE)
    finally:
        conn.close()
    log.info("done: %s", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in res.items()})
    return res
