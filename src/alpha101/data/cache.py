"""Parquet cache of `yahoo_ohlcv` and the `load_panel` entry point.

Cached per market under `cache_dir` (default data/cache, git-ignored):
  yahoo_<market>.parquet         long frame (yahoo.COLUMNS) of the market's EQUITY symbols + calendar index
  yahoo_<market>_master.parquet  the market's `security_master` rows (sector, industry, shares, ...)
  yahoo_<market>.json            sidecar: DB fingerprint and the `start` the parquet was read with

Fingerprint rule: the cache is valid only if the sidecar fingerprint equals the live
`db_fingerprint` (row count / max fetched_at / symbol count of the market's equities, the same for
the calendar index and for security_master). The cache is a regenerable copy, never a source of truth.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import reader
from ..panel import Panel
from . import yahoo
from .yahoo import read_long

log = logging.getLogger(__name__)

CALENDAR_SYMBOL = {"us": "^GSPC", "ca": "^GSPTSE"}
WARMUP_ROWS = 400


def _market(market: str) -> str:
    m = str(market).lower()
    if m not in CALENDAR_SYMBOL:
        raise ValueError(f"market must be 'us' or 'ca', got {market!r}")
    return m.upper()


def _iso(v):
    return None if v is None else pd.Timestamp(v).isoformat()


def db_fingerprint(conn, market: str) -> dict:
    """Cheap change detector for one market (three aggregate SELECTs, pure %s parameters)."""
    mk = _market(market)
    cal = CALENDAR_SYMBOL[mk.lower()]
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MAX(y.fetched_at), COUNT(DISTINCT y.symbol) FROM yahoo_ohlcv y "
                    "JOIN security_master m ON m.symbol = y.symbol "
                    "WHERE m.market = %s AND m.quote_type = 'EQUITY'", [mk])
        rows, mx, nsym = cur.fetchone()
        cur.execute("SELECT COUNT(*), MAX(fetched_at) FROM yahoo_ohlcv WHERE symbol = %s", [cal])
        crows, cmx = cur.fetchone()
        cur.execute("SELECT COUNT(*), MAX(fetched_at) FROM security_master WHERE market = %s", [mk])
        mrows, mmx = cur.fetchone()
    return {"rows": int(rows), "max_fetched_at": _iso(mx), "symbols": int(nsym),
            "calendar_rows": int(crows), "calendar_max_fetched_at": _iso(cmx),
            "master_rows": int(mrows), "master_max_fetched_at": _iso(mmx)}


def _read_master(conn, market: str) -> pd.DataFrame:
    cols = ["symbol", "quote_type", "sector", "industry", "shares_outstanding"]
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, quote_type, sector, industry, shares_outstanding "
                    "FROM security_master WHERE market = %s", [_market(market)])
        rows = cur.fetchall()
    df = pd.DataFrame(list(rows), columns=cols)
    df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce").astype(float)
    return df


def _paths(market: str, cache_dir: Path):
    m = market.lower()
    return (cache_dir / f"yahoo_{m}.parquet", cache_dir / f"yahoo_{m}_master.parquet",
            cache_dir / f"yahoo_{m}.json")


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")  # unique per writer
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_long(market: str, start: str = "2004-01-01", refresh: bool = False,
              cache_dir: Path = Path("data/cache")) -> pd.DataFrame:
    """Long frame of the market's EQUITY symbols plus its calendar index, date >= start."""
    mk = _market(market)
    cache_dir = Path(cache_dir)
    pq, mq, sc = _paths(mk, cache_dir)
    with reader() as conn:
        fp = db_fingerprint(conn, mk)  # taken BEFORE the rows: a concurrent ingest re-reads next time
        if not refresh and sc.exists() and pq.exists() and mq.exists():
            side = json.loads(sc.read_text())
            if side.get("fingerprint") == fp and pd.Timestamp(side["start"]) <= pd.Timestamp(start):
                df = pd.read_parquet(pq)
                return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        master = _read_master(conn, mk)
        symbols = sorted(set(master.loc[master["quote_type"] == "EQUITY", "symbol"])
                         | {CALENDAR_SYMBOL[mk.lower()]})
        df = read_long(conn, symbols=symbols, start=start)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sc.unlink(missing_ok=True)  # sidecar goes first out, last in: a crash mid-publish leaves a MISS, never a mismatched hit
    _atomic(pq, lambda p: df.to_parquet(p, index=False))
    _atomic(mq, lambda p: master.to_parquet(p, index=False))
    _atomic(sc, lambda p: p.write_text(json.dumps({"fingerprint": fp, "start": str(start)})))  # LAST
    return df


def load_master(market: str, cache_dir: Path = Path("data/cache")) -> pd.DataFrame:
    """security_master rows cached by the last `load_long` (FileNotFoundError if absent)."""
    mq = _paths(_market(market), Path(cache_dir))[1]
    if not mq.exists():
        raise FileNotFoundError(f"{mq} missing; call load_long first")
    return pd.read_parquet(mq)


def _load_start(start: str) -> str:
    return (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")


def load_panel(market: str, start: str = "2005-01-01", basis: str = "raw",
               universe_source: str | None = "liquidity", with_groups: bool = True,
               vwap_proxy: str = "typical", refresh: bool = False,
               cache_dir: Path = Path("data/cache")) -> Panel:
    """Equities-only Panel on the market's index calendar, with WARMUP_ROWS rows before `start`.

    universe_source: None = every equity on every row >= start; "liquidity" = the market's default
    liquidity source (`universe.liquidity_source`); any other string is a `universe_membership.source`.
    The source name is resolved with `universe.resolve_source` (a bare name picks its latest `name@version`,
    else the legacy bare name; an explicit `name@version` is used as given) and the resolved name is logged.
    A source's mask is loaded with `universe.load_universe` and cut to rows >= start; a source without
    members raises ValueError (run `universe.build_universe` first).
    """
    mk = _market(market)
    if basis not in ("raw", "adjusted"):
        raise ValueError("basis must be 'raw' or 'adjusted'")
    long = load_long(mk, start=_load_start(start), refresh=refresh, cache_dir=cache_dir)
    master = load_master(mk, cache_dir)
    cal_sym = CALENDAR_SYMBOL[mk.lower()]
    calendar = pd.DatetimeIndex(sorted(long.loc[long["symbol"] == cal_sym, "date"].unique()))
    i0 = int(calendar.searchsorted(pd.Timestamp(start)))
    if i0 < WARMUP_ROWS:
        log.warning("only %d calendar rows before %s (wanted %d warm-up rows)", i0, start, WARMUP_ROWS)
    dates = calendar[max(0, i0 - WARMUP_ROWS):]

    equities = set(master.loc[master["quote_type"] == "EQUITY", "symbol"])
    eq = long[long["symbol"].isin(equities) & ~long["symbol"].str.startswith("^")
              & (long["symbol"] != cal_sym) & (long["date"] >= dates[0])]
    sp = eq[eq["split"] > 0]
    bad = sp[~sp["date"].isin(dates) | ~(sp["close"] > 0)]  # NaN close fails `> 0`
    if len(bad):
        pairs = [(r.symbol, r.date.strftime("%Y-%m-%d")) for r in bad.head(5).itertuples()]
        raise ValueError(f"{len(bad)} split rows would be dropped by panel_from_long "
                         f"(off-calendar or non-positive close), e.g. {pairs}")

    p = yahoo.panel_from_long(eq, basis="adjusted", calendar=dates, vwap_proxy=vwap_proxy)
    m = master.drop_duplicates("symbol").set_index("symbol")
    shares = m["shares_outstanding"].reindex(p.symbols).to_numpy(dtype=float)
    fields = dict(p.fields)
    fields["shares_outstanding"] = np.tile(shares, (len(p.dates), 1))
    groups = {}
    if with_groups:
        groups = {lv: Panel.encode_groups(m[lv], p.symbols) for lv in ("sector", "industry")}
    after_start = (p.dates >= pd.Timestamp(start))[:, None]
    if universe_source is None:
        universe = np.tile(after_start, (1, len(p.symbols)))
    else:
        from . import universe as uni_mod  # lazy: universe imports cache
        source = uni_mod.liquidity_source(mk) if universe_source == "liquidity" else universe_source
        with reader() as conn:
            source = uni_mod.resolve_source(conn, source)
            log.info("universe source resolved to %r", source)
            loaded = uni_mod.load_universe(conn, p, source)
        if not loaded.any():
            raise ValueError(f"no members for source {source!r}; run build_universe first")
        universe = loaded & after_start
    p = Panel(dates=p.dates, symbols=p.symbols, fields=fields, universe=universe, groups=groups,
              vwap_proxy=vwap_proxy)
    return p.to_raw_basis() if basis == "raw" else p
