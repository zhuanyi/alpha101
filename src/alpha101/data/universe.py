"""Universe builder: liquidity masks and index-membership references -> `universe_membership`.

Liquidity universe (docs/design.md section 5): on the first trading day of each month, candidates with
enough history and a minimum as-traded close are ranked by trailing median dollar volume measured at the
last trading day of the PREVIOUS month; the top N are members for the whole month. Only rows up to that
decision day are read, so the mask adds no look-ahead given the candidate pool.

Optional spread screen (amend-1): at the decision day a name is also required to have an Abdi-Ranaldo (2017)
spread estimate (`ar_spread`, trailing window) at or below a per-market bar; the screen precedes the top-N
ranking, so a screened name is replaced by the next-ranked one. Stored source names carry a version
(`name@YYYYMMDD`, `versioned`); readers resolve a bare name to its latest version (`resolve_source`).

Membership is stored as state-change events (symbol, source, as_of, in_universe): the state holds from
`as_of` until the next event of the same (symbol, source).
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import reader, writer
from ..panel import Panel
from . import constituents
from .constituents import candidate_pool, check_identity, yahoo_symbol

log = logging.getLogger("alpha101.universe")

# spread bar: recommended thresholds, not yet validated
DEFAULTS = {"us": {"top_n": 500, "min_price": 5.0, "max_spread": 0.02, "spread_window": 63},
            "ca": {"top_n": 200, "min_price": 2.0, "max_spread": 0.03, "spread_window": 63}}
VERSION_RE = r"^\d{8}[a-z]?$"
SOURCE_MAX = 40   # universe_membership.source is VARCHAR(40)
MIN_ROWS = 20   # design.md section 5 "Minimum history"
CHUNK = 5000
EVENT_COLUMNS = ["symbol", "market", "source", "as_of", "in_universe"]

# design.md section 8, verbatim.
UNIVERSE_MEMBERSHIP_DDL = """CREATE TABLE IF NOT EXISTS universe_membership (
  symbol      VARCHAR(20) NOT NULL,
  market      CHAR(2)     NOT NULL,
  source      VARCHAR(40) NOT NULL,   -- sp500_wiki_current | sp500_wiki_changes | tsx_wiki_current
                                      -- | liquidity_top500_us | liquidity_top200_ca
  as_of       DATE        NOT NULL,   -- state holds from as_of until the next row for (symbol, source)
  in_universe TINYINT(1)  NOT NULL,
  created_at  TIMESTAMP   NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, source, as_of),
  KEY idx_source_date (source, as_of)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

# NOTE: every VALUES element is a %s placeholder (see ingest.py).
UPSERT_SQL = ("INSERT INTO universe_membership (symbol,market,source,as_of,in_universe) VALUES (%s,%s,%s,%s,%s)"
              " ON DUPLICATE KEY UPDATE market=VALUES(market),in_universe=VALUES(in_universe)")


def liquidity_source(market: str, top_n: int | None = None) -> str:
    """"liquidity_top500_us", "liquidity_top200_ca" (top_n defaults per market)."""
    m = str(market).lower()
    if m not in DEFAULTS:
        raise ValueError(f"market must be 'us' or 'ca', got {market!r}")
    n = DEFAULTS[m]["top_n"] if top_n is None else int(top_n)
    return f"liquidity_top{n}_{m}"


def versioned(source: str, version: str) -> str:
    """`source@version` (version = YYYYMMDD plus an optional letter); ValueError on a malformed name."""
    if not re.match(VERSION_RE, str(version)):
        raise ValueError(f"version must match {VERSION_RE}, got {version!r}")
    if "@" in source:
        raise ValueError(f"source {source!r} already carries a version")
    out = f"{source}@{version}"
    if len(out) > SOURCE_MAX:
        raise ValueError(f"source name {out!r} is longer than {SOURCE_MAX} characters")
    return out


def resolve_source(conn, source: str) -> str:
    """The stored source a reader should use. Explicit `name@version`: itself if it has rows, else ValueError.
    Bare name: the latest stored `name@version` (string max of the version), else the bare legacy name if
    it has rows, else ValueError. No other implicit choice is ever made."""
    with conn.cursor() as cur:
        if "@" in source:
            cur.execute("SELECT 1 FROM universe_membership WHERE source=%s LIMIT 1", [source])
            if cur.fetchall():
                return source
            raise ValueError(f"no stored source {source!r}")
        cur.execute("SELECT DISTINCT source FROM universe_membership WHERE source LIKE %s", [source + "@%"])
        found = [r[0] for r in cur.fetchall()]   # LIKE is a superset ("_" is a wildcard): filter below
        vers = [x[len(source) + 1:] for x in found if x.startswith(source + "@")]
        vers = [v for v in vers if re.match(VERSION_RE, v)]
        if vers:
            return versioned(source, max(vers))
        cur.execute("SELECT 1 FROM universe_membership WHERE source=%s LIMIT 1", [source])
        if cur.fetchall():
            return source
    raise ValueError(f"no stored source {source!r} (bare or versioned)")


def ar_spread(high, low, close, window: int, split_age=None, min_frac: float = 0.8,
              volume=None) -> np.ndarray:
    """Abdi-Ranaldo (2017) close-high-low bid-ask spread estimate, (T, N) fraction (0.01 = 1%).

    Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices",
    Review of Financial Studies 30(12), 4437-4480: S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})], c = log close,
    eta = (log high + log low) / 2. Inputs are as-traded prices (non-positive / non-finite -> NaN). The pair
    dated t uses days t-1 and t, p[t] = (c[t-1] - eta[t-1]) * (c[t-1] - eta[t]); p[0] is NaN, and p[t] is NaN
    where split_age[t] == 0 (the pair straddles a split on the raw basis). S[t] = sqrt(max(4 * mean of the
    finite p over rows t-window+1..t, 0)), NaN with fewer than ceil(min_frac * window) finite pairs.
    If `volume` is given, p[t] is also NaN where volume[t-1] or volume[t] is not finite or <= 0: a no-trade
    day carries no close/high/low information (a stale O=H=L=C row would add a 0 product and bias the
    estimate toward "liquid"). S[t] uses prices up to row t only.
    """
    def lg(a):
        a = np.asarray(a, dtype=float)
        with np.errstate(all="ignore"):
            return np.where(np.isfinite(a) & (a > 0), np.log(a), np.nan)
    c, eta = lg(close), (lg(high) + lg(low)) / 2.0
    pair = np.full(c.shape, np.nan)
    if c.shape[0] > 1:
        pair[1:] = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    if split_age is not None:
        pair[np.asarray(split_age) == 0] = np.nan
    if volume is not None:
        v = np.asarray(volume, dtype=float)
        traded = np.isfinite(v) & (v > 0)
        pair[~traded] = np.nan
        pair[1:][~traded[:-1]] = np.nan
    need = max(1, int(np.ceil(min_frac * window - 1e-9)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = pd.DataFrame(pair).rolling(window, min_periods=need).mean().to_numpy()
        return np.sqrt(np.maximum(4.0 * m, 0.0))


def liquidity_universe(panel: Panel, top_n: int, min_price: float, min_history: int = 252,
                       lookback: int = 63, eligible: np.ndarray | None = None,
                       max_spread: float | None = None, spread_window: int = 63,
                       top_n_by_month: dict | None = None) -> np.ndarray:
    """Bool (T, N) liquidity mask on a raw-basis panel.

    For each month start row i the decision day is e = i - 1 (last trading day of the previous month; the
    first panel month has no members). Symbol j is eligible if `eligible[j]`, it has >= min_history finite
    closes in rows 0..e, close[e, j] is finite and >= min_price, and its score is finite. Score = median of
    close*volume over rows e-lookback+1..e (NaN volume ignored). Top `top_n` by score (ties: symbol
    ascending) are members on every row of the month.

    max_spread: if not None, `ar_spread` (window `spread_window`, split pairs excluded via `split_age`) must
    be finite and <= max_spread at e (NaN fails); applied before the ranking, so the next name replaces a
    screened one. top_n_by_month ({pd.Period("YYYY-MM", "M"): K}): the month starting at row i keeps
    min(top_n, K[month]) names (0 if the month is absent) instead of top_n.
    """
    if panel.basis != "raw":
        raise ValueError("liquidity_universe needs a raw-basis panel (as-traded prices)")
    dates = panel.dates
    T, N = len(dates), len(panel.symbols)
    close = panel.get("close")
    volume = panel.get("volume")
    dv = close * volume
    fin_count = np.cumsum(np.isfinite(close), axis=0)
    syms = np.asarray(panel.symbols, dtype=str)
    elig = np.ones(N, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if elig.shape != (N,):
        raise ValueError("eligible must have shape (N,)")
    spread = None
    if max_spread is not None:
        spread = ar_spread(panel.get("high"), panel.get("low"), close, spread_window,
                           split_age=panel.fields.get("split_age"), volume=volume)
    months = dates.to_period("M")
    starts = [i for i in range(1, T) if months[i] != months[i - 1]]
    mask = np.zeros((T, N), dtype=bool)
    for k, i in enumerate(starts):
        e = i - 1
        end = starts[k + 1] if k + 1 < len(starts) else T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            score = np.nanmedian(dv[max(0, e - lookback + 1):e + 1], axis=0)
        ok = (elig & (fin_count[e] >= min_history) & np.isfinite(close[e])
              & (close[e] >= min_price) & np.isfinite(score))
        if spread is not None:
            with np.errstate(invalid="ignore"):
                ok &= np.isfinite(spread[e]) & (spread[e] <= max_spread)
        n_keep = top_n if top_n_by_month is None else min(top_n, int(top_n_by_month.get(months[i], 0)))
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        order = np.lexsort((syms[idx], -score[idx]))   # primary: score descending, then symbol ascending
        mask[i:end, idx[order[:n_keep]]] = True
    return mask


def _compress(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep each symbol's first row and the rows whose state differs from the previous row."""
    rows = rows.sort_values(["symbol", "as_of"], kind="stable").reset_index(drop=True)
    if rows.empty:
        return rows
    same_sym = rows["symbol"].eq(rows["symbol"].shift())
    changed = rows["in_universe"].ne(rows["in_universe"].shift())
    return rows[~same_sym | changed].reset_index(drop=True)


def membership_events(mask: np.ndarray, dates, symbols, market: str, source: str) -> pd.DataFrame:
    """[symbol, market, source, as_of, in_universe]: a first row per symbol at dates[0], then each change."""
    mask = np.asarray(mask, dtype=bool)
    dates = pd.DatetimeIndex(dates)
    T, N = mask.shape
    if T == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    pydates = [d.date() for d in dates]
    rows = []
    t_idx, j_idx = np.nonzero(mask[1:] != mask[:-1])
    changes: dict[int, list[int]] = {}
    for t, j in zip(t_idx, j_idx):
        changes.setdefault(int(j), []).append(int(t) + 1)
    for j in range(N):
        sym = str(symbols[j])
        rows.append((sym, market.upper(), source, pydates[0], int(mask[0, j])))
        for t in changes.get(j, ()):
            rows.append((sym, market.upper(), source, pydates[t], int(mask[t, j])))
    out = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return out.sort_values(["symbol", "as_of"], kind="stable").reset_index(drop=True)


def _map_tickers(raw, market: str) -> list[str]:
    out = []
    for t in raw:
        if t is None or (not isinstance(t, str) and pd.isna(t)):
            continue
        try:
            s = yahoo_symbol(t, market)
        except ValueError as e:
            log.warning("dropping index ticker: %s", e)
            continue
        if s:
            out.append(s)
    return out


def index_membership_events(current: pd.DataFrame, changes: pd.DataFrame, market: str, source: str,
                            since: str) -> pd.DataFrame:
    """Index membership events reconstructed by walking the Wikipedia changes table backwards.

    `current["symbol"]` are raw tickers of today's list; `changes` is [date, added, removed] (raw tickers).
    members = today's list; changes dated >= since are undone newest first (an `added` ticker leaves the
    set, a `removed` ticker joins it), which gives the state at `since`. Emits (sym, since, state) for every
    symbol in that set or touched by a change, plus the change events, compressed to state changes. If two
    rows share (symbol, as_of) the state after all changes of that date in forward order wins (within one
    date the table is assumed newest-first, so the first row processed is the last in forward order).
    Anomalies (an `added` ticker not in the set, a `removed` ticker already in it) are ticker-reuse or
    rename artefacts (e.g. AT&T Corp.'s `T` removed in 2005 while today's `T` is AT&T Inc.): they are
    counted and logged, but emit NO event and leave the set unchanged. Invariant: replaying the returned
    events forward ends at today's list (mapped to Yahoo symbols).

    With an empty `changes` this is today's list projected back to `since`: a survivorship-biased
    reference mask (`sp500_wiki_current`, `tsx_wiki_current`), NOT a point-in-time universe.
    """
    since_d = pd.Timestamp(since).date()
    members = set(_map_tickers(current["symbol"], market))
    events: list[tuple] = []   # (symbol, as_of, state, priority); lower priority value wins on ties
    anomalies = 0
    if changes is not None and len(changes):
        ch = changes[pd.to_datetime(changes["date"]) >= pd.Timestamp(since)].copy()
        ch["date"] = pd.to_datetime(ch["date"])
        ch = ch.sort_values("date", ascending=False, kind="stable")
        seq = 0
        for date, added, removed in zip(ch["date"], ch["added"], ch["removed"]):
            d = date.date()
            for a in _map_tickers([added], market):
                if a not in members:
                    anomalies += 1   # ticker reuse/rename artefact: no event, the set is unchanged
                    continue
                members.discard(a)
                events.append((a, d, 1, seq))
                seq += 1
            for r in _map_tickers([removed], market):
                if r in members:
                    anomalies += 1   # ticker reuse/rename artefact: no event, the set is unchanged
                    continue
                members.add(r)
                events.append((r, d, 0, seq))
                seq += 1
    if anomalies:
        log.warning("%s: %d walk-back anomalies (added not in set / removed already in set)", source, anomalies)
    touched = {e[0] for e in events}
    base = [(s, since_d, 1 if s in members else 0, 10**12) for s in sorted(members | touched)]
    allrows = pd.DataFrame(base + events, columns=["symbol", "as_of", "in_universe", "prio"])
    if allrows.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    allrows = allrows.sort_values(["symbol", "as_of", "prio"], kind="stable") \
        .drop_duplicates(["symbol", "as_of"], keep="first")
    allrows["market"] = market.upper()
    allrows["source"] = source
    allrows = _compress(allrows[EVENT_COLUMNS])
    allrows["in_universe"] = allrows["in_universe"].astype(int)
    return allrows


def _chunks(items: list, n: int = 500):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def eligible_symbols(conn, pool: pd.DataFrame, market: str, min_rows: int = MIN_ROWS) -> tuple[list[str], dict]:
    """Pool symbols of `market` that may enter a universe, and why the others do not.

    Exclusions in order: no_data (no yahoo_ohlcv rows), not_equity (security_master quote_type != EQUITY
    or no master row), reused_ticker (change-only S&P symbol failing `check_identity`; {symbol: reason}),
    short_history (< min_rows rows).
    """
    pool = pool[pool["market"] == market.upper()].drop_duplicates("symbol")
    syms = list(pool["symbol"])
    counts: dict[str, tuple[int, object]] = {}
    master: dict[str, tuple] = {}
    with conn.cursor() as cur:
        for part in _chunks(syms):
            ph = ",".join(["%s"] * len(part))
            cur.execute(f"SELECT symbol, COUNT(*), MIN(date) FROM yahoo_ohlcv WHERE symbol IN ({ph}) "
                        "GROUP BY symbol", part)
            for s, n, mn in cur.fetchall():
                counts[s] = (int(n), mn)
            cur.execute(f"SELECT symbol, name, quote_type FROM security_master WHERE symbol IN ({ph})", part)
            for s, name, qt in cur.fetchall():
                master[s] = (name, qt)
    meta = pool.set_index("symbol")
    info = {"no_data": [], "not_equity": [], "reused_ticker": {}, "short_history": []}
    elig = []
    for s in syms:
        n, first = counts.get(s, (0, None))
        if n == 0:
            info["no_data"].append(s)
            continue
        name, qt = master.get(s, (None, None))
        if qt != "EQUITY":
            info["not_equity"].append(s)
            continue
        if meta.at[s, "source"] == "sp500_changes":
            why = check_identity(first, name, meta.at[s, "change_date"], meta.at[s, "wiki_names"], symbol=s)
            if why:
                info["reused_ticker"][s] = why
                continue
        if n < min_rows:
            info["short_history"].append(s)
            continue
        elig.append(s)
    info["pool"] = len(syms)
    info["eligible"] = len(elig)
    return sorted(elig), info


def source_symbols(conn, source: str) -> list[str]:
    """Sorted distinct symbols of exactly this stored source (no version resolution)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM universe_membership WHERE source=%s", [source])
        return sorted(str(r[0]) for r in cur.fetchall())


def store_events(conn, events: pd.DataFrame) -> int:
    """Upsert events into universe_membership; returns the number of rows sent.

    Refuses (RuntimeError, before any write) if a source already holds a (symbol, as_of) key that the new
    events do not contain: rows cannot be removed, so such a key would silently corrupt the state.
    `as_of` may be a date, Timestamp, datetime64 or datetime; it is normalised to `datetime.date`.
    """
    too_long = sorted({str(x) for x in events["source"] if len(str(x)) > SOURCE_MAX})
    if too_long:
        raise ValueError(f"source names longer than {SOURCE_MAX} characters: {too_long}")
    with conn.cursor() as cur:
        cur.execute(UNIVERSE_MEMBERSHIP_DDL)
    events = events.copy()
    events["as_of"] = [pd.Timestamp(d).date() for d in events["as_of"]]   # date, whatever the input dtype
    for source, grp in events.groupby("source"):
        new_keys = set(zip(grp["symbol"], grp["as_of"]))
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, as_of FROM universe_membership WHERE source=%s", [source])
            old = {(s, pd.Timestamp(d).date()) for s, d in cur.fetchall()}
        stale = old - new_keys
        if stale:
            raise RuntimeError(f"source {source!r}: {len(stale)} stored (symbol, as_of) keys are not in the new "
                               "events; use a new source name or get approval (rows are never removed)")
    data = [(str(r.symbol), str(r.market), str(r.source), r.as_of, int(r.in_universe))
            for r in events.itertuples(index=False)]
    with conn.cursor() as cur:
        for i in range(0, len(data), CHUNK):
            cur.executemany(UPSERT_SQL, data[i:i + CHUNK])
            conn.commit()
    return len(data)


def load_universe(conn, panel: Panel, source: str) -> np.ndarray:
    """(T, N) bool mask: state on date t = last event with as_of <= t; False before a symbol's first event."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, as_of, in_universe FROM universe_membership WHERE source=%s "
                    "ORDER BY symbol, as_of", [source])
        rows = list(cur.fetchall())
    T, N = len(panel.dates), len(panel.symbols)
    out = np.zeros((T, N), dtype=bool)
    if not rows:
        return out
    df = pd.DataFrame(rows, columns=["symbol", "as_of", "in_universe"])
    df["as_of"] = pd.to_datetime(df["as_of"])
    col = {s: j for j, s in enumerate(panel.symbols)}
    pdates = panel.dates.values
    for sym, g in df.groupby("symbol"):
        j = col.get(sym)
        if j is None:
            continue
        ev_d = g["as_of"].values
        ev_s = g["in_universe"].to_numpy(dtype=int) != 0
        k = np.searchsorted(ev_d, pdates, side="right") - 1
        out[:, j] = np.where(k >= 0, ev_s[np.maximum(k, 0)], False)
    return out


def _setup_logging(market: str) -> Path:
    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / f"universe_{market}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.log"
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return path


def _cacap_k(conn, since: str, version: str) -> tuple[str, dict]:
    """(CA liquidity source used, {month Period: max over the month's CA rows of the per-row member count})."""
    from .cache import load_panel
    base = liquidity_source("ca")
    try:
        src = resolve_source(conn, versioned(base, version))
    except ValueError:
        try:
            src = resolve_source(conn, base)
        except ValueError:
            raise RuntimeError("build the CA universe first") from None
    cp = load_panel("ca", start=since, universe_source=None, with_groups=False)
    cnt = pd.Series(load_universe(conn, cp, src).sum(axis=1), index=cp.dates)
    cnt = cnt[cnt.index >= pd.Timestamp(since)]
    k = cnt.groupby(cnt.index.to_period("M")).max()
    return src, {m: int(v) for m, v in k.items()}


def build_universe(market: str, since: str = "2005-01-01", top_n: int | None = None,
                   min_price: float | None = None, store: bool = True, version: str | None = None,
                   max_spread="default", spread_window: int | None = None) -> dict:
    """Build (and by default store) every universe source of one market. See the module docstring.

    Every stored source is `versioned(<base>, version)`, version defaulting to the UTC date YYYYMMDD.
    max_spread: "default" = DEFAULTS[market], None = no screen, float = that bar. The result reports the
    names the screen removed per year (`screen_removed_by_year`, vs the unscreened mask). For "us" the H2
    capped source `liquidity_cacap_us@<version>` is also built: the liquidity mask with the month's member
    count capped at K_m, the maximum per-row member count of that month in the CA liquidity source.

    The Wikipedia pages are fetched twice: once inside `candidate_pool` and once more for the index tables
    (`fetch_sp500` / `fetch_tsx_composite`), so the pool and the index events may in principle come from two
    page versions; the `elig_set` filter limits the effect.
    """
    from .cache import load_panel
    market = market.lower()
    if market not in DEFAULTS:
        raise ValueError("market must be 'us' or 'ca'")
    top_n = DEFAULTS[market]["top_n"] if top_n is None else top_n
    min_price = DEFAULTS[market]["min_price"] if min_price is None else min_price
    max_spread = DEFAULTS[market]["max_spread"] if isinstance(max_spread, str) and max_spread == "default" \
        else max_spread
    if not (max_spread is None or (isinstance(max_spread, (int, float)) and not isinstance(max_spread, bool)
                                    and np.isfinite(max_spread) and max_spread > 0)):
        raise ValueError("max_spread must be 'default', None, or a finite float > 0")
    spread_window = DEFAULTS[market]["spread_window"] if spread_window is None else int(spread_window)
    version = f"{dt.datetime.now(dt.timezone.utc):%Y%m%d}" if version is None else version
    logfile = _setup_logging(market)
    stamp = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    html = Path("data/raw") / f"html_{stamp}"
    pool = candidate_pool(html, since=since)
    with reader() as conn:
        elig, info = eligible_symbols(conn, pool, market)
    log.info("market=%s pool=%d eligible=%d log=%s", market, info["pool"], info["eligible"], logfile)

    p = load_panel(market, start=since, universe_source=None, with_groups=False)
    args = dict(eligible=np.isin(p.symbols, elig), spread_window=spread_window)
    mask = liquidity_universe(p, top_n, min_price, max_spread=max_spread, **args)
    unscreened = mask if max_spread is None else liquidity_universe(p, top_n, min_price, max_spread=None, **args)
    keep = np.asarray(p.dates >= pd.Timestamp(since))
    removed = (unscreened & ~mask)[keep]
    years = p.dates[keep].year
    screen_removed = {int(y): {"mean_daily_removed": float(removed[years == y].sum(axis=1).mean()),
                               "symbols_removed": int(removed[years == y].any(axis=0).sum())}
                      for y in sorted(set(years))}
    elig_set = set(elig)
    parts = []
    src_liq = versioned(liquidity_source(market, top_n), version)
    ev = membership_events(mask[keep], p.dates[keep], p.symbols, market, src_liq)
    parts.append(ev[ev["symbol"].isin(elig_set)])
    cacap = None
    empty = pd.DataFrame(columns=["date", "added", "removed"])
    if market == "us":
        with reader() as conn:
            ca_src, k_m = _cacap_k(conn, since, version)
        capped = liquidity_universe(p, top_n, min_price, max_spread=max_spread, top_n_by_month=k_m, **args)
        ev = membership_events(capped[keep], p.dates[keep], p.symbols, market,
                               versioned("liquidity_cacap_us", version))
        parts.append(ev[ev["symbol"].isin(elig_set)])
        ccount = capped[keep].sum(axis=1)
        cacap = {"ca_source": ca_src, "months": len(k_m),
                 "member_count_range": (int(ccount.min()), int(ccount.max()))}
        cur, ch = constituents.fetch_sp500(html)
        for src, chg in (("sp500_wiki_current", empty), ("sp500_wiki_changes", ch)):
            ev = index_membership_events(cur, chg, "us", versioned(src, version), since)
            parts.append(ev[ev["symbol"].isin(elig_set)])
    else:
        tsx = constituents.fetch_tsx_composite(html)
        ev = index_membership_events(tsx, empty, "ca", versioned("tsx_wiki_current", version), since)
        parts.append(ev[ev["symbol"].isin(elig_set)])
    events = pd.concat(parts, ignore_index=True)

    written = 0
    if store:
        conn = writer()
        try:
            written = store_events(conn, events)
        finally:
            conn.close()
    counts = mask[keep].sum(axis=1)
    sources = {}
    for src, g in events.groupby("source"):
        sources[src] = {"events": int(len(g)), "symbols": int(g["symbol"].nunique()),
                        "min_as_of": str(g["as_of"].min()), "max_as_of": str(g["as_of"].max())}
    res = {"market": market, "version": version, "pool_info": info, "sources": sources,
           "member_count_range": (int(counts.min()), int(counts.max())), "rows_written": written,
           "screen": {"estimator": "abdi_ranaldo_2017", "max_spread": max_spread, "spread_window": spread_window},
           "screen_removed_by_year": screen_removed}
    if cacap is not None:
        res["cacap"] = cacap
    log.info("done: %s", {k: v for k, v in res.items() if k != "pool_info"})
    return res
