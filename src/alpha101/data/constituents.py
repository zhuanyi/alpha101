"""Candidate pool: current S&P 500 + S&P changes since a date + current S&P/TSX Composite.

Wikipedia HTML is untrusted data: it is written to a fresh cache directory, parsed with
pandas/lxml, and never executed. The raw tables are also saved as CSV under `data/raw/` with the
fetch date in the filename. Both pools start from today's constituents, so they are
survivorship-biased (design.md section 5); TSX results are indicative only.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# The "Selected changes" table moved off the list page to this one (checked 2026-10-18).
SP500_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
TSX_URL = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
USER_AGENT = "alpha101-research/0.1" + (f" ({os.environ['ALPHA101_CONTACT']})" if os.environ.get("ALPHA101_CONTACT") else "")

log = logging.getLogger("alpha101.constituents")

_FOOTNOTE = re.compile(r"\[[^\]]*\]")
# A Yahoo symbol as stored in yahoo_ohlcv: BRK-B, RCI-B.TO, IIP-UN.TO. No whitespace, no "|".
_SYMBOL = re.compile(r"^[A-Z0-9]+(-[A-Z0-9]+)*(\.TO)?$")
DOWNLOAD_ATTEMPTS = 3
MAX_DROPPED_DATE_SHARE = 0.05   # a changes table losing more rows than this to date parsing is rejected
FIRST_TRADE_SLACK = dt.timedelta(days=7)
_NAME_NOISE = {"inc", "corp", "corporation", "co", "company", "the", "group", "holdings", "plc", "ltd", "of",
               "and", "limited", "class", "a", "b", "new", "trust", "international", "common", "stock"}

# Verified renames that the name rule rejects: the Yahoo name and the Wikipedia Security name are the
# same issuer under different names. These skip the name test only; the first-trade-date test still applies.
NAME_CHECK_OVERRIDES = {
    "AIV": "Aimco (Wikipedia) is Apartment Investment and Management Company (Yahoo), same issuer since 1994",
    "CNX": "Consol Energy (Wikipedia) was renamed CNX Resources Corporation (Yahoo) in 2017, same issuer",
}


def valid_symbol(symbol: str) -> bool:
    """True for a clean Yahoo symbol (`^[A-Z0-9]+(-[A-Z0-9]+)*(\\.TO)?$`)."""
    return isinstance(symbol, str) and _SYMBOL.match(symbol) is not None


def yahoo_symbol(ticker: str, market: str) -> str:
    """US: "BRK.B" -> "BRK-B". CA: "BBD.B" -> "BBD-B.TO", "RY" -> "RY.TO". Strips whitespace.

    An empty ticker gives "" (the caller drops it). Anything else that is not a clean symbol
    afterwards (inner whitespace, "|", other punctuation) raises ValueError.
    """
    t = _FOOTNOTE.sub("", str(ticker)).split("|")[0].strip().upper().replace(".", "-")
    m = market.strip().lower()
    if m not in ("us", "ca"):
        raise ValueError(f"unknown market {market!r}")
    if not t:
        return ""
    sym = t + ".TO" if m == "ca" else t
    if not valid_symbol(sym):
        raise ValueError(f"invalid Yahoo symbol {sym!r} from ticker {ticker!r}")
    return sym


def _name_tokens(name, split=r"[\s\-/]+") -> list[str]:
    toks = []
    for w in re.split(split, str(name).lower()):
        w = re.sub(r"[^a-z0-9]", "", w)
        if w:
            toks.append(w)
    return toks


def names_match(a, b) -> bool:
    """Loose company-name equality: a shared word, a word contained in a longer one (FMC / TechnipFMC),
    or an initialism (ATI = Allegheny Technologies Inc, VFC ~ V.F.). False if either name is empty."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False
    ka = [w for w in ta if w not in _NAME_NOISE] or ta
    kb = [w for w in tb if w not in _NAME_NOISE] or tb
    for x in ka:
        for y in kb:
            if x == y or (min(len(x), len(y)) >= 3 and (x in y or y in x)):
                return True
    # abbreviations are whole words: "O-I" is "oi" here, not "o", "i"
    wa, wb = _name_tokens(a, r"\s+"), _name_tokens(b, r"\s+")
    for short, full in ((wa, tb), (wb, ta)):
        init = "".join(w[0] for w in full)
        init_core = "".join(w[0] for w in full if w not in _NAME_NOISE)
        # trailing "i": the abbreviation may carry the "Inc" the other name lacks (ATI / Allegheny Technologies)
        if any(len(i) >= 2 and w in (i, i + "i")
               for w in short for i in (init, init_core)):
            return True
    return "".join(ka) == "".join(kb)


def check_identity(first_trade, yahoo_name, change_date, wiki_names, symbol=None) -> str | None:
    """Why a change-only ticker is NOT the issuer the S&P changes table meant, or None if it passes.

    Yahoo resolves a removed ticker to whoever holds it today. Two tests: the Yahoo history must not
    start after the (earliest) change date, and the Yahoo name must match a Wikipedia Security name.
    Symbols in `NAME_CHECK_OVERRIDES` skip the name test only.
    """
    if first_trade is not None and change_date is not None and not pd.isna(change_date):
        if pd.Timestamp(first_trade) > pd.Timestamp(change_date) + FIRST_TRADE_SLACK:
            return f"first trade {pd.Timestamp(first_trade).date()} after change date {pd.Timestamp(change_date).date()}"
    names = [n for n in (wiki_names or ()) if n]
    if symbol in NAME_CHECK_OVERRIDES:
        return None
    if yahoo_name and names and not any(names_match(yahoo_name, n) for n in names):
        return f"name {yahoo_name!r} matches none of {sorted(names)!r}"
    return None


def _stamp() -> str:
    return f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"


def _download(url: str, cache_dir: Path, name: str) -> str:
    """GET `url` (retry with backoff on network errors, 429 and 5xx), store the HTML under cache_dir
    (UTC timestamp in the name), return the text."""
    import requests
    for attempt in range(DOWNLOAD_ATTEMPTS):
        r = None
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            client_error = r is not None and 400 <= r.status_code < 500 and r.status_code != 429
            if client_error or attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            log.warning("GET %s failed (%s), retrying", url, type(e).__name__)
            time.sleep(2.0 * 2 ** attempt)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{name}_{_stamp()}.html").write_text(r.text, encoding="utf-8")
    return r.text


def _tables(html: str) -> list[pd.DataFrame]:
    # keep_default_na=False: the TSX ticker "NA" (National Bank) must not become NaN
    return pd.read_html(io.StringIO(html), flavor="lxml", keep_default_na=False)


def _flat(cols) -> list[str]:
    """Flatten (possibly MultiIndex) column labels to 'a b' strings, dropping repeats."""
    out = []
    for c in cols:
        parts = c if isinstance(c, tuple) else (c,)
        seen: list[str] = []
        for p in parts:
            p = _FOOTNOTE.sub("", str(p)).replace("\xa0", " ").strip()
            if p and p not in seen:
                seen.append(p)
        out.append(" ".join(seen))
    return out


def parse_sp500_current(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for t in tables:
        cols = _flat(t.columns)
        if "Symbol" in cols and "GICS Sector" in cols:
            d = t.copy()
            d.columns = cols
            out = pd.DataFrame({
                "symbol": d["Symbol"].map(lambda x: _FOOTNOTE.sub("", str(x)).strip()),
                "name": d["Security"],
                "gics_sector": d["GICS Sector"],
                "gics_sub_industry": d["GICS Sub-Industry"],
            })
            return out.reset_index(drop=True)
    raise ValueError("S&P 500 constituents table not found")


def _clean_cell(x):
    return None if pd.isna(x) else (_FOOTNOTE.sub("", str(x)).replace("\xa0", " ").split("|")[0].strip() or None)


def _clean_date(x):
    return pd.NaT if pd.isna(x) else pd.to_datetime(
        _FOOTNOTE.sub("", str(x)).replace("\xa0", " ").strip(), format="%B %d, %Y", errors="coerce")


def parse_sp500_changes(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """[date, added, added_name, removed, removed_name, reason] (raw tickers, None where a side is empty),
    or None if no changes table is present. Raises ValueError if a table is present but unusable
    (no row with a parsable date, or more than MAX_DROPPED_DATE_SHARE of the dated rows lost)."""
    for t in tables:
        cols = _flat(t.columns)
        if not any(c.startswith("Added") for c in cols) or not any(c.startswith("Removed") for c in cols):
            continue
        d = t.copy()
        d.columns = cols
        date_c = next(c for c in cols if "Date" in c)
        add_c = next(c for c in cols if c.startswith("Added") and "Ticker" in c)
        rem_c = next(c for c in cols if c.startswith("Removed") and "Ticker" in c)
        add_n = next((c for c in cols if c.startswith("Added") and "Security" in c), None)
        rem_n = next((c for c in cols if c.startswith("Removed") and "Security" in c), None)
        reason_c = next((c for c in cols if c.startswith("Reason")), None)
        dates = d[date_c].map(_clean_date)
        # a repeated header row has no digit in its date cell; it is structure, not a lost row
        dated = d[date_c].map(lambda x: bool(re.search(r"\d", str(x))))
        lost = int((dates.isna() & dated).sum())
        if dates.notna().sum() == 0 or lost > MAX_DROPPED_DATE_SHARE * int(dated.sum()):
            raise ValueError(f"S&P 500 changes table: {int(dates.notna().sum())} parsable dates, "
                             f"{lost} of {int(dated.sum())} dated rows lost")
        if lost:
            log.warning("S&P 500 changes table: %d dated rows dropped (unparsable date)", lost)
        out = pd.DataFrame({"date": dates,
                            "added": d[add_c].map(_clean_cell),
                            "added_name": d[add_n].map(_clean_cell) if add_n else None,
                            "removed": d[rem_c].map(_clean_cell),
                            "removed_name": d[rem_n].map(_clean_cell) if rem_n else None,
                            "reason": d[reason_c].map(_clean_cell) if reason_c else None})
        return out.dropna(subset=["date"]).reset_index(drop=True)
    return None


def parse_tsx(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for t in tables:
        cols = _flat(t.columns)
        if "Ticker" in cols and "Company" in cols:
            d = t.copy()
            d.columns = cols
            sector_c = next(c for c in cols if c.startswith("Sector"))
            return pd.DataFrame({
                "symbol": d["Ticker"].map(lambda x: _FOOTNOTE.sub("", str(x)).strip()),
                "name": d["Company"],
                "sector": d[sector_c],
            }).reset_index(drop=True)
    raise ValueError("S&P/TSX Composite constituents table not found")


def _save_raw(df: pd.DataFrame, raw_dir: Path, name: str) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / f"{name}_{_stamp()}.csv", index=False)


def fetch_sp500(cache_dir: Path, raw_dir: Path = Path("data/raw")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(current[symbol, name, gics_sector, gics_sub_industry],
    changes[date, added, added_name, removed, removed_name, reason]). Both are saved whole under raw_dir."""
    page = _tables(_download(SP500_URL, cache_dir, "sp500_list"))
    current = parse_sp500_current(page)
    try:
        changes = parse_sp500_changes(page)  # old layout: the table sits on the list page
    except ValueError as e:
        log.warning("changes table on the list page is unusable (%s); trying %s", e, SP500_CHANGES_URL)
        changes = None
    if changes is None:
        changes = parse_sp500_changes(_tables(_download(SP500_CHANGES_URL, cache_dir, "sp500_changes")))
    if changes is None:
        raise ValueError("S&P 500 changes table not found")
    _save_raw(current, raw_dir, "sp500_current")
    _save_raw(changes, raw_dir, "sp500_changes")
    return current, changes


def fetch_tsx_composite(cache_dir: Path, raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """current[symbol, name, sector]."""
    current = parse_tsx(_tables(_download(TSX_URL, cache_dir, "tsx_composite")))
    _save_raw(current, raw_dir, "tsx_current")
    return current


def candidate_pool(cache_dir: Path, since: str = "2005-01-01",
                   raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """[symbol, market, source, change_date, wiki_names]: current S&P 500, S&P adds/removes since
    `since`, current TSX.

    Order is stable: current names in page order, then change-only names by first appearance.
    A symbol keeps the first source it appears under. For change-only symbols `change_date` is the
    earliest change row naming the ticker and `wiki_names` the Security names Wikipedia gives it
    (inputs of `check_identity`); both are empty otherwise. Cells that are not clean Yahoo symbols
    are dropped with a warning.
    """
    cur, ch = fetch_sp500(cache_dir, raw_dir)
    tsx = fetch_tsx_composite(cache_dir, raw_dir)
    return _pool(cur, ch, tsx, since)


def _pool(cur: pd.DataFrame, ch: pd.DataFrame, tsx: pd.DataFrame, since: str) -> pd.DataFrame:
    """The `candidate_pool` frame from the three parsed tables (shared with `pool_from_raw`)."""
    rows: list[tuple] = []

    def add(raw, market, source, date=pd.NaT, name=None):
        try:
            sym = yahoo_symbol(raw, market)
        except ValueError as e:
            log.warning("dropping pool entry: %s", e)
            return
        if sym:
            rows.append((sym, market.upper(), source, date, name))

    for s in cur["symbol"]:
        add(s, "us", "sp500_current")
    recent = ch[ch["date"] >= pd.Timestamp(since)]
    for col, ncol in (("added", "added_name"), ("removed", "removed_name")):
        names = recent[ncol] if ncol in recent else [None] * len(recent)
        for s, dte, n in zip(recent[col], recent["date"], names):
            if s is not None and not pd.isna(s):
                add(s, "us", "sp500_changes", dte, n)
    for s in tsx["symbol"]:
        add(s, "ca", "tsx_current")
    allrows = pd.DataFrame(rows, columns=["symbol", "market", "source", "change_date", "wiki_name"])
    first = allrows.drop_duplicates("symbol", keep="first").set_index("symbol")
    ch_rows = allrows[allrows["source"] == "sp500_changes"]
    first["change_date"] = pd.NaT
    first["wiki_names"] = [() for _ in range(len(first))]
    only = first.index[first["source"] == "sp500_changes"]
    grp = ch_rows[ch_rows["symbol"].isin(only)].groupby("symbol")
    first.loc[only, "change_date"] = grp["change_date"].min().reindex(only)
    first.loc[only, "wiki_names"] = pd.Series(
        {k: tuple(dict.fromkeys(n for n in g["wiki_name"] if isinstance(n, str) and n)) for k, g in grp}
    ).reindex(only).map(lambda v: v if isinstance(v, tuple) else ())
    pool = first.drop(columns="wiki_name").reset_index()
    return pool[["symbol", "market", "source", "change_date", "wiki_names"]]


def pool_from_raw(raw_dir: Path, since: str = "2005-01-01") -> pd.DataFrame:
    """`candidate_pool` rebuilt offline from the latest saved `sp500_current_*.csv`, `sp500_changes_*.csv` and
    `tsx_current_*.csv` in raw_dir (latest by the UTC timestamp in the file name, each independently).

    FileNotFoundError if any of the three is missing. `pool.attrs["snapshot"]` lists the three file names, `["since"]` is `since`.
    """
    raw_dir = Path(raw_dir)
    files = []
    for stem in ("sp500_current", "sp500_changes", "tsx_current"):
        found = sorted(raw_dir.glob(f"{stem}_*.csv"), key=lambda p: p.name)
        if not found:
            raise FileNotFoundError(f"no {stem}_*.csv in {raw_dir}")
        files.append(found[-1])

    def read(p):   # "NA" is a ticker, not a missing value; an empty cell stays "" (dropped by yahoo_symbol)
        return pd.read_csv(p, dtype=str, keep_default_na=False)

    cur, ch, tsx = (read(p) for p in files)
    ch["date"] = pd.to_datetime(ch["date"])
    pool = _pool(cur, ch, tsx, since)
    pool.attrs["snapshot"] = [p.name for p in files]
    pool.attrs["since"] = since
    return pool
