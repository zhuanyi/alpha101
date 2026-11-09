"""Read `alpha101.yahoo_ohlcv` into a Panel.

Stored values are what yfinance returns with auto_adjust=False: open/high/low/close and volume
split-adjusted to the share basis on the fetch date, NOT dividend-adjusted; adj_close is split
and dividend adjusted; `split` is the split ratio on its effective date (0 otherwise).
"""
from __future__ import annotations

import pandas as pd

from ..panel import Panel

COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]


def read_long(conn, symbols: list[str] | None = None, start: str | None = None,
              end: str | None = None) -> pd.DataFrame:
    """Rows of yahoo_ohlcv as a long DataFrame (pure %s placeholders, no literals)."""
    where, params = [], []
    if symbols:
        where.append("symbol IN (" + ",".join(["%s"] * len(symbols)) + ")")
        params.extend(symbols)
    if start:
        where.append("date >= %s")
        params.append(start)
    if end:
        where.append("date <= %s")
        params.append(end)
    sql = f"SELECT {','.join(COLUMNS)} FROM yahoo_ohlcv"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY symbol, date"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    for c in COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def panel_from_long(df: pd.DataFrame, basis: str = "raw", calendar: pd.DatetimeIndex | None = None,
                    **kw) -> Panel:
    """Wide Panel from read_long output. basis="raw" (as-traded, default) or "adjusted".

    calendar: the market's trading days (e.g. the dates of ^GSPC or ^GSPTSE). Build ONE panel
    per market: US and TSX holidays differ, and on a union calendar every foreign holiday
    becomes a NaN row that, under the strict NaN policy, blanks d days of every window op.
    Rows off the calendar are dropped; a symbol missing on a calendar day gets NaN.
    Rows with a non-positive or missing close are dropped (Yahoo placeholders); zero volume is
    kept as NaN volume so it cannot enter log(volume) or adv as a real observation.
    """
    d = df[(df["close"] > 0)].copy()
    if calendar is not None:
        d = d[d["date"].isin(calendar)]
    d.loc[d["volume"] <= 0, "volume"] = float("nan")
    d["split"] = d["split"].fillna(0.0)
    fields = ["open", "high", "low", "close", "adj_close", "volume", "split"]
    p = Panel.from_long(d[["symbol", "date"] + fields], **kw)
    if calendar is not None:
        p = p.reindex_dates(pd.DatetimeIndex(calendar))
    p.fields["split"] = pd.DataFrame(p.fields["split"]).fillna(0.0).to_numpy(dtype=float, copy=True)
    if basis == "raw":
        return p.to_raw_basis()
    if basis != "adjusted":
        raise ValueError("basis must be 'raw' or 'adjusted'")
    return p
