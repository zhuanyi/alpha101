"""Synthetic world for the study tests: designed price paths, no DB, no network.

Names i = 0..n-1 (ysym f"S{i:02d}.TO"); XIC holds the first `xic_n`, XCS holds the rest plus an overlap. YTD (D_prev -> R)
of name i is -0.50 + 0.0125 i, so name 0 is the worst loser; from L to X name i returns `post(i)` (default +5% for the
20 worst names, 0 for the rest), so the P basket (N=20) gross excess over the EW universe is exactly 2.5% when the
universe is the 40 names.
"""
import datetime as dt

import numpy as np
import pandas as pd

import prices as pr

D = dt.date


def sym(i):
    return f"S{i:02d}.TO"


def make_world(cal, years=(2012, 2013), n=40, xic_n=25, post=None, extra_days=(), yahoo_for=None, seed=1):
    """-> (snap DataFrame, YahooData). Dates: D_prev, Dec 1 .. X3 trading days, plus `extra_days` (month-ends)."""
    post = post or (lambda i: 0.05 if i < 20 else 0.0)
    rows, frames = [], {}
    yahoo_for = set(range(n)) if yahoo_for is None else set(yahoo_for)
    for y in years:
        e = cal.event_row(y)
        days = sorted({e["D_prev"], *[d for d in cal.tdays if dt.date(y, 12, 1) <= d <= e["VX3_on_or_after_feb15"]],
                       *[d for d in extra_days if d.year in (y - 1, y, y + 1)]})
        idx = {d: k for k, d in enumerate(cal.tdays)}
        L, X, R = e["L"], e["jan_exit_X"], e["R"]
        for i in range(n):
            p0 = 10.0 + 0.1 * i
            pR = p0 * (1 - 0.50 + 0.0125 * i)
            for d in days:
                if d <= e["D_prev"]:
                    p = p0
                elif d <= R:
                    p = p0 + (pR - p0) * min(1.0, (idx[d] - idx[e["D_prev"]]) / max(1, idx[R] - idx[e["D_prev"]]))
                elif d <= L:
                    p = pR
                else:
                    p = pR * (1 + post(i) * min(1.0, (idx[d] - idx[L]) / (idx[X] - idx[L])))
                if d > X:
                    p = pR * (1 + post(i))
                etfs = ["XIC"] if i < xic_n else ["XCS"]
                if 15 <= i < xic_n:
                    etfs = ["XIC", "XCS"]
                for etf in etfs:
                    rows.append({"etf": etf, "as_of": d, "ticker": f"S{i:02d}", "ysym": sym(i), "name": f"NAME {i:02d} CORP",
                                 "sector": "x", "price": p, "shares": 1000.0, "market_value": p * 1000.0,
                                 "weight": 0.5})
                if i in yahoo_for:
                    frames.setdefault(sym(i), {})[pd.Timestamp(d)] = p
    snap = pd.DataFrame(rows).drop_duplicates(["etf", "as_of", "ticker"]).reset_index(drop=True)
    yf = {}
    for s, dct in frames.items():
        c = pd.Series(dct).sort_index()
        yf[s] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "adj_close": c,
                              "volume": 1000.0, "dividend": 0.0, "split": 0.0})
    for s in ("XIC.TO",):
        c = pd.Series({pd.Timestamp(d): 20.0 for d in sorted(snap["as_of"].unique())})
        yf[s] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c, "volume": 1.0, "dividend": 0.0, "split": 0.0})
    first = {s: pd.Timestamp("2005-01-03") for s in yf}
    return snap, pr.YahooData(yf, first)


def mini_snap(rows):
    """rows: (etf, date, ticker, price, shares) -> snapshot frame with ysym = ticker + '.TO'."""
    return pd.DataFrame([{"etf": e, "as_of": d, "ticker": t, "ysym": t + ".TO", "name": t + " CORP", "sector": "x",
                          "price": p, "shares": s, "market_value": p * s, "weight": 1.0} for e, d, t, p, s in rows])
