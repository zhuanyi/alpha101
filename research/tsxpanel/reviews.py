"""Index-review dates and the daily-file fetch plan around them.

S&P/TSX quarterly reviews take effect after the close on the third Friday of Mar/Jun/Sep/Dec, so the review date T is
that Friday (the last day a deleted name is still in the index) and the first trading day after T is the first day it
is gone. The plan asks for the 10 trading days before and after T (T itself included: 21 dates per review).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "taxloss"))

REVIEW_MONTHS = (3, 6, 9, 12)
YEARS = range(2012, 2026)
WINDOW = 10


def third_friday(y: int, m: int) -> dt.date:
    fridays = [dt.date(y, m, d) for d in range(1, 29) if dt.date(y, m, d).weekday() == 4]
    return fridays[2]


def review_dates(cal, years=YEARS) -> pd.DataFrame:
    """One row per quarterly review: year, month, nominal third Friday, review_date (T = the last trading day on or
    before the third Friday: it differs only if that Friday is a market holiday, e.g. Good Friday is never a third
    Friday of Mar/Jun/Sep/Dec in range but the rule is kept general)."""
    rows = []
    for y in years:
        for m in REVIEW_MONTHS:
            f = third_friday(y, m)
            t = f if f in set(cal.tdays) else cal.before(f)
            rows.append({"year": y, "month": m, "third_friday": f, "review_date": t})
    return pd.DataFrame(rows)


def window_dates(cal, review_date, k: int = WINDOW) -> list[dt.date]:
    return [cal.shift(review_date, i) for i in range(-k, k + 1)]


def fetch_plan(cal, years=YEARS, k: int = WINDOW) -> pd.DataFrame:
    """(etf, date, review_date, offset) for every window date of every review, both ETFs."""
    from ishares import ETFS
    rows = []
    for r in review_dates(cal, years).itertuples(index=False):
        for i, d in enumerate(window_dates(cal, r.review_date, k)):
            for etf in ETFS:
                rows.append({"etf": etf, "date": d, "review_date": r.review_date, "offset": i - k})
    return pd.DataFrame(rows)
