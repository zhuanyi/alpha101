"""Puts research/taxloss and research/tsxpanel on sys.path; weekday calendars, no DB, no network."""
import datetime as dt
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("taxloss", "tsxpanel"):
    p = os.path.join(ROOT, "research", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

D = dt.date


def weekdays(start, end, closed=()):
    closed = set(closed)
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in closed:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


@pytest.fixture(scope="session")
def cal():
    import taxcal
    closed = [D(y, m, d) for y in range(2010, 2028) for m, d in ((12, 25), (12, 26), (1, 1))]
    return taxcal.Calendar(weekdays(D(2010, 1, 4), D(2027, 3, 31), closed))
