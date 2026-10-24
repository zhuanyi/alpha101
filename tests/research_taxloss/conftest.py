"""Puts research/taxloss on sys.path; synthetic TSX calendars (weekdays minus explicit closures), no DB, no network."""
import datetime as dt
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAXLOSS = os.path.join(ROOT, "research", "taxloss")
if TAXLOSS not in sys.path:
    sys.path.insert(0, TAXLOSS)

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
def make_cal():
    import taxcal

    def f(closed, start=D(2005, 1, 3), end=D(2027, 3, 31)):
        return taxcal.Calendar(weekdays(start, end, closed))
    return f


@pytest.fixture(scope="session")
def generic_cal(make_cal):
    """Dec 25/26 and Jan 1 closed in every year (structure tests, not date fixtures)."""
    closed = [D(y, m, d) for y in range(2005, 2028) for m, d in ((12, 25), (12, 26), (1, 1))]
    return make_cal(closed)
