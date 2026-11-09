"""TSX tax-year calendar for the tax-loss rebound study (pre-registered spec, 2.1/3/8).

Everything is a method of `Calendar`, which is built from a sorted list of TSX trading days, so tests need no DB.

Settlement regime by TRADE date (verified 2026-10-20, spec 2.1):
  T+3 before 2017-09-05  https://fcnb.ca/en/news-alerts/canadian-securities-regulators-outline-steps-to-support-transition-to-t2
                         https://www.mondaq.com/canada/securities/628312/canada-welcomes-t%2b2-settlement
  T+2 2017-09-05 .. 2024-05-24
  T+1 from 2024-05-27    https://www.asc.ca/en/news-and-publications/news-releases/2024/05/may-27-canadian-securities-regulators-announce-move-to-t1-settlement-cycle
TSX settlement schedules (banking holidays on which TSX trades but nothing settles): https://www.tsx.com/en/resource/3407
"""
from __future__ import annotations

import bisect
import datetime as dt

T2_START = dt.date(2017, 9, 5)
T1_START = dt.date(2024, 5, 27)
REGIME_URLS = {
    "T+3->T+2": ["https://fcnb.ca/en/news-alerts/canadian-securities-regulators-outline-steps-to-support-transition-to-t2",
                 "https://www.mondaq.com/canada/securities/628312/canada-welcomes-t%2b2-settlement"],
    "T+2->T+1": ["https://www.asc.ca/en/news-and-publications/news-releases/2024/05/"
                 "may-27-canadian-securities-regulators-announce-move-to-t1-settlement-cycle"],
    "settlement schedule 2026": ["https://www.tsx.com/en/resource/3407"],
}

# Days on which the TSX trades but nothing settles. Sep 30 is the National Day for Truth and Reconciliation.
# Its first observed year is 2021 (federal statutory holiday); the spec marks this UNVERIFIED, see run.log.
TRUTH_RECONCILIATION_FIRST_YEAR = 2021

# TSX closures after the last ^GSPTSE row in the DB (2026 TSX schedule: Dec 25 and Dec 28 in lieu of Boxing Day;
# 2027 dates are derived by rule: Jan 1, Family Day = 3rd Monday of Feb, Good Friday).
FUTURE_HOLIDAYS = [dt.date(2026, 12, 25), dt.date(2026, 12, 28), dt.date(2027, 1, 1),
                   dt.date(2027, 2, 15), dt.date(2027, 3, 26)]
FUTURE_END = dt.date(2027, 3, 31)


# Manual cross-check of L against published notices (2026-10-20). "page" = page opened and the sentence read;
# "search" = only a search-result summary. 2016 and 2012-2015/2018-2022/2024 were NOT cross-checked.
SETTLEMENT_NOTICES = [
    (2017, dt.date(2017, 12, 27), "page", "https://www.advisor.ca/tax/tax-news/year-end-tips-for-tax-loss-selling/",
     '"To complete settlement by Dec. 31, the trade date must be no later than Dec. 27, 2017."'),
    (2023, dt.date(2023, 12, 27), "page", "https://www.advisor.ca/tax/tax-strategies/year-end-tax-planning-for-2023/",
     '"clients have to settle trades no later than Dec. 27 to be able to use their loss for this year"'),
    (2025, dt.date(2025, 12, 30), "page+search", "https://money.tmx.com/content-hub/navigating-volatility-content-hub/primer-tax-loss-selling-canada/",
     '"placing trades by no later than December 30" (TMX page names no year; the Dec 30 2025 date is from search-result '
     'summaries of https://www.nasdaq.com/articles/mark-these-tax-loss-selling-dates-your-calendar and other secondary pages)'),
]
# Sep 30 first observed 2021: TMX Operational Notice 2021-041 (2021-09-24)
SEP30_NOTICE = "https://www.tsx.com/trading/toronto-stock-exchange/trading-notices?id=104"


def settlement_regime(trade_date) -> int:
    """k = number of settlement days after the trade date: 3, 2 or 1 by the regime on the TRADE date."""
    d = _d(trade_date)
    return 3 if d < T2_START else 2 if d < T1_START else 1


def banking_holiday(d: dt.date) -> bool:
    """TSX trades but nothing settles: Remembrance Day, and Sep 30 from 2021."""
    return (d.month, d.day) == (11, 11) or ((d.month, d.day) == (9, 30) and d.year >= TRUTH_RECONCILIATION_FIRST_YEAR)


def _d(x) -> dt.date:
    return x.date() if hasattr(x, "date") and callable(x.date) else x


def extend_trading_days(days, future_holidays=FUTURE_HOLIDAYS, end=FUTURE_END) -> list[dt.date]:
    """Observed ^GSPTSE dates, extended with weekdays minus the published/derived closures after the last row."""
    out = sorted({_d(x) for x in days})
    hol = set(future_holidays)
    d = out[-1] + dt.timedelta(days=1)
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def tsx_trading_days(conn=None) -> list[dt.date]:
    """Dates of ^GSPTSE in yahoo_ohlcv (rows with a positive close), extended with the closures above."""
    from alpha101.data.yahoo import read_long
    if conn is None:
        from alpha101.db import reader
        conn = reader()
    df = read_long(conn, ["^GSPTSE"])
    return extend_trading_days(df.loc[df["close"] > 0, "date"].dt.date)


class Calendar:
    def __init__(self, trading_days):
        self.tdays = sorted({_d(x) for x in trading_days})
        self.sd = [d for d in self.tdays if not banking_holiday(d)]       # settlement calendar

    # --- basic navigation -------------------------------------------------------------------------------------
    def settlement_days(self) -> list[dt.date]:
        return list(self.sd)

    def settle(self, t) -> dt.date:
        """Settlement date of a trade on t: the k-th settlement day strictly after t, k by the regime on t."""
        i = bisect.bisect_right(self.sd, _d(t)) - 1 + settlement_regime(t)
        if i >= len(self.sd):
            raise ValueError(f"calendar too short to settle {t}")
        return self.sd[i]

    def last_trading_day_of_month(self, y: int, m: int) -> dt.date:
        c = [d for d in self.tdays if (d.year, d.month) == (y, m)]
        if not c:
            raise ValueError(f"no trading days in {y}-{m:02d}")
        return c[-1]

    def first_trading_day_of_month(self, y: int, m: int) -> dt.date:
        c = [d for d in self.tdays if (d.year, d.month) == (y, m)]
        if not c:
            raise ValueError(f"no trading days in {y}-{m:02d}")
        return c[0]

    def shift(self, d, n: int) -> dt.date:
        """The trading day n trading days after (n>0) or before (n<0) trading day d."""
        i = self.tdays.index(_d(d)) + n
        if not 0 <= i < len(self.tdays):
            raise ValueError("calendar too short")
        return self.tdays[i]

    def on_or_after(self, d) -> dt.date:
        i = bisect.bisect_left(self.tdays, _d(d))
        if i >= len(self.tdays):
            raise ValueError("calendar too short")
        return self.tdays[i]

    def before(self, d) -> dt.date:
        """Last trading day strictly before d."""
        i = bisect.bisect_left(self.tdays, _d(d)) - 1
        if i < 0:
            raise ValueError("calendar too short")
        return self.tdays[i]

    # --- the study's dates ------------------------------------------------------------------------------------
    def placebo_L(self, y: int, m: int) -> dt.date:
        """Latest trading day t in month m whose settlement is on or before the last settlement day of month m
        (k by the regime on t). For m=12 this is the tax-loss deadline L_y (Dec 31 itself never settles trades)."""
        last_settle = [d for d in self.sd if (d.year, d.month) == (y, m)][-1]
        for t in reversed([d for d in self.tdays if (d.year, d.month) == (y, m)]):
            if self.settle(t) <= last_settle:
                return t
        raise ValueError(f"no L for {y}-{m:02d}")

    def last_taxloss_day(self, y: int) -> dt.date:
        return self.placebo_L(y, 12)

    def rank_day(self, y: int) -> dt.date:
        return self.shift(self.last_taxloss_day(y), -1)

    def placebo_rank_day(self, y: int, m: int) -> dt.date:
        return self.shift(self.placebo_L(y, m), -1)

    def jan_exit(self, y: int) -> dt.date:
        """P exit: last trading day of January y+1."""
        return self.last_trading_day_of_month(y + 1, 1)

    def dec_review_effective(self, y: int) -> dt.date:
        """Monday after the third Friday of December (or the next trading day if that Monday is closed)."""
        fridays = [dt.date(y, 12, d) for d in range(1, 32) if dt.date(y, 12, d).weekday() == 4]
        return self.on_or_after(fridays[2] + dt.timedelta(days=3))

    def nth_trading_day_of_month(self, y: int, m: int, n: int) -> dt.date:
        c = [d for d in self.tdays if (d.year, d.month) == (y, m)]
        if len(c) < n:
            raise ValueError(f"calendar too short: fewer than {n} trading days in {y}-{m:02d}")
        return c[n - 1]

    def event_row(self, y: int) -> dict:
        L = self.last_taxloss_day(y)
        eff = self.dec_review_effective(y)
        return {
            "year": y, "regime": f"T+{settlement_regime(L)}",
            "D_prev": self.last_trading_day_of_month(y - 1, 12),
            "pre_review_day": self.before(eff), "dec_review_effective": eff,
            "R": self.rank_day(y), "L": L, "L_settles": self.settle(L),
            "VE0_rank_day": self.shift(L, -6), "VE0_entry": self.shift(L, -5),
            "first_jan_day": self.first_trading_day_of_month(y + 1, 1),
            "VX1_5th_jan_day": self.nth_trading_day_of_month(y + 1, 1, 5),
            "jan_exit_X": self.jan_exit(y),
            "VX3_on_or_after_feb15": self.on_or_after(dt.date(y + 1, 2, 15)),
        }
