import datetime as dt

import reviews

D = dt.date


def test_third_friday():
    assert reviews.third_friday(2024, 12) == D(2024, 12, 20)
    assert reviews.third_friday(2025, 3) == D(2025, 3, 21)
    assert reviews.third_friday(2012, 9) == D(2012, 9, 21)


def test_review_dates_count_and_weekday(cal):
    r = reviews.review_dates(cal)
    assert len(r) == 14 * 4
    assert all(d.weekday() == 4 for d in r["review_date"])


def test_window_is_21_trading_days_around_T(cal):
    T = D(2024, 12, 20)
    w = reviews.window_dates(cal, T)
    assert len(w) == 21 and w[10] == T
    assert all(d in set(cal.tdays) for d in w)
    assert w[0] < T < w[-1]


def test_fetch_plan_covers_both_etfs(cal):
    p = reviews.fetch_plan(cal, years=[2020])
    assert set(p["etf"]) == {"XIC", "XCS"}
    assert len(p) == 4 * 21 * 2
    assert p["offset"].min() == -10 and p["offset"].max() == 10
