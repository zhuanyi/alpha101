import datetime as dt

import pytest

import taxcal

D = dt.date


def test_fixture_2025_t1(make_cal):
    cal = make_cal([D(2025, 12, 25), D(2026, 1, 1)])
    assert cal.last_taxloss_day(2025) == D(2025, 12, 30)
    assert cal.rank_day(2025) == D(2025, 12, 29)
    assert cal.settle(D(2025, 12, 31)) == D(2026, 1, 2)       # Dec 31 + 1 settlement day: Jan 1 is closed


def test_fixture_2023_t2(make_cal):
    cal = make_cal([D(2023, 12, 25), D(2023, 12, 26), D(2024, 1, 1)])
    assert cal.settle(D(2023, 12, 27)) == D(2023, 12, 29)
    assert cal.settle(D(2023, 12, 28)) == D(2024, 1, 2)
    assert cal.last_taxloss_day(2023) == D(2023, 12, 27)


def test_fixture_2016_t3(make_cal):
    cal = make_cal([D(2016, 12, 26), D(2016, 12, 27), D(2017, 1, 2)])
    assert cal.settle(D(2016, 12, 23)) == D(2016, 12, 30)
    assert cal.settle(D(2016, 12, 28)) == D(2017, 1, 3)
    assert cal.last_taxloss_day(2016) == D(2016, 12, 23)


def test_fixture_2026_live(make_cal):
    cal = make_cal([D(2026, 12, 25), D(2026, 12, 28), D(2027, 1, 1)])
    assert cal.last_taxloss_day(2026) == D(2026, 12, 30)
    assert cal.rank_day(2026) == D(2026, 12, 29)
    assert cal.settle(D(2026, 12, 31)) == D(2027, 1, 4)       # TSX schedule: Dec 31 -> Monday Jan 4


def test_regime_switch_dates():
    assert taxcal.settlement_regime(D(2017, 9, 1)) == 3
    assert taxcal.settlement_regime(D(2017, 9, 5)) == 2
    assert taxcal.settlement_regime(D(2024, 5, 24)) == 2
    assert taxcal.settlement_regime(D(2024, 5, 27)) == 1
    assert taxcal.settlement_regime(D(2012, 1, 3)) == 3


def test_banking_holidays_are_trading_days_but_not_settlement_days(make_cal):
    cal = make_cal([])
    assert D(2026, 11, 11) in cal.tdays and D(2026, 11, 11) not in cal.sd
    assert cal.settle(D(2026, 11, 10)) == D(2026, 11, 12)     # TSX 2026 schedule
    assert cal.settle(D(2026, 9, 29)) == D(2026, 10, 1)
    assert D(2020, 9, 30) in cal.sd                           # before the first observance (2021)
    assert D(2021, 9, 30) not in cal.sd
    assert D(2018, 11, 12) in cal.sd and D(2018, 11, 11) not in cal.tdays      # Sunday: not a trading day at all


def test_placebo_L_uses_regime_on_trade_date(make_cal):
    cal = make_cal([])
    assert cal.placebo_L(2017, 9) == D(2017, 9, 27)           # T+2: Sep 27 -> Sep 29
    assert cal.placebo_L(2017, 8) == D(2017, 8, 28)           # T+3: Aug 28 -> Aug 31
    assert cal.placebo_L(2024, 5) == D(2024, 5, 30)           # T+1 from May 27: May 30 -> May 31
    assert cal.placebo_L(2024, 4) == D(2024, 4, 26)           # T+2: Apr 26 -> Apr 30


def test_december_placebo_is_the_primary_L(generic_cal):
    for y in range(2012, 2027):
        assert generic_cal.placebo_L(y, 12) == generic_cal.last_taxloss_day(y)
        assert generic_cal.placebo_rank_day(y, 12) == generic_cal.rank_day(y)


def test_dec_review_effective_is_monday_after_third_friday(generic_cal):
    assert generic_cal.dec_review_effective(2018) == D(2018, 12, 24)
    assert generic_cal.dec_review_effective(2025) == D(2025, 12, 22)
    for y in range(2012, 2027):
        e = generic_cal.dec_review_effective(y)
        assert e.weekday() == 0 and 18 <= e.day <= 24


def test_event_row_variant_dates(generic_cal):
    r = generic_cal.event_row(2025)
    assert r["VE0_entry"] == generic_cal.shift(r["L"], -5) and r["VE0_rank_day"] == generic_cal.shift(r["L"], -6)
    assert r["first_jan_day"] == D(2026, 1, 2) and r["VX1_5th_jan_day"] == D(2026, 1, 8)
    assert r["jan_exit_X"] == D(2026, 1, 30)
    assert r["VX3_on_or_after_feb15"] == D(2026, 2, 16)       # Feb 15 2026 is a Sunday
    assert r["D_prev"] == D(2024, 12, 31) and r["pre_review_day"] < r["dec_review_effective"]


def test_extension_adds_future_closures():
    days = taxcal.extend_trading_days([D(2026, 10, 15), D(2026, 10, 16)])
    assert D(2026, 12, 25) not in days and D(2026, 12, 28) not in days
    assert D(2026, 12, 30) in days and D(2027, 1, 29) in days and D(2027, 2, 15) not in days


def test_sep30_2021_matches_tmx_notice(make_cal):
    cal = make_cal([])
    assert cal.settle(D(2021, 9, 28)) == D(2021, 10, 1)      # TMX Operational Notice 2021-041
    assert cal.settle(D(2021, 9, 29)) == D(2021, 10, 4)
    assert cal.settle(D(2021, 9, 30)) == D(2021, 10, 4)


def test_published_notices_match_calendar(make_cal):
    cal = make_cal([D(2017, 12, 25), D(2017, 12, 26), D(2018, 1, 1), D(2023, 12, 25), D(2023, 12, 26), D(2024, 1, 1),
                    D(2025, 12, 25), D(2026, 1, 1)])
    for y, claimed, *_ in taxcal.SETTLEMENT_NOTICES:
        assert cal.last_taxloss_day(y) == claimed


def test_event_row_short_calendar_raises_value_error(make_cal):
    cal = make_cal([], start=D(2012, 1, 3), end=D(2013, 1, 3))
    with pytest.raises(ValueError):
        cal.nth_trading_day_of_month(2013, 1, 5)
