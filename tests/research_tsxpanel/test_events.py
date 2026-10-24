import datetime as dt

import pandas as pd

import events

D = dt.date
COLS = ["date", "etf", "ticker", "ysym", "name", "isin", "weight", "price", "shares", "sector", "source"]


def mk(rows):
    return pd.DataFrame([{"date": d, "etf": e, "ticker": y.split(".")[0], "ysym": y, "name": n, "isin": None,
                          "weight": 1.0, "price": 10.0, "shares": 100.0, "sector": "X", "source": "parsed"}
                         for d, e, y, n in rows], columns=COLS)


DAYS = [D(2024, 12, 17), D(2024, 12, 18), D(2024, 12, 19), D(2024, 12, 20), D(2024, 12, 23), D(2024, 12, 24),
        D(2024, 12, 27), D(2024, 12, 30)]


def base_rows(skip=()):
    rows = []
    for d in DAYS:
        for e, y, n in (("XIC", "AAA.TO", "AAA CORP"), ("XIC", "BBB.TO", "BBB INC"), ("XCS", "SMA.TO", "SMALL CORP")):
            if (d, y) not in skip:
                rows.append((d, e, y, n))
    return rows


def run(rows, cal, reviews=(D(2024, 12, 20),)):
    return events.build_events(mk(rows), cal, list(reviews))


def test_clean_deletion_at_review(cal):
    gone = {(d, "BBB.TO") for d in DAYS if d > D(2024, 12, 20)}
    ev = run(base_rows(gone), cal)
    e = ev[ev["direction"] == "delete"]
    assert list(e["ysym"]) == ["BBB.TO"]
    r = e.iloc[0]
    assert (r["last_in"], r["first_out"]) == (D(2024, 12, 20), D(2024, 12, 23))
    assert r["class"] == "left_both" and r["review_date"] == D(2024, 12, 20) and r["review_offset"] == 1
    assert r["review_aligned"]
    assert r["source"] == "daily" and r["flags"] == ""


def test_move_to_xcs_is_not_left_both(cal):
    gone = {(d, "BBB.TO") for d in DAYS if d > D(2024, 12, 20)}
    rows = base_rows(gone) + [(d, "XCS", "BBB.TO", "BBB INC") for d in DAYS if d > D(2024, 12, 20)]
    ev = run(rows, cal)
    assert list(ev[ev["direction"] == "delete"]["class"]) == ["to_xcs"]
    add = ev[(ev["direction"] == "add") & (ev["etf"] == "XCS")]
    assert list(add["class"]) == ["from_xic"]


def test_transient_absence_is_not_an_event(cal):
    ev = run(base_rows({(D(2024, 12, 20), "BBB.TO")}), cal)           # missing from one file only
    assert ev.empty


def test_ticker_change_by_name(cal):
    rows = [r for r in base_rows() if not (r[0] > D(2024, 12, 20) and r[2] == "BBB.TO")]
    rows += [(d, "XIC", "BBX.TO", "BBB INC") for d in DAYS if d > D(2024, 12, 20)]
    ev = run(rows, cal)
    assert set(ev["class"]) == {"ticker_change"} and len(ev) == 2


def test_unconfirmed_when_file_history_ends(cal):
    rows = [r for r in base_rows() if r[0] <= D(2024, 12, 23)]
    rows = [r for r in rows if not (r[0] == D(2024, 12, 23) and r[2] == "BBB.TO")]
    ev = run(rows, cal)
    assert "unconfirmed" in ev.iloc[0]["flags"]


def test_wide_bracket_flags_and_off_cycle_counts(cal):
    rows = [(D(2024, 7, 31), "XIC", "AAA.TO", "AAA CORP"), (D(2024, 7, 31), "XIC", "BBB.TO", "BBB INC"),
            (D(2024, 8, 30), "XIC", "AAA.TO", "AAA CORP"), (D(2024, 9, 30), "XIC", "AAA.TO", "AAA CORP"),
            (D(2024, 10, 31), "XIC", "AAA.TO", "AAA CORP")]
    ev = run(rows, cal, reviews=(D(2024, 9, 20),))
    r = ev.iloc[0]
    assert r["source"] == "month_end" and "wide_bracket" in r["flags"] and r["n_review_candidates"] == 0
    oc = events.off_cycle(ev)
    assert int(oc["certain_off_cycle"].sum()) == 1 and int(oc["month_end_resolution"].sum()) == 1


def test_counts_per_year_and_offset_distribution(cal):
    gone = {(d, "BBB.TO") for d in DAYS if d > D(2024, 12, 20)}
    ev = run(base_rows(gone), cal)
    c = events.counts_per_year(ev)
    assert c.loc[(c["direction"] == "delete") & (c["class"] == "left_both"), 2024].iloc[0] == 1
    o = events.review_offset_distribution(ev)
    assert o.iloc[0]["review_offset"] == 1 and o.iloc[0]["n"] == 1


def test_rename_candidate_flag(cal):
    rows = [r for r in base_rows() if not (r[0] > D(2024, 12, 20) and r[2] == "BBB.TO")]
    rows += [(d, "XIC", "NEWB.TO", "BBBTECH HOLDINGS") for d in DAYS if d > D(2024, 12, 20)]
    rows = [(d, e, y, "BBBTECH INC" if y == "BBB.TO" else n) for d, e, y, n in rows]
    ev = run(rows, cal)
    assert set(ev["class"]) == {"left_both", "new"}
    assert ev["flags"].str.contains("rename_candidate").all()


def test_two_day_fund_lag_is_review_aligned_and_three_is_not(cal):
    T = D(2024, 12, 20)
    for gone_from, aligned in ((D(2024, 12, 24), True), (D(2024, 12, 27), False)):      # first_out T+2 / T+4
        gone = {(d, "BBB.TO") for d in DAYS if d >= gone_from}
        ev = run(base_rows(gone), cal, reviews=(T,))
        assert bool(ev[ev["direction"] == "delete"].iloc[0]["review_aligned"]) is aligned
