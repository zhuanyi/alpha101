import datetime as dt

import numpy as np
import pandas as pd

import coverage
import rebound

D = dt.date


def closes(start, n, px=10.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(px, index=idx)


def hold(px=10.0):
    return pd.DataFrame({"as_of": [D(2024, 6, 3), D(2024, 6, 4), D(2024, 6, 5)], "price": [px] * 3})


def test_covered_when_identity_and_post_rows_ok():
    c = closes("2024-05-01", 120)
    a = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(), c, None, c.index.min())
    assert a["ident_ok"] and a["post_rows"] >= 20 and a["covered"]


def test_not_covered_when_series_ends_at_deletion():
    c = closes("2024-05-01", 25)                     # ends ~2024-06-04
    a = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(), c, None, c.index.min())
    assert a["post_rows"] < 20 and not a["covered"]


def test_ticker_reuse_fails_identity():
    c = closes("2024-05-01", 120, px=50.0)           # another issuer's prices under the same symbol
    a = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(10.0), c, None, c.index.min())
    assert not a["ident_ok"] and not a["covered"] and a["reason"] == "price_mismatch"


def test_split_unadjusting_rescues_identity():
    c = closes("2024-05-01", 120, px=100.0)          # Yahoo basis after a later 1-for-10 consolidation: 10x as-traded
    splits = pd.Series({pd.Timestamp("2024-09-02"): 0.1})
    a = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(10.0), c, splits, c.index.min())
    assert a["ident_ok"] and a["covered"]
    b = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(10.0), c, None, c.index.min())
    assert not b["ident_ok"]


def test_no_data(tmp_path):
    a = coverage.assess("X.TO", D(2024, 6, 5), D(2024, 6, 6), hold(), None, None, None)
    assert not a["covered"] and a["reason"] == "no_yahoo" and a["post_rows"] == 0


def test_yf_frame_caches_hits_and_misses(tmp_path):
    calls = []
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3), "close": [1.0, 1.0, 1.0], "split": 0.0})

    def fetch(s):
        calls.append(s)
        return df if s == "A.TO" else None

    assert len(coverage.yf_frame("A.TO", tmp_path, fetch)) == 3
    assert len(coverage.yf_frame("A.TO", tmp_path, fetch)) == 3
    assert coverage.yf_frame("B.TO", tmp_path, fetch).empty and coverage.yf_frame("B.TO", tmp_path, fetch).empty
    assert calls == ["A.TO", "B.TO"]


def test_targets_and_per_year():
    ev = pd.DataFrame([
        {"etf": "XIC", "direction": "delete", "ysym": "A.TO", "name": "A", "last_in": D(2020, 3, 20),
         "first_out": D(2020, 3, 23), "class": "left_both", "weight_ref": 0.5, "flags": ""},
        {"etf": "XIC", "direction": "delete", "ysym": "B.TO", "name": "B", "last_in": D(2020, 3, 20),
         "first_out": D(2020, 3, 23), "class": "to_xcs", "weight_ref": 0.5, "flags": ""},
        {"etf": "XIC", "direction": "delete", "ysym": "C.TO", "name": "C", "last_in": D(2020, 3, 20),
         "first_out": D(2020, 3, 23), "class": "left_both", "weight_ref": 0.5, "flags": "unconfirmed"}])
    t = coverage.targets(ev)
    assert list(t["ysym"]) == ["A.TO"]
    cov = t.assign(source="none")
    py = coverage.per_year(cov)
    assert py.loc[py["year"] == 2020, "uncovered"].iloc[0] == 1


def test_exit_class_uses_tax_loss_class_names():
    import rebound
    ev = pd.DataFrame([
        {"etf": "XIC", "direction": "delete", "ysym": "A.TO", "class": "to_xcs", "first_out": D(2020, 3, 20)},
        {"etf": "XIC", "direction": "delete", "ysym": "B.TO", "class": "left_both", "first_out": D(2020, 3, 20)},
        {"etf": "XIC", "direction": "delete", "ysym": "C.TO", "class": "left_both", "first_out": D(2020, 3, 20)},
        {"etf": "XIC", "direction": "delete", "ysym": "D.TO", "class": "ticker_change", "first_out": D(2020, 3, 20)},
        {"etf": "XIC", "direction": "add", "ysym": "E.TO", "class": "new", "first_out": D(2020, 3, 20)}])
    cov = pd.DataFrame([{"etf": "XIC", "ysym": "B.TO", "first_out": D(2020, 3, 20), "source": "yfinance"},
                        {"etf": "XIC", "ysym": "C.TO", "first_out": D(2020, 3, 20), "source": "none"}])
    out = coverage.add_exit_class(ev, cov)
    assert list(out["exit_class"]) == ["other_etf", "yahoo", "unresolved", "", ""]
    assert set(out["exit_class"]) - {""} <= set(rebound.CLASSES)
