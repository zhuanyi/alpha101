import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "slow_long_only", Path(__file__).resolve().parents[1] / "scripts" / "slow_long_only.py")
slo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slo)


def _frame(n=504, active_daily=0.0004, tickets_per_rebalance=20.0, every=20):
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    net = bench + active_daily
    return pd.DataFrame({"gross": net + 0.0001, "net": net, "benchmark": bench,
                         "turnover": 0.5, "n_trades": np.where(np.arange(n) % every == 0, tickets_per_rebalance, 0.0)},
                        index=idx)


def test_active_is_net_minus_benchmark_annualised():
    m = slo.row_metrics(_frame(), 100_000.0)
    assert m["active_ann"] == pytest.approx(0.0004 * 252)
    assert m["net_ann"] - m["bench_ann"] == pytest.approx(m["active_ann"])
    assert m["gross_ann"] > m["net_ann"]


def test_ticket_accounting_and_breakeven():
    # n_trades is sparse as in composite_backtest: 20 tickets on every 20th row (26 rebalances in 504 days)
    f = _frame(tickets_per_rebalance=20.0, every=20)
    m = slo.row_metrics(f, 100_000.0)
    expected = f["n_trades"].sum() / (504 / 252)
    assert m["trades_per_year"] == pytest.approx(expected)
    assert m["commission_bps_per_year"] == pytest.approx(expected * 9.99 / 100_000 * 1e4)
    # at the break-even fee the active return is zero: active + (9.99 - fee) * tickets / book == 0
    assert m["active_ann"] + (9.99 - m["breakeven_fee"]) * m["trades_per_year"] / 100_000 == pytest.approx(0.0)


def test_holdout_rows_are_dropped_and_ci_brackets_mean():
    f = _frame(n=1200)                      # runs past 2019-01-01
    m = slo.row_metrics(f, 100_000.0)
    assert m["days"] == int((f.index < pd.Timestamp(slo.HOLDOUT)).sum())
    assert m["active_lo"] < m["active_ann"] < m["active_hi"]


def test_fee_drag_skips_tickets_the_library_never_charges():
    # the last in-sample rebalance sits in the NaN straddle band: its tickets carry no cost in `net`
    a = _frame(tickets_per_rebalance=20.0, every=20)
    b = a.copy()
    a["net"] = a["net"] - np.where(a["n_trades"] > 0, 20 * 9.99 / 100_000, 0.0)
    a.loc[a.index[-1], ["gross", "net", "benchmark"]] = np.nan
    a.loc[a.index[-1], "n_trades"] = 36.0
    lib, flat = slo.fee_drag(a, b, 100_000.0)
    assert flat == pytest.approx(lib)
    assert flat == pytest.approx(a["n_trades"].iloc[:-1].sum() * 9.99 / 100_000 / (503 / 252))


def test_rows_without_a_finite_return_are_dropped_with_their_tickets():
    f = _frame()
    f.iloc[:40, f.columns.get_loc("gross")] = np.nan       # flat warm-up: no book, no return
    f.iloc[:40, f.columns.get_loc("net")] = np.nan
    m = slo.row_metrics(f, 100_000.0)
    assert m["days"] == 464
    assert m["trades_per_year"] == pytest.approx(f["n_trades"].iloc[40:].sum() / (464 / 252))


def test_ci_is_mean_plus_minus_1_96_nw_se():
    f = _frame()
    f["net"] = f["benchmark"] + np.random.default_rng(1).normal(0.0004, 0.003, len(f))
    m = slo.row_metrics(f, 100_000.0)
    act = f["net"] - f["benchmark"]
    t = slo.newey_west_t(act, lags=21)
    assert m["active_t"] == pytest.approx(t)
    half = 1.96 * abs(act.mean() / t) * 252
    assert m["active_hi"] - m["active_ann"] == pytest.approx(half)
    assert m["active_ann"] - m["active_lo"] == pytest.approx(half)


def test_report_headline_counts_positive_active_with_t_above_2(tmp_path, monkeypatch):
    monkeypatch.setattr(slo, "TMP", tmp_path)
    monkeypatch.setattr(slo, "OUT", tmp_path)
    sel = {m: {str(h): [{"alpha_id": f"alpha{i:03d}", "mean_ic": 0.01, "t_nw": 3.0, "n_days": 100}
                        for i in range(1, 11)] for h in slo.HORIZONS} for m in slo.MARKETS}
    (tmp_path / "sel.json").write_text(__import__("json").dumps(sel))
    monkeypatch.setattr(slo, "SEL_JSON", tmp_path / "sel.json")
    base = {k: 0.1 for k in ("gross_ann", "net_ann", "bench_ann", "active_lo", "active_hi", "turnover_oneway",
                             "trades_per_year", "commission_bps_per_year", "breakeven_fee")}
    rows = {"us": [("composite10", 0.02, 2.5), ("alpha001", 0.02, 2.0), ("alpha002", -0.02, 3.0)],
            "ca": [("composite10", 0.01, 2.1), ("alpha001", 0.0, 0.0)]}
    for m, rs in rows.items():
        pd.DataFrame([{"market": m, "strategy": s, "rebalance": "M", "book": 100000, "error": "", "days": 10,
                       **base, "active_ann": a, "active_t": t} for s, a, t in rs]).to_csv(tmp_path / f"slow_long_only_{m}.csv",
                                                                                         index=False)
    slo.report(5)
    text = (tmp_path / "slow_long_only_tables.md").read_text()
    assert "**Rows with active > 0 and NW t > 2 after costs: 2 of 5.**" in text
    assert "Rows with active > 0 (any t): 3." in text
