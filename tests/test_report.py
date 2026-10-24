"""Offline tests for report.py: a synthetic two-market RunOutput, no DB, no network."""
import dataclasses
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha101 import report as R
from alpha101.costs import CostModel
from alpha101.data import constituents, universe
from alpha101.data.constituents import candidate_pool, parse_sp500_current, parse_tsx, pool_from_raw
from alpha101.evaluate import RunConfig, evaluate_alpha, ic_correlation, multiple_testing
from alpha101.panel import Panel
from alpha101.report import (MarketOutput, RunOutput, benchmark_gap, delisting_effect, ic_ci, periods,
                             render_markdown, report, screen_summary, years_to_significance)
from alpha101.synthetic import random_panel

ERAS = (("early", "2015-01-01", "2015-03-31"), ("mid", "2015-04-01", "2015-06-30"),
        ("holdout_era", "2015-07-01", "2015-12-31"))
IDS = (1, 6, 12, 42, 56, 101)


def _cfg(unlock=False):
    return RunConfig(markets=("us", "ca"), horizons=(1, 5), primary_horizon=5, start="2015-01-01",
                     holdout_start="2015-07-01", unlock_holdout=unlock, min_names={"us": 10, "ca": 10},
                     eras=ERAS)


@pytest.fixture(scope="module")
def built():
    """(panels, market outputs) evaluated once under an unlocked config; locked runs re-wrap them."""
    cfg = _cfg(unlock=False)
    costs = CostModel()
    panels, outs = {}, {}
    for seed, mk in enumerate(("us", "ca")):
        p = random_panel(T=200, N=30, seed=seed)
        panels[mk] = p
        res = {}
        for i in IDS:
            r = evaluate_alpha(p, i, cfg, None, market=mk)
            res[r.alpha_id] = r
        summ = pd.DataFrame({a: {"t_nw": r.summary[5]["is"]["t_nw"]} for a, r in res.items()}).T
        corr = ic_correlation({a: r.ic[5]["ic"] for a, r in res.items()}, before=cfg.holdout_start)
        bench = pd.Series(100.0 * 1.0003 ** np.arange(len(p.dates)), index=p.dates)
        w = "equal" if mk == "us" else "cap"
        outs[mk] = MarketOutput(
            market=mk, universe_source=f"liquidity_top500_{mk}@20261019", results=res,
            failures={"alpha099": "MissingField: panel has no field 'zzz'"},
            multiple_testing=multiple_testing(summ, corr), ic_corr=corr,
            member_counts=pd.Series(np.where(np.arange(200) < 100, 28, 30), index=p.dates),
            survivorship=({"pool": 863, "with_data": 653, "no_data": 210, "reused_ticker": 17, "not_equity": 4,
                           "short_history": 3, "eligible": 629, "current": 503, "change_only": 360,
                           "change_only_with_data": 150, "coverage": 150 / 360, "snapshot": ["sp500_current_X.csv"]}
                          if mk == "us" else
                          {"pool": 234, "with_data": 231, "no_data": 3, "reused_ticker": 0, "not_equity": 1,
                           "short_history": 2, "eligible": 228, "change_history": False}),
            benchmark={"name": "BENCH", "weighting": w, "table": benchmark_gap(p, bench, cfg, w)},
            screen={"estimator": "abdi_ranaldo_2017", "max_spread": 0.02, "spread_window": 63,
                    "by_year": {2015: {"mean_daily_removed": 3.25, "symbols_removed": 7, "members_mean": 27.5}}},
            delisting_changed=7,
            cap_series=(pd.Series([21, 19, 25], index=pd.period_range("2015-01", periods=3, freq="M"))
                        if mk == "us" else None))
    return cfg, costs, outs


def _run(built, unlock=False, paired=True):
    cfg, costs, outs = built
    cfg = dataclasses.replace(cfg, unlock_holdout=unlock)
    pr = pd.DataFrame({"market": ["us"], "alpha_id": ["alpha001"], "mean_diff": [0.01234], "t_nw": [1.5],
                       "n_days": [88]}) if paired else None
    return RunOutput(run_id="run-20261019-abc", git_sha="deadbeef1234", created_at="2026-10-19T12:34:56+00:00",
                     cfg=cfg, costs=costs, markets=outs,
                     fingerprints={mk: {"rows": 123456, "symbols": 777, "max_fetched_at": "2026-10-18T01:02:03"}
                                   for mk in outs},
                     prereg_sha256="f" * 64 if unlock else None, paired=pr,
                     baseline_run_id="base-20261001" if paired else None)


# Numbers the markdown may contain without having been formatted: structural text and the legend.
ALLOW = {"95%", "50%", "252", "1.96"}


def stray_numbers(md, seen, run):
    """Number tokens of `md` that neither the formatter recorded nor the explicit allowlist covers."""
    lit = [run.run_id, run.git_sha, run.created_at, run.baseline_run_id or "", run.prereg_sha256 or "",
           "delay0", "bonferroni_101", "S&P 500", "SHA-256", "abdi_ranaldo_2017"]
    lit += [m.universe_source for m in run.markets.values()]
    lit += [msg for m in run.markets.values() for msg in m.failures.values()]
    lit += [e[0] for e in run.cfg.eras]
    rx = [r"\d{4}-\d{2}-\d{2}(T[\d:.+Z-]*)?", r"[\w.]+\.(?:png|csv)", r"alpha\s*\d+", r"\bA\d\b", r"\bH\d\b",
          r"section \d"]
    for d in [re.escape(x) for x in lit if x] + rx:
        md = re.sub(d, " ", md)
    return [t for t in re.findall(r"-?\d+(?:\.\d+)?%?", md) if t not in seen and t not in ALLOW]


def test_files_exist(built, tmp_path):
    path = report(_run(built), tmp_path)
    assert path == tmp_path / "report.md"
    for name in ("report.md", "ic_heatmap.png", "neff_scree.png", "cum_ic_us.png", "cum_ic_ca.png",
                 "deciles_us.png", "deciles_ca.png"):
        assert (tmp_path / name).stat().st_size > 0, name
    md = path.read_text()
    for name in ("ic_heatmap.png", "cum_ic_ca.png", "deciles_us.png", "neff_scree.png"):
        assert f"]({name})" in md


def test_every_number_is_recorded(built):
    run = _run(built, unlock=True)
    md, seen = render_markdown(run, {"cap": "x.png"})
    assert not stray_numbers(md, seen, run)
    # the scan has teeth: a stray literal is caught
    assert stray_numbers(md + "\nmean IC 0.1234\n", seen, run) == ["0.1234"]


def test_spot_checks(built):
    run = _run(built)
    md, seen = render_markdown(run, {})
    us = run.markets["us"]
    row = next(l for l in md.splitlines() if l.startswith("| alpha001 |"))
    assert row.split("|")[2].strip() == format(us.results["alpha001"].summary[5]["is"]["mean_ic"], ".4f")
    assert format(us.multiple_testing["n_eff"], ".1f") in md
    head = md[:md.index("## Run metadata")]
    assert "863" in head and "210" in head and "Survivorship" in head and "TSX results are indicative" in head
    assert "pit=false" in row or "pit=false" in next(l for l in md.splitlines() if l.startswith("| alpha056 |"))
    assert "delay0" in next(l for l in md.splitlines() if l.startswith("| alpha042 |"))
    assert "MissingField" in md and "No baseline run given." not in md and "base-20261001" in md
    assert "No baseline run given." in render_markdown(_run(built, paired=False), {})[0]


def test_locked_hides_holdout_era(built):
    locked, unlocked = _run(built), _run(built, unlock=True)
    assert [p[0] for p in periods(locked.cfg)] == ["is", "early", "mid"]
    assert [p[0] for p in periods(unlocked.cfg)] == ["is", "early", "mid", "holdout_era"]
    assert periods(locked.cfg)[0] == ("is", "2015-01-01", "2015-06-30")
    md_l, seen = render_markdown(locked, {})
    md_u, _ = render_markdown(unlocked, {})
    assert "holdout_era" not in md_l and "holdout_era IC" in md_u
    assert "UNLOCKED" in md_u and "locked" in md_l and "nan" not in md_l.lower()
    assert not stray_numbers(md_l, seen, locked)


def test_helpers_hand_cases():
    lo, hi = ic_ci({"mean_ic": 0.02, "t_nw": 4.0})
    assert lo == pytest.approx(0.02 - 1.96 * 0.005) and hi == pytest.approx(0.02 + 1.96 * 0.005)
    assert all(np.isnan(ic_ci({"mean_ic": 0.02, "t_nw": 0.0})))
    assert all(np.isnan(ic_ci({"mean_ic": 0.02, "t_nw": float("nan")})))
    assert years_to_significance(0.1, 1) == pytest.approx((1.96 / (0.1 * np.sqrt(252))) ** 2)
    assert years_to_significance(-0.1, 5) == pytest.approx((1.96 / (0.1 * np.sqrt(252 / 5))) ** 2)
    assert np.isnan(years_to_significance(0.0, 5)) and np.isnan(years_to_significance(float("nan"), 5))


def _toy_panel(adj, **extra):
    adj = np.asarray(adj, dtype=float)
    T, N = adj.shape
    f = {"adj_close": adj, "close": adj.copy(), **extra}
    return Panel(dates=pd.bdate_range("2020-01-01", periods=T), symbols=pd.Index([f"S{i}" for i in range(N)]),
                 fields=f)


def test_benchmark_gap_constant_returns():
    T, g, gb = 120, 0.001, 0.0004
    t = np.arange(T)[:, None]
    p = _toy_panel(np.hstack([100 * (1 + g) ** t, 50 * (1 + g) ** t]), shares_outstanding=np.full((T, 2), 1e6))
    bench = pd.Series(10 * (1 + gb) ** np.arange(T), index=p.dates)
    cfg = RunConfig(start="2020-01-01", holdout_start="2030-01-01")
    for w in ("equal", "cap"):
        tab = benchmark_gap(p, bench, cfg, w)
        rows = T - 2                                   # lag 1 + the last row has no exit price
        months = len(p.dates[:rows].to_period("M").unique())
        u, b = (1 + g) ** rows, (1 + gb) ** rows
        assert list(tab.index) == ["is", "2005-2012", "2013-2018", "2019-2026"]
        r = tab.loc["is"]
        assert r["months"] == months
        assert r["universe"] == pytest.approx(u ** (12 / months) - 1)
        assert r["benchmark"] == pytest.approx(b ** (12 / months) - 1)
        assert r["gap"] == pytest.approx(r["universe"] - r["benchmark"])
        assert tab.loc["2005-2012", "months"] == 0 and np.isnan(tab.loc["2005-2012", "gap"])
        assert tab.loc["2019-2026", "universe"] == pytest.approx(r["universe"])


def test_delisting_effect_hand_count():
    T, L = 40, 20
    px = 100 * 1.001 ** np.arange(T)[:, None] * np.ones((1, 2))
    ended = px.copy()
    ended[L + 1:, 0] = np.nan                           # symbol 0's last close is row L, > DELIST_GRACE before the end
    cfg = RunConfig(start="2020-01-01", holdout_start="2030-01-01", primary_horizon=5)
    # moved cells: entry row t+1 <= L and L < t+1+5  ->  t = 15..19
    assert delisting_effect(_toy_panel(ended), cfg) == 5
    assert delisting_effect(_toy_panel(px), cfg) == 0
    assert delisting_effect(_toy_panel(ended), dataclasses.replace(cfg, delisting="none")) == 0


def _wide_panel():
    T, N = 330, 6
    close = np.full((T, N), 50.0)
    high, low = close * 1.001, close / 1.001
    # name 0: close alternates +-5% around the high-low centre
    sign = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    close[:, 0] = 50.0 * np.exp(0.05 * sign)
    high[:, 0], low[:, 0] = 50.0 * np.exp(0.1), 50.0 * np.exp(-0.1)
    f = {"open": close.copy(), "high": high, "low": low, "close": close, "adj_close": close.copy(),
         "volume": np.full((T, N), 1e6), "split_age": np.full((T, N), 1e9)}
    syms = [f"S{i}" for i in range(N)]
    p = Panel(dates=pd.bdate_range("2020-01-01", periods=T), symbols=pd.Index(syms), fields=f)
    return p, syms, N


def test_screen_summary_wide_spread_name():
    p, syms, N = _wide_panel()
    d = universe.DEFAULTS["us"]
    p.universe = universe.liquidity_universe(p, d["top_n"], d["min_price"], eligible=np.ones(N, bool),
                                             max_spread=d["max_spread"], spread_window=d["spread_window"])
    assert not p.universe[:, 0].any() and p.universe[:, 1:].any()
    s = screen_summary(p, "us", syms)
    assert s["estimator"] == "abdi_ranaldo_2017" and s["max_spread"] == d["max_spread"]
    assert s["spread_window"] == d["spread_window"] and set(s["by_year"]) == {2021}   # 252 rows of history first exist in 2021
    for v in s["by_year"].values():
        assert v["mean_daily_removed"] == pytest.approx(1.0) and v["symbols_removed"] == 1
        assert v["members_mean"] == pytest.approx(N - 1)
    # an ineligible symbol is not in the unscreened set, so it is not "removed" by the screen either
    s2 = screen_summary(p, "us", syms[1:])
    assert all(v["mean_daily_removed"] == 0.0 and v["symbols_removed"] == 0 for v in s2["by_year"].values())


def test_pool_from_raw_equals_candidate_pool(monkeypatch, tmp_path):
    cur = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "NA"], "Security": ["Apple", "Berkshire", "Nano"],
                        "GICS Sector": ["IT", "Fin", "IT"], "GICS Sub-Industry": ["HW", "Ins", "HW"]})
    ch = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2010-05-05", "2012-01-01"]),
                       "added": ["TSLA", None, "ZZZ"], "added_name": ["Tesla", None, "Zed"],
                       "removed": [None, "ZZZ", "A B"], "removed_name": [None, "Zed Corp", "bad"]})
    tsx = pd.DataFrame({"Ticker": ["RY", "BBD.B"], "Company": ["Royal", "Bombardier"],
                        "Sector [10]": ["Fin", "Ind"], "Industry [10]": ["a", "b"]})
    cur_p, tsx_p = parse_sp500_current([cur]), parse_tsx([tsx])
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError):
        pool_from_raw(raw)
    stale = cur_p.assign(symbol="OLDX")                 # an older snapshot that must lose
    stale.to_csv(raw / "sp500_current_20250101T000000Z.csv", index=False)
    cur_p.to_csv(raw / "sp500_current_20260101T000000Z.csv", index=False)
    ch.to_csv(raw / "sp500_changes_20260101T000000Z.csv", index=False)
    tsx_p.to_csv(raw / "tsx_current_20260101T000000Z.csv", index=False)
    monkeypatch.setattr(constituents, "fetch_sp500", lambda cache_dir, raw_dir=None: (cur_p, ch))
    monkeypatch.setattr(constituents, "fetch_tsx_composite", lambda cache_dir, raw_dir=None: tsx_p)
    want = candidate_pool(tmp_path)
    got = pool_from_raw(raw)
    assert got.attrs["snapshot"] == ["sp500_current_20260101T000000Z.csv", "sp500_changes_20260101T000000Z.csv",
                                     "tsx_current_20260101T000000Z.csv"]
    assert "NA" in set(got["symbol"]) and "OLDX" not in set(got["symbol"])
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True), check_dtype=False)


class _Cur:
    def __init__(self, rows):
        self.rows, self.args = rows, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.args = (sql, args)

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows):
        self.c = _Cur(rows)

    def cursor(self):
        return self.c


def test_source_symbols_fake_cursor():
    conn = _Conn([("ZZ",), ("AA",), ("MM",)])
    assert universe.source_symbols(conn, "src@20261019") == ["AA", "MM", "ZZ"]
    assert conn.c.args[1] == ["src@20261019"] and "%s" in conn.c.args[0]


def test_survivorship_counts(monkeypatch):
    pool = pd.DataFrame({"symbol": ["A", "B", "C", "D", "E", "T1"], "market": ["US"] * 5 + ["CA"],
                         "source": ["sp500_current", "sp500_current", "sp500_changes", "sp500_changes",
                                    "sp500_changes", "tsx_current"]})
    pool.attrs["snapshot"] = ["f.csv"]
    info = {"no_data": ["C"], "not_equity": ["B"], "reused_ticker": {"D": "name"}, "short_history": [],
            "pool": 5, "eligible": 2}
    monkeypatch.setattr(universe, "eligible_symbols", lambda conn, pool, market: (["A", "E"], info))
    s = R.survivorship_counts(None, pool, "us")
    assert (s["pool"], s["with_data"], s["no_data"], s["reused_ticker"]) == (5, 4, 1, 1)
    assert (s["current"], s["change_only"], s["change_only_with_data"]) == (2, 3, 1)
    assert s["coverage"] == pytest.approx(1 / 3) and s["snapshot"] == ["f.csv"]
    assert s["since"] is None
    monkeypatch.setattr(universe, "eligible_symbols",
                        lambda conn, pool, market: ([], {"no_data": [], "not_equity": [], "reused_ticker": {},
                                                         "short_history": [], "pool": 1, "eligible": 1}))
    assert R.survivorship_counts(None, pool, "ca")["change_history"] is False


# ---------------------------------------------------------------------- additional checks
def test_screen_summary_start_end_hand_case():
    p, syms, N = _wide_panel()
    d = universe.DEFAULTS["us"]
    p.universe = universe.liquidity_universe(p, d["top_n"], d["min_price"], eligible=np.ones(N, bool),
                                             max_spread=d["max_spread"], spread_window=d["spread_window"])
    first = int(np.argmax(p.universe.any(axis=1)))
    cut = first + 10
    p.universe[cut:, 1] = False                          # the universe shrinks from row `cut` on
    full = screen_summary(p, "us", syms)["by_year"][2021]
    assert full["members_mean"] < N - 1
    early = screen_summary(p, "us", syms, end=p.dates[cut].strftime("%Y-%m-%d"))["by_year"][2021]
    assert early["members_mean"] == pytest.approx(N - 1)           # rows on/after `end` are ignored
    late = screen_summary(p, "us", syms, start=p.dates[cut].strftime("%Y-%m-%d"))["by_year"][2021]
    assert late["members_mean"] == pytest.approx(N - 2)            # rows before `start` are ignored
    assert screen_summary(p, "us", syms, end=p.dates[first].strftime("%Y-%m-%d"))["by_year"] == {}
    assert screen_summary(p, "us", syms, start="2030-01-01")["by_year"] == {}


def test_locked_tables_cut_exactly_at_holdout(built):
    since = "2005-01-01"
    screen = {"estimator": "abdi_ranaldo_2017", "max_spread": 0.02, "spread_window": 63,
              "by_year": {2014: {"mean_daily_removed": 1.5, "symbols_removed": 1, "members_mean": 811.3},
                          2015: {"mean_daily_removed": 2.5, "symbols_removed": 2, "members_mean": 812.3},
                          2016: {"mean_daily_removed": 3.5, "symbols_removed": 3, "members_mean": 913.7}}}
    cap = pd.Series([4111, 4222, 4321], index=pd.PeriodIndex(["2014-12", "2015-06", "2015-07"], freq="M"))
    for unlock in (False, True):
        run = _run(built, unlock=unlock)
        mo = dataclasses.replace(run.markets["us"], screen=screen, cap_series=cap)
        run.markets = {**run.markets, "us": mo}
        md, seen = render_markdown(run, {})
        assert not stray_numbers(md, seen, run)
        assert "811.3" not in md and "4111" not in md              # before cfg.start: never shown
        assert "812.3" in md and "4222" in md                      # visible in both
        assert ("913.7" in md) == unlock and ("4321" in md) == unlock   # on/after the holdout: only unlocked
        assert "screen_summary" in md


def test_pool_from_raw_empty_symbol_cells(monkeypatch, tmp_path):
    cur = pd.DataFrame({"symbol": ["AAPL", "", "NA"], "name": ["Apple", "x", "Nano"]})
    ch = pd.DataFrame({"date": ["2020-01-01", "2012-01-01"], "added": ["TSLA", ""], "added_name": ["Tesla", ""],
                       "removed": ["", "ZZZ"], "removed_name": ["", "Zed"]})
    tsx = pd.DataFrame({"symbol": ["RY.TO", "", "BBD-B.TO"], "name": ["Royal", "y", "Bombardier"]})
    raw = tmp_path / "raw"
    raw.mkdir()
    cur.to_csv(raw / "sp500_current_20260101T000000Z.csv", index=False)
    ch.to_csv(raw / "sp500_changes_20260101T000000Z.csv", index=False)
    tsx.to_csv(raw / "tsx_current_20260101T000000Z.csv", index=False)
    chd = ch.assign(date=pd.to_datetime(ch["date"])).replace("", None)
    monkeypatch.setattr(constituents, "fetch_sp500", lambda cache_dir, raw_dir=None: (cur, chd))
    monkeypatch.setattr(constituents, "fetch_tsx_composite", lambda cache_dir, raw_dir=None: tsx)
    got = pool_from_raw(raw)
    assert not {"NAN", "NAN.TO", ""} & set(got["symbol"]) and "NA" in set(got["symbol"])
    assert got.attrs["since"] == "2005-01-01"
    pd.testing.assert_frame_equal(got.reset_index(drop=True), candidate_pool(tmp_path).reset_index(drop=True),
                                  check_dtype=False)


def _stub_market(ts):
    return SimpleNamespace(results={f"alpha{i + 1:03d}": SimpleNamespace(summary={5: {"is": {"t_nw": t}}})
                                    for i, t in enumerate(ts)})


def test_top_drops_nan_before_sorting():
    cfg = _cfg()
    m = _stub_market([float("nan"), -3.0, 2.0, float("nan"), 3.0, 1.0])
    assert R._top(m, cfg, 10) == ["alpha002", "alpha005", "alpha003", "alpha006"]
    assert R._top(m, cfg, 2) == ["alpha002", "alpha005"]


def test_survivorship_since(built):
    run = _run(built)
    us = dict(run.markets["us"].survivorship, since="2005-01-01")
    run.markets = {**run.markets, "us": dataclasses.replace(run.markets["us"], survivorship=us)}
    head = render_markdown(run, {})[0].split("## Run metadata")[0]
    assert "since 2005" in head and "since 2015" not in head      # the pool's since, not cfg.start
    assert " since " not in render_markdown(_run(built), {})[0].split("## Run metadata")[0]   # absent: no clause


def test_heatmap_not_evaluated_is_muted(built, tmp_path):
    run = _run(built)
    name = R.fig_ic_heatmap(run, tmp_path)
    assert name == "ic_heatmap.png" and (tmp_path / name).stat().st_size > 0
    assert R.MUTED == "#898781"


def test_spread_stats_hand_case(monkeypatch):
    cfg = _cfg()
    idx = pd.DatetimeIndex(["2015-06-24", "2015-06-25", "2015-06-26", "2015-06-29", "2015-06-30", "2015-07-01"])
    gross = pd.Series([0.01, 0.02, -0.01, np.nan, 0.03, 0.5], index=idx)
    tidx = pd.DatetimeIndex(["2015-06-24", "2015-06-25", "2015-06-26", "2015-06-29", "2015-07-01"])
    res = SimpleNamespace(deciles=pd.DataFrame({"long_short": gross}), trades=None,
                          turnover=pd.Series([0.2, 0.5, 0.7, 0.3, 0.9], index=tidx))
    monkeypatch.setattr(R, "apply_costs", lambda g, tr, mk, c: g - 0.001)
    s = R.spread_stats(res, "us", CostModel(), cfg)
    assert s["gross_ann"] == pytest.approx(0.0125 * 252) and s["net_ann"] == pytest.approx(0.0115 * 252)
    assert s["turnover_mean"] == pytest.approx(0.425)
    assert s["turnover_gt50"] == pytest.approx(0.25)               # 0.5 itself is not > 0.5; 0.7 is


def test_periods_era_spanning_holdout():
    cfg = dataclasses.replace(_cfg(), eras=ERAS[:2] + (("span", "2015-05-01", "2015-08-31"),))
    assert [p[0] for p in periods(cfg)] == ["is", "early", "mid"]
    assert [p[0] for p in periods(dataclasses.replace(cfg, unlock_holdout=True))] == ["is", "early", "mid", "span"]


def test_benchmark_gap_ignores_pre_start_rows():
    T, g, gb, S = 120, 0.001, 0.0004, 40
    t = np.arange(T)
    u = 100 * (1 + g) ** t
    u[10:] *= 1.5                                       # a pre-start jump in the universe only
    p = _toy_panel(np.column_stack([u, u]))
    bench = pd.Series(10 * (1 + gb) ** t, index=p.dates)
    cfg = RunConfig(start=p.dates[S].strftime("%Y-%m-%d"), holdout_start="2030-01-01")
    row = benchmark_gap(p, bench, cfg, "equal").loc["is"]
    n = T - 2 - S
    months = len(p.dates[S:T - 2].to_period("M").unique())
    assert row["months"] == months
    assert row["universe"] == pytest.approx(((1 + g) ** n) ** (12 / months) - 1)
    assert row["benchmark"] == pytest.approx(((1 + gb) ** n) ** (12 / months) - 1)


def test_cum_ic_locked_plots_nothing_from_holdout(built, tmp_path, monkeypatch):
    import matplotlib.axes
    seen = []
    orig = matplotlib.axes.Axes.plot

    def spy(self, *a, **k):
        seen.append(pd.DatetimeIndex(a[0]))
        return orig(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
    run = _run(built)
    assert R.fig_cum_ic(run, run.markets["us"], tmp_path)
    assert seen and all(len(x) and x.max() < pd.Timestamp(run.cfg.holdout_start) for x in seen)
