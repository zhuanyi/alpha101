"""Offline tests for cli.py: synthetic panels, fake readers/writers, no DB, no network."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha101 import cli, persist
from alpha101.data.yahoo import COLUMNS
from alpha101.evaluate import RunConfig
from alpha101.synthetic import random_panel

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


class FakeCur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))

    def fetchall(self):
        return self.conn.rows


class FakeConn:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return FakeCur(self)

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch):
    """Patch every outside dependency of `run`; returns a namespace recording what happened."""
    ns = type("Env", (), {})()
    ns.events, ns.cfgs, ns.loaded = [], [], []
    ns.baseline_rows, ns.dirty, ns.fail, ns.conns, ns.T = [], False, set(), [], 300

    def load_panel(m, start, basis="raw", universe_source=None, **kw):
        ns.loaded.append((m, start, basis, universe_source))
        return random_panel(T=ns.T, N=40, seed=3)

    monkeypatch.setattr(cli, "load_panel", load_panel)

    def reader():
        ns.conns.append(FakeConn(ns.baseline_rows))
        return ns.conns[-1]

    monkeypatch.setattr(cli, "reader", reader)
    monkeypatch.setattr(cli, "writer", lambda: FakeConn())
    monkeypatch.setattr(cli, "db_fingerprint", lambda conn, m: {"rows": 10, "symbols": 3, "max_fetched_at": "x"})
    monkeypatch.setattr(cli, "read_long", lambda conn, symbols=None, start=None: pd.DataFrame(columns=COLUMNS))
    monkeypatch.setattr(cli.universe, "resolve_source", lambda conn, s: s + "@20261019" if "@" not in s else s)
    monkeypatch.setattr(cli.universe, "source_symbols", lambda conn, s: ["S000"])
    monkeypatch.setattr(cli.constituents, "pool_from_raw", lambda raw_dir, since="2005-01-01": pd.DataFrame())
    monkeypatch.setattr(cli.report_mod, "survivorship_counts", lambda conn, pool, m: {})
    monkeypatch.setattr(cli.report_mod, "screen_summary", lambda *a, **k: {})
    monkeypatch.setattr(persist, "git_state", lambda repo=None: (SHA, ns.dirty))
    monkeypatch.setattr(persist, "ensure_tables", lambda conn: ns.events.append("ensure"))

    def register_run(conn, run_id, cfg, costs, fps, prereg, **kw):
        if ns.dirty:
            raise RuntimeError("uncommitted changes under src/ or scripts/")
        ns.cfgs.append(cfg)
        ns.events.append("register")

    def write_ic(conn, run_id, alpha_id, market, horizon, ic):
        ns.events.append(("ic", alpha_id, market, horizon))
        return len(ic)

    def write_summary(conn, run_id, alpha_id, market, result, cfg, costs, universe, report_path=None):
        ns.events.append(("summary", alpha_id, universe))
        return 1

    monkeypatch.setattr(persist, "register_run", register_run)
    monkeypatch.setattr(persist, "write_ic", write_ic)
    monkeypatch.setattr(persist, "write_summary", write_summary)

    real_eval = cli.evaluate_alpha

    def evaluate_alpha(panel, i, cfg, out_dir, market=None):
        if i in ns.fail:
            raise ValueError("boom")
        if not ns.cfgs or ns.cfgs[-1] is not cfg:
            ns.cfgs.append(cfg)
        return real_eval(panel, i, cfg, out_dir, market=market)

    monkeypatch.setattr(cli, "evaluate_alpha", evaluate_alpha)
    return ns


BASE = ["run", "--start", "2015-04-01", "--horizons", "1,5"]


def test_unlock_without_prereg_exits_2_before_loading(env, tmp_path):
    assert cli.main(["run", "--alphas", "1", "--unlock-holdout", "--out", str(tmp_path)]) == 2
    assert env.loaded == [] and env.events == []


def test_bad_arguments_exit_2(env, tmp_path):
    out = ["--out", str(tmp_path), "--no-db"]
    assert cli.main([*BASE, "--alphas", "1", "--universe", "X", "--markets", "us,ca", *out]) == 2
    assert cli.main([*BASE, "--alphas", "0,abc", *out]) == 2
    assert cli.main([*BASE, "--alphas", "1,500", *out]) == 2
    assert cli.main([*BASE, "--alphas", "1", "--markets", "uk", *out]) == 2
    assert env.loaded == []


def test_run_no_db(env, tmp_path, capsys):
    rc = cli.main([*BASE, "--alphas", "1,6", "--markets", "ca", "--no-db", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    run_id = [ln for ln in out.splitlines() if ln.startswith("run_id=")][0][len("run_id="):]
    report = tmp_path / run_id / "report.md"
    assert report.exists() and f"report={report}" in out
    assert "backtest_results_rows=0" in out and "failures=0" in out
    assert not any(e in ("ensure", "register") or isinstance(e, tuple) for e in env.events)
    assert "not registered" in report.read_text()
    assert (tmp_path / run_id / "alpha_values" / "ca" / "alpha001.parquet").exists()


def test_horizon_is_added_and_primary(env, tmp_path):
    rc = cli.main(["run", "--start", "2015-04-01", "--alphas", "1", "--markets", "ca", "--no-db",
                   "--horizon", "3", "--horizons", "1,5", "--out", str(tmp_path)])
    assert rc == 0
    cfg = env.cfgs[0]
    assert 3 in cfg.horizons and cfg.primary_horizon == 3 and set(cfg.horizons) == {1, 3, 5}


def test_db_path_order_counts_and_failure(env, tmp_path, capsys):
    env.fail = {6}
    rc = cli.main([*BASE, "--alphas", "1,6,12", "--markets", "ca", "--out", str(tmp_path)])
    assert rc == 1                    # a failed alpha makes the exit code non-zero; the report is still written
    ev = env.events
    assert ev[0] == "ensure" and ev[1] == "register"
    ics = [e for e in ev if isinstance(e, tuple) and e[0] == "ic"]
    sums = [e for e in ev if isinstance(e, tuple) and e[0] == "summary"]
    assert len(ics) == 2 * 2          # alphas evaluated x horizons {1, 5}
    assert len(sums) == 2 and {s[1] for s in sums} == {"alpha001", "alpha012"}
    assert sums[0][2] == "liquidity_top200_ca@20261019"
    assert ev.index("register") < ev.index(ics[0])
    out = capsys.readouterr().out
    assert "failures=1" in out and "backtest_results_rows=2" in out
    md = next(tmp_path.glob("*/report.md")).read_text()
    assert "alpha006: ValueError: boom" in md


def test_dirty_tree_with_db_exits_1(env, tmp_path):
    env.dirty = True
    assert cli.main([*BASE, "--alphas", "1", "--markets", "ca", "--out", str(tmp_path)]) == 1
    assert env.loaded == []
    env.dirty = True
    assert cli.main([*BASE, "--alphas", "1", "--markets", "ca", "--no-db", "--out", str(tmp_path)]) == 0
    assert "dirty tree, not registered" in next(tmp_path.glob("*/report.md")).read_text()


def test_unlock_without_db_exits_2(env, tmp_path):
    assert cli.main([*BASE, "--alphas", "1", "--no-db", "--unlock-holdout", "--prereg", "p.md",
                     "--out", str(tmp_path)]) == 2


def test_list_prints_101_lines(capsys):
    assert cli.main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 101 and "max_lookback_days=" in lines[0]


def test_expr_prints_is_table_only(env, capsys):
    assert cli.main(["expr", "rank(close)", "--market", "ca", "--start", "2015-04-01"]) == 0
    out = capsys.readouterr().out
    assert "mean IC" in out and "2019-2026" not in out and "holdout" not in out.lower()
    assert len([ln for ln in out.splitlines() if ln.strip()[:1].isdigit()]) == 4   # horizons 1, 5, 10, 21


def test_baseline_paired_difference(env, tmp_path):
    env.T = 1200                      # the panel runs past holdout_start (2019-01-01)
    p = random_panel(T=1200, N=40, seed=3)
    cfg = RunConfig(markets=("ca",), horizons=(1, 5), start="2015-04-01")
    from alpha101.evaluate import evaluate_alpha
    r = evaluate_alpha(p, 1, cfg, None, market="ca")
    ic = r.ic[5]["ic"].dropna()
    d = ic.index[:50]
    # baseline = this IC shifted by a known +0.01 on 40 dates, plus 10 dates only the baseline has
    rows = [("alpha001", "CA", 5, ts.date(), float(ic.loc[ts]) + 0.01) for ts in d[:40]]
    rows += [("alpha001", "CA", 5, (pd.Timestamp("2014-01-01") + pd.Timedelta(days=i)).date(), 0.5)
             for i in range(10)]
    rows += [("alpha999", "CA", 5, d[0].date(), 0.3)]
    late = r.ic[5].index[r.ic[5].index >= pd.Timestamp(cfg.holdout_start)][:20]
    assert len(late) == 20
    rows += [("alpha001", "CA", 5, ts.date(), 0.4) for ts in late]      # holdout dates: must not count
    env.baseline_rows = rows
    rc = cli.main([*BASE, "--alphas", "1", "--markets", "ca", "--no-db", "--baseline-run", "RUN0",
                   "--out", str(tmp_path)])
    assert rc == 0
    md = next(tmp_path.glob("*/report.md")).read_text()
    assert "Baseline run: `RUN0`" in md
    assert "-0.0100" in md          # variant minus baseline = -0.01 on the 40 common days
    assert "| ca | alpha001 |" in md and "| 40 |" in md
    sql, params = next(q for c in env.conns for q in c.executed if "alpha_ic" in q[0])
    assert "date < %s" in sql and params == ["RUN0", 5, cfg.holdout_start]


def test_baseline_query_unfiltered_when_unlocked(env):
    env.baseline_rows = [("alpha001", "CA", 5, pd.Timestamp("2015-05-01").date(), 0.1)]
    cli._read_baseline("RUN0", 5)
    sql, params = env.conns[-1].executed[-1]
    assert "date <" not in sql and params == ["RUN0", 5]


def test_expr_prints_only_is_values(env, monkeypatch, capsys):
    def stat(v):
        return {"mean_ic": v, "t_nw": v, "n_days": 7}
    summary = {h: {"is": stat(0.1111), "2019-2022": stat(0.7777), "2023-2026": stat(0.8888),
                   "holdout": stat(0.9999)} for h in (1, 5)}
    res = type("R", (), {"alpha_id": "expr", "expression": "rank(close)", "summary": summary})()
    monkeypatch.setattr(cli, "evaluate_alpha", lambda *a, **k: res)
    assert cli.main(["expr", "rank(close)", "--market", "ca"]) == 0
    out = capsys.readouterr().out
    assert "0.1111" in out
    assert not any(x in out for x in ("0.7777", "0.8888", "0.9999"))


def test_n_eff_counts_failed_alpha(env, monkeypatch, tmp_path):
    env.fail = {6}
    seen = {}
    real_mt, real_corr = cli.multiple_testing, cli.ic_correlation

    def mt(summ, corr):
        seen["summ"] = summ
        return real_mt(summ, corr)

    def corr(ics, **kw):
        seen["ics"] = ics
        return real_corr(ics, **kw)

    monkeypatch.setattr(cli, "multiple_testing", mt)
    monkeypatch.setattr(cli, "ic_correlation", corr)
    cli.main([*BASE, "--alphas", "1,6", "--markets", "ca", "--no-db", "--out", str(tmp_path)])
    assert set(seen["summ"].index) == {"alpha001", "alpha006"}          # one row per attempted alpha
    assert np.isnan(seen["summ"].loc["alpha006", "t_nw"]) and not np.isnan(seen["summ"].loc["alpha001", "t_nw"])
    assert set(seen["ics"]) == {"alpha001", "alpha006"} and seen["ics"]["alpha006"].isna().all()


def test_uncaught_exception_exits_1_with_traceback(env, monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise RuntimeError("panel exploded")

    monkeypatch.setattr(cli, "load_panel", boom)
    assert cli.main([*BASE, "--alphas", "1", "--markets", "ca", "--no-db", "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "RuntimeError: panel exploded" in err


def test_alphas_zero_exits_2(env, tmp_path):
    assert cli.main([*BASE, "--alphas", "0", "--no-db", "--out", str(tmp_path)]) == 2
    assert env.loaded == []


def test_paired_unit():
    idx = pd.bdate_range("2015-01-05", periods=10)
    r = type("R", (), {"ic": {5: pd.DataFrame({"ic": np.arange(10) * 0.01, "n": 30}, index=idx)}})()
    base = pd.DataFrame({"alpha_id": "alpha001", "market": "CA", "horizon": 5, "date": idx[:8],
                         "ic": np.arange(8) * 0.01 - 0.02})
    cfg = RunConfig(markets=("ca",), primary_horizon=5, holdout_start="2015-01-12")
    out = cli._paired({"ca": {"alpha001": r}}, base, cfg)
    row = out.iloc[0]
    assert row["n_days"] == 5 and row["mean_diff"] == pytest.approx(0.02)   # idx[:5] < 2015-01-12


def test_help_exits_0():
    p = subprocess.run([sys.executable, "-m", "alpha101", "--help"], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0 and "usage" in p.stdout.lower()
