"""SQL-shape and logic tests for persist.py; fake connections, no DB."""
import hashlib
import inspect
import json
import math
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha101 import persist
from alpha101.costs import CostModel
from alpha101.evaluate import AlphaResult, RunConfig

ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.lastrowid = conn.next_id

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.calls.append(("execute", sql, params))

    def executemany(self, sql, rows):
        self.conn.calls.append(("executemany", sql, list(rows)))

    def fetchall(self):
        return self.conn.select_rows


class FakeConn:
    def __init__(self, select_rows=(), next_id=42):
        self.calls = []
        self.commits = 0
        self.select_rows = list(select_rows)
        self.next_id = next_id

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def _design_ddls():
    text = (ROOT / "docs" / "design.md").read_text()
    sec = text.split("## 8. DDL", 1)[1]
    block = re.search(r"```sql\n(.*?)```", sec, re.S).group(1)
    stmts = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) .*?;", block, re.S)
    full = re.findall(r"(CREATE TABLE IF NOT EXISTS \w+ .*?;)", block, re.S)
    return dict(zip(stmts, full))


def _norm(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def test_ddl_matches_design():
    design = _design_ddls()
    assert list(design) == ["security_master", "universe_membership", "alpha_runs", "alpha_ic"]
    assert list(persist.DDL) == list(design)
    for name, ddl in persist.DDL.items():
        assert ddl.startswith("CREATE TABLE IF NOT EXISTS")
        assert _norm(ddl) == _norm(design[name])


def test_ensure_tables_executes_the_four_ddls():
    conn = FakeConn()
    persist.ensure_tables(conn)
    assert [c[1] for c in conn.calls] == list(persist.DDL.values())
    assert all(c[0] == "execute" for c in conn.calls)
    assert conn.commits == 1


def test_insert_statements_are_pure_placeholders():
    inserts = [v for k, v in vars(persist).items() if k.startswith("INSERT_") and k.endswith("_SQL")]
    assert len(inserts) == 4
    for sql in inserts:
        m = re.search(r"INSERT INTO \w+ \(([^)]*)\) VALUES \(([^)]*)\)", sql)
        cols = m.group(1).split(",")
        vals = m.group(2).split(",")
        assert all(v == "%s" for v in vals), sql
        assert len(cols) == len(vals), sql
    src = inspect.getsource(persist)
    assert not re.search(r"DELETE|DROP|TRUNCATE|ALTER", src)
    assert "shift(-" not in src


def test_new_run_id():
    now = datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc)
    rid = persist.new_run_id("ab12cd3ef456", now)
    assert rid == "20261020T0300Z-ab12cd3"
    assert len(rid) <= 40
    assert persist.new_run_id("ab12cd3ef456", datetime(2026, 10, 20, 3, 0)) == rid  # naive = UTC
    est = timezone(timedelta(hours=-5))
    assert persist.new_run_id("ab12cd3", datetime(2026, 10, 19, 22, 0, tzinfo=est)) == rid


def _clean(monkeypatch, sha="a" * 40):
    monkeypatch.setattr(persist, "git_state", lambda repo=None: (sha, False))
    monkeypatch.setattr(persist, "_path_is_committed", lambda p, repo=None: True)


def test_register_run_dirty_tree(monkeypatch):
    monkeypatch.setattr(persist, "git_state", lambda repo=None: ("a" * 40, True))
    conn = FakeConn()
    with pytest.raises(RuntimeError, match="uncommitted"):
        persist.register_run(conn, "r1", RunConfig(), CostModel(), {}, None)
    assert conn.calls == []


def test_register_run_unlock_needs_prereg(monkeypatch):
    _clean(monkeypatch)
    conn = FakeConn()
    with pytest.raises(ValueError):
        persist.register_run(conn, "r1", RunConfig(unlock_holdout=True), CostModel(), {}, None)
    assert conn.calls == []


def test_register_run_long_id(monkeypatch):
    _clean(monkeypatch)
    with pytest.raises(ValueError):
        persist.register_run(FakeConn(), "x" * 41, RunConfig(), CostModel(), {}, None)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "reports").mkdir()
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / "scripts" / "s.sh").write_text("echo\n")
    (r / "prereg.md").write_text("frozen plan\n")
    (r / ".gitignore").write_text("reports/\n")
    _git(r, "init", "-q")
    _git(r, "config", "user.name", "T")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_git_state_real(repo):
    sha, dirty = persist.git_state(repo)
    assert len(sha) == 40 and not dirty
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert persist.git_state(repo)[1]
    _git(repo, "checkout", "--", "src/a.py")
    assert not persist.git_state(repo)[1]
    (repo / "src" / "new.py").write_text("y = 1\n")
    assert persist.git_state(repo)[1]


def test_path_is_committed_real(repo, tmp_path):
    assert persist._path_is_committed((repo / "prereg.md").resolve(), repo)
    (repo / "prereg.md").write_text("changed\n")
    assert not persist._path_is_committed((repo / "prereg.md").resolve(), repo)
    (repo / "untracked.md").write_text("u\n")
    assert not persist._path_is_committed((repo / "untracked.md").resolve(), repo)
    (repo / "reports" / "prereg.md").write_text("ignored\n")
    assert not persist._path_is_committed((repo / "reports" / "prereg.md").resolve(), repo)
    outside = tmp_path / "outside.md"
    outside.write_text("o\n")
    assert not persist._path_is_committed(outside.resolve(), repo)


def test_register_run_prereg_hash_and_options(repo):
    conn = FakeConn()
    persist.register_run(conn, "r1", RunConfig(unlock_holdout=True), CostModel(), {"n": 1},
                         repo / "prereg.md", repo=repo)
    (kind, sql, params), = conn.calls
    assert sql == persist.INSERT_RUN_SQL and conn.commits == 1
    run_id, sha, script, options_json, fp, prereg, unlocked = params
    assert (run_id, script) == ("r1", persist.SCRIPT) and len(sha) == 40
    assert prereg == hashlib.sha256(b"frozen plan\n").hexdigest()
    assert unlocked == 1
    opts = json.loads(options_json)
    assert "argmax_convention" in opts["run_config"]["options"]
    assert "half_spread_bps" in opts["costs"]
    assert json.loads(fp) == {"n": 1}


@pytest.mark.parametrize("kind", ["modified", "untracked", "ignored", "outside"])
def test_register_run_rejects_uncommitted_prereg(repo, tmp_path, kind):
    if kind == "modified":
        pre = repo / "prereg.md"
        pre.write_text("changed\n")
    elif kind == "untracked":
        pre = repo / "u.md"
        pre.write_text("u\n")
    elif kind == "ignored":
        pre = repo / "reports" / "prereg.md"
        pre.write_text("i\n")
    else:
        pre = tmp_path / "outside.md"
        pre.write_text("o\n")
    conn = FakeConn()
    with pytest.raises(RuntimeError, match="not committed"):
        persist.register_run(conn, "r1", RunConfig(unlock_holdout=True), CostModel(), {}, pre, repo=repo)
    assert conn.calls == []


def test_register_run_relative_prereg_resolved_once(repo, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    (other / "prereg.md").write_text("DIFFERENT\n")  # same relative name as the committed repo file
    monkeypatch.chdir(other)
    conn = FakeConn()
    with pytest.raises(RuntimeError, match="not committed"):
        persist.register_run(conn, "r1", RunConfig(unlock_holdout=True), CostModel(), {},
                             Path("prereg.md"), repo=repo)
    assert conn.calls == []


def test_register_run_dirty_repo_real(repo):
    (repo / "src" / "new.py").write_text("y = 1\n")
    conn = FakeConn()
    with pytest.raises(RuntimeError, match="uncommitted"):
        persist.register_run(conn, "r1", RunConfig(), CostModel(), {}, None, repo=repo)
    assert conn.calls == []


def test_register_run_text_guard(monkeypatch):
    _clean(monkeypatch)
    with pytest.raises(ValueError, match="panel_fingerprint"):
        persist.register_run(FakeConn(), "r1", RunConfig(), CostModel(), {"k": "x" * 70000}, None)


def test_register_run_nan_is_null(monkeypatch):
    _clean(monkeypatch)
    conn = FakeConn()
    persist.register_run(conn, "r1", RunConfig(), CostModel(), {"a": float("nan"), "b": np.float64("inf")}, None)
    fp = conn.calls[0][2][4]
    assert "NaN" not in fp and "Infinity" not in fp and json.loads(fp) == {"a": None, "b": None}


def test_run_insert_is_not_an_upsert():
    assert "ON DUPLICATE KEY" not in persist.INSERT_RUN_SQL.upper()


def test_register_run_locked_flag(monkeypatch):
    _clean(monkeypatch)
    conn = FakeConn()
    persist.register_run(conn, "r1", RunConfig(), CostModel(), {}, None)
    assert conn.calls[0][2][5] is None and conn.calls[0][2][6] == 0


def test_write_ic_values_and_chunks():
    idx = pd.date_range("2010-01-01", periods=3)
    ic = pd.DataFrame({"ic": [0.1, np.nan, np.inf], "n": [10.0, np.nan, 5.0]}, index=idx)
    conn = FakeConn()
    assert persist.write_ic(conn, "r1", "alpha001", "ca", 5, ic) == 3
    rows = conn.calls[0][2]
    assert rows[0] == ("r1", "alpha001", "CA", 5, idx[0].date(), 0.1, 10)
    assert rows[1][5] is None and rows[1][6] == 0
    assert rows[2][5] is None and rows[2][6] == 5
    assert conn.calls[0][1] == persist.INSERT_IC_SQL

    big = pd.DataFrame({"ic": 0.0, "n": 1.0}, index=pd.date_range("1990-01-01", periods=12000))
    conn = FakeConn()
    assert persist.write_ic(conn, "r1", "alpha001", "us", 1, big) == 12000
    assert [c[0] for c in conn.calls] == ["executemany"] * 3
    assert [len(c[2]) for c in conn.calls] == [5000, 5000, 2000]
    assert conn.commits == 3


def test_write_ic_validation():
    ic = pd.DataFrame({"ic": [0.1], "n": [1]}, index=pd.date_range("2010-01-01", periods=1))
    with pytest.raises(ValueError):
        persist.write_ic(FakeConn(), "r1", "a" * 41, "us", 5, ic)
    with pytest.raises(ValueError):
        persist.write_ic(FakeConn(), "r1", "alpha001", "uk", 5, ic)


CM = CostModel(commission=0.0, half_spread_bps={"us": 0.0, "ca": 10.0}, borrow_bps_per_year=0.0)


def _result(gross, traded, ntr, alpha_id="alpha012"):
    idx = pd.date_range("2010-01-04", periods=len(gross), freq="B")
    deciles = pd.DataFrame({"long_short": gross}, index=idx)
    trades = pd.DataFrame({"traded": traded, "short": 0.0, "n_trades": ntr}, index=idx)
    summ = {5: {"is": {"mean_ic": 0.0123456, "ic_ir": 0.1, "t_nw": 2.345, "hit_rate": 0.5, "n_days": 100}}}
    return AlphaResult(alpha_id=alpha_id, expression="x", market="ca", basis="raw", ic={}, summary=summ,
                       deciles=deciles, turnover=pd.Series(dtype=float), trades=trades)


def test_write_summary_numbers():
    gross = [0.01, 0.02, np.nan, -0.01, 0.0]
    res = _result(gross, [1, 0, 0, 0, 0], [3, 2, 7, 1, 0])
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    rowid = persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(), CM, "liquidity_top200_ca",
                                  "out/report.md")
    assert rowid == 42 and conn.commits == 1
    sel, ins = conn.calls
    assert sel[1] == persist.SELECT_RUN_SQL and sel[2] == ("r1",)
    assert ins[1] == persist.INSERT_SUMMARY_SQL
    (name, start, end, univ, params, ann, sharpe, trades, status, notes, path) = ins[2]
    net = np.array([0.009, 0.02, -0.01, 0.0])
    assert name == "alpha101_012_ca_h5"
    assert start == pd.Timestamp("2010-01-04").date() and end == pd.Timestamp("2010-01-08").date()
    assert ann == round(net.mean() * 252, 4)
    assert abs(ann - 1.197) < 1e-9
    assert sharpe == round(net.mean() / net.std(ddof=1) * math.sqrt(252), 4)
    assert trades == 3 + 2 + 1 + 0
    assert status == "pending" and univ == "liquidity_top200_ca" and path == "out/report.md"
    assert notes == "IS h=5: mean IC 0.0123, NW t 2.35, 100 days; net decile spread annualised 1.1970"
    p = json.loads(params)
    assert p["run_id"] == "r1" and p["git_sha"] == "c" * 40 and p["script"] == persist.SCRIPT
    assert "argmax_convention" in p["options"] and p["options"]["primary_horizon"] == 5
    assert p["options"]["unlock_holdout"] is False and "half_spread_bps" in p["costs"]


def test_write_summary_none_cases():
    # constant net return: zero std -> sharpe NULL; huge annualised value -> NULL
    res = _result([1e4, 1e4, 1e4], [0, 0, 0], [1, 1, 1], "expr_0123456789")
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    persist.write_summary(conn, "r1", "expr_0123456789", "ca", res, RunConfig(), CM, "u")
    (name, _s, _e, _u, _p, ann, sharpe, trades, _st, notes, path) = conn.calls[1][2]
    assert name == "alpha101_expr_0123456789_ca_h5"
    assert ann is None and sharpe is None and path is None and trades == 3
    assert "annualised 2520000.0000" in notes


def test_write_summary_nan_notes():
    res = _result([0.01, 0.02], [0, 0], [1, 1])
    res.summary[5]["is"]["t_nw"] = float("nan")
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(), CM, "u")
    assert "NW t nan" in conn.calls[1][2][9]


def test_dec_rounds_before_range_check():
    assert persist._dec(999999.99996) is None
    assert persist._dec(-999999.99996) is None
    assert persist._dec(999999.9999) == 999999.9999
    assert persist._dec(float("nan")) is None and persist._dec(None) is None


def test_write_summary_holdout_flag_mismatch():
    res = _result([0.01, 0.02], [0, 0], [1, 1])
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    with pytest.raises(RuntimeError, match="holdout"):
        persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(unlock_holdout=True), CM, "u")
    conn = FakeConn(select_rows=[("c" * 40, 1)])
    with pytest.raises(RuntimeError, match="holdout"):
        persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(), CM, "u")
    assert len(conn.calls) == 1
    conn = FakeConn(select_rows=[("c" * 40, 1)])
    persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(unlock_holdout=True), CM, "u")
    assert len(conn.calls) == 2


def test_write_summary_params_size_guard():
    res = _result([0.01, 0.02], [0, 0], [1, 1])
    big = CostModel(commission=0.0, half_spread_bps={"us": 0.0, "ca": 10.0, "k" * 70000: 1.0},
                    borrow_bps_per_year=0.0)
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    with pytest.raises(ValueError, match="params"):
        persist.write_summary(conn, "r1", "alpha012", "ca", res, RunConfig(), big, "u")
    assert len(conn.calls) == 1


def test_write_summary_errors():
    res = _result([0.01, 0.02], [0, 0], [1, 1])
    with pytest.raises(RuntimeError):
        persist.write_summary(FakeConn(select_rows=[]), "r1", "alpha012", "ca", res, RunConfig(), CM, "u")
    allnan = _result([np.nan, np.nan], [0, 0], [1, 1])
    conn = FakeConn(select_rows=[("c" * 40, 0)])
    with pytest.raises(ValueError):
        persist.write_summary(conn, "r1", "alpha012", "ca", allnan, RunConfig(), CM, "u")
    assert len(conn.calls) == 1  # only the SELECT


def test_write_hypothesis():
    conn = FakeConn(next_id=7)
    rid = persist.write_hypothesis(conn, "h", "c", "ic_nw", {"b": 1, "a": float("nan")}, "supported", 3)
    assert rid == 7 and conn.commits == 1
    sql, params = conn.calls[0][1:]
    assert sql == persist.INSERT_HYPOTHESIS_SQL
    assert params[3] == '{"a": null, "b": 1}' and params[4:] == ("supported", 3)
    assert json.loads(params[3]) == {"a": None, "b": 1}
    with pytest.raises(ValueError):
        persist.write_hypothesis(FakeConn(), "h", "c", "m", {}, "maybe", None)
    with pytest.raises(ValueError):
        persist.write_hypothesis(FakeConn(), "h", "c", "m" * 101, {}, "rejected", None)


def test_json_numpy_scalars_become_python_types():
    out = persist._json({"a": np.int64(5), "b": np.bool_(True), "c": np.float64("nan")})
    assert out == '{"a": 5, "b": true, "c": null}'


def test_git_state_scripts_change_is_dirty(repo):
    assert not persist.git_state(repo)[1]
    (repo / "scripts" / "s.sh").write_text("echo changed\n")
    assert persist.git_state(repo)[1]


def test_register_run_options_json_size_guard(monkeypatch):
    _clean(monkeypatch)
    cfg = RunConfig(min_names={"us": 30, "ca": 20, "k" * 70000: 1})
    conn = FakeConn()
    with pytest.raises(ValueError, match="options_json"):
        persist.register_run(conn, "r1", cfg, CostModel(), {}, None)
    assert conn.calls == []


def test_register_run_stored_sha_is_head(repo):
    conn = FakeConn()
    persist.register_run(conn, "r1", RunConfig(), CostModel(), {}, None, repo=repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    assert conn.calls[0][2][1] == head


def test_register_run_relative_prereg_hashes_resolved_file(repo, monkeypatch):
    monkeypatch.chdir(repo)
    conn = FakeConn()
    persist.register_run(conn, "r1", RunConfig(unlock_holdout=True), CostModel(), {}, Path("prereg.md"),
                         repo=repo)
    assert conn.calls[0][2][5] == hashlib.sha256(b"frozen plan\n").hexdigest()
