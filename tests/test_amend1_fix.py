"""amend-1 AT&T fix script: fake connections only, no database."""
import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "amend1_fix", Path(__file__).resolve().parents[1] / "scripts" / "amend1_fix_sp500_changes.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)

D = dt.date


def _csvs(tmp_path):
    cur = pd.DataFrame({"symbol": ["IR", "T", "AAA"]})
    ch = pd.DataFrame({"date": ["2020-03-02", "2010-11-17", "2005-11-18"], "added": ["IR", "", ""],
                       "removed": ["", "", ""]})
    cur.to_csv(tmp_path / "cur.csv", index=False)
    ch.to_csv(tmp_path / "ch.csv", index=False)
    return str(tmp_path / "cur.csv"), str(tmp_path / "ch.csv")


class DB:
    def __init__(self, rows, rowcount=2):
        self.rows, self.rowcount, self.executed, self.many = rows, rowcount, [], []
        self.commits = self.rollbacks = 0

    def cursor(self):
        return Cur(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Cur:
    def __init__(self, db):
        self.db, self.out = db, []

    @property
    def rowcount(self):
        return self.db.rowcount

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.db.executed.append((sql, params))
        if sql.startswith("SELECT symbol, as_of, in_universe"):
            self.out = list(self.db.rows)
        elif sql.startswith("SELECT symbol, market"):
            self.out = [("IR", "US", fix.SOURCE, D(2010, 11, 17), 1, dt.datetime(2026, 1, 1)),
                        ("T", "US", fix.SOURCE, D(2005, 11, 18), 0, dt.datetime(2026, 1, 1))]

    def executemany(self, sql, data):
        self.db.many.append((sql, list(data)))

    def fetchall(self):
        return self.out


STORED = [("IR", D(2005, 1, 3), 1), ("IR", D(2010, 11, 17), 1), ("T", D(2005, 11, 18), 0), ("AAA", D(2005, 1, 3), 1)]


def _setup(monkeypatch, rdb, wdb):
    monkeypatch.setattr(fix, "reader", lambda: rdb)
    monkeypatch.setattr(fix, "writer", lambda: wdb)
    # the corrected events: stored minus the two stale rows, plus the new IR row
    ev = pd.DataFrame({"symbol": ["IR", "IR", "AAA"], "market": "US", "source": fix.SOURCE,
                       "as_of": [D(2005, 1, 3), D(2020, 3, 2), D(2005, 1, 3)], "in_universe": [1, 1, 1]})
    monkeypatch.setattr(fix.universe, "index_membership_events", lambda *a, **k: ev)


def _args(tmp_path, *extra):
    c, ch = _csvs(tmp_path)
    return ["--current", c, "--changes", ch, "--backup", str(tmp_path / "bk.json"), *extra]


def test_wrong_diff_writes_nothing(monkeypatch, tmp_path):
    rdb, wdb = DB(STORED + [("ZZZ", D(2006, 1, 1), 1)]), DB([])
    _setup(monkeypatch, rdb, wdb)
    assert fix.main(_args(tmp_path, "--execute")) == 1
    assert wdb.executed == [] and not (tmp_path / "bk.json").exists()


def test_dry_run_deletes_nothing(monkeypatch, tmp_path, capsys):
    rdb, wdb = DB(STORED), DB([])
    _setup(monkeypatch, rdb, wdb)
    assert fix.main(_args(tmp_path)) == 0
    assert wdb.executed == [] and wdb.many == []
    assert not any(s.startswith("DELETE") for s, _ in rdb.executed)
    assert not (tmp_path / "bk.json").exists()
    out = capsys.readouterr().out
    assert "dry run" in out and "would back up 2" in out


def test_existing_backup_not_overwritten(monkeypatch, tmp_path):
    rdb, wdb = DB(STORED), DB([])
    _setup(monkeypatch, rdb, wdb)
    (tmp_path / "bk.json").write_text("keep")
    assert fix.main(_args(tmp_path, "--execute")) == 1
    assert rdb.executed == [] and wdb.executed == []
    assert (tmp_path / "bk.json").read_text() == "keep"


def test_execute_single_scoped_delete_and_upsert(monkeypatch, tmp_path):
    after = [("IR", D(2005, 1, 3), 1), ("IR", D(2020, 3, 2), 1), ("AAA", D(2005, 1, 3), 1)]
    rdb, wdb = DB(STORED), DB([], rowcount=2)
    reads = iter([STORED, after])
    monkeypatch.setattr(fix, "read_stored", lambda conn: next(reads))
    _setup(monkeypatch, rdb, wdb)
    assert fix.main(_args(tmp_path, "--execute")) == 0
    dels = [(s, p) for s, p in wdb.executed if s.startswith("DELETE")]
    assert len(dels) == 1
    sql, params = dels[0]
    assert params == [fix.SOURCE, "IR", D(2010, 11, 17), "T", D(2005, 11, 18), "IR", D(2020, 3, 2)]
    assert sql.count("%s") == 7 and "source=%s" in sql
    assert wdb.commits == 1 and wdb.rollbacks == 0
    assert len(fix.json.loads((tmp_path / "bk.json").read_text())) == 2
    assert wdb.many[0][0] == fix.universe.UPSERT_SQL
    assert wdb.many[0][1] == [("IR", "US", fix.SOURCE, D(2020, 3, 2), 1)]


def test_rowcount_mismatch_rolls_back(monkeypatch, tmp_path):
    rdb, wdb = DB(STORED), DB([], rowcount=3)
    _setup(monkeypatch, rdb, wdb)
    assert fix.main(_args(tmp_path, "--execute")) == 1
    assert wdb.rollbacks == 1 and wdb.commits == 0 and wdb.many == []
