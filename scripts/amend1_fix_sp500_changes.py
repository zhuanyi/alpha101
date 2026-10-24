"""One-off amend-1 fix: rewrite exactly 3 stale keys of source `sp500_wiki_changes` in universe_membership.

Scoped DELETE: the AT&T / Ingersoll-Rand ticker-reuse rows written before the
walk-back anomaly rule. Steps: rebuild the corrected events from the Wikipedia snapshot CSVs, diff them against
the stored rows (must be EXACTLY the known 3-key difference, else exit 1 without writing), back up the stored
rows of the 3 keys to JSON, then (only with --execute) delete those keys and upsert the corrected rows in one
transaction and re-verify. Without --execute nothing is written to the database.

    PYTHONPATH=src python3 scripts/amend1_fix_sp500_changes.py --current CSV --changes CSV --backup JSON [--execute]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from alpha101.data import universe
from alpha101.db import reader, writer

SOURCE = "sp500_wiki_changes"
SINCE = "2005-01-01"
DB_ONLY = {("IR", dt.date(2010, 11, 17), 1), ("T", dt.date(2005, 11, 18), 0)}
NEW_ONLY = {("IR", dt.date(2020, 3, 2), 1)}
KEYS = [("IR", dt.date(2010, 11, 17)), ("T", dt.date(2005, 11, 18)), ("IR", dt.date(2020, 3, 2))]
COLUMNS = ["symbol", "market", "source", "as_of", "in_universe", "created_at"]
KEY_COND = " OR ".join(["(symbol=%s AND as_of=%s)"] * len(KEYS))
DELETE_SQL = f"DELETE FROM universe_membership WHERE source=%s AND ({KEY_COND})"
SELECT_SQL = f"SELECT {', '.join(COLUMNS)} FROM universe_membership WHERE source=%s AND ({KEY_COND})"
KEY_PARAMS = [SOURCE] + [x for k in KEYS for x in k]


def triples(rows) -> set:
    return {(str(s), pd.Timestamp(d).date(), int(v)) for s, d, v in rows}


def read_stored(conn) -> list:
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, as_of, in_universe FROM universe_membership WHERE source=%s", [SOURCE])
        return list(cur.fetchall())


def corrected_events(current_csv, changes_csv, stored_symbols) -> pd.DataFrame:
    ev = universe.index_membership_events(pd.read_csv(current_csv), pd.read_csv(changes_csv), "us", SOURCE, SINCE)
    ev = ev[ev["symbol"].isin(set(stored_symbols))].copy()
    ev["as_of"] = [pd.Timestamp(d).date() for d in ev["as_of"]]
    return ev.reset_index(drop=True)


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--current", required=True)
    ap.add_argument("--changes", required=True)
    ap.add_argument("--backup", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    bpath = Path(a.backup)
    if a.execute and bpath.exists():
        print(f"refusing to overwrite existing backup {bpath}")
        return 1

    with reader() as conn:
        stored = read_stored(conn)
        if not stored:
            print(f"no stored rows for source {SOURCE!r}")
            return 1
        ev = corrected_events(a.current, a.changes, {r[0] for r in stored})
        db_t, new_t = triples(stored), triples(zip(ev["symbol"], ev["as_of"], ev["in_universe"]))
        db_only, new_only = db_t - new_t, new_t - db_t
        if db_only != DB_ONLY or new_only != NEW_ONLY:
            print("unexpected diff, nothing written")
            print("db_only :", sorted(db_only, key=str))
            print("new_only:", sorted(new_only, key=str))
            return 1
        before = len(stored)
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL, KEY_PARAMS)
            backup = [dict(zip(COLUMNS, [_iso(v) for v in r])) for r in cur.fetchall()]

    keyset = set(KEYS)
    fix = ev[[(s, d) in keyset for s, d in zip(ev["symbol"], ev["as_of"])]]
    inserts = [(str(r.symbol), str(r.market), str(r.source), r.as_of, int(r.in_universe))
               for r in fix.itertuples(index=False)]
    print("planned:", DELETE_SQL, KEY_PARAMS)
    print("planned upserts:", inserts)
    if not a.execute:
        print(f"would back up {len(backup)} stored rows of the 3 keys to {bpath}:")
        print(json.dumps(backup, indent=1))
        print("dry run: nothing written (use --execute)")
        return 0

    bpath.write_text(json.dumps(backup, indent=1))
    print(f"backed up {len(backup)} stored rows of the 3 keys to {bpath}")
    conn = writer()
    try:
        with conn.cursor() as cur:
            cur.execute(DELETE_SQL, KEY_PARAMS)
            if cur.rowcount != len(backup):
                conn.rollback()
                print(f"deleted {cur.rowcount} rows, expected {len(backup)}: rolled back")
                return 1
            cur.executemany(universe.UPSERT_SQL, inserts)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    with reader() as conn:
        after = read_stored(conn)
    ok = triples(after) == new_t
    print(f"rows before {before}, after {len(after)}; stored == corrected: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
