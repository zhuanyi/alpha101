"""CLI: fetch XIC/XCS daily files in the +-10 trading-day window of every quarterly review, 2012-2025.

Reuses taxloss.ishares.fetch (1.1 s gap, backoff, cache, .missing markers). Every miss is logged. Resumable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "taxloss"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import ishares  # noqa: E402
import reviews  # noqa: E402

RAW = HERE / "data" / "raw" / "ishares"
log = logging.getLogger("tsxpanel.fetch")


def run(plan: pd.DataFrame, raw_dir: Path = RAW, session=None) -> pd.DataFrame:
    import requests
    session = session or requests.Session()
    out = []
    for i, r in enumerate(plan.itertuples(index=False)):
        try:
            res = ishares.fetch(r.etf, r.date, session, raw_dir)
        except ishares.Blocked as e:
            log.error("BLOCKED at %s %s: %s", r.etf, r.date, e)
            out.append({**r._asdict(), "status": "blocked", "from_cache": False})
            break
        if res["status"] != "present":
            log.warning("MISS %s %s status=%s", r.etf, r.date, res["status"])
        out.append({**r._asdict(), "status": res["status"], "from_cache": res["from_cache"]})
        if i % 100 == 0:
            log.info("%d/%d", i, len(plan))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "window_fetch_status.csv"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from taxcal import Calendar, tsx_trading_days
    cal = Calendar(tsx_trading_days())
    plan = reviews.fetch_plan(cal)
    # cheap ones first is irrelevant; sort so the log reads chronologically
    plan = plan.sort_values(["date", "etf"]).reset_index(drop=True)
    log.info("plan: %d requests, %d already cached", len(plan),
             sum(ishares.is_cached(r.etf, r.date, RAW) for r in plan.itertuples()))
    res = run(plan)
    res.to_csv(a.out, index=False)
    print(res["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
