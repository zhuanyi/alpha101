"""CLI: price coverage after deletion -> price_coverage.csv, price_coverage_per_year.csv (read-only DB, cached yfinance)."""
import logging

import pandas as pd

import _path
import coverage


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from alpha101.db import reader
    ev = pd.read_csv(_path.DATA / "events.csv", parse_dates=["last_in", "first_out", "review_date"])
    for c in ("last_in", "first_out", "review_date"):
        ev[c] = ev[c].dt.date.where(ev[c].notna(), None)
    panel = pd.read_parquet(_path.DATA / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    conn = reader()
    try:
        tgt = coverage.targets(ev)
        cov = coverage.run(tgt, panel, conn, _path.DATA / "yf_cache")
    finally:
        conn.close()
    cov.to_csv(_path.DATA / "price_coverage.csv", index=False)
    from taxcal import Calendar, tsx_trading_days
    ev = coverage.add_exit_class(ev, cov)
    ev.to_csv(_path.DATA / "events.csv", index=False)
    coverage.review_aligned_exit_table(ev, panel, Calendar(tsx_trading_days())).to_csv(
        _path.DATA / "review_aligned_exit.csv", index=False)
    py = coverage.per_year(cov)
    py.to_csv(_path.DATA / "price_coverage_per_year.csv", index=False)
    print(py.to_string(index=False))


if __name__ == "__main__":
    main()
