"""CLI: panel.parquet -> events.csv and the count tables (all under the git-ignored data/ dir)."""
import pandas as pd

import _path
import events
import reviews


def main():
    from taxcal import Calendar, tsx_trading_days
    cal = Calendar(tsx_trading_days())
    panel = pd.read_parquet(_path.DATA / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    rv = list(reviews.review_dates(cal)["review_date"])
    ev = events.build_events(panel, cal, rv)
    ev.to_csv(_path.DATA / "events.csv", index=False)
    events.counts_per_year(ev).to_csv(_path.DATA / "events_per_year.csv", index=False)
    events.review_offset_distribution(ev).to_csv(_path.DATA / "review_offsets.csv", index=False)
    events.off_cycle(ev).to_csv(_path.DATA / "off_cycle.csv", index=False)
    print(len(ev), "events")
    print(ev.groupby(["etf", "direction", "class"]).size().to_string())


if __name__ == "__main__":
    main()
