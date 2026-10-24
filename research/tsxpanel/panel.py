"""Point-in-time TSX panel: every cached XIC / XCS holdings file parsed into one long table.

Columns: date, etf, ticker, ysym, name, isin, weight, price, shares, sector, source. The BlackRock CA files carry no
ISIN column in any era (checked 2012, 2019, 2024, 2025 headers), so `isin` is a null column kept for the schema.
`ysym` is the identity key everywhere downstream (the raw ticker scheme changes on 2012-05-31).
Placeholder rows (Ticker "-") are recovered by name match via taxloss.placeholders and tagged source="placeholder".
Raw files are licensed (personal, non-commercial): the parquet is derived from them and stays git-ignored.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

import _path  # noqa: F401
import ishares
import placeholders

COLUMNS = ["date", "etf", "ticker", "ysym", "name", "isin", "weight", "price", "shares", "sector", "source"]


def list_files(raw_dir: Path) -> pd.DataFrame:
    """Every dated .csv in the cache: etf, file_date, path, is_stub."""
    rows = []
    for etf_dir in sorted(Path(raw_dir).glob("*")):
        for p in sorted(etf_dir.glob("*.csv")):
            if not re.fullmatch(r"\d{8}", p.stem):
                continue
            text = p.read_text(encoding="utf-8-sig", errors="replace")
            hd = ishares.header_date(text)
            rows.append({"etf": etf_dir.name, "file_date": pd.Timestamp(p.stem).date(), "path": p,
                         "is_stub": hd is None, "header_date": hd})
    return pd.DataFrame(rows, columns=["etf", "file_date", "path", "is_stub", "header_date"])


def build(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (panel, files inventory, placeholder match log). A file whose header date differs from its file name is
    kept under the header date (the date the holdings are as of) and listed in the inventory with mismatch=True."""
    files = list_files(raw_dir)
    ok = files[~files["is_stub"]].copy()
    ok["mismatch"] = ok["header_date"] != ok["file_date"]
    parts = []
    ref = {}
    for etf, g in ok.groupby("etf"):
        late = g[g["file_date"] >= ishares.DUAL_SCHEME_END].sort_values("file_date")
        if len(late):
            ref[etf] = set(ishares.parse(late.iloc[0]["path"], etf)[0]["ticker"])
    for r in ok.sort_values(["etf", "file_date"]).itertuples(index=False):
        use_ref = ref.get(r.etf) if r.file_date < ishares.DUAL_SCHEME_END else None
        df, _ = ishares.parse(r.path, r.etf, use_ref)
        parts.append(df)
    snap = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ishares.PARSED_COLUMNS)
    ph = placeholders.scan_raw(raw_dir)
    plog, rec = placeholders.match(ph, snap) if len(ph) else (pd.DataFrame(), pd.DataFrame(columns=snap.columns))
    snap = snap.assign(source="parsed")
    if len(rec):
        snap = pd.concat([snap, rec.assign(source="placeholder")], ignore_index=True)
    panel = snap.rename(columns={"as_of": "date"}).assign(isin=None)[COLUMNS]
    panel = panel.sort_values(["etf", "date", "ysym"]).reset_index(drop=True)
    return panel, ok.drop(columns=["path"]), plog


def month_end_coverage(panel: pd.DataFrame, cal, years=range(2011, 2026), fallback_bdays: int = 3) -> pd.DataFrame:
    """Per (etf, year, month): the last trading day, the file date that covers it (the day itself or up to
    `fallback_bdays` business days earlier) and the number of rows. Months with no covering file show file_date None."""
    have = {e: set(g["date"]) for e, g in panel.groupby("etf")}
    n = panel.groupby(["etf", "date"]).size()
    rows = []
    for etf in sorted(have):
        for y in years:
            for m in range(1, 13):
                try:
                    me = cal.last_trading_day_of_month(y, m)
                except ValueError:
                    continue
                if me > dt.date(2025, 12, 31) or (y, m) < (2011, 2):
                    continue
                used, d = None, me
                for step in range(fallback_bdays + 1):
                    if d in have[etf]:
                        used = d
                        break
                    d -= dt.timedelta(days=1)
                    while d.weekday() >= 5:
                        d -= dt.timedelta(days=1)
                rows.append({"etf": etf, "year": y, "month": m, "month_end": me, "file_date": used,
                             "rows": int(n.get((etf, used), 0)) if used else 0})
    return pd.DataFrame(rows)


def main():
    from taxcal import Calendar, tsx_trading_days
    out = _path.DATA
    panel, files, plog = build(_path.RAW)
    panel.to_parquet(out / "panel.parquet", index=False)
    files.to_csv(out / "panel_files.csv", index=False)
    plog.to_csv(out / "placeholder_matches.csv", index=False)
    cal = Calendar(tsx_trading_days())
    month_end_coverage(panel, cal).to_csv(out / "month_end_coverage.csv", index=False)
    print(len(panel), "rows;", panel["date"].nunique(), "dates;", panel.groupby("etf")["date"].nunique().to_dict())


if __name__ == "__main__":
    main()
