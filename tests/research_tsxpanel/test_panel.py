import datetime as dt

import pandas as pd

import panel

D = dt.date
HEAD = 'Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency'


def write(root, etf, ymd, as_of, rows):
    d = root / etf
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'"{t}","{n}","Fin","Equity","1,000.00","1.00","-","100.00","{px}","Canada","TSX","CAD","1.00","CAD"'
                     for t, n, px in rows)
    (d / f"{ymd}.csv").write_text(f'Fund Holdings as of,"{as_of}"\n \n{HEAD}\n{body}\n', encoding="utf-8")


def test_build_skips_stub_and_keeps_schema(tmp_path):
    write(tmp_path, "XIC", "20240628", "Jun 28, 2024", [("RY", "ROYAL BANK", "10.00"), ("TD", "TD BANK", "20.00")])
    (tmp_path / "XIC" / "20240627.csv").write_text('Fund Holdings as of,"-"\n', encoding="utf-8")
    write(tmp_path, "XCS", "20240628", "Jun 28, 2024", [("ABC", "ABC INC", "3.00")])
    p, files, plog = panel.build(tmp_path)
    assert list(p.columns) == panel.COLUMNS
    assert len(p) == 3 and p["isin"].isna().all() and set(p["source"]) == {"parsed"}
    assert files["file_date"].nunique() == 1 and not files["mismatch"].any()


def test_month_end_coverage_falls_back_up_to_3_bdays(cal):
    rows = [{"date": D(2024, 5, 29), "etf": "XIC"}, {"date": D(2024, 6, 28), "etf": "XIC"}]
    p = pd.DataFrame(rows).assign(ysym="A.TO")
    c = panel.month_end_coverage(p, cal, years=[2024])
    may = c[(c["month"] == 5) & (c["etf"] == "XIC")].iloc[0]       # 2024-05-31 is a Friday, file is 05-29 (2 bdays)
    assert may["file_date"] == D(2024, 5, 29)
    apr = c[(c["month"] == 4) & (c["etf"] == "XIC")].iloc[0]
    assert apr["file_date"] is None and apr["rows"] == 0
