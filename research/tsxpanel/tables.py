"""CLI: markdown tables (counts and date spans only) for the coverage note, from the git-ignored CSVs in data/."""
import datetime as dt

import pandas as pd

import _path

D = _path.DATA


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    parts = {}
    files = pd.read_csv(D / "panel_files.csv")
    files["year"] = files["file_date"].str[:4]
    parts["files_per_year"] = md(files.pivot_table(index="year", columns="etf", values="file_date", aggfunc="count", fill_value=0).reset_index())
    me = pd.read_csv(D / "month_end_coverage.csv")
    me["covered"] = me["file_date"].notna()
    t = me.groupby(["year", "etf"])["covered"].agg(["sum", "count"]).reset_index()
    t["month_ends"] = t["sum"].astype(int).astype(str) + "/" + t["count"].astype(str)
    parts["month_end"] = md(t.pivot(index="year", columns="etf", values="month_ends").reset_index())
    ws = pd.read_csv(D / "window_fetch_status.csv")
    ws["year"] = ws["date"].str[:4]
    parts["window_status"] = md(ws.pivot_table(index="year", columns="status", values="date", aggfunc="count", fill_value=0).reset_index())
    miss = ws[~ws["status"].eq("present")]
    parts["window_miss_dates"] = md(miss.groupby(["etf", "status"])["date"].agg(["count", "min", "max"]).reset_index())
    ev = pd.read_csv(D / "events.csv")
    ev["year"] = ev["first_out"].str[:4]
    parts["events_year_delete_xic"] = md(ev[(ev.etf == "XIC") & (ev.direction == "delete")].pivot_table(index="year", columns="class", values="ysym", aggfunc="count", fill_value=0).reset_index())
    parts["events_year_delete_xcs"] = md(ev[(ev.etf == "XCS") & (ev.direction == "delete")].pivot_table(index="year", columns="class", values="ysym", aggfunc="count", fill_value=0).reset_index())
    parts["events_year_add"] = md(ev[ev.direction == "add"].pivot_table(index="year", columns=["etf", "class"], values="ysym", aggfunc="count", fill_value=0).reset_index().rename(columns=lambda c: " ".join(c) if isinstance(c, tuple) else c))
    parts["offsets"] = md(pd.read_csv(D / "review_offsets.csv").pivot(index="review_offset", columns="etf", values="n").fillna(0).astype(int).reset_index())
    parts["off_cycle"] = md(pd.read_csv(D / "off_cycle.csv").groupby("etf")[["events", "certain_off_cycle", "narrow_not_review", "month_end_resolution"]].sum().reset_index())
    parts["flags"] = md(ev.assign(f=ev["flags"].fillna("").str.split("|")).explode("f").query("f != ''").groupby(["etf", "direction", "f"]).size().rename("n").reset_index())
    pc = D / "price_coverage_per_year.csv"
    if pc.exists():
        parts["price_cov"] = md(pd.read_csv(pc))
    (D / "tables.md").write_text("\n\n".join(f"<!-- {k} -->\n{v}" for k, v in parts.items()))
    print("\n\n".join(f"## {k}\n{v}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
