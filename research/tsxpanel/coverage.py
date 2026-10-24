"""Price coverage after deletion: does a daily price series exist for names that left XIC / XCS?

Counts rows and spans only (no returns, no price ratios). A Yahoo series counts as coverage only if it passes the
tax-loss identity check (taxloss.yahootopup.identity_ok: un-adjusted Yahoo close within 2% of the holdings price on the
pre-deletion overlap) AND has at least MIN_POST closes in the POST_DAYS calendar days after first_out. Sources are
tried in order: the `yahoo_ohlcv` table (read-only), then yfinance (cached under data/yf_cache,
nothing is written to the database).
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

import _path  # noqa: F401
import yahootopup

log = logging.getLogger("tsxpanel.coverage")
MIN_POST = 20
POST_DAYS = 90
TARGET_CLASSES = ("left_both",)


def targets(ev: pd.DataFrame, classes=TARGET_CLASSES) -> pd.DataFrame:
    """Confirmed deletions of the given classes, one row per (etf, ysym, first_out)."""
    t = ev[(ev["direction"] == "delete") & ev["class"].isin(classes)
           & ~ev["flags"].fillna("").str.contains("unconfirmed")].copy()
    t["year"] = t["first_out"].map(lambda d: d.year)
    return t.drop_duplicates(["etf", "ysym", "first_out"]).reset_index(drop=True)


def assess(ysym: str, last_in, first_out, holdings: pd.DataFrame, close: pd.Series | None, splits: pd.Series | None,
           first_date, min_post: int = MIN_POST, post_days: int = POST_DAYS) -> dict:
    """holdings: columns as_of, price for this ysym (pre-deletion files). -> {post_rows, post_last, ident_ok, reason,
    covered}. post_rows counts closes strictly after first_out - 1 day (first_out itself counts) within post_days."""
    ident = yahootopup.identity_ok(ysym, holdings, close, splits, first_date, pd.Timestamp(last_in))
    post_rows, post_last = 0, None
    if close is not None and len(close):
        w = close[(close.index >= pd.Timestamp(first_out)) & (close.index <= pd.Timestamp(first_out) + pd.Timedelta(days=post_days))]
        w = w[w > 0]
        post_rows = len(w)
        post_last = w.index.max().date() if len(w) else None
    return {"post_rows": post_rows, "post_last": post_last, "ident_ok": bool(ident["ok"]), "reason": ident["reason"],
            "covered": bool(ident["ok"]) and post_rows >= min_post}


def _load_cache(cache_dir: Path, ysym: str):
    p = Path(cache_dir) / f"{ysym}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if (Path(cache_dir) / f"{ysym}.none").exists():
        return pd.DataFrame()
    return None


def yf_frame(ysym: str, cache_dir: Path, fetch=None) -> pd.DataFrame:
    """Cached yfinance history (empty frame when Yahoo has nothing). `fetch(ysym)` defaults to ingest.fetch_history."""
    cache_dir = Path(cache_dir)
    got = _load_cache(cache_dir, ysym)
    if got is not None:
        return got
    if fetch is None:
        from alpha101.data import ingest
        fetch = lambda s: ingest.fetch_history(s, pause=1.0)   # noqa: E731
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = fetch(ysym)
    except Exception as e:  # noqa: BLE001 - rate limit / invalid symbol: leave uncached so a rerun retries
        log.warning("%s: yfinance %s", ysym, type(e).__name__)
        return pd.DataFrame()
    if df is None or df.empty:
        (cache_dir / f"{ysym}.none").write_text("")
        return pd.DataFrame()
    df.to_parquet(cache_dir / f"{ysym}.parquet", index=False)
    return df


def _series(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None, None
    d = df.assign(date=pd.to_datetime(df["date"])).set_index("date")
    close = d["close"][d["close"] > 0]
    sp = d["split"].fillna(0.0) if "split" in d else pd.Series(dtype=float)
    return close, sp[sp > 0], close.index.min()


def run(tgt: pd.DataFrame, panel: pd.DataFrame, conn, cache_dir: Path, fetch=None, use_yf: bool = True) -> pd.DataFrame:
    """One row per target with db_* and yf_* assessments and the best `source`."""
    syms = sorted(set(tgt["ysym"]))
    closes, splits, firsts = yahootopup.load_yahoo(conn, syms, start="2005-01-01", end=dt.date.today().isoformat())
    hold = {s: g[["date", "price"]].rename(columns={"date": "as_of"}) for s, g in panel[panel["ysym"].isin(syms)].groupby("ysym")}
    rows = []
    for r in tgt.itertuples(index=False):
        h = hold.get(r.ysym, pd.DataFrame(columns=["as_of", "price"]))
        h = h[h["as_of"] <= r.last_in]
        a = assess(r.ysym, r.last_in, r.first_out, h, closes.get(r.ysym), splits.get(r.ysym), firsts.get(r.ysym))
        row = {"etf": r.etf, "ysym": r.ysym, "name": r.name, "year": r.year, "last_in": r.last_in,
               "first_out": r.first_out, "weight_ref": r.weight_ref, **{f"db_{k}": v for k, v in a.items()},
               "source": "yahoo_ohlcv" if a["covered"] else "none"}
        if not a["covered"] and use_yf:
            c, s, f = _series(yf_frame(r.ysym, cache_dir, fetch))
            b = assess(r.ysym, r.last_in, r.first_out, h, c, s, f)
            row.update({f"yf_{k}": v for k, v in b.items()})
            if b["covered"]:
                row["source"] = "yfinance"
        rows.append(row)
    return pd.DataFrame(rows)


def per_year(cov: pd.DataFrame) -> pd.DataFrame:
    """Counts by ETF and deletion year: names, covered by DB, added by yfinance, uncovered, weight-share uncovered."""
    def agg(g):
        w = g["weight_ref"].fillna(0)
        return pd.Series({"deleted": len(g), "db": int((g["source"] == "yahoo_ohlcv").sum()),
                          "yfinance_only": int((g["source"] == "yfinance").sum()),
                          "uncovered": int((g["source"] == "none").sum()),
                          "uncovered_pct": round(100 * (g["source"] == "none").mean(), 1),
                          "uncovered_weight_pct": round(100 * w[g["source"] == "none"].sum() / w.sum(), 1) if w.sum() else None})
    out = cov.groupby(["etf", "year"]).apply(agg, include_groups=False).reset_index()
    tot = cov.groupby("etf").apply(agg, include_groups=False).reset_index().assign(year="ALL")
    return pd.concat([out, tot], ignore_index=True)


# --- exit classes (the tax-loss classes of rebound.CLASSES) -------------------------------------------------------
def add_exit_class(ev: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """`exit_class` for deletions, using the tax-loss class names (imported, not re-declared): the sibling ETF's file
    prices the name -> other_etf; a left_both name with an identity-checked Yahoo series after deletion -> yahoo;
    anything else -> unresolved. `takeover` needs a logged source (takeover_sources.csv holds no rows), so acquired
    names sit inside unresolved. Renames and additions get ''."""
    import rebound
    cov_src = cov.set_index(["etf", "ysym", "first_out"])["source"].to_dict() if len(cov) else {}
    out = []
    for r in ev.to_dict("records"):
        if r["direction"] != "delete" or r["class"] == "ticker_change":
            out.append("")
        elif r["class"] in ("to_xcs", "to_xic"):
            out.append("other_etf")
        else:
            out.append("yahoo" if cov_src.get((r["etf"], r["ysym"], r["first_out"]), "none") != "none" else "unresolved")
    assert set(out) - {""} <= set(rebound.CLASSES)
    return ev.assign(exit_class=out)


def review_aligned_exit_table(ev: pd.DataFrame, panel: pd.DataFrame, cal) -> pd.DataFrame:
    """Counts for the review-aligned XIC deletions per year: names, by exit class, how many `other_etf` names are
    actually in the XCS file at T+10 trading days (the exit day of the deletion draft), and the share of ETF weight
    (at the last holdings file) per exit class. Counts and weight shares only."""
    d = ev[(ev["direction"] == "delete") & (ev["etf"] == "XIC") & ev["review_aligned"]
           & ev["exit_class"].isin(["other_etf", "yahoo", "unresolved"])].copy()
    xcs = panel[panel["etf"] == "XCS"].groupby("date")["ysym"].apply(set).to_dict()
    xcs_dates = sorted(xcs)

    def in_xcs_t10(r):
        try:
            x = cal.shift(r.review_date, 10)
        except ValueError:
            return False
        near = [dd for dd in xcs_dates if abs((dd - x).days) <= 1]
        return bool(near) and r.ysym in xcs[near[0]]

    d["xcs_at_t10"] = [in_xcs_t10(r) for r in d.itertuples(index=False)] if len(d) else []
    d["year"] = d["first_out"].map(lambda x: x.year)
    d["w"] = d["weight_ref"].fillna(0)

    def agg(g):
        tot = g["w"].sum()
        return pd.Series({"deleted": len(g), "other_etf": int((g["exit_class"] == "other_etf").sum()),
                          "other_etf_in_xcs_at_T10": int(((g["exit_class"] == "other_etf") & g["xcs_at_t10"]).sum()),
                          "yahoo": int((g["exit_class"] == "yahoo").sum()),
                          "unresolved": int((g["exit_class"] == "unresolved").sum()),
                          "unresolved_weight_pct": round(100 * g.loc[g["exit_class"] == "unresolved", "w"].sum() / tot, 1) if tot else None,
                          "no_price_at_T10_weight_pct": round(100 * g.loc[(g["exit_class"] == "unresolved") | ((g["exit_class"] == "other_etf") & ~g["xcs_at_t10"]), "w"].sum() / tot, 1) if tot else None})
    per = d.groupby("year").apply(agg, include_groups=False).reset_index()
    tot = agg(d).to_frame().T.assign(year="ALL")
    return pd.concat([per, tot], ignore_index=True)
