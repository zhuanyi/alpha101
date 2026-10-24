"""Yahoo top-up for the iShares-derived eligible universe, plus the identity check (spec 8).

Coverage only. The bottom-20 membership below uses a crude YTD sort (holdings prices, the 1.8x/0.55x fund-share
guard of the 1.3 probe, applied to share ratios normalised by the fund-wide median ratio as in spec 7) for ONE purpose: to count how many of those 20 names have an identity-checked Yahoo
series. The YTD values are never stored, logged or written; this is not the study's `rank_ytd`.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("taxloss.yahoo")

STATE = Path(__file__).resolve().parent / "data" / "raw" / "yahoo_fetch_state.csv"       # git-ignored
HAVE_BEFORE = Path(__file__).resolve().parent / "data" / "raw" / "yahoo_have_before.txt"
PRICE_FLOOR_FETCH = 1.0       # fetch universe = V-P1 floor (superset of P's C$2); coverage is reported at C$2
PRICE_FLOOR_P = 2.0
ID_TOL = 0.02
MIN_OVERLAP = 3
SHARE_GUARD = (0.55, 1.8)
MAX_CONSECUTIVE_LIMITED = 15
CLASS_SUFFIX = re.compile(r"^([A-Z0-9]+)[.-][A-Z]$")


# --- eligible universe (coverage definition of 2.3 rules 1, 2, 4) -------------------------------------------------
def issuer_key(ticker: str) -> str:
    """Share classes of one issuer ('BBD.B', 'BBD.A') share a key; units ('NWH.UN') and plain tickers are their own."""
    m = CLASS_SUFFIX.match(ticker.upper())
    return m.group(1) if m else ticker.upper()


def _file(snap, d):
    """XIC rows first, then XCS-only names, on date d: one row per ysym."""
    f = snap[snap["as_of"] == d].copy()
    f["_o"] = (f["etf"] != "XIC").astype(int)
    return f.sort_values(["_o", "ysym"]).drop_duplicates("ysym").drop(columns="_o")


def resolve_R(snap: pd.DataFrame, R):
    """The ranking-day file actually used: R itself, else the latest earlier file within 5 days (2021 has no R file)."""
    return max([d for d in set(snap["as_of"]) if d <= R and (R - d).days <= 5], default=R)


def eligible_universe(snap: pd.DataFrame, D_prev, R, floor: float = PRICE_FLOOR_P) -> pd.DataFrame:
    """Names in the D_{y-1} file and the R file (XIC or XCS), price at R >= floor, one share class per issuer (larger
    fund market value). Columns: ysym, ticker, etf, name, price_D, price_R, mv_R, guard_ok (fund share-count guard)."""
    cols = ["ysym", "ticker", "etf", "name", "price_D", "price_R", "mv_R", "guard_ok"]
    a, b = _file(snap, D_prev), _file(snap, R)
    if a.empty or b.empty:
        return pd.DataFrame(columns=cols)
    m = b.merge(a[["ysym", "etf", "price", "shares"]].rename(columns={"etf": "etf_D", "price": "price_D",
                                                                      "shares": "shares_D"}), on="ysym")
    m = m[m["price"] >= floor].copy()
    ratio = m["shares"] / m["shares_D"]
    same = m["etf"] == m["etf_D"]
    med = ratio[same].groupby(m.loc[same, "etf"]).median()        # fund-wide creations/redemptions move every holding
    adj = ratio / m["etf"].map(med).fillna(1.0)
    m["guard_ok"] = ~(same & ((adj < SHARE_GUARD[0]) | (adj > SHARE_GUARD[1])))
    m["_k"] = m["ticker"].map(issuer_key)
    m = m.sort_values(["_k", "market_value", "ticker"], ascending=[True, False, True]).drop_duplicates("_k")
    m = m.rename(columns={"price": "price_R", "market_value": "mv_R"})
    return m[cols].sort_values("ysym").reset_index(drop=True)


def bottom20(elig: pd.DataFrame, n: int = 20) -> list[str]:
    """ysyms of the n lowest crude-YTD names among guard_ok names. Values stay local; only names are returned."""
    e = elig[elig["guard_ok"].astype(bool)]
    ytd = pd.Series((e["price_R"] / e["price_D"] - 1).to_numpy(), index=e["ysym"].to_numpy()).sort_values(kind="mergesort")
    return list(ytd.index[:n])


# --- identity check ------------------------------------------------------------------------------------------------
def unadjust(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Yahoo close is split-adjusted to the fetch date: multiply by every split ratio dated AFTER the row's date to
    get the as-traded price. `splits`: ratio by effective date (2.0 = 2-for-1, 0.1 = 1-for-10)."""
    out = close.astype(float).copy()
    for sd, r in splits.items():
        if r and r > 0:
            out[out.index < pd.Timestamp(sd)] *= r
    return out


def identity_ok(ysym: str, holdings: pd.DataFrame, close: pd.Series | None, splits: pd.Series | None, first_date,
                d_prev, tol: float = ID_TOL, min_overlap: int = MIN_OVERLAP) -> dict:
    """holdings: columns as_of, price (as-traded) for this ysym; close: Yahoo close by date (fetch-date split basis);
    splits: Yahoo split ratios by effective date (ALL dates, also after the window); first_date: first stored Yahoo
    date. ok = Yahoo starts on/before d_prev AND median |un-adjusted Yahoo close / holdings price - 1| <= tol over
    >= min_overlap overlapping dates. after_unadjust = passes only because of the split un-adjusting."""
    res = {"ysym": ysym, "ok": False, "reason": "", "n_overlap": 0, "after_unadjust": False}
    if close is None or close.empty or first_date is None:
        res["reason"] = "no_yahoo"
        return res
    if pd.Timestamp(first_date) > pd.Timestamp(d_prev):
        res["reason"] = "starts_after_D_prev"
        return res
    h = holdings.drop_duplicates("as_of")
    h = h.set_index(pd.to_datetime(h["as_of"]))["price"]
    close = close.dropna()
    close = close[close > 0]
    j = pd.concat([h.rename("h"), close.rename("y")], axis=1, join="inner")
    res["n_overlap"] = len(j)
    if len(j) < min_overlap:
        res["reason"] = "insufficient_overlap"
        return res
    adj = unadjust(j["y"], splits if splits is not None else pd.Series(dtype=float))
    err = float(np.median(np.abs(adj / j["h"] - 1)))
    raw = float(np.median(np.abs(j["y"] / j["h"] - 1)))
    res["ok"] = err <= tol
    res["after_unadjust"] = res["ok"] and raw > tol
    res["reason"] = "" if res["ok"] else "price_mismatch"
    return res


# --- fetch ---------------------------------------------------------------------------------------------------------
def _read_state() -> dict[str, str]:
    if not STATE.exists():
        return {}
    with STATE.open() as f:
        return {r["symbol"]: r["status"] for r in csv.DictReader(f)}


def _write_state(sym: str, status: str, rows: int = 0) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    new = not STATE.exists()
    with STATE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["symbol", "status", "rows", "at"])
        w.writerow([sym, status, rows, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")])


def stored_symbols(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM yahoo_ohlcv")
        return {r[0] for r in cur.fetchall()}


def topup(wconn, symbols: list[str], have: set[str], pause: float = 1.0) -> dict:
    """Fetch symbols not in yahoo_ohlcv with the per-symbol ingest functions (never the ingest() driver) and upsert.
    Resumable: symbols already written / known no-data / invalid are skipped; rate-limited ones are retried."""
    from alpha101.data import ingest as ing
    state = _read_state()
    out = {"attempted": 0, "written": 0, "rows": 0, "no_data": [], "invalid": [], "rate_limited": [], "error": []}
    consecutive = 0
    for sym in symbols:
        if sym in have or state.get(sym) in ("written", "no_data", "invalid"):
            continue
        out["attempted"] += 1
        fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
        try:
            df = ing.fetch_history(sym, pause=pause)
        except ValueError:
            out["invalid"].append(sym)
            _write_state(sym, "invalid")
            continue
        except ing.RateLimited:
            out["rate_limited"].append(sym)
            _write_state(sym, "rate_limited")
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_LIMITED:
                log.error("Yahoo rate-limited %d symbols in a row; stopping top-up (rerun resumes)", consecutive)
                break
            continue
        except Exception as e:  # noqa: BLE001 - one malformed frame must not stop the run
            out["error"].append(sym)
            _write_state(sym, "error")
            log.warning("%s: fetch failed (%s)", sym, type(e).__name__)
            continue
        consecutive = 0
        if df is None:
            out["no_data"].append(sym)
            _write_state(sym, "no_data")
        else:
            try:
                wconn.ping(reconnect=True)
                n = ing.upsert_ohlcv(wconn, sym, df, fetched_at)
            except Exception as e:  # noqa: BLE001 - DB errors can echo credentials: class name only
                out["error"].append(sym)
                _write_state(sym, "error")
                log.error("%s: write failed (%s)", sym, type(e).__name__)
                continue
            out["written"] += 1
            out["rows"] += n
            _write_state(sym, "written", n)
        if out["attempted"] % 50 == 0:
            log.info("top-up %d attempted, %d written, %d no_data", out["attempted"], out["written"], len(out["no_data"]))
        time.sleep(pause)
    return out


def load_yahoo(conn, symbols: list[str], start="2011-11-01", end="2026-03-31"):
    """-> (closes {ysym: Series by date within [start, end]}, splits {ysym: Series ratio by date, ALL dates},
    firsts {ysym: first stored date})."""
    from alpha101.data.yahoo import read_long
    closes, splits, firsts = {}, {}, {}
    for i in range(0, len(symbols), 200):
        part = symbols[i:i + 200]
        ph = ",".join(["%s"] * len(part))
        with conn.cursor() as cur:
            cur.execute(f"SELECT symbol, MIN(date) FROM yahoo_ohlcv WHERE symbol IN ({ph}) GROUP BY symbol", part)
            firsts.update({s: pd.Timestamp(d) for s, d in cur.fetchall()})
            cur.execute(f"SELECT symbol, date, split FROM yahoo_ohlcv WHERE split > %s AND symbol IN ({ph})", [0, *part])
            for s, d, r in cur.fetchall():
                splits.setdefault(s, {})[pd.Timestamp(d)] = float(r)
        d = read_long(conn, part, start, end)
        for s, g in d.groupby("symbol"):
            closes[s] = g.set_index("date")["close"]
    return closes, {s: pd.Series(v) for s, v in splits.items()}, firsts


# --- driver --------------------------------------------------------------------------------------------------------
YEARS = range(2012, 2026)


def event_coverage(snap, cal, have_before, closes, splits, firsts, y) -> tuple[dict, list[dict]]:
    """One yahoo_coverage.csv row for event y (+ identity details of the names that pass only after un-adjusting)."""
    e = cal.event_row(y)
    R_used = resolve_R(snap, e["R"])
    elig = eligible_universe(snap, e["D_prev"], R_used, PRICE_FLOOR_P)
    win = snap[(snap["as_of"] >= e["D_prev"]) & (snap["as_of"] <= e["VX3_on_or_after_feb15"])]
    win = win.assign(_o=(win["etf"] != "XIC").astype(int)).sort_values(["_o"]).drop_duplicates(["ysym", "as_of"])
    by_sym = {s: g[["as_of", "price"]] for s, g in win.groupby("ysym")}
    ids = {s: identity_ok(s, by_sym.get(s, pd.DataFrame(columns=["as_of", "price"])), closes.get(s), splits.get(s),
                          firsts.get(s), e["D_prev"]) for s in elig["ysym"]}
    ok = {s for s, r in ids.items() if r["ok"]}
    b20 = bottom20(elig)
    xrows = snap[snap["as_of"] == e["jan_exit_X"]]
    x_present = len(xrows) > 0
    x_syms = set(xrows["ysym"])
    xd = pd.Timestamp(e["jan_exit_X"])
    cls = {"holdings": 0, "yahoo": 0, "unresolved_ub": 0}
    for s in b20:
        if s in x_syms:
            cls["holdings"] += 1
        elif s in ok and s in closes and xd in closes[s].dropna().index:
            cls["yahoo"] += 1
        else:
            cls["unresolved_ub"] += 1
    reasons = pd.Series([r["reason"] for r in ids.values() if not r["ok"]]).value_counts()
    xcs_only = elig[elig["ysym"].isin(b20) & (elig["etf"] != "XIC")]
    row = {"year": y, "R_used": R_used, "eligible_n": len(elig), "guard_excluded_n": int((~elig["guard_ok"]).sum()),
           "in_db_before_n": int(elig["ysym"].isin(have_before).sum()),
           "identity_ok_n": len(ok), "identity_after_unadjust_n": sum(1 for r in ids.values() if r["after_unadjust"]),
           **{f"fail_{k}": int(reasons.get(k, 0)) for k in ("no_yahoo", "starts_after_D_prev", "insufficient_overlap",
                                                              "price_mismatch")},
           "bottom20_n": len(b20), "bottom20_xcs_only_n": len(xcs_only),
           "bottom20_in_db_before_n": sum(1 for s in b20 if s in have_before),
           "bottom20_identity_ok_n": sum(1 for s in b20 if s in ok),
           "X_file_present": x_present, "b20_exit_holdings_n": cls["holdings"], "b20_exit_yahoo_n": cls["yahoo"],
           "b20_unresolved_upper_bound_n": cls["unresolved_ub"]}
    n_el, n_b = max(len(elig), 1), max(len(b20), 1)
    row["in_db_before_share"] = round(row["in_db_before_n"] / n_el, 4)
    row["identity_ok_share"] = round(row["identity_ok_n"] / n_el, 4)
    row["bottom20_in_db_before_share"] = round(row["bottom20_in_db_before_n"] / n_b, 4)
    row["bottom20_identity_ok_share"] = round(row["bottom20_identity_ok_n"] / n_b, 4)
    row["unresolved_upper_bound_share"] = round(cls["unresolved_ub"] / n_b, 4)
    det = [{"year": y, "ysym": s, "n_overlap": r["n_overlap"]} for s, r in ids.items() if r["after_unadjust"]]
    return row, det


def run(cal, out: Path) -> None:
    from alpha101.db import reader, writer
    import ishares
    out = Path(out)
    rconn = reader()
    try:
        snap = ishares.read_snapshots(rconn)
        have_now = stored_symbols(rconn)
    finally:
        rconn.close()
    if snap.empty:
        raise SystemExit("etf_holdings_snapshot is empty: run --fetch-holdings first")
    ev = {y: cal.event_row(y) for y in YEARS}
    fetch_syms = set()
    for y, e in ev.items():
        fetch_syms |= set(eligible_universe(snap, e["D_prev"], resolve_R(snap, e["R"]), PRICE_FLOOR_FETCH)["ysym"])
    fetch_syms |= {"XCS.TO"}
    log.info("fetch universe: %d distinct ysyms (C$%.0f floor), %d already in yahoo_ohlcv",
             len(fetch_syms), PRICE_FLOOR_FETCH, len(fetch_syms & have_now))
    if HAVE_BEFORE.exists():
        have_before = set(HAVE_BEFORE.read_text().split())
    else:
        have_before = set(have_now)
        HAVE_BEFORE.parent.mkdir(parents=True, exist_ok=True)
        HAVE_BEFORE.write_text("\n".join(sorted(have_before)))
    wconn = writer()
    try:
        res = topup(wconn, sorted(fetch_syms), have_now)
    finally:
        wconn.close()
    log.info("yahoo_ohlcv top-up: attempted=%d written=%d rows=%d no_data=%d invalid=%d rate_limited=%d error=%d",
             res["attempted"], res["written"], res["rows"], len(res["no_data"]), len(res["invalid"]),
             len(res["rate_limited"]), len(res["error"]))
    rconn = reader()
    try:
        closes, splits, firsts = load_yahoo(rconn, sorted(fetch_syms))
    finally:
        rconn.close()
    rows, details = [], []
    for y in YEARS:
        r, d = event_coverage(snap, cal, have_before, closes, splits, firsts, y)
        rows.append(r)
        details += d
    df = pd.DataFrame(rows)
    df["partial_exit"] = ~df["X_file_present"]
    df.to_csv(out / "yahoo_coverage.csv", index=False)
    pd.DataFrame(details, columns=["year", "ysym", "n_overlap"]).to_csv(out / "identity_after_unadjust.csv", index=False)
    state = _read_state()
    pd.DataFrame(sorted(state.items()), columns=["symbol", "last_status"]).to_csv(out / "yahoo_topup_status.csv", index=False)
    write_summary(out)


def write_summary(out: Path) -> None:
    """coverage_summary.md, generated from the CSVs in `out` (nothing typed in)."""
    out = Path(out)
    lines = ["# TSX tax-loss study: coverage summary", "", "Generated from the CSVs in this directory. Coverage counts only; "
             "no ranks, baskets or returns are computed here.", ""]
    hc = out / "holdings_coverage.csv"
    if hc.exists():
        h = pd.read_csv(hc)
        lines += ["## iShares holdings coverage (requested dates; holdout dates are not requested)", "",
                  h[h["year"].isin(["ALL", "ALL_ex_2017_gap"])].to_markdown(index=False), ""]
    yc = out / "yahoo_coverage.csv"
    if yc.exists():
        y = pd.read_csv(yc)
        cols = ["year", "eligible_n", "in_db_before_share", "identity_ok_share", "bottom20_n", "bottom20_xcs_only_n",
                "bottom20_in_db_before_share", "bottom20_identity_ok_share", "unresolved_upper_bound_share", "partial_exit"]
        lines += ["## Yahoo identity-checked coverage per event (eligible = 2.3 rules 1, 2, 4 at C$2)", "",
                  y[cols].to_markdown(index=False), ""]
        m_all, m_ex = y["unresolved_upper_bound_share"].mean(), y.loc[~y["partial_exit"], "unresolved_upper_bound_share"].mean()
        lines += [f"Projected `unresolved` share of basket weight (upper bound: no takeover credit, no other-ETF "
                  f"fallback beyond the X files): mean over {len(y)} events = {m_all:.1%}; "
                  f"excluding partial-exit events = {m_ex:.1%}. EODHD trigger = 10%.", ""]
    (out / "coverage_summary.md").write_text("\n".join(lines))
