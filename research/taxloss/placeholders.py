"""Recover holdings rows whose Ticker is the placeholder "-".

`ishares.parse` drops these rows (unclean ticker, or no price). Here the raw cache is scanned for them, each row is
matched by normalised Name to a real-ticker row of the SAME ETF in the nearest earlier / later file, and the match is
accepted only if the prices of the two files are within a sane band. Recovery is in memory only (no DB write);
every row, matched or not, is logged to placeholder_ticker_matches.csv. Unmatched rows stay excluded.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

MAX_GAP_DAYS = 200
PRICE_BAND = (0.6, 1.6)          # placeholder price / matched-file price
STOP_TOKENS = {"COM", "CORP", "CORPORATION", "INC", "INCORPORATED", "LTD", "LIMITED", "CO", "THE", "COMPANY", "CL",
               "CLASS", "SHS", "VOTING", "SUBORDINATE", "SUB", "VTG", "NEW", "ORD", "TRUST", "INCOME", "FUND"}
SKIP_NAME = re.compile(r"WARRANT|\bRTS\b|RIGHTS?\b|INTERIM|SPIN-OFF|WIND-DOWN")
COLS = ["etf", "as_of", "name", "sector", "asset_class", "price", "shares", "market_value", "weight"]


def norm_name(name: str) -> str:
    """Upper-case, punctuation removed, corporate-form tokens dropped: 'CAMECO CORP COM' == 'CAMECO CORP'."""
    toks = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    return " ".join(t for t in toks if t not in STOP_TOKENS)


def _num(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def scan_file(path, etf: str) -> list[dict]:
    """Rows with Ticker '-' (CAD, positive price and shares) of one cached holdings file."""
    from ishares import header_date
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    as_of = header_date(text)
    if as_of is None:
        return []
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Ticker"), None)
    if hi is None:
        return []
    col = {h.strip(): i for i, h in enumerate(rows[hi])}
    shares_col = "Shares" if "Shares" in col else "Par Value"
    g = lambda r, k: r[col[k]].strip() if k in col and col[k] < len(r) else ""   # noqa: E731
    out = []
    for r in rows[hi + 1:]:
        if not r or r[0].strip() != "-" or g(r, "Currency") != "CAD":
            continue
        mv, sh, px = _num(g(r, "Market Value")), _num(g(r, shares_col)), _num(g(r, "Price"))
        if px is None and mv is not None and sh:
            px = mv / sh
        out.append({"etf": etf, "as_of": as_of, "name": g(r, "Name"), "sector": g(r, "Sector"),
                    "asset_class": g(r, "Asset Class"), "price": px, "shares": sh, "market_value": mv,
                    "weight": _num(g(r, "Weight (%)"))})
    return out


def scan_raw(raw_dir, dates=None) -> pd.DataFrame:
    """All placeholder rows in the raw cache (only files whose date is in `dates`, when given)."""
    rows = []
    for etf_dir in sorted(Path(raw_dir).glob("*")):
        for p in sorted(etf_dir.glob("*.csv")):
            if not re.fullmatch(r"\d{8}", p.stem):         # a stray file in the cache is not a holdings snapshot
                continue
            if dates is not None and pd.Timestamp(p.stem).date() not in dates:
                continue
            rows += scan_file(p, etf_dir.name)
    return pd.DataFrame(rows, columns=COLS)


def match(ph: pd.DataFrame, snap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (log, recovered rows in snapshot format). `snap` = parsed rows of the same files (real tickers)."""
    s = snap.assign(_k=snap["name"].map(norm_name))
    by = {k: g.sort_values("as_of") for k, g in s.groupby(["etf", "_k"])}
    have = set(zip(snap["etf"], snap["as_of"], snap["ysym"]))
    log, rec = [], []
    for r in ph.itertuples(index=False):
        row = {"etf": r.etf, "as_of": r.as_of, "name": r.name, "price": r.price, "weight": r.weight,
               "matched_ticker": "", "ysym": "", "matched_as_of": None, "price_ratio": None, "accepted": False,
               "reason": ""}
        log.append(row)
        key = norm_name(r.name)
        if SKIP_NAME.search(str(r.name).upper()):
            row["reason"] = "interim/warrant/rights line"
            continue
        if not r.price or r.price <= 0 or not r.shares or r.shares <= 0:
            row["reason"] = "no price or shares"
            continue
        g = by.get((r.etf, key))
        if g is None:
            row["reason"] = "no same-name real-ticker row in this ETF"
            continue
        cands = []
        for side, sub in (("before", g[g["as_of"] < r.as_of].tail(1)), ("after", g[g["as_of"] > r.as_of].head(1))):
            for m in sub.itertuples(index=False):
                gap = abs((m.as_of - r.as_of).days)
                ratio = r.price / m.price
                if gap <= MAX_GAP_DAYS and PRICE_BAND[0] <= ratio <= PRICE_BAND[1]:
                    cands.append((gap, side, m, ratio))
        if not cands:
            row["reason"] = "nearest same-name file too far or price outside band"
            continue
        if len({c[2].ysym for c in cands}) > 1:
            row["reason"] = "ambiguous (before/after files disagree on ticker)"
            continue
        gap, side, m, ratio = min(cands, key=lambda c: c[0])
        if (r.etf, r.as_of, m.ysym) in have:
            row["reason"] = "matched ticker already in the same file"
            continue
        row.update(matched_ticker=m.ticker, ysym=m.ysym, matched_as_of=m.as_of, price_ratio=round(ratio, 4),
                   accepted=True, reason=f"name match to {side} file ({gap} days)")
        have.add((r.etf, r.as_of, m.ysym))
        rec.append({"etf": r.etf, "as_of": r.as_of, "ticker": m.ticker, "ysym": m.ysym, "name": r.name,
                    "sector": r.sector, "price": r.price, "shares": r.shares, "market_value": r.market_value,
                    "weight": r.weight})
    cols = ["etf", "as_of", "ticker", "ysym", "name", "sector", "price", "shares", "market_value", "weight"]
    return pd.DataFrame(log), pd.DataFrame(rec, columns=cols)


def unmatched_weight_share(log: pd.DataFrame, snap_with_recovered: pd.DataFrame, etf: str, as_of) -> dict:
    """Share of the file's total weight that sits in unmatched placeholder rows (rows excluded from the study)."""
    f = snap_with_recovered[(snap_with_recovered["etf"] == etf) & (snap_with_recovered["as_of"] == as_of)]
    um = log[(log["etf"] == etf) & (log["as_of"] == as_of) & ~log["accepted"]]
    w_um = float(um["weight"].fillna(0).sum())
    w_all = float(f["weight"].fillna(0).sum()) + w_um
    return {"unmatched_weight": w_um, "total_weight": w_all, "share": (w_um / w_all) if w_all else 0.0,
            "n_unmatched": len(um)}
