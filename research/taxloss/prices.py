"""Price panel and corporate actions for the tax-loss rebound study (spec 1.4, 2.5, 7).

Holdings prices are as-traded; Yahoo `close` is split-adjusted to the fetch date. The two bases are never mixed:
a Yahoo exit is the holdings entry price times a Yahoo close RATIO, and identity checks un-adjust Yahoo first.

Contents
  detect_splits   spec 7 rule (price ratio between consecutive files > 40% while the fund share count, normalised by the
                  fund-wide median ratio, moves inversely within 5%) OR the Yahoo split field
  PricePanel      holdings lookups, split factors, Yahoo identity, dividends, exit hierarchy (1.4), spreads inputs
  gate            big_moves.csv: |window return| > 100% must be cleared by hand before decide()
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

import yahootopup

PRICE_JUMP = 0.40
INVERSE_TOL = 0.05
YAHOO_MIN_LN = math.log(1.5)                # a Yahoo-only 'split' with |ratio| inside 1.5x is a unit/stock-dividend artefact
NONROUND_MIN_LN = math.log(2.5)           # a holdings-only hit with a non-round ratio is applied only for a >= 2.5x price move
YAHOO_CONSISTENT_LN = math.log(2.0)      # a Yahoo split is applied if the holdings price ratio times q is within 2x of 1
WINDOW_MISMATCH_LN = math.log(1.5)       # holdings and Yahoo returns over one name's own consecutive observations disagree by > 1.5x
WINDOW_NICE_TOL = 0.03                   # ... and the implied factor is within 3% of a round split ratio
WINDOW_SHARES_LN = math.log(1.25)        # ... and the ETF share count moved inversely to q within 25% (else a ticker re-assignment)
WINDOW_MAX_DAYS = 45                     # own-observation pairs (the name missing from files in between) at most this far apart
UNRESOLVED_HAIRCUT = 0.30
BIG_MOVE = 1.0
NICE = sorted({1 / k for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 500, 1000)}
              | {float(k) for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)})
HOLDOUT_FLOOR = dt.date(2011, 2, 1)       # pass 1 never loads holdings dated before this (2007-2011 M-series)

BIG_COLUMNS = ["year", "ysym", "name", "cells", "scope", "max_window_return", "entry_date", "exit_date",
               "entry_price", "exit_price", "exit_class", "exit_source", "yahoo_return", "yahoo_close_entry",
               "yahoo_close_exit", "split_factor", "clearance", "reason"]
CLEARANCES = {"keep", "exclude"}


def snap_nice(x: float) -> float:
    """Snap a share-count ratio to a round split ratio when within 5%, else keep it."""
    j = min(NICE, key=lambda v: abs(math.log(x / v)))
    return j if abs(x / j - 1) <= 0.05 else float(x)


def is_nice(q: float) -> bool:
    return any(abs(q / j - 1) < 1e-9 for j in NICE)


def _bdays_before(d: dt.date, k: int) -> dt.date:
    while k > 0:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            k -= 1
    return d


# --- split detection ---------------------------------------------------------------------------------------------
HIT_COLUMNS = ["ysym", "d0", "d1", "etf", "price_ratio", "share_ratio_adj", "q_holdings", "q_yahoo", "trigger",
               "q_applied", "applied", "note"]


def detect_splits(snap: pd.DataFrame, yahoo_splits: dict | None = None, yahoo_close=None) -> pd.DataFrame:
    """Every detection between consecutive holdings files of an ETF (one row per ysym and file pair).

    q = new shares per old share (2.0 = 2-for-1, 0.1 = 1-for-10), as in yahoo_ohlcv.split. Trigger 'holdings' =
    price ratio r departs from 1 by > 40% AND |r * s_adj - 1| <= 5% (s_adj = share ratio / fund-wide median share
    ratio of the two files). Trigger 'yahoo' = a Yahoo split dated in (d0, d1]. Applied q: Yahoo's when it has one and
    the holdings price ratio is consistent with it (within 2x), else the holdings-implied q for a holdings trigger.
    Trigger 'window' (needs `yahoo_close(ysym) -> close Series`): over a name's own consecutive observations in one ETF
    (also across files that omit the name, <= 45 days apart) the holdings price ratio r and the Yahoo close ratio c
    disagree by > 1.5x and q = c / r is within 3% of a round split ratio, and the ETF share count moved by about q (within 25%). Skipped where a hit already covers the pair.
    """
    yahoo_splits = yahoo_splits or {}
    rows = []
    for etf, g in snap.groupby("etf"):
        dates = sorted(g["as_of"].unique())
        files = {d: f.sort_values("market_value", ascending=False, na_position="last").drop_duplicates("ysym")
                    .set_index("ysym") for d, f in g.groupby("as_of")}
        for d0, d1 in zip(dates[:-1], dates[1:]):
            a, b = files[d0], files[d1]
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            r = b.loc[common, "price"] / a.loc[common, "price"]
            s = b.loc[common, "shares"] / a.loc[common, "shares"]
            sadj = s / float(s.median())
            ysplit = {}
            for sym in common:
                sp = yahoo_splits.get(sym)
                if sp is not None and len(sp):
                    inn = sp[(sp.index > pd.Timestamp(d0)) & (sp.index <= pd.Timestamp(d1))]
                    if len(inn):
                        ysplit[sym] = float(np.prod(inn.to_numpy(dtype=float)))
            trig = ((r - 1).abs() > PRICE_JUMP) & ((r * sadj - 1).abs() <= INVERSE_TOL)
            for sym in set(common[trig.to_numpy()]) | set(ysplit):
                rv, sv = float(r[sym]), float(sadj[sym])
                qh = snap_nice(sv) if bool(trig[sym]) else None
                qy = ysplit.get(sym)
                note = ""
                if qy is not None and abs(math.log(rv * qy)) <= YAHOO_CONSISTENT_LN and abs(math.log(rv * qy)) < abs(math.log(rv)) and (qh is not None or abs(math.log(qy)) >= YAHOO_MIN_LN):
                    q, applied = qy, True
                elif qh is not None and not is_nice(qh) and abs(math.log(rv)) < NONROUND_MIN_LN:
                    q, applied = qh, False
                    note = "non-round share ratio with a small price move (index share-cap change?): detected, not applied; review"
                elif qh is not None:
                    q, applied = qh, True
                    if qy is not None:
                        note = "Yahoo split inconsistent with holdings price ratio; holdings-implied q used"
                else:
                    q, applied = qy, False
                    note = "Yahoo split not applied (ratio inside 1.5x, or not visible in holdings prices)"
                rows.append({"ysym": sym, "d0": d0, "d1": d1, "etf": etf, "price_ratio": rv, "share_ratio_adj": sv,
                             "q_holdings": qh, "q_yahoo": qy,
                             "trigger": "both" if (qh is not None and qy is not None) else
                                        ("holdings" if qh is not None else "yahoo"),
                             "q_applied": q, "applied": applied, "note": note})
    if yahoo_close is not None:
        rows.extend(_window_hits(snap, yahoo_close, rows))
    df = pd.DataFrame(rows, columns=HIT_COLUMNS)
    if df.empty:
        return df
    # XIC and XCS see the same corporate action: keep one row per (ysym, d0, d1), preferring an applied one
    df = df.sort_values(["ysym", "d0", "applied", "etf"], ascending=[True, True, False, True])
    return df.drop_duplicates(["ysym", "d0", "d1"]).sort_values(["d0", "ysym"]).reset_index(drop=True)


def _window_hits(snap: pd.DataFrame, yahoo_close, found: list) -> list:
    """The 'window' trigger of detect_splits. Round factors only; WPRT/HUT style names (Yahoo closes a later consolidation
    multiple of the holdings price, same return) give q = 1 and never trigger."""
    covered = {}
    for h in found:
        if h["applied"]:
            covered.setdefault(h["ysym"], []).append((h["d0"], h["d1"]))
    rows = []
    for (etf, sym), g in snap.sort_values("as_of").drop_duplicates(["etf", "ysym", "as_of"]).groupby(["etf", "ysym"]):
        if len(g) < 2:
            continue
        c = yahoo_close(sym)
        if c is None:
            continue
        d, p, sh = g["as_of"].tolist(), g["price"].to_numpy(dtype=float), g["shares"].to_numpy(dtype=float)
        for i in range(len(d) - 1):
            d0, d1 = d[i], d[i + 1]
            if (d1 - d0).days > WINDOW_MAX_DAYS or not (p[i] > 0 and p[i + 1] > 0):
                continue
            if any(a0 < d1 and d0 < a1 for a0, a1 in covered.get(sym, ())):
                continue
            r = p[i + 1] / p[i]
            if abs(math.log(r)) < WINDOW_MISMATCH_LN:
                continue
            c0, c1 = c.get(pd.Timestamp(d0)), c.get(pd.Timestamp(d1))
            if c0 is None or c1 is None or not (c0 > 0 and c1 > 0):
                continue
            q = float(c1) / float(c0) / r
            j = min(NICE, key=lambda v: abs(math.log(q / v)))
            if abs(math.log(q)) < WINDOW_MISMATCH_LN or abs(q / j - 1) > WINDOW_NICE_TOL:
                continue
            if not (sh[i] > 0 and sh[i + 1] > 0) or abs(math.log(sh[i + 1] / sh[i] / j)) > WINDOW_SHARES_LN:
                continue
            rows.append({"ysym": sym, "d0": d0, "d1": d1, "etf": etf, "price_ratio": r,
                         "share_ratio_adj": float(sh[i + 1] / sh[i]),
                         "q_holdings": None, "q_yahoo": None, "trigger": "window", "q_applied": j, "applied": True,
                         "note": f"holdings ratio {r:.4g} vs Yahoo ratio {float(c1) / float(c0):.4g}: implied factor {q:.4g} snapped to {j:.4g}"})
            covered.setdefault(sym, []).append((d0, d1))     # the same action seen via the other ETF's pairs is not applied twice
    return rows


# --- Yahoo --------------------------------------------------------------------------------------------------------
class YahooData:
    """frames: {ysym: DataFrame indexed by Timestamp with open/high/low/close/adj_close/volume/dividend/split};
    firsts: {ysym: first stored date (whole history)}. Splits cover ALL stored dates, also after any window."""

    def __init__(self, frames: dict, firsts: dict | None = None):
        self.frames = {s: f.sort_index() for s, f in frames.items()}
        self.firsts = firsts or {s: f.index.min() for s, f in self.frames.items() if len(f)}
        self.splits = {}
        for s, f in self.frames.items():
            sp = f["split"].fillna(0.0) if "split" in f else pd.Series(dtype=float)
            sp = sp[sp > 0]
            if len(sp):
                self.splits[s] = sp

    def close(self, ysym):
        f = self.frames.get(ysym)
        return None if f is None else f["close"][f["close"] > 0]

    def close_on(self, ysym, d):
        c = self.close(ysym)
        if c is None:
            return None
        v = c.get(pd.Timestamp(d))
        return None if v is None or not np.isfinite(v) else float(v)


def load_yahoo(conn, symbols: list[str], start="2011-06-01") -> YahooData:
    """Full OHLCV frames from `start` on (all later dates, so splits after any window are seen), plus whole-history
    first dates. Read-only, %s placeholders."""
    from alpha101.data.yahoo import read_long
    frames, firsts = {}, {}
    for i in range(0, len(symbols), 200):
        part = symbols[i:i + 200]
        ph = ",".join(["%s"] * len(part))
        with conn.cursor() as cur:
            cur.execute(f"SELECT symbol, MIN(date) FROM yahoo_ohlcv WHERE symbol IN ({ph}) GROUP BY symbol", part)
            firsts.update({s: pd.Timestamp(d) for s, d in cur.fetchall()})
        d = read_long(conn, part, start)
        for s, g in d.groupby("symbol"):
            frames[s] = g.set_index("date")[["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]]
    return YahooData(frames, firsts)


# --- the panel ---------------------------------------------------------------------------------------------------
class PricePanel:
    """Holdings snapshots + Yahoo + split hits, with the study's price lookups (exit hierarchy of spec 1.4)."""

    def __init__(self, snap: pd.DataFrame, cal, yahoo: YahooData, hits: pd.DataFrame | None = None,
                 takeovers: pd.DataFrame | None = None, excluded: set | None = None):
        if snap["as_of"].min() < HOLDOUT_FLOOR:
            raise ValueError("snapshots dated before 2011-02-01 are the sealed 2007-2011 M-series inputs: not allowed in pass 1")
        self.snap, self.cal, self.yahoo = snap, cal, yahoo
        self.hits = detect_splits(snap, yahoo.splits, yahoo.close) if hits is None else hits
        self.takeovers = takeovers if takeovers is not None else pd.DataFrame(columns=["year", "ysym", "cash_per_share", "source_url"])
        self.excluded = excluded or set()                              # {(year, ysym)} cleared as "exclude"
        self.files = {}                                                # (etf, date) -> Series ysym -> price
        for (etf, d), g in snap.groupby(["etf", "as_of"]):
            self.files[(etf, d)] = g.sort_values("market_value", ascending=False, na_position="last") \
                .drop_duplicates("ysym").set_index("ysym")["price"]
        self.dates = sorted({d for _, d in self.files})
        self.by_sym = {s: g.sort_values("as_of")[["as_of", "price", "etf"]] for s, g in snap.groupby("ysym")}
        self._applied = {}
        if len(self.hits):
            for h in self.hits[self.hits["applied"]].itertuples(index=False):
                self._applied.setdefault(h.ysym, []).append((h.d1, float(h.q_applied)))
        self._ident = {}

    # file access --------------------------------------------------------------------------------------------------
    def has_file(self, d) -> bool:
        return d in self._dateset()

    def _dateset(self):
        if not hasattr(self, "_ds"):
            self._ds = set(self.dates)
        return self._ds

    def resolve_date(self, d, fallback_days: int = 0):
        """Latest file date on/before d within `fallback_days` weekdays (month-end roles only), else None."""
        for k in range(fallback_days + 1):
            dd = d if k == 0 else _bdays_before(d, k)
            if dd in self._dateset():
                return dd
        return None

    def holdings_price(self, ysym, d, pref_etf: str | None = None):
        """(price, etf) from the file dated d: preferred ETF first, then the other; (None, None) if absent."""
        order = [pref_etf] + [e for e in ("XIC", "XCS") if e != pref_etf] if pref_etf else ["XIC", "XCS"]
        for e in order:
            s = self.files.get((e, d))
            if s is not None and ysym in s.index:
                return float(s[ysym]), e
        return None, None

    def last_observed(self, ysym, d_from, d_to):
        """(price, date) of the latest holdings row of ysym in [d_from, d_to] (either ETF), else (None, None)."""
        g = self.by_sym.get(ysym)
        if g is None:
            return None, None
        g = g[(g["as_of"] >= d_from) & (g["as_of"] <= d_to)]
        if g.empty:
            return None, None
        r = g.iloc[-1]
        return float(r["price"]), r["as_of"]

    def split_factor(self, ysym, d0, d1) -> float:
        """Product of applied q for splits whose detection interval ends in (d0, d1]: value of one d0 share at d1 prices."""
        f = 1.0
        for dd, q in self._applied.get(ysym, ()):
            if d0 < dd <= d1:
                f *= q
        return f

    def hits_between(self, d0, d1) -> pd.DataFrame:
        return self.hits[(self.hits["d1"] > d0) & (self.hits["d1"] <= d1)] if len(self.hits) else self.hits

    # Yahoo identity ----------------------------------------------------------------------------------------------
    def identity(self, ysym, d_prev, d_end) -> bool:
        key = (ysym, d_prev, d_end)
        if key not in self._ident:
            g = self.by_sym.get(ysym)
            if g is None or ysym not in self.yahoo.frames:
                self._ident[key] = False
            else:
                h = g[(g["as_of"] >= d_prev) & (g["as_of"] <= d_end)]
                res = yahootopup.identity_ok(ysym, h, self.yahoo.close(ysym), self.yahoo.splits.get(ysym),
                                             self.yahoo.firsts.get(ysym), d_prev)
                self._ident[key] = bool(res["ok"])
        return self._ident[key]

    # dividends -----------------------------------------------------------------------------------------------------
    def dividend_yield(self, ysym, d0, d1) -> float:
        """Sum of Yahoo dividend / close of the day before the ex-date, ex-dates in (d0, d1]. Yahoo basis only."""
        f = self.yahoo.frames.get(ysym)
        if f is None:
            return 0.0
        div = f["dividend"].fillna(0.0)
        div = div[(div > 0) & (div.index > pd.Timestamp(d0)) & (div.index <= pd.Timestamp(d1))]
        if div.empty:
            return 0.0
        close = f["close"]
        tot = 0.0
        for ex, v in div.items():
            prev = close[close.index < ex]
            prev = prev[prev > 0]
            if len(prev):
                tot += float(v) / float(prev.iloc[-1])
        return tot

    # the trade -----------------------------------------------------------------------------------------------------
    def entry_price(self, ysym, pref_etf, d_entry, d_ref, ref_price, d_prev, d_end):
        """(price, source). Holdings file at d_entry (own ETF, then the other); if there is no usable holdings price, a
        Yahoo-ratio price from the rank-day holdings price (identity-checked); else (None, 'none')."""
        p, e = self.holdings_price(ysym, d_entry, pref_etf)
        if p is not None:
            return p, f"holdings_{e.lower()}"
        if ref_price and self.identity(ysym, d_prev, d_end):
            c1, c0 = self.yahoo.close_on(ysym, d_entry), self.yahoo.close_on(ysym, d_ref)
            if c1 and c0:
                return ref_price * c1 / c0, "yahoo"
        return None, "none"

    def trade(self, year, ysym, pref_etf, d_entry, d_exit, d_ref, ref_price, d_prev, d_end, unresolved="base",
              entry=None) -> dict:
        """One name's buy at d_entry and sell at d_exit. unresolved: 'base' (last holdings price x 0.7), 'zero' (0%),
        'wipe' (-100%). Returns dict with ok, ret (gross price + dividend yield), classes and prices.
        Exit classes: holdings (own ETF), other_etf (1.4 a), yahoo (b), takeover (c), unresolved (d)."""
        out = {"ysym": ysym, "ok": False, "ret": np.nan, "entry_price": None, "entry_source": "none", "exit_price": None,
               "exit_class": "no_entry", "exit_source": "", "yahoo_return": np.nan, "yahoo_close_entry": np.nan,
               "yahoo_close_exit": np.nan, "split_factor": 1.0, "div_yield": 0.0, "last_obs_date": None,
               "entry_date": d_entry, "exit_date": d_exit}
        pe, src = entry if entry is not None else self.entry_price(ysym, pref_etf, d_entry, d_ref, ref_price, d_prev, d_end)
        out["entry_price"], out["entry_source"] = pe, src
        if pe is None or pe <= 0:
            return out
        ident = self.identity(ysym, d_prev, d_end)
        c_e, c_x = self.yahoo.close_on(ysym, d_entry), self.yahoo.close_on(ysym, d_exit)
        if ident and c_e and c_x:
            out["yahoo_close_entry"], out["yahoo_close_exit"], out["yahoo_return"] = c_e, c_x, c_x / c_e - 1
        out["ok"] = True
        px, ex = self.holdings_price(ysym, d_exit, pref_etf)
        if px is not None:
            q = self.split_factor(ysym, d_entry, d_exit)
            ret = q * px / pe - 1
            out.update(exit_price=px, exit_class="holdings" if ex == pref_etf else "other_etf",
                       exit_source=f"holdings_{ex.lower()}", split_factor=q)
            out["div_yield"] = self.dividend_yield(ysym, d_entry, d_exit) if ident else 0.0
        elif ident and c_e and c_x:
            ret = c_x / c_e - 1
            out.update(exit_price=pe * (1 + ret), exit_class="yahoo", exit_source="yahoo")
            out["div_yield"] = self.dividend_yield(ysym, d_entry, d_exit)
        else:
            tk = self.takeovers[(self.takeovers["year"] == year) & (self.takeovers["ysym"] == ysym)
                                & self.takeovers["source_url"].fillna("").astype(str).str.len().gt(0)] \
                if len(self.takeovers) else self.takeovers
            lp, ld = self.last_observed(ysym, min(d_ref, d_entry), d_exit)
            out["last_obs_date"] = ld
            if len(tk):
                cash = tk.iloc[0]["cash_per_share"]
                cash = float(cash) if pd.notna(cash) and str(cash) != "" else lp
                q = self.split_factor(ysym, d_entry, ld) if ld else 1.0
                ret = q * cash / pe - 1
                out.update(exit_price=cash, exit_class="takeover", exit_source="takeover", split_factor=q)
            else:
                q = self.split_factor(ysym, d_entry, ld) if ld else 1.0
                if unresolved == "zero":
                    ret = 0.0
                elif unresolved == "wipe":
                    ret = -1.0
                else:
                    ret = q * (lp if lp else pe) * (1 - UNRESOLVED_HAIRCUT) / pe - 1
                out.update(exit_price=(lp * (1 - UNRESOLVED_HAIRCUT) if lp else None), exit_class="unresolved",
                           exit_source="unresolved", split_factor=q)
        out["ret"] = ret + out["div_yield"]
        return out


# --- big_moves gate -----------------------------------------------------------------------------------------------
def read_big_moves(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        return pd.DataFrame(columns=BIG_COLUMNS)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BIG_COLUMNS)
    for c in BIG_COLUMNS:
        if c not in df:
            df[c] = ""
    return df


def uncleared(df: pd.DataFrame) -> pd.DataFrame:
    """Rows without clearance keep/exclude and a written reason."""
    if df.empty:
        return df
    ok = df["clearance"].str.strip().str.lower().isin(CLEARANCES) & df["reason"].str.strip().ne("")
    return df[~ok]


def merge_big_moves(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Carry hand-written clearance/reason from `old` over to the regenerated `new` rows (key: year, ysym)."""
    new = new.copy()
    if len(old):
        prev = {(str(r.year), r.ysym): (r.clearance, r.reason) for r in old.itertuples(index=False)}
        for i, r in new.iterrows():
            c, why = prev.get((str(r["year"]), r["ysym"]), ("", ""))
            new.at[i, "clearance"], new.at[i, "reason"] = c, why
    return new.sort_values(["year", "ysym"]).reset_index(drop=True)


def excluded_names(df: pd.DataFrame) -> set:
    """{(year, ysym)} cleared as 'exclude'."""
    if df.empty:
        return set()
    e = df[df["clearance"].str.strip().str.lower() == "exclude"]
    return {(int(r.year), r.ysym) for r in e.itertuples(index=False)}


class GateError(RuntimeError):
    """decide() refused: big_moves.csv has uncleared rows."""
