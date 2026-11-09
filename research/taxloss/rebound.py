"""Signal, costs, benchmarks, placebos, statistics and the pre-registered decision (spec 2-5).

`rank_ytd` is the only function that sees ranking data and it truncates its inputs at the rank day itself. Everything
that needs later data (entry/exit prices, the L-day volume filter, spreads) lives outside it.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats as sst

import prices as pr
import yahootopup

SEED = 20261019
N_BASKET = 20
BOOK = 100_000.0
COMMISSION = 9.99
S_CLOSE_BP = 10.0
HALF_TICK = 0.005
FALLBACK_H_BP = {("XIC", 0): 40, ("XIC", 1): 15, ("XIC", 2): 5, ("XCS", 0): 60, ("XCS", 1): 30, ("XCS", 2): 15}
MIN_BUCKET = 5
SPREAD_WINDOW = 60
MIN_SPREAD_DAYS = 20
T_ADVANCE, T_GROSS_V1, T_REJECT = 2.0, 1.771, 1.0


# --- cells and scenarios -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    id: str
    rank_role: str = "R"
    entry_role: str = "L"
    exit_role: str = "X"
    n: int = N_BASKET
    floor: float = 2.0


CELLS = [Cell("P"), Cell("V-E0", "L-6", "L-5"), Cell("V-E2", entry_role="first_jan"), Cell("V-X1", exit_role="X1"),
         Cell("V-X3", exit_role="X3"), Cell("V-N10", n=10), Cell("V-N40", n=40), Cell("V-P1", floor=1.0),
         Cell("V-P5", floor=5.0)]


@dataclass(frozen=True)
class Scenario:
    name: str = "base"
    unresolved: str = "base"        # base | zero | wipe
    e2h: bool = False               # entry side pays the full spread 2h
    commission: float = COMMISSION
    s_close_bp: float = S_CLOSE_BP
    hmult: float = 1.0
    book: float = BOOK


BASE = Scenario()


def event_dates(cal, y: int) -> dict:
    e = cal.event_row(y)
    return {"D_prev": e["D_prev"], "R": e["R"], "L-6": e["VE0_rank_day"], "L": e["L"], "L-5": e["VE0_entry"],
            "first_jan": e["first_jan_day"], "X": e["jan_exit_X"], "X1": e["VX1_5th_jan_day"],
            "X3": e["VX3_on_or_after_feb15"], "pre_review": e["pre_review_day"]}


# --- ranking (the only function that sees ranking data) ---------------------------------------------------------
def rank_ytd(snap: pd.DataFrame, hits: pd.DataFrame, d_prev, rank_day, floor: float = 2.0) -> pd.DataFrame:
    """Eligible names (2.3 rules 1, 2, 4) sorted by YTD ascending (worst first). Rows dated after `rank_day` are
    dropped on entry, so the output cannot depend on them. YTD = P(R) * split factor / P(D_prev) - 1 with the
    applied q of every hit whose file pair ends in (D_prev, R]. Rule 3 (halts) needs the L close and is applied
    outside, in `halted_names`."""
    snap = snap[snap["as_of"] <= rank_day]
    hits = hits[hits["d1"] <= rank_day] if len(hits) else hits
    r_used = yahootopup.resolve_R(snap, rank_day)
    el = yahootopup.eligible_universe(snap, d_prev, r_used, floor)
    if el.empty:
        return el.assign(ytd=[], R_used=[])
    fac = {}
    if len(hits):
        for h in hits[hits["applied"] & (hits["d1"] > d_prev) & (hits["d1"] <= r_used)].itertuples(index=False):
            fac[h.ysym] = fac.get(h.ysym, 1.0) * float(h.q_applied)
    el = el.copy()
    el["ytd"] = el["price_R"] * el["ysym"].map(fac).fillna(1.0) / el["price_D"] - 1
    el["R_used"] = r_used
    el = el.drop(columns=["guard_ok"])                  # the crude 1.8x guard is replaced by the split adjustment
    return el.sort_values(["ytd", "ysym"], kind="mergesort").reset_index(drop=True)


def halted_names(panel, names, rank_day, entry_day, d_prev, d_end) -> set:
    """Rule 2.3.3, applied outside the ranking: where an identity-checked Yahoo series exists (and spans the date),
    volume must be > 0 on the rank day and the entry day. Names without Yahoo data are kept ('no_volume_data')."""
    out = set()
    for s in names:
        f = panel.yahoo.frames.get(s)
        if f is None or not panel.identity(s, d_prev, d_end):
            continue
        last = f.index.max()
        for d in (rank_day, entry_day):
            ts = pd.Timestamp(d)
            if ts > last:
                continue
            v = f["volume"].get(ts)
            if v is None or not np.isfinite(v) or v <= 0:
                out.add(s)
                break
    return out


def basket(ranked: pd.DataFrame, n: int, halted: set | None = None) -> pd.DataFrame:
    """The n lowest-YTD eligible names that are not halted."""
    r = ranked if not halted else ranked[~ranked["ysym"].isin(halted)]
    return r.head(n)


# --- spreads and costs ---------------------------------------------------------------------------------------------
def abdi_ranaldo(high, low, close) -> float:
    """Abdi & Ranaldo (2017) close-high-low full-spread estimate over the arrays given (numpy slicing, no look-forward
    shifts): S^2 = 4 mean[(c_t - eta_t)(c_t - eta_{t+1})], S = sqrt(max(S^2, 0)). NaN if fewer than 20 days."""
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    ok = (h > 0) & (l > 0) & (c > 0)
    h, l, c = h[ok], l[ok], c[ok]
    if len(c) < MIN_SPREAD_DAYS:
        return float("nan")
    eta, lc = np.log((h + l) / 2), np.log(c)
    s2 = 4 * np.mean((lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:]))
    return float(math.sqrt(max(s2, 0.0)))


def corwin_schultz(high, low) -> float:
    """Corwin & Schultz (2012) two-day high-low full-spread estimate, averaged over consecutive pairs (floored at 0)."""
    h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    ok = (h > 0) & (l > 0)
    h, l = h[ok], l[ok]
    if len(h) < MIN_SPREAD_DAYS:
        return float("nan")
    k = 3 - 2 * math.sqrt(2)
    beta = np.log(h[:-1] / l[:-1]) ** 2 + np.log(h[1:] / l[1:]) ** 2
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(l[:-1], l[1:])) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.mean(np.maximum(s, 0.0)))


def price_band(p: float) -> int:
    return 0 if p < 5 else 1 if p < 20 else 2


def half_spreads(panel, universe: pd.DataFrame, rank_day, d_prev, d_end) -> pd.DataFrame:
    """Per-name half-spread h (fraction of price). Own value = S_AR / 2 over the 60 trading days ending at the rank day
    (identity-checked Yahoo OHLC), clamped to half a tick / price; names without it get their bucket's median
    ({XIC, XCS-only} x price band), or the fallback table if the bucket has fewer than 5 names with OHLC."""
    rows = []
    for u in universe.itertuples(index=False):
        own = cs = float("nan")
        f = panel.yahoo.frames.get(u.ysym)
        if f is not None and panel.identity(u.ysym, d_prev, d_end):
            w = f[f.index <= pd.Timestamp(rank_day)].tail(SPREAD_WINDOW)
            s = abdi_ranaldo(w["high"], w["low"], w["close"])
            if np.isfinite(s):
                own = max(s / 2, HALF_TICK / u.price_R)
                c2 = corwin_schultz(w["high"], w["low"])
                cs = max(c2 / 2, HALF_TICK / u.price_R) if np.isfinite(c2) else float("nan")
        rows.append({"ysym": u.ysym, "bucket_etf": "XIC" if u.etf == "XIC" else "XCS", "band": price_band(u.price_R),
                     "h_own": own, "h_cs": cs})
    d = pd.DataFrame(rows, columns=["ysym", "bucket_etf", "band", "h_own", "h_cs"])
    med = d.dropna(subset=["h_own"]).groupby(["bucket_etf", "band"])["h_own"].agg(["median", "size"])
    h, src = [], []
    for r in d.itertuples(index=False):
        if np.isfinite(r.h_own):
            h.append(r.h_own)
            src.append("own")
            continue
        k = (r.bucket_etf, r.band)
        if k in med.index and med.at[k, "size"] >= MIN_BUCKET:
            h.append(float(med.at[k, "median"]))
            src.append("bucket")
        else:
            h.append(FALLBACK_H_BP[k] / 1e4)
            src.append("fallback")
    d["h"], d["h_source"] = h, src
    return d


def per_side_cost(h, scen: Scenario, n: int, entry: bool) -> np.ndarray:
    """c = commission + half-spread + timing, as a fraction of the ticket, for one side (entry pays 2h under E2h)."""
    comm = scen.commission / (scen.book / n)
    hh = np.asarray(h, dtype=float) * scen.hmult * (2.0 if (entry and scen.e2h) else 1.0)
    return comm + hh + scen.s_close_bp / 1e4


def apply_scenario(tbl: pd.DataFrame, scen: Scenario, n: int) -> pd.DataFrame:
    """Add gross return under the scenario's unresolved mode and the net return after costs (needs column h)."""
    t = tbl.copy()
    ret = t["ret"].to_numpy(dtype=float).copy()
    unr = (t["exit_class"] == "unresolved").to_numpy()
    if scen.unresolved == "zero":
        ret[unr] = 0.0
    elif scen.unresolved == "wipe":
        ret[unr] = -1.0
    t["ret_s"] = ret
    cin, cout = per_side_cost(t["h"], scen, n, True), per_side_cost(t["h"], scen, n, False)
    t["net_s"] = (1 + ret) * (1 - cout) / (1 + cin) - 1
    return t


# --- one window (year x cell, placebo month, M series) -----------------------------------------------------------
def build_window(panel, year: int, d_prev, rank_day, entry_day, exit_day, floor: float, id_end, excluded=frozenset(),
                 with_h: bool = True) -> pd.DataFrame:
    """Universe table for one window: ranked eligible names (minus halted and cleared exclusions), their trades
    (base unresolved rule) and half-spreads. Sorted worst YTD first; `rank` is the 1-based position."""
    ranked = rank_ytd(panel.snap, panel.hits, d_prev, rank_day, floor)
    if ranked.empty:
        return ranked
    ranked = ranked[[(year, s) not in excluded for s in ranked["ysym"]]]
    halted = halted_names(panel, ranked["ysym"], rank_day, entry_day, d_prev, id_end)
    ranked = ranked[~ranked["ysym"].isin(halted)].reset_index(drop=True)
    rows = []
    for u in ranked.itertuples(index=False):
        t = panel.trade(year, u.ysym, u.etf, entry_day, exit_day, u.R_used, u.price_R, d_prev, id_end)
        rows.append(t)
    t = pd.DataFrame(rows)
    w = pd.concat([ranked.reset_index(drop=True), t.drop(columns=["ysym"])], axis=1)
    w["rank"] = np.arange(1, len(w) + 1)
    if with_h:
        hs = half_spreads(panel, ranked, rank_day, d_prev, id_end)
        w = w.merge(hs[["ysym", "h", "h_cs", "h_source"]], on="ysym", how="left")
    return w


def basket_rows(w: pd.DataFrame, n: int) -> pd.DataFrame:
    """The basket = n lowest YTD of the window universe; names with no entry price stay in (ok False) and are dropped
    from the means, never replaced."""
    return w.head(n)


def mean_ret(w: pd.DataFrame, col: str) -> float:
    x = w.loc[w["ok"], col]
    return float(x.mean()) if len(x) else float("nan")


CLASSES = ["holdings", "other_etf", "yahoo", "takeover", "unresolved", "no_entry"]


def class_shares(w: pd.DataFrame) -> dict:
    n = max(len(w), 1)
    return {f"share_{c}": float((w["exit_class"] == c).sum()) / n for c in CLASSES}


# --- Study: all windows, cached ----------------------------------------------------------------------------------
class Study:
    def __init__(self, panel, cal, years=range(2012, 2026), excluded=frozenset()):
        self.panel, self.cal, self.years, self.excluded = panel, cal, list(years), frozenset(excluded)
        self._w = {}

    def dates(self, y):
        return event_dates(self.cal, y)

    def window(self, y: int, cell: Cell) -> pd.DataFrame:
        key = (y, cell.rank_role, cell.entry_role, cell.exit_role, cell.floor)
        if key not in self._w:
            d = self.dates(y)
            self._w[key] = build_window(self.panel, y, d["D_prev"], d[cell.rank_role], d[cell.entry_role],
                                        d[cell.exit_role], cell.floor, d["X3"], self.excluded)
        return self._w[key]

    def partial(self, y: int, cell: Cell) -> bool:
        d = self.dates(y)
        return not (self.panel.has_file(d[cell.entry_role]) and self.panel.has_file(d[cell.exit_role]))

    def xic_tr(self, y, cell: Cell):
        d = self.dates(y)
        return adj_ratio(self.panel.yahoo, "XIC.TO", d[cell.entry_role], d[cell.exit_role])

    def cell_event(self, y: int, cell: Cell, scen: Scenario = BASE) -> dict:
        w = self.window(y, cell)
        if w.empty:
            return {"year": y, "cell": cell.id, "n_universe": 0}
        t = apply_scenario(w, scen, cell.n)
        b = basket_rows(t, cell.n)
        bg, bn, eg = mean_ret(b, "ret_s"), mean_ret(b, "net_s"), mean_ret(t, "ret_s")
        xic = self.xic_tr(y, cell)
        return {"year": y, "cell": cell.id, "scenario": scen.name, "n_universe": int(t["ok"].sum()),
                "n_basket": int(b["ok"].sum()), "n_basket_no_entry": int((~b["ok"]).sum()),
                "basket_gross": bg, "basket_net": bn, "ew_gross": eg, "g": bg - eg, "g_net": bn - eg,
                "xic_tr": xic, "n_y": bn - xic if xic is not None else float("nan"),
                "partial_exit": self.partial(y, cell), **class_shares(b)}

    def cell_table(self, cell: Cell, scen: Scenario = BASE) -> pd.DataFrame:
        return pd.DataFrame([self.cell_event(y, cell, scen) for y in self.years])


def adj_ratio(yahoo: pr.YahooData, ysym: str, d0, d1):
    """Total-return ratio - 1 from Yahoo adj_close, None if either date is missing."""
    f = yahoo.frames.get(ysym)
    if f is None:
        return None
    a, b = f["adj_close"].get(pd.Timestamp(d0)), f["adj_close"].get(pd.Timestamp(d1))
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return float(b / a - 1)


def sanity_reproduction(study: Study) -> dict:
    """For names that have both, the window return from holdings prices vs the Yahoo close ratio (P cell, names
    exiting via a holdings file with an identity-checked Yahoo series). Median absolute difference, dividends excluded."""
    d = []
    for y in study.years:
        w = study.window(y, CELLS[0])
        if w.empty:
            continue
        x = w[w["ok"] & (w["exit_class"] == "holdings") & w["yahoo_return"].notna()]
        d += list(np.abs((x["ret"] - x["div_yield"]) - x["yahoo_return"]))
    return {"n": len(d), "median_abs_diff": float(np.median(d)) if d else float("nan")}


# --- gate: big_moves ---------------------------------------------------------------------------------------------
def big_moves(study: Study, cells=CELLS) -> pd.DataFrame:
    """Every universe (hence basket) name whose |window return| > 100% in any cell: one row per (year, name), listing
    the cells and carrying the prices of the cell with the largest move."""
    hits = {}
    for y in study.years:
        for c in cells:
            w = study.window(y, c)
            if w.empty:
                continue
            for r in w[w["ok"] & (w["ret"].abs() > pr.BIG_MOVE)].itertuples(index=False):
                hits.setdefault((y, r.ysym), []).append((c.id, "basket" if r.rank <= c.n else "universe", r))
    rows = []
    for (y, sym), lst in sorted(hits.items()):
        cid, scope, r = max(lst, key=lambda t: abs(t[2].ret))
        rows.append({"year": y, "ysym": sym, "name": r.name, "cells": "|".join(t[0] for t in lst),
                     "scope": "basket" if any(t[1] == "basket" for t in lst) else "universe",
                     "max_window_return": r.ret, "entry_date": r.entry_date, "exit_date": r.exit_date,
                     "entry_price": r.entry_price, "exit_price": r.exit_price, "exit_class": r.exit_class,
                     "exit_source": f"{r.entry_source}->{r.exit_source}", "yahoo_return": r.yahoo_return,
                     "yahoo_close_entry": r.yahoo_close_entry, "yahoo_close_exit": r.yahoo_close_exit,
                     "split_factor": r.split_factor, "clearance": "", "reason": ""})
    return pd.DataFrame(rows, columns=pr.BIG_COLUMNS)


# --- placebos ----------------------------------------------------------------------------------------------------
def random_placebo(per_year: dict, n: int = N_BASKET, draws: int = 1000, seed: int = SEED) -> dict:
    """per_year: {year: (net_returns_array, gross_returns_array)} over each year's eligible universe (names with a
    trade). Each draw takes n names without replacement per year, independently across years. Returns per-draw
    per-year net and gross excess (basket minus the EW universe's gross mean), shape (draws, years)."""
    rng = np.random.default_rng(seed)
    years = sorted(per_year)
    gn, gg = np.empty((draws, len(years))), np.empty((draws, len(years)))
    for j, y in enumerate(years):
        net, gross = per_year[y]
        m = len(net)
        k = min(n, m)
        idx = np.argsort(rng.random((draws, m)), axis=1)[:, :k]
        ew = float(np.mean(gross))
        gn[:, j] = net[idx].mean(axis=1) - ew
        gg[:, j] = gross[idx].mean(axis=1) - ew
    return {"years": years, "g_net": gn, "g": gg}


def percentile(draw_means: np.ndarray, actual: float) -> float:
    """Mid-rank percentile of `actual` among the draw means."""
    d = np.asarray(draw_means)
    return float(100 * ((d < actual).mean() + 0.5 * (d == actual).mean()))


def placebo_inputs(study: Study, scen: Scenario = BASE) -> dict:
    out = {}
    for y in study.years:
        w = study.window(y, CELLS[0])
        if w.empty:
            continue
        t = apply_scenario(w, scen, N_BASKET)
        t = t[t["ok"]]
        out[y] = (t["net_s"].to_numpy(), t["ret_s"].to_numpy())
    return out


def random_placebo_summary(pl: dict, events: pd.DataFrame) -> dict:
    """Percentile of the actual P mean (net and gross) and per-event percentiles."""
    ev = events.set_index("year")
    cols = [ev.loc[y] for y in pl["years"]]
    a_net = float(np.mean([r["g_net"] for r in cols]))
    a_g = float(np.mean([r["g"] for r in cols]))
    return {"pct_net": percentile(pl["g_net"].mean(axis=1), a_net), "pct_gross": percentile(pl["g"].mean(axis=1), a_g),
            "actual_mean_g_net": a_net, "actual_mean_g": a_g,
            "per_event_pct_net": {int(y): percentile(pl["g_net"][:, j], float(ev.loc[y, "g_net"])) for j, y in enumerate(pl["years"])},
            "per_event_pct_gross": {int(y): percentile(pl["g"][:, j], float(ev.loc[y, "g"])) for j, y in enumerate(pl["years"])},
            "draws": int(pl["g"].shape[0])}


def window_excess(panel, year, d_prev, rank_day, entry_day, exit_day, id_end, n=N_BASKET, floor=2.0, excluded=frozenset()) -> dict:
    """Gross excess (basket - EW universe) of one window with the identical construction (placebo months, M series)."""
    w = build_window(panel, year, d_prev, rank_day, entry_day, exit_day, floor, id_end, excluded, with_h=False)
    if w.empty:
        return {"g": float("nan"), "n_universe": 0, "n_basket": 0, "reason": "empty universe"}
    b = w.head(n)
    return {"g": mean_ret(b, "ret") - mean_ret(w, "ret"), "n_universe": int(w["ok"].sum()), "n_basket": int(b["ok"].sum()),
            "reason": ""}


def placebo_months(study: Study, p_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Months Feb..Oct (placebo) plus December (= P gross g). -> (per year/month rows, per-month summary with
    December's rank). Windows whose files are missing (2017 gap) are skipped and logged in `reason`."""
    p, cal = study.panel, study.cal
    rows = []
    for y in study.years:
        for m in range(2, 11):
            base_d = cal.last_trading_day_of_month(y - 1, m)
            L = cal.placebo_L(y, m)
            rk = cal.shift(L, -1)
            ex_d = cal.last_trading_day_of_month(y, m + 1)
            base = p.resolve_date(base_d, 3)
            exit_f = p.resolve_date(ex_d, 3)
            r_ok = p.resolve_date(rk, 5)
            if base is None or exit_f is None or r_ok is None or not p.has_file(L):
                rows.append({"year": y, "month": m, "g": np.nan, "n_universe": 0, "n_basket": 0,
                             "reason": "missing holdings file(s): " + ",".join(
                                 k for k, v in (("base", base), ("exit", exit_f), ("rank", r_ok), ("entry", L if p.has_file(L) else None)) if v is None)})
                continue
            r = window_excess(p, y, base, rk, L, exit_f, exit_f, excluded=study.excluded)
            rows.append({"year": y, "month": m, **r})
    for r in p_events.itertuples(index=False):
        rows.append({"year": r.year, "month": 12, "g": r.g, "n_universe": r.n_universe, "n_basket": r.n_basket, "reason": ""})
    df = pd.DataFrame(rows)
    s = df.groupby("month").agg(mean_excess=("g", "mean"), n_years=("g", lambda x: int(x.notna().sum()))).reset_index()
    s["rank_of_10"] = s["mean_excess"].rank(ascending=False, method="min")
    s9 = s[s["month"] != 10]
    s["rank_of_9_ex_oct"] = s["month"].map(dict(zip(s9["month"], s9["mean_excess"].rank(ascending=False, method="min"))))
    s["note"] = np.where(s["month"] == 10, "window ends inside the November selling season", "")
    return df, s


def december_first(s: pd.DataFrame) -> bool:
    """December ranks STRICTLY first among the 10 months (a tie is not first)."""
    dec = float(s.loc[s["month"] == 12, "mean_excess"].iloc[0])
    others = s.loc[s["month"] != 12, "mean_excess"].dropna()
    return bool(len(others) and dec > float(others.max()))


def m_series(panel, cal, years, excluded=frozenset(), unmask: bool = False) -> pd.DataFrame:
    """M rule: rank YTD at the Nov-30 month-end file vs the prior Dec-31 file, buy the Dec-31 close, sell the Jan-31
    close. Gross excess only. Event years before 2012 are the sealed holdout and are refused unless unmask=True."""
    years = list(years)
    if not unmask and any(y < 2012 for y in years):
        raise PermissionError("M-series years 2007-2011 are sealed (pass 2 only: --unmask-m)")
    rows = []
    for y in years:
        d_prev = panel.resolve_date(cal.last_trading_day_of_month(y - 1, 12), 3)
        nov = panel.resolve_date(cal.last_trading_day_of_month(y, 11), 3)
        dec = panel.resolve_date(cal.last_trading_day_of_month(y, 12), 3)
        jan = panel.resolve_date(cal.last_trading_day_of_month(y + 1, 1), 3)
        if None in (d_prev, nov, dec, jan):
            rows.append({"year": y, "g": np.nan, "n_universe": 0, "n_basket": 0, "reason": "missing month-end file"})
            continue
        rows.append({"year": y, **window_excess(panel, y, d_prev, nov, dec, jan, jan, excluded=excluded)})
    return pd.DataFrame(rows, columns=["year", "g", "n_universe", "n_basket", "reason"])


def deciles(study: Study) -> pd.DataFrame:
    """Mean gross excess by YTD decile (1 = worst) per event and pooled over events, for P."""
    rows = []
    for y in study.years:
        w = study.window(y, CELLS[0])
        w = w[w["ok"]].reset_index(drop=True)
        if len(w) < 10:
            continue
        ew = float(w["ret"].mean())
        w["decile"] = np.ceil(10 * (np.arange(len(w)) + 1) / len(w)).astype(int)
        for d, g in w.groupby("decile"):
            rows.append({"year": y, "decile": int(d), "n": len(g), "mean_ret": float(g["ret"].mean()),
                         "excess": float(g["ret"].mean()) - ew})
    return pd.DataFrame(rows, columns=["year", "decile", "n", "mean_ret", "excess"])


# --- statistics --------------------------------------------------------------------------------------------------
def t_plain(x) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def t_nw(x, lags: int = 1) -> float:
    from alpha101.stats import newey_west_t
    return newey_west_t(x, lags)


def t_dec(x) -> float:
    """Decision statistic (5.2): the smaller of the plain t and the Newey-West 1-lag t."""
    a, b = t_plain(x), t_nw(x, 1)
    v = [z for z in (a, b) if np.isfinite(z)]
    return min(v) if v else float("nan")


def stats_block(ev: pd.DataFrame, seed: int = SEED, reps: int = 10_000) -> dict:
    """Tests of 5.2 on per-event g and g_net, plus the diversifier numbers (alpha, beta, Sharpe, correlation)."""
    out = {}
    for col in ("g", "g_net"):
        x = ev[col].dropna().to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        boot = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
        try:
            w = float(sst.wilcoxon(x).pvalue)
        except ValueError:
            w = float("nan")
        out[col] = {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t_plain": t_plain(x),
                    "t_nw1": t_nw(x, 1), "t_dec": t_dec(x), "wilcoxon_p": w, "share_positive": float((x > 0).mean()),
                    "boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "lag1_autocorr": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 3 else float("nan")}
    j = ev.dropna(subset=["basket_net", "xic_tr"])
    if len(j) > 3 and j["xic_tr"].std(ddof=1) > 0:
        y, xx = j["basket_net"].to_numpy(float), j["xic_tr"].to_numpy(float)
        beta = float(np.cov(y, xx, ddof=1)[0, 1] / np.var(xx, ddof=1))
        out["capm"] = {"alpha": float(y.mean() - beta * xx.mean()), "beta": beta, "sharpe": float(y.mean() / y.std(ddof=1)),
                       "corr": float(np.corrcoef(y, xx)[0, 1]), "n": len(j)}
    return out


def neff(cell_tables: dict, df: int = 13) -> dict:
    """n_eff of the per-event g_net series across cells (alpha101.stats) and the labelled adjusted bars."""
    from alpha101.stats import adjusted_threshold, effective_number_of_tests
    m = pd.DataFrame({c: t.set_index("year")["g_net"] for c, t in cell_tables.items()})
    corr = np.array(m.corr().fillna(0.0), dtype=float)
    np.fill_diagonal(corr, 1.0)
    ne = effective_number_of_tests(corr)
    z = adjusted_threshold(ne)
    p = 1 - (1 - 0.05) ** (1 / max(ne, 1.0))
    return {"n_cells": int(m.shape[1]), "n_eff": ne, "sidak_two_sided_normal_bar": z,
            "student_t_df13_equivalent": float(sst.t.ppf(1 - p / 2, df)), "df": df,
            "note": "two-sided normal-quantile Sidak bar (alpha101.stats.adjusted_threshold), not a one-sided t bar"}


# --- the decision ------------------------------------------------------------------------------------------------
@dataclass
class Evidence:
    """Inputs of 5.4 for one (scenario, book size)."""
    years: list
    g: np.ndarray
    g_net: np.ndarray
    n: np.ndarray
    pct_net: float
    dec_first: bool
    m_oos: float | None = None          # None = 2007-2011 M series not computed (pass 1: sealed)

    def drop_year(self, year: int) -> "Evidence":
        k = np.array([y != year for y in self.years])
        return replace(self, years=[y for y in self.years if y != year], g=self.g[k], g_net=self.g_net[k], n=self.n[k])


def conditions(e: Evidence) -> dict:
    tn, tg = t_dec(e.g_net), t_dec(e.g)
    c1 = bool(np.mean(e.g_net) > 0 and tn >= T_ADVANCE)
    c2 = bool(e.pct_net >= 95)
    c3 = bool(np.nanmean(e.n) > 0 and int((e.n > 0).sum()) >= 9)
    c4 = bool(e.dec_first)
    c5 = None if e.m_oos is None else bool(e.m_oos > 0)
    reject = bool(np.mean(e.g) <= 0 or tg < T_REJECT or e.pct_net < 75)
    weak = bool(c2 and c4 and tg >= T_GROSS_V1 and (not c1 or not c3))
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "t_dec_net": tn, "t_dec_gross": tg, "reject": reject,
            "weak": weak, "n_events": len(e.g), "n_positive_n": int((e.n > 0).sum())}


def core_verdict(e: Evidence) -> str:
    """REJECT, ADVANCE, WEAK, UNDERPOWERED in that order, ignoring condition 6. A missing M value (pass 1) gives
    ADVANCE_PENDING_M when conditions 1-4 hold."""
    c = conditions(e)
    if c["reject"]:
        return "REJECT"
    if c["c1"] and c["c2"] and c["c3"] and c["c4"]:
        if c["c5"] is None:
            return "ADVANCE_PENDING_M"
        if c["c5"]:
            return "ADVANCE"
    if c["weak"]:
        return "WEAK"
    return "UNDERPOWERED"


def decide(base: Evidence, scenarios: dict | None = None, ev250: Evidence | None = None, big_moves_path=None,
           drop_year: int = 2016) -> dict:
    """Spec 5.4 coded literally. Condition 6: the verdict (ignoring 6) is unchanged without `drop_year`, with
    unresolved exits at -100% ('wipe') and under E2h ('e2h'); `scenarios` supplies those two Evidence objects.
    Refuses (GateError) if big_moves.csv has uncleared rows."""
    if big_moves_path is not None:
        bm = pr.read_big_moves(big_moves_path)
        un = pr.uncleared(bm)
        if len(un):
            raise pr.GateError(f"{len(un)} uncleared row(s) in big_moves.csv")
    scenarios = scenarios or {}
    core = core_verdict(base)
    checks = {"drop_" + str(drop_year): core_verdict(base.drop_year(drop_year)) if drop_year in base.years else core}
    for k, e in scenarios.items():
        checks[k] = core_verdict(e)
    c6 = all(v == core for v in checks.values())
    c = conditions(base)
    if c["reject"]:
        verdict = "REJECT"
    elif c["c1"] and c["c2"] and c["c3"] and c["c4"] and c6 and c["c5"] is True:
        verdict = "ADVANCE"
    elif c["c1"] and c["c2"] and c["c3"] and c["c4"] and c6 and c["c5"] is None:
        verdict = "ADVANCE_PENDING_M"
    elif c["weak"]:
        verdict = "WEAK"
    else:
        verdict = "UNDERPOWERED"
    out = {"verdict": verdict, "conditions": {**c, "c6": c6}, "core_verdict": core, "c6_checks": checks,
           "treated_as": None}
    if verdict == "WEAK" and ev250 is not None:
        c250 = conditions(ev250)
        out["conditions_c250"] = c250
        if c250["c1"] and c250["c3"] and (c["c5"] is not False) and c6:
            out["treated_as"] = "ADVANCE_C250K_ONLY" if c["c5"] else "ADVANCE_C250K_ONLY_PENDING_M"
    return out


def evidence_from(table: pd.DataFrame, pl: dict, dec_first: bool, m_oos=None) -> Evidence:
    t = table.sort_values("year")
    return Evidence(years=list(t["year"]), g=t["g"].to_numpy(float), g_net=t["g_net"].to_numpy(float),
                    n=t["n_y"].to_numpy(float), pct_net=pl["pct_net"], dec_first=dec_first, m_oos=m_oos)
