#!/usr/bin/env python3
"""for a TSX / NYSE-Nasdaq interlisted stock, which line is cheaper to trade?

    python3 scripts/cost_edge_duallisting.py [--snapshot DIR] [--out docs/research_notes/2026-10-23_duallisting_venue_results.md]

Default: fetch fresh Yahoo + Bank of Canada data into research/cost_edge_duallisting/snapshot_<today>/ and recompute
everything from it. `--snapshot DIR` recomputes from an existing snapshot without any network.

The FX cost functions, the spread estimators and the DLR / DLR-U half-spreads are reused from
scripts/cost_edge_norbert.py (imported, not copied) and a local Norbert snapshot (see scripts/cost_edge_norbert.py).

Routes (all costs are ROUND-TRIP, i.e. buy then sell and end in the starting currency, as C$ of the position V):
  a  TSX line from CAD cash              2*h_tsx*V + 2*k_cad
  b  US line from USD cash already held  2*h_us*V + 2*k_usd*fx
  c  US line from CAD, FX desk        b + 2*c_fx_desk(V/fx)*V             (cash conversion tiers)
  c' US line from CAD, FX inside trade   b + 2*c_fx_desk(V/fx, in_trade)*V   (UNVERIFIED reading of the broker's page)
  d  US line from CAD, Norbert's gambit  b + 2*c_ng(V)*V                (only where c_ng < c_fx_desk at this V)
  e  TSX line from USD cash              a + the cheaper of the two FX round trips
  h = half-spread as a fraction of price, k = commission per order (C$9.99 on the TSX line, US$9.99 on the US line).
The gambit uses d = 0 (the rate-deviation and journal-delay terms are dropped; the half-spread estimates are biased upward, so the net direction of error is unknown) and a commission on the DLR-U.TO leg that is an UNRESOLVED parameter (default US$9.99, alternative US$0).
FX is paid twice in a round trip (in and out). Everything is a fraction/amount of V, V in C$.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cost_edge_norbert", ROOT / "scripts" / "cost_edge_norbert.py")
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

NORBERT_SNAPSHOT = ROOT / "research" / "cost_edge_norbert" / "snapshot_2026-10-21"
SIZES = [10_000, 25_000, 100_000, 250_000]
TOP_N = 40
WINDOW = 252            # sessions used for the cost-time spread, ADV and parity statistics
MIN_COMMON = 504        # minimum common sessions to keep a pair
MIN_PARITY = 252        # minimum sessions with a BoC rate to keep a pair
MAX_MEDIAN_ABS_GAP = 0.03   # sanity filter: a pair whose median |parity gap| exceeds this is a wrong mapping / non-1:1 line
COMMISSION = ce.COMMISSION  # C$9.99 on the TSX line, US$9.99 on the US line
TIE_BP = 1.0            # cost differences below this (bp of V) are called a tie
STALE_DAYS = 10         # a pair whose last common session is more than this many calendar days before the newest session is dropped
# Commission on the DLR-U.TO leg of Norbert's gambit, in US$. UNRESOLVED: whether the broker bills a USD-quoted TSX order in US$ (and at
# 9.99 or 0 if DLR-U is on the $0 Select ETF list) is not confirmed. The default is the conservative US$9.99; US$0 is reported as
# a sensitivity. The DLR.TO leg stays at C$9.99.
DLRU_COMMISSION_USD = 9.99
DLRU_ALT_USD = 0.0
# Domicile matters for the US 15% dividend withholding (see the withholding section of the generated report). UNVERIFIED hand-curated: names known to be
# incorporated in the US even though they trade on the TSX. Check every name before relying on it.
US_DOMICILED = {"OVV"}

# Candidate interlisted pairs: (Yahoo TSX symbol, Yahoo US symbol). Hand-curated from the large end of the TSX with a US
# listing; completeness is UNVERIFIED. Yahoo writes share classes with dashes. The final list is the TOP_N by computed TSX
# dollar ADV that pass the data filters (see select_universe), so a missing candidate can only matter if it would rank high.
CANDIDATES = [
    ("RY.TO", "RY"), ("TD.TO", "TD"), ("BNS.TO", "BNS"), ("BMO.TO", "BMO"), ("CM.TO", "CM"),
    ("ENB.TO", "ENB"), ("TRP.TO", "TRP"), ("CNQ.TO", "CNQ"), ("SU.TO", "SU"), ("CVE.TO", "CVE"),
    ("IMO.TO", "IMO"), ("OVV.TO", "OVV"), ("CP.TO", "CP"), ("CNR.TO", "CNI"), ("SHOP.TO", "SHOP"),
    ("MFC.TO", "MFC"), ("SLF.TO", "SLF"), ("BAM.TO", "BAM"), ("BN.TO", "BN"), ("NTR.TO", "NTR"),
    ("TRI.TO", "TRI"), ("WCN.TO", "WCN"), ("ABX.TO", "B"), ("AEM.TO", "AEM"), ("WPM.TO", "WPM"),
    ("FNV.TO", "FNV"), ("CCO.TO", "CCJ"), ("T.TO", "TU"), ("BCE.TO", "BCE"), ("RCI-B.TO", "RCI"),
    ("QSR.TO", "QSR"), ("GIB-A.TO", "GIB"), ("TECK-B.TO", "TECK"), ("FTS.TO", "FTS"), ("EMA.TO", "EMA"),
    ("K.TO", "KGC"), ("AGI.TO", "AGI"), ("PAAS.TO", "PAAS"), ("MG.TO", "MGA"), ("OTEX.TO", "OTEX"),
    ("GFL.TO", "GFL"), ("TFII.TO", "TFII"), ("CAE.TO", "CAE"), ("BB.TO", "BB"), ("BTE.TO", "BTE"),
    ("HBM.TO", "HBM"), ("CIGI.TO", "CIGI"), ("STN.TO", "STN"), ("BEP-UN.TO", "BEP"), ("BIP-UN.TO", "BIP"),
    ("PPL.TO", "PBA"), ("AQN.TO", "AQN"), ("DSG.TO", "DSGX"), ("GIL.TO", "GIL"), ("SJR-B.TO", "SJR"),
    ("ENS.TO", "ENS"), ("IFP.TO", "IFP"), ("LSPD.TO", "LSPD"), ("DOO.TO", "DOO"), ("NGD.TO", "NGD"),
    ("CLS.TO", "CLS"),
]


# ----------------------------------------------------------------------------- cost formulas
def route_costs(v: float, fx: float, h_tsx: float, h_us: float, h_ng_sum: float, k: float = COMMISSION,
                basis: str = "usd", k_dlru_usd: float | None = None, use_ng: bool = True) -> dict:
    """Round-trip C$ cost of every route for a position of C$ v. h_* are half-spreads as fractions of price.

    basis = 'usd': the desk's tier is read off v / fx (primary); 'cad': off v itself (check).
    k_dlru_usd: commission in US$ on the DLR-U.TO leg of the gambit (None = C$ k, as in the Norbert module).
    use_ng = False removes the gambit from the choice (FX desk only): ng_wins is False and e uses the desk. Route d is still
    computed from the gambit cost (for reference); it does not fall back to the desk.
    Returns the route costs plus the one-way FX fractions used and whether NG beats the FX desk at this size.
    """
    amt = v / fx if basis == "usd" else v
    a = 2.0 * h_tsx * v + 2.0 * k
    b = 2.0 * h_us * v + 2.0 * k * fx
    desk = ce.c_fx_desk(amt)
    desk_in = ce.c_fx_desk(amt, in_trade=True)
    k_eff = k if k_dlru_usd is None else (k + k_dlru_usd * fx) / 2.0     # c_ng charges 2 orders of k_eff per one-way gambit
    ng = ce.c_ng(v, k_eff, h_ng_sum, 0.0, 0.0)
    fx_desk, fx_in, fx_ng = 2.0 * desk * v, 2.0 * desk_in * v, 2.0 * ng * v
    ng_wins = use_ng and ng < desk
    fx_best = min(fx_desk, fx_ng) if use_ng else fx_desk
    return {"a": a, "b": b, "c": b + fx_desk, "c_in": b + fx_in, "d": b + fx_ng, "e": a + fx_best,
            "e_desk": a + fx_desk, "ng_wins": ng_wins, "c_fx_desk": desk, "c_ng": ng}


def pick(costs: dict, v: float, tie_bp: float = TIE_BP):
    """Cheapest route among `costs` (name -> C$ cost). Returns (winner, margin_bp_of_v); winner is 'tie' if the margin < tie_bp."""
    ranked = sorted(costs.items(), key=lambda kv: kv[1])
    margin = (ranked[1][1] - ranked[0][1]) / v * 1e4
    return ("tie" if margin < tie_bp else ranked[0][0]), margin


def parity_gap_bp(p_tsx: pd.Series, p_us: pd.Series, boc: pd.Series) -> pd.Series:
    """(TSX close / (US close x BoC USDCAD) - 1) in bp, on dates present in all three."""
    j = pd.concat({"t": p_tsx, "u": p_us, "r": boc}, axis=1, join="inner").dropna()
    return (j["t"] / (j["u"] * j["r"]) - 1.0) * 1e4


def window_lines(t_all: pd.DataFrame, u_all: pd.DataFrame, n: int = WINDOW):
    """The last n sessions of the COMMON calendar fix a date span; each line is then cut to that span on its OWN calendar.

    Returns (tsx_own, us_own, common_index_in_window). A one-sided holiday leaves that line's neighbouring sessions adjacent
    in its own series, which is what the two-day (Corwin-Schultz) and next-day-mid (Abdi-Ranaldo) estimators assume.
    """
    w = t_all.index.intersection(u_all.index)[-n:]
    return t_all.loc[w[0]:w[-1]], u_all.loc[w[0]:w[-1]], w


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact sign test p-value for k successes out of n at p = 1/2."""
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


# ----------------------------------------------------------------------------- data
def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["High", "Low", "Close", "Volume"]].copy()
    d = d.dropna(subset=["High", "Low", "Close"])
    return d[(d["Low"] > 0) & (d["High"] >= d["Low"]) & (d["Close"] > 0)]


def fetch(snap: Path, candidates=CANDIDATES) -> None:
    import requests
    import yfinance as yf
    snap.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []
    for tsx, us in candidates:
        for sym, side in ((tsx, "tsx"), (us, "us")):
            try:
                h = yf.Ticker(sym).history(period="max", interval="1d", auto_adjust=False)
            except Exception as e:  # noqa: BLE001 - record and continue; the filter reports missing lines
                failed.append(f"{sym}: {type(e).__name__}")
                continue
            if h.empty:
                failed.append(f"{sym}: empty")
                continue
            h.index = h.index.tz_localize(None) if h.index.tz is not None else h.index
            h.index.name = "date"
            out = h[["High", "Low", "Close", "Volume"]].reset_index()
            out.insert(0, "side", side)
            out.insert(0, "pair", f"{tsx}|{us}")
            frames.append(out)
    pd.concat(frames).to_csv(snap / "prices.csv.gz", index=False, float_format="%.6g")
    r = requests.get("https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date=2000-01-01", timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    pd.DataFrame({"date": [o["d"] for o in obs], "FXUSDCAD": [float(o["FXUSDCAD"]["v"]) for o in obs]}) \
        .to_csv(snap / "boc_fxusdcad.csv", index=False)
    (snap / "meta.json").write_text(json.dumps({
        "fetched": dt.date.today().isoformat(),
        "source": "yfinance Ticker.history (auto_adjust=False, interval 1d); BoC Valet FXUSDCAD",
        "candidates": len(candidates), "failed_downloads": failed}))


def load(snap: Path) -> dict:
    px = pd.read_csv(snap / "prices.csv.gz", parse_dates=["date"])
    lines = {}
    for (pair, side), g in px.groupby(["pair", "side"]):
        lines.setdefault(pair, {})[side] = clean(g.set_index("date").sort_index().drop(columns=["pair", "side"]))
    boc = pd.read_csv(snap / "boc_fxusdcad.csv", parse_dates=["date"]).set_index("date")["FXUSDCAD"]
    return {"lines": lines, "boc": boc, "meta": json.loads((snap / "meta.json").read_text())}


# ----------------------------------------------------------------------------- analysis
def common(lines: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = lines["tsx"].index.intersection(lines["us"].index)
    return lines["tsx"].loc[idx], lines["us"].loc[idx]


# Route variants stored per name: key -> kwargs for route_costs (defaults: DLR-U leg at US$ DLRU_COMMISSION_USD, desk tier off USD amount)
def _variants(k_dlru_usd: float) -> dict:  # cost-route variants; the spread-estimator variants are built in analyse_pair
    return {"routes": {"k_dlru_usd": k_dlru_usd},
            "routes_dlru0": {"k_dlru_usd": DLRU_ALT_USD},
            "routes_cadbasis": {"k_dlru_usd": k_dlru_usd, "basis": "cad"},
            "routes_desk_usd": {"k_dlru_usd": k_dlru_usd, "use_ng": False},
            "routes_desk_cad": {"k_dlru_usd": k_dlru_usd, "use_ng": False, "basis": "cad"}}


def analyse_pair(pair: str, lines: dict, boc: pd.Series, fx: float, h_ng_sum: float,
                 k_dlru_usd: float = DLRU_COMMISSION_USD, asof: pd.Timestamp | None = None) -> dict:
    """All per-name statistics, or a dict with `drop` set to the reason if the pair fails the data filters."""
    t_sym, u_sym = pair.split("|")
    missing = [sym for sym, side in ((t_sym, "tsx"), (u_sym, "us")) if side not in lines]
    if missing:
        return {"pair": pair, "drop": f"no price history from Yahoo for {', '.join(missing)} (empty response: delisted, renamed or wrong symbol)"}
    t, u = common(lines)
    if len(t) < MIN_COMMON:
        return {"pair": pair, "drop": f"only {len(t)} common sessions"}
    if asof is not None and (asof - t.index.max()).days > STALE_DAYS:
        return {"pair": pair, "drop": f"stale: last common session {t.index.max().date()} is more than {STALE_DAYS} days before {asof.date()}"}
    gap = parity_gap_bp(t["Close"], u["Close"], boc)
    if len(gap) < MIN_PARITY:
        return {"pair": pair, "drop": f"only {len(gap)} sessions with a BoC rate"}
    gap_w = gap.tail(WINDOW)
    if gap_w.abs().median() / 1e4 > MAX_MEDIAN_ABS_GAP:
        return {"pair": pair, "drop": f"median |parity gap| {gap_w.abs().median():.0f} bp (wrong mapping / non-1:1 line)"}
    t_own, u_own, w = window_lines(lines["tsx"], lines["us"])
    st, su = ce.spread_window(t_own), ce.spread_window(u_own)                  # each line on its own calendar
    st_c, su_c = ce.spread_window(t.loc[w]), ce.spread_window(u.loc[w])        # previous method: common calendar (reported only)
    tw, uw = t.loc[w], u.loc[w]
    adv_tsx = float((tw["Close"] * tw["Volume"]).mean())            # C$ per day, Yahoo TSX-line volume
    adv_us = float((uw["Close"] * uw["Volume"]).mean())             # US$ per day, Yahoo (consolidated) volume
    c0, c1 = t.index.min(), t.index.max()
    sy_t, sy_u = ce.spread_table(lines["tsx"].loc[c0:c1]), ce.spread_table(lines["us"].loc[c0:c1])
    res = {"pair": pair, "drop": None, "common_start": c0, "common_end": c1, "n_common": len(t),
           "h_tsx": st["used"], "h_us": su["used"], "st": st, "su": su,
           "h_tsx_common": st_c["used"], "h_us_common": su_c["used"],
           "adv_tsx_cad": adv_tsx, "adv_us_cad": adv_us * fx, "spread_by_year": (sy_t, sy_u),
           "gap_all": gap, "gap_win": gap_w,
           "gap_stats": {"median": float(gap_w.median()), "abs_median": float(gap_w.abs().median()),
                         "p5": float(gap_w.quantile(0.05)), "p95": float(gap_w.quantile(0.95)),
                         "abs_p95": float(gap_w.abs().quantile(0.95))}}
    for key, kw in _variants(k_dlru_usd).items():
        res[key] = {v: route_costs(v, fx, res["h_tsx"], res["h_us"], h_ng_sum, **kw) for v in SIZES}
    # spread-estimator readings: each line's half-spread from one estimator only (still floored at the half tick), or half of the used value
    for key, f_t, f_u, f_ng in (
            ("routes_cs", np.nanmax([st["cs_half"], st["half_tick"]]), np.nanmax([su["cs_half"], su["half_tick"]]), 1.0),
            ("routes_ar", np.nanmax([st["ar_half"], st["half_tick"]]), np.nanmax([su["ar_half"], su["half_tick"]]), 1.0),
            ("routes_halfspread", st["used"] / 2.0, su["used"] / 2.0, 0.5)):
        res[key] = {v: route_costs(v, fx, float(f_t), float(f_u), h_ng_sum * f_ng, k_dlru_usd=k_dlru_usd) for v in SIZES}
    res["routes_commoncal"] = {v: route_costs(v, fx, res["h_tsx_common"], res["h_us_common"], h_ng_sum, k_dlru_usd=k_dlru_usd)
                               for v in SIZES}
    return res


def select_universe(results: list[dict], top_n: int = TOP_N) -> tuple[list[dict], list[dict]]:
    """Keep the top_n passing pairs by TSX-line C$ ADV. Returns (kept, dropped-with-reason incl. 'ranked below the cut')."""
    ok = sorted((r for r in results if not r["drop"]), key=lambda r: -r["adv_tsx_cad"])
    dropped = [r for r in results if r["drop"]]
    dropped += [{"pair": r["pair"], "drop": f"ranked {i + top_n + 1} by TSX ADV (below the cut)"} for i, r in enumerate(ok[top_n:])]
    return ok[:top_n], dropped


def median_route(names: list[dict], v: float, key: str = "routes") -> dict:
    return {r: float(np.median([n[key][v][r] for n in names])) for r in ("a", "b", "c", "c_in", "d", "e", "e_desk")}


def us_from_cad(r: dict) -> float:
    """Cheapest way to hold the US line from CAD cash for one name: the FX desk (c), or the gambit (d) where it beats the desk."""
    return min(r["c"], r["d"]) if r["ng_wins"] else r["c"]


def _decide(diffs: list[float], v: float, first: str, second: str):
    """diffs[i] = cost(second) - cost(first) for name i, in C$. The verdict uses the MEDIAN of the per-name differences, so the
    reported saving is a median of per-name savings, not a difference of medians. Returns (winner or 'tie', saving bp of V)."""
    return pick({first: 0.0, second: float(np.median(diffs))}, v)


def verdicts(names: list[dict], key: str = "routes") -> dict:
    """(account, V) -> dict(winner, margin_bp, wins=#names where the line wins, n, median costs of the compared routes).

    margin_bp = median over names of the per-name saving of the cheaper line (bp of V)."""
    out = {}
    for v in SIZES:
        med = median_route(names, v, key)
        # CAD cash: TSX line (a) vs best US route from CAD (cheapest of c, d where NG wins at this V; c_in is reported only)
        ng_ok = np.median([n[key][v]["ng_wins"] for n in names]) > 0.5
        us_route = "d" if ng_ok and med["d"] < med["c"] else "c"
        w, m = _decide([us_from_cad(n[key][v]) - n[key][v]["a"] for n in names], v, "TSX", "US")
        tsx_wins = sum(n[key][v]["a"] < us_from_cad(n[key][v]) for n in names)
        out[("CAD", v)] = {"winner": w, "margin_bp": m, "tsx_wins": int(tsx_wins), "n": len(names),
                           "tsx_cost": med["a"], "us_cost": med[us_route], "us_route": us_route}
        # USD cash: US line (b) vs TSX line with FX (e)
        w, m = _decide([n[key][v]["e"] - n[key][v]["b"] for n in names], v, "US", "TSX")
        us_wins = sum(n[key][v]["b"] < n[key][v]["e"] for n in names)
        out[("USD", v)] = {"winner": w, "margin_bp": m, "us_wins": int(us_wins), "n": len(names),
                           "us_cost": med["b"], "tsx_cost": med["e"]}
    return out


ALT_READINGS = {"routes_cadbasis": "desk tier read off the C$ amount", "routes_desk_usd": "gambit removed (FX desk only)",
                "routes_desk_cad": "gambit removed, tier off the C$ amount", "routes_dlru0": f"DLR-U.TO commission US${DLRU_ALT_USD:g}",
                "routes_cs": "spreads from Corwin-Schultz only", "routes_ar": "spreads from Abdi-Ranaldo only",
                "routes_halfspread": "all half-spreads (both lines and the gambit) halved"}


def alt_check(vd: dict, names: list[dict]) -> dict:
    """(account, V) -> list of alternative readings under which the verdict differs from the primary one (empty = same under all)."""
    alts = {k: verdicts(names, k) for k in ALT_READINGS}
    return {kv: [ALT_READINGS[k] for k, a in alts.items() if a[kv]["winner"] != x["winner"]] for kv, x in vd.items()}


def spread_findings(names: list[dict]) -> dict:
    """TSX-vs-US half-spread comparison and the sign of the per-name median parity gap, as counts with exact sign-test p-values."""
    d = np.array([n["h_tsx"] - n["h_us"] for n in names])
    n_wider, n_narrower = int((d > 0).sum()), int((d < 0).sum())
    gaps = np.array([n["gap_stats"]["median"] for n in names])
    n_gap_pos = int((gaps > 0).sum())
    return {"n": len(names), "n_tsx_wider": n_wider, "n_tsx_narrower": n_narrower,
            "p_spread": sign_test_p(n_wider, n_wider + n_narrower), "median_diff_bp": float(np.median(d)) * 1e4,
            "n_gap_pos": n_gap_pos, "p_gap": sign_test_p(n_gap_pos, int((gaps != 0).sum())), "median_gap_bp": float(np.median(gaps))}


REMAPPED_PAIRS = {"ABX.TO|B", "CLS.TO|CLS", "PPL.TO|PBA"}   # pairs whose Yahoo mapping was corrected or added (section 2)


def gap_common_factor(names: list[dict]) -> dict:
    """Per-name median parity gap after removing the daily cross-name mean. All names share one BoC rate per day, so the raw
    per-name medians are not independent; this removes the common daily component and re-counts the positive medians."""
    g = pd.concat({n["pair"]: n["gap_win"] for n in names}, axis=1)
    dm = g.sub(g.mean(axis=1), axis=0)
    med = dm.median()
    n_pos, n_nz = int((med > 0).sum()), int((med != 0).sum())
    return {"n_pos": n_pos, "n": len(med), "p": sign_test_p(n_pos, n_nz),
            "common_share": 1.0 - float(np.nanvar(dm.values)) / float(np.nanvar(g.values))}


def _tsx_vs_us(d, thr_bp: float = 0.0) -> tuple[int, int, float]:
    """(# TSX wider, # TSX narrower, exact sign-test p) over differences d in bp, ignoring |d| <= thr_bp."""
    d = np.asarray(d, dtype=float)
    wd, nr = int((d > thr_bp).sum()), int((d < -thr_bp).sum())
    return wd, nr, sign_test_p(wd, wd + nr)


def spread_sensitivity(names: list[dict], passing: list[dict] | None = None) -> dict:
    """How fragile the TSX-vs-US half-spread sign test is, and how robust its direction is. All values (wider, narrower, p)."""
    bp = lambda rs, a="h_tsx", b="h_us": [(r[a] - r[b]) * 1e4 for r in rs]
    wd, nr, _ = _tsx_vs_us(bp(names))
    n_min = min(wd, nr)
    passing = names if passing is None else passing
    top_us = sorted(passing, key=lambda r: -r["adv_us_cad"])[:TOP_N]
    return {"one_name_moves_p": sign_test_p(n_min + 1, wd + nr),
            "min_1bp": _tsx_vs_us(bp(names), 1.0), "min_2bp": _tsx_vs_us(bp(names), 2.0),
            "common_calendar": _tsx_vs_us(bp(names, "h_tsx_common", "h_us_common")),
            "all_passing": (len(passing), *_tsx_vs_us(bp(passing))),
            "top_by_us_adv": _tsx_vs_us(bp(top_us)),
            "without_remapped": _tsx_vs_us(bp([r for r in names if r["pair"] not in REMAPPED_PAIRS]))}


def one_leg_check(names: list[dict], key: str = "routes") -> dict:
    """If the FX is paid only once (you never convert back), does the venue ordering hold? Counts name x size cases and the
    ratio (one cheapest FX leg) / (round-trip cost advantage the line that needs FX has before FX), over cases with an advantage."""
    holds = total = 0
    ratios, fails = [], []
    for n in names:
        for v in SIZES:
            r = n[key][v]
            half_cad = (us_from_cad(r) - r["b"]) / 2.0            # one FX leg of the cheapest CAD -> US route
            half_usd = (r["e"] - r["a"]) / 2.0                    # one FX leg of the cheapest USD -> TSX route
            for acct, adv, half, own_cost, other_cost in (("CAD", r["a"] - r["b"], half_cad, r["a"], r["b"] + half_cad),
                                                          ("USD", r["b"] - r["a"], half_usd, r["b"], r["a"] + half_usd)):
                total += 1
                holds += own_cost < other_cost
                if not own_cost < other_cost:
                    fails.append((n["pair"], v, acct))
                if adv > 0:
                    ratios.append((half / adv, n["pair"], v))
    ratios.sort()
    return {"holds": holds, "total": total, "fails": fails, "n_ratios": len(ratios), "min": ratios[0] if ratios else None,
            "median": float(np.median([x[0] for x in ratios])) if ratios else None}


def stress_ordering(fx: float, h_ng: float, k_dlru_usd: float, vd: dict) -> dict:
    """Zero-spread stress: does each account's winning line stay the same as in the verdict? -> {account: [sizes where it changes]}"""
    changed = {"CAD": [], "USD": []}
    for v in SIZES:
        z = route_costs(v, fx, 0.0, 0.0, h_ng, k_dlru_usd=k_dlru_usd)
        cad = "TSX" if z["a"] < us_from_cad(z) else "US"
        usd = "US" if z["b"] < z["e"] else "TSX"
        for acct, got in (("CAD", cad), (("USD"), usd)):
            if vd[(acct, v)]["winner"] not in (got, "tie"):
                changed[acct].append(v)
    return changed


def required_spread_edge_bp(v: float, fx: float, h_ng_sum: float, k_dlru_usd: float = DLRU_COMMISSION_USD) -> dict:
    """How much cheaper (in one-way half-spread, bp of price) the US line must be than the TSX line before CAD cash should
    buy the US line: the round-trip FX cost per side (2 legs / 2 sides = one conversion per side), plus the commission gap."""
    r = route_costs(v, fx, 0.0, 0.0, h_ng_sum, k_dlru_usd=k_dlru_usd)
    fx_per_side = min(2 * r["c_fx_desk"], 2 * r["c_ng"]) / 2.0 * 1e4
    comm_gap = (2 * COMMISSION * fx - 2 * COMMISSION) / v * 1e4 / 2.0
    return {"fx_per_side_bp": fx_per_side, "commission_gap_bp_per_side": comm_gap}


# ----------------------------------------------------------------------------- render
def pct(x, nd=3):
    return f"{x * 100:.{nd}f}%"


def bps(x, nd=1):
    return f"{x * 1e4:.{nd}f}"


def money(x):
    return f"{x:,.0f}"


def render(names: list[dict], dropped: list[dict], data: dict, ng: dict, fx: float, fx_date, tax: str,
           passing: list[dict] | None = None) -> str:
    vd = verdicts(names)
    alt = alt_check(vd, names)
    sf = spread_findings(names)
    sens = spread_sensitivity(names, passing)
    gcf = gap_common_factor(names)
    boc = data["boc"]
    k_def, k_alt = DLRU_COMMISSION_USD, DLRU_ALT_USD
    L = []
    w = L.append
    w("# Dual-listed stocks: which line to trade, TSX or NYSE/Nasdaq\n")
    w("Cost parameters (commission, spreads, FX tiers): see [`docs/cost_assumptions.md`](../cost_assumptions.md).\n")
    w(f"Data fetched {data['meta']['fetched']} ({data['meta']['source']}). Generated by `scripts/cost_edge_duallisting.py` "
      "from the local snapshot in `research/cost_edge_duallisting/`. FX cost functions, spread estimators and the DLR / DLR-U.TO "
      "half-spreads are imported from `scripts/cost_edge_norbert.py` and the Norbert snapshot "
      f"(`{NORBERT_SNAPSHOT.name}`; see `docs/research_notes/2026-10-21_norbert_gambit_results.md`). "
      "The broker's tiers, the C$9.99 / US$9.99 commission and the 1-cent tick are input constants read from the broker's published fee pages on 2026-10-18, not re-verified here. "
      "All spreads are **estimates** (Corwin-Schultz, Abdi-Ranaldo), not quotes.\n")
    w("## 1. Short answer\n")
    s_cad = [vd[("CAD", v)] for v in SIZES]
    s_usd = [vd[("USD", v)] for v in SIZES]
    w(f"- **Holding CAD:** the TSX line is the cheaper venue at {sum(x['winner'] == 'TSX' for x in s_cad)} of {len(SIZES)} sizes "
      f"(median name). Buying the US line with CAD means paying the desk's FX spread (or Norbert's gambit) twice; the cheapest FX leg costs "
      f"{ng['fx_range_text']} bp of V per conversion (cheaper of FX desk and gambit, C$10k to C$250k), against a TSX-minus-US half-spread "
      f"difference of {bps(float(np.median([n['h_tsx'] - n['h_us'] for n in names])))} bp (median name; largest over all names "
      f"{bps(max(n['h_tsx'] - n['h_us'] for n in names))} bp).")
    w(f"- **Holding USD:** the US line is the cheaper venue at {sum(x['winner'] == 'US' for x in s_usd)} of {len(SIZES)} sizes; "
      "buying the TSX line from USD cash needs the same two conversions. The spread estimates are inflated in absolute terms "
      "(section 3), but the verdict depends on the line-to-line difference and on FX, not on the absolute level.")
    w("- The gambit cost uses d = 0 and upward-biased half-spread estimates, so its net bias direction is unknown (sections 4 and 8).")
    w("- Details, per-size numbers and the exceptions are below; the verdict table is section 6.\n")
    w("## 2. Universe and data\n")
    w(f"- **How the list was built (read this before trusting it).** A hand-curated list of {len(CANDIDATES)} candidate TSX/US pairs "
      "(large TSX names that also list in the US; Yahoo symbols, share classes with dashes; completeness UNVERIFIED) was downloaded. "
      f"The {TOP_N} with the largest TSX-line dollar ADV (mean close x volume over the last {WINDOW} sessions) that pass the filters were kept. "
      "Ranking by computed ADV rather than market cap avoids a second data source; it can differ from a market-cap ranking.")
    w(f"- Filters: at least {MIN_COMMON} common sessions; at least {MIN_PARITY} sessions with a BoC rate; median |parity gap| over the "
      f"last {WINDOW} sessions at most {MAX_MEDIAN_ABS_GAP * 100:.0f}% (catches wrong mappings and non-1:1 lines).")
    n_cut = sum("below the cut" in r["drop"] for r in dropped)
    w(f"- Not in the final list: {len(dropped)} of {len(CANDIDATES)} candidates ({len(dropped) - n_cut} failed a data filter or had no Yahoo "
      f"history, {n_cut} passed but ranked below the cut).")
    for r in sorted(dropped, key=lambda r: r["pair"]):
        w(f"  - `{r['pair']}`: {r['drop']}")
    if data["meta"].get("failed_downloads"):
        w(f"- Failed downloads: {', '.join(data['meta']['failed_downloads'])}.")
    starts = [n["common_start"] for n in names]
    w(f"- Common history per name starts between {min(starts).date()} and {max(starts).date()} (median {pd.Series(starts).median().date()}), "
      f"ending {max(n['common_end'] for n in names).date()}. Each line's spreads are estimated on its OWN exchange calendar "
      f"(restricted to the date span of the common history, or of the last {WINDOW} common sessions for the cost-time spreads); ADV and parity "
      f"use the last {WINDOW} common sessions. Pairs whose last common session is more than {STALE_DAYS} days before the newest session are dropped.")
    w("- **Symbol mapping.** `PPL.TO|PBA` (Pembina), `DOO.TO|DOO` (BRP), `CLS.TO|CLS` (Celestica) and `ABX.TO|B` (Barrick: TSX ticker unchanged, NYSE ticker "
      "changed from GOLD to B; Yahoo's `GOLD` series no longer tracks the same security and failed the parity filter in an earlier run of this note) "
      "were corrected or added after review. A symbol with an empty Yahoo response is reported below as such, not as a data problem of the pair.")
    w(f"- The Bank of Canada FXUSDCAD series in the snapshot starts {boc.index.min().date()} and ends {boc.index.max().date()}. "
      "The parity gap (section 5) is limited to dates with a BoC rate. Latest rate used for costs: "
      f"{fx:.4f} ({fx_date.date()}).")
    w("- **Data source:** yfinance. `alpha101.yahoo_ohlcv` was not used: it holds 50 symbols (mostly US/TSX ETFs and 32 US large "
      "caps), not these interlisted pairs. Yahoo `.TO` volume is probably TSX-only while the US line's volume is consolidated across US "
      "venues (UNVERIFIED), which biases the ADV comparison toward the US line.\n")
    w("## 3. Half-spreads (ESTIMATES) and ADV\n")
    w("Half-spread as a fraction of price. Per line: **used = max(CS, AR, half-tick)** from `cost_edge_norbert.spread_table` "
      "(Corwin-Schultz 2012 two-day windows, negatives clamped to 0; Abdi-Ranaldo 2017 monthly, months with at least 8 sessions; "
      "the half-tick floor is one step beyond max(CS, AR) and is labelled as such). High-low estimators overstate the spread of "
      "mega-caps whose true spread is one tick, so differences of 1-2 bp between lines are inside estimator noise.\n")
    h_t = np.array([n["h_tsx"] for n in names])
    h_u = np.array([n["h_us"] for n in names])
    vd_cc = verdicts(names, "routes_commoncal")
    chg_t = np.array([n["h_tsx"] - n["h_tsx_common"] for n in names]) * 1e4
    chg_u = np.array([n["h_us"] - n["h_us_common"] for n in names]) * 1e4
    worst = max(names, key=lambda n: max(abs(n["h_tsx"] - n["h_tsx_common"]), abs(n["h_us"] - n["h_us_common"])))
    diff_sizes = [f"{acct} C${v:,}" for (acct, v), x in vd.items() if vd_cc[(acct, v)]["winner"] != x["winner"]]
    direction = "wider" if sf["n_tsx_wider"] >= sf["n_tsx_narrower"] else "narrower"
    n_dir = max(sf["n_tsx_wider"], sf["n_tsx_narrower"])
    w(f"Over the {len(names)} names (last {WINDOW} sessions): TSX line median half-spread {bps(np.median(h_t))} bp "
      f"(p10 {bps(np.quantile(h_t, .1))}, p90 {bps(np.quantile(h_t, .9))}); US line {bps(np.median(h_u))} bp "
      f"(p10 {bps(np.quantile(h_u, .1))}, p90 {bps(np.quantile(h_u, .9))}). The TSX half-spread is {direction} than the US one for "
      f"{n_dir} of {len(names)} names (wider for {sf['n_tsx_wider']}, narrower for {sf['n_tsx_narrower']}; two-sided exact sign test p = "
      f"{sf['p_spread']:.1e}). Median (TSX - US) = {sf['median_diff_bp']:.2f} bp; share of names where the two lines are within 1 bp of each "
      f"other: {np.mean(np.abs(h_t - h_u) < 1e-4) * 100:.0f}%. "
      "**The absolute levels are far above what these mega-caps quote** (a high-low estimator on daily bars picks up intraday volatility; "
      "typical quoted spreads of the large banks are a fraction of that, UNVERIFIED), so read them as upper bounds. That direction is the "
      f"majority one but not universal ({n_dir} of {len(names)}), and the difference is small against the level and "
      "inside the 1-2 bp noise of the estimator, so it is a tilt, not a measurement of the true spread gap. See section 6 for whether it decides a verdict.\n")
    (w1, n1, p1), (w2, n2, p2) = sens["min_1bp"], sens["min_2bp"]
    cc, ap, tu, wr = sens["common_calendar"], sens["all_passing"], sens["top_by_us_adv"], sens["without_remapped"]
    w(f"**The sign test is fragile; the direction is more robust.** p = {sf['p_spread']:.3f} on {sf['n_tsx_wider']} vs {sf['n_tsx_narrower']} is only just "
      f"below 5%: moving one name to the other side gives p = {sens['one_name_moves_p']:.3f}; counting only names whose half-spreads differ by at least 1 bp, "
      f"TSX wider for {w1} and narrower for {n1} (p = {p1:.3f}); at least 2 bp, {w2} vs {n2} (p = {p2:.2f}). Direction (TSX wider / narrower) under other "
      f"choices: common-calendar estimates {cc[0]} / {cc[1]}; all {ap[0]} pairs that pass the filters {ap[1]} / {ap[2]}; the {TOP_N} with the largest "
      f"US-line ADV {tu[0]} / {tu[1]}; without the {len(REMAPPED_PAIRS)} pairs whose mapping was corrected {wr[0]} / {wr[1]}. Consolidated US high-low versus "
      "TSX-only high-low (venue coverage) is an unresolved confound.\n")
    w(f"Calendar check: estimating each line on its own exchange calendar (this version) rather than on the TSX-and-US intersection (previous "
      f"version) changed the TSX half-spread by {chg_t.mean():+.2f} bp on average (largest |change| {np.abs(chg_t).max():.1f} bp) and the US one by "
      f"{chg_u.mean():+.2f} bp (largest {np.abs(chg_u).max():.1f} bp); the biggest single move is {worst['pair']}. "
      + ("The verdict table is unchanged against the common-calendar estimates.\n" if not diff_sizes else
         f"Verdicts that differ from the common-calendar estimates: {', '.join(diff_sizes)}.\n"))
    # per-year medians across names
    w("Median across names of the per-year used half-spread (bp), by line. Survivor-only (names with no Yahoo history are absent) and on "
      "split-adjusted prices, which inflates the half-tick floor in early years:\n")
    w("| year | names | TSX | US |\n|---|---|---|---|")
    years = sorted({y for n in names for y in n["spread_by_year"][0].index})
    for y in years:
        tv = [n["spread_by_year"][0].loc[y, "used"] for n in names if y in n["spread_by_year"][0].index]
        uv = [n["spread_by_year"][1].loc[y, "used"] for n in names if y in n["spread_by_year"][1].index]
        if len(tv) >= 5:
            w(f"| {y} | {len(tv)} | {bps(np.median(tv))} | {bps(np.median(uv))} |")
    w("\nPer name and year, with CS, AR and tick columns: `research/cost_edge_duallisting/snapshot_<date>/spreads_by_year.csv` (route costs: `route_costs.csv` in the same folder).\n")
    w("| pair | h TSX bp | h US bp | TSX ADV C$M | US ADV C$M | US/TSX ADV | median parity gap bp | median abs gap bp |\n|---|---|---|---|---|---|---|---|")
    for n in names:
        w(f"| {n['pair']} | {bps(n['h_tsx'])} | {bps(n['h_us'])} | {n['adv_tsx_cad'] / 1e6:,.0f} | {n['adv_us_cad'] / 1e6:,.0f} | "
          f"{n['adv_us_cad'] / n['adv_tsx_cad']:.1f} | {n['gap_stats']['median']:.1f} | {n['gap_stats']['abs_median']:.1f} |")
    w("")
    w("## 4. Cost of a round-trip position\n")
    w(f"Routes (all round trip, C$): a = TSX line from CAD; b = US line from USD already held; c = US line from CAD via the FX desk "
      f"(cash-conversion tiers, tier read off the USD amount V/{fx:.4f}); c' = same with the FX done inside the trade (UNVERIFIED "
      f"reading: amounts of ${ce.DESK_IN_TRADE_FROM:,} or more pay the ${50_000:,}-${59_999:,} tier); d = US line from CAD via "
      f"Norbert's gambit; e = TSX line from USD (best of FX desk and gambit). Commission C${COMMISSION} per TSX order and US${COMMISSION} "
      f"per US order (= C${COMMISSION * fx:.2f} at {fx:.4f}); the gambit's own commission is C${COMMISSION} on each DLR.TO order and "
      f"US${k_def:g} (default) on each DLR-U.TO order, two of each per round trip (UNRESOLVED, below). A round trip pays FX **twice** (in and out). If you never convert back "
      "(you keep USD) the FX cost is one-sided, i.e. half of the FX part.\n")
    h_ng = ng["h_sum"]
    w(f"NG half-spread sum (DLR.TO + DLR-U.TO, Norbert window {ng['window']}): {bps(h_ng)} bp. NG beats the FX desk (one way, base case, "
      "d = 0), per `cost_edge_norbert.breakeven` (USD basis): from "
      f"{ng['bt'][k_def][0]}.\n")
    w(f"**Gambit assumptions.** (1) d = 0: the rate-deviation and journal-delay terms are dropped, so the gambit cost uses d = 0 (routes d and e and the "
      f"break-even above). The half-spread sum used for the gambit is a high-low estimate that is biased upward (section 3), so the net direction of the "
      f"error in the NG numbers is unknown: dropping d biases them down, the spread estimate biases them up. (2) **UNRESOLVED, not confirmed by the broker:** the commission on the "
      f"DLR-U.TO leg. Default here: US${k_def:g} per order (converted at {fx:.4f}); alternative: US${k_alt:g} (if it is on the broker's $0 list). "
      f"The DLR.TO leg is C${COMMISSION}. Results for both are in the table below; the verdict table (section 6) uses the default and flags any "
      "verdict that differs under the alternative.\n")
    ngw = {k: all(n[key][v]["ng_wins"] for n in names for v in SIZES) for k, key in ((k_def, "routes"), (k_alt, "routes_dlru0"))}
    w("| V (C$) | NG one-way cost bp, DLR-U.TO at US$%g | NG beats FX desk | NG one-way cost bp, DLR-U.TO at US$%g | NG beats FX desk | FX desk one-way bp |\n|---|---|---|---|---|---|" % (k_def, k_alt))
    for v in SIZES:
        r0, r1 = names[0]["routes"][v], names[0]["routes_dlru0"][v]
        w(f"| {v:,} | {r0['c_ng'] * 1e4:.1f} | {'yes' if r0['ng_wins'] else 'no'} | {r1['c_ng'] * 1e4:.1f} | {'yes' if r1['ng_wins'] else 'no'} | {r0['c_fx_desk'] * 1e4:.1f} |")
    w("")
    w("Median across names, C$ (bp of V in brackets):\n")
    w("| V (C$) | a TSX from CAD | b US from USD | c US from CAD, FX desk | c' US from CAD, in-trade | d US from CAD, NG | e TSX from USD |\n|---|---|---|---|---|---|---|")
    for v in SIZES:
        m = median_route(names, v)
        cells = [f"{money(m[r])} ({m[r] / v * 1e4:.1f})" for r in ("a", "b", "c", "c_in", "d", "e")]
        w(f"| {v:,} | " + " | ".join(cells) + " |")
    nofix = [f"C${v:,}" for v in SIZES if not all(n["routes"][v]["ng_wins"] for n in names)]
    w("\nWhere NG is cheaper than the FX desk it is used for d and e; where it is not, d is shown for reference only and the verdict ignores it. "
      + (f"NG is cheaper than the FX desk at every modelled size for every name (DLR-U.TO at US${k_def:g}: {'yes' if ngw[k_def] else 'no'}; "
         f"at US${k_alt:g}: {'yes' if ngw[k_alt] else 'no'}). " if all(ngw.values()) else
         f"NG is NOT cheaper than the FX desk at every size (default commission: {'yes' if ngw[k_def] else 'no'}; "
         f"at US${k_alt:g}: {'yes' if ngw[k_alt] else 'no'}; sizes where it loses under the default: {', '.join(nofix) or 'none'}). ")
      + "See the `ng_wins` flag in the CSV.\n")
    w("**What the US line would have to save.** One conversion per side costs the following (the cheaper of FX desk and NG, bp of V), "
      "against the extra commission cost of the US line (US$ order vs C$ order):\n")
    w("| V (C$) | cheapest FX per side bp | extra US commission per side bp | TSX-minus-US half-spread needed bp | median actual bp | largest actual bp | names where the US route wins from CAD |\n|---|---|---|---|---|---|---|")
    for v in SIZES:
        rq = required_spread_edge_bp(v, fx, h_ng, k_def)
        need = rq["fx_per_side_bp"] + rq["commission_gap_bp_per_side"]
        wins = sum(min(n["routes"][v]["c"], n["routes"][v]["d"] if n["routes"][v]["ng_wins"] else np.inf) < n["routes"][v]["a"] for n in names)
        act = np.median([(n["h_tsx"] - n["h_us"]) * 1e4 for n in names])
        w(f"| {v:,} | {rq['fx_per_side_bp']:.1f} | {rq['commission_gap_bp_per_side']:.2f} | {need:.1f} | {act:.2f} | {max((n['h_tsx'] - n['h_us']) * 1e4 for n in names):.2f} | {wins} of {len(names)} |")
    w("")
    z = {v: route_costs(v, fx, 0.0, 0.0, h_ng, k_dlru_usd=k_def) for v in SIZES}
    chg = stress_ordering(fx, h_ng, k_def, vd)
    w("Stress with zero spread on both lines (only commission and FX left), C$ round trip: "
      + "; ".join(f"V = {v:,}: TSX from CAD {z[v]['a']:.0f}, US from CAD {us_from_cad(z[v]):.0f}, "
                  f"US from USD {z[v]['b']:.0f}, TSX from USD {z[v]['e']:.0f}" for v in SIZES)
      + (". The winning line is the same as in the verdict table for every size and both accounts.\n" if not chg["CAD"] and not chg["USD"] else
         f". The winning line differs from the verdict table for: CAD {', '.join(f'C${v:,}' for v in chg['CAD']) or 'none'}; USD "
         f"{', '.join(f'C${v:,}' for v in chg['USD']) or 'none'}.\n"))
    w("## 5. Price-parity gap (caveat)\n")
    allg = pd.concat([n["gap_win"] for n in names])
    w(f"gap = TSX close / (US close x BoC FXUSDCAD) - 1, in bp, over the last {WINDOW} sessions per name. The BoC rate is a session "
      "average of one-minute quotes (08:00-16:00 ET), not a closing rate (same caveat as the Norbert's gambit report), and both exchanges close at "
      "16:00 ET, so much of the dispersion is likely that averaging, not a tradable mispricing (cause UNVERIFIED).\n")
    w("| statistic | bp |\n|---|---|")
    w(f"| pooled median | {allg.median():.1f} |\n| pooled mean | {allg.mean():.1f} |\n| pooled p5 / p95 | {allg.quantile(.05):.1f} / {allg.quantile(.95):.1f} |\n"
      f"| pooled median abs | {allg.abs().median():.1f} |\n| pooled p95 abs | {allg.abs().quantile(.95):.1f} |")
    w("\nPer name median and median-abs gap are in the table in section 3. "
      f"The per-name median gap is positive for {sf['n_gap_pos']} of {len(names)} names (sign test p = {sf['p_gap']:.1e}, which treats the "
      f"{len(names)} names as independent; they are not, because every name uses the same BoC rate each day); the median of the per-name medians is "
      f"{sf['median_gap_bp']:.1f} bp. After removing the daily cross-name mean, the per-name median gap is positive for {gcf['n_pos']} of {gcf['n']} names "
      f"(p = {gcf['p']:.2f}); the common daily component accounts for {gcf['common_share'] * 100:.0f}% of the gap variance. So the one-sided sign is largely one "
      "common effect (the TSX close slightly above the US close converted at the BoC session-average rate; cause UNVERIFIED: the averaging convention, "
      "or close timing, are candidates), not many independent confirmations. The level is small "
      "and not a cost you can capture, and a close-to-close comparison cannot resolve line differences below roughly 10 bp (the pooled dispersion above).\n")
    w("## 6. Verdict\n")
    w("Median name; primary basis = desk tier read off the USD amount; 'wins in N of M' counts names. "
      f"Differences under {TIE_BP:g} bp of V are called a tie. All numbers are estimates.\n")
    w("| account cash | V (C$) | verdict | median cost of cheaper line | median cost of other line | saving (median of per-name savings) | verdict differs under (alternative readings) |\n|---|---|---|---|---|---|---|")
    for v in SIZES:
        for acct in ("CAD", "USD"):
            x = vd[(acct, v)]
            if acct == "CAD":
                lines = {"TSX": x["tsx_cost"], "US": x["us_cost"]}
                route = {"TSX": "TSX line", "US": f"US line via {'Norbert' if x['us_route'] == 'd' else 'FX desk'}"}
                wins = f"TSX wins in {x['tsx_wins']} of {x['n']} names"
            else:
                lines = {"US": x["us_cost"], "TSX": x["tsx_cost"]}
                route = {"US": "US line (no FX)", "TSX": "TSX line (+ FX both ways)"}
                wins = f"US wins in {x['us_wins']} of {x['n']} names"
            txt = "no meaningful difference; use the line with the deeper book" if x["winner"] == "tie" else f"{route[x['winner']]} ({wins})"
            flip = "none" if not alt[(acct, v)] else "; ".join(alt[(acct, v)])
            cheap, other = min(lines.values()), max(lines.values())
            w(f"| {acct} | {v:,} | {txt} | {money(cheap)} ({cheap / v * 1e4:.1f} bp) | {money(other)} ({other / v * 1e4:.1f} bp) | "
              f"{x['margin_bp']:.1f} bp | {flip} |")
    n_alt = sum(bool(x) for x in alt.values())
    w(f"\nAlternative readings tested: {'; '.join(ALT_READINGS.values())}. The check can fail: it recomputes every verdict under each reading and lists the readings "
      + ("where the winner differs; none differ, so the verdicts do not depend on these choices.\n" if not n_alt else
         f"where the winner differs ({n_alt} of {len(alt)} account-size cells differ, see the last column).\n"))
    ol = one_leg_check(names)

    def phrase(acct):
        by = {}
        for v in SIZES:
            by.setdefault(vd[(acct, v)]["winner"], []).append(v)
        lab = {"TSX": "the TSX line", "US": "the US line", "tie": "no meaningful difference (tie)"}
        if len(by) == 1:
            return f"{lab[next(iter(by))]} at every size modelled (C${SIZES[0]:,} to C${SIZES[-1]:,})"
        return "; ".join(f"{lab[k]} at {', '.join(f'C${v:,}' for v in vs)}" for k, vs in by.items())
    plain = (f"Plain words (generated from the table above): with CAD cash the cheaper line is {phrase('CAD')}; with USD cash it is {phrase('USD')}. "
             f"The saving is the median over names of the per-name cost difference between the two lines. ")
    if ol["min"] is not None:
        plain += (f"If you will never convert back (only one FX leg is paid), the same winner holds in {ol['holds']} of {ol['total']} name-size-account cases; "
                  f"the single cheapest FX leg is at least {ol['min'][0]:.2f}x (worst: {ol['min'][1]} at C${ol['min'][2]:,}) and a median {ol['median']:.1f}x "
                  "the round-trip cost advantage the line that needs FX has before FX."
                  + (f" Cases where the ordering does not hold: {', '.join(f'{p} {a} C${v:,}' for p, v, a in ol['fails'][:8])}"
                     f"{' ...' if len(ol['fails']) > 8 else ''}." if ol["fails"] else ""))
    else:
        plain += f"The line that needs FX is never cheaper before FX in any name-size-account case ({ol['holds']} of {ol['total']} orderings hold with one FX leg)."
    w(plain + "\n")
    w("## 7. Tax note (text only, not modelled)\n")
    w(tax + "\n")
    w("## 8. Caveats\n")
    w("- Spreads are high-low estimates, not quotes; depth and market impact at C$100k-250k are not captured. Check live bid/ask for the less liquid names.")
    w("- Yahoo `.TO` volume probably excludes US venues and US-line volume is consolidated (UNVERIFIED), so US/TSX ADV ratios are biased up.")
    w("- m_aff (desk affiliate margin over mid) is 0 as in the Norbert's gambit report, so the FX-desk routes are lower bounds on cost.")
    w("- In-trade FX (c') is an UNVERIFIED reading of the broker's page. The currency basis of the broker's tier is unverified; the last column of the verdict table shows whether any verdict flips under the alternative readings.")
    w("- The universe is hand-curated and ranked by computed dollar ADV; see section 2.")
    w("- Close-to-close parity uses the BoC session-average rate; the gap is small and one-sided in sign (section 5), cause UNVERIFIED, and not treated as a tradable edge.")
    w("- Journal delay risk of Norbert's gambit is a risk, not a cost, and is not in these numbers (d = 0), and the gambit's half-spreads are upward-biased estimates; the net direction of the error in the NG numbers is unknown (see the Norbert note).")
    w(f"- UNRESOLVED: the commission the broker charges on the DLR-U.TO leg of the gambit (default US${k_def:g}, alternative US${k_alt:g}); not confirmed.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
TAX_TEXT = """**UNVERIFIED. Not reviewed by a tax adviser; do not rely on this section.** It records what was read and what was not.

- Common statement: "US-line dividends in a TFSA lose 15% to US withholding with no recovery; in an RRSP the treaty exempts it". It is stated for dividends from **US-domiciled issuers**. Whether the same withholding applies to a Canadian-incorporated issuer's dividend on its US line was not established here.
- Source rule read: dividends from a domestic (US) corporation are US-source income; dividends from a foreign corporation are US-source only if at least 25% of its gross income over the prior three years was effectively connected with a US trade or business, and then only in the pro-rata share that income bears to its total (26 U.S.C. 861(a)(2)(B), https://www.law.cornell.edu/uscode/text/26/861, read 2026-10-23). Source is not the same as withholding: the withholding rules themselves (26 U.S.C. 871/881 and 1441/1442, the Canada-US treaty, the W-8BEN and the custody chain at the broker) were NOT read.
- **The 25% test of 861(a)(2)(B) was NOT applied** to WCN, CP, TRI or QSR (or to any other name in the list); issuers with large US operations may meet it. Treat every Canadian-incorporated name as unchecked, not as cleared.
- Where US withholding does apply (US-domiciled issuer): 15% in a TFSA, not recoverable, because the treaty does not treat a TFSA as a retirement account; RRSP/RRIF exempt, per secondary sources that cite Canada-US treaty Article XXI(2) (the article number and the non-recoverability reason are from those sources and unverified here). Sources (secondary, not the treaty text): https://www.fool.ca/2026/10/15/your-tfsa-pays-15-tax-on-u-s-dividends-your-rrsp-doesnt/, plus three broker or adviser educational pages (URLs omitted). The treaty text and a CRA / IRS publication for the non-recoverability point were not read.
- Domicile was checked for **{flagged} only** (hand-curated set `US_DOMICILED`). {us_dom} The {n_unchecked} names in the final list that are not flagged are **unchecked** (country of incorporation not looked up); this is not a finding that they are Canadian-domiciled. Partnership units (e.g. Bermuda LPs) were not considered.
- Not modelled: the cost of converting USD dividends from the US line back to CAD, and the capital-gains and superficial-loss treatment of the two lines. This is not tax advice."""


def tax_text(names: list[dict]) -> str:
    us_dom = sorted(n["pair"] for n in names if n["pair"].split("|")[1] in US_DOMICILED)
    return TAX_TEXT.format(us_dom=(f"Flagged names in the final list: {', '.join(us_dom)}." if us_dom else
                                  "None of the flagged names is in the final list, so no name in the final list has had its domicile checked."),
                           flagged=", ".join(sorted(US_DOMICILED)), n_unchecked=len(names) - len(us_dom))


def ng_inputs(fx: float) -> dict:
    nd = ce.load(NORBERT_SNAPSHOT)
    res = ce.analyse(nd)
    h_sum = res["win_dlr"]["used"] + res["win_dlru"]["used"]
    bt = {}
    for k_usd in (DLRU_COMMISSION_USD, DLRU_ALT_USD):
        k_eff = (ce.COMMISSION + k_usd * fx) / 2.0
        v_star, holds = ce.breakeven(k_eff, h_sum, 0.0, fx=fx, v_max=max(SIZES))
        txt = "never" if v_star is None else f"C${v_star:,.0f}, and stays cheaper at every size up to C${max(SIZES):,}" if holds \
            else f"C${v_star:,.0f}, but loses again at a higher desk tier below C${max(SIZES):,}"
        fxs = [min(ce.c_fx_desk(v / fx), ce.c_ng(v, k_eff, h_sum, 0.0, 0.0)) for v in SIZES]
        bt[k_usd] = (txt, f"{min(fxs) * 1e4:.1f} to {max(fxs) * 1e4:.1f}")
    return {"h_sum": h_sum, "window": f"{res['window'][0].date()} to {res['window'][1].date()}", "bt": bt,
            "fx_range_text": bt[DLRU_COMMISSION_USD][1]}


def write_csvs(names: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        for v in SIZES:
            r = n["routes"][v]
            rows.append({"pair": n["pair"], "V_cad": v, **{k: r[k] for k in ("a", "b", "c", "c_in", "d", "e", "e_desk", "ng_wins")},
                         **{f"{k}_dlru0": n["routes_dlru0"][v][k] for k in ("d", "e", "ng_wins")},
                         "h_tsx": n["h_tsx"], "h_us": n["h_us"], "h_tsx_commoncal": n["h_tsx_common"], "h_us_commoncal": n["h_us_common"]})
    pd.DataFrame(rows).to_csv(outdir / "route_costs.csv", index=False, float_format="%.6g")
    yrs = []
    for n in names:
        for side, tab in zip(("tsx", "us"), n["spread_by_year"]):
            t = tab.reset_index()
            t.insert(0, "side", side)
            t.insert(0, "pair", n["pair"])
            yrs.append(t)
    pd.concat(yrs).to_csv(outdir / "spreads_by_year.csv", index=False, float_format="%.6g")


def run(snap: Path, out: Path | None) -> str:
    data = load(snap)
    boc = data["boc"]
    fx, fx_date = float(boc.iloc[-1]), boc.index[-1]
    ng = ng_inputs(fx)
    asof = max(d.index.max() for l in data["lines"].values() for d in l.values())
    results = [analyse_pair(p, data["lines"].get(p, {}), boc, fx, ng["h_sum"], asof=asof) for p in
               sorted({f"{t}|{u}" for t, u in CANDIDATES})]
    names, dropped = select_universe(results)
    doc = render(names, dropped, data, ng, fx, fx_date, tax_text(names), passing=[r for r in results if not r["drop"]])
    write_csvs(names, snap)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="reuse an existing snapshot dir instead of fetching")
    ap.add_argument("--out", default=str(ROOT / "docs" / "research_notes" / "2026-10-23_duallisting_venue_results.md"))
    a = ap.parse_args()
    snap = Path(a.snapshot) if a.snapshot else ROOT / "research" / "cost_edge_duallisting" / f"snapshot_{dt.date.today().isoformat()}"
    if not a.snapshot:
        fetch(snap)
    print(run(snap, Path(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
