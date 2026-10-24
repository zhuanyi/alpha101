#!/usr/bin/env python3
"""Norbert's gambit (DLR.TO <-> DLR-U.TO) vs the retail broker FX conversion.

    python3 scripts/cost_edge_norbert.py [--snapshot DIR] [--out docs/research_notes/2026-10-21_norbert_gambit_results.md]

Default: fetch fresh Yahoo + Bank of Canada data into research/cost_edge_norbert/snapshot_<today>/ and
recompute everything from it. `--snapshot DIR` recomputes from an existing snapshot without any network.
Computed from the snapshot: the spreads, the implied-rate deviation, the volatilities and every saving/breakeven figure.
Input constants (not fetched or re-verified here): The broker's tiers, commission and tick below, taken from the broker's published fee pages as read on 2026-10-18.

Conventions (one-way costs are fractions of the CAD-equivalent notional V):
  c_FX(V)  = listed_spread(tier) * 4/3 + m_aff              (m_aff = 0: unknown, so NG savings are a lower bound)
  c_NG(V)  = 2k/V + h_DLR + h_DLRU + d_dir                   (h = half-spread, d = implied-rate deviation)
  d_c2u    = 1 - r_boc / r_imp   (CAD->USD loses when the implied rate is above BoC)
  d_u2c    = 1 - r_imp / r_boc   (USD->CAD loses when it is below)
  r_imp    = DLR.TO close / DLR-U.TO close (CAD per USD); r_boc = Bank of Canada FXUSDCAD.
  The BoC daily rate is a truncated mean of 480 one-minute indicative mid quotes (08:00-16:00 ET, top/bottom 2.5%
  dropped), so the implied-rate deviation vs the ~16:00 closes is mostly averaging-vs-close noise: the base case sets
  d = 0 and only the adverse (p95) deviation is used as a sensitivity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

NOTIONALS = [10_000, 25_000, 100_000, 250_000, 500_000]
DELAYS = [0, 1, 3]
COMMISSION = 9.99
TICK = 0.01
# desk "Gross Transaction Amount" tiers, listed spread. Input constants read from the broker's published fee pages on 2026-10-18, NOT re-verified by this script.
# (lower bound of the tier, listed spread). All-in = listed * 4/3 (affiliate surcharge = one third).
FX_DESK_TIERS = [(0.0, 0.0126), (25_000.0, 0.0078), (100_000.0, 0.0046), (500_000.0, 0.0027), (1_000_000.0, 0.0019)]
DESK_IN_TRADE_FLOOR = 0.0078  # "$60,000 or more" in a trade FX is charged the $50,000-$59,999 tier
DESK_IN_TRADE_FROM = 60_000
SURCHARGE = 4.0 / 3.0
CS_K = 3.0 - 2.0 * np.sqrt(2.0)


# ----------------------------------------------------------------------------- cost formulas
def desk_listed(amount: float, in_trade: bool = False) -> float:
    s = [sp for lo, sp in FX_DESK_TIERS if amount >= lo][-1]
    return max(s, DESK_IN_TRADE_FLOOR) if in_trade and amount >= DESK_IN_TRADE_FROM else s


def c_fx_desk(amount_in_basis: float, m_aff: float = 0.0, in_trade: bool = False) -> float:
    return desk_listed(amount_in_basis, in_trade) * SURCHARGE + m_aff


def c_ng(v: float, k: float, h_dlr: float, h_dlru: float, d: float) -> float:
    return 2.0 * k / v + h_dlr + h_dlru + d


def breakeven_intervals(k: float, h_sum: float, d: float, fx: float = 1.0, m_aff: float = 0.0):
    """Per desk tier, the CAD notional interval (lo, hi) where NG is cheaper than the FX desk (tiers with no such interval are omitted).

    The desk tier is chosen from V/fx (fx=1 means the tier is read off the CAD amount; fx=CAD per USD means
    off the USD-equivalent). c_FX is piecewise constant, so solve 2k/V = c_tier - h_sum - d tier by tier.
    """
    edges = [lo * fx for lo, _ in FX_DESK_TIERS] + [np.inf]
    out = []
    for i, (lo, sp) in enumerate(FX_DESK_TIERS):
        denom = sp * SURCHARGE + m_aff - h_sum - d
        if denom <= 0:
            continue
        start = max(2.0 * k / denom, edges[i])
        if start < edges[i + 1]:
            out.append((start, edges[i + 1]))
    return out


def breakeven(k: float, h_sum: float, d: float, fx: float = 1.0, m_aff: float = 0.0, v_max: float = max(NOTIONALS)):
    """(v_star, wins_all_above): v_star = smallest CAD notional where NG beats the FX desk (None if never).

    the FX desk's cost steps DOWN at higher tiers, so NG can lose again above v_star; wins_all_above is True only if NG is
    cheaper at every notional from v_star up to v_max (default: the largest notional modelled).
    """
    iv = breakeven_intervals(k, h_sum, d, fx, m_aff)
    if not iv:
        return None, False
    v_star, end = iv[0]
    for lo, hi in iv[1:]:
        if lo > end:
            break
        end = hi
    return v_star, end > v_max


# ----------------------------------------------------------------------------- spread estimators
def cs_half_spread(df: pd.DataFrame) -> pd.Series:
    """Corwin-Schultz (2012) two-day-window spread S, negatives clamped to 0; returns HALF spread per window."""
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    beta = hl2 + hl2.shift(-1)
    h2 = np.maximum(df["High"], df["High"].shift(-1))
    l2 = np.minimum(df["Low"], df["Low"].shift(-1))
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return (s.clip(lower=0.0) / 2.0).dropna()


def cs_clamp_share(df: pd.DataFrame) -> pd.Series:
    """Share of two-day windows whose raw CS spread was <= 0 (clamped), per window index."""
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    beta = hl2 + hl2.shift(-1)
    gamma = np.log(np.maximum(df["High"], df["High"].shift(-1)) / np.minimum(df["Low"], df["Low"].shift(-1))) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    return (alpha <= 0).where(alpha.notna()).dropna()


def ar_half_spread_month(df: pd.DataFrame, min_obs: int = 8) -> pd.Series:
    """Abdi-Ranaldo (2017) close-high-low estimator, one value per calendar month.

    s^2 = 4 * mean[(c_t - eta_t)(c_t - eta_{t+1})], eta = log mid-range; s = sqrt(max(s^2, 0)).
    Returns the HALF spread (s/2), indexed by month period.
    """
    c = np.log(df["Close"])
    eta = (np.log(df["High"]) + np.log(df["Low"])) / 2.0
    prod = ((c - eta) * (c - eta.shift(-1))).dropna()
    out = {}
    for m, g in prod.groupby(prod.index.to_period("M")):
        if len(g) < min_obs:
            continue
        out[m] = np.sqrt(max(4.0 * g.mean(), 0.0)) / 2.0
    return pd.Series(out, dtype=float)


def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year: CS half, AR half, half-tick floor, used = max of the three (all fractions of price)."""
    cs = cs_half_spread(df)
    ar = ar_half_spread_month(df)
    rows = []
    for y in sorted(set(df.index.year)):
        cs_y = cs[cs.index.year == y]
        ar_y = ar[[p.year == y for p in ar.index]] if len(ar) else ar
        sub = df[df.index.year == y]
        cs_v = float(cs_y.mean()) if len(cs_y) else np.nan
        ar_v = float(ar_y.mean()) if len(ar_y) else np.nan
        tick = TICK / 2.0 / float(sub["Close"].median())
        used = np.nanmax([cs_v, ar_v, tick])
        rows.append({"year": y, "days": len(sub), "cs_half": cs_v, "ar_half": ar_v, "half_tick": tick, "used": used})
    return pd.DataFrame(rows).set_index("year")


def spread_window(df: pd.DataFrame) -> dict:
    cs = cs_half_spread(df)
    ar = ar_half_spread_month(df)
    cs_v = float(cs.mean()) if len(cs) else np.nan
    ar_v = float(ar.mean()) if len(ar) else np.nan
    tick = TICK / 2.0 / float(df["Close"].median())
    return {"cs_half": cs_v, "ar_half": ar_v, "half_tick": tick, "used": float(np.nanmax([cs_v, ar_v, tick])),
            "cs_clamped_share": float(cs_clamp_share(df).mean()), "days": len(df)}


# ----------------------------------------------------------------------------- data
def daily_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    d = h.copy()
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index).normalize()
    g = d.groupby(level=0)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                         "Close": g["Close"].last(), "Volume": g["Volume"].sum()})


def fetch(snap: Path) -> None:
    import requests
    import yfinance as yf
    snap.mkdir(parents=True, exist_ok=True)
    for sym, tag in (("DLR.TO", "dlr"), ("DLR-U.TO", "dlru")):
        t = yf.Ticker(sym)
        for kind, kw in (("daily", {"interval": "1d"}), ("hourly", {"interval": "1h"})):
            h = t.history(period="max", auto_adjust=False, **kw)
            h.index = h.index.tz_localize(None) if h.index.tz is not None else h.index
            h.index.name = "ts"
            h[["Open", "High", "Low", "Close", "Volume"]].to_csv(snap / f"{tag}_{kind}.csv")
    r = requests.get("https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date=2000-01-01", timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    pd.DataFrame({"date": [o["d"] for o in obs], "FXUSDCAD": [float(o["FXUSDCAD"]["v"]) for o in obs]}) \
        .to_csv(snap / "boc_fxusdcad.csv", index=False)
    (snap / "meta.json").write_text(json.dumps({"fetched": dt.date.today().isoformat(),
                                                "source": "yfinance Ticker.history (auto_adjust=False); BoC Valet FXUSDCAD"}))


def load(snap: Path) -> dict:
    rd = lambda n: pd.read_csv(snap / n, index_col=0, parse_dates=True)  # noqa: E731
    boc = pd.read_csv(snap / "boc_fxusdcad.csv", parse_dates=["date"]).set_index("date")["FXUSDCAD"]
    return {"dlr_daily": rd("dlr_daily.csv").dropna(subset=["Close"]), "dlru_daily": rd("dlru_daily.csv"), "dlr_hourly": rd("dlr_hourly.csv"), "dlru_hourly": rd("dlru_hourly.csv"), "boc": boc,
            "meta": json.loads((snap / "meta.json").read_text())}


# ----------------------------------------------------------------------------- analysis
def deviation_series(dlr: pd.DataFrame, dlru: pd.DataFrame, boc: pd.Series) -> pd.DataFrame:
    j = pd.concat({"p_cad": dlr["Close"], "p_usd": dlru["Close"], "boc": boc}, axis=1, join="inner").dropna()
    j["r_imp"] = j["p_cad"] / j["p_usd"]
    j["dev_bp"] = (j["r_imp"] / j["boc"] - 1.0) * 1e4
    j["d_c2u"] = 1.0 - j["boc"] / j["r_imp"]
    j["d_u2c"] = 1.0 - j["r_imp"] / j["boc"]
    return j


def dist(x: pd.Series) -> dict:
    return {"n": int(x.count()), "mean": float(x.mean()), "median": float(x.median()),
            "p5": float(x.quantile(0.05)), "p95": float(x.quantile(0.95))}


def analyse(data: dict) -> dict:
    dlr_h, dlru_h = daily_from_hourly(data["dlr_hourly"]), daily_from_hourly(data["dlru_hourly"])
    start, end = dlru_h.index.min(), dlru_h.index.max()
    dlr_ov = dlr_h.loc[start:end]
    dev = deviation_series(dlr_ov, dlru_h, data["boc"])
    fx = float(data["boc"].iloc[-1])
    res = {"window": (start, end), "fx": fx, "fx_date": data["boc"].index[-1], "dev": dev,
           "dev_by_year": {y: {"dev_bp": dist(g["dev_bp"]), "c2u_bp": dist(g["d_c2u"] * 1e4), "u2c_bp": dist(g["d_u2c"] * 1e4)}
                           for y, g in dev.groupby(dev.index.year)},
           "spreads_dlr_by_year": spread_table(data["dlr_daily"]),
           "spreads_dlru_by_year": spread_table(dlru_h),
           "win_dlr": spread_window(dlr_ov), "win_dlru": spread_window(dlru_h),
           "missing_boc": [x for x in dlru_h.index if x not in data["boc"].index]}
    ret = lambda df: np.log(df["Close"]).diff().dropna()  # noqa: E731
    res["sigma"] = {"dlr_full": float(ret(data["dlr_daily"]).std()), "dlr_overlap": float(ret(dlr_ov).std()),
                    "dlru_overlap": float(ret(dlru_h).std())}
    return res


# scenario name -> (half-spreads doubled, p95 adverse deviation)
SCEN_FLAGS = {"base": (False, False), "half-spread x2": (True, False), "adverse d (p95)": (False, True), "x2 + adverse": (True, True)}
SCENARIOS = list(SCEN_FLAGS)


def scenario_saving(v, fx, d_exp, d_adv, h_sum, k, basis, scen):
    """Return (c_FX, c_NG, saving fraction) for one notional under a scenario name."""
    if scen not in SCEN_FLAGS:
        raise ValueError(f"unknown scenario {scen!r}; expected one of {SCENARIOS}")
    x2, adverse = SCEN_FLAGS[scen]
    h = h_sum * (2.0 if x2 else 1.0)
    d = d_adv if adverse else d_exp
    amt = v / fx if basis == "usd" else v
    desk = c_fx_desk(amt)
    ng = c_ng(v, k, h, 0.0, d)
    return desk, ng, desk - ng




def pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"


def bp(x, nd=1):
    return f"{x * 1e4:.{nd}f}"


def desk_tier_text() -> str:
    """Human-readable desk cash-FX tier list, built from FX_DESK_TIERS (the single source of the tier text)."""
    parts = []
    for i, (lo, sp) in enumerate(FX_DESK_TIERS):
        if i == 0:
            parts.append(f"{sp * 100:.2f}% below ${FX_DESK_TIERS[1][0]:,.0f}")
        elif i + 1 < len(FX_DESK_TIERS):
            parts.append(f"{sp * 100:.2f}% for ${lo:,.0f}-${FX_DESK_TIERS[i + 1][0] - 1:,.0f}")
        else:
            parts.append(f"{sp * 100:.2f}% from ${lo:,.0f}")
    return "; ".join(parts)


def render(res: dict, data: dict) -> str:
    fx, start, end = res["fx"], *res["window"]
    dev = res["dev"]
    h_sum = res["win_dlr"]["used"] + res["win_dlru"]["used"]
    L = []
    w = L.append
    w("# Norbert's gambit vs the broker's FX conversion\n")
    w("Cost parameters (commission, spreads, FX tiers): see [`docs/cost_assumptions.md`](../cost_assumptions.md).\n")
    fetched = data["meta"]["fetched"]
    w(f"Data fetched {fetched} ({data['meta']['source']}). Generated by `scripts/cost_edge_norbert.py`; "
      "the spread, deviation, volatility and saving figures are computed from the local snapshot in `research/cost_edge_norbert/`. "
      "The broker's tiers, commission and tick size are input constants read from the broker's published fee pages on 2026-10-18 and were not re-verified here.\n")
    w("## 1. Data coverage (read first)\n")
    n_sess = int(res["spreads_dlru_by_year"]["days"].sum())
    n_daily = len(data["dlru_daily"])
    n_daily_txt = f"{ {1: 'one'}.get(n_daily, n_daily) } row{'' if n_daily == 1 else 's'}"
    w(f"- **DLR-U.TO daily history on Yahoo is essentially absent**: the daily endpoint returned {n_daily_txt} on {fetched}, and the "
      f"hourly endpoint starts on {start.date()} ({n_sess} sessions to {end.date()}). Daily bars for DLR-U.TO are therefore built "
      "from the 60-minute bars (open of first bar, max high, min low, close of the last bar).")
    w(f"- So the implied-rate deviation and the DLR-U.TO spread exist **only for {start.date()} to {end.date()}**. "
      "No older period is stitched in or proxied. Treat every NG number below as a ~3-month sample.")
    w("- DLR.TO has a long daily history, used only for its per-year spread estimates and for its volatility.\n")
    w("## 2. Implied rate vs Bank of Canada\n")
    miss = res["missing_boc"]
    miss_txt = (f"{n_sess} sessions in section 1, {n_sess - len(miss)} with a BoC rate: "
                + ", ".join(str(x.date()) for x in miss) + " has no rate"
                + (" (National Day for Truth and Reconciliation, the Bank of Canada was closed)"
                   if [str(x.date()) for x in miss] == ["2026-09-30"] else "") + ". ") if miss else ""
    w("Implied rate = DLR.TO close / DLR-U.TO close, both from the last 60-minute bar of the session (same source, same bar). "
      "The BoC daily rate FXUSDCAD is a truncated mean of 480 one-minute indicative mid quotes between 08:00 and 16:00 ET "
      "(top and bottom 2.5% dropped), published by 16:30 ET "
      "([method](https://www.bankofcanada.ca/rates/exchange/background-information-on-foreign-exchange-rates/)). "
      "It is a session average, not a closing rate, so the deviation against ~16:00 closes is mostly averaging-vs-close noise, not cost. "
      "**The base case therefore sets d = 0 and does not credit the mean deviation**; the p95 adverse deviation is kept as a separate sensitivity. "
      + miss_txt + "Positive deviation = implied rate above BoC (CAD->USD loses, USD->CAD gains).\n")
    w("| year | n | mean bp | median bp | p5 bp | p95 bp |\n|---|---|---|---|---|---|")
    for y, s in res["dev_by_year"].items():
        d = s["dev_bp"]
        w(f"| {y} | {d['n']} | {d['mean']:.1f} | {d['median']:.1f} | {d['p5']:.1f} | {d['p95']:.1f} |")
    w("\nCost of the deviation by direction (bp of notional; mean is NOT credited in the base case, p95 = adverse case):\n")
    w("| direction | mean | median | p5 | p95 |\n|---|---|---|---|---|")
    for name, col in (("CAD->USD", "d_c2u"), ("USD->CAD", "d_u2c")):
        d = dist(dev[col] * 1e4)
        w(f"| {name} | {d['mean']:.1f} | {d['median']:.1f} | {d['p5']:.1f} | {d['p95']:.1f} |")
    w("")
    w("## 3. Half-spread estimates (ESTIMATES, no quote data)\n")
    w("Half-spread as % of price. CS = Corwin-Schultz (2012) on two-day windows; AR = Abdi-Ranaldo (2017), one value per calendar "
      "month with at least 8 observations. Negative per-window estimates are clamped to 0 (standard convention). "
      f"Half-tick = ${TICK / 2:.3f} / median close (the physical minimum: both lines trade in ${TICK:.2f} ticks). "
      "**Used = max(CS, AR, half-tick)** - the half-tick term goes one step beyond the plain CS/AR estimators because on tick-bound "
      "ETFs the high-low estimators can sit below the minimum tick.\n")
    for title, key in (("DLR.TO, daily Yahoo bars, per year", "spreads_dlr_by_year"),
                       ("DLR-U.TO, daily bars built from hourly bars", "spreads_dlru_by_year")):
        w(f"**{title}**\n")
        w("| year | days | CS half | AR half | half-tick | used |\n|---|---|---|---|---|---|")
        for y, r in res[key].iterrows():
            f = lambda v: "n/a" if pd.isna(v) else pct(v, 3)  # noqa: E731
            w(f"| {y} | {int(r['days'])} | {f(r['cs_half'])} | {f(r['ar_half'])} | {f(r['half_tick'])} | {f(r['used'])} |")
        w("")
    wd, wu = res["win_dlr"], res["win_dlru"]
    w(f"Window used for the cost model ({start.date()} to {end.date()}, same days for both lines): "
      f"DLR.TO half = {pct(wd['used'], 3)} (CS {pct(wd['cs_half'], 3)}, AR {pct(wd['ar_half'], 3)}, tick {pct(wd['half_tick'], 3)}; "
      f"{wd['cs_clamped_share'] * 100:.0f}% of CS windows clamped to 0); DLR-U.TO half = {pct(wu['used'], 3)} "
      f"(CS {pct(wu['cs_half'], 3)}, AR {pct(wu['ar_half'], 3)}, tick {pct(wu['half_tick'], 3)}; "
      f"{wu['cs_clamped_share'] * 100:.0f}% clamped). Sum h = {pct(h_sum, 3)} ({bp(h_sum)} bp).\n")
    w("## 4. Cost model\n")
    w(f"c_FX(V) = listed_spread(tier) x 4/3 (+ m_aff, set to 0). c_NG(V) = 2k/V + h_DLR + h_DLRU + d, k = C${COMMISSION}, d = 0 in the base case. "
      f"V is the CAD-equivalent notional; USD-equivalent uses the latest BoC rate {fx:.4f} ({res['fx_date'].date()}). "
      "the desk's tier is read off the *Gross Transaction Amount*, whose currency basis is UNVERIFIED, so both bases are shown: "
      "**USD basis** (amount = V / BoC, primary, matches the page's USD examples) and **CAD basis**.\n")
    d_exp = {"c2u": 0.0, "u2c": 0.0}  # BoC is a session mean, not a close: the observed mean deviation is not credited
    d_adv = {"c2u": float(dev["d_c2u"].quantile(0.95)), "u2c": float(dev["d_u2c"].quantile(0.95))}
    verdicts, cad_stress = {}, {}
    for dname, key in (("CAD->USD", "c2u"), ("USD->CAD", "u2c")):
        w(f"### 4.{1 if key == 'c2u' else 2} {dname}\n")
        w(f"Base d = {bp(d_exp[key])} bp (observed mean {bp(float(dev['d_c2u' if key == 'c2u' else 'd_u2c'].mean()))} bp not credited), "
          f"adverse (p95) d = {bp(d_adv[key])} bp.\n")
        w("| V (CAD) | desk cost, USD basis | NG cost | saving C$ | saving bp | desk cost, CAD basis | saving C$ | saving bp |\n|---|---|---|---|---|---|---|---|")
        for v in NOTIONALS:
            tdu, ng, su = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "usd", "base")
            tdc, _, sc = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "cad", "base")
            w(f"| {v:,} | {pct(tdu)} | {pct(ng)} | {su * v:,.0f} | {bp(su)} | {pct(tdc)} | {sc * v:,.0f} | {bp(sc)} |")
        w("\nSensitivity, saving C$ (bp) on the USD basis:\n")
        w("| V (CAD) | " + " | ".join(SCENARIOS) + " |\n|---|" + "---|" * len(SCENARIOS))
        for v in NOTIONALS:
            cells = []
            for s in SCENARIOS:
                _, _, sv = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "usd", s)
                cells.append(f"{sv * v:,.0f} ({bp(sv)})")
            w(f"| {v:,} | " + " | ".join(cells) + " |")
        w(f"\nBreakeven notional V*, C$: the first V where NG is cheaper than the FX desk. The FX desk's cost steps down at higher tiers, so "
          f"'holds' states whether NG stays cheaper at every higher notional up to C${max(NOTIONALS):,}; otherwise the "
          "cell says NG loses again above a tier edge.\n")
        w("| scenario | USD basis | CAD basis |\n|---|---|---|")
        for s_ in SCENARIOS:
            hh = h_sum * (2.0 if SCEN_FLAGS[s_][0] else 1.0)
            dd = d_adv[key] if SCEN_FLAGS[s_][1] else d_exp[key]

            def fmt(b):
                v_star, holds = b
                return "never" if v_star is None else f"{v_star:,.0f} ({'holds' if holds else 'NG loses again at a higher tier'})"
            w(f"| {s_} | {fmt(breakeven(COMMISSION, hh, dd, fx=fx))} | {fmt(breakeven(COMMISSION, hh, dd, fx=1.0))} |")
        # verdict per size (USD basis)
        for v in NOTIONALS:
            base = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "usd", "base")[2]
            stress = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "usd", "x2 + adverse")[2]
            cadb = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "cad", "base")[2]
            cad_td, cad_ng, cads = scenario_saving(v, fx, d_exp[key], d_adv[key], h_sum, COMMISSION, "cad", "x2 + adverse")
            verdicts[(key, v)] = (base, stress, cadb, cads)
            cad_stress[(key, v)] = (cad_td, cad_ng, cads)
        w("")
    for sub, (dname, key) in zip("ab", (("CAD->USD", "c2u"), ("USD->CAD", "u2c"))):
        w(f"### 4.3{sub} Commission variants ({dname}, USD basis, base scenario; saving C$)\n")
        w("| V (CAD) | k = 9.99 both legs | k = 0 (if DLR were a $0 Select ETF) | USD-leg commission US$9.99 (= C$ at BoC) |\n|---|---|---|---|")
        for v in NOTIONALS:
            desk = c_fx_desk(v / fx)
            s1 = (desk - c_ng(v, COMMISSION, h_sum, 0.0, d_exp[key])) * v
            s0 = (desk - c_ng(v, 0.0, h_sum, 0.0, d_exp[key])) * v
            s2 = desk * v - (v * (h_sum + d_exp[key]) + COMMISSION * (1 + fx))
            w(f"| {v:,} | {s1:,.0f} | {s0:,.0f} | {s2:,.0f} |")
        w("")
    w("### 4.4 If the desk's FX were done inside a trade (UNVERIFIED reading of the page)\n")
    w(f"In-trade FX with gross amount >= ${DESK_IN_TRADE_FROM:,} is charged the $50,000-$59,999 tier ({DESK_IN_TRADE_FLOOR * 100:.2f}% listed = "
      f"{DESK_IN_TRADE_FLOOR * SURCHARGE * 100:.2f}% all-in). Saving C$ (bp), CAD->USD, USD basis:\n")
    w("| V (CAD) | desk in-trade cost | saving C$ | saving bp |\n|---|---|---|---|")
    for v in NOTIONALS:
        desk = c_fx_desk(v / fx, in_trade=True)
        sv = desk - c_ng(v, COMMISSION, h_sum, 0.0, d_exp["c2u"])
        w(f"| {v:,} | {pct(desk)} | {sv * v:,.0f} | {bp(sv)} |")
    w("\n## 5. Journal-delay risk (risk, NOT a cost)\n")
    w("After the journal you sell the *other* line, so the risk is the price risk of the line you sell: "
      "CAD->USD sells DLR-U.TO (USD-quoted, close to cash; the CAD->USD rate was locked when you bought DLR.TO, so what remains is tick noise), "
      "USD->CAD sells DLR.TO (effectively USD/CAD volatility). 1-sigma = daily log-return std x sqrt(days); "
      f"std: DLR-U.TO {pct(res['sigma']['dlru_overlap'], 3)} (overlap window, mostly tick bounce), DLR.TO {pct(res['sigma']['dlr_full'], 3)} (full history), "
      f"{pct(res['sigma']['dlr_overlap'], 3)} (overlap window).\n")
    w("| delay (business days) | CAD->USD 1-sigma bp | USD->CAD 1-sigma bp (full history) | C$ at 100k, CAD->USD | C$ at 100k, USD->CAD | C$ at 500k, USD->CAD |\n|---|---|---|---|---|---|")
    for n in DELAYS:
        a, b = res["sigma"]["dlru_overlap"] * np.sqrt(n), res["sigma"]["dlr_full"] * np.sqrt(n)
        w(f"| {n} | {bp(a)} | {bp(b)} | {a * 100_000:,.0f} | {b * 100_000:,.0f} | {b * 500_000:,.0f} |")
    w("\nWhether the second line can be sold before the journal settles (which would remove the delay) is UNVERIFIED.\n")
    w("## 6. Verdict\n")
    w("Basis: USD-basis tier, base scenario, d = 0; 'robust' = NG still wins with half-spreads doubled and the p95 adverse deviation. "
      "Because m_aff is set to 0 the desk cost is a lower bound, so NG savings are lower bounds too.\n")
    w("| V (CAD) | CAD->USD | USD->CAD | CAD-basis tier check |\n|---|---|---|---|")
    for v in NOTIONALS:
        cells = []
        for key in ("c2u", "u2c"):
            base, stress = verdicts[(key, v)][:2]
            cells.append("Norbert's gambit (robust)" if stress > 0 else
                         ("Norbert's gambit (wins at base, not in stress)" if base > 0 else "desk FX"))
        cad = ", ".join("NG" if verdicts[(k, v)][2] > 0 else "desk" for k in ("c2u", "u2c"))
        w(f"| {v:,} | {cells[0]} | {cells[1]} | {cad} |")
    w("")
    w("CAD-basis tier, combined stress (half-spreads doubled and p95 adverse deviation); cost of each route as a % of the notional:\n")
    w("| V (CAD) | CAD->USD | USD->CAD |\n|---|---|---|")
    for v in NOTIONALS:
        cells = []
        for key in ("c2u", "u2c"):
            desk, ng, sv = cad_stress[(key, v)]
            cells.append(f"{'NG' if sv > 0 else 'desk'} (NG {pct(ng, 3)} vs desk {pct(desk, 3)})")
        w(f"| {v:,} | {cells[0]} | {cells[1]} |")
    fails = [(dname, v) for dname, key in (("CAD->USD", "c2u"), ("USD->CAD", "u2c")) for v in NOTIONALS if cad_stress[(key, v)][2] <= 0]
    n_cells = 2 * len(NOTIONALS)
    if fails:
        parts = []
        for dname, v in fails:
            desk, ng, _ = cad_stress[("c2u" if dname == "CAD->USD" else "u2c", v)]
            parts.append(f"{dname} at C${v:,} (NG {pct(ng, 3)} vs desk {pct(desk, 3)}, NG costs {bp(ng - desk)} bp more)")
        w(f"\n**Qualification:** 'robust' above holds at every size on the USD-basis tier. On the CAD-basis tier, {len(fails)} of "
          f"{n_cells} stressed cells loses under the combined stress: " + "; ".join(parts) + ".\n")
    else:
        w(f"\n'Robust' holds at every size on both tier bases (all {n_cells} CAD-basis stressed cells favour NG).\n")
    w("## 7. Caveats, verified and UNVERIFIED items\n")
    w("**Input constants, not re-verified:** the broker's tiers, commission and tick size are taken from the broker's published fee pages as read on 2026-10-18 and were not re-checked "
      f"against the broker's pages by this script. Currency Conversion Spread tiers (cash FX): {desk_tier_text()}; the one-third affiliate surcharge "
      f"(all-in = listed x 4/3); the in-trade sentence ('${DESK_IN_TRADE_FROM:,} or more' is charged the $50,000-$59,999 tier); "
      f"the ${COMMISSION} commission and the tick size ${TICK}.\n")
    w("**UNVERIFIED** (each one is a stated assumption above):\n")
    for t in ("Whether DLR / DLR.U is on the broker's $0 'Select ETFs' list: the pricing page names no tickers (a link to the list was not followed), so $9.99 per trade is kept.",
              "Whether the DLR.U sale is charged in USD (the variant column in 4.3a and 4.3b shows the effect).",
              "The currency basis of the broker's 'Gross Transaction Amount' (both bases shown; notionals near a tier boundary give different savings).",
              "m_aff, the margin of the broker's affiliate rate over mid-market, is unknown and set to 0: The FX desk's true cost is higher, NG savings are lower bounds.",
              "the desk's journaling procedure and how many business days it takes (0, 1, 3 modelled as risk only).",
              "Whether the second line can be sold before the journal settles (would remove the delay risk).",
              "The Global X fund terms (issuer pages returned HTTP 403 earlier).",
              "Half-spreads are ESTIMATES (Corwin-Schultz, Abdi-Ranaldo, half-tick floor), not quotes. Depth and market impact are not captured, which matters most at C$250k-500k; "
              "check live bid/ask and size before a large trade.",
              "The deviation sample is one quarter of sessions (n in section 2); the BoC rate is a session mean, so the deviation vs ~16:00 closes is mostly averaging noise (the base case does not credit it), and the adverse (p95) deviation partly overlaps the half-spread (closes sit inside the quotes), so adding both is conservative."):
        w(f"- {t}")
    w("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="reuse an existing snapshot dir instead of fetching")
    ap.add_argument("--out", default="docs/research_notes/2026-10-21_norbert_gambit_results.md")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    snap = Path(a.snapshot) if a.snapshot else root / "research" / "cost_edge_norbert" / f"snapshot_{dt.date.today().isoformat()}"
    if not a.snapshot:
        fetch(snap)
    data = load(snap)
    doc = render(analyse(data), data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
