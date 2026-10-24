#!/usr/bin/env python3
"""Options proxy study (stage 0, free data only): is there premium worth buying MX/ORATS data for?

    python3 scripts/options_proxy_study.py [--cache DIR] [--out docs/research_notes/2026-10-24_options_proxy.md]

Parts (every number in the generated note is computed here):
  1. Cboe BXM/PUT/BXMD/PPUT vs ^SP500TR (Yahoo): CAGR, vol, Sharpe (rf = 0), max DD, Newey-West t of monthly
     excess return, plus CAPM-style alpha. Index results, no retail cost. Monthly last-observation levels.
  2. Retail cost of a monthly covered call on XIU (commission + half-spread, parameters below) in bp/yr
     at several position sizes, vs the index premium; break-even size.
  3. Price-only bound for short-put assignment: TSX large caps, forward 21-day return after a 5-10% drawdown
     from the 20-day high vs unconditional; P(ret < -X) with X from a ~10-delta put strike set by realised vol.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import urllib.request
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
INDICES = ["BXM", "PUT", "BXMD", "PPUT"]
BENCH = "^SP500TR"
PANEL = Path(os.environ.get("TSX_PANEL", Path(__file__).resolve().parents[1] / "research/tsxpanel/data/panel.parquet"))

# Fallback hand list (Yahoo symbols) if the panel is unreadable.
HAND_LIST = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO", "MFC.TO", "SLF.TO", "BAM.TO", "ENB.TO",
    "TRP.TO", "CNQ.TO", "SU.TO", "CVE.TO", "IMO.TO", "TOU.TO", "CNR.TO", "CP.TO", "SHOP.TO", "CSU.TO",
    "ATD.TO", "L.TO", "DOL.TO", "WCN.TO", "NTR.TO", "ABX.TO", "AEM.TO", "WPM.TO", "FNV.TO", "TECK-B.TO",
    "BCE.TO", "T.TO", "RCI-B.TO", "FTS.TO", "H.TO", "TRI.TO", "GIB-A.TO", "QSR.TO", "POW.TO", "BN.TO",
]

# --- cost-model parameters (assumptions, not measurements) ---
COMMISSION_BASE = 9.99          # C$ per options order
COMMISSION_PER_CONTRACT = 1.25  # C$ per contract
HALF_SPREAD = 0.03      # C$ per share; XIU monthly ATM/OTM option half-spread. ASSUMPTION - the point of buying data.
SIZES = [25_000, 100_000, 250_000]
HORIZON = 21
DD_LO, DD_HI = 0.05, 0.10
TARGET_DELTA = 0.10
VOL_WIN = 60


# ---------------------------------------------------------------- part 1
def month_levels(s: pd.Series) -> pd.Series:
    """Last observation of each calendar month, indexed by month-end Period."""
    s = s.dropna().sort_index()
    m = s.groupby(s.index.to_period("M")).last()
    return m.reindex(pd.period_range(m.index[0], m.index[-1], freq="M"))


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """t-stat of the mean with Bartlett-weighted long-run variance."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    lr = e @ e / n
    for k in range(1, lags + 1):
        lr += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / sqrt(lr / n))


def perf(levels: pd.Series) -> dict:
    """Stats from monthly levels (NaN levels break the chain; those months' returns are dropped)."""
    r = levels.pct_change(fill_method=None).dropna()
    yrs = len(r) / 12
    eq = (1 + r).cumprod()
    return {"n_months": len(r), "cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * sqrt(12),
            "sharpe": r.mean() * 12 / (r.std() * sqrt(12)), "maxdd": (pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)]).pipe(lambda e: e / e.cummax() - 1)).min()}


def compare(idx: pd.Series, bench: pd.Series, start: str | None = None) -> dict:
    """Index vs benchmark on common months; excess-return t-stat and CAPM alpha."""
    a, b = month_levels(idx), month_levels(bench)
    df = pd.concat([a.rename("i"), b.rename("b")], axis=1).dropna()
    if start:
        df = df[df.index >= pd.Period(start, "M")]
    full = df.reindex(pd.period_range(df.index[0], df.index[-1], freq="M"))
    ri = full.i.pct_change(fill_method=None)
    rb = full.b.pct_change(fill_method=None)
    ok = pd.concat([ri, rb], axis=1).dropna()
    ok.columns = ["i", "b"]
    ex = (ok.i - ok.b).values
    beta = np.cov(ok.i, ok.b)[0, 1] / ok.b.var()
    alpha_m = ok.i.mean() - beta * ok.b.mean()
    res = ok.i - beta * ok.b - alpha_m
    se = sqrt(res.var() / len(ok))
    out = {"start": str(df.index[0]), "end": str(df.index[-1]),
           "idx": perf(full.i), "bench": perf(full.b),
           "excess_mean_ann": ex.mean() * 12, "excess_nw_t": newey_west_t(ex),
           "beta": beta, "alpha_ann": alpha_m * 12, "alpha_t": alpha_m / se}
    out["cagr_diff"] = out["idx"]["cagr"] - out["bench"]["cagr"]
    return out


# ---------------------------------------------------------------- part 2
def covered_call_cost_bp(size: float, price: float, half_spread: float = HALF_SPREAD,
                         close_frac: float = 0.0) -> float:
    """Annual cost in bp of size for 12 monthly short calls, 1 contract per 100 shares.

    Each month: sell-to-open commission + half-spread on the sold call; with probability close_frac the
    call is bought back (second commission + second half-spread) instead of expiring/being assigned.
    """
    n = size / (100 * price)  # continuous contracts; commission fixed part is per order
    per_month = (COMMISSION_BASE + COMMISSION_PER_CONTRACT * n + 100 * n * half_spread) * (1 + close_frac)
    return 12 * per_month / size * 1e4


def breakeven_size(premium_bp: float, price: float, half_spread: float = HALF_SPREAD,
                   close_frac: float = 0.0):
    """Smallest size where cost <= premium; None if the variable-cost floor already exceeds it."""
    f = 1 + close_frac
    floor = 12 * f * (COMMISSION_PER_CONTRACT + 100 * half_spread) / (100 * price) * 1e4
    if premium_bp <= floor:
        return None
    return 12 * f * COMMISSION_BASE * 1e4 / (premium_bp - floor)


# ---------------------------------------------------------------- part 3
def put_strike_drop(sigma_ann: float, delta: float = TARGET_DELTA, T: float = HORIZON / 252) -> float:
    """X such that a put struck at S*(1-X) has |BS delta| = delta (r = q = 0)."""
    d1 = norm.ppf(1 - delta)
    return 1 - exp(-(d1 * sigma_ann * sqrt(T) - 0.5 * sigma_ann ** 2 * T))


def episode_starts(cond: pd.Series, sep: int = HORIZON) -> pd.Series:
    """Keep a flagged day only if no kept day lies within the previous `sep` trading days (non-overlapping windows)."""
    keep = np.zeros(len(cond), bool)
    last = -10 ** 9
    for i, c in enumerate(cond.values):
        if c and i - last >= sep:
            keep[i], last = True, i
    return pd.Series(keep, index=cond.index)


def drawdown_events(px: pd.Series) -> pd.DataFrame:
    """Per-day frame: dd from 20d high, forward 21d return, trailing realised vol, strike drop X."""
    px = px.dropna()
    dd = px / px.rolling(20).max() - 1
    fwd = px.shift(-HORIZON) / px - 1
    vol = np.log(px).diff().rolling(VOL_WIN).std() * sqrt(252)
    df = pd.DataFrame({"dd": dd, "fwd": fwd, "vol": vol}).dropna()
    df["X"] = df.vol.map(put_strike_drop)
    df["hit"] = df.fwd < -df.X
    df["cond"] = (df.dd <= -DD_LO) & (df.dd > -DD_HI)
    return df


def pool_events(prices: dict[str, pd.Series]) -> dict:
    frames, conds = [], []
    for t, s in prices.items():
        d = drawdown_events(s)
        if len(d) < 300:
            continue
        d["ticker"] = t
        d["start"] = episode_starts(d.cond)
        frames.append(d)
    al = pd.concat(frames)
    cond = al[al.start]
    def summ(x):
        return {"n": len(x), "mean": x.fwd.mean(), "median": x.fwd.median(), "p05": x.fwd.quantile(.05),
                "p_hit": x.hit.mean(), "meanX": x.X.mean(), "p_lt5": (x.fwd < -.05).mean(),
                "p_lt10": (x.fwd < -.10).mean()}
    return {"uncond": summ(al), "cond": summ(cond), "n_tickers": al.ticker.nunique(),
            "period": (str(al.index.min().date()), str(al.index.max().date()))}


# ---------------------------------------------------------------- IO / report
def fetch_cboe(name: str, cache: Path) -> pd.Series:
    f = cache / f"{name}.csv"
    if not f.exists():
        req = urllib.request.Request(CBOE_URL.format(name), headers={"User-Agent": "Mozilla/5.0"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    d = pd.read_csv(f, parse_dates=["DATE"], date_format="%m/%d/%Y").set_index("DATE")[name]
    return pd.to_numeric(d, errors="coerce").dropna()


def fetch_yahoo(sym: str, cache: Path, start="1985-01-01", adj=True) -> pd.Series:
    import yfinance as yf
    f = cache / f"{sym.replace('^', '_')}.parquet"
    if f.exists():
        return pd.read_parquet(f)["c"]
    d = yf.download(sym, start=start, auto_adjust=adj, progress=False)
    s = d["Close"].squeeze().dropna().rename("c")
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.to_frame().to_parquet(f)
    return s


def panel_tickers(n=40):
    try:
        d = pd.read_parquet(PANEL, columns=["date", "etf", "ysym", "weight"])
        x = d[d.etf == "XIC"]
        last = x[x.date == x.date.max()].sort_values("weight", ascending=False)
        return list(last.ysym.head(n)), f"top {n} XIC weights at {pd.Timestamp(last.date.iloc[0]).date()} (panel, read-only)"
    except Exception:
        return HAND_LIST, "hand list of 40 TSX large caps"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def build_report(cache: Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    bench = fetch_yahoo(BENCH, cache, adj=False)
    L = [f"# Options proxy study (stage 0, free data) - generated {dt.date.today()}", "",
         "All numbers below are generated by `scripts/options_proxy_study.py`. No paid data.", "",
         "Cost parameters: see [`docs/cost_assumptions.md`](../cost_assumptions.md).", "",
         "## 1. Cboe option-strategy indices vs S&P 500 TR", ""]
    L += ["Sources: " + ", ".join(f"`{CBOE_URL.format(n)}`" for n in INDICES) + f"; benchmark Yahoo `{BENCH}` (price column, TR index).",
          "Index results, **no retail cost, no tax**. Monthly last-observation levels (months without an observation are dropped, not bridged - PUT is sparse before 2007, hence its low month count); Sharpe uses rf = 0; "
          "NW t = Newey-West (6 lags) t of mean monthly (index - SP500TR); alpha = intercept of index on SP500TR (monthly, annualised).",
          "BXM = ATM 1-month SPX buy-write; PUT = ATM 1-month SPX put-write; BXMD = 30-delta buy-write; "
          "**PPUT = put-protection (long SPX + long 5%-OTM puts), NOT a put-write**.", ""]
    res = {}
    for label, start in [("full common history", None), ("post-2012", "2012-01")]:
        L += [f"### {label}", "",
              "| index | window | months | CAGR | SP500TR CAGR | CAGR diff | vol | SP500TR vol | Sharpe | SP500TR Sharpe | maxDD | SP500TR maxDD | excess/yr | NW t | beta | alpha/yr | alpha t |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in INDICES:
            r = compare(fetch_cboe(n, cache), bench, start)
            res[(n, label)] = r
            i, b = r["idx"], r["bench"]
            L.append(f"| {n} | {r['start']}..{r['end']} | {i['n_months']} | {pct(i['cagr'])} | {pct(b['cagr'])} | {pct(r['cagr_diff'])} | "
                     f"{pct(i['vol'])} | {pct(b['vol'])} | {i['sharpe']:.2f} | {b['sharpe']:.2f} | {pct(i['maxdd'])} | {pct(b['maxdd'])} | "
                     f"{pct(r['excess_mean_ann'])} | {r['excess_nw_t']:.2f} | {r['beta']:.2f} | {pct(r['alpha_ann'])} | {r['alpha_t']:.2f} |")
        L.append("")

    # part 2
    xiu = fetch_yahoo("XIU.TO", cache, start="2024-01-01", adj=False)
    price = float(xiu.iloc[-1])
    bxm = res[("BXM", "post-2012")]
    prem_alpha = bxm["alpha_ann"] * 1e4
    prem_cagr = bxm["cagr_diff"] * 1e4
    L += ["## 2. Retail cost overlay: monthly covered call on XIU", "",
          f"Assumptions (parameters, not measurements): commission C${COMMISSION_BASE} + C${COMMISSION_PER_CONTRACT}/contract per order; "
          f"option half-spread C${HALF_SPREAD:.2f}/share on the sold call (the quantity ORATS/MX data would measure); "
          f"XIU last close C${price:.2f} ({xiu.index[-1].date()}); 1 contract per 100 shares, rounded to continuous contracts; "
          "12 sells-to-open per year. 'Open only' = calls expire or are assigned; 'open + close' = every call is bought back (second commission and half-spread).", "",
          "| size C$ | contracts/mo | cost bp/yr (open only) | cost bp/yr (open + close) |", "|---|---|---|---|"]
    for s in SIZES:
        L.append(f"| {s:,} | {s / (100 * price):.1f} | {covered_call_cost_bp(s, price):.0f} | {covered_call_cost_bp(s, price, close_frac=1.0):.0f} |")
    L += ["", f"Premium benchmarks (BXM, post-2012): BXM-minus-SPX CAGR = {prem_cagr:.0f} bp/yr; "
          f"CAPM alpha of BXM on SP500TR = {prem_alpha:.0f} bp/yr (t = {bxm['alpha_t']:.2f}). "
          "BXM is ATM SPX with no costs, a stand-in for the gross premium - XIU options are thinner, so the real gross premium is not measured here.", "",
          "| premium measure | bp/yr | break-even size, open only | break-even size, open + close |", "|---|---|---|---|"]
    be = {}
    for name, p in [("CAGR diff", prem_cagr), ("CAPM alpha", prem_alpha)]:
        a, b = breakeven_size(p, price), breakeven_size(p, price, close_frac=1.0)
        be[name] = (a, b)
        f = lambda v: "never (variable-cost floor >= premium)" if v is None else f"C${v:,.0f}"
        L.append(f"| {name} | {p:.0f} | {f(a)} | {f(b)} |")
    floor = covered_call_cost_bp(1e12, price)
    L += ["", f"Variable-cost floor (size -> infinity, open only): {floor:.0f} bp/yr.", ""]

    # part 3
    tickers, src = panel_tickers()
    px = {}
    for t in tickers:
        try:
            px[t] = fetch_yahoo(t, cache, start="2005-01-01")
        except Exception:
            pass
    px = {t: s for t, s in px.items() if len(s) > 400}
    ev = pool_events(px)
    u, c = ev["uncond"], ev["cond"]
    L += ["## 3. Price-only bound on short-put assignment", "",
          f"Universe: {src}; {ev['n_tickers']} of {len(tickers)} had usable Yahoo history; {ev['period'][0]}..{ev['period'][1]}. "
          "**SURVIVOR-ONLY**: current constituents with full histories, so drift is overstated and left-tail frequencies are understated; treat as a lower bound on assignment.",
          f"Event = day with drawdown from the 20-day high in [-{DD_HI:.0%}, -{DD_LO:.0%}], de-clustered to one event per {HORIZON} trading days per ticker. "
          f"Forward return = {HORIZON}-day adjusted close return. Strike: Black-Scholes {TARGET_DELTA:.0%}-delta put with sigma = trailing {VOL_WIN}-day realised vol (r=q=0), "
          "so X varies by row; 'hit' = forward return < -X (put finishes ITM = assignment).", "",
          "| sample | n | mean fwd | median fwd | 5th pct | mean X | P(ret < -X) | P(ret < -5%) | P(ret < -10%) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in [("unconditional (all days)", u), ("after 5-10% drawdown", c)]:
        L.append(f"| {name} | {s['n']:,} | {pct(s['mean'])} | {pct(s['median'])} | {pct(s['p05'])} | {pct(s['meanX'])} | "
                 f"{pct(s['p_hit'])} | {pct(s['p_lt5'])} | {pct(s['p_lt10'])} |")
    L += ["", "Strikes use realised vol; market IV is usually above RV, which puts a true 10-delta strike further OTM and lowers the hit rate shown. "
          "Events are cross-sectionally clustered in crash periods (2008, 2020), so the conditional-vs-unconditional gap is descriptive, not a significance claim.",
          "", "## 4. Verdict (rule-based on the numbers above)", ""]
    best_alpha_t = bxm["alpha_t"]
    ok_size = [s for s in SIZES if covered_call_cost_bp(s, price) < max(prem_alpha, 0)]
    L.append(f"- Index-level premium after 2012: BXM-minus-SPX CAGR {prem_cagr:.0f} bp/yr, CAPM alpha {prem_alpha:.0f} bp/yr (t {best_alpha_t:.2f}).")
    put = res[("PUT", "post-2012")]
    L.append(f"- PUT (put-write) post-2012 CAPM alpha {put['alpha_ann'] * 1e4:.0f} bp/yr (t {put['alpha_t']:.2f}).")
    L.append(f"- Retail covered-call cost at the three sizes: {', '.join(f'{covered_call_cost_bp(s, price):.0f}' for s in SIZES)} bp/yr (open only).")
    if ok_size:
        L.append(f"- Net of retail costs the alpha measure stays positive at sizes: {', '.join(f'C${s:,}' for s in ok_size)}. A thin-chain XIU spread is the open question that MX/ORATS data would answer.")
    else:
        L.append("- After retail costs the index premium does not cover costs at any tested size; buying MX/ORATS data is NOT justified on the covered-call premium alone.")
    L.append(f"- Short-put side: after a 5-10% drawdown P(assignment-type loss) = {pct(c['p_hit'])} vs {pct(u['p_hit'])} unconditional (survivor-only lower bound).")
    L += ["", "## Limits", "",
          "- Cboe indices are SPX, not XIU/TSX; no TSX option data used. Index returns are gross of any retail cost and tax.",
          "- Half-spread and commission mix are assumptions; the real XIU spread is exactly what is unmeasured.",
          "- Survivor-only equity panel; realised-vol strikes instead of market IV."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / ".cache/options_proxy"))
    ap.add_argument("--out", default="docs/research_notes/2026-10-24_options_proxy.md")
    a = ap.parse_args()
    Path(a.out).write_text(build_report(Path(a.cache)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
