#!/usr/bin/env python3
"""Backtest of a rule-based "sector momentum" ETF portfolio, in CAD, with retail costs.

The rule (a paper-traded reference implementation; line numbers in older notes are not meaningful here):

* Universe: UNIVERSE below.  Tickers with no data are dropped; VNQ has no data in the price source
  used for the published run, so it was never tradable (33 of 34 names used).
* Score: prices.iloc[-1] / prices.iloc[-126] - 1, i.e. 125 rows back (not 126).
  A ticker is eligible once both ends of that window exist (>= 126 rows of history).
* Picks: top TOP_N=5.  The signal is recomputed every trading day.
* Weights: min-variance QP on 1y of daily returns, 5% floor / 40% cap,
  equal-weight fallback if < 100 rows or solver failure.  (Called "risk parity" in the original
  script, it is a constrained minimum-variance portfolio.)
* Rebalance triggers, evaluated every day on the latest close:
    1. no positions -> initial allocation
    2. held set != picks set -> rebalance (ANY day, not only month start)
    3. any |actual - target| weight > DRIFT_THRESHOLD (positions only, cash ignored)
    4. first run of the month with day-of-month <= 5, if no rebalance yet this calendar month
  Orders: whole shares, int(equity * w / last_close); positions not in the targets are sold in
  full; other trades below MIN_TRADE_NOTIONAL are skipped.
* Execution: signal on close of day t, orders fill on day t+1 (open if a trustworthy open exists,
  otherwise that day's close).

Nothing is written outside results/; prices are read once into a local parquet cache.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = Path(os.environ.get("SECTORMOM_CACHE", HERE / ".cache"))
DEXCAUS_CSV = os.environ.get("DEXCAUS_CSV")      # optional FRED DEXCAUS export (observation_date, DEXCAUS); else CAD=X only

# ---- rule parameters --------------------------------
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    "EFA", "EEM", "VWO",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "UNG", "GDX",
    "VNQ",
    "MTUM", "QUAL", "VLUE", "SIZE", "USMV",
]
TOP_N = 5
LOOKBACK = 126
DRIFT_THRESHOLD = 0.05
MIN_TRADE_NOTIONAL = 2000.0
W_MIN, W_MAX = 0.05, 0.40
REBALANCE_DAY = 1  # monthly rebalance allowed while day-of-month <= REBALANCE_DAY + 4

# ---- backtest conventions ---------------------------------------------------------------------
# Reference book. MIN_TRADE is scaled as a constant fraction of *current* equity (2000 / 100000 = 2%),
# so it is US$2,000 on a US$100k book and grows with the book / with larger book sizes.
REFERENCE_BOOK_USD = 100_000.0
COMMISSION_CAD = 9.99          # the retail broker, per order
SPDR_SECTORS = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY"}
SPREAD_BP_SPDR = 2.0           # half-spread per side: SPDR sector ETFs and SPY
SPREAD_BP_OTHER = 5.0          # everything else
BOOKS_CAD = [100_000, 250_000, 500_000]
HEADLINE_BOOK = 250_000
MIN_ELIGIBLE_FOR_START = 10
END_DATE = "2026-10-16"        # end of the published run
PERIODS = {
    "full": (None, None),
    "pre-2010": (None, "2009-12-31"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-present": ("2020-01-01", None),
}
ENERGY = ["XLE", "USO", "UNG"]


def spread_bp(ticker: str) -> float:
    return SPREAD_BP_SPDR if (ticker in SPDR_SECTORS or ticker == "SPY") else SPREAD_BP_OTHER


# =============================================================================================
# Data
# =============================================================================================
def load_from_yahoo() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Pull adjusted closes / adjusted opens and USDCAD from Yahoo Finance (needs the `data` extra: yfinance).

    Adjusted open = raw open scaled by the same dividend factor as the close.  An open is trusted only if it
    is not a flat open=high=low=close stub.  The published numbers were produced from a stored copy of Yahoo
    data whose final months were close-only; a fresh pull will differ slightly."""
    import yfinance as yf

    raw = yf.download(UNIVERSE, period="max", auto_adjust=False, actions=False, progress=False, group_by="column")
    prices = raw["Adj Close"].dropna(how="all")
    close, op, hi, lo = (raw[c].reindex_like(prices) for c in ("Close", "Open", "High", "Low"))
    flat = (op == hi) & (hi == lo) & (lo == close)
    ok = (op > 0) & (close > 0) & ~flat
    opens = (op * prices / close).where(ok)
    meta = {"open_valid_share": float(ok.stack().mean())}

    yfx = yf.download("CAD=X", period="max", auto_adjust=False, progress=False)["Close"].squeeze().dropna().astype(float)
    fx = yfx
    if DEXCAUS_CSV:
        f = pd.read_csv(DEXCAUS_CSV)
        f["observation_date"] = pd.to_datetime(f["observation_date"])
        f = pd.to_numeric(f.set_index("observation_date")["DEXCAUS"], errors="coerce").dropna()
        overlap = f.index.intersection(yfx.index)
        overlap = overlap[overlap >= overlap.max() - pd.Timedelta(days=365)]
        meta["fx_overlap_n"] = int(len(overlap))
        meta["fx_overlap_max_abs_diff"] = float((f[overlap] - yfx[overlap]).abs().max())
        meta["fx_overlap_mean_abs_diff"] = float((f[overlap] - yfx[overlap]).abs().mean())
        meta["fred_last"] = str(f.index.max().date())
        fx = pd.concat([f, yfx[yfx.index > f.index.max()]]).sort_index()   # FRED history, extended with CAD=X
    meta["fx_last"] = str(fx.index.max().date())
    meta["tickers_missing"] = sorted(set(UNIVERSE) - set(prices.columns))
    return prices, opens, fx, meta


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    pp, op, fp, mp = CACHE / "prices.parquet", CACHE / "opens.parquet", CACHE / "fx.parquet", CACHE / "meta.json"
    if refresh or not all(p.exists() for p in (pp, op, fp, mp)):
        prices, opens, fx, meta = load_from_yahoo()
        prices.to_parquet(pp)
        opens.to_parquet(op)
        fx.to_frame("usdcad").to_parquet(fp)
        mp.write_text(json.dumps(meta, indent=2))
    return (pd.read_parquet(pp), pd.read_parquet(op), pd.read_parquet(fp)["usdcad"],
            json.loads(mp.read_text()))


class Market:
    """Dense numpy view of the panel; all indexing is positional on the trading calendar."""

    def __init__(self, prices: pd.DataFrame, opens: pd.DataFrame, fx: pd.Series):
        prices = prices.sort_index()
        self.dates = prices.index
        self.tickers = list(prices.columns)
        self.P = prices.to_numpy(float)                                   # NaN where no row
        self.O = opens.reindex_like(prices).to_numpy(float)               # NaN where open untrusted
        self.Pv = np.nan_to_num(prices.ffill().to_numpy(float), nan=0.0)  # valuation prices
        self.fx = fx.reindex(prices.index.union(fx.index)).ffill().reindex(prices.index).bfill().to_numpy(float)
        self.hs = np.array([spread_bp(t) for t in self.tickers]) / 1e4
        d = prices.index.values.astype("datetime64[D]")
        self.win_start = np.searchsorted(d, d - np.timedelta64(365, "D"), side="left")
        self.month_key = (prices.index.year * 12 + prices.index.month).to_numpy()
        self.day = prices.index.day.to_numpy()
        self._mom: dict[int, np.ndarray] = {}

    def momentum(self, lookback: int) -> np.ndarray:
        """Row i: P[i] / P[i-(lookback-1)] - 1  (original rule: iloc[-1] / iloc[-lookback] - 1)."""
        if lookback not in self._mom:
            lag = lookback - 1
            M = np.full_like(self.P, np.nan)
            num, den = self.P[lag:], self.P[:-lag]
            with np.errstate(invalid="ignore", divide="ignore"):
                M[lag:] = np.where((den > 0) & np.isfinite(num), num / den - 1, np.nan)
            self._mom[lookback] = M
        return self._mom[lookback]

    def eligible_count(self, lookback: int = LOOKBACK) -> np.ndarray:
        return np.isfinite(self.momentum(lookback)).sum(axis=1)

    def start_index(self, min_eligible: int = MIN_ELIGIBLE_FOR_START) -> int:
        """First trading day of the first calendar month whose first trading day has >= N eligible."""
        cnt = self.eligible_count()
        first = np.r_[True, self.month_key[1:] != self.month_key[:-1]]
        idx = np.nonzero(first & (cnt >= min_eligible))[0]
        return int(idx[0])


# =============================================================================================
# Weights
# =============================================================================================
def window_cov(mkt: Market, i: int, picks: tuple[int, ...]) -> np.ndarray | None:
    """Lines 302-322: last 365 calendar days, prices[top_picks].dropna(), pct_change().dropna().cov()."""
    X = mkt.P[mkt.win_start[i]:i + 1][:, list(picks)]
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 100:                                   
        return None
    R = X[1:] / X[:-1] - 1
    R = R[np.isfinite(R).all(axis=1)]
    return np.cov(R, rowvar=False).reshape(len(picks), len(picks))


def reference_qp_cvxpy(cov: np.ndarray, n: int) -> np.ndarray | None:
    """Verbatim optimisation of the original rule (OSQP via cvxpy, default settings)."""
    import cvxpy as cp
    weights = cp.Variable(n)
    portfolio_variance = cp.quad_form(weights, cov)
    objective = cp.Minimize(portfolio_variance)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0.05,
        weights <= 0.40,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != "optimal":
        return None
    optimal_weights = weights.value
    return optimal_weights / np.sum(optimal_weights)


def qp_active_set(cov: np.ndarray, lo: float = W_MIN, hi: float = W_MAX) -> np.ndarray | None:
    """Fast exact solver for min w'Cw, sum w = 1, lo <= w <= hi (small n).  Returns None if the
    KKT check fails, in which case the caller falls back to SLSQP."""
    n = cov.shape[0]
    low = np.zeros(n, bool)
    up = np.zeros(n, bool)
    for _ in range(6 * n):
        fixed = low | up
        free = ~fixed
        wfix = np.where(low, lo, np.where(up, hi, 0.0))
        w = wfix.copy()
        if not free.any():
            return None
        S = cov[np.ix_(free, free)]
        c = cov[np.ix_(free, fixed)] @ wfix[fixed] if fixed.any() else np.zeros(free.sum())
        try:
            s1 = np.linalg.solve(S, np.ones(S.shape[0]))
            sc = np.linalg.solve(S, c)
        except np.linalg.LinAlgError:
            return None
        mu = (1.0 - wfix.sum() + sc.sum()) / s1.sum()
        w[free] = mu * s1 - sc
        viol = np.where(free, np.maximum(lo - w, w - hi), -np.inf)
        j = int(np.argmax(viol))
        if viol[j] > 1e-12:                      # fix the single worst violator, then re-solve
            if w[j] < lo:
                low[j] = True
            else:
                up[j] = True
            continue
        g = 2.0 * cov @ w
        lam = g[free].mean()
        rel_l = low & (g < lam - 1e-10)
        rel_u = up & (g > lam + 1e-10)
        if not (rel_l.any() or rel_u.any()):
            return w
        low &= ~rel_l
        up &= ~rel_u
    return None


def qp_slsqp(cov: np.ndarray, lo: float = W_MIN, hi: float = W_MAX) -> np.ndarray | None:
    from scipy.optimize import minimize
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * cov @ w,
                   bounds=[(lo, hi)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return res.x / res.x.sum() if res.success else None


def solve_weights(cov: np.ndarray | None, n: int, solver: str) -> np.ndarray:
    """Fallbacks: too little history / solver failure -> equal weight."""
    eq = np.full(n, 1.0 / n)
    if cov is None:
        return eq
    try:
        if solver == "cvxpy":
            w = reference_qp_cvxpy(cov, n)
        else:
            w = qp_active_set(cov)
            if w is None:
                w = qp_slsqp(cov)
    except Exception:  # noqa: BLE001  (the original rule catches everything and falls back to equal weight)
        w = None
    return eq if w is None else w


def target_weights(mkt: Market, i: int, picks: tuple[int, ...], solver: str, cache: dict | None) -> np.ndarray:
    key = (solver, i, picks)
    if cache is not None and key in cache:
        return cache[key]
    w = solve_weights(window_cov(mkt, i, picks), len(picks), solver)
    if cache is not None:
        cache[key] = w
    return w


# =============================================================================================
# Simulation engine
# =============================================================================================
def simulate(mkt: Market, book_cad: float, *, start_idx: int, end_idx: int,
             mode: str = "momentum", lookback: int = LOOKBACK, top_n: int = TOP_N,
             weighting: str = "rp", solver: str = "fast", costs: bool = True,
             holdings_trigger: bool = True, drift_trigger: bool = True, monthly_trigger: bool = True,
             exclude: tuple[str, ...] = (), fixed_ticker: str | None = None,
             seed: int | None = None, cache: dict | None = None, record: bool = False,
             book_usd: float | None = None, signal_prices: np.ndarray | None = None,
             fill_mode: str = "next_open", min_trade_usd: float | None = None) -> dict:
    """Day-by-day share-level simulation in USD, valued in CAD with daily USDCAD.

    mode: 'momentum'          reference rule: picks recomputed daily from momentum
          'momentum_monthly'  picks refreshed only on the first trading day of a month
          'random'            random `top_n` eligible tickers, refreshed monthly (placebo)
          'all'               every eligible ticker, equal weight
          'fixed'             one ticker (buy & hold)
    fill_mode: 'next_open' (default; falls back to next close when no trustworthy open) or
               'signal_close' (diagnostic bound only: fill at the close used for sizing, not tradable).
    min_trade_usd: absolute US$ MIN_TRADE override (default None = constant fraction of equity).
    signal_prices: optional alternative price matrix used ONLY for the momentum signal
                   (look-ahead self-test hook).
    """
    K = len(mkt.tickers)
    M = None
    if mode in ("momentum", "momentum_monthly"):
        if signal_prices is None:
            M = mkt.momentum(lookback).copy()
        else:
            assert signal_prices.shape == mkt.P.shape, "signal_prices must match the market panel"
            tmp = Market.__new__(Market)
            tmp.P = signal_prices
            tmp._mom = {}
            M = Market.momentum(tmp, lookback).copy()
        for t in exclude:
            M[:, mkt.tickers.index(t)] = np.nan
    elig = mkt.momentum(lookback)
    if exclude:
        elig = elig.copy()
        for t in exclude:
            elig[:, mkt.tickers.index(t)] = np.nan
    rng = np.random.default_rng(seed)

    sh = np.zeros(K)
    fx0 = mkt.fx[start_idx]
    cash = float(book_usd if book_usd is not None else book_cad / fx0)
    n_days = end_idx - start_idx + 1
    nav = np.zeros(n_days)
    traded = np.zeros(n_days)       # USD notional traded (buys + sells) at fill price
    comm = np.zeros(n_days)         # USD commissions
    spr = np.zeros(n_days)          # USD half-spread cost
    norders = np.zeros(n_days)
    pnl = np.zeros((n_days, K)) if record else None
    commission_usd_flag = 1.0 if costs else 0.0
    spread_flag = 1.0 if costs else 0.0

    pending: tuple[int, list[tuple[int, int]]] | None = None
    last_reb_month = -1
    cur_picks: tuple[int, ...] | None = None
    last_pick_month = -1
    trig = {"initial": 0, "holdings": 0, "drift": 0, "monthly": 0}
    n_open_fills = n_close_fills = 0
    order_log: list[dict] = [] if record else None
    pick_log: list[tuple] = [] if record else None

    for i in range(start_idx, end_idx + 1):
        d = i - start_idx
        prev_close = mkt.Pv[i - 1] if i > 0 else mkt.Pv[i]
        sh_pre = sh.copy()
        # ---- 1. execute orders decided on the previous close at today's open/close ----------
        if pending is not None:
            sig_idx, orders = pending
            assert sig_idx < i, "look-ahead: signal date must precede trade date"
            for k, delta in sorted(orders, key=lambda o: o[1]):     # sells first
                px = mkt.O[i, k] if fill_mode == "next_open" else np.nan
                if fill_mode == "signal_close":
                    px = mkt.Pv[sig_idx, k]
                    n_close_fills += 1
                elif np.isfinite(px):
                    n_open_fills += 1
                else:
                    px = mkt.Pv[i, k]                                # next-close fallback
                    n_close_fills += 1
                side = 1.0 if delta > 0 else -1.0
                fill = px * (1.0 + side * mkt.hs[k] * spread_flag)
                cash -= delta * fill
                c_usd = COMMISSION_CAD / mkt.fx[i] * commission_usd_flag
                cash -= c_usd
                comm[d] += c_usd
                spr[d] += abs(delta) * px * mkt.hs[k] * spread_flag
                traded[d] += abs(delta) * fill
                norders[d] += 1
                sh[k] += delta
                if record:
                    pnl[d, k] += delta * (mkt.Pv[i, k] - fill)
                    order_log.append({"signal_date": mkt.dates[sig_idx], "trade_date": mkt.dates[i],
                                      "ticker": mkt.tickers[k], "shares": int(delta), "fill": fill})
            pending = None
        if record:
            pnl[d] += sh_pre * (mkt.Pv[i] - prev_close)
        nav_i = cash + float(sh @ mkt.Pv[i])
        nav[d] = nav_i
        if i == end_idx:
            break

        # ---- 2. decide on today's close; orders fill tomorrow -------------------------------
        held = tuple(np.nonzero(sh)[0])
        picks = None
        if mode in ("momentum", "momentum_monthly"):
            refresh = mode == "momentum" or mkt.month_key[i] != last_pick_month or cur_picks is None
            if refresh:
                row = M[i]
                valid = np.nonzero(np.isfinite(row))[0]
                if len(valid) >= top_n:
                    order = valid[np.argsort(-row[valid], kind="stable")][:top_n]
                    cur_picks = tuple(int(k) for k in order)
                else:
                    cur_picks = None
                last_pick_month = mkt.month_key[i]
            picks = cur_picks
        elif mode == "random":
            if mkt.month_key[i] != last_pick_month or cur_picks is None:
                valid = np.nonzero(np.isfinite(elig[i]))[0]
                cur_picks = tuple(int(k) for k in rng.choice(valid, top_n, replace=False)) if len(valid) >= top_n else None
                last_pick_month = mkt.month_key[i]
            picks = cur_picks
        elif mode == "all":
            valid = np.nonzero(np.isfinite(elig[i]))[0]
            picks = tuple(int(k) for k in valid) if len(valid) else None
        elif mode == "fixed":
            picks = (mkt.tickers.index(fixed_ticker),)
        if picks is None:
            continue
        if record and (not pick_log or pick_log[-1][1] != picks):
            pick_log.append((mkt.dates[i], picks))

        def targets():
            if mode in ("all", "fixed") or weighting == "ew":
                return np.full(len(picks), 1.0 / len(picks))
            return target_weights(mkt, i, tuple(picks), solver, cache)

        reason = None
        w_t = None
        if not held:
            reason = "initial"
        elif holdings_trigger and set(held) != set(picks):
            reason = "holdings"
        else:
            if drift_trigger:
                w_t = targets()
                tw = dict(zip(picks, w_t))
                mv = sh[list(held)] * mkt.Pv[i, list(held)]
                tot = mv.sum()
                if tot > 0:
                    drift = max(abs(m / tot - tw.get(k, 0.0)) for k, m in zip(held, mv))
                    if drift > DRIFT_THRESHOLD:
                        reason = "drift"
            if reason is None and monthly_trigger and mkt.day[i] <= REBALANCE_DAY + 4 \
                    and mkt.month_key[i] != last_reb_month:
                reason = "monthly"
        if reason is None:
            continue
        trig[reason] += 1
        last_reb_month = mkt.month_key[i]
        if w_t is None:
            w_t = targets()
        px = mkt.Pv[i]
        equity = nav_i
        # noise filter, scaled to book size; the passive benchmarks ('all', 'fixed') have none
        min_trade = MIN_TRADE_NOTIONAL * equity / REFERENCE_BOOK_USD if mode not in ("all", "fixed") else 0.0
        if min_trade_usd is not None and mode not in ("all", "fixed"):
            min_trade = min_trade_usd                              # absolute US$ floor (sensitivity)
        orders = []
        tgt_keys = set(picks)
        for k in held:                                              # sell everything not targeted
            if k not in tgt_keys:
                orders.append((int(k), -int(sh[k])))
        for k, w in zip(picks, w_t):
            if px[k] <= 0:
                continue
            tsh = int(equity * w / px[k])                           # int() truncation
            delta = tsh - int(sh[k])
            if abs(delta) * px[k] < min_trade or delta == 0:        
                continue
            orders.append((int(k), delta))
        if orders:
            pending = (i, orders)

    out = {"nav_usd": nav, "traded": traded, "comm": comm, "spread": spr, "norders": norders,
           "dates": mkt.dates[start_idx:end_idx + 1], "fx": mkt.fx[start_idx:end_idx + 1],
           "triggers": trig, "open_fills": n_open_fills, "close_fills": n_close_fills}
    out["nav_cad"] = pd.Series(nav * mkt.fx[start_idx:end_idx + 1], index=out["dates"])
    if record:
        out["pnl"] = pnl
        out["orders"] = pd.DataFrame(order_log)
        out["pick_log"] = pick_log
    return out


# =============================================================================================
# Statistics
# =============================================================================================
def monthly_returns(nav: pd.Series) -> pd.Series:
    m = nav.groupby([nav.index.year, nav.index.month]).last()
    m.index = nav.groupby([nav.index.year, nav.index.month]).tail(1).index
    prev = m.shift(1)
    prev.iloc[0] = nav.iloc[0]
    return m / prev - 1


def newey_west_t(x: np.ndarray, lag: int = 12) -> tuple[float, float]:
    """t-stat of the mean with Bartlett-kernel HAC variance."""
    x = np.asarray(x, float)
    n = len(x)
    if n < lag + 2:
        return float("nan"), float("nan")
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    se = math.sqrt(s / n)
    return float(x.mean()), float(x.mean() / se) if se > 0 else float("nan")


def perf(nav: pd.Series, rf: float = 0.0) -> dict:
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    vol = r.std() * math.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else float("nan")
    mdd = float((nav / nav.cummax() - 1).min())
    mr = monthly_returns(nav)
    w12 = float((1 + mr).rolling(12).apply(np.prod, raw=True).min() - 1) if len(mr) >= 12 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": mdd, "worst12m": w12,
            "cum_return": float(nav.iloc[-1] / nav.iloc[0] - 1), "years": yrs}


def slice_period(s, lo, hi):
    a = pd.Timestamp(lo) if lo else s.index[0]
    b = pd.Timestamp(hi) if hi else s.index[-1]
    return a, b


def summarise(res: dict, spy: dict, label: str, book: int) -> list[dict]:
    """One summary row per sub-period for a simulation result vs the SPY buy-and-hold result."""
    nav, bnav = res["nav_cad"], spy["nav_cad"]
    rows = []
    for pname, (lo, hi) in PERIODS.items():
        a, b = slice_period(nav, lo, hi)
        sel = (nav.index >= a) & (nav.index <= b)
        if sel.sum() < 5:
            continue
        n = nav[sel]
        p = perf(n)
        yrs = max(p["years"], 1e-9)
        usd = res["nav_usd"][sel]
        mean_nav = float(np.mean(usd))
        row = {"strategy": label, "book_cad": book, "period": pname, "start": n.index[0].date(),
               "end": n.index[-1].date(), **p,
               "turnover_per_yr": float(res["traded"][sel].sum() / 2 / mean_nav / yrs),
               "trades_per_yr": float(res["norders"][sel].sum() / yrs),
               "commission_drag_per_yr": float(res["comm"][sel].sum() / mean_nav / yrs),
               "spread_drag_per_yr": float(res["spread"][sel].sum() / mean_nav / yrs)}
        row["cost_drag_per_yr"] = row["commission_drag_per_yr"] + row["spread_drag_per_yr"]
        mr, bm = monthly_returns(n), monthly_returns(bnav[sel])
        ex = (mr - bm).dropna()
        if len(ex) < 14:
            row.update({"excess_ann": float("nan"), "nw_t": float("nan"), "n_months": len(ex)})
        else:
            mean, t = newey_west_t(ex.to_numpy())
            row.update({"excess_ann": mean * 12, "nw_t": t, "n_months": len(ex)})
        row["excess_cum_vs_spy"] = float(n.iloc[-1] / n.iloc[0] - bnav[sel].iloc[-1] / bnav[sel].iloc[0])
        rows.append(row)
    return rows


# =============================================================================================
# Orchestration
# =============================================================================================
_G: dict = {}


def _placebo_worker(seed: int) -> tuple[int, float, float]:
    mkt, s, e, book = _G["mkt"], _G["s"], _G["e"], _G["book"]
    r = simulate(mkt, book, start_idx=s, end_idx=e, mode="random", seed=seed, solver="fast",
                 costs=_G.get("costs", True))
    p = perf(r["nav_cad"])
    return seed, p["cagr"], p["sharpe"]


def placebo_gross(mkt: Market, s: int, e: int, runs: int, workers: int) -> tuple[pd.DataFrame, dict]:
    """Cost-free random-5 placebo vs cost-free reference rules.  The net placebo re-picks monthly while the
    reference rules re-pick daily, so the net percentile mixes selection skill with trading frequency;
    removing costs from both isolates selection."""
    _G.update(mkt=mkt, s=s, e=e, book=HEADLINE_BOOK, costs=False)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_placebo_worker, range(runs), chunksize=5))
    _G["costs"] = True
    d = pd.DataFrame(out, columns=["seed", "cagr", "sharpe"])
    ref = perf(simulate(mkt, HEADLINE_BOOK, start_idx=s, end_idx=e, solver="fast", costs=False)["nav_cad"])
    info = {"runs": runs, "reference_gross_cagr": ref["cagr"], "reference_gross_sharpe": ref["sharpe"],
            "pct_cagr": float((d.cagr < ref["cagr"]).mean() * 100),
            "pct_sharpe": float((d.sharpe < ref["sharpe"]).mean() * 100),
            "placebo_gross_cagr_mean": float(d.cagr.mean()), "placebo_gross_sharpe_mean": float(d.sharpe.mean())}
    return d, info


def run_all(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prices, opens, fx, meta = load_data(args.refresh_data)
    prices = prices.loc[:args.end]
    mkt = Market(prices, opens.loc[:args.end], fx)
    s = mkt.start_index()
    e = len(mkt.dates) - 1
    print(f"universe with data: {len(mkt.tickers)} (missing: {meta['tickers_missing']}); "
          f"start {mkt.dates[s].date()} end {mkt.dates[e].date()}", flush=True)

    # --- universe evolution ---------------------------------------------------------------
    cnt = pd.Series(mkt.eligible_count(), index=mkt.dates)
    ue = cnt.groupby(cnt.index.year).last().rename("eligible_at_year_end").to_frame()
    ue.to_csv(RESULTS / "universe_evolution.csv")

    cache: dict = {}
    # --- solver agreement: cvxpy (verbatim original) vs fast solver on every date of the reference path --
    ref_c = simulate(mkt, HEADLINE_BOOK, start_idx=s, end_idx=e, solver="cvxpy", cache=cache, record=True)
    ref_f = simulate(mkt, HEADLINE_BOOK, start_idx=s, end_idx=e, solver="fast", cache=cache)
    keys = [k for k in cache if k[0] == "cvxpy"]
    from_fast = []
    for (_, i, picks) in keys:
        wf = solve_weights(window_cov(mkt, i, picks), len(picks), "fast")
        from_fast.append(np.abs(wf - cache[("cvxpy", i, picks)]).max())
    agree = {"n_weight_vectors": len(keys), "max_abs_weight_diff": float(np.max(from_fast)),
             "p99_abs_weight_diff": float(np.percentile(from_fast, 99)),
             "cagr_cvxpy": perf(ref_c["nav_cad"])["cagr"], "cagr_fast": perf(ref_f["nav_cad"])["cagr"]}
    print("solver agreement", agree, flush=True)

    # --- main strategies per book ---------------------------------------------------------
    summary, monthly = [], {}
    ref_runs = {}
    for book in BOOKS_CAD:
        runs = {
            "reference_rules": simulate(mkt, book, start_idx=s, end_idx=e, solver="cvxpy", cache=cache),
            "spy_buy_hold": simulate(mkt, book, start_idx=s, end_idx=e, mode="fixed", fixed_ticker="SPY",
                                     holdings_trigger=False, drift_trigger=False, monthly_trigger=False),
            "ew_all_eligible_monthly": simulate(mkt, book, start_idx=s, end_idx=e, mode="all",
                                                holdings_trigger=False, drift_trigger=False),
            "ew_top5_same_picks": simulate(mkt, book, start_idx=s, end_idx=e, weighting="ew", cache=cache),
        }
        if book == HEADLINE_BOOK:
            runs["reference_rules_gross_of_costs"] = simulate(mkt, book, start_idx=s, end_idx=e, solver="cvxpy",
                                                         cache=cache, costs=False)
            runs["ew_all_eligible_gross"] = simulate(mkt, book, start_idx=s, end_idx=e, mode="all", costs=False,
                                                     holdings_trigger=False, drift_trigger=False)
            runs["ew_top5_gross"] = simulate(mkt, book, start_idx=s, end_idx=e, weighting="ew", costs=False,
                                             cache=cache)
            runs["variant_monthly_picks_only"] = simulate(mkt, book, start_idx=s, end_idx=e, mode="momentum_monthly",
                                                          solver="fast", cache=cache)
            runs["diag_ex_energy_XLE_USO_UNG"] = simulate(mkt, book, start_idx=s, end_idx=e, exclude=tuple(ENERGY),
                                                          solver="fast", cache=cache)
        spy = runs["spy_buy_hold"]
        for lab, r in runs.items():
            summary += summarise(r, spy, lab, book)
        monthly[book] = pd.DataFrame({lab: monthly_returns(r["nav_cad"]) for lab, r in runs.items()
                                      if not lab.startswith(("diag", "variant")) and "gross" not in lab})
        monthly[book].index = monthly[book].index.to_period("M").to_timestamp("M")
        monthly[book].to_csv(RESULTS / f"monthly_returns_C{book // 1000}k.csv", index_label="month_end")
        ref_runs[book] = runs
        print(f"book C${book:,} done", flush=True)
    summ = pd.DataFrame(summary)
    summ.to_csv(RESULTS / "summary.csv", index=False)

    # --- robustness grid (C$250k): lookback x top-N x weighting -----------------------------
    grid = []
    spy250 = ref_runs[HEADLINE_BOOK]["spy_buy_hold"]
    for lb in (63, 126, 252):
        for n in (3, 5, 8):
            for wt in ("rp", "ew"):
                r = simulate(mkt, HEADLINE_BOOK, start_idx=s, end_idx=e, lookback=lb, top_n=n,
                             weighting=wt, solver="fast", cache=cache)
                row = [x for x in summarise(r, spy250, "grid", HEADLINE_BOOK) if x["period"] == "full"][0]
                row.update({"lookback": lb, "top_n": n, "weighting": wt,
                            "is_reference_rule": (lb == 126 and n == 5 and wt == "rp")})
                grid.append(row)
    pd.DataFrame(grid)[["lookback", "top_n", "weighting", "is_reference_rule", "cagr", "vol", "sharpe", "maxdd",
                        "worst12m", "turnover_per_yr", "trades_per_yr", "cost_drag_per_yr",
                        "excess_ann", "nw_t"]].to_csv(RESULTS / "robustness_grid.csv", index=False)
    print("grid done", flush=True)

    # --- placebo ---------------------------------------------------------------------------
    plc = None
    if args.placebo_runs > 0:
        _G.update(mkt=mkt, s=s, e=e, book=HEADLINE_BOOK)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(_placebo_worker, range(args.placebo_runs), chunksize=5))
        plc = pd.DataFrame(out, columns=["seed", "cagr", "sharpe"])
        plc.to_csv(RESULTS / "placebo_distribution.csv", index=False)
        print("placebo done", flush=True)
        gd, ginfo = placebo_gross(mkt, s, e, args.placebo_runs, args.workers)
        gd.to_csv(RESULTS / "placebo_distribution_gross.csv", index=False)
        (RESULTS / "placebo_gross.json").write_text(json.dumps(ginfo, indent=2))
        print("gross placebo", ginfo, flush=True)


    (RESULTS / "solver_agreement.json").write_text(json.dumps(agree, indent=2))
    (RESULTS / "run_meta.json").write_text(json.dumps({
        **meta, "start": str(mkt.dates[s].date()), "end": str(mkt.dates[e].date()),
        "open_fills_headline": ref_runs[HEADLINE_BOOK]["reference_rules"]["open_fills"],
        "close_fills_headline": ref_runs[HEADLINE_BOOK]["reference_rules"]["close_fills"],
        "triggers_headline": ref_runs[HEADLINE_BOOK]["reference_rules"]["triggers"],
        "tickers_in_universe": mkt.tickers}, indent=2, default=str))
    write_tables(summ, plc)


def min_trade_sensitivity(mkt: Market, s: int, e: int) -> pd.DataFrame:
    """Reference rules, exact-QP solver, MIN_TRADE variants: scaled 2% of equity (headline) vs absolute US$ floors."""
    spy = simulate(mkt, HEADLINE_BOOK, start_idx=s, end_idx=e, mode="fixed", fixed_ticker="SPY",
                   holdings_trigger=False, drift_trigger=False, monthly_trigger=False)
    cache: dict = {}
    rows = []
    for book in (HEADLINE_BOOK, 136_000):                  # 136k ~ a US$100k book
        for lab, mt in (("scaled 2% of equity (headline)", None), ("absolute US$2,000", 2000.0),
                        ("absolute US$5,000", 5000.0), ("no filter", 0.0)):
            r = simulate(mkt, book, start_idx=s, end_idx=e, solver="fast", cache=cache, min_trade_usd=mt)
            x = [y for y in summarise(r, spy, lab, book) if y["period"] == "full"][0]
            rows.append({"book_cad": book, "min_trade": lab, **{k: x[k] for k in (
                "cagr", "sharpe", "trades_per_yr", "turnover_per_yr", "cost_drag_per_yr", "excess_ann", "nw_t")}})
    return pd.DataFrame(rows)


def verdict(summ: pd.DataFrame, plc: pd.DataFrame | None) -> tuple[str, dict]:
    """Pre-registered rule (fixed before looking at results), headline = exact reference rules, C$250k, full period:
    EDGE        : NW t(excess vs SPY) >= 2.0  AND  placebo percentile >= 95 on CAGR and Sharpe
                  AND positive excess vs SPY in >= 2 of the 3 historical sub-periods.
    NO EDGE     : NW t < 1.0  AND  placebo percentile < 90 on CAGR or Sharpe.
    INCONCLUSIVE: anything else."""
    h = summ[(summ.strategy == "reference_rules") & (summ.book_cad == HEADLINE_BOOK)].set_index("period")
    spy = summ[(summ.strategy == "spy_buy_hold") & (summ.book_cad == HEADLINE_BOOK)].set_index("period")
    full = h.loc["full"]
    subs_pos = sum(h.loc[p, "cagr"] > spy.loc[p, "cagr"] for p in ("pre-2010", "2010-2019", "2020-present"))
    info = {"nw_t": float(full.nw_t), "excess_ann": float(full.excess_ann), "subperiods_beating_spy": int(subs_pos)}
    if plc is not None:
        info["pct_cagr"] = float((plc.cagr < full.cagr).mean() * 100)
        info["pct_sharpe"] = float((plc.sharpe < full.sharpe).mean() * 100)
        pc, ps = info["pct_cagr"], info["pct_sharpe"]
        if full.nw_t >= 2.0 and pc >= 95 and ps >= 95 and subs_pos >= 2:
            return "EDGE", info
        if full.nw_t < 1.0 and (pc < 90 or ps < 90):
            return "NO EDGE", info
    return "INCONCLUSIVE", info


def write_tables(summ, plc) -> None:
    def pct(x):
        return f"{x * 100:.2f}%"
    lines = []
    h = summ[(summ.book_cad == HEADLINE_BOOK) & (summ.period == "full") & (~summ.strategy.str.startswith(("diag", "variant")))]
    lines.append("## Headline table (full period, C$250k)\n")
    lines.append("| strategy | CAGR | vol | Sharpe | maxDD | worst 12m | turnover/yr | trades/yr | cost drag/yr | excess vs SPY (ann.) | NW t |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in h.iterrows():
        lines.append(f"| {r.strategy} | {pct(r.cagr)} | {pct(r.vol)} | {r.sharpe:.2f} | {pct(r.maxdd)} | {pct(r.worst12m)} | "
                     f"{r.turnover_per_yr:.2f}x | {r.trades_per_yr:.0f} | {pct(r.cost_drag_per_yr)} | "
                     f"{pct(r.excess_ann) if pd.notna(r.excess_ann) else 'n/a'} | {r.nw_t:.2f} |")
    v, info = verdict(summ, plc)
    lines.append(f"\nVERDICT (pre-registered rule): **{v}**  {json.dumps(info)}\n")
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh-data", action="store_true", help="re-download prices from Yahoo Finance")
    ap.add_argument("--placebo-runs", type=int, default=1000)
    ap.add_argument("--min-trade-sensitivity", action="store_true",
                    help="only (re)compute the MIN_TRADE_NOTIONAL sensitivity table")
    ap.add_argument("--gross-placebo-only", action="store_true",
                    help="only (re)compute the cost-free placebo comparison")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--end", default=END_DATE)
    args = ap.parse_args()
    if args.min_trade_sensitivity:
        prices, opens, fx, _ = load_data(False)
        mkt = Market(prices.loc[:args.end], opens.loc[:args.end], fx)
        df = min_trade_sensitivity(mkt, mkt.start_index(), len(mkt.dates) - 1)
        df.to_csv(RESULTS / "min_trade_sensitivity.csv", index=False)
        print(df.to_string())
        return
    if args.gross_placebo_only:
        prices, opens, fx, _ = load_data(False)
        mkt = Market(prices.loc[:args.end], opens.loc[:args.end], fx)
        gd, ginfo = placebo_gross(mkt, mkt.start_index(), len(mkt.dates) - 1, args.placebo_runs, args.workers)
        RESULTS.mkdir(parents=True, exist_ok=True)
        gd.to_csv(RESULTS / "placebo_distribution_gross.csv", index=False)
        (RESULTS / "placebo_gross.json").write_text(json.dumps(ginfo, indent=2))
        print(json.dumps(ginfo, indent=2))
        return
    run_all(args)


if __name__ == "__main__":
    main()
