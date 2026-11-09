"""Run report: one markdown file and four kinds of PNG from a `RunOutput` .

Everything shown is computed from the run objects. Every number in the markdown goes through one
formatter (`_Fmt`, used as `_f`) that records the string it printed, so a test can scan the text and
check that no literal slipped in. `periods(cfg)` is the one place that decides what the report may show:
a locked run never displays a holdout-era number.

Helpers the CLI calls to fill a `MarketOutput` live here too (`spread_stats`, `survivorship_counts`,
`benchmark_gap`, `delisting_effect`, `screen_summary`, `ic_ci`, `years_to_significance`).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as _st  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .costs import CostModel, apply_costs  # noqa: E402
from .data import universe as _universe  # noqa: E402
from .evaluate import AlphaResult, RunConfig, censor_holdout  # noqa: E402
from .formulas import get_alpha  # noqa: E402
from .lag import forward_returns  # noqa: E402
from .panel import Panel  # noqa: E402

TRADING_DAYS = 252
ALPHA = 0.05   # the family-wise level behind evaluate.multiple_testing's Sidak bar

# Palette (dataviz reference palette, light surface).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, BAND = "#e1e0d9", "#c3c2b7", "#2a78d6", "#f0efec"
DIVERGING = ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"]


@dataclass
class MarketOutput:
    market: str                         # "us" | "ca"
    universe_source: str                # resolved source name actually loaded, e.g. "liquidity_top500_us@20261019"
    results: dict[str, AlphaResult]     # alpha_id -> result, every alpha that evaluated
    failures: dict[str, str]            # alpha_id -> "ExcType: message"; still counted as attempted
    multiple_testing: dict              # evaluate.multiple_testing(...) on IS, primary horizon
    ic_corr: pd.DataFrame               # evaluate.ic_correlation(IS, primary horizon)
    member_counts: pd.Series            # universe members per row, dates >= cfg.start
    survivorship: dict                  # survivorship_counts(...)
    # benchmark_gap(...) wrapped as {"name": str, "weighting": "equal"|"cap", "table": DataFrame}; {} if no data
    benchmark: dict
    screen: dict                        # screen_summary(...) output, {} if not computed
    delisting_changed: int              # delisting_effect(...) output
    cap_series: pd.Series | None = None  # H2 capped universe: K_m per month (Period/Timestamp index), else None


@dataclass
class RunOutput:
    run_id: str
    git_sha: str
    created_at: str                     # ISO-8601 UTC
    cfg: RunConfig
    costs: CostModel
    markets: dict[str, MarketOutput]
    fingerprints: dict                  # market -> cache.db_fingerprint dict
    prereg_sha256: str | None = None
    paired: pd.DataFrame | None = None  # columns market, alpha_id, mean_diff, t_nw, n_days
    baseline_run_id: str | None = None


# ---------------------------------------------------------------------- helpers (also used by the CLI)
def periods(cfg: RunConfig) -> list[tuple[str, str, str]]:
    """[("is", start, day before holdout_start)] plus the cfg.eras that END before holdout_start; with
    cfg.unlock_holdout also the eras on/after it. The report shows nothing outside this list."""
    hs = pd.Timestamp(cfg.holdout_start)
    out = [("is", cfg.start, (hs - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))]
    return out + [(n, lo, hi) for n, lo, hi in cfg.eras if cfg.unlock_holdout or pd.Timestamp(hi) < hs]


def ic_ci(summary: dict) -> tuple[float, float]:
    """95% CI of the mean IC from its Newey-West t: se = |mean_ic / t_nw|, mean +- 1.96 se. NaN if t is 0/NaN."""
    m, t = summary["mean_ic"], summary["t_nw"]
    if not (np.isfinite(m) and np.isfinite(t)) or t == 0:
        return float("nan"), float("nan")
    se = abs(m / t)
    return m - 1.96 * se, m + 1.96 * se


def years_to_significance(ic_ir: float, horizon: int) -> float:
    """Years of data for the mean IC to reach |t| = 1.96 at the observed IC-IR: (1.96 / (ic_ir sqrt(252/h)))^2.

    h-day forward ICs overlap, so a year holds about 252/h independent observations; the t of the mean
    grows as ic_ir * sqrt(n). NaN if ic_ir is 0/NaN."""
    if not np.isfinite(ic_ir) or ic_ir == 0:
        return float("nan")
    return float((1.96 / (ic_ir * np.sqrt(TRADING_DAYS / horizon))) ** 2)


def spread_stats(result: AlphaResult, market: str, costs: CostModel, cfg: RunConfig) -> dict:
    """In-sample (dates < holdout_start, finite) long-short stats: gross_ann and net_ann (mean daily * 252),
    turnover_mean and turnover_gt50 (share of IS rebalances with one-way turnover > 0.5)."""
    hs = pd.Timestamp(cfg.holdout_start)
    gross = result.deciles["long_short"]
    net = apply_costs(gross, result.trades, market, costs)
    ok = (gross.index < hs) & np.isfinite(gross.to_numpy()) & np.isfinite(net.to_numpy())
    t = result.turnover.loc[result.turnover.index < hs].dropna()
    return {"gross_ann": float(gross[ok].mean() * TRADING_DAYS) if ok.any() else float("nan"),
            "net_ann": float(net[ok].mean() * TRADING_DAYS) if ok.any() else float("nan"),
            "turnover_mean": float(t.mean()) if len(t) else float("nan"),
            "turnover_gt50": float((t > 0.5).mean()) if len(t) else float("nan")}


def survivorship_counts(conn, pool: pd.DataFrame, market: str) -> dict:
    """Pool coverage from `universe.eligible_symbols` (read-only): pool, with_data (= pool - no_data), no_data,
    reused_ticker, not_equity, short_history, eligible; US also current, change_only, change_only_with_data
    (change-only, has rows, not a reused ticker) and coverage = change_only_with_data / change_only; CA has
    change_history False. `snapshot` and `since` = pool.attrs.get(...)."""
    _, info = _universe.eligible_symbols(conn, pool, market)
    p = pool[pool["market"] == market.upper()].drop_duplicates("symbol")
    out = {"pool": int(info["pool"]), "no_data": len(info["no_data"]), "reused_ticker": len(info["reused_ticker"]),
           "not_equity": len(info["not_equity"]), "short_history": len(info["short_history"]),
           "eligible": int(info["eligible"])}
    out["with_data"] = out["pool"] - out["no_data"]
    if market.lower() == "us":
        co = set(p.loc[p["source"] == "sp500_changes", "symbol"])
        good = co - set(info["no_data"]) - set(info["reused_ticker"])
        out.update(current=int((p["source"] == "sp500_current").sum()), change_only=len(co),
                   change_only_with_data=len(good), coverage=len(good) / len(co) if co else float("nan"))
    else:
        out["change_history"] = False
    out["snapshot"] = pool.attrs.get("snapshot")
    out["since"] = pool.attrs.get("since")
    return out


def _r1_censored(panel: Panel, cfg: RunConfig) -> np.ndarray:
    """Censored 1-day forward returns (T, N) via the one lag function."""
    r = censor_holdout(forward_returns(panel, 1, execution="close", delisting=cfg.delisting), cfg, horizon=1)
    return r.to_numpy(dtype=float, copy=True)


def benchmark_gap(panel: Panel, bench_adj_close: pd.Series, cfg: RunConfig, weighting: str) -> pd.DataFrame:
    """Universe return minus benchmark return, annualised, per period of `periods(cfg)`.

    Daily universe return = mean (weighting "equal") or `cap`-weighted mean (finite cap > 0) of the censored
    1-day forward returns over in-universe names; the benchmark goes through the same `forward_returns` and
    `censor_holdout` as a one-symbol panel. Rows are dates >= cfg.start, each calendar month (by signal date)
    is compounded over rows where both are finite, and per period annualised = prod(1+m)^(12/n_months) - 1.
    Columns [universe, benchmark, gap, months], index = period name."""
    if weighting not in ("equal", "cap"):
        raise ValueError("weighting must be 'equal' or 'cap'")
    R = _r1_censored(panel, cfg)
    ok = np.isfinite(R) & (np.ones(R.shape, bool) if panel.universe is None else np.asarray(panel.universe, bool))
    if weighting == "cap":
        cap = panel.get("cap")
        w = np.where(np.isfinite(cap) & (cap > 0), cap, 0.0)
        ok &= w > 0
    else:
        w = np.ones(R.shape)
    wk = np.where(ok, w, 0.0)
    with np.errstate(all="ignore"):
        ru = np.where(wk.sum(axis=1) > 0, (np.where(ok, R, 0.0) * wk).sum(axis=1) / wk.sum(axis=1), np.nan)
    b = bench_adj_close.reindex(panel.dates).to_numpy(dtype=float, copy=True)[:, None]
    bp = Panel(dates=panel.dates, symbols=pd.Index(["BENCH"]), fields={"adj_close": b, "close": b})
    rb = _r1_censored(bp, cfg)[:, 0]
    df = pd.DataFrame({"u": ru, "b": rb}, index=panel.dates)
    df = df.loc[df.index >= pd.Timestamp(cfg.start)].dropna()
    rows = {}
    for name, lo, hi in periods(cfg):
        sub = df.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        monthly = (1.0 + sub).groupby(sub.index.to_period("M")).prod() - 1.0
        n = len(monthly)
        ann = (lambda s: float((1.0 + s).prod() ** (12.0 / n) - 1.0)) if n else (lambda s: float("nan"))
        u, bm = ann(monthly["u"]), ann(monthly["b"])
        rows[name] = {"universe": u, "benchmark": bm, "gap": u - bm, "months": n}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["universe", "benchmark", "gap", "months"])


def delisting_effect(panel: Panel, cfg: RunConfig) -> int:
    """In-universe, in-sample (start <= date < holdout_start) cells of the primary-horizon forward return that
    differ (NaN vs finite counts) between delisting=cfg.delisting and delisting="none", both censored."""
    h = cfg.primary_horizon

    def fwd(mode):
        r = forward_returns(panel, h, execution=cfg.execution, delisting=mode)
        return censor_holdout(r, cfg, horizon=h).to_numpy(dtype=float, copy=True)

    a, b = fwd(cfg.delisting), fwd("none")
    differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    rows = ((panel.dates >= pd.Timestamp(cfg.start)) & (panel.dates < pd.Timestamp(cfg.holdout_start)))[:, None]
    uni = np.ones(a.shape, bool) if panel.universe is None else np.asarray(panel.universe, bool)
    return int((differ & uni & rows).sum())


def screen_summary(panel_raw: Panel, market: str, eligible_symbols: list[str],
                   start: str | None = None, end: str | None = None) -> dict:
    """What the spread screen removed. Unscreened = `liquidity_universe` with the market's DEFAULTS and no
    spread bar over `eligible_symbols`; screened = `panel_raw.universe` (the stored source as loaded). Per year,
    over rows >= the first universe row: mean_daily_removed (#unscreened & ~screened), symbols_removed
    (distinct symbols ever removed that year) and members_mean (mean daily screened count). Only rows with
    start <= date < end enter (None = unbounded); the CLI passes cfg.start and, when locked, holdout_start."""
    d = _universe.DEFAULTS[market]
    elig = np.isin(np.asarray(panel_raw.symbols, dtype=str), np.asarray(eligible_symbols, dtype=str))
    un = _universe.liquidity_universe(panel_raw, d["top_n"], d["min_price"], eligible=elig, max_spread=None)
    sc = np.asarray(panel_raw.universe, dtype=bool)
    first = int(np.argmax(sc.any(axis=1))) if sc.any() else len(sc)
    dates = panel_raw.dates
    keep = np.arange(len(dates)) >= first
    if start is not None:
        keep &= np.asarray(dates >= pd.Timestamp(start))
    if end is not None:
        keep &= np.asarray(dates < pd.Timestamp(end))
    rm = (un & ~sc)[keep]
    sck = sc[keep]
    years = dates[keep].year
    by_year = {int(y): {"mean_daily_removed": float(rm[years == y].sum(axis=1).mean()),
                        "symbols_removed": int(rm[years == y].any(axis=0).sum()),
                        "members_mean": float(sck[years == y].sum(axis=1).mean())}
               for y in np.unique(years)}
    return {"estimator": "abdi_ranaldo_2017", "max_spread": d["max_spread"], "spread_window": d["spread_window"],
            "by_year": by_year}


# ---------------------------------------------------------------------- number formatter
class _Fmt:
    """The one number formatter. `f(x, spec)` returns the string and records it in `.seen`; NaN/inf/None
    print "n/a" (not recorded). spec: "d" integer, "pct1"/"pct2" percent with 1/2 decimals, else a format spec."""

    def __init__(self):
        self.seen: set[str] = set()

    def __call__(self, x, spec: str) -> str:
        if x is None or not np.isfinite(x):
            return "n/a"
        if spec == "d":
            s = str(int(round(float(x))))
        elif spec in ("pct1", "pct2"):
            s = f"{float(x) * 100:.{spec[-1]}f}%"
        else:
            s = format(float(x), spec)
        self.seen.add(s)
        return s


_f = _Fmt   # public-ish alias: _f() builds a recording formatter


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
            + ["| " + " | ".join(r) + " |" for r in rows] + [""])


def _visible(index: pd.Index, cfg: RunConfig) -> np.ndarray:
    """Bool mask of the dates the report may show: >= start, and < holdout_start unless unlocked."""
    ok = np.asarray(index >= pd.Timestamp(cfg.start))
    return ok if cfg.unlock_holdout else ok & np.asarray(index < pd.Timestamp(cfg.holdout_start))


def _year_visible(year: int, cfg: RunConfig) -> bool:
    """True if the calendar year has at least one day the report may show (see `_visible`)."""
    lo = max(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(cfg.start))
    hi = pd.Timestamp(year=year, month=12, day=31)
    if not cfg.unlock_holdout:
        hi = min(hi, pd.Timestamp(cfg.holdout_start) - pd.Timedelta(days=1))
    return bool(lo <= hi)


def _month_visible(month_start: pd.DatetimeIndex, cfg: RunConfig) -> np.ndarray:
    """Bool mask of months lying wholly in the visible range: month start >= first day of cfg.start's month and,
    when locked, month end < holdout_start."""
    ok = np.asarray(month_start >= pd.Timestamp(cfg.start).to_period("M").to_timestamp())
    if not cfg.unlock_holdout:
        ok &= np.asarray((month_start + pd.offsets.MonthEnd(0)) < pd.Timestamp(cfg.holdout_start))
    return ok


def _spec(alpha_id: str):
    m = re.fullmatch(r"alpha(\d{3})", alpha_id)
    try:
        return get_alpha(int(m.group(1))) if m else None
    except KeyError:
        return None


def _flags(alpha_id: str) -> str:
    spec = _spec(alpha_id)
    out = []
    if spec is not None:
        out += ["industry"] * spec.needs_industry + ["pit=false"] * (spec.id == 56) + ["delay0"] * (spec.delay == 0)
    return ", ".join(out)


def _generic(f: _Fmt, v) -> str:
    if isinstance(v, bool) or isinstance(v, str) or v is None:
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f(v, "d")
    if isinstance(v, (float, np.floating)):
        return f(v, ".8g")
    if isinstance(v, dict):
        return ", ".join(f"{k} {_generic(f, x)}" for k, x in v.items())
    if isinstance(v, (tuple, list)):
        return ", ".join(_generic(f, x) for x in v)
    return str(v)


def _fields(f: _Fmt, obj, names: list[str] | None = None) -> str:
    return "; ".join(f"{fl.name} = {_generic(f, getattr(obj, fl.name))}" for fl in dataclasses.fields(obj)
                     if names is None or fl.name in names)


# ---------------------------------------------------------------------- markdown
def render_markdown(run: RunOutput, figures: dict[str, str]) -> tuple[str, set[str]]:
    """(markdown, every number string formatted). `figures` maps a caption to a PNG file name."""
    f = _Fmt()
    cfg, h = run.cfg, run.cfg.primary_horizon
    pers = periods(cfg)
    shown = {p[0] for p in pers}
    L: list[str] = [f"# alpha101 run {run.run_id}", ""]

    # 1. survivorship
    since = next((m.survivorship["since"] for m in run.markets.values() if m.survivorship.get("since")), None)
    since_txt = f" since {f(pd.Timestamp(since).year, 'd')}" if since else ""
    L += ["> **Survivorship (read first).** The candidate pools are today's index constituents: the current S&P 500 "
          f"plus the adds and removes of the Wikipedia \"Selected changes\" table{since_txt}, "
          "and today's S&P/TSX "
          "Composite with no change history. Names that were delisted and left no trace in either list are absent."]
    for mk, m in run.markets.items():
        s = m.survivorship
        if not s:
            L += [f"> - {mk.upper()}: pool counts unavailable (no saved pool snapshot was given to the run)."]
            continue
        if "change_only" in s:
            L += [f"> - {mk.upper()}: pool {f(s['pool'], 'd')} tickers, of which current {f(s['current'], 'd')} and "
                  f"change-only {f(s['change_only'], 'd')}; {f(s['with_data'], 'd')} with data, "
                  f"{f(s['no_data'], 'd')} no data, {f(s['reused_ticker'], 'd')} reused tickers excluded; "
                  f"change-only coverage {f(s['change_only_with_data'], 'd')} of {f(s['change_only'], 'd')} "
                  f"({f(s['coverage'], 'pct1')})."]
        else:
            L += [f"> - {mk.upper()}: pool {f(s['pool'], 'd')} tickers (today's constituents, no change history); "
                  f"{f(s['with_data'], 'd')} with data, {f(s['no_data'], 'd')} no data, "
                  f"{f(s['reused_ticker'], 'd')} reused tickers excluded."]
        if s.get("snapshot"):
            L += [f"> - {mk.upper()} snapshot: " + ", ".join(f"`{x}`" for x in s["snapshot"]) + "."]
    L += ["> - IC is conditioned on survivors, so the bias is expected to be upward on alpha, not symmetric "
          "(research_plan.md A5).",
          "> - TSX results are indicative only until a point-in-time TSX constituent history is used.",
          "> - Delisting convention `" + cfg.delisting + "`: "
          + ", ".join(f"{mk.upper()} {f(m.delisting_changed, 'd')}" for mk, m in run.markets.items())
          + " in-sample forward-return cells changed against no delisting exit (research_plan.md A2).", ""]

    # 2. run metadata
    sha = run.prereg_sha256 or "none"
    L += ["## Run metadata", "", f"- run_id: `{run.run_id}`", f"- git SHA: `{run.git_sha}`",
          f"- created_at: {run.created_at}",
          "- holdout: " + ("UNLOCKED" if cfg.unlock_holdout else "locked") + f" (holdout_start {cfg.holdout_start})",
          f"- pre-registration SHA-256: {sha}",
          f"- RunConfig: horizons = {_generic(f, cfg.horizons)}; primary_horizon = {f(h, 'd')}; start = {cfg.start}; "
          f"basis = {cfg.basis}; vwap_proxy = {cfg.vwap_proxy}; execution = {cfg.execution}; "
          f"delisting = {cfg.delisting}; options: {_fields(f, cfg.options)}; min_names: {_generic(f, cfg.min_names)}; "
          f"n_bins = {f(cfg.n_bins, 'd')}",
          "- periods shown: " + ", ".join(p[0] for p in pers),
          f"- CostModel: {_fields(f, run.costs)}"]
    for mk, m in run.markets.items():
        fp = run.fingerprints.get(mk, {})
        L += [f"- {mk.upper()}: universe source `{m.universe_source}`; fingerprint rows {_generic(f, fp.get('rows'))}, "
              f"symbols {_generic(f, fp.get('symbols'))}, max_fetched_at {fp.get('max_fetched_at')}"]
    L += [""]

    # 3. universe
    L += ["## Universe", ""]
    for mk, m in run.markets.items():
        L += [f"### {mk.upper()}", ""]
        mc = m.member_counts
        mc = mc[_visible(mc.index, cfg)] if len(mc) else mc
        if len(mc):
            g = mc.groupby(mc.index.year)
            L += _table(["year", "min members", "mean members", "max members"],
                        [[f(y, "d"), f(g.min()[y], "d"), f(g.mean()[y], ".1f"), f(g.max()[y], "d")]
                         for y in g.min().index])
        sc = m.screen
        if sc:
            L += [f"Spread screen: estimator {sc['estimator']}, bar {f(sc['max_spread'], 'pct1')}, trailing window "
                  f"{f(sc['spread_window'], 'd')} days. The yearly numbers were computed by screen_summary over "
                  "[start, end) only: from the run start, and before holdout_start in a locked run.", ""]
            L += _table(["year", "mean daily removed", "symbols removed", "mean daily members"],
                        [[f(y, "d"), f(v["mean_daily_removed"], ".2f"), f(v["symbols_removed"], "d"),
                          f(v["members_mean"], ".1f")]
                         for y, v in sorted(sc["by_year"].items()) if _year_visible(y, cfg)])
        else:
            L += ["No spread screen summary for this universe.", ""]
        if m.cap_series is not None and len(m.cap_series):
            ci = m.cap_series.index
            ms = ci.to_timestamp() if isinstance(ci, pd.PeriodIndex) else pd.DatetimeIndex(ci)   # month start
            keep = _month_visible(ms, cfg)
            yrs = np.asarray(ms.year)
            ks = m.cap_series[keep]
            g = ks.groupby(yrs[keep])
            L += ["Capped universe (K_m = names kept per month, set to the CA universe size of that month); the H2 "
                  "headline compares this universe to CA (research_plan.md A1).", ""]
            L += _table(["year", "min K_m", "max K_m"], [[f(y, "d"), f(g.min()[y], "d"), f(g.max()[y], "d")]
                                                         for y in g.min().index])
        bm = m.benchmark
        if bm and len(bm.get("table", [])):
            t = bm["table"]
            t = t.loc[[i for i in t.index if i in shown]]
            note = " Cap-weighted with today's share counts, indicative." if bm.get("weighting") == "cap" else ""
            L += [f"Benchmark {bm.get('name', 'n/a')}, {bm.get('weighting', 'equal')}-weighted universe, "
                  f"annualised by period.{note}", ""]
            L += _table(["period", "universe", "benchmark", "gap", "months"],
                        [[i, f(r["universe"], "pct2"), f(r["benchmark"], "pct2"), f(r["gap"], "pct2"),
                          f(r["months"], "d")] for i, r in t.iterrows()])
        else:
            L += ["No benchmark data.", ""]
    L += ["The spread estimator measures the high-low range, not the bid-ask half-spread; it does not validate the "
          "cost model's half-spread of " + ", ".join(f"{k.upper()} {f(v, 'g')} bps"
                                                      for k, v in run.costs.half_spread_bps.items()) + ".", ""]

    # 4. multiple testing
    L += ["## Multiple testing", ""]
    for mk, m in run.markets.items():
        mt = m.multiple_testing
        n_eff = mt["n_eff"]
        n_tests = len(m.results) + len(m.failures)
        p_raw = 2 * (1 - float(_st.norm.cdf(mt["thresholds"]["raw"])))
        p_sidak = 1 - (1 - ALPHA) ** (1 / max(n_eff, 1.0))
        L += [f"### {mk.upper()}", "",
              f"N_eff = {f(n_eff, '.1f')} effective tests out of {f(n_tests, 'd')} attempted alphas "
              "(failed alphas count: they are tests).", ""]
        L += _table(["bar", "|t| threshold", "alphas passing"],
                    [[k, f(v, ".2f"), f(mt["passing"][k], "d")] for k, v in mt["thresholds"].items()])
        L += [f"Expected false positives, N_eff x p: {f(n_eff * p_raw, '.1f')} at the raw bar (p = {f(p_raw, '.2f')}) "
              f"and {f(n_eff * p_sidak, '.2f')} at the Sidak bar (p = {f(p_sidak, '.4f')}).", ""]

    # 5. per-alpha results
    L += ["## Per-alpha results", ""]
    for mk, m in run.markets.items():
        L += [f"### {mk.upper()}", "", f"Primary horizon h = {f(h, 'd')}, in-sample.", ""]
        ids = sorted(m.results)
        rows = []
        for a in ids:
            r = m.results[a]
            s = r.summary[h]["is"]
            lo, hi = ic_ci(s)
            sp = spread_stats(r, mk, run.costs, cfg)
            rows.append([a, f(s["mean_ic"], ".4f"), f(s["ic_ir"], ".4f"), f(s["t_nw"], ".2f"), f(lo, ".4f"),
                         f(hi, ".4f"), f(s["hit_rate"], "pct1"), f(s["n_days"], "d"), f(sp["turnover_mean"], ".3f"),
                         f(sp["turnover_gt50"], "pct1"), f(sp["gross_ann"], "pct2"), f(sp["net_ann"], "pct2"),
                         f(years_to_significance(s["ic_ir"], h), ".1f"), _flags(a)])
        L += _table(["alpha", "mean IC", "IC-IR", "NW t", "95% CI low", "95% CI high", "hit rate", "days",
                     "turnover", "share > 50%", "gross ann.", "net ann.", "years to sig.", "flags"], rows)
        later = [p[0] for p in pers if p[0] != "is"]
        if later and ids:
            L += ["Mean IC and NW t per period:", ""]
            head = ["alpha"] + [x for e in later for x in (f"{e} IC", f"{e} t")]
            L += _table(head, [[a] + [x for e in later for x in (f(m.results[a].summary[h][e]["mean_ic"], ".4f"),
                                                                  f(m.results[a].summary[h][e]["t_nw"], ".2f"))]
                               for a in ids])
        if ids:
            L += ["In-sample mean IC and NW t per horizon:", ""]
            head = ["alpha"] + [x for k in cfg.horizons for x in (f"h={f(k, 'd')} IC", f"h={f(k, 'd')} t")]
            L += _table(head, [[a] + [x for k in cfg.horizons
                                      for x in (f(m.results[a].summary[k]["is"]["mean_ic"], ".4f"),
                                                f(m.results[a].summary[k]["is"]["t_nw"], ".2f"))] for a in ids])
        L += ["Failed alphas: none." if not m.failures else "Failed alphas:", ""]
        L += [f"- {a}: {msg}" for a, msg in sorted(m.failures.items())] + ([""] if m.failures else [])

    # 6. convention variants
    L += ["## Convention variants", ""]
    if run.paired is not None:
        L += [f"Baseline run: `{run.baseline_run_id}`. Mean IC difference is variant minus baseline; "
              "t is Newey-West on the paired daily difference.", ""]
        L += _table(["market", "alpha", "mean IC difference", "paired NW t", "days"],
                    [[str(r.market), str(r.alpha_id), f(r.mean_diff, ".4f"), f(r.t_nw, ".2f"), f(r.n_days, "d")]
                     for r in run.paired.itertuples(index=False)])
    else:
        L += ["No baseline run given.", ""]

    # 7. caveats
    L += ["## Caveats", "",
          "- Sector and industry labels are today's classification (static), used for every date.",
          "- Market cap uses a static share count (shares outstanding today); alpha 56 is not point-in-time.",
          "- Alphas with zero delay (flag delay0) assume a close that a retail account cannot trade; read them as upper bounds.",
          "- The vwap field is a proxy (" + cfg.vwap_proxy + " price), not a traded VWAP.",
          "- The spread estimator measures the high-low range, not the bid-ask half-spread.",
          "- Survivorship: see the box at the top.",
          "- Legend: years to significance = (1.96 / (IC-IR x sqrt(252 / h))) squared, because overlapping h-day ICs give "
          "about 252 / h independent observations a year; 95% CI = mean IC +- 1.96 x |mean IC / NW t|.", ""]

    # 8. figures
    L += ["## Figures", ""] + [f"![{cap}]({fn})" for cap, fn in figures.items()] + [""]
    return "\n".join(L), f.seen


# ---------------------------------------------------------------------- figures
def _style():
    return plt.rc_context({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})


def _top(m: MarketOutput, cfg: RunConfig, k: int) -> list[str]:
    """Top k alpha ids by in-sample |NW t| at the primary horizon (NaN dropped, ties by id)."""
    t = {a: abs(r.summary[cfg.primary_horizon]["is"]["t_nw"]) for a, r in m.results.items()}
    t = {a: v for a, v in t.items() if np.isfinite(v)}      # drop non-finite BEFORE sorting (NaN breaks the order)
    return [a for a, v in sorted(t.items(), key=lambda kv: (-kv[1], kv[0]))][:k]


def _save(fig, out_dir: Path, name: str) -> str:
    fig.savefig(Path(out_dir) / name, dpi=150)
    plt.close(fig)
    return name


def fig_ic_heatmap(run: RunOutput, out_dir: Path) -> str | None:
    mks = [m for m in run.markets.values() if m.results]
    if not mks:
        return None
    hs = list(run.cfg.horizons)
    mats = []
    for m in mks:
        a = np.full((101, len(hs)), np.nan)
        for i in range(101):
            r = m.results.get(f"alpha{i + 1:03d}")
            if r is not None:
                a[i] = [r.summary[k]["is"]["mean_ic"] for k in hs]
        mats.append(a)
    vmax = max([np.nanmax(np.abs(a)) for a in mats if np.isfinite(a).any()] or [0.01])
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING).copy()
    cmap.set_bad(MUTED)   # not evaluated: distinct from the neutral (zero) colour
    with _style():
        fig, axes = plt.subplots(1, len(mks), figsize=(2.6 * len(mks) + 1.4, 14), squeeze=False, sharey=True)
        for ax, m, a in zip(axes[0], mks, mats):
            im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks(range(len(hs)), [str(k) for k in hs])
            ax.set_yticks(range(0, 101, 10), [str(i + 1) for i in range(0, 101, 10)])
            ax.set_title(m.market.upper())
            ax.set_xlabel("horizon (trading days)")
            ax.grid(False)
        axes[0][0].set_ylabel("alpha id")
        fig.suptitle("In-sample mean rank IC by alpha and horizon", color=INK)
        fig.colorbar(im, ax=list(axes[0]), label="mean rank IC (in-sample); gray = not evaluated", shrink=0.5)
        return _save(fig, out_dir, "ic_heatmap.png")


def fig_cum_ic(run: RunOutput, m: MarketOutput, out_dir: Path) -> str | None:
    cfg, h = run.cfg, run.cfg.primary_horizon
    top = _top(m, cfg, 10)
    if not top:
        return None
    bands = [p for p in periods(cfg) if p[0] != "is"]
    with _style():
        fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True, squeeze=False)
        xmax = None
        for ax, a in zip(axes.ravel(), top):
            ic = m.results[a].ic[h]["ic"]
            ic = ic[_visible(ic.index, cfg)].dropna()
            cum = ic.cumsum()
            ax.plot(cum.index, cum.to_numpy(), color=BLUE, lw=1.5)
            ax.set_title(f"{a} (t={m.results[a].summary[h]['is']['t_nw']:.2f})", pad=12)
            xmax = cum.index.max() if xmax is None else max(xmax, cum.index.max())
        for ax in axes.ravel()[len(top):]:
            ax.set_visible(False)
        lo = pd.Timestamp(cfg.start)
        for ax in axes.ravel()[:len(top)]:
            if xmax is None or pd.isna(xmax):
                break
            ax.set_xlim(lo, xmax)
            for k, (name, b0, b1) in enumerate(bands):
                x0, x1 = max(pd.Timestamp(b0), lo), min(pd.Timestamp(b1), xmax)
                if x1 <= x0:
                    continue
                if k % 2 == 0:
                    ax.axvspan(x0, x1, color=BAND, zorder=0)
                if ax in axes[0]:
                    ax.text(x0 + (x1 - x0) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=6, color=INK2)
        for ax in axes[:, 0]:
            ax.set_ylabel("cumulative IC")
        for ax in axes[1]:
            ax.tick_params(axis="x", labelrotation=30)
        fig.suptitle(f"{m.market.upper()}: cumulative daily rank IC at h = {h}, top alphas by in-sample |NW t|",
                     color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return _save(fig, out_dir, f"cum_ic_{m.market}.png")


def fig_deciles(run: RunOutput, m: MarketOutput, out_dir: Path) -> str | None:
    cfg = run.cfg
    top = _top(m, cfg, 5)
    if not top:
        return None
    nb = cfg.n_bins
    with _style():
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True, squeeze=False)
        for ax, a in zip(axes[0], top):
            d = m.results[a].deciles
            d = d.loc[(d.index >= pd.Timestamp(cfg.start)) & (d.index < pd.Timestamp(cfg.holdout_start))]
            ax.bar(range(1, nb + 1), [d[f"bin_{i}"].mean() * 1e4 for i in range(1, nb + 1)], width=0.6, color=BLUE)
            ax.axhline(0, color=BASE, lw=1.0)
            ax.set_title(a)
            ax.set_xlabel("bin (1 = lowest alpha)")
            ax.set_xticks(range(1, nb + 1))
        for ax in axes[0][len(top):]:
            ax.set_visible(False)
        axes[0][0].set_ylabel("mean daily return (bps)")
        fig.suptitle(f"{m.market.upper()}: in-sample mean daily return by bin, top alphas by |NW t|", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return _save(fig, out_dir, f"deciles_{m.market}.png")


def fig_neff(run: RunOutput, out_dir: Path) -> str | None:
    mks = [m for m in run.markets.values() if m.results and m.ic_corr.shape[0] > 0]
    if not mks:
        return None
    with _style():
        fig, axes = plt.subplots(1, len(mks), figsize=(4.2 * len(mks), 3.4), squeeze=False)
        for ax, m in zip(axes[0], mks):
            c = m.ic_corr.to_numpy(dtype=float, copy=True)
            c[~np.isfinite(c)] = 0.0
            np.fill_diagonal(c, 1.0)
            eig = np.sort(np.clip(np.linalg.eigvalsh(c), 0, None))[::-1]
            ax.plot(range(1, len(eig) + 1), eig, color=BLUE, lw=1.5)
            ax.set_title(f"{m.market.upper()} in-sample IC correlation")
            ax.set_xlabel("eigenvalue rank")
            ax.set_ylabel("eigenvalue")
            ax.text(0.97, 0.95, f"N_eff = {m.multiple_testing['n_eff']:.1f}", transform=ax.transAxes, ha="right",
                    va="top", color=INK2)
        fig.tight_layout()
        return _save(fig, out_dir, "neff_scree.png")


def report(run: RunOutput, out_dir: Path) -> Path:
    """Write out_dir/report.md and the PNGs; returns the markdown path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str] = {}

    def add(caption, name):
        if name:
            figs[caption] = name

    add("In-sample mean IC by alpha and horizon", fig_ic_heatmap(run, out_dir))
    for mk, m in run.markets.items():
        if not m.results:
            continue
        add(f"{mk.upper()} cumulative IC of the top alphas", fig_cum_ic(run, m, out_dir))
        add(f"{mk.upper()} mean return by bin of the top alphas", fig_deciles(run, m, out_dir))
    add("Eigenvalues of the in-sample IC correlation", fig_neff(run, out_dir))
    md, _ = render_markdown(run, figs)
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    return path
