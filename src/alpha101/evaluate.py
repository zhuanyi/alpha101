"""Evaluation layer: holdout censoring, IC by horizon and era, decile long-short, turnover, multiple
testing and the H3 composite (docs/design.md section 6).

Everything consumes an alpha (dates x symbols) and `lag.forward_returns`; nothing here shifts data
forward. Daily book returns are the 1-day forward returns, so a book formed from the signal on day s
earns rows s, s+1, ... of `forward_returns(panel, 1)` until its next rebalance.

Integrity rule: `censor_holdout` is the only place the holdout is cut. Alpha values are censored by
their own date, forward returns by their EXIT date, so no in-sample statistic sees a holdout price.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from .costs import CostModel, apply_costs
from .expr.evaluator import EvalOptions, Evaluator
from .expr.parser import canonical
from .formulas import get_alpha
from .lag import forward_returns
from .panel import Panel
from .stats import adjusted_threshold, effective_number_of_tests, ic_summary, rank_ic

LAG = 1  # implementation lag of every research run (docs/operators.md section 4)


@dataclass(frozen=True)
class RunConfig:
    markets: tuple[str, ...] = ("us", "ca")
    horizons: tuple[int, ...] = (1, 5, 10, 21)
    primary_horizon: int = 5
    start: str = "2005-01-01"
    holdout_start: str = "2019-01-01"
    unlock_holdout: bool = False
    options: EvalOptions = EvalOptions()
    basis: str = "raw"
    vwap_proxy: str = "typical"
    execution: str = "close"
    delisting: str = "last_close"
    min_names: dict = field(default_factory=lambda: {"us": 30, "ca": 20})
    n_bins: int = 10
    eras: tuple = (("2005-2012", "2005-01-01", "2012-12-31"), ("2013-2018", "2013-01-01", "2018-12-31"),
                   ("2019-2026", "2019-01-01", "2026-12-31"))


@dataclass
class AlphaResult:
    alpha_id: str                    # alpha001..alpha101, or expr_<sha1 of the canonical form>
    expression: str
    market: str
    basis: str                       # basis of the panel actually evaluated
    ic: dict                         # h -> DataFrame[ic, n], dates >= cfg.start
    summary: dict                    # h -> {"is" | era name -> ic_summary dict}
    deciles: pd.DataFrame            # decile_spread at the primary horizon
    turnover: pd.Series              # one-way, per sub-book rebalance, primary-horizon long-short book
    trades: pd.DataFrame             # costs.apply_costs input for deciles["long_short"]
    values_path: Path | None = None


def censor_holdout(df: pd.DataFrame, cfg: RunConfig, horizon: int | None = None,
                   lag: int = LAG) -> pd.DataFrame:
    """The ONLY place the holdout is cut.

    Locked (cfg.unlock_holdout False): returns a copy with rows dated >= holdout_start set to NaN.
    For forward returns pass their `horizon` (and `lag`): row t is then censored when its EXIT date
    t + lag + horizon is on or after holdout_start, so no in-sample statistic sees a holdout price.

    Unlocked: alpha values (horizon None) are returned unchanged (df itself, not a copy). Forward
    returns (horizon given) keep the boundary cut: a copy is returned where only the straddle band,
    rows [max(cut - lag - horizon, 0), cut) with cut the first row >= holdout_start, is NaN. Those are the
    in-sample signal dates whose exit is on or after holdout_start; rows >= cut (holdout signal dates)
    are kept. A return is thus used iff its signal and exit dates lie on the same side of
    holdout_start, and in-sample statistics are identical in locked and unlocked runs.

    Exit dates are counted in rows of `df.index`, which must therefore be the full panel calendar.
    """
    if cfg.unlock_holdout and horizon is None:
        return df
    cut = int(df.index.searchsorted(pd.Timestamp(cfg.holdout_start)))  # first row >= holdout_start
    out = df.copy()
    if horizon is None:
        out.iloc[cut:] = np.nan
    elif cfg.unlock_holdout:
        out.iloc[max(cut - lag - horizon, 0):cut] = np.nan
    else:
        out.iloc[max(cut - lag - horizon, 0):] = np.nan
    return out


def _scored(alpha: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Alpha as the statistics see it: censored, and NaN before cfg.start (warm-up rows)."""
    out = censor_holdout(alpha, cfg).copy()     # censor_holdout returns alpha itself when unlocked
    out.loc[out.index < pd.Timestamp(cfg.start)] = np.nan
    return out


def _r1(panel: Panel, cfg: RunConfig) -> np.ndarray:
    """Censored 1-day forward returns, the daily return of every book."""
    fwd = forward_returns(panel, 1, execution=cfg.execution, lag=LAG, delisting=cfg.delisting)
    r = censor_holdout(fwd, cfg, horizon=1)
    return r.to_numpy(dtype=float, copy=True)


def _market(cfg: RunConfig, market: str | None) -> str:
    return cfg.markets[0] if market is None else market


def _min_names(cfg: RunConfig, market: str) -> int:
    try:
        return cfg.min_names[market]
    except KeyError:
        raise ValueError(f"unknown market {market!r}; valid markets: {sorted(cfg.min_names)}") from None


def _resolve(alpha_id_or_expr) -> tuple[str, str]:
    if isinstance(alpha_id_or_expr, (int, np.integer)):
        spec = get_alpha(int(alpha_id_or_expr))
        return spec.name, spec.expression
    expr = str(alpha_id_or_expr)
    return "expr_" + hashlib.sha1(canonical(expr).encode()).hexdigest()[:10], expr


def evaluate_alpha(panel: Panel, alpha_id_or_expr, cfg: RunConfig, out_dir: Path | None,
                   market: str | None = None) -> AlphaResult:
    """Evaluate one alpha once (fresh Evaluator) and compute every per-alpha statistic.

    The panel is used as given (its basis and vwap proxy come from the loader); its universe mask
    (None = everyone) is applied by the Evaluator. `market` picks cfg.min_names and defaults to
    cfg.markets[0]. If `out_dir` is given, the censored alpha is written there as float32 parquet.

    `trades` of a locked run can carry the end-of-IS liquidation: its `n_trades` on the first holdout
    rows (the book goes flat there) and its `traded` on the last held in-sample row before them.
    No holdout return is ever charged or read.
    """
    market = _market(cfg, market)
    min_names = _min_names(cfg, market)
    name, expr = _resolve(alpha_id_or_expr)
    alpha = Evaluator(panel, cfg.options).evaluate(expr)
    values_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        values_path = out_dir / f"{name}.parquet"
        censor_holdout(alpha, cfg).astype("float32").to_parquet(values_path)
    a = _scored(alpha, cfg)
    start = pd.Timestamp(cfg.start)
    holdout = pd.Timestamp(cfg.holdout_start)
    ics, summary = {}, {}
    for h in sorted(set(cfg.horizons) | {cfg.primary_horizon}):
        fwd = censor_holdout(forward_returns(panel, h, execution=cfg.execution, lag=LAG,
                                             delisting=cfg.delisting), cfg, horizon=h)
        ic = rank_ic(a, fwd, min_names=min_names)
        ic = ic.loc[ic.index >= start]
        ics[h] = ic
        per = {"is": ic_summary(ic["ic"].loc[ic.index < holdout], h)}
        for era, lo, hi in cfg.eras:
            per[era] = ic_summary(ic["ic"].loc[lo:hi], h)
        summary[h] = per
    deciles, turn, trades = _decile_book(a, panel, cfg.primary_horizon, cfg.n_bins, cfg,
                                         min_names)
    return AlphaResult(alpha_id=name, expression=expr, market=market, basis=panel.basis, ic=ics,
                       summary=summary, deciles=deciles, turnover=turn, trades=trades,
                       values_path=values_path)


# ---------------------------------------------------------------------- books
def _block(R: np.ndarray, fin: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold over the rows of R from weights w (N, k columns of non-negative weights).

    Returns (daily returns (L, k), growth of each name over the whole block (N,)). A name with a
    missing return that day earns nothing and drops out of that day's denominator.
    """
    R0 = np.where(fin, R, 0.0)
    G = np.cumprod(1.0 + R0, axis=0)
    Gprev = np.vstack([np.ones((1, R.shape[1])), G[:-1]])
    num = (Gprev * R0) @ w
    den = (Gprev * fin) @ w
    with np.errstate(all="ignore"):
        ret = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return ret, G[-1]


def _rebase(w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """One side's weights drifted by the block growth g and rebased to sum 1. A side whose growth sums
    to 0 (every name went to 0) is worth 0, as in `composite_backtest`: zeros are returned, nothing is
    sold, and rebuilding the side trades its full new weight (no NaN)."""
    grown = w * g
    tot = grown.sum()
    return grown / tot if tot > 0 else np.zeros_like(w)


def _holdout_cut(panel: Panel, cfg: RunConfig) -> int | None:
    """First row >= holdout_start in a locked run; None when unlocked (nothing is censored)."""
    return None if cfg.unlock_holdout else int(panel.dates.searchsorted(pd.Timestamp(cfg.holdout_start)))


def _charge_liquidations(traded: np.ndarray, gross: np.ndarray, liquidations: list, n_trades: np.ndarray,
                         start_row: int = 0, cut: int | None = None) -> None:
    """Add each liquidation (s, lo, amount, tickets) to `traded` on the last row of lo..s-1 with a finite gross.

    In a locked run (`cut` = first holdout row) a liquidation whose flat row s >= cut is skipped: the
    book is flat only because the alphas are censored at the holdout, so no in-sample row is charged
    for it (its tickets stay in `n_trades` on row s, which is in the holdout). Locked and unlocked runs
    then agree on every row before the holdout, provided the alphas exist on/after `cut`.

    A liquidation into a flat book opens no book, so the row s that carries it has no return to charge
    (`apply_costs` would give NaN). It is charged on the closing book's last row with a finite gross
    (for a flat s in [cut-1, cut+1] that is cut-3, because the straddle band is NaN, not s-1).

    A liquidation that cannot be charged is dropped from `traded` and its tickets are removed from
    `n_trades[s]`, so the two always agree. That is the case when the closing book has no finite gross
    row, and when the charge row is before `start_row` (the first output row): the build of that book
    before the start was free, so its liquidation is too."""
    fin = np.isfinite(gross)
    for s, lo, amount, tickets in liquidations:
        if cut is not None and s >= cut:
            continue
        rows = np.flatnonzero(fin[lo:s])
        if len(rows) and lo + rows[-1] >= start_row:
            traded[lo + rows[-1]] += amount
        else:
            n_trades[s] -= tickets


def _bins(A: np.ndarray, n_bins: int, min_names: int) -> np.ndarray:
    """Int (T, N) bin per cell, 1 = lowest alpha .. n_bins = highest; 0 = not held.

    Ordinal ranks, ties broken by symbol order (the alphalens convention), so every bin is
    populated even for alphas with few distinct values (alpha 1 takes five). Rows with fewer than
    max(min_names, n_bins) finite alphas hold nothing.
    """
    pct = pd.DataFrame(A).rank(axis=1, method="first", pct=True).to_numpy(dtype=float, copy=True)
    b = np.where(np.isfinite(pct), np.clip(np.ceil(pct * n_bins), 1, n_bins), 0).astype(np.int64)
    b[np.isfinite(A).sum(axis=1) < max(min_names, n_bins)] = 0
    return b


def _decile_book(alpha: pd.DataFrame, panel: Panel, horizon: int, n_bins: int, cfg: RunConfig,
                 min_names: int):
    """(daily DataFrame, turnover Series, trades DataFrame) of the staggered decile book."""
    A = alpha.reindex(index=panel.dates, columns=panel.symbols).to_numpy(dtype=float, copy=True)
    R = _r1(panel, cfg)
    fin = np.isfinite(R)
    T, N = A.shape
    bins = _bins(A, n_bins, min_names)
    ls = np.full((horizon, T), np.nan)
    per_bin = np.full((horizon, T, n_bins), np.nan)
    traded = np.zeros(T)
    n_trades = np.zeros(T)
    liquidations = []                              # (flat row s, first row of the closing book, traded)
    turns = []
    labels = np.arange(1, n_bins + 1)
    for j in range(horizon):                       # sub-book j rebalances on rows j, j+h, j+2h, ...
        reb = np.arange(j, T, horizon)
        ends = np.append(reb[1:], T)
        W_new = np.zeros((len(reb), N))
        W_drift = np.zeros((len(reb), N))
        prev = None                                # (long weights, short weights, growth over last block)
        for k, (s, e) in enumerate(zip(reb, ends)):
            if prev is not None:                   # w * (1 + r) / (1 + r_side): each side rebased to 1
                wl, wsh, g = prev
                W_drift[k] = _rebase(wl, g) - _rebase(wsh, g)
            b = bins[s]
            if not (b == n_bins).any() or not (b == 1).any():
                if prev is not None:               # the book goes flat: the liquidation is a trade
                    flat = np.abs(W_drift[k])
                    tickets = int((flat > 1e-12).sum())
                    liquidations.append((s, reb[k - 1], flat.sum() / horizon, tickets))
                    n_trades[s] += tickets
                prev = None
                continue
            M = (b[:, None] == labels[None, :]).astype(float)
            M /= M.sum(axis=0, keepdims=True).clip(min=1.0)    # equal weight within each bin
            ret, g = _block(R[s:e], fin[s:e], M)
            per_bin[j, s:e] = ret
            ls[j, s:e] = ret[:, -1] - ret[:, 0]
            W_new[k] = M[:, -1] - M[:, 0]
            dw = np.abs(W_new[k] - W_drift[k])
            traded[s] += dw.sum() / horizon         # sum |dw| (no 0.5), unlike `turnover`
            n_trades[s] += int((dw > 1e-12).sum())
            prev = (M[:, -1], M[:, 0], g)
        idx = panel.dates[reb]
        turns.append(turnover(pd.DataFrame(W_new, index=idx), pd.DataFrame(W_drift, index=idx)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # rows where no sub-book is active
        daily = {"long_short": np.nanmean(ls, axis=0)}
        mb = np.nanmean(per_bin, axis=0)
    for i in range(n_bins):
        daily[f"bin_{i + 1}"] = mb[:, i]
    out = pd.DataFrame(daily, index=panel.dates)
    keep = panel.dates >= pd.Timestamp(cfg.start)
    turn = pd.concat(turns).sort_index()
    turn = turn.loc[turn.index >= pd.Timestamp(cfg.start)]
    _charge_liquidations(traded, out["long_short"].to_numpy(), liquidations, n_trades,
                         int(panel.dates.searchsorted(pd.Timestamp(cfg.start))), _holdout_cut(panel, cfg))
    trades = pd.DataFrame({"traded": traded, "short": np.isfinite(out["long_short"]).astype(float),
                           "n_trades": n_trades}, index=panel.dates)
    return out.loc[keep], turn, trades.loc[keep]


def decile_spread(alpha: pd.DataFrame, panel: Panel, horizon: int, n_bins: int, cfg: RunConfig,
                  min_names: int | None = None) -> pd.DataFrame:
    """Daily long-short (top bin - bottom bin) and per-bin returns, columns long_short, bin_1..bin_n.

    `horizon` sub-books are staggered: sub-book j rebalances on rows j, j+h, ... and holds h days,
    and the daily return is the mean over active sub-books, so every day contributes. At each
    rebalance names are binned by cross-sectional rank of the alpha (equal weight within a bin, ties
    broken by symbol order) and then held buy-and-hold. The long-short book is long the top bin and
    short the bottom bin with unit capital per side; its turnover drifts each side separately. Daily returns are `forward_returns(panel, 1, cfg.execution)`
    only, censored by exit date. Pass the alpha already scored (`_scored`): NaN cells are not held.
    A rebalance with fewer than max(min_names, n_bins) names holds nothing (min_names defaults to
    n_bins). Rows are dates >= cfg.start.

    A book that goes flat (too few names) is liquidated; the liquidation is a trade, charged in
    `trades` on the closing book's last row with a finite long-short return (row s-1 for a flat
    rebalance row s), while `n_trades` counts its tickets on the flat rebalance row s itself. A
    liquidation with no finite row to charge, or whose charge row is before `cfg.start`, is neither
    in `traded` nor in `n_trades`. In a locked run a liquidation whose flat row is on or after the
    holdout (the alphas are censored there) is not charged at all, so locked and unlocked runs agree
    on every pre-holdout row when the alphas exist on/after the holdout. A side
    whose names all went to 0 is worth 0: nothing is sold, rebuilding it trades its full new weight.
    """
    return _decile_book(alpha, panel, horizon, n_bins, cfg, n_bins if min_names is None else min_names)[0]


def turnover(weights_by_rebalance: pd.DataFrame, drifted: pd.DataFrame | None = None) -> pd.Series:
    """One-way turnover per rebalance: 0.5 * sum |w_new - w_drifted| (both sides count).

    Rows are rebalance dates, columns names; 0 or NaN = not held. `drifted` holds the pre-trade
    weights at each rebalance, i.e. the previous book drifted to that date,
    w_prev * (1 + r) / (1 + r_p) (docs/design.md section 6). Without it the previous row is used
    undrifted. A row with no new book or no previous book (the first build, a re-entry) is NaN.
    For a long-short book with unit capital per side the maximum is 2 (both sides flip).
    """
    new = weights_by_rebalance.fillna(0.0).to_numpy(dtype=float, copy=True)
    if drifted is None:
        old = np.vstack([np.zeros((1, new.shape[1])), new[:-1]])
    else:
        old = drifted.reindex_like(weights_by_rebalance).fillna(0.0).to_numpy(dtype=float, copy=True)
    tv = 0.5 * np.abs(new - old).sum(axis=1)
    tv[~(new != 0).any(axis=1) | ~(old != 0).any(axis=1)] = np.nan
    return pd.Series(tv, index=weights_by_rebalance.index, name="turnover")


# ---------------------------------------------------------------------- cross-alpha statistics
def ic_correlation(ics: dict[str, pd.Series], before: str | None = None,
                   min_periods: int = 60) -> pd.DataFrame:
    """Pairwise-complete correlation of daily IC series (alphas as columns); pairs with fewer than
    `min_periods` overlapping days are NaN (`multiple_testing` counts them as 0).

    N_eff is estimated on the in-sample period (docs/design.md section 6): pass
    before=cfg.holdout_start to drop holdout rows when the series come from an unlocked run.
    """
    df = pd.DataFrame(ics)
    if before is not None:
        df = df.loc[df.index < pd.Timestamp(before)]
    return df.corr(min_periods=min_periods)


def multiple_testing(summary: pd.DataFrame, ic_corr: pd.DataFrame) -> dict:
    """N_eff, |t| thresholds and how many alphas clear each.

    `summary` has one row per alpha and a `t_nw` column (in-sample, primary horizon). NaN entries of
    `ic_corr` (pairs with too little overlap) count as 0 off the diagonal. An alpha with fewer than
    60 in-sample IC days therefore has NaN correlations with everyone and stays counted in N_eff as a
    fully independent test: every attempted alpha is a test, however little data it has.
    """
    c = ic_corr.to_numpy(dtype=float, copy=True)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    n_eff = effective_number_of_tests(c)
    thresholds = {"raw": 1.96, "sidak_neff": adjusted_threshold(n_eff),
                  "bonferroni_101": float(_st.norm.ppf(1 - 0.05 / 101 / 2)), "hlz": 3.0}
    t = summary["t_nw"].abs()
    return {"n_eff": n_eff, "n_tests": int(len(summary)), "thresholds": thresholds,
            "passing": {k: int((t > v).sum()) for k, v in thresholds.items()}}


# ---------------------------------------------------------------------- H3 composite
def _zscore(a: pd.DataFrame) -> np.ndarray:
    x = a.to_numpy(dtype=float, copy=True)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        z = (x - np.nanmean(x, axis=1, keepdims=True)) / np.nanstd(x, axis=1, ddof=1, keepdims=True)
    z[~np.isfinite(z)] = np.nan
    return z


def composite_backtest(panel: Panel, members: list[int], weights, cfg: RunConfig, costs: CostModel,
                       book: str = "long_short", n_long: int = 10, n_short: int = 10,
                       rebalance: str = "W-FRI", market: str | None = None) -> pd.DataFrame:
    """H3: a z-score composite of fixed members traded as an n_long / n_short book.

    `members` and `weights` must come from in-sample data only (docs/research_plan.md H3); this function
    never selects. weights: "equal" (+1 each) or a dict {alpha id: signed weight}, e.g. -1 for a
    member with a negative in-sample IC. The composite is sum(w * z) / sum(|w|) over the members
    available in a cell, z = cross-sectional z-score of each scored alpha.

    Rebalances on the last trading day of each `rebalance` period (pandas period alias). Trading
    rule: only membership changes trade. Names that leave are sold, names that enter share the
    weight they free equally, and names that stay keep their drifted weight (no re-equalising
    trades at $9.99 a ticket). Each side is rebased to 1 at every rebalance. book="long_only" holds
    the top n_long only; `benchmark` is the equal-weight return of the universe names.

    Columns (dates >= cfg.start): gross, net, cost, long, short, benchmark, turnover (one-way, at
    rebalance dates), n_trades, and `active` = net - benchmark for long_only.

    A rebalance with too few finite composite names leaves the book flat and liquidates it. The
    liquidation is charged (`cost`, `net`) on the closing book's last row with a finite gross, i.e.
    row s-1 for a flat rebalance row s; `n_trades` counts its tickets on the flat row s itself. In a
    locked run a liquidation whose flat row is on or after the holdout is not charged (the book is flat
    only because the alphas are censored there), so locked and unlocked runs agree before the holdout.
    """
    if book not in ("long_short", "long_only"):
        raise ValueError("book must be 'long_short' or 'long_only'")
    market = _market(cfg, market)
    if isinstance(weights, str):
        if weights != "equal":
            raise ValueError("weights must be 'equal' or a dict {alpha id: weight}")
        wts = {m: 1.0 for m in members}
    else:
        wts = {m: float(weights[m]) for m in members}
    num = np.zeros(panel.shape)
    den = np.zeros(panel.shape)
    for m in members:
        z = _zscore(_scored(Evaluator(panel, cfg.options).evaluate(get_alpha(m).expression), cfg))
        ok = np.isfinite(z)
        num += np.where(ok, wts[m] * z, 0.0)
        den += np.where(ok, abs(wts[m]), 0.0)
    with np.errstate(all="ignore"):
        comp = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    R = _r1(panel, cfg)
    fin = np.isfinite(R)
    T, N = comp.shape
    period = pd.Series(np.arange(T), index=panel.dates).groupby(panel.dates.to_period(rebalance)).max()
    reb = np.sort(period.to_numpy())
    # the panel's last row cannot be known to close its period, and its book would earn nothing
    reb = reb[(panel.dates[reb] >= pd.Timestamp(cfg.start)) & (reb < T - 1)]
    ends = np.append(reb[1:], T)
    sides = [("long", n_long, 1.0)] + ([("short", n_short, -1.0)] if book == "long_short" else [])
    need = n_long + (n_short if book == "long_short" else 0)
    ret = {k: np.full(T, np.nan) for k, _, _ in sides}
    traded = np.zeros(T)
    n_trades = np.zeros(T)
    liquidations = []                                      # (flat row s, first row of the closing book, traded)
    built = None                                           # first row of the last block held (None: flat)
    W_new = np.zeros((len(reb), N))
    W_drift = np.zeros((len(reb), N))
    held = {k: np.zeros(N) for k, _, _ in sides}          # per-side weights, drifted, sum 1
    order = np.arange(N)
    for i, (s, e) in enumerate(zip(reb, ends)):
        c = comp[s]
        ok = np.isfinite(c)
        if ok.sum() < need:
            for k, _, _ in sides:                  # the book goes flat: liquidate (charged on its last row)
                tickets = int((held[k] > 1e-12).sum())
                if built is not None:
                    liquidations.append((s, built, held[k].sum(), tickets))
                n_trades[s] += tickets
            built = None
            held = {k: np.zeros(N) for k, _, _ in sides}
            continue
        idx = order[ok][np.argsort(-c[ok], kind="stable")]   # best first; ties by symbol order
        for k, n, sign in sides:
            target = idx[:n] if sign > 0 else idx[::-1][:n]
            w = held[k]
            W_drift[i] += sign * w
            keep = np.zeros(N, dtype=bool)
            keep[target] = True
            new = np.where(keep, w, 0.0)
            enter = keep & (w == 0)
            if enter.any():
                new[enter] = (1.0 - new.sum()) / enter.sum()
            W_new[i] += sign * new
            dw = np.abs(new - w)
            traded[s] += dw.sum()
            n_trades[s] += int((dw > 1e-12).sum())
            r, g = _block(R[s:e], fin[s:e], new[:, None])
            ret[k][s:e] = r[:, 0]
            grown = new * g
            held[k] = grown / grown.sum() if grown.sum() > 0 else np.zeros(N)
        built = s
    idx = panel.dates[reb]
    turn = turnover(pd.DataFrame(W_new, index=idx), pd.DataFrame(W_drift, index=idx))
    uni = np.ones((T, N), dtype=bool) if panel.universe is None else np.asarray(panel.universe, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bench = np.nanmean(np.where(uni & fin, R, np.nan), axis=1)
    out = pd.DataFrame({"long": ret["long"], "short": ret.get("short", np.full(T, np.nan)),
                        "benchmark": bench}, index=panel.dates)
    out["gross"] = out["long"] - out["short"] if book == "long_short" else out["long"]
    _charge_liquidations(traded, out["gross"].to_numpy(), liquidations, n_trades,
                         int(panel.dates.searchsorted(pd.Timestamp(cfg.start))), _holdout_cut(panel, cfg))
    trades = pd.DataFrame({"traded": traded,
                           "short": np.isfinite(out["short"]).astype(float)}, index=panel.dates)
    out["net"] = apply_costs(out["gross"], trades, market, costs)
    out["cost"] = out["gross"] - out["net"]
    out["turnover"] = turn.reindex(panel.dates)
    out["n_trades"] = n_trades
    if book == "long_only":
        out["active"] = out["net"] - out["benchmark"]
    return out.loc[panel.dates >= pd.Timestamp(cfg.start)]
