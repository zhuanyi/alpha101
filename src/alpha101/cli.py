"""Command line: list, expr, ingest, universe, run .

`list` and `expr` never touch the holdout. `run` prints only identifiers and row counts; every statistic goes
to the report. Unlocking the holdout needs a committed pre-registration file and a DB registration.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import persist
from . import report as report_mod
from .costs import CostModel
from .data import constituents, ingest as ingest_mod, universe
from .data.cache import db_fingerprint, load_panel
from .data.yahoo import read_long
from .db import reader, writer
from .evaluate import RunConfig, evaluate_alpha, ic_correlation, multiple_testing
from .expr.evaluator import EvalOptions
from .formulas import load_alphas
from .stats import newey_west_t

log = logging.getLogger(__name__)

MARKETS = ("us", "ca")
BENCHMARKS = {"us": ("RSP", "equal"), "ca": ("XIC.TO", "cap")}


class UsageError(ValueError):
    """A bad command line value (exit code 2)."""


# ---------------------------------------------------------------------- parsing
def _ints(text: str, what: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",")]
    except ValueError:
        raise UsageError(f"{what}: expected comma-separated integers, got {text!r}") from None
    if not vals or any(v < 1 for v in vals):
        raise UsageError(f"{what}: values must be positive integers, got {text!r}")
    return list(dict.fromkeys(vals))


def _alphas(text: str) -> list[int]:
    known = load_alphas()
    if text == "all":
        return sorted(known)
    ids = _ints(text, "--alphas")
    bad = [i for i in ids if i not in known]
    if bad:
        raise UsageError(f"--alphas: unknown alpha ids {bad}")
    return ids


def _markets(text: str) -> list[str]:
    ms = [x.strip().lower() for x in text.split(",")]
    if not text or any(m not in MARKETS for m in ms):
        raise UsageError(f"--markets: expected a comma-separated subset of {MARKETS}, got {text!r}")
    return list(dict.fromkeys(ms))


def _markets_arg(text: str) -> list[str]:
    if text == "all":
        return ["ca", "us"]   # CA first: the US capped source needs the CA source
    if text not in MARKETS:
        raise UsageError(f"--market: expected us, ca or all, got {text!r}")
    return [text]


# ---------------------------------------------------------------------- list / expr / ingest / universe
def cmd_list(args) -> int:
    for i, a in sorted(load_alphas().items()):
        print(f"{i:3d}  inputs={','.join(a.inputs)}  max_lookback_days={a.max_lookback_days}  delay={a.delay}  "
              f"price_scale_invariant={a.price_scale_invariant}  needs_industry={a.needs_industry}")
    return 0


def cmd_expr(args) -> int:
    market = args.market
    cfg = RunConfig(markets=(market,), start=args.start)
    panel = load_panel(market, start=args.start, universe_source="liquidity")
    res = evaluate_alpha(panel, args.expression, cfg, None, market=market)
    print(f"{res.alpha_id}: {res.expression}")
    print(f"{market.upper()}, in-sample only, start {cfg.start}")
    print(f"{'horizon':>7} {'mean IC':>9} {'NW t':>7} {'days':>6}")
    for h in sorted(res.summary):
        s = res.summary[h]["is"]
        print(f"{h:>7d} {s['mean_ic']:>9.4f} {s['t_nw']:>7.2f} {s['n_days']:>6d}")
    return 0


def _compact(res: dict) -> dict:
    return {k: (len(v) if isinstance(v, (list, dict, tuple, set)) else v) for k, v in res.items()}


def cmd_ingest(args) -> int:
    for m in _markets_arg(args.market):
        res = ingest_mod.ingest(m, limit=args.limit)
        print(f"ingest {m}: " + ", ".join(f"{k}={v}" for k, v in _compact(res).items()))
    return 0


def cmd_universe(args) -> int:
    for m in _markets_arg(args.market):
        res = universe.build_universe(m, version=args.version)
        print(f"universe {m}: version={res['version']} rows_written={res['rows_written']} "
              f"member_count_range={res['member_count_range']}")
        for src, v in sorted(res["sources"].items()):
            print(f"  source {src}: {v}")
        print(f"  screen={res['screen']}")
        for y, v in sorted(res["screen_removed_by_year"].items()):
            print(f"  screen_removed {y}: {v}")
        if "cacap" in res:
            print(f"  cacap={res['cacap']}")
    return 0


# ---------------------------------------------------------------------- run
def _paired(this: dict, baseline: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Per (market, alpha) the daily IC difference this - baseline on common in-sample dates."""
    h = cfg.primary_horizon
    hs = pd.Timestamp(cfg.holdout_start)
    rows = []
    for mk, results in this.items():
        for aid, r in results.items():
            b = baseline[(baseline["market"] == mk.upper()) & (baseline["alpha_id"] == aid)]
            base = pd.Series(b["ic"].to_numpy(dtype=float), index=pd.to_datetime(b["date"]))
            cur = r.ic[h]["ic"]
            idx = cur.index.intersection(base.index)
            idx = idx[idx < hs]
            d = (cur.loc[idx] - base.loc[idx]).dropna()
            rows.append({"market": mk, "alpha_id": aid,
                         "mean_diff": float(d.mean()) if len(d) else float("nan"),
                         "t_nw": newey_west_t(d, lags=h), "n_days": int(len(d))})
    return pd.DataFrame(rows, columns=["market", "alpha_id", "mean_diff", "t_nw", "n_days"])


def _read_baseline(run_id: str, horizon: int, before: str | None = None) -> pd.DataFrame:
    """Baseline daily ICs at one horizon; `before` (a locked run's holdout_start) keeps holdout rows unread."""
    sql = "SELECT alpha_id, market, horizon, date, ic FROM alpha_ic WHERE run_id=%s AND horizon=%s"
    params = [run_id, horizon]
    if before is not None:
        sql += " AND date < %s"
        params.append(before)
    with reader() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = list(cur.fetchall())
    if not rows:
        raise ValueError(f"baseline run {run_id!r} has no alpha_ic rows at the primary horizon")
    df = pd.DataFrame(rows, columns=["alpha_id", "market", "horizon", "date", "ic"])
    df["ic"] = pd.to_numeric(df["ic"], errors="coerce").astype(float)
    return df


def _market_run(m: str, args, cfg: RunConfig, costs: CostModel, ids: list[int], out_dir: Path, raw_dir: Path):
    """Evaluate every alpha on one market and gather the report inputs. Returns (MarketOutput, source)."""
    with reader() as conn:
        source = universe.resolve_source(conn, args.universe or universe.liquidity_source(m))
    print(f"[{m}] universe source {source}", file=sys.stderr)
    panel = load_panel(m, args.start, cfg.basis, universe_source=source)
    raw = panel if cfg.basis == "raw" else load_panel(m, args.start, "raw", universe_source=source)
    results, failures = {}, {}
    for i in ids:
        name = f"alpha{i:03d}"
        t0 = time.time()
        try:
            results[name] = evaluate_alpha(panel, i, cfg, out_dir / "alpha_values" / m, market=m)
        except Exception as e:  # noqa: BLE001 - one alpha must not stop the run
            failures[name] = f"{type(e).__name__}: {e}"
            log.error("%s %s failed: %s", m, name, failures[name])
        print(f"[{m}] {name} {time.time() - t0:.1f}s" + ("  FAILED" if name in failures else ""), file=sys.stderr)

    h = cfg.primary_horizon
    dates = panel.dates[panel.dates >= pd.Timestamp(cfg.start)]
    summ = pd.DataFrame({"t_nw": {**{a: r.summary[h]["is"]["t_nw"] for a, r in results.items()},
                                  **{a: float("nan") for a in failures}}})
    ics = {a: r.ic[h]["ic"] for a, r in results.items()}
    ics.update({a: pd.Series(np.nan, index=dates) for a in failures})
    corr = ic_correlation(ics, before=cfg.holdout_start)
    mt = multiple_testing(summ, corr)

    uni = np.ones(panel.shape, bool) if panel.universe is None else np.asarray(panel.universe, bool)
    keep = np.asarray(panel.dates >= pd.Timestamp(cfg.start))
    counts = pd.Series(uni[keep].sum(axis=1), index=panel.dates[keep])

    with reader() as conn:
        try:
            surv = report_mod.survivorship_counts(conn, constituents.pool_from_raw(raw_dir, since="2005-01-01"), m)
        except FileNotFoundError as e:
            log.warning("no pool snapshot in %s (%s); survivorship counts left empty", raw_dir, e)
            surv = {}
        sym, weighting = BENCHMARKS[m]
        bl = read_long(conn, symbols=[sym], start=str(panel.dates[0].date()))
        screen = {}
        if source.startswith("liquidity_top"):
            elig = universe.source_symbols(conn, source)
            screen = report_mod.screen_summary(raw, m, elig, start=cfg.start,
                                               end=None if cfg.unlock_holdout else cfg.holdout_start)
    bench = {}
    if len(bl):
        adj = bl.set_index("date")["adj_close"].astype(float)
        bench = {"name": sym, "weighting": weighting, "table": report_mod.benchmark_gap(panel, adj, cfg, weighting)}
    else:
        log.warning("no %s rows in yahoo_ohlcv; benchmark left empty", sym)
    cap = None
    if source.startswith("liquidity_cacap_us"):
        cap = counts.groupby(counts.index.to_period("M")).max()
    mo = report_mod.MarketOutput(
        market=m, universe_source=source, results=results, failures=failures, multiple_testing=mt, ic_corr=corr,
        member_counts=counts, survivorship=surv, benchmark=bench, screen=screen,
        delisting_changed=report_mod.delisting_effect(panel, cfg), cap_series=cap)
    return mo


def cmd_run(args) -> int:
    if args.unlock_holdout and not args.prereg:
        print("error: --unlock-holdout requires --prereg <pre-registration file>", file=sys.stderr)
        return 2
    if args.unlock_holdout and args.no_db:
        print("error: --unlock-holdout needs DB registration (the pre-registration SHA is stored there); "
              "drop --no-db", file=sys.stderr)
        return 2
    try:
        ids = _alphas(args.alphas)
        markets = _markets(args.markets)
        horizons = _ints(args.horizons, "--horizons")
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.horizon < 1:
        print("error: --horizon must be a positive integer", file=sys.stderr)
        return 2
    if args.universe and len(markets) != 1:
        print("error: --universe is only valid with exactly one market", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")

    cfg = RunConfig(markets=tuple(markets), horizons=tuple(sorted(set(horizons) | {args.horizon})),
                    primary_horizon=args.horizon, start=args.start, unlock_holdout=args.unlock_holdout,
                    options=EvalOptions(argmax_convention=args.argmax_convention, adv_units=args.adv_units),
                    basis=args.basis, execution=args.execution, delisting=args.delisting)
    costs = CostModel()
    git_sha, dirty = persist.git_state()
    run_id = persist.new_run_id(git_sha)
    out_dir = Path(args.out) / run_id
    raw_dir = Path(args.raw_dir)
    prereg = Path(args.prereg) if args.prereg else None

    fps = {}
    with reader() as conn:
        for m in markets:
            fps[m] = db_fingerprint(conn, m)
    prereg_sha = None
    if not args.no_db:
        conn = writer()
        try:
            persist.ensure_tables(conn)
            persist.register_run(conn, run_id, cfg, costs, fps, prereg)   # refuses a dirty tree
        finally:
            conn.close()
        if prereg is not None:
            prereg_sha = hashlib.sha256(prereg.resolve().read_bytes()).hexdigest()
    sha_note = git_sha if not args.no_db else git_sha + (" (dirty tree, not registered)" if dirty
                                                         else " (not registered)")
    baseline = (_read_baseline(args.baseline_run, cfg.primary_horizon,
                               None if cfg.unlock_holdout else cfg.holdout_start)
                if args.baseline_run else None)

    outputs, ic_rows, summaries = {}, 0, 0
    report_path = out_dir / "report.md"
    for m in markets:
        mo = _market_run(m, args, cfg, costs, ids, out_dir, raw_dir)
        outputs[m] = mo
        if not args.no_db:
            conn = writer()
            try:
                for a, r in mo.results.items():
                    for h, ic in r.ic.items():
                        ic_rows += persist.write_ic(conn, run_id, a, m, h, ic)
                    persist.write_summary(conn, run_id, a, m, r, cfg, costs, mo.universe_source, str(report_path))
                    summaries += 1
            finally:
                conn.close()
    paired = _paired({m: o.results for m, o in outputs.items()}, baseline, cfg) if baseline is not None else None
    run = report_mod.RunOutput(
        run_id=run_id, git_sha=sha_note, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cfg=cfg, costs=costs, markets=outputs, fingerprints=fps, prereg_sha256=prereg_sha, paired=paired,
        baseline_run_id=args.baseline_run)
    path = report_mod.report(run, out_dir)
    print(f"run_id={run_id}")
    print(f"report={path}")
    print(f"alpha_ic_rows={ic_rows}")
    print(f"backtest_results_rows={summaries}")
    n_fail = sum(len(o.failures) for o in outputs.values())
    print(f"failures={n_fail}")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpha101", description="alpha101 research CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the 101 alphas with their static properties").set_defaults(fn=cmd_list)

    e = sub.add_parser("expr", help="ad-hoc in-sample IC table of one expression")
    e.add_argument("expression")
    e.add_argument("--market", choices=MARKETS, required=True)
    e.add_argument("--start", default="2015-01-01")
    e.set_defaults(fn=cmd_expr)

    g = sub.add_parser("ingest", help="fetch Yahoo history into yahoo_ohlcv (writes the DB)")
    g.add_argument("--market", choices=(*MARKETS, "all"), required=True)
    g.add_argument("--limit", type=int)
    g.set_defaults(fn=cmd_ingest)

    u = sub.add_parser("universe", help="build and store the universe sources (writes the DB)")
    u.add_argument("--market", choices=(*MARKETS, "all"), required=True)
    u.add_argument("--version", help="YYYYMMDD[a-z]; default today (UTC)")
    u.set_defaults(fn=cmd_universe)

    r = sub.add_parser("run", help="evaluate alphas, write the report (and the DB unless --no-db)",
                       description="Evaluate alphas and write the report. Exit code 0 only for a clean run; "
                                   "1 if any alpha failed (the report is still written) or on an uncaught error; "
                                   "2 for a usage error.")
    r.add_argument("--alphas", default="all", help="all or a comma-separated list of ids")
    r.add_argument("--start", default="2005-01-01")
    r.add_argument("--horizon", type=int, default=5)
    r.add_argument("--horizons", default="1,5,10,21")
    r.add_argument("--markets", default="us,ca")
    r.add_argument("--universe", help="universe source (one market only); default the market's liquidity source")
    r.add_argument("--argmax-convention", choices=("oldest_1based", "days_ago"), default="oldest_1based")
    r.add_argument("--adv-units", choices=("shares", "dollar"), default="shares")
    r.add_argument("--basis", choices=("raw", "adjusted"), default="raw")
    r.add_argument("--execution", choices=("close", "open"), default="close")
    r.add_argument("--delisting", choices=("last_close", "none", "shumway"), default="last_close")
    r.add_argument("--baseline-run", help="run_id whose alpha_ic is the baseline of the paired table")
    r.add_argument("--raw-dir", default="data/raw", help="saved Wikipedia snapshots (survivorship counts)")
    r.add_argument("--no-db", action="store_true", help="no DB writes and no run registration")
    r.add_argument("--out", default="reports/")
    r.add_argument("--unlock-holdout", action="store_true")
    r.add_argument("--prereg", help="committed pre-registration file (required with --unlock-holdout)")
    r.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    try:
        return args.fn(args)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
