"""Slowed-down long-only test : selection, backtests and CSV. In-sample only, holdout never unlocked.

  select  top-10 alphas per market by |NW t| of the persisted daily rank IC (default run)
  run     long-only n_long=20 backtests at W-FRI / M / Q rebalances, flat C$9.99 per ticket
Usage: PYTHONPATH=src python3 scripts/slow_long_only.py select|run [market ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alpha101.stats import newey_west_t

BASE_RUN = "20261019T2057Z-78fe86b"      # `default` run (alpha_ic persisted in alpha101)
HOLDOUT = "2019-01-01"
MARKETS = ("us", "ca")
HORIZONS = (5, 21)
OUT = Path("docs")
SEL_JSON = OUT / "slow_long_only_selection.json"
TMP = Path("reports")                     # git-ignored intermediates


def select() -> dict:
    from alpha101.db import reader
    sel = {}
    with reader() as conn:
        for h in HORIZONS:
            with conn.cursor() as cur:
                cur.execute("SELECT alpha_id, market, date, ic FROM alpha_ic WHERE run_id=%s AND horizon=%s "
                            "AND date < %s", [BASE_RUN, h, HOLDOUT])
                rows = list(cur.fetchall())
            df = pd.DataFrame(rows, columns=["alpha_id", "market", "date", "ic"])
            df["ic"] = pd.to_numeric(df["ic"], errors="coerce").astype(float)
            assert df["date"].max() < pd.Timestamp(HOLDOUT).date()
            for (m, a), g in df.groupby(["market", "alpha_id"]):
                x = g.sort_values("date")["ic"].dropna()
                sel.setdefault(m.lower(), {}).setdefault(str(h), []).append(
                    {"alpha_id": a, "mean_ic": float(x.mean()), "t_nw": newey_west_t(x, lags=h), "n_days": len(x)})
    for m in sel:
        for h in sel[m]:
            sel[m][h] = sorted(sel[m][h], key=lambda r: -abs(r["t_nw"]))
    SEL_JSON.write_text(json.dumps(sel, indent=1))
    return sel


NL = 20
BOOKS = (100_000.0, 250_000.0)
REBALS = ("W-FRI", "M", "Q")            # pandas 3.0 Period aliases; "ME"/"QE" are rejected by to_period
FEE = 9.99
NW_LAGS = 21


def _ann(x: pd.Series) -> float:
    return float(x.mean() * 252)


def row_metrics(out: pd.DataFrame, book: float) -> dict:
    """In-sample metrics of one composite_backtest frame (arithmetic annualisation, 252 days)."""
    d = out.loc[out.index < pd.Timestamp(HOLDOUT)]
    ok = d[["gross", "net", "benchmark"]].notna().all(axis=1)
    d = d.loc[ok]
    n = len(d)
    years = n / 252
    act = d["net"] - d["benchmark"]
    mu = float(act.mean())
    t = newey_west_t(act, lags=NW_LAGS)
    se = abs(mu / t) if t and np.isfinite(t) and t != 0 else float("nan")
    tickets_py = float(d["n_trades"].sum() / years)
    comm_py = tickets_py * FEE / book            # flat-fee drag per year, fraction of book
    return {"days": n, "gross_ann": _ann(d["gross"]), "net_ann": _ann(d["net"]), "bench_ann": _ann(d["benchmark"]),
            "active_ann": mu * 252, "active_t": t, "active_lo": (mu - 1.96 * se) * 252,
            "active_hi": (mu + 1.96 * se) * 252, "turnover_oneway": float(d["turnover"].mean()),
            "trades_per_year": tickets_py, "commission_bps_per_year": comm_py * 1e4,
            "breakeven_fee": FEE + mu * 252 / (tickets_py / book) if tickets_py > 0 else float("nan")}


def run(markets) -> None:
    from alpha101.costs import CostModel
    from alpha101.data import universe
    from alpha101.data.cache import load_panel
    from alpha101.db import reader
    from alpha101.evaluate import RunConfig, composite_backtest
    sel = json.loads(SEL_JSON.read_text())
    cfg = RunConfig()
    assert not cfg.unlock_holdout
    for m in markets:
        with reader() as conn:
            source = universe.resolve_source(conn, universe.liquidity_source(m))
        print(f"[{m}] universe {source}", flush=True)
        panel = load_panel(m, cfg.start, cfg.basis, universe_source=source)
        top = sel[m]["5"][:10]
        ids = {int(r["alpha_id"][5:]): (1.0 if r["mean_ic"] > 0 else -1.0) for r in top}
        strategies = {f"alpha{i:03d}": ([i], {i: w}) for i, w in ids.items()}
        strategies["composite10"] = (list(ids), ids)
        rows = []
        for name, (mem, w) in strategies.items():
            for reb in REBALS:
                for book in BOOKS:
                    if book != BOOKS[0] and name != "composite10" and SKIP_BIG:
                        continue
                    cm = CostModel(commission=FEE, notional_per_name=book / NL)
                    t0 = time.time()
                    try:
                        out = composite_backtest(panel, mem, w, cfg, cm, book="long_only", n_long=NL,
                                                 rebalance=reb, market=m)
                        r = {"market": m, "strategy": name, "rebalance": reb, "book": int(book), "error": "",
                             **row_metrics(out, book)}
                    except Exception as e:  # noqa: BLE001
                        r = {"market": m, "strategy": name, "rebalance": reb, "book": int(book),
                             "error": f"{type(e).__name__}: {e}"}
                    rows.append(r)
                    print(f"[{m}] {name} {reb} {int(book)} {time.time() - t0:.1f}s {r['error']}", flush=True)
        pd.DataFrame(rows).to_csv(TMP / f"slow_long_only_{m}.csv", index=False)


SKIP_BIG = False


def check_fee_accounting(m: str) -> list[str]:
    """Library pro-rata commission vs exact flat $9.99 per ticket, on the composite (book C$100k)."""
    from alpha101.costs import CostModel
    from alpha101.data import universe
    from alpha101.data.cache import load_panel
    from alpha101.db import reader
    from alpha101.evaluate import RunConfig, composite_backtest
    sel = json.loads(SEL_JSON.read_text())
    ids = {int(r["alpha_id"][5:]): (1.0 if r["mean_ic"] > 0 else -1.0) for r in sel[m]["5"][:10]}
    cfg = RunConfig()
    with reader() as conn:
        source = universe.resolve_source(conn, universe.liquidity_source(m))
    panel = load_panel(m, cfg.start, cfg.basis, universe_source=source)
    lines = []
    for reb in REBALS:
        kw = dict(book="long_only", n_long=NL, rebalance=reb, market=m)
        a = composite_backtest(panel, list(ids), ids, cfg, CostModel(commission=FEE, notional_per_name=5000.0), **kw)
        b = composite_backtest(panel, list(ids), ids, cfg, CostModel(commission=0.0, notional_per_name=5000.0), **kw)
        lib, flat = fee_drag(a, b, 100_000.0)
        lines.append(f"| {m.upper()} | {reb} | {lib * 1e4:.1f} | {flat * 1e4:.1f} | {(lib / flat - 1) * 100:+.1f}% |")
    return lines


def fee_drag(with_fee: pd.DataFrame, no_fee: pd.DataFrame, book: float) -> tuple[float, float]:
    """(library commission drag, tickets x 9.99 / book) per year, on the rows `row_metrics` keeps.

    Tickets on a dropped row (the last in-sample rebalance, whose book only earns in the NaN straddle
    band) are never charged by the library, so they are not counted on the flat side either."""
    d = with_fee.loc[with_fee.index < pd.Timestamp(HOLDOUT)]
    d = d.loc[d[["gross", "net", "benchmark"]].notna().all(axis=1)]
    years = len(d) / 252
    lib = float((no_fee["net"] - with_fee["net"]).loc[d.index].sum() / years)
    flat = float(d["n_trades"].sum() * FEE / book / years)
    return lib, flat


def _fmt(r, col, pct=True, nd=2):
    v = r[col]
    return "n/a" if pd.isna(v) else (f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}")


def _table(df: pd.DataFrame) -> list[str]:
    out = ["| market | strategy | rebal | book | gross | net | bench | active | NW t | 95% CI | turnover | tickets/yr "
           "| comm bps/yr | break-even $/ticket |", "|" + "---|" * 14]
    for _, r in df.iterrows():
        out.append(f"| {r['market'].upper()} | {r['strategy']} | {r['rebalance']} | {int(r['book']) // 1000}k "
                   f"| {_fmt(r, 'gross_ann')} | {_fmt(r, 'net_ann')} | {_fmt(r, 'bench_ann')} | {_fmt(r, 'active_ann')} "
                   f"| {_fmt(r, 'active_t', False)} | [{_fmt(r, 'active_lo')}, {_fmt(r, 'active_hi')}] "
                   f"| {_fmt(r, 'turnover_oneway', False)} | {_fmt(r, 'trades_per_year', False, 0)} "
                   f"| {_fmt(r, 'commission_bps_per_year', False, 0)} | {_fmt(r, 'breakeven_fee', False, 1)} |")
    return out


def report(planned: int) -> None:
    sel = json.loads(SEL_JSON.read_text())
    df = pd.concat([pd.read_csv(TMP / f"slow_long_only_{m}.csv") for m in MARKETS], ignore_index=True)
    df.to_csv(OUT / "slow_long_only.csv", index=False)
    done = df[df["error"].fillna("") == ""]
    errs = df[df["error"].fillna("") != ""]
    win = done[(done["active_ann"] > 0) & (done["active_t"] > 2)]
    L = [f"Backtests: {len(done)} completed of {planned} planned; {len(errs)} failed."]
    for _, r in errs.iterrows():
        L.append(f"- FAILED {r['market']} {r['strategy']} {r['rebalance']} {r['book']}: {r['error']}")
    L += ["", f"**Rows with active > 0 and NW t > 2 after costs: {len(win)} of {len(done)}.**",
          f"Rows with active > 0 (any t): {int((done['active_ann'] > 0).sum())}."]
    if len(win):
        L += ["", *_table(win.sort_values("active_t", ascending=False))]
    L += ["", "## Headline", "", "Composite (equal-sign-weighted top-10) at each frequency and book size:", "",
          *_table(done[done["strategy"] == "composite10"]), "",
          "Best single alpha per market and frequency by in-sample active NW t (book C$100k):", ""]
    best = (done[(done["strategy"] != "composite10") & (done["book"] == int(BOOKS[0]))]
            .sort_values("active_t", ascending=False).groupby(["market", "rebalance"], sort=False).head(1))
    L += _table(best.sort_values(["market", "rebalance"]))
    L += ["", "## Selection", ""]
    for m in MARKETS:
        for h in HORIZONS:
            L += [f"{m.upper()}, top 10 by |NW t| of daily rank IC at horizon {h}, in-sample "
                  f"(sign = sign of mean IC; weight -1 = long the bottom of the ranking):", "",
                  "| alpha | mean IC | NW t | weight |", "|---|---|---|---|"]
            for r in sel[m][str(h)][:10]:
                L.append(f"| {r['alpha_id']} | {r['mean_ic']:.4f} | {r['t_nw']:.2f} | {1 if r['mean_ic'] > 0 else -1:+d} |")
            L.append("")
        a5 = {r["alpha_id"] for r in sel[m]["5"][:10]}
        a21 = {r["alpha_id"] for r in sel[m]["21"][:10]}
        L += [f"{m.upper()}: {len(a5 & a21)} of the top-10 at h=5 are also in the top-10 at h=21 ({sorted(a5 & a21)}).", ""]
    L += ["## All rows", "", *_table(done)]
    (TMP / "slow_long_only_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:3]))

if __name__ == "__main__":
    if sys.argv[1] == "select":
        s = select()
        for m in s:
            for h in s[m]:
                print(m, h, [(r["alpha_id"], round(r["t_nw"], 2)) for r in s[m][h][:10]])
    elif sys.argv[1] == "check":
        for m in sys.argv[2:] or MARKETS:
            print("\n".join(check_fee_accounting(m)))
    elif sys.argv[1] == "report":
        report(int(sys.argv[2]))
    else:
        run(sys.argv[2:] or list(MARKETS))
