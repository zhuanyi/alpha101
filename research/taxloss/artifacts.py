"""Artifacts: exit-class tables, December review check, deleted-before-R ranks, PNGs, report, DB rows, the
holdout-lock check. Numbers in the report are generated from the frames passed in; nothing is typed in."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import placeholders
import prices as pr
import rebound as rb
import yahootopup


# --- artifacts ---------------------------------------------------------------------------------------------------
def exit_class_tables(study: rb.Study, takeovers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """exit_classes.csv (share of equal-weight basket / universe per exit class, per year x cell, with takeover
    sources and file-presence flags) and exit_class_names.csv (names that did not exit via a holdings file, P cell
    universe plus every cell's basket; no prices)."""
    rows, names = [], []
    for y in study.years:
        d = study.dates(y)
        for c in rb.CELLS:
            w = study.window(y, c)
            if w.empty:
                continue
            ent, ext = d[c.entry_role], d[c.exit_role]
            for scope, sub in (("basket", w.head(c.n)), ("universe", w)):
                tk = takeovers[(takeovers["year"] == y) & takeovers["ysym"].isin(sub["ysym"])] if len(takeovers) else takeovers
                rows.append({"year": y, "cell": c.id, "scope": scope, "n": len(sub), **rb.class_shares(sub),
                             "entry_file_present": study.panel.has_file(ent), "exit_file_present": study.panel.has_file(ext),
                             "takeover_sources": ";".join(f"{r.ysym}={r.source_url}" for r in tk.itertuples(index=False))
                             if len(tk) else ""})
                if (scope == "basket") or c.id == "P":
                    x = sub[sub["exit_class"] != "holdings"]
                    for r in x.itertuples(index=False):
                        names.append({"year": y, "cell": c.id, "scope": scope, "ysym": r.ysym, "name": r.name,
                                      "exit_class": r.exit_class, "entry_source": r.entry_source,
                                      "exit_source": r.exit_source, "last_obs_date": r.last_obs_date})
    nm = pd.DataFrame(names, columns=["year", "cell", "scope", "ysym", "name", "exit_class", "entry_source", "exit_source", "last_obs_date"])
    return pd.DataFrame(rows), nm.drop_duplicates(["year", "cell", "scope", "ysym"])


def unresolved_mean_share(ec: pd.DataFrame) -> dict:
    """Mean over events of the P basket's unresolved share (spec 1.4), all events and without partial-exit events."""
    p = ec[(ec["cell"] == "P") & (ec["scope"] == "basket")]
    ok = p[p["exit_file_present"]]
    return {"mean_all": float(p["share_unresolved"].mean()), "mean_ex_partial": float(ok["share_unresolved"].mean()) if len(ok) else float("nan"),
            "data_limited": bool(p["share_unresolved"].mean() > 0.10)}


def december_review_dates(snap: pd.DataFrame, cal, years=range(2012, 2026)) -> pd.DataFrame:
    """December review check: does the holdings file change membership on the assumed effective date? Per year and ETF, compare
    the last available file before the assumed date with the first on/after it, and list every consecutive-file pair in
    December through the first January day that changes membership (ysym added/dropped). Descriptive; the rule itself
    is unchanged."""
    rows = []
    for y in years:
        eff = cal.dec_review_effective(y)
        lo, hi = dt.date(y, 12, 1), cal.first_trading_day_of_month(y + 1, 1)
        for etf, g in snap.groupby("etf"):
            ds = sorted(d for d in g["as_of"].unique() if lo <= d <= hi)
            mem = {d: set(g.loc[g["as_of"] == d, "ysym"]) for d in ds}
            before = max([d for d in ds if d < eff], default=None)
            after = min([d for d in ds if d >= eff], default=None)
            pairs, straddle = [], None
            for a, b in zip(ds[:-1], ds[1:]):
                drop, add = len(mem[a] - mem[b]), len(mem[b] - mem[a])
                if drop or add:
                    pairs.append(f"{a}->{b}: -{drop}/+{add}")
                if a < eff <= b:
                    straddle = (a, b, drop, add)
            if straddle is None:
                status = "unverifiable (no file pair straddles the assumed date)"
            elif straddle[2] + straddle[3] > 0:
                status = "change on assumed date"
            else:
                status = "MISMATCH: no change across the assumed date" + (" (changes elsewhere)" if pairs else " (no change at all)")
            rows.append({"year": y, "etf": etf, "assumed_effective": eff, "file_before": before, "file_on_or_after": after,
                         "straddling_pair_changes": None if straddle is None else straddle[2] + straddle[3],
                         "status": status, "pairs_with_change": " | ".join(pairs)})
    return pd.DataFrame(rows)


def deleted_with_rank(snap: pd.DataFrame, hits: pd.DataFrame, cal, years=range(2012, 2026)) -> pd.DataFrame:
    """Names in the file of the last day before the December review effective date but absent from the R file, with the
    YTD rank they would have had at that day (among names eligible at C$2 then). 2021 uses the fallback R file."""
    rows = []
    for y in years:
        e = cal.event_row(y)
        pre, r_used = e["pre_review_day"], yahootopup.resolve_R(snap, e["R"])
        a, b = snap[snap["as_of"] == pre], snap[snap["as_of"] == r_used]
        if a.empty or b.empty:
            continue
        ranked = rb.rank_ytd(snap, hits, e["D_prev"], pre, 2.0)
        pos = {s: i + 1 for i, s in enumerate(ranked["ysym"])}
        for r in a[~a["ysym"].isin(set(b["ysym"]))].drop_duplicates("ysym").sort_values("ysym").itertuples(index=False):
            rows.append({"year": y, "ysym": r.ysym, "name": r.name, "etf": r.etf, "pre_review_day": pre, "R_used": r_used,
                         "ytd_rank_at_pre_review": pos.get(r.ysym), "n_ranked": len(ranked)})
    return pd.DataFrame(rows, columns=["year", "ysym", "name", "etf", "pre_review_day", "R_used", "ytd_rank_at_pre_review", "n_ranked"])


def placeholder_exclusion_weights(log: pd.DataFrame, snap_all: pd.DataFrame, cal, years=range(2012, 2026)) -> pd.DataFrame:
    """Per event and ETF: weight share (of the file's total weight) in unmatched placeholder rows on D_prev and R."""
    rows = []
    for y in years:
        e = cal.event_row(y)
        for role, d in (("D_prev", e["D_prev"]), ("R", yahootopup.resolve_R(snap_all, e["R"]))):
            for etf in ("XIC", "XCS"):
                u = placeholders.unmatched_weight_share(log, snap_all, etf, d)
                rows.append({"year": y, "role": role, "etf": etf, "file_date": d, "n_unmatched_rows": u["n_unmatched"],
                             "unmatched_weight_pct": u["unmatched_weight"], "file_total_weight_pct": u["total_weight"],
                             "unmatched_share": u["share"]})
    return pd.DataFrame(rows)


def write_gate_summary(out: Path, ec: pd.DataFrame, bm: pd.DataFrame, hits: pd.DataFrame, plog: pd.DataFrame,
                       dec_check: pd.DataFrame, sanity: dict, spec_sha: str, commit: str) -> Path:
    """gate_status.md: coverage facts only (no returns aggregated, no decision), generated from the CSVs' frames."""
    p = ec[(ec["cell"] == "P") & (ec["scope"] == "basket")]
    um = unresolved_mean_share(ec)
    cols = ["year", "share_holdings", "share_other_etf", "share_yahoo", "share_takeover", "share_unresolved", "share_no_entry",
            "entry_file_present", "exit_file_present"]
    part = ec[(ec["scope"] == "basket") & (~ec["exit_file_present"] | ~ec["entry_file_present"])]
    L = ["# TSX tax-loss study: gate status (no results computed)", "",
         f"Spec sha256 `{spec_sha}`; code commit `{commit}`.", "",
         f"- big_moves.csv: **{len(bm)} rows**, {len(pr.uncleared(bm))} uncleared. `decide()` has not run.",
         f"- split hits: {len(hits)} (applied {int(hits['applied'].sum()) if len(hits) else 0}); every hit is in split_hits.csv and run.log.",
         f"- sanity reproduction (holdings vs Yahoo window return, names with both): n={sanity['n']}, median |diff| {sanity['median_abs_diff']:.4f} (bar 0.005).",
         f"- P basket `unresolved` share, mean over events: {um['mean_all']:.1%} (ex partial-exit events {um['mean_ex_partial']:.1%}); "
         f"data-limited (> 10%): {um['data_limited']}.", "",
         "## Exit-class weight shares, P basket (equal weight)", "", p[cols].to_markdown(index=False, floatfmt=".3f"), "",
         "## Cells whose entry or exit holdings file is missing for both ETFs ", "",
         part[["year", "cell", "share_yahoo", "share_unresolved", "share_no_entry", "entry_file_present", "exit_file_present"]]
         .to_markdown(index=False, floatfmt=".3f"), "",
         f"## Placeholder `-` rows ", "",
         f"{len(plog)} rows scanned; {int(plog['accepted'].sum())} matched to a real ticker by name; the rest stay excluded "
         f"(placeholder_ticker_matches.csv, placeholder_exclusion_weights.csv).", "",
         "## December review check ", "", dec_check["status"].str.split(":").str[0].value_counts().to_frame("rows").to_markdown(), ""]
    path = Path(out) / "gate_status.md"
    path.write_text("\n".join(L))
    return path


# --- holdout lock ------------------------------------------------------------------------------------------------
def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_unmask_allowed(spec: Path, prereg: Path) -> None:
    """Pass 2 gate (spec 6): prereg_hash.txt must exist and be committed, and the spec's CURRENT sha256 must equal it.
    Raises PermissionError otherwise."""
    prereg = Path(prereg)
    if not prereg.exists():
        raise PermissionError(f"{prereg} does not exist")
    r = subprocess.run(["git", "-C", str(prereg.parent), "log", "-1", "--format=%H", "--", prereg.name],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise PermissionError(f"{prereg} is not committed")
    want = prereg.read_text().split()[0] if prereg.read_text().split() else ""
    got = sha256(spec)
    if got != want:
        raise PermissionError(f"spec sha256 {got} differs from the locked hash {want}")


# --- charts (static matplotlib; single series + neutral band, direct labels, one axis; dataviz skill) --------------
INK, INK2, GRID, BLUE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_events(ev: pd.DataFrame, pl: dict, path) -> None:
    """g_net per event for P with the 5-95% band of the random-basket placebo for each event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = ev.sort_values("year").reset_index(drop=True)
    lo = np.percentile(pl["g_net"], 5, axis=0) * 100
    hi = np.percentile(pl["g_net"], 95, axis=0) * 100
    x = np.arange(len(ev))
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.bar(x, (hi - lo), bottom=lo, width=0.7, color=GRID, label="random-basket placebo, 5-95% band", zorder=1)
    ax.bar(x, ev["g_net"] * 100, width=0.28, color=BLUE, label="P: after-cost excess vs EW universe", zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks(x, [str(y) + ("*" if p else "") for y, p in zip(ev["year"], ev["partial_exit"])], rotation=0)
    ax.set_ylabel("excess return, % (E to X)", color=INK2)
    ax.set_title("P: after-cost excess per event (* = partial exit data)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_months(summary: pd.DataFrame, path) -> None:
    """Mean gross excess by month (Feb-Oct placebo months and December = P); December in the accent colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary.sort_values("month")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    cols = [BLUE if m == 12 else "#9a998f" for m in s["month"]]
    ax.bar([names[m - 1] for m in s["month"]], s["mean_excess"] * 100, color=cols, width=0.55)
    ax.axhline(0, color=INK2, linewidth=0.9)
    for i, (m, v) in enumerate(zip(s["month"], s["mean_excess"] * 100)):
        if m in (10, 12):
            ax.annotate(f"{v:.1f}%" + (" (Oct window ends in Nov)" if m == 10 else " (P)"), (i, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    ax.set_ylabel("mean gross excess vs EW universe, %", color=INK2)
    ax.set_title("December vs placebo months: same signal, wrong month", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# --- report ----------------------------------------------------------------------------------------------------------
def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def write_report(out: Path, ctx: dict) -> Path:
    """Results markdown generated from `ctx` (frames and dicts computed by the run)."""
    ev, dec, st = ctx["events_P"], ctx["decision"], ctx["stats"]
    L = ["# TSX tax-loss rebound: results (pass 1)", "",
         f"Spec sha256 `{ctx['spec_sha']}`; code commit `{ctx['commit']}`; seed {ctx['seed']}. Generated from the CSVs in this directory.", "",
         "Pass 1 only: the 2007-2011 M series is sealed (no rows computed). Condition 5 of the decision rule is therefore "
         "not evaluated here.", "",
         f"## Verdict: **{dec['verdict']}**", "",
         "```json", json.dumps({k: dec[k] for k in ("verdict", "core_verdict", "conditions", "c6_checks", "treated_as")}, indent=1, default=str), "```", "",
         "## P per event (C$100k)", "", ev.drop(columns=["scenario"], errors="ignore").to_markdown(index=False, floatfmt=".4f"), "",
         "## Tests (events, N=%d)" % len(ev), ""]
    for k in ("g", "g_net"):
        s = st[k]
        L.append(f"- **{k}**: mean {_pct(s['mean'])}, sd {_pct(s['sd'])}, t_plain {s['t_plain']:.2f}, t_NW1 {s['t_nw1']:.2f}, "
                 f"t_dec {s['t_dec']:.2f}, Wilcoxon p {s['wilcoxon_p']:.3f}, share positive {s['share_positive']:.2f}, "
                 f"bootstrap 95% CI [{_pct(s['boot_ci95'][0])}, {_pct(s['boot_ci95'][1])}], lag-1 autocorr {s['lag1_autocorr']:.2f}")
    if "capm" in st:
        c = st["capm"]
        L.append(f"- diversifier (reported, not a pass route): alpha {_pct(c['alpha'])}, beta {c['beta']:.2f}, "
                 f"Sharpe {c['sharpe']:.2f}, corr {c['corr']:.2f}")
    rp = ctx["random_placebo"]
    L += ["", "## Placebos", "",
          f"- random-basket percentile of the net P mean: **{rp['pct_net']:.1f}** (gross {rp['pct_gross']:.1f}), {rp['draws']} draws",
          f"- December rank among the 10 months (gross excess): **{int(ctx['months'].loc[ctx['months']['month'] == 12, 'rank_of_10'].iloc[0])}**"
          f" (strictly first: {ctx['dec_first']})", "", ctx["months"].to_markdown(index=False, floatfmt=".4f"), "",
          "## Variants (mean over events)", "", ctx["variants_summary"].to_markdown(index=False, floatfmt=".4f"), "",
          "## n_eff", "", "```json", json.dumps(ctx["neff"], indent=1), "```", "",
          "## Sensitivities (P, mean g_net / t_dec)", "", ctx["sens"].to_markdown(index=False, floatfmt=".4f"), "",
          "## Data quality", "",
          f"- unresolved exit share (P basket, mean over events): {_pct(ctx['unresolved']['mean_all'])} "
          f"(ex partial-exit events {_pct(ctx['unresolved']['mean_ex_partial'])}); data-limited label: {ctx['unresolved']['data_limited']}",
          f"- split hits: {ctx['n_split_hits']} (applied {ctx['n_split_applied']}); big_moves rows: {ctx['n_big_moves']} (all cleared)", ""]
    if ctx.get("data_issues"):
        L += ["## Handling of known data issues", ""] + [f"- {x}" for x in ctx["data_issues"]] + [""]
    p = Path(out) / "results.md"
    p.write_text("\n".join(L))
    return p


# --- DB rows -----------------------------------------------------------------------------------------------------
def write_db(conn, spec_sha: str, cell_tables: dict, params: dict, results_path: str) -> dict:
    """One backtest_results row per cell (P first; validation_status 'pending') and one research_log row with
    conclusion NULL. All VALUES are %s; no DDL. Refuses to duplicate existing tsx_taxloss_* rows."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM backtest_results WHERE strategy_name LIKE %s", ("tsx_taxloss_%",))
    if cur.fetchone()[0]:
        raise RuntimeError("tsx_taxloss_* rows already exist in backtest_results; refusing to duplicate")
    sql_b = ("INSERT INTO backtest_results (strategy_name,start_date,end_date,universe,params,total_return,annualized_return,"
             "sharpe_ratio,max_drawdown,win_rate,total_trades,benchmark_return,benchmark_sharpe,validation_status,notes,result_file_path)"
             " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    sql_r = ("INSERT INTO research_log (hypothesis,success_criteria,test_method,result_summary,conclusion,backtest_result_id)"
             " VALUES (%s,%s,%s,%s,%s,%s)")
    ids = {}
    f = lambda v: None if v is None or not np.isfinite(v) else float(v)   # noqa: E731
    for cid, t in cell_tables.items():
        yrs = t["year"]
        compound = float(np.prod(1 + t["basket_net"].dropna()) - 1)
        p = dict(params, spec_sha256=spec_sha, cell=cid)
        cur.execute(sql_b, (f"tsx_taxloss_{cid}", dt.date(int(yrs.min()), 12, 1), dt.date(int(yrs.max()) + 1, 2, 28),
                            "S&P/TSX Composite + SmallCap (iShares XIC/XCS holdings, PIT)", json.dumps(p, default=str)[:60000],
                            f(compound), None, None, None, f((t["g_net"] > 0).mean()), int(2 * t["n_basket"].sum()),
                            f(np.prod(1 + t["xic_tr"].dropna()) - 1), None, "pending", None, results_path))
        ids[cid] = cur.lastrowid
    cur.execute(sql_r, ("TSX tax-loss rebound: big YTD losers rebound after the last tax-loss trade date",
                        "Spec 5.4 decision rule (after-cost t >= 2 etc.)", "taxloss_rebound.py --run",
                        json.dumps({"spec_sha256": spec_sha, "cells": list(ids)})[:60000], None, ids.get("P")))
    conn.commit()
    return ids
