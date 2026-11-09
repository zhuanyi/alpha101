#!/usr/bin/env python3
"""TSX tax-loss rebound study CLI (pre-registered spec, not distributed).

  --build-calendar   taxloss_calendar.csv
  --fetch-holdings   iShares XIC/XCS holdings (cached, resumable) -> etf_holdings_snapshot, holdings_coverage.csv
  --fetch-yahoo      Yahoo top-up + identity check -> yahoo_coverage.csv, coverage_summary.md
  --run              pass 1: price panel, split hits, exit classes, big_moves gate; once big_moves.csv is
                     fully cleared: events, variants, placebos, decision, report. The 2007-2011 M series stays sealed.
  --unmask-m         pass 2 (needs a committed prereg_hash.txt matching the spec's current sha256)
  --calibrate-spreads  November live-quote calibration (code only; needs --quotes)
  --write-db         with --run: backtest_results / research_log rows (only after the gate is clear)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

# The pre-registered spec is not distributed; point TAXLOSS_SPEC at your copy (its sha256 is checked against data/prereg_hash.txt).
SPEC = Path(os.environ.get("TAXLOSS_SPEC", HERE / "data" / "prereg_spec.md"))
DEFAULT_OUT = Path(os.environ.get("TAXLOSS_OUT", "taxloss_results"))
PREREG = HERE / "data" / "prereg_hash.txt"
TAKEOVERS = HERE / "data" / "takeover_sources.csv"
SEED = 20261019
EVENT_YEARS = range(2012, 2026)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_commit() -> str:
    g = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "HEAD") + (" (dirty)" if g("status", "--porcelain", "--", ".") else "")


def setup_log(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.FileHandler(out / "run.log"), logging.StreamHandler()):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("taxloss").info("spec sha256=%s code commit=%s", sha256(SPEC), code_commit())


def build_calendar(out: Path):
    import pandas as pd
    import taxcal
    log = logging.getLogger("taxloss.cal")
    cal = taxcal.Calendar(taxcal.tsx_trading_days())
    df = pd.DataFrame([cal.event_row(y) for y in range(2012, 2027)])
    df.to_csv(out / "taxloss_calendar.csv", index=False)
    for k, urls in taxcal.REGIME_URLS.items():
        log.info("regime source %s: %s", k, "; ".join(urls))
    log.info("Sep 30 non-settlement from 2021: %s", taxcal.SEP30_NOTICE)
    for y, claimed, how, url, quote in taxcal.SETTLEMENT_NOTICES:
        got = cal.last_taxloss_day(y)
        log.info("broker/exchange notice %d (%s): published L=%s, calendar L=%s -> %s | %s | %s", y, how, claimed, got,
                 "MATCH" if got == claimed else "MISMATCH", url, quote)
    log.info("taxloss_calendar.csv: %d rows", len(df))
    return cal, df


def load_inputs(raw_dir: Path, log):
    """Holdings (>= 2011-02-01 only) + recovered placeholder rows, Yahoo frames, takeover sources."""
    import pandas as pd
    import ishares
    import placeholders
    import prices
    from alpha101.db import reader
    conn = reader()
    try:
        snap = ishares.read_snapshots(conn, start=prices.HOLDOUT_FLOOR)
        ph = placeholders.scan_raw(raw_dir, dates=set(snap["as_of"]))
        plog, rec = placeholders.match(ph, snap)
        log.info("placeholder rows: %d scanned, %d matched (recovered), %d unmatched", len(plog),
                 int(plog["accepted"].sum()) if len(plog) else 0, int((~plog["accepted"]).sum()) if len(plog) else 0)
        snap_all = pd.concat([snap, rec.assign(as_of=rec["as_of"])], ignore_index=True)
        syms = sorted(set(snap_all["ysym"]) | {"XIC.TO", "XCS.TO", "^GSPTSE"})
        yahoo = prices.load_yahoo(conn, syms)
    finally:
        conn.close()
    tk = pd.read_csv(TAKEOVERS) if TAKEOVERS.exists() and TAKEOVERS.stat().st_size else pd.DataFrame(
        columns=["year", "ysym", "cash_per_share", "source_url"])
    return snap, snap_all, plog, yahoo, tk


def run_pass1(a, cal) -> int:
    import prices
    log = logging.getLogger("taxloss.run")
    snap, snap_all, plog, yahoo, tk = load_inputs(a.raw_dir, log)
    panel = prices.PricePanel(snap_all, cal, yahoo, takeovers=tk)
    return run_from_panel(panel, cal, a.out_dir, tk, a.seed, plog, snap_all, a.write_db)


def run_from_panel(panel, cal, out: Path, tk, seed: int, plog, snap_all, write_db: bool = False, years=EVENT_YEARS) -> int:
    """Everything after loading: artifacts, the big_moves gate, and (gate clear) results and the report."""
    import numpy as np
    import pandas as pd
    import artifacts as art
    import prices
    import rebound as rb
    log = logging.getLogger("taxloss.run")
    out = Path(out)
    plog.to_csv(out / "placeholder_ticker_matches.csv", index=False)
    art.placeholder_exclusion_weights(plog, snap_all, cal, years).to_csv(out / "placeholder_exclusion_weights.csv", index=False)
    hits = panel.hits.merge(snap_all.drop_duplicates("ysym")[["ysym", "name"]], on="ysym", how="left")
    hits.to_csv(out / "split_hits.csv", index=False)
    for h in hits.itertuples(index=False):
        log.info("SPLIT HIT %s %s..%s ratio=%.3f q_applied=%s trigger=%s applied=%s %s", h.ysym, h.d0, h.d1,
                 h.price_ratio, h.q_applied, h.trigger, h.applied, h.note)
    log.info("split hits: %d (applied %d)", len(hits), int(hits["applied"].sum()) if len(hits) else 0)
    art.december_review_dates(snap_all, cal, years).to_csv(out / "december_review_dates.csv", index=False)
    art.deleted_with_rank(snap_all, panel.hits, cal, years).to_csv(out / "deleted_before_R.csv", index=False)

    raw_study = rb.Study(panel, cal, years=years)
    bm_path = out / "big_moves.csv"
    bm = prices.merge_big_moves(rb.big_moves(raw_study), prices.read_big_moves(bm_path))
    bm.to_csv(bm_path, index=False)
    ec, names = art.exit_class_tables(raw_study, tk)
    ec.to_csv(out / "exit_classes.csv", index=False)
    names.to_csv(out / "exit_class_names.csv", index=False)
    sr = rb.sanity_reproduction(raw_study)
    log.info("sanity reproduction (holdings vs Yahoo window return, P cell): n=%d, median |diff| = %.4f (bar 0.005)", sr["n"], sr["median_abs_diff"])
    art.write_gate_summary(out, ec, bm, hits, plog, pd.read_csv(out / "december_review_dates.csv"), sr, sha256(SPEC), code_commit())
    un = prices.uncleared(bm)
    log.info("big_moves.csv: %d rows, %d uncleared", len(bm), len(un))
    if len(un):
        log.warning("GATE: %d uncleared big_moves row(s); stopping before decide() (no results computed or written)", len(un))
        print(f"GATE: {len(un)} uncleared big_moves.csv row(s) of {len(bm)}; stopped before decide().")
        return 0
    # ----- gate clear: final study with the hand-cleared exclusions -------------------------------------------------
    excl = prices.excluded_names(bm)
    panel.excluded = excl
    study = rb.Study(panel, cal, years=years, excluded=excl) if excl else raw_study
    if excl:
        ec, names = art.exit_class_tables(study, tk)
        ec.to_csv(out / "exit_classes.csv", index=False)
        names.to_csv(out / "exit_class_names.csv", index=False)
    cell_tables = {c.id: study.cell_table(c) for c in rb.CELLS}
    p = cell_tables["P"]
    p.to_csv(out / "events_P.csv", index=False)
    var = pd.concat([t for k, t in cell_tables.items() if k != "P"], ignore_index=True)
    var.to_csv(out / "variants.csv", index=False)
    rb.deciles(study).to_csv(out / "deciles.csv", index=False)
    pl = rb.random_placebo(rb.placebo_inputs(study), seed=seed)
    rp = rb.random_placebo_summary(pl, p)
    pd.DataFrame({"draw": np.arange(pl["g_net"].shape[0]), "mean_g_net": pl["g_net"].mean(axis=1),
                  "mean_g": pl["g"].mean(axis=1)}).to_csv(out / "placebo_random.csv", index=False)
    mdf, months = rb.placebo_months(study, p)
    mdf.to_csv(out / "placebo_months.csv", index=False)
    months.to_csv(out / "placebo_months_summary.csv", index=False)
    dec_first = rb.december_first(months)
    rb.m_series(panel, cal, years, excl).to_csv(out / "m_series.csv", index=False)
    (out / "neff.json").write_text(json.dumps(rb.neff(cell_tables), indent=1))
    # scenarios for condition 6 and the sensitivity table
    def scen_ev(scen, book=None):
        t = study.cell_table(rb.CELLS[0], scen)
        pls = rb.random_placebo(rb.placebo_inputs(study, scen), seed=seed)
        return t, rb.evidence_from(t, rb.random_placebo_summary(pls, t), dec_first)
    wipe_t, wipe_e = scen_ev(rb.Scenario("wipe", unresolved="wipe"))
    e2h_t, e2h_e = scen_ev(rb.Scenario("e2h", e2h=True))
    big_t, big_e = scen_ev(rb.Scenario("book250k", book=250_000.0))
    base_e = rb.evidence_from(p, rp, dec_first)
    decision = rb.decide(base_e, {"wipe": wipe_e, "e2h": e2h_e}, ev250=big_e, big_moves_path=bm_path)
    (out / "decision.json").write_text(json.dumps(decision, indent=1, default=str))
    sens_rows = [("base", p)]
    for scen in (rb.Scenario("commission_7", commission=7.0), rb.Scenario("h_x2", hmult=2.0), rb.Scenario("s_close_0", s_close_bp=0.0),
                 rb.Scenario("s_close_25", s_close_bp=25.0), rb.Scenario("unresolved_0pct", unresolved="zero")):
        sens_rows.append((scen.name, study.cell_table(rb.CELLS[0], scen)))
    sens_rows += [("unresolved_-100pct", wipe_t), ("E2h", e2h_t), ("book_250k", big_t)]
    sens = pd.DataFrame([{"scenario": n, "mean_g_net": t["g_net"].mean(), "t_dec_net": rb.t_dec(t["g_net"]),
                          "mean_g": t["g"].mean(), "t_dec_gross": rb.t_dec(t["g"])} for n, t in sens_rows])
    stats = rb.stats_block(p)
    vs = pd.DataFrame([{"cell": k, "mean_g": t["g"].mean(), "mean_g_net": t["g_net"].mean(), "t_dec_net": rb.t_dec(t["g_net"]),
                        "events": int(t["g"].notna().sum())} for k, t in cell_tables.items()])
    art.plot_events(p, pl, out / "events_P.png")
    art.plot_months(months, out / "december_vs_placebo_months.png")
    um = art.unresolved_mean_share(ec)
    issues = data_issue_notes(ec, out, plog)
    art.write_report(out, {"events_P": p, "decision": decision, "stats": stats, "random_placebo": rp, "months": months,
                           "dec_first": dec_first, "variants_summary": vs, "neff": rb.neff(cell_tables), "sens": sens,
                           "unresolved": um, "n_split_hits": len(hits), "n_split_applied": int(hits["applied"].sum()) if len(hits) else 0,
                           "n_big_moves": len(bm), "spec_sha": sha256(SPEC), "commit": code_commit(), "seed": seed,
                           "data_issues": issues})
    log.info("pass 1 complete: verdict %s", decision["verdict"])
    if write_db:
        from alpha101.db import writer
        conn = writer()
        try:
            ids = art.write_db(conn, sha256(SPEC), cell_tables, {"seed": seed, "costs": {"commission": 9.99, "s_close_bp": 10}},
                               str(out / "results.md"))
        finally:
            conn.close()
        log.info("backtest_results ids %s", ids)
    return 0


def data_issue_notes(ec, out: Path, plog) -> list[str]:
    import pandas as pd
    notes = []
    miss = ec[(ec["scope"] == "basket") & (~ec["exit_file_present"] | ~ec["entry_file_present"])]
    for r in miss.itertuples(index=False):
        notes.append(f"issue 1: {r.year} {r.cell}: entry/exit holdings file missing for both ETFs; basket weight via Yahoo "
                     f"{r.share_yahoo:.0%}, unresolved {r.share_unresolved:.0%}, no entry {r.share_no_entry:.0%}")
    dr = pd.read_csv(out / "december_review_dates.csv")
    bad = dr[dr["status"].str.startswith("MISMATCH") | dr["status"].str.startswith("unverifiable")]
    notes.append(f"issue 2: December review check: {len(dr)} (year, ETF) rows, {len(bad)} not confirming a change on the assumed date "
                 f"(see december_review_dates.csv); the rule is unchanged")
    if len(plog):
        notes.append(f"issue 3: placeholder '-' rows: {len(plog)} scanned, {int(plog['accepted'].sum())} matched by name to a real ticker, "
                     f"the rest stay excluded (placeholder_exclusion_weights.csv)")
    return notes


def calibrate_spreads(quotes_csv: Path, out: Path) -> int:
    """November calibration (spec 4): ratio of quoted to estimated half-spread by bucket. Input CSV columns: ysym, bid,
    ask, h_est, bucket. Code and test only until November."""
    import pandas as pd
    q = pd.read_csv(quotes_csv)
    q["h_quoted"] = (q["ask"] - q["bid"]) / (q["ask"] + q["bid"])
    q["ratio"] = q["h_quoted"] / q["h_est"]
    t = q.groupby("bucket")["ratio"].agg(["median", "size"]).reset_index()
    t["exceeds_1_5"] = t["median"] > 1.5
    t.to_csv(out / "spread_calibration.csv", index=False)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-calendar", action="store_true")
    ap.add_argument("--fetch-holdings", action="store_true")
    ap.add_argument("--fetch-yahoo", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--unmask-m", action="store_true")
    ap.add_argument("--calibrate-spreads", action="store_true")
    ap.add_argument("--quotes", type=Path)
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--raw-dir", type=Path, default=None, help="iShares raw cache (default: research/taxloss/data/raw/ishares)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if not (a.build_calendar or a.fetch_holdings or a.fetch_yahoo or a.run or a.unmask_m or a.calibrate_spreads):
        ap.error("nothing to do")
    if a.unmask_m:
        import artifacts as art
        try:
            art.check_unmask_allowed(SPEC, PREREG)
        except PermissionError as e:
            print(f"--unmask-m refused: {e}", file=sys.stderr)
            return 2
        print("--unmask-m: pass 2 is not part of this build", file=sys.stderr)
        return 2
    if a.calibrate_spreads:
        if not a.quotes:
            ap.error("--calibrate-spreads needs --quotes")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        return calibrate_spreads(a.quotes, a.out_dir)
    if a.raw_dir is None:
        import ishares
        a.raw_dir = ishares.RAW_DIR
    setup_log(a.out_dir)
    cal = None
    if a.build_calendar or a.fetch_holdings or a.fetch_yahoo or a.run:
        cal, _ = build_calendar(a.out_dir)
    if a.fetch_holdings:
        import ishares
        log = logging.getLogger("taxloss.holdings")
        status = ishares.fetch_all(ishares.request_plan(cal))
        status.to_csv(a.out_dir / "holdings_dates.csv", index=False)
        ishares.coverage(status).to_csv(a.out_dir / "holdings_coverage.csv", index=False)
        if status.attrs.get("blocked"):
            log.error("BLOCKED by BlackRock: %s; coverage recorded as far as it got", status.attrs["blocked"])
        res = ishares.load_to_db(status)
        res["exclusions"].groupby(["ticker", "name", "reason"]).agg(
            files=("as_of", "size"), first=("as_of", "min"), last=("as_of", "max")).reset_index().to_csv(
            a.out_dir / "ticker_exclusions.csv", index=False)
        log.info("etf_holdings_snapshot: upserted %d rows from %d files", res["rows"], res["files"])
        from alpha101.db import reader
        conn = reader()
        try:
            snap = ishares.read_snapshots(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT etf, COUNT(*) FROM etf_holdings_snapshot GROUP BY etf")
                log.info("etf_holdings_snapshot rows in DB: %s", dict(cur.fetchall()))
        finally:
            conn.close()
        dele, summ = ishares.deleted_before_R(snap, cal)
        dele.to_csv(a.out_dir / "deleted_before_R.csv", index=False)
        summ.to_csv(a.out_dir / "december_review_check.csv", index=False)
        log.info("deleted_before_R.csv: %d rows (no YTD rank column: ranks are results)", len(dele))
    if a.fetch_yahoo:
        import yahootopup
        yahootopup.run(cal, a.out_dir)
    if a.run:
        return run_pass1(a, cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
