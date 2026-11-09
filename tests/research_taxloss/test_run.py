"""Placeholders, December check, artifacts, the holdout lock and an end-to-end synthetic run (gate, determinism, grep)."""
import datetime as dt
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import artifacts as art
import placeholders as ph
import prices as pr
import rebound as rb
import taxloss_rebound as cli
from synth import make_world, mini_snap, sym

D = dt.date
YEARS = (2012, 2014)


# --- placeholders -----------------------------------------------------------------------------------------------------
def test_norm_name_and_match_by_name_and_price():
    assert ph.norm_name("CAMECO CORP COM") == ph.norm_name("Cameco Corp.") == "CAMECO"
    snap = mini_snap([("XIC", D(2013, 5, 31), "CCO", 22.0, 100.0), ("XIC", D(2014, 12, 31), "CCO", 21.0, 100.0)])
    snap["name"] = "CAMECO CORP"
    p = pd.DataFrame([
        {"etf": "XIC", "as_of": D(2013, 6, 28), "name": "CAMECO CORP COM", "sector": "-", "asset_class": "Equity", "price": 22.4,
         "shares": 10.0, "market_value": 224.0, "weight": 0.5},
        {"etf": "XIC", "as_of": D(2013, 6, 28), "name": "CHORUS AVIATION INC", "sector": "-", "asset_class": "-", "price": 4.0,
         "shares": 10.0, "market_value": 40.0, "weight": 0.4},
        {"etf": "XIC", "as_of": D(2013, 6, 28), "name": "DULUTH METALS LTD WARRANT", "sector": "-", "asset_class": "-", "price": 1.0,
         "shares": 10.0, "market_value": 10.0, "weight": 0.1},
        {"etf": "XIC", "as_of": D(2013, 7, 31), "name": "CAMECO CORP COM", "sector": "-", "asset_class": "Equity", "price": 70.0,
         "shares": 10.0, "market_value": 700.0, "weight": 0.5}])
    log, rec = ph.match(p, snap)
    assert list(log["accepted"]) == [True, False, False, False]
    assert rec.iloc[0]["ticker"] == "CCO" and rec.iloc[0]["ysym"] == "CCO.TO"
    assert "price outside band" in log.iloc[3]["reason"] and "interim/warrant" in log.iloc[2]["reason"]
    snap_all = pd.concat([snap, rec])
    u = ph.unmatched_weight_share(log, snap_all[snap_all["as_of"] == D(2013, 6, 28)], "XIC", D(2013, 6, 28))
    assert u["n_unmatched"] == 2 and u["unmatched_weight"] == pytest.approx(0.5)      # Chorus line + warrant


def test_match_skips_when_ticker_already_in_the_file_and_when_ambiguous():
    snap = mini_snap([("XIC", D(2020, 1, 31), "AA", 10.0, 100.0), ("XIC", D(2020, 3, 31), "AA", 10.0, 100.0)])
    snap["name"] = "ALPHA CORP"
    both = snap.copy()
    both["ticker"], both["ysym"] = "BB", "BB.TO"
    both["as_of"] = [D(2020, 3, 31), D(2020, 5, 29)]
    row = {"etf": "XIC", "as_of": D(2020, 2, 28), "name": "ALPHA CORP", "sector": "-", "asset_class": "Equity", "price": 10.0,
           "shares": 1.0, "market_value": 10.0, "weight": 0.1}
    log, rec = ph.match(pd.DataFrame([row]), snap)
    assert rec.shape[0] == 1
    amb, rec2 = ph.match(pd.DataFrame([row]), pd.concat([snap.iloc[:1], both.iloc[:1].assign(as_of=D(2020, 3, 31))]))
    assert rec2.empty and "ambiguous" in amb.iloc[0]["reason"]
    # same ysym already present in the placeholder's own file
    own = snap.copy()
    own.loc[0, "as_of"] = D(2020, 2, 28)
    log3, rec3 = ph.match(pd.DataFrame([row]), own)
    assert rec3.empty and "already" in log3.iloc[0]["reason"]


def test_scan_file_reads_placeholder_rows(tmp_path):
    f = tmp_path / "XIC" / "20131231.csv"
    f.parent.mkdir()
    f.write_text('Fund Holdings as of,"Dec 31, 2013"\n \nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency\n'
                 '"-","CAMECO CORP COM","-","Equity","6,969,973.68","0.52","-","316,242.00","22.04","Canada","TSX","CAD","1.00","CAD"\n'
                 '"AC","AIR CANADA","Industrials","Equity","2,603,466.45","1.76","-","351,345.00","7.41","Canada","TSX","CAD","1.00","CAD"\n'
                 '"-","USD LINE","-","Equity","1.0","0.0","-","1.0","1.0","US","NYSE","USD","1.00","USD"\n')
    r = ph.scan_raw(tmp_path)
    assert len(r) == 1 and r.iloc[0]["name"] == "CAMECO CORP COM" and r.iloc[0]["price"] == 22.04 and r.iloc[0]["as_of"] == D(2013, 12, 31)


def test_scan_raw_skips_files_that_are_not_dated_snapshots(tmp_path):
    (tmp_path / "XIC").mkdir()
    (tmp_path / "XIC" / "notes.csv").write_text("not a holdings file\n")
    (tmp_path / "XIC" / "20131231.csv").write_text(
        'Fund Holdings as of,"Dec 31, 2013"\n \nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency\n'
        '"-","CAMECO CORP COM","-","Equity","6,969,973.68","0.52","-","316,242.00","22.04","Canada","TSX","CAD","1.00","CAD"\n')
    assert len(ph.scan_raw(tmp_path, dates={D(2013, 12, 31)})) == 1


# --- December review check, exit tables, deleted-before-R ------------------------------------------------------------
@pytest.fixture(scope="module")
def world(generic_cal):
    extra = [generic_cal.last_trading_day_of_month(y, 11) for y in (2011, 2012, 2013, 2014)]
    snap, yd = make_world(generic_cal, years=YEARS, extra_days=extra)
    return snap, yd


def test_december_review_check_statuses(generic_cal, world):
    snap, yd = world
    eff = generic_cal.dec_review_effective(2012)
    s = snap.copy()
    s = s[~((s["ysym"] == sym(30)) & (s["as_of"] >= eff))]            # S30 leaves exactly on the assumed date
    r = art.december_review_dates(s, generic_cal, years=[2012, 2014])
    got = r.set_index(["year", "etf"])["status"]
    assert got[(2012, "XCS")] == "change on assumed date"
    assert got[(2014, "XCS")].startswith("MISMATCH") or got[(2014, "XCS")].startswith("unverifiable")
    assert set(r.columns) >= {"straddling_pair_changes", "pairs_with_change", "assumed_effective"}


def test_deleted_before_r_gets_a_ytd_rank(generic_cal, world):
    snap, yd = world
    e = generic_cal.event_row(2014)
    s = snap[~((snap["ysym"] == sym(2)) & (snap["as_of"] >= e["R"]))]
    d = art.deleted_with_rank(s, pr.detect_splits(s), generic_cal, years=[2014])
    assert list(d["ysym"]) == [sym(2)] and int(d.iloc[0]["ytd_rank_at_pre_review"]) == 3


def test_exit_class_shares_sum_to_one(generic_cal, world):
    snap, yd = world
    # drop S03 from the X file in 2014: Yahoo still has it (as_of dates in the frame) -> yahoo exit
    e = generic_cal.event_row(2014)
    s = snap[~((snap["ysym"] == sym(3)) & (snap["as_of"] == e["jan_exit_X"]))]
    st = rb.Study(pr.PricePanel(s, generic_cal, yd), generic_cal, years=[2014])
    ec, names = art.exit_class_tables(st, pd.DataFrame(columns=["year", "ysym", "cash_per_share", "source_url"]))
    p = ec[(ec["cell"] == "P") & (ec["scope"] == "basket")].iloc[0]
    assert p["share_yahoo"] == pytest.approx(1 / 20) and p["share_holdings"] == pytest.approx(19 / 20)
    cols = [c for c in ec.columns if c.startswith("share_")]
    assert np.allclose(ec[cols].sum(axis=1), 1.0)
    assert list(names[(names["cell"] == "P") & (names["scope"] == "basket")]["ysym"]) == [sym(3)]
    um = art.unresolved_mean_share(ec)
    assert um["mean_all"] == 0.0 and not um["data_limited"]


def test_missing_exit_file_is_flagged_partial_and_goes_through_yahoo(generic_cal, world):
    snap, yd = world
    e = generic_cal.event_row(2014)
    s = snap[snap["as_of"] != e["jan_exit_X"]]
    st = rb.Study(pr.PricePanel(s, generic_cal, yd), generic_cal, years=[2014])
    r = st.cell_event(2014, rb.CELLS[0])
    assert r["partial_exit"] and r["share_yahoo"] == 1.0 and r["basket_gross"] == pytest.approx(0.05)


# --- the holdout lock -------------------------------------------------------------------------------------------------
def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)


def test_unmask_requires_committed_matching_hash(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("frozen spec")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    pre = repo / "prereg_hash.txt"
    with pytest.raises(PermissionError, match="does not exist"):
        art.check_unmask_allowed(spec, pre)
    pre.write_text(f"{art.sha256(spec)}  spec.md\n")
    with pytest.raises(PermissionError, match="not committed"):
        art.check_unmask_allowed(spec, pre)
    git(repo, "add", "prereg_hash.txt")
    git(repo, "commit", "-qm", "lock")
    art.check_unmask_allowed(spec, pre)                                     # committed and matching: allowed
    spec.write_text("frozen spec, edited")
    with pytest.raises(PermissionError, match="differs"):
        art.check_unmask_allowed(spec, pre)


def test_cli_unmask_exits_nonzero_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PREREG", tmp_path / "nope.txt")
    assert cli.main(["--unmask-m", "--out-dir", str(tmp_path)]) != 0
    spec = tmp_path / "spec.md"
    spec.write_text("x")
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "prereg_hash.txt").write_text("0" * 64 + "  spec.md\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "lock")
    monkeypatch.setattr(cli, "PREREG", repo / "prereg_hash.txt")
    monkeypatch.setattr(cli, "SPEC", spec)
    assert cli.main(["--unmask-m", "--out-dir", str(tmp_path)]) != 0         # hash differs from the spec


def test_m_series_refuses_the_sealed_years(generic_cal, world):
    snap, yd = world
    p = pr.PricePanel(snap, generic_cal, yd)
    with pytest.raises(PermissionError):
        rb.m_series(p, generic_cal, [2011, 2012])
    m = rb.m_series(p, generic_cal, [2012])
    assert list(m["year"]) == [2012] and m.iloc[0]["n_universe"] == 40


def test_calibrate_spreads_ratio_by_bucket(tmp_path):
    q = pd.DataFrame({"ysym": list("abcd"), "bid": [9.9, 9.9, 4.9, 4.9], "ask": [10.1, 10.1, 5.1, 5.1],
                      "h_est": [0.005, 0.005, 0.02, 0.02], "bucket": ["x", "x", "y", "y"]})
    f = tmp_path / "q.csv"
    q.to_csv(f, index=False)
    assert cli.calibrate_spreads(f, tmp_path) == 0
    t = pd.read_csv(tmp_path / "spread_calibration.csv").set_index("bucket")
    assert t.loc["x", "median"] == pytest.approx(0.01 / 0.005) and bool(t.loc["x", "exceeds_1_5"]) and not bool(t.loc["y", "exceeds_1_5"])


# --- end to end, synthetic ---------------------------------------------------------------------------------------------
def plog_empty():
    return pd.DataFrame({"etf": pd.Series(dtype=str), "as_of": pd.Series(dtype=object), "name": pd.Series(dtype=str),
                         "weight": pd.Series(dtype=float), "accepted": pd.Series(dtype=bool)})


def run_world(cal, snap, yd, out):
    panel = pr.PricePanel(snap, cal, yd)
    return cli.run_from_panel(panel, cal, out, pd.DataFrame(columns=["year", "ysym", "cash_per_share", "source_url"]),
                              rb.SEED, plog_empty(), snap, False, years=list(YEARS))


@pytest.fixture()
def patched_years(monkeypatch, tmp_path):
    spec = tmp_path / "prereg_spec.md"                                  # stand-in for the (undistributed) spec
    spec.write_text("synthetic spec\n")
    monkeypatch.setattr(cli, "SPEC", spec)
    monkeypatch.setattr(art, "plot_events", lambda *a, **k: None)      # the PNGs have their own smoke test
    monkeypatch.setattr(art, "plot_months", lambda *a, **k: None)


def test_gate_stops_before_any_result_then_runs_when_cleared(generic_cal, tmp_path, patched_years):
    extra = [generic_cal.last_trading_day_of_month(y, 11) for y in (2011, 2012, 2013, 2014)]
    snap, yd = make_world(generic_cal, years=YEARS, extra_days=extra, post=lambda i: 1.5 if i == 5 else (0.05 if i < 20 else 0.0))
    out = tmp_path / "out"
    out.mkdir()
    assert run_world(generic_cal, snap, yd, out) == 0
    bm = pd.read_csv(out / "big_moves.csv")
    assert len(bm) == 2 and set(bm["ysym"]) == {sym(5)}                       # one row per event year
    for f in ("events_P.csv", "variants.csv", "decision.json", "placebo_random.csv", "results.md", "deciles.csv", "m_series.csv"):
        assert not (out / f).exists(), f"{f} written before the gate was cleared"
    for f in ("exit_classes.csv", "split_hits.csv", "big_moves.csv", "deleted_before_R.csv"):
        assert (out / f).exists()
    # The rows are cleared by hand; the rerun keeps that text and produces results
    bm["clearance"], bm["reason"] = "exclude", "synthetic +150% pop, excluded in the test"
    bm.to_csv(out / "big_moves.csv", index=False)
    assert run_world(generic_cal, snap, yd, out) == 0
    d = json.loads((out / "decision.json").read_text())
    assert d["verdict"] in {"REJECT", "WEAK", "UNDERPOWERED", "ADVANCE_PENDING_M"} and d["conditions"]["c5"] is None
    assert pd.read_csv(out / "big_moves.csv")["clearance"].eq("exclude").all()
    p = pd.read_csv(out / "events_P.csv")
    assert list(p["year"]) == list(YEARS) and (p["n_universe"] == 39).all()


def test_pass1_outputs_have_no_2007_2011_event_rows_and_runs_are_deterministic(generic_cal, tmp_path, patched_years):
    extra = [generic_cal.last_trading_day_of_month(y, 11) for y in (2011, 2012, 2013, 2014)]
    snap, yd = make_world(generic_cal, years=YEARS, extra_days=extra)
    digests = []
    for k in range(2):
        out = tmp_path / f"o{k}"
        out.mkdir()
        assert run_world(generic_cal, snap, yd, out) == 0
        assert (out / "decision.json").exists()
        digests.append({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("*.csv")) + sorted(out.glob("*.json")) + [out / "results.md"]
                        if f.name != "run.log"})
    assert digests[0] == digests[1]
    out = tmp_path / "o0"
    for f in out.glob("*.csv"):
        df = pd.read_csv(f)
        for col in ("year",):
            if col in df.columns:
                assert not df[col].isin(range(2007, 2012)).any(), f"{f.name} has a 2007-2011 event row"
    m = pd.read_csv(out / "m_series.csv")
    assert m["year"].min() >= 2012


def test_sanity_reproduction_matches_yahoo(generic_cal, world):
    snap, yd = world
    st = rb.Study(pr.PricePanel(snap, generic_cal, yd), generic_cal, years=[2014])
    s = rb.sanity_reproduction(st)
    assert s["n"] == 40 and s["median_abs_diff"] <= 0.005


def test_pngs_render(tmp_path):
    ev = pd.DataFrame({"year": [2012, 2013, 2014], "g_net": [0.02, -0.01, 0.03], "partial_exit": [False, True, False]})
    pl = {"g_net": np.random.default_rng(0).normal(0, 0.03, (200, 3))}
    art.plot_events(ev, pl, tmp_path / "a.png")
    months = pd.DataFrame({"month": [2, 3, 10, 12], "mean_excess": [0.01, -0.01, 0.0, 0.02]})
    art.plot_months(months, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 5000 and (tmp_path / "b.png").stat().st_size > 5000
