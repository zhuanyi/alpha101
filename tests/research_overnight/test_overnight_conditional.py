import copy
import hashlib
import os
import re

import numpy as np
import pandas as pd
import pytest

import overnight_conditional_test as oc
from overnight_intraday_test import build_returns, newey_west_t


def _prep(synthetic):
    frames, events = synthetic
    frames = oc.apply_era_mask(oc.clean_frames(frames))
    vix = frames["^VIX"]["close"]
    rets, tds = {}, {}
    for s in oc.LOAD_SYMBOLS:
        if s == "^VIX":
            continue
        r = build_returns(frames[s])
        r["volume"] = frames[s]["volume"].reindex(r.index)
        rets[s] = r
        tds[s] = oc.trading_days(frames[s])
    return rets, oc.Calendar(events, tds), vix


def test_lookahead_selftest_passes(synthetic):
    rets, cal, vix = _prep(synthetic)
    flags = oc.condition_flags(rets, cal, vix)
    lines, ok = oc.lookahead_selftest(rets, cal, vix, flags, seed=3, k=5)
    assert ok, lines
    assert len(lines) == 5 and all(l.endswith("PASS") for l in lines)
    # the recompute really ran on truncated inputs: every truncated frame is strictly shorter than the full one
    rows = [int(re.search(r"rows=(\d+)", l).group(1)) for l in lines]
    assert all(r < len(flags) for r in rows), (rows, len(flags))


def test_lookahead_selftest_detects_a_leak(synthetic, monkeypatch):
    """Mutation check: a flag that peeks one row ahead must make the self-test FAIL (it is not a tautology)."""
    rets, cal, vix = _prep(synthetic)
    real = oc.condition_flags

    def leaky(r, c, v):
        out = real(r, c, v)
        out[("SPY", "H7")] = r["SPY"]["total"].shift(-1).notna().astype(float).reindex(out.index)
        return out

    monkeypatch.setattr(oc, "condition_flags", leaky)
    lines, ok = oc.lookahead_selftest(rets, cal, vix, leaky(rets, cal, vix), seed=3, k=5)
    assert not ok and any(l.endswith("FAIL") for l in lines), lines


def test_no_forward_shift_in_module_source():
    assert oc.source_has_forward_shift([oc.__file__]) == []


def test_flag_semantics(synthetic):
    rets, cal, vix = _prep(synthetic)
    f = oc.condition_flags(rets, cal, vix)
    idx = rets["SPY"].index
    h1 = f[("SPY", "H1")].dropna()
    assert (h1.astype(bool) == np.isin(h1.index.dayofweek, [1, 2])).all()
    h2 = f[("SPY", "H2")].dropna()
    assert h2[h2.index.dayofweek == 0].all()
    # a CPI release on a trading day flags that day's open; FOMC flags the statement day itself
    d = cal.event_dates("US_CPI")[5]
    assert f[("SPY", "H4")].loc[d] == 1.0
    d = cal.event_dates("FOMC_STATEMENT")[5]
    assert f[("SPY", "H3")].loc[d] == 1.0
    # turn of month: exactly -1,+1,+2,+3 -> 4 per full month
    h6 = f[("SPY", "H6")].dropna()
    m = h6["2010-01-01":"2010-12-31"].groupby(h6["2010-01-01":"2010-12-31"].index.month).sum()
    assert (m == 4).all(), m.to_dict()
    assert idx.is_monotonic_increasing


def test_hac_intercept_matches_phase0_newey_west():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 800)
    beta, se = oc.ols_hac(x, np.ones((800, 1)))
    assert beta[0] / se[0] == pytest.approx(newey_west_t(x), rel=1e-9)


def test_placebo_rate_on_iid_noise_is_near_nominal():
    rng = np.random.default_rng(5)
    on = rng.normal(0, 50, 3000)
    dow = np.tile(np.arange(5), 600)
    rate, hits, sd, lags = oc.placebo_calibration(on, dow, 150, 300, seed=2)
    assert 0.02 <= rate <= 0.09          # 300 draws at 5%: about +-2.4 binomial sd
    assert hits == round(rate * 300) and lags == oc.nw_lags(len(on))


def test_holdout_mask_drops_2019_plus(synthetic):
    frames, _ = synthetic
    masked = oc.apply_era_mask(frames)
    assert all(df.index.max() < oc.HO_START for df in masked.values())
    unlocked = oc.apply_era_mask(frames, unlock=True)
    assert unlocked["SPY"].index.max() >= oc.HO_START


def _run(synthetic, out):
    frames, events = synthetic
    msgs = []
    return oc.run_stage_a(frames, events, str(out), seed=11, n_perm=30, n_placebo=20, phase0_dir=None, log=msgs.append)


DATE_RE = re.compile(r"\b20(19|2\d)-\d\d-\d\d\b")


def test_stage_a_outputs_contain_no_holdout_dates(synthetic, tmp_path):
    res = _run(synthetic, tmp_path)
    assert res["lookahead_ok"]
    hits = []
    for fn in os.listdir(tmp_path):
        if fn == "contaminated_holdout_weekday.csv":
            continue
        text = ""
        if not fn.endswith(".png"):
            with open(tmp_path / fn, errors="ignore") as fh:
                text = fh.read()
        if DATE_RE.search(text):
            hits.append(fn)
    assert hits == []
    contam = pd.read_csv(tmp_path / "contaminated_holdout_weekday.csv")
    assert set(contam["label"]) == {"contaminated_holdout"} and len(contam) == 5 * len(oc.HYP_SYMBOLS)
    # the stage-A returns never saw a 2019+ row
    assert all(r.index.max() < oc.HO_START for r in res["rets"].values())


def test_determinism_same_seed_same_bytes(synthetic, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(synthetic, a)
    _run(synthetic, b)
    names = sorted(f for f in os.listdir(a) if f.endswith((".csv", ".json", ".txt")))
    assert names and names == sorted(f for f in os.listdir(b) if f.endswith((".csv", ".json", ".txt")))
    for n in names:
        with open(a / n, "rb") as fa, open(b / n, "rb") as fb:
            ha, hb = hashlib.sha256(fa.read()).hexdigest(), hashlib.sha256(fb.read()).hexdigest()
        assert ha == hb, n


def test_stage_b_refuses_without_frozen(tmp_path):
    with pytest.raises(SystemExit) as e:
        oc.main(["--unlock-holdout", "--from-db", "--out-dir", str(tmp_path)])
    assert e.value.code not in (0, None)
    with pytest.raises(SystemExit) as e:
        oc.main(["--unlock-holdout", "--frozen", str(tmp_path / "missing.json"), "--out-dir", str(tmp_path)])
    assert e.value.code not in (0, None)


def test_stage_b_not_implemented_even_with_frozen(tmp_path):
    fz = tmp_path / "frozen.json"
    fz.write_text("{}")
    with pytest.raises(SystemExit) as e:
        oc.main(["--unlock-holdout", "--frozen", str(fz), "--out-dir", str(tmp_path)])
    assert e.value.code not in (0, None)


def test_cost_and_tax_basics():
    c = oc.cost_per_trip("SPY", 100_000)
    assert c == pytest.approx(2 * 9.99 + 100_000 * (2 * 0.5 + 1.0 + 2 * 1.0) / 1e4)
    assert oc.cost_per_trip("SPY", 100_000, L=3) < c
    assert oc.cost_per_trip("SPY", 100_000, fx=True) > 5 * c
    by_y = pd.DataFrame({"net": [1000.0, -500.0], "div": [100.0, 0.0]}, index=[2015, 2016])
    assert oc.annual_tax(by_y, "RRSP", "us").sum() == 0
    assert oc.annual_tax(by_y, "TFSA", "us").sum() == pytest.approx(15.0)
    mg = oc.annual_tax(by_y, "MARGIN_capgains", "us")
    assert mg.loc[2016] == 0
    assert mg.loc[2015] == pytest.approx(900 * 0.2676 + 100 * 0.5353)


# ---- spec 3.2 cost sensitivities -------------------------------------------------------------------------------
E100 = 100_000
SPY_DEFAULT = 2 * 9.99 + E100 * (2 * 0.5 + 1.0 + 2 * 1.0) / 1e4        # = 59.98
XIC_DEFAULT = 2 * 9.99 + E100 * (2 * 2.0 + 3.0 + 2 * 2.0) / 1e4        # = 131.98


def test_cost_sens_default_scenario_is_unchanged():
    assert oc.scenario_costs("default") == oc.COSTS
    assert oc.scenario_costs("default") is not oc.COSTS
    assert oc.cost_per_trip("SPY", E100, costs=oc.scenario_costs("default")) == oc.cost_per_trip("SPY", E100)
    assert oc.cost_per_trip("SPY", E100) == pytest.approx(SPY_DEFAULT)
    assert oc.cost_per_trip("XIC.TO", E100) == pytest.approx(XIC_DEFAULT)
    assert oc.cost_frac_for("panel_US") == pytest.approx(SPY_DEFAULT / E100)


def test_cost_sens_commission_7():
    c = oc.scenario_costs("commission_7")
    assert c["commission"] == 7.0
    assert oc.cost_per_trip("SPY", E100, costs=c) == pytest.approx(SPY_DEFAULT - 2 * (9.99 - 7.0))
    assert oc.cost_per_trip("XIC.TO", E100, costs=c) == pytest.approx(XIC_DEFAULT - 2 * (9.99 - 7.0))


def test_cost_sens_spread_x2():
    c = oc.scenario_costs("spread_x2")
    assert c["half_spread_bp"]["SPY"] == 1.0 and c["half_spread_bp"]["XIC.TO"] == 4.0
    assert oc.cost_per_trip("SPY", E100, costs=c) == pytest.approx(SPY_DEFAULT + E100 * 2 * 0.5 / 1e4)
    assert oc.cost_per_trip("XIC.TO", E100, costs=c) == pytest.approx(XIC_DEFAULT + E100 * 2 * 2.0 / 1e4)


def test_cost_sens_timing_0bp():
    c = oc.scenario_costs("timing_0bp")
    assert oc.cost_per_trip("SPY", E100, costs=c) == pytest.approx(SPY_DEFAULT - E100 * 2 * 1.0 / 1e4)
    assert oc.cost_per_trip("XIC.TO", E100, costs=c) == pytest.approx(XIC_DEFAULT - E100 * 2 * 2.0 / 1e4)


def test_cost_sens_timing_3bp_is_absolute_for_both_markets():
    c = oc.scenario_costs("timing_3bp")
    assert c["timing_slippage_bp_per_side"] == {"us": 3.0, "ca": 3.0}
    assert oc.cost_per_trip("SPY", E100, costs=c) == pytest.approx(SPY_DEFAULT + E100 * 2 * 2.0 / 1e4)
    assert oc.cost_per_trip("XIC.TO", E100, costs=c) == pytest.approx(XIC_DEFAULT + E100 * 2 * 1.0 / 1e4)


def test_cost_sens_scenarios_do_not_mutate_module_costs():
    before = copy.deepcopy(oc.COSTS)
    for name in oc.SCENARIOS:
        oc.scenario_costs(name)
    assert oc.COSTS == before


def test_cost_sens_pipeline_default_reproduces_and_costs_move_verdict_inputs(synthetic, tmp_path):
    res = _run(synthetic, tmp_path)
    cs = oc.cost_sensitivity(res, str(tmp_path), lambda m: None)      # raises if 'default' differs from the main table
    assert set(cs.scenario) == set(oc.SCENARIOS)
    assert (tmp_path / "cost_sensitivity.csv").exists()
    key = ["condition", "unit"]
    by = {n: cs[cs.scenario == n].set_index(key) for n in oc.SCENARIOS}
    d = by["default"]
    assert (by["commission_7"].cost_bp < d.cost_bp).all()
    assert (by["spread_x2"].cost_bp > d.cost_bp).all()
    assert (by["timing_0bp"].cost_bp < d.cost_bp).all()
    assert (by["timing_3bp"].cost_bp > d.cost_bp).all()
    assert (by["timing_0bp"].mean_d_ann > d.mean_d_ann).all()
    assert (by["spread_x2"].t_b == d.t_b).all()                        # cost-independent inputs untouched
    assert (cs.net_bp_D2 == cs.mean_on_C_bp - cs.cost_bp).all()
    assert (cs.flipped == (cs.verdict != cs.default_verdict)).all()
    assert not cs[cs.scenario == "default"].flipped.any()
