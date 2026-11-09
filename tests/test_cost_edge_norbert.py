import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "cost_edge_norbert", Path(__file__).resolve().parents[1] / "scripts" / "cost_edge_norbert.py")
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def test_desk_cost_each_tier_all_in_is_four_thirds():
    assert ce.c_fx_desk(10_000) == pytest.approx(0.0126 * 4 / 3)      # 1.68%
    assert ce.c_fx_desk(24_999.99) == pytest.approx(0.0168)
    assert ce.c_fx_desk(25_000) == pytest.approx(0.0104)
    assert ce.c_fx_desk(99_999) == pytest.approx(0.0104)
    assert ce.c_fx_desk(100_000) == pytest.approx(0.0046 * 4 / 3)     # 0.6133%
    assert ce.c_fx_desk(500_000) == pytest.approx(0.0036)
    assert ce.c_fx_desk(1_000_000) == pytest.approx(0.0019 * 4 / 3)


def test_desk_in_trade_floor_only_from_60k():
    assert ce.c_fx_desk(59_999, in_trade=True) == pytest.approx(0.0104)
    assert ce.c_fx_desk(200_000, in_trade=True) == pytest.approx(0.0104)   # cash would be 0.6133%
    assert ce.c_fx_desk(10_000, in_trade=True) == pytest.approx(0.0168)


def test_c_ng_hand_value():
    # 2*9.99/10000 = 0.001998; + 0.0005 + 0.0004 + 0.0001 = 0.002998
    assert ce.c_ng(10_000, 9.99, 0.0005, 0.0004, 0.0001) == pytest.approx(0.002998)


def test_breakeven_within_first_tier():
    # 2k / (0.0168 - 0.002) = 19.98 / 0.0148; tiers below cost more than NG everywhere above, so it holds
    v_star, holds = ce.breakeven(9.99, 0.002, 0.0, fx=1.0)
    assert v_star == pytest.approx(19.98 / 0.0148)
    assert holds is True


def test_breakeven_deviation_sign_flip():
    good = ce.breakeven(9.99, 0.002, -0.0005)[0]   # deviation in our favour
    bad = ce.breakeven(9.99, 0.002, +0.0005)[0]
    assert good == pytest.approx(19.98 / 0.0153)
    assert bad == pytest.approx(19.98 / 0.0143)
    assert good < bad


def test_breakeven_never_when_costs_exceed_desk():
    assert ce.breakeven(9.99, 0.05, 0.0) == (None, False)


def test_breakeven_skips_tier_where_denominator_nonpositive():
    # h_sum 1.2%: first tier denom 0.0168-0.012 > 0 -> V* = 19.98/0.0048 = 4162.5 (inside tier)
    assert ce.breakeven(9.99, 0.012, 0.0)[0] == pytest.approx(19.98 / 0.0048)
    # h_sum 1.5%: tier 1 denom 0.0018 -> V*=11100 (inside, <25k)
    assert ce.breakeven(9.99, 0.015, 0.0)[0] == pytest.approx(19.98 / 0.0018)
    # h_sum 1.7% > tier1 cost: skip tier 1; tier 2 (1.04%) also <; all tiers < h -> never
    assert ce.breakeven(9.99, 0.017, 0.0) == (None, False)


def test_breakeven_non_monotone_ng_loses_again_at_higher_tier():
    # h=1.5%, d=0, fx=1: NG wins from 11,100, but the 25k+ tier costs 1.04% < 1.5% half-spreads, so it loses again
    v_star, holds = ce.breakeven(9.99, 0.015, 0.0, fx=1.0)
    assert v_star == pytest.approx(11_100)
    assert holds is False
    assert ce.c_ng(11_100, 9.99, 0.015, 0.0, 0.0) <= ce.c_fx_desk(11_100)
    assert ce.c_ng(30_000, 9.99, 0.015, 0.0, 0.0) > ce.c_fx_desk(30_000)     # 1.567% vs 1.04%
    assert ce.c_ng(100_000, 9.99, 0.015, 0.0, 0.0) > ce.c_fx_desk(100_000)   # 1.52% vs 0.61%
    assert ce.breakeven_intervals(9.99, 0.015, 0.0) == [(pytest.approx(11_100), 25_000.0)]


def test_breakeven_holds_only_up_to_v_max():
    # intervals (1k..25k) lose nowhere below 25k, and the next tier wins from its edge: contiguous -> holds
    assert ce.breakeven(9.99, 0.002, 0.0)[1] is True
    # h=0.8%: tier 3 (0.61%) has denom < 0 -> NG loses from 100k; holds up to 500k is False, up to 90k is True
    assert ce.breakeven(9.99, 0.008, 0.0)[1] is False
    assert ce.breakeven(9.99, 0.008, 0.0, v_max=90_000)[1] is True


def test_breakeven_intervals_clamp_to_tier_edge():
    # h=0.2%: every tier has denom > 0 and 2k/denom is below the tier's lower edge for tiers 2+, so each tier's
    # interval must start at its own edge (not at the stray root below it)
    iv = ce.breakeven_intervals(9.99, 0.002, 0.0)
    assert iv[0] == (pytest.approx(19.98 / 0.0148), 25_000.0)
    assert iv[1] == (25_000.0, 100_000.0)
    assert iv[2] == (100_000.0, 500_000.0)
    assert [lo for lo, _ in iv] == sorted(lo for lo, _ in iv)


def test_breakeven_usd_basis_shifts_tier_edges():
    # tier-1 denom = 0.0168-0.0165=0.0003 -> V*=66,600; USD basis (fx 1.4): tier 1 spans up to 35,000 -> V* outside,
    # tier 2 denom = 0.0104-0.0165 < 0 -> skip -> never. CAD basis: tier1 up to 25,000 -> V* outside -> never too.
    assert ce.breakeven(9.99, 0.0165, 0.0, fx=1.4)[0] is None
    # tier 2 reachable: h=0.0085, tier 1 V*=19.98/0.0083=2407 -> inside
    assert ce.breakeven(9.99, 0.0085, 0.0, fx=1.4)[0] == pytest.approx(19.98 / 0.0083)


def test_breakeven_tier_edges_scale_by_fx_multiplicatively():
    # tier-1 root 19.98 / (0.0168 - 0.016133) = 30,000 CAD. With fx 1.4 tier 1 spans up to 25,000 * 1.4 = 35,000, so the
    # root is inside tier 1 and is returned. Dividing instead (25,000 / 1.4 = 17,857) would push it out of tier 1.
    h = 0.0168 - 19.98 / 30_000
    v_star, holds = ce.breakeven(9.99, h, 0.0, fx=1.4)
    assert v_star == pytest.approx(30_000)
    assert holds is False        # tier 2 desk cost (1.04%) is below h (1.6%): NG loses again from 35,000
    assert ce.breakeven(9.99, h, 0.0, fx=1.0)[0] is None   # CAD basis: 30,000 is already in tier 2 where desk = 1.04% < h


def _ohlc(rows):
    return pd.DataFrame(rows, columns=["High", "Low", "Close"], index=pd.bdate_range("2026-01-05", periods=len(rows)))


def test_cs_clamps_to_zero_when_two_day_range_dwarfs_single_day_ranges():
    # a 10% jump between two tight days: gamma >> beta, raw alpha < 0, spread clamped to 0
    df = _ohlc([(101.0, 100.0, 100.5), (111.0, 110.0, 110.5)])
    assert ce.cs_half_spread(df).iloc[0] == 0.0
    assert ce.cs_clamp_share(df).iloc[0] == 1.0


def test_cs_identical_days_closed_form():
    # same H/L both days: beta = 2b, gamma = b, alpha = sqrt(b) * ((2 - sqrt 2)/k - 1/sqrt k) = sqrt(b) (k = 3 - 2 sqrt 2)
    df = _ohlc([(10.02, 10.00, 10.01)] * 3)
    alpha = np.sqrt(np.log(10.02 / 10.00) ** 2)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    assert ce.cs_half_spread(df).iloc[0] == pytest.approx(s / 2, rel=1e-9)


def test_cs_hand_value():
    # Day1 H/L=1.01, day2 H/L=1.01 with higher range 2-day gamma = ln(1.0201)^2 (stacked ranges 1.01 -> 1.0201)
    h = [101.0, 102.01]
    l = [100.0, 101.0]
    df = _ohlc([(h[0], l[0], 100.5), (h[1], l[1], 101.5)])
    b1 = np.log(1.01) ** 2
    beta = 2 * b1
    gamma = np.log(102.01 / 100.0) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    got = ce.cs_half_spread(df)
    assert len(got) == 1
    assert got.iloc[0] == pytest.approx(max(s, 0.0) / 2)


def test_ar_zero_when_close_equals_mid():
    df = _ohlc([(10.02, 10.00, float(np.sqrt(10.02 * 10.00)))] * 12)
    assert ce.ar_half_spread_month(df).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_ar_hand_value_and_clamp():
    # close above mid every day by x in logs, mid constant -> product = x*x -> s = 2x, half = x
    mid = 100.0
    x = 0.001
    hi, lo = mid * np.exp(0.002), mid * np.exp(-0.002)
    df = _ohlc([(hi, lo, mid * np.exp(x))] * 12)
    assert ce.ar_half_spread_month(df).iloc[0] == pytest.approx(x, rel=1e-9)
    # eta constant, close constant: product = x^2 on every day, so the sign of c - eta does not matter
    rows = [(hi, lo, mid * np.exp(x * (1 if i % 2 == 0 else -1))) for i in range(12)]
    assert ce.ar_half_spread_month(_ohlc(rows)).iloc[0] == pytest.approx(x, rel=1e-9)
    # mid alternates 99.9 / 100.1 around a constant close of 100: (c - eta_t)(c - eta_t+1) < 0 -> s^2 < 0 -> clamped to 0
    rows = [((100.0 + (0.1 if i % 2 else -0.1)) + 0.05, (100.0 + (0.1 if i % 2 else -0.1)) - 0.05, 100.0) for i in range(12)]
    assert ce.ar_half_spread_month(_ohlc(rows)).iloc[0] == 0.0


def test_ar_requires_min_obs():
    df = _ohlc([(10.02, 10.00, 10.01)] * 5)
    assert len(ce.ar_half_spread_month(df)) == 0


def test_deviation_direction_signs():
    idx = pd.bdate_range("2026-01-05", periods=2)
    dlr = pd.DataFrame({"Close": [14.2, 14.2]}, index=idx)
    dlru = pd.DataFrame({"Close": [10.0, 10.0]}, index=idx)
    boc = pd.Series([1.40, 1.44], index=idx)
    j = ce.deviation_series(dlr, dlru, boc)
    # implied 1.42: above BoC 1.40 -> CAD->USD loses, USD->CAD gains; reversed on day 2
    assert j["d_c2u"].iloc[0] == pytest.approx(1 - 1.40 / 1.42) and j["d_c2u"].iloc[0] > 0
    assert j["d_u2c"].iloc[0] == pytest.approx(1 - 1.42 / 1.40) and j["d_u2c"].iloc[0] < 0
    assert j["d_c2u"].iloc[1] < 0 < j["d_u2c"].iloc[1]
    assert j["dev_bp"].iloc[0] == pytest.approx((1.42 / 1.40 - 1) * 1e4)


def test_daily_from_hourly_aggregates():
    ix = pd.to_datetime(["2026-01-05 09:30", "2026-01-05 10:30", "2026-01-06 09:30"])
    h = pd.DataFrame({"Open": [1, 2, 3], "High": [5, 6, 7], "Low": [0.5, 0.4, 0.9], "Close": [2, 3, 4], "Volume": [1, 1, 1]}, index=ix)
    d = ce.daily_from_hourly(h)
    assert d.loc["2026-01-05", ["Open", "High", "Low", "Close"]].tolist() == [1, 6, 0.4, 3]
    assert len(d) == 2


def _tight(n=14):
    # near-zero ranges and close at the log mid: CS and AR estimates are ~0, far below the half-tick
    return _ohlc([(100.0001, 100.0, float(np.sqrt(100.0001 * 100.0)))] * n)


def test_spread_table_floors_at_half_tick():
    df = _tight()
    t = ce.spread_table(df).loc[2026]
    tick = ce.TICK / 2.0 / float(df["Close"].median())
    assert t["cs_half"] < tick and t["ar_half"] < tick
    assert t["used"] == pytest.approx(tick)


def test_spread_window_floors_at_half_tick():
    df = _tight()
    w = ce.spread_window(df)
    tick = ce.TICK / 2.0 / float(df["Close"].median())
    assert w["cs_half"] < tick and w["ar_half"] < tick
    assert w["used"] == pytest.approx(tick)


def test_scenario_saving_wiring():
    v, fx, d_exp, d_adv, h, k = 100_000, 1.4, 0.0003, 0.0020, 0.0010, 9.99
    # base: USD basis reads the tier off v/fx = 71,429 (0.78% listed), uses d_exp and h as given
    desk, ng, sv = ce.scenario_saving(v, fx, d_exp, d_adv, h, k, "usd", "base")
    assert desk == pytest.approx(0.0104)
    assert ng == pytest.approx(2 * k / v + h + d_exp)
    assert sv == pytest.approx(desk - ng)
    # CAD basis reads the tier off v = 100,000 (0.46% listed)
    assert ce.scenario_saving(v, fx, d_exp, d_adv, h, k, "cad", "base")[0] == pytest.approx(0.0046 * 4 / 3)
    # adverse swaps d_exp for d_adv; x2 doubles h; both together do both
    assert ce.scenario_saving(v, fx, d_exp, d_adv, h, k, "usd", "adverse d (p95)")[1] == pytest.approx(2 * k / v + h + d_adv)
    assert ce.scenario_saving(v, fx, d_exp, d_adv, h, k, "usd", "half-spread x2")[1] == pytest.approx(2 * k / v + 2 * h + d_exp)
    assert ce.scenario_saving(v, fx, d_exp, d_adv, h, k, "usd", "x2 + adverse")[1] == pytest.approx(2 * k / v + 2 * h + d_adv)
    assert set(ce.SCENARIOS) == {"base", "half-spread x2", "adverse d (p95)", "x2 + adverse"}


def test_desk_tier_text_is_built_from_the_tier_table():
    t = ce.desk_tier_text()
    assert t == ("1.26% below $25,000; 0.78% for $25,000-$99,999; 0.46% for $100,000-$499,999; "
                 "0.27% for $500,000-$999,999; 0.19% from $1,000,000")


def test_scenario_saving_rejects_unknown_scenario():
    with pytest.raises(ValueError):
        ce.scenario_saving(100_000, 1.4, 0.0, 0.002, 0.001, 9.99, "usd", "adverse")      # substring of a real name
    with pytest.raises(ValueError):
        ce.scenario_saving(100_000, 1.4, 0.0, 0.002, 0.001, 9.99, "usd", "nonsense")


def test_m_aff_added_after_surcharge():
    # surcharge order: listed * 4/3 + m_aff (not (listed + m_aff) * 4/3, and m_aff is not dropped)
    assert ce.c_fx_desk(10_000, m_aff=0.001) == pytest.approx(0.0126 * 4 / 3 + 0.001)
    assert ce.c_fx_desk(10_000, m_aff=0.001) != pytest.approx((0.0126 + 0.001) * 4 / 3)
    # breakeven_intervals uses the same cost: tier-1 denominator = 0.0168 + 0.001 - 0.002
    iv = ce.breakeven_intervals(9.99, 0.002, 0.0, m_aff=0.001)
    assert iv[0][0] == pytest.approx(19.98 / (0.0126 * 4 / 3 + 0.001 - 0.002))


def test_breakeven_default_v_max_is_largest_notional():
    # h = 0.3%: tiers up to 1,000,000 win (0.36% all-in at 500k+), the 1M+ tier (0.2533%) loses -> NG wins to 1,000,000 only
    v_star, holds = ce.breakeven(9.99, 0.003, 0.0)
    assert v_star is not None
    assert holds is True                                       # default v_max = 500,000 < 1,000,000
    assert ce.breakeven(9.99, 0.003, 0.0, v_max=2_000_000)[1] is False
    assert ce.breakeven(9.99, 0.003, 0.0, v_max=max(ce.NOTIONALS))[1] is True


_SNAP = Path(__file__).resolve().parents[1] / "research" / "cost_edge_norbert" / "snapshot_2026-10-21"


@pytest.fixture(scope="module")
def rendered():
    if not _SNAP.exists():
        pytest.skip("vendor price snapshot is not redistributed; run scripts/cost_edge_norbert.py to fetch one")
    data = ce.load(_SNAP)
    res = ce.analyse(data)
    return ce.render(res, data), res, data


def _section(doc, start, end):
    i = doc.index(start)
    return doc[i:doc.index(end, i)]


def test_render_deviation_table_directions_not_swapped(rendered):
    doc, res, _ = rendered
    dev = res["dev"]
    for name, col in (("CAD->USD", "d_c2u"), ("USD->CAD", "d_u2c")):
        d = ce.dist(dev[col] * 1e4)
        row = f"| {name} | {d['mean']:.1f} | {d['median']:.1f} | {d['p5']:.1f} | {d['p95']:.1f} |"
        assert row in doc
    assert ce.dist(dev["d_c2u"] * 1e4) != ce.dist(dev["d_u2c"] * 1e4)    # guard: the two rows must be distinguishable


def test_render_adverse_column_uses_direction_p95(rendered):
    doc, res, _ = rendered
    dev, fx = res["dev"], res["fx"]
    h_sum = res["win_dlr"]["used"] + res["win_dlru"]["used"]
    p95 = {"c2u": float(dev["d_c2u"].quantile(0.95)), "u2c": float(dev["d_u2c"].quantile(0.95))}
    assert p95["c2u"] != pytest.approx(p95["u2c"])
    for sub, dname, key in (("4.1", "CAD->USD", "c2u"), ("4.2", "USD->CAD", "u2c")):
        sec = _section(doc, f"### {sub} {dname}", "Breakeven notional")
        assert f"adverse (p95) d = {ce.bp(p95[key])} bp" in sec
        row = next(r for r in sec.splitlines() if r.startswith("| 100,000 |") and r.count("(") >= 4)
        _, _, sv = ce.scenario_saving(100_000, fx, 0.0, p95[key], h_sum, ce.COMMISSION, "usd", "x2 + adverse")
        assert row.rstrip().endswith(f"{sv * 100_000:,.0f} ({ce.bp(sv)}) |")
        other = "u2c" if key == "c2u" else "c2u"
        _, _, wrong = ce.scenario_saving(100_000, fx, 0.0, p95[other], h_sum, ce.COMMISSION, "usd", "x2 + adverse")
        assert not row.rstrip().endswith(f"{wrong * 100_000:,.0f} ({ce.bp(wrong)}) |")


def test_render_verdict_prints_cad_basis_stress_exception(rendered):
    doc, res, _ = rendered
    sec = doc[doc.index("## 6. Verdict"):doc.index("## 7. Caveats")]
    assert "| 500,000 | NG (NG 0.336% vs desk 0.360%) | desk (NG 0.374% vs desk 0.360%) |" in sec
    assert "USD->CAD at C$500,000 (NG 0.374% vs desk 0.360%" in sec
    assert "**Qualification:**" in sec
    assert "1 of 10 stressed cells" in sec


def test_render_daily_row_count_comes_from_snapshot(rendered):
    doc, res, data = rendered
    assert "the daily endpoint returned one row on" in doc
    data2 = dict(data, dlru_daily=pd.concat([data["dlru_daily"]] * 3))
    doc2 = ce.render(res, data2)
    assert "the daily endpoint returned 3 rows on" in doc2
