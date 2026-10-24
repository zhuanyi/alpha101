"""Hand-computed expectations for every operator on tiny panels (docs/operators.md)."""
import numpy as np
import pandas as pd
import pytest

from alpha101 import operators as op
from alpha101.expr.analysis import window
from alpha101.expr.evaluator import EvalOptions, Evaluator
from alpha101.panel import Panel

nan = np.nan


def col(*v):
    """A single-symbol (T, 1) array."""
    return np.array(v, dtype=float)[:, None]


def eq(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-12, atol=1e-12,
                               equal_nan=True)


# ----------------------------------------------------------------- windows and rounding
def test_window_floors_non_integer():
    assert window(3.72469) == 3
    assert window(1.06157) == 1
    assert window(20.0) == 20
    assert window(9.99) == 9
    # 1e-9 guard against float representation error (e.g. 2.9999999999999996 -> 3)
    assert window(2.9999999999999996) == 3


def test_window_below_one_rejected():
    with pytest.raises(ValueError):
        window(0.5)


# ----------------------------------------------------------------- delay / delta
def test_delay():
    eq(op.delay(col(1, 2, 3, 4, 5), 2), col(nan, nan, 1, 2, 3))


def test_delay_longer_than_series_is_all_nan():
    eq(op.delay(col(1, 2), 5), col(nan, nan))


def test_delta():
    eq(op.delta(col(1, 4, 9, 16), 1), col(nan, 3, 5, 7))
    eq(op.delta(col(1, 4, 9, 16), 2), col(nan, nan, 8, 12))


# ----------------------------------------------------------------- sum / product / stddev / min / max
def test_ts_sum_includes_today():
    eq(op.ts_sum(col(1, 2, 3, 4, 5), 3), col(nan, nan, 6, 9, 12))


def test_ts_sum_nan_in_window_is_nan():
    eq(op.ts_sum(col(1, nan, 3, 4, 5, 6), 2), col(nan, nan, nan, 7, 9, 11))


def test_ts_product():
    eq(op.ts_product(col(1, 2, 3, -1), 2), col(nan, 2, 6, -3))


def test_ts_stddev_is_sample():
    eq(op.ts_stddev(col(1, 2, 3, 5), 3), col(nan, nan, 1.0, np.std([2, 3, 5], ddof=1)))


def test_ts_min_max():
    x = col(3, 1, 2, 5, 4)
    eq(op.ts_min(x, 3), col(nan, nan, 1, 1, 2))
    eq(op.ts_max(x, 3), col(nan, nan, 3, 5, 5))


# ----------------------------------------------------------------- ts_rank
def test_ts_rank_pct_with_ties():
    # windows: [1,2,3] -> today 3 is top: 3/3; [2,3,2] -> today 2 ties with one other: 1.5/3
    # [3,2,2] -> 1.5/3
    eq(op.ts_rank(col(1, 2, 3, 2, 2), 3), col(nan, nan, 1.0, 0.5, 0.5))


def test_ts_rank_lowest_and_nan():
    eq(op.ts_rank(col(5, 4, 3, nan, 1, 2, 3), 3), col(nan, nan, 1 / 3, nan, nan, nan, 1.0))


# ----------------------------------------------------------------- ts_argmax / ts_argmin
def test_ts_argmax_oldest_1based():
    # windows [1,3,2] -> max at position 2 of 3; [3,2,5] -> 3 (today); [2,5,4] -> 2
    eq(op.ts_argmax(col(1, 3, 2, 5, 4), 3), col(nan, nan, 2, 3, 2))


def test_ts_argmax_days_ago():
    eq(op.ts_argmax(col(1, 3, 2, 5, 4), 3, convention="days_ago"), col(nan, nan, 1, 0, 1))


def test_ts_argmax_ties_take_oldest():
    eq(op.ts_argmax(col(2, 2, 1), 3), col(nan, nan, 1))


def test_ts_argmin():
    eq(op.ts_argmin(col(3, 1, 2, 0), 3), col(nan, nan, 2, 3))


def test_bad_argmax_convention():
    with pytest.raises(ValueError):
        op.ts_argmax(col(1, 2, 3), 2, convention="zero_based")


# ----------------------------------------------------------------- decay_linear
def test_decay_linear_newest_weighted_most():
    # weights oldest..today = 1,2,3 over 6
    eq(op.decay_linear(col(1, 2, 3, 6), 3), col(nan, nan, (1 + 4 + 9) / 6, (2 + 6 + 18) / 6))


def test_decay_linear_window_one_is_identity():
    eq(op.decay_linear(col(1, 2, 3), 1), col(1, 2, 3))


def test_decay_linear_nan_window():
    eq(op.decay_linear(col(1, nan, 3, 4), 2), col(nan, nan, nan, (3 + 8) / 3))


# ----------------------------------------------------------------- correlation / covariance
def test_correlation_perfect_and_negative():
    x = col(1, 2, 3, 4)
    eq(op.correlation(x, col(2, 4, 6, 8), 3), col(nan, nan, 1, 1))
    eq(op.correlation(x, col(8, 6, 4, 2), 3), col(nan, nan, -1, -1))


def test_correlation_constant_window_is_zero_not_nan():
    eq(op.correlation(col(1, 2, 3, 4), col(5, 5, 5, 5), 3), col(nan, nan, 0, 0))


def test_correlation_nan_in_window_is_nan():
    eq(op.correlation(col(1, 2, nan, 4, 5, 7), col(1, 3, 2, 5, 4, 6), 3),
       col(nan, nan, nan, nan, nan, op.correlation(col(4, 5, 7), col(5, 4, 6), 3)[-1, 0]))


def test_correlation_hand_value():
    # x=[1,2,4], y=[1,3,2]: cov = 0.5, sx = 1.5275, sy = 1 -> r = 0.32733
    r = op.correlation(col(1, 2, 4), col(1, 3, 2), 3)[-1, 0]
    assert r == pytest.approx(0.5 / np.sqrt(7 / 3), rel=1e-12)


def test_covariance_sample():
    eq(op.covariance(col(1, 2, 3), col(1, 2, 3), 3), col(nan, nan, 1.0))
    eq(op.covariance(col(1, 2, 4), col(1, 3, 2), 3), col(nan, nan, 0.5))


# ----------------------------------------------------------------- cross-sectional
def test_rank_pct_average_ties_and_nan():
    x = np.array([[10, 30, 20], [5, 5, 1], [nan, 2, 1]], dtype=float)
    eq(op.rank(x), [[1 / 3, 1, 2 / 3], [2.5 / 3, 2.5 / 3, 1 / 3], [nan, 1, 0.5]])


def test_rank_respects_universe_mask():
    x = np.array([[1, 2, 3, 4]], dtype=float)
    mask = np.array([[True, False, True, True]])
    eq(op.rank(x, mask), [[1 / 3, nan, 2 / 3, 1]])


def test_scale_default_and_a():
    x = np.array([[1, -3, nan]], dtype=float)
    eq(op.scale(x), [[0.25, -0.75, nan]])
    eq(op.scale(x, 2.0), [[0.5, -1.5, nan]])
    eq(op.scale(np.zeros((1, 2))), [[nan, nan]])


def test_indneutralize_demeans_within_group():
    x = np.array([[1, 3, 10, 20, 30]], dtype=float)
    lab = np.array([0, 0, 1, 1, 1])
    market = np.zeros(5, dtype=int)
    eq(op.indneutralize(x, [lab, market], min_group_size=2), [[-1, 1, -10, 0, 10]])


def test_indneutralize_small_group_falls_back_to_market():
    x = np.array([[1, 3, 10, 20, 30]], dtype=float)
    lab = np.array([0, 0, 1, 1, 1])
    market = np.zeros(5, dtype=int)
    m = 64 / 5
    eq(op.indneutralize(x, [lab, market], min_group_size=3), [[1 - m, 3 - m, -10, 0, 10]])


def test_indneutralize_unlabelled_symbol_uses_coarser_level():
    x = np.array([[1, 3, 5]], dtype=float)
    lab = np.array([0, 0, -1])
    market = np.zeros(3, dtype=int)
    eq(op.indneutralize(x, [lab, market], min_group_size=2), [[-1, 1, 2]])


def test_indneutralize_counts_only_non_nan_members():
    x = np.array([[1, nan, 5, 7]], dtype=float)
    lab = np.array([0, 0, 1, 1])
    market = np.zeros(4, dtype=int)
    # group 0 has a single valid member -> falls back to market mean of [1, 5, 7] = 13/3
    eq(op.indneutralize(x, [lab, market], min_group_size=2), [[1 - 13 / 3, nan, -1, 1]])


# ----------------------------------------------------------------- element-wise
def test_signedpower_preserves_sign():
    eq(op.signedpower(np.array([-2.0, 3.0, 0.0]), 2.0), [-4, 9, 0])
    eq(op.signedpower(np.array([-8.0]), 1 / 3), [-2])


def test_power_negative_base_fractional_exponent_is_nan():
    eq(op.power(np.array([-8.0, 2.0]), np.array([1 / 3, 3.0])), [nan, 8])


def test_division_by_zero_is_nan():
    eq(op.div(np.array([1.0, 0.0, -1.0]), 0.0), [nan, nan, nan])


def test_log_nonpositive_is_nan():
    eq(op.log(np.array([np.e, 0.0, -1.0])), [1, nan, nan])


def test_sign_and_abs():
    eq(op.sign(np.array([-2.0, 0.0, 3.0, nan])), [-1, 0, 1, nan])
    eq(op.absolute(np.array([-2.0, nan])), [2, nan])


def test_compare_propagates_nan():
    eq(op.compare("<", np.array([1.0, 2.0, nan]), np.array([2.0, 1.0, 1.0])), [1, 0, nan])
    eq(op.compare("==", np.array([1.0]), 1.0), [1])


def test_logical_or_and():
    a = np.array([1.0, 0.0, 0.0, nan])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    eq(op.logical("||", a, b), [1, 0, 1, nan])
    eq(op.logical("&&", a, b), [0, 0, 0, nan])


def test_where_nan_condition():
    eq(op.where(np.array([1.0, 0.0, nan]), 10.0, 20.0), [10, 20, nan])


def test_elementwise_min_max():
    eq(op.elementwise_max(np.array([1.0, 5.0, nan]), np.array([3.0, 2.0, 1.0])), [3, 5, nan])
    eq(op.elementwise_min(np.array([1.0, 5.0]), np.array([3.0, 2.0])), [1, 2])


# ----------------------------------------------------------------- through the evaluator
def _toy_panel():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = np.array([[1, 10], [3, 9], [2, 8], [5, 12], [4, 11], [6, 7]], dtype=float)
    vol = np.array([[100, 50], [110, 40], [90, 60], [120, 70], [80, 30], [130, 90]], dtype=float)
    return Panel(dates=dates, symbols=pd.Index(["A", "B"]),
                 fields={"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
                         "volume": vol, "adj_close": close},
                 groups={"sector": np.array([0, 0])})


def test_evaluator_non_integer_window_floors():
    p = _toy_panel()
    a = Evaluator(p).evaluate("ts_rank(close, 3.72469)").to_numpy()
    eq(a, op.ts_rank(p.fields["close"], 3))


def test_evaluator_min_with_literal_is_ts_min_and_with_expr_is_elementwise():
    p = _toy_panel()
    ev = Evaluator(p)
    eq(ev.evaluate("min(close, 2)").to_numpy(), op.ts_min(p.fields["close"], 2))
    eq(ev.evaluate("min(close, volume / 10)").to_numpy(),
       np.minimum(p.fields["close"], p.fields["volume"] / 10))


def test_evaluator_ts_argmax_is_1_based_by_default_and_configurable():
    p = _toy_panel()
    a = Evaluator(p).evaluate("Ts_ArgMax(close, 3)").to_numpy()
    eq(a[:, 0], [nan, nan, 2, 3, 2, 3])
    b = Evaluator(p, EvalOptions(argmax_convention="days_ago")).evaluate("ts_argmax(close, 3)")
    eq(b.to_numpy()[:, 0], [nan, nan, 1, 0, 1, 0])


def test_evaluator_adv_units():
    p = _toy_panel()
    shares = Evaluator(p).evaluate("adv2").to_numpy()
    dollar = Evaluator(p, EvalOptions(adv_units="dollar")).evaluate("adv2").to_numpy()
    eq(shares[:, 0], [nan, 105, 100, 105, 100, 105])
    eq(dollar[:, 0], [nan, (100 + 330) / 2, (330 + 180) / 2, (180 + 600) / 2, (600 + 320) / 2,
                      (320 + 780) / 2])


def test_evaluator_returns_vwap_and_constants():
    p = _toy_panel()
    ev = Evaluator(p)
    eq(ev.evaluate("returns").to_numpy()[:, 0], [nan, 2, -1 / 3, 1.5, -0.2, 0.5])
    c = p.fields["close"][:, 0]
    eq(ev.evaluate("vwap").to_numpy()[:, 0], ((c + 1) + (c - 1) + c) / 3)
    eq(ev.evaluate("(-1 * 1)").to_numpy(), np.full((6, 2), -1.0))


def test_evaluator_indneutralize_needs_labels():
    p = _toy_panel()
    out = Evaluator(p, EvalOptions(min_group_size=2)).evaluate(
        "indneutralize(close, IndClass.industry)").to_numpy()
    c = p.fields["close"]
    eq(out, c - c.mean(axis=1, keepdims=True))  # industry missing -> sector (2 members)
    bare = Panel(dates=p.dates, symbols=p.symbols, fields=p.fields)
    with pytest.raises(KeyError):
        Evaluator(bare).evaluate("indneutralize(close, IndClass.sector)")


def test_evaluator_window_must_be_literal():
    with pytest.raises(ValueError):
        Evaluator(_toy_panel()).evaluate("ts_sum(close, volume)")


def test_evaluator_universe_mask_applies_to_rank_and_output():
    p = _toy_panel()
    uni = np.ones((6, 2), dtype=bool)
    uni[0, 1] = False
    p2 = Panel(dates=p.dates, symbols=p.symbols, fields=p.fields, universe=uni)
    out = Evaluator(p2).evaluate("rank(close)").to_numpy()
    eq(out[0], [1.0, nan])          # A alone in the universe on day 0
    eq(out[1], [0.5, 1.0])


def test_memoised_values_are_read_only():
    ev = Evaluator(_toy_panel())
    ev.evaluate("delta(close, 1) + delta(close, 1)")
    for v in ev._memo.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            assert not v.flags.writeable


# ----------------------------------------------------------------- multi-column / chunking
def test_correlation_matches_per_column_numpy_on_many_columns():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 7)), rng.normal(size=(40, 7))
    got = op.correlation(x, y, 10)
    for j in range(7):
        for t in range(9, 40):
            want = np.corrcoef(x[t - 9:t + 1, j], y[t - 9:t + 1, j])[0, 1]
            assert got[t, j] == pytest.approx(want, abs=1e-10)
    assert np.isnan(got[:9]).all()


def test_sliding_kernels_identical_across_chunk_boundaries(monkeypatch):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 13))
    x[17, 4] = np.nan
    full = [op.ts_rank(x, 7), op.ts_argmax(x, 7), op.decay_linear(x, 7), op.ts_product(x, 3)]
    monkeypatch.setattr(op, "_CHUNK_BYTES", 8 * 54 * 7 * 3)  # forces 3-column chunks
    chunked = [op.ts_rank(x, 7), op.ts_argmax(x, 7), op.decay_linear(x, 7), op.ts_product(x, 3)]
    for a, b in zip(full, chunked):
        eq(a, b)
    assert np.isnan(full[0][17:24, 4]).all() and np.isfinite(full[0][24, 4])
