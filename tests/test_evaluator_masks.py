"""Warm-up and split cells must not take part in cross-sectional operators.

A cell inside a symbol's warm-up (fewer than `lookback` rows since its first finite close) or
whose window straddles its own split is not point-in-time computable. rank / scale /
indneutralize must ignore such cells, so the other symbols' values are exactly what they would
be if the invalid symbol were absent from the cross-section.
"""
import numpy as np
import pandas as pd
import pytest

from alpha101 import Evaluator, Panel, load_alphas
from alpha101 import operators as ops
from alpha101.expr.analysis import lookback
from alpha101.expr.evaluator import EvalOptions
from alpha101.expr.parser import parse
from alpha101.synthetic import random_panel

T, START_D, L = 40, 20, 5
# delta(close, 5): D's warm-up cells are already NaN through the strict NaN policy.
# The ternary's short branch (`close`, lookback 0) emits finite values for D on rows 21..24
# (returns >= 0 there because D's close rises), so without the fix they leak into the
# cross-section.
X_PLAIN = "delta(close, 5)"
X_TERNARY = "((returns < 0) ? delta(close, 5) : close)"


def _panel(start_d=START_D, split=None):
    rng = np.random.default_rng(7)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(T, 4)), axis=0))
    close[:, 3] = 100.0 + np.arange(T)  # D: strictly rising
    close[:start_d, 3] = np.nan
    fields = {"open": close.copy(), "high": close * 1.01, "low": close * 0.99, "close": close,
              "volume": np.full((T, 4), 1e6), "adj_close": close.copy()}
    if split is not None:
        fields["split"] = split
    return Panel(dates=pd.bdate_range("2024-01-01", periods=T), symbols=pd.Index(list("ABCD")),
                 fields=fields, groups={"sector": np.array([0, 0, 1, 1])})


def _drop(panel: Panel, j: int) -> Panel:
    keep = [k for k in range(len(panel.symbols)) if k != j]
    return Panel(dates=panel.dates, symbols=panel.symbols[keep],
                 fields={k: v[:, keep] for k, v in panel.fields.items()},
                 groups={k: np.asarray(v)[keep] for k, v in panel.groups.items()})


def _eval(panel, expr, **opt):
    return Evaluator(panel, EvalOptions(**opt)).evaluate(expr).to_numpy()


def _patch_legacy(mp):
    """Pre-fix behaviour: cross-sectional ops see only the universe (None here), not validity."""
    rank, scale, ind = ops.rank, ops.scale, ops.indneutralize
    mp.setattr(ops, "rank", lambda x, mask=None: rank(x, None))
    mp.setattr(ops, "scale", lambda x, a=1.0, mask=None: scale(x, a, None))
    mp.setattr(ops, "indneutralize", lambda x, c, mask=None, mg=5: ind(x, c, None, mg))


@pytest.fixture
def legacy(monkeypatch):
    _patch_legacy(monkeypatch)


CASES = [("rank({x})", {}), ("scale({x})", {}),
         ("indneutralize({x}, IndClass.sector)", {"min_group_size": 1})]


@pytest.mark.parametrize("x", [X_PLAIN, X_TERNARY])
@pytest.mark.parametrize("tmpl,opt", CASES)
def test_warmup_cells_do_not_affect_others(tmpl, opt, x):
    expr = tmpl.format(x=x)
    full = _eval(_panel(), expr, **opt)
    without_d = _eval(_drop(_panel(), 3), expr, **opt)
    rows = slice(0, START_D + L)
    np.testing.assert_allclose(full[rows, :3], without_d[rows, :3], equal_nan=True)
    assert np.isnan(full[:START_D + L, 3]).all()
    assert np.isfinite(full[START_D + L:, 3]).all()


def test_scale_row_sums_exclude_warmup():
    out = _eval(_panel(), f"scale({X_TERNARY})")
    s = np.nansum(np.abs(out[:, :3]), axis=1)
    np.testing.assert_allclose(s[L:START_D + L], 1.0)  # A, B, C alone carry the whole book


@pytest.mark.parametrize("tmpl,opt", CASES)
def test_legacy_mask_leaks_warmup(legacy, tmpl, opt):
    """The tests above have teeth: the pre-fix masking changes A, B, C on D's warm-up rows."""
    expr = tmpl.format(x=X_TERNARY)
    full = _eval(_panel(), expr, **opt)
    without_d = _eval(_drop(_panel(), 3), expr, **opt)
    rows = slice(START_D + 1, START_D + L)
    assert not np.allclose(full[rows, :3], without_d[rows, :3], equal_nan=True)


def _split_case():
    split = np.zeros((T, 4))
    split[30, 1] = 2.0  # B splits 2:1 on row 30: its 5-day windows straddle it on rows 30..34
    raw = _panel(start_d=0, split=split).to_raw_basis()
    expr = "rank(delta(close, 5))"
    return _eval(raw, expr), _eval(_drop(raw, 1), expr)


def test_split_cells_do_not_affect_others():
    full, without_b = _split_case()
    np.testing.assert_allclose(full[30:35, [0, 2, 3]], without_b[30:35], equal_nan=True)
    assert np.isnan(full[30:35, 1]).all()
    assert np.isfinite(full[35:, 1]).all() and np.isfinite(full[L:30, 1]).all()


def test_legacy_mask_leaks_split(legacy):
    full, without_b = _split_case()
    assert not np.allclose(full[30:35, [0, 2, 3]], without_b[30:35], equal_nan=True)


def test_symbol_without_close_is_never_valid():
    """np.argmax of an all-NaN close column is 0, which used to make D valid from row L."""
    base = _panel()
    fields = {k: np.array(v, dtype=float, copy=True) for k, v in base.fields.items()}
    fields["close"][:, 3] = np.nan
    panel = Panel(dates=base.dates, symbols=base.symbols, fields=fields, groups=base.groups)
    expr = "rank(sum(volume, 5))"
    full = _eval(panel, expr)
    assert np.isnan(full[:, 3]).all()
    np.testing.assert_allclose(full[:, :3], _eval(_drop(_panel(), 3), expr), equal_nan=True)


# --------------------------------------------------------------------------- regression
REGRESSION = (1, 28, 100)  # rank over a ternary short branch; scale; indneutralize + scale + rank


def _stagger(panel: Panel, starts: dict[int, int]) -> Panel:
    """Same panel with symbol j's history (every field) NaN before row starts[j]."""
    fields = {k: np.array(v, dtype=float, copy=True) for k, v in panel.fields.items()}
    for j, s in starts.items():
        for v in fields.values():
            v[:s, j] = np.nan
    return Panel(dates=panel.dates, symbols=panel.symbols, fields=fields, groups=panel.groups)


def _old_new(panel, expr, monkeypatch):
    new = _eval(panel, expr)
    with monkeypatch.context() as m:
        _patch_legacy(m)
        old = _eval(panel, expr)
    return old, new


@pytest.mark.parametrize("alpha_id", REGRESSION)
def test_regression_standard_panel_unchanged(alpha_id, monkeypatch):
    """Every symbol starts on row 0 and there is no split: nothing invalid is ever in the
    cross-section on a row where the output is defined, so old and new agree exactly."""
    expr = load_alphas()[alpha_id].expression
    old, new = _old_new(random_panel(T=420, N=40, seed=3), expr, monkeypatch)
    np.testing.assert_array_equal(np.isfinite(old), np.isfinite(new))
    fin = np.isfinite(old)
    np.testing.assert_allclose(new[fin], old[fin], rtol=0, atol=0)


@pytest.mark.parametrize("alpha_id", REGRESSION)
def test_regression_staggered_changes_confined_to_warmup_rows(alpha_id, monkeypatch):
    """With late listings, finite values may legitimately change, but only on rows where some
    symbol is inside its warm-up window [first, first + lookback): those are exactly the rows on
    which the old code let a not-yet-computable cell into the cross-section."""
    expr = load_alphas()[alpha_id].expression
    Lb = lookback(parse(expr))
    starts = {3: 60, 11: 100, 25: 150, 38: 300}
    old, new = _old_new(_stagger(random_panel(T=420, N=40, seed=3), starts), expr, monkeypatch)
    warm = np.zeros(old.shape[0], dtype=bool)
    for s in starts.values():
        warm[s:s + Lb] = True
    fin = np.isfinite(old)
    changed = (fin & ~np.isclose(new, old, rtol=0, atol=0, equal_nan=True)).any(axis=1)
    assert not (changed & ~warm).any()
    np.testing.assert_array_equal(np.isfinite(old), np.isfinite(new))
