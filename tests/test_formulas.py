"""alphas.yaml must agree with what the parser computes, and every alpha must evaluate."""
import numpy as np
import pytest

from alpha101 import Evaluator, load_alphas
from alpha101.expr.analysis import group_levels, inputs, lookback, needs_industry
from alpha101.synthetic import random_panel

ALPHAS = load_alphas()


def test_ids_are_1_to_101():
    assert sorted(ALPHAS) == list(range(1, 102))


def test_delay0_alphas_match_paper_footnote():
    assert sorted(i for i, a in ALPHAS.items() if a.delay == 0) == [42, 48, 53, 54]


def test_industry_alphas():
    assert sorted(i for i, a in ALPHAS.items() if a.needs_industry) == [
        48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]


@pytest.mark.parametrize("alpha_id", sorted(ALPHAS))
def test_yaml_computed_fields_are_current(alpha_id):
    a = ALPHAS[alpha_id]
    node = a.ast()
    assert list(a.inputs) == inputs(node)
    assert a.needs_industry == needs_industry(node)
    assert list(a.group_levels) == group_levels(node)
    assert a.max_lookback_days == lookback(node)


@pytest.fixture(scope="module")
def panel():
    return random_panel(T=420, N=40, seed=3)


@pytest.mark.parametrize("alpha_id", sorted(ALPHAS))
def test_every_alpha_evaluates_with_coverage(panel, alpha_id):
    """After the lookback, most cells must be finite (guards against NaN-propagation bugs)."""
    a = ALPHAS[alpha_id]
    out = Evaluator(panel).evaluate(a.expression).to_numpy()
    tail = out[max(a.max_lookback_days, 0) + 5:]
    assert tail.shape[0] > 50
    finite = np.isfinite(tail).mean()
    assert finite > 0.5, f"alpha {alpha_id}: only {finite:.0%} finite after lookback"


@pytest.mark.parametrize("alpha_id", sorted(ALPHAS))
def test_first_value_not_before_lookback(panel, alpha_id):
    """No finite value may appear before max_lookback_days rows of history exist."""
    a = ALPHAS[alpha_id]
    out = Evaluator(panel).evaluate(a.expression).to_numpy()
    assert not np.isfinite(out[: a.max_lookback_days]).any()


def _hand_panel():
    import pandas as pd
    from alpha101.panel import Panel
    o = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
    h = np.array([[11.0, 21.0], [12.5, 20.0], [12.0, 22.0]])
    l = np.array([[9.5, 19.0], [10.5, 18.0], [11.0, 20.0]])
    c = np.array([[10.5, 20.5], [12.0, 18.5], [11.5, 21.5]])
    v = np.array([[100.0, 200.0], [150.0, 100.0], [150.0, 300.0]])
    return Panel(dates=pd.bdate_range("2024-01-01", periods=3), symbols=pd.Index(["A", "B"]),
                 fields={"open": o, "high": h, "low": l, "close": c, "volume": v, "adj_close": c})


def test_alpha101_by_hand():
    p = _hand_panel()
    out = Evaluator(p).evaluate(ALPHAS[101].expression).to_numpy()
    o, h, l, c = (p.fields[k] for k in ("open", "high", "low", "close"))
    np.testing.assert_allclose(out, (c - o) / ((h - l) + 0.001))
    assert out[0, 0] == pytest.approx(0.5 / 1.501)


def test_alpha012_by_hand():
    p = _hand_panel()
    out = Evaluator(p).evaluate(ALPHAS[12].expression).to_numpy()
    # day 1: A volume up (+1) * -(12 - 10.5) = -1.5 ; B volume down (-1) * -(18.5 - 20.5) = -2
    # day 2: A volume flat (0) -> 0 ; B up (+1) * -(21.5 - 18.5) = -3
    np.testing.assert_allclose(out, [[np.nan, np.nan], [-1.5, -2.0], [0.0, -3.0]], equal_nan=True)
