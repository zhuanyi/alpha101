import numpy as np
import pandas as pd
import pytest

from alpha101 import Evaluator, Panel, forward_returns, load_alphas
from alpha101.synthetic import random_panel, rescaled

nan = np.nan


def _panel(split=None):
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = np.array([[10.0], [11.0], [12.0], [3.0], [3.3], [3.6]])  # 4:1 split on day 3 (raw)
    adj_close = close.copy()
    fields = {"open": close.copy(), "high": close + 0.1, "low": close - 0.1, "close": close,
              "volume": np.full((6, 1), 100.0), "adj_close": adj_close}
    if split is not None:
        fields["split"] = split
    return Panel(dates=dates, symbols=pd.Index(["A"]), fields=fields)


def test_forward_returns_close_default_lag1():
    p = _panel()
    p.fields["adj_close"] = np.array([[1.0], [2.0], [4.0], [8.0], [16.0], [32.0]])
    r = forward_returns(p, horizon=1).to_numpy()[:, 0]
    # alpha at t trades at close t+1, exits at close t+2: 4/2-1, 8/4-1, ...
    np.testing.assert_allclose(r, [1, 1, 1, 1, nan, nan], equal_nan=True)
    r5 = forward_returns(p, horizon=2).to_numpy()[:, 0]
    np.testing.assert_allclose(r5, [3, 3, 3, nan, nan, nan], equal_nan=True)


def test_forward_returns_open_uses_dividend_factor():
    p = _panel()
    p.fields["open"] = np.array([[10.0], [10.0], [10.0], [10.0], [10.0], [10.0]])
    p.fields["close"] = np.full((6, 1), 10.0)
    # dividend of 10% between day 2 and 3: adj_close/close drops from 1 to... back-adjusted
    p.fields["adj_close"] = np.array([[9.0], [9.0], [9.0], [10.0], [10.0], [10.0]])
    r = forward_returns(p, horizon=1, execution="open").to_numpy()[:, 0]
    np.testing.assert_allclose(r, [0, 10 / 9 - 1, 0, 0, nan, nan], equal_nan=True)


def test_forward_returns_refuses_delay0_by_default():
    with pytest.raises(ValueError):
        forward_returns(_panel(), horizon=1, lag=0)
    r = forward_returns(_panel(), horizon=1, lag=0, allow_delay0=True)
    assert r.shape == (6, 1)


def test_only_lag_module_shifts_forward():
    """The ONE-lag-function rule: no negative shifts/forward indexing outside lag.py."""
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "alpha101"
    offenders = []
    for f in src.rglob("*.py"):
        if f.name == "lag.py":
            continue
        text = f.read_text()
        if re.search(r"shift\(\s*-", text) or re.search(r"\[\s*\w+\s*\+\s*(lag|horizon)", text):
            offenders.append(f.name)
    assert offenders == []


def test_raw_basis_reconstructs_as_traded_prices():
    # adjusted (Yahoo) history for a 4:1 split effective day 3
    p = _panel(split=np.array([[0.0], [0.0], [0.0], [4.0], [0.0], [0.0]]))
    p.fields["close"] = np.array([[2.5], [2.75], [3.0], [3.0], [3.3], [3.6]])
    p.fields["volume"] = np.array([[400.0], [400.0], [400.0], [100.0], [100.0], [100.0]])
    raw = p.to_raw_basis()
    np.testing.assert_allclose(raw.fields["close"][:, 0], [10, 11, 12, 3, 3.3, 3.6])
    np.testing.assert_allclose(raw.fields["volume"][:, 0], [100, 100, 100, 100, 100, 100])
    np.testing.assert_allclose(raw.fields["split_age"][3:, 0], [0, 1, 2])
    # a 2-day lookback (delta 2) straddling the split is masked; returns are untouched
    out = Evaluator(raw).evaluate("delta(close, 2)").to_numpy()[:, 0]
    assert np.isnan(out[3]) and np.isnan(out[4])
    assert out[5] == pytest.approx(0.6)


def test_scale_invariant_flags_are_reproducible():
    base = random_panel(T=300, N=25, seed=5)
    f = np.random.default_rng(1).choice([0.5, 2.0, 10.0], size=25)
    A = load_alphas()
    for i in (6, 12, 101):
        a = Evaluator(base).evaluate(A[i].expression).rank(axis=1).to_numpy()[-60:]
        b = Evaluator(rescaled(base, f)).evaluate(A[i].expression).rank(axis=1).to_numpy()[-60:]
        same = np.allclose(a, b, equal_nan=True)
        assert same == A[i].price_scale_invariant, i


# ---- delisting exit convention (amend-1) ----
def _ended_panel(adj_cols, T=12):
    """T-row panel; adj_cols: list of per-symbol adj_close lists (NaN-padded to T). open = adj + 1."""
    adj = np.full((T, len(adj_cols)), np.nan)
    for j, c in enumerate(adj_cols):
        adj[: len(c), j] = c
    fields = {"open": adj + 1.0, "high": adj + 1.0, "low": adj, "close": adj.copy(), "volume": np.ones_like(adj),
              "adj_close": adj}
    return Panel(dates=pd.bdate_range("2024-01-01", periods=T),
                 symbols=pd.Index([f"S{j}" for j in range(len(adj_cols))]), fields=fields)


def test_delisting_last_close_default_and_none():
    p = _ended_panel([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]])   # L=5 < T-1-5=6: ended
    r = forward_returns(p, horizon=2).to_numpy()[:, 0]
    # t=3: entry 16 (row 4), exit row 6 missing -> last close 32; t=4: entry 32, exit 32; t=5: entry missing
    np.testing.assert_allclose(r[:7], [3, 3, 3, 1, 0, nan, nan], equal_nan=True)
    assert np.isnan(r[7:]).all()
    n = forward_returns(p, horizon=2, delisting="none").to_numpy()[:, 0]
    np.testing.assert_allclose(n[:7], [3, 3, 3, nan, nan, nan, nan], equal_nan=True)
    assert np.isnan(n[7:]).all()
    # open execution: entry = open (adj + 1) at t+1, exit = last adj_close
    ro = forward_returns(p, horizon=2, execution="open").to_numpy()[:, 0]
    np.testing.assert_allclose(ro[3:5], [32 / 17 - 1, 32 / 33 - 1])
    assert ro[0] == pytest.approx(9 / 3 - 1) and np.isnan(ro[5])


def test_delisting_shumway_moves_only_exit_cells():
    p = _ended_panel([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]])
    r = forward_returns(p, horizon=2, delisting="shumway").to_numpy()[:, 0]
    np.testing.assert_allclose(r[:5], [3, 3, 3, 2 * 0.7 - 1, 1 * 0.7 - 1])
    assert np.isnan(r[5:]).all()


def test_delisting_two_symbols_and_grace():
    # S0 ended at L=4; S1 trades to L=T-3 (within grace) -> not ended; S2 trades to the end
    full = [float(i + 1) for i in range(12)]
    p = _ended_panel([[1.0, 2.0, 4.0, 8.0, 16.0], full[:9], full])
    r = forward_returns(p, horizon=3).to_numpy()
    assert r[2, 0] == pytest.approx(16 / 8 - 1) and r[3, 0] == pytest.approx(0.0)   # entry row 4 = L
    assert np.isnan(r[4, 0])
    nn = forward_returns(p, horizon=3, delisting="none").to_numpy()
    np.testing.assert_allclose(r[:, 1:], nn[:, 1:], equal_nan=True)   # not-ended symbols untouched
    assert np.isnan(r[5, 1]) and r[4, 1] == pytest.approx(9 / 6 - 1)
    assert np.isnan(r[9, 2])


def test_delisting_internal_gap_and_entry_beyond_last():
    a = [1.0, 2.0, np.nan, 8.0, 16.0, 32.0]
    p = _ended_panel([a])
    r = forward_returns(p, horizon=1).to_numpy()[:, 0]
    # t=0: exit row 2 is an internal gap -> NaN; t=1: entry row 2 NaN -> NaN
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert r[2] == pytest.approx(16 / 8 - 1)
    assert r[4] == pytest.approx(0.0)       # entry row 5 = L, exit moved to L
    assert np.isnan(r[5:]).all()            # entry beyond L stays NaN


def test_delisting_bad_value_raises():
    with pytest.raises(ValueError):
        forward_returns(_panel(), horizon=1, delisting="bogus")


def test_delisting_zero_price_is_data():
    # (a) adj goes to 0 and stays 0 to the panel end: not ended, same as "none"
    p = _ended_panel([[4.0, 4.0, 4.0, 0.0] + [0.0] * 8])
    r = forward_returns(p, horizon=2).to_numpy()
    np.testing.assert_allclose(r, forward_returns(p, horizon=2, delisting="none").to_numpy(), equal_nan=True)
    assert r[1, 0] == pytest.approx(-1.0)
    # (b) adj goes to 0 at row L=4 then NaN: ended; the moved exit is 0, r = -1
    q = _ended_panel([[4.0, 4.0, 4.0, 4.0, 0.0]])
    rq = forward_returns(q, horizon=3).to_numpy()[:, 0]
    assert rq[2] == pytest.approx(-1.0) and rq[0] == pytest.approx(-1.0)
    assert np.isnan(forward_returns(q, horizon=3, delisting="none").to_numpy()[2, 0])


def test_delisting_grace_boundary():
    import alpha101.lag as lag
    T, G = 12, lag.DELIST_GRACE
    assert (T, G) == (12, 5)
    def exit_filled(L):
        p = _ended_panel([[float(i + 1) for i in range(L + 1)]], T=T)
        return np.isfinite(forward_returns(p, horizon=2).to_numpy()[L - 2, 0])   # entry row L-1, exit beyond L
    assert not exit_filled(6)    # L = T-1-G: not ended
    assert exit_filled(5)        # L = T-2-G: ended
