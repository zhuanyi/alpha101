"""Shared fixtures for the Phase 1 overnight tests: puts research/overnight on sys.path and builds synthetic data."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OVERNIGHT = os.path.join(ROOT, "research", "overnight")
if OVERNIGHT not in sys.path:
    sys.path.insert(0, OVERNIGHT)

SYMBOLS = ["SPY", "QQQ", "IWM", "XIU.TO", "XIC.TO", "SSO", "UPRO", "TQQQ", "QLD"]


def _frame(rng, idx, drift=0.0002, vol=0.01):
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0003, 0.004, len(idx)))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close,
                         "adj_close": close * 0.99, "volume": rng.integers(1_000_000, 2_000_000, len(idx)).astype(float),
                         "dividend": 0.0, "split": 0.0}, index=idx)


@pytest.fixture(scope="session")
def synthetic():
    """Synthetic daily frames 2003-2022 (crosses the 2019 holdout boundary) and a calendar of events."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2003-01-02", "2022-12-30", name="date")
    frames = {s: _frame(rng, idx) for s in SYMBOLS}
    vix = pd.DataFrame({"close": 15 + 5 * np.abs(np.sin(np.arange(len(idx)) / 40.0)) + rng.normal(0, 1, len(idx))}, index=idx)
    for c in ("open", "high", "low", "adj_close"):
        vix[c] = vix["close"]
    vix["volume"] = 0.0
    vix["dividend"] = 0.0
    vix["split"] = 0.0
    frames["^VIX"] = vix
    ev = []
    for y in range(2003, 2023):
        for m in range(1, 13):
            first = idx[(idx.year == y) & (idx.month == m)]
            ev.append(("US_CPI", first[9], 1))
            ev.append(("US_NFP", first[3], 1))
            if m in (1, 3, 5, 6, 7, 9, 11, 12):
                ev.append(("FOMC_STATEMENT", first[14], 1))
    events = pd.DataFrame(ev, columns=["event_type", "event_date", "scheduled"])
    return frames, events
