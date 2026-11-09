"""Deterministic synthetic panels for tests, timing and invariance checks (no market data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


def random_panel(T: int = 300, N: int = 40, seed: int = 0, n_sectors: int = 4,
                 n_industries: int = 8) -> Panel:
    """Geometric random-walk OHLCV panel with sector/industry labels and a full universe."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.02, size=(T, N))
    close = 50.0 * np.exp(np.cumsum(r, axis=0)) * rng.uniform(0.2, 5.0, size=N)
    gap = rng.normal(0, 0.005, size=(T, N))
    open_ = close * np.exp(gap) * np.exp(-r / 2)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, size=(T, N))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, size=(T, N))))
    vol = np.round(rng.lognormal(13, 0.5, size=(T, N)) * rng.uniform(0.2, 5.0, size=N))
    dates = pd.bdate_range("2015-01-02", periods=T)
    symbols = pd.Index([f"S{i:03d}" for i in range(N)])
    industries = np.arange(N) % n_industries
    sectors = industries % n_sectors
    return Panel(dates=dates, symbols=symbols,
                 fields={"open": open_, "high": hi, "low": lo, "close": close, "volume": vol,
                         "adj_close": close.copy(), "shares_outstanding": np.full((T, N), 1e8)},
                 groups={"sector": sectors, "industry": industries})


def rescaled(panel: Panel, factors: np.ndarray) -> Panel:
    """Same panel with symbol s's prices multiplied by factors[s] and volume divided by it.

    This is exactly what a split-adjustment basis change does to one symbol's history.
    """
    f = np.asarray(factors, dtype=float)[None, :]
    fields = dict(panel.fields)
    for k in ("open", "high", "low", "close", "adj_close", "vwap"):
        if k in fields:
            fields[k] = fields[k] * f
    if "volume" in fields:
        fields["volume"] = fields["volume"] / f
    if "shares_outstanding" in fields:
        fields["shares_outstanding"] = fields["shares_outstanding"] / f
    return Panel(dates=panel.dates, symbols=panel.symbols, fields=fields, universe=panel.universe,
                 groups=panel.groups, vwap_proxy=panel.vwap_proxy)
