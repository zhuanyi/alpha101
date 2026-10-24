"""The ONE lag-enforcing function. Nothing else in the library may shift prices forward.

Convention (docs/operators.md section 4): an alpha value dated t uses data through the CLOSE of
day t. It is traded `lag` trading days later (default 1) at the chosen execution price and held
`horizon` trading days:

    execution="close": r[t] = P_close[t+lag+h] / P_close[t+lag] - 1     (default)
    execution="open":  r[t] = P_open [t+lag+h] / P_open [t+lag] - 1

P_close is adj_close (split + dividend adjusted). P_open is open scaled by the same day's
adj_close/close factor, so dividends paid inside the holding period are included.

lag=0 with execution="close" models the paper's "delay-0" alphas (42, 48, 53, 54), traded at
the close they are computed from; it is refused unless allow_delay0=True, because a retail
account cannot trade at a close it has not yet observed.

Delisting exit (amend-1): a symbol whose data end more than DELIST_GRACE rows before the panel end has
*ended*; a position whose exit falls after its last valid adj_close exits at that last traded close
(delisting="last_close", default), instead of dropping out as NaN ("none"). "shumway" additionally
applies the Shumway (1997) delisting return to the cells moved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel

DELIST_GRACE = 5          # rows; data ending within this many rows of the panel end is NOT a delisting
SHUMWAY_RETURN = -0.30    # Shumway (1997) performance-delisting return, off by default


def forward_returns(panel: Panel, horizon: int, execution: str = "close", lag: int = 1,
                    allow_delay0: bool = False, delisting: str = "last_close") -> pd.DataFrame:
    """Forward returns (T, N) frame; see the module docstring for the convention.

    delisting: "none" leaves a missing exit price NaN. "last_close" (default): for a symbol that has
    ended (last finite adj_close (0 counts) at row L < T - 1 - DELIST_GRACE) and a row whose entry row is <= L
    and whose exit row is > L, the exit is adj_close[L] (also for execution="open"). Internal gaps, missing
    entries and symbols that have not ended are untouched. "shumway": as "last_close", then
    r = (1 + r) * (1 + SHUMWAY_RETURN) - 1 on the moved cells only. We cannot tell performance delistings
    from M&A with our data, so this hits every ended symbol; it is an off-by-default sensitivity option.
    """
    if delisting not in ("none", "last_close", "shumway"):
        raise ValueError("delisting must be 'none', 'last_close' or 'shumway'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if lag < 0 or (lag == 0 and not allow_delay0):
        raise ValueError("lag must be >= 1 (lag=0 needs allow_delay0=True and execution='close')")
    if lag == 0 and execution != "close":
        raise ValueError("delay-0 is only defined for execution='close'")
    adj = panel.get("adj_close")
    if execution == "close":
        px = adj
    elif execution == "open":
        with np.errstate(all="ignore"):
            px = panel.get("open") * adj / panel.get("close")
    else:
        raise ValueError("execution must be 'close' or 'open'")
    T = px.shape[0]
    entry = np.full_like(px, np.nan)
    exit_ = np.full_like(px, np.nan)
    if lag < T:
        entry[: T - lag] = px[lag:]
    if lag + horizon < T:
        exit_[: T - lag - horizon] = px[lag + horizon:]
    moved = np.zeros(px.shape, dtype=bool)
    if delisting != "none" and T:
        valid = np.isfinite(adj)   # a zero price is data (total loss), not the end of the data
        last = np.where(valid.any(axis=0), T - 1 - np.argmax(valid[::-1], axis=0), -1)   # L_j (-1: no data)
        ended = (last >= 0) & (last < T - 1 - DELIST_GRACE)
        exit_last = adj[np.maximum(last, 0), np.arange(px.shape[1])]
        er = np.arange(T)[:, None] + lag   # entry row of each alpha row
        moved = (ended[None, :] & np.isfinite(entry) & (er <= last[None, :])
                 & (last[None, :] < er + horizon))
        exit_ = np.where(moved, exit_last[None, :], exit_)
    with np.errstate(all="ignore"):
        r = exit_ / entry - 1.0
        if delisting == "shumway":
            r = np.where(moved, (1.0 + r) * (1.0 + SHUMWAY_RETURN) - 1.0, r)
    r[~np.isfinite(r)] = np.nan
    return panel.frame(r)
