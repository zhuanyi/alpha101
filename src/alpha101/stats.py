"""Cross-sectional IC and time-series statistics used by the evaluation layer."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as _st


def rank_ic(alpha: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    """Daily Spearman rank IC between alpha[t] and fwd[t] (both dates x symbols).

    Only cells where both are non-NaN count. Dates with fewer than `min_names` such cells get
    NaN. Returns a DataFrame with columns ic and n, indexed by date.
    """
    a, f = alpha.align(fwd, join="inner")
    av = a.to_numpy(dtype=float, copy=True)
    fv = f.to_numpy(dtype=float, copy=True)
    bad = np.isnan(av) | np.isnan(fv)
    av[bad] = np.nan
    fv[bad] = np.nan
    ra = pd.DataFrame(av).rank(axis=1).to_numpy(dtype=float, copy=True)
    rf = pd.DataFrame(fv).rank(axis=1).to_numpy(dtype=float, copy=True)
    n = (~bad).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days -> NaN mean, handled below
        ra -= np.nanmean(ra, axis=1, keepdims=True)
        rf -= np.nanmean(rf, axis=1, keepdims=True)
        ic = np.nansum(ra * rf, axis=1) / np.sqrt(np.nansum(ra ** 2, axis=1) * np.nansum(rf ** 2, axis=1))
    ic[(n < min_names) | ~np.isfinite(ic)] = np.nan
    return pd.DataFrame({"ic": ic, "n": n}, index=a.index)


def newey_west_t(x, lags: int) -> float:
    """t-statistic of the mean of x with a Newey-West (Bartlett) HAC variance."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    if T < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / T
    for k in range(1, min(lags, T - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    return float(x.mean() / np.sqrt(s / T)) if s > 0 else float("nan")


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """Mean IC, IC-IR (mean/std, per period), NW t-stat with lags = horizon, hit rate, days."""
    x = ic.dropna()
    sd = x.std(ddof=1)
    return {
        "mean_ic": float(x.mean()) if len(x) else float("nan"),
        "ic_ir": float(x.mean() / sd) if len(x) > 1 and sd > 0 else float("nan"),
        "t_nw": newey_west_t(x, lags=horizon),
        "hit_rate": float((x > 0).mean()) if len(x) else float("nan"),
        "n_days": int(len(x)),
    }


def effective_number_of_tests(corr: np.ndarray) -> float:
    """(sum eig)^2 / sum(eig^2) of a correlation matrix."""
    c = np.asarray(corr, dtype=float)
    eig = np.linalg.eigvalsh(c)
    eig = np.clip(eig, 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def adjusted_threshold(n_eff: float, alpha: float = 0.05) -> float:
    """Two-sided |t| bar after a Sidak correction for n_eff independent tests."""
    p = 1 - (1 - alpha) ** (1 / max(n_eff, 1.0))
    return float(_st.norm.ppf(1 - p / 2))
