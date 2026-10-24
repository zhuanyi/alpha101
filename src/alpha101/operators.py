"""Vectorised operators of "101 Formulaic Alphas" (paper Appendix A.2).

Every operator takes and returns 2-D float64 numpy arrays shaped (T dates, N symbols), time on
axis 0 in ascending date order. Semantics are specified in docs/operators.md; the short version:

* Windows cover the past d rows INCLUDING today: x[t-d+1 .. t]. delay/delta reach x[t-d].
* Strict NaN policy: a time-series window containing any NaN yields NaN.
* +/-inf produced by any operator is converted to NaN (see `clean`).
* Cross-sectional operators (rank, scale, indneutralize) only see cells where `mask` is True
  (the point-in-time universe, less warm-up and raw-basis split-window cells; see `Evaluator`)
  and return NaN elsewhere.

pandas 3 note: with copy-on-write, `DataFrame.to_numpy()` may hand back a read-only view, so
every helper that writes into an array obtains it with `np.array(..., copy=True)` first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ARGMAX_CONVENTIONS = ("oldest_1based", "days_ago")
# keep each sliding-window temporary below ~256 MB
_CHUNK_BYTES = 256 * 1024 * 1024


def clean(x: np.ndarray) -> np.ndarray:
    """Replace +/-inf with NaN (returns a new array)."""
    out = np.array(x, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --------------------------------------------------------------------------- element-wise
def add(a, b): return _arith(np.add, a, b)
def sub(a, b): return _arith(np.subtract, a, b)
def mul(a, b): return _arith(np.multiply, a, b)
def div(a, b): return _arith(np.divide, a, b)


def power(a, b):
    """x ^ y. A negative base with a non-integer exponent gives NaN."""
    return _arith(np.power, a, b)


def _arith(fn, a, b):
    with np.errstate(all="ignore"):
        return clean(fn(_f(a), _f(b)))


def neg(a): return -_f(a)
def absolute(a): return np.abs(_f(a))
def sign(a): return np.sign(_f(a))  # sign(NaN) = NaN, sign(0) = 0


def log(a):
    """Natural log; x <= 0 -> NaN."""
    with np.errstate(all="ignore"):
        return clean(np.log(_f(a)))


def signedpower(x, a):
    """sign(x) * |x|^a. The paper writes x^a; the name and WorldQuant usage mean sign-preserving."""
    x = _f(x)
    with np.errstate(all="ignore"):
        return clean(np.sign(x) * np.power(np.abs(x), _f(a)))


def compare(op: str, a, b):
    """Comparison returning 1.0/0.0, and NaN where either side is NaN."""
    a, b = np.broadcast_arrays(_f(a), _f(b))
    fn = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal,
          "==": np.equal, "!=": np.not_equal}[op]
    out = fn(a, b).astype(float)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def logical(op: str, a, b):
    """|| and && on truthiness (non-zero = true); NaN if either side is NaN."""
    a, b = np.broadcast_arrays(_f(a), _f(b))
    fn = np.logical_or if op == "||" else np.logical_and
    out = fn(a != 0, b != 0).astype(float)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def where(cond, a, b):
    """cond ? a : b, with NaN where cond is NaN."""
    cond, a, b = np.broadcast_arrays(_f(cond), _f(a), _f(b))
    out = np.where(cond != 0, a, b).astype(float)
    out[np.isnan(cond)] = np.nan
    return out


def elementwise_min(a, b): return np.minimum(_f(a), _f(b))  # NaN propagates
def elementwise_max(a, b): return np.maximum(_f(a), _f(b))


# --------------------------------------------------------------------------- cross-sectional
def _masked(x, mask):
    x = np.array(x, dtype=float, copy=True)
    if mask is not None:
        x[~mask] = np.nan
    return x


def rank(x, mask=None):
    """Cross-sectional percentile rank r/n in (0, 1]; ties get the average rank; NaN ignored."""
    x = _masked(x, mask)
    return pd.DataFrame(x).rank(axis=1, method="average", pct=True).to_numpy(dtype=float, copy=True)


def scale(x, a=1.0, mask=None):
    """Rescale each row so that sum(|x|) = a. A row with sum(|x|) = 0 becomes NaN."""
    x = _masked(x, mask)
    s = np.nansum(np.abs(x), axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return clean(x * float(a) / s)


def indneutralize(x, labels_chain, mask=None, min_group_size=5):
    """Demean x within groups, falling back to coarser groups when a group is too small.

    labels_chain: list of int label arrays, finest first, each shaped (N,) or (T, N); -1 means
    "no label". On each row a cell is demeaned within the finest level whose group has at least
    `min_group_size` non-NaN members that row; the last level (normally "market", all zeros)
    always applies when it has >= 1 member. Cells with no usable label stay NaN.
    """
    x = _masked(x, mask)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    done = np.zeros((T, N), dtype=bool)
    valid = ~np.isnan(x)
    rows = np.broadcast_to(np.arange(T)[:, None], (T, N))
    for i, labels in enumerate(labels_chain):
        lab = np.broadcast_to(np.asarray(labels, dtype=np.int64), (T, N))
        need = 1 if i == len(labels_chain) - 1 else min_group_size
        ok = valid & (lab >= 0)
        G = int(lab.max()) + 1 if ok.any() else 0
        if G == 0:
            continue
        idx = rows[ok] * G + lab[ok]
        cnt = np.bincount(idx, minlength=T * G)
        tot = np.bincount(idx, weights=x[ok], minlength=T * G)
        cell = rows * G + np.where(lab >= 0, lab, 0)
        c = cnt[cell]
        use = ok & ~done & (c >= need)
        with np.errstate(all="ignore"):
            mean = tot[cell] / c
        out[use] = x[use] - mean[use]
        done |= use
    return out


# --------------------------------------------------------------------------- time-series helpers
def _rolling(x, d):
    return pd.DataFrame(_f(x)).rolling(d, min_periods=d)


def _np(df) -> np.ndarray:
    return clean(df.to_numpy(dtype=float, copy=True))


def _window_nan(x, d):
    """Boolean (T, N): True where the window x[t-d+1..t] is incomplete or contains NaN."""
    isn = np.isnan(x).astype(np.int64)
    c = np.cumsum(isn, axis=0)
    c_prev = np.zeros_like(c)
    c_prev[d:] = c[:-d]
    bad = (c - c_prev) > 0
    bad[: d - 1] = True
    return bad


def _sliding_apply(x, d, fn):
    """Apply fn(windows) -> (rows, cols) over windows shaped (rows, cols, d), oldest first.

    NaNs are zero-filled before fn sees them and the affected outputs are reset to NaN.
    Processed in column chunks to bound memory.
    """
    x = _f(x)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < d:
        return out
    bad = _window_nan(x, d)
    filled = np.where(np.isnan(x), 0.0, x)
    step = max(1, int(_CHUNK_BYTES // max(1, (T - d + 1) * d * 8)))
    for j in range(0, N, step):
        v = sliding_window_view(filled[:, j:j + step], d, axis=0)  # (T-d+1, n, d)
        out[d - 1:, j:j + step] = fn(v)
    out[bad] = np.nan
    return out


# --------------------------------------------------------------------------- time-series operators
def delay(x, d: int):
    """x[t-d]; the first d rows are NaN."""
    x = _f(x)
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d] if d > 0 else x
    return out


def delta(x, d: int):
    return _f(x) - delay(x, d)


def ts_sum(x, d: int):
    return _np(_rolling(x, d).sum())


def ts_product(x, d: int):
    return clean(_sliding_apply(x, d, lambda v: np.prod(v, axis=-1)))


def ts_stddev(x, d: int):
    """Sample standard deviation (ddof=1) over the window."""
    return _np(_rolling(x, d).std(ddof=1))


def ts_min(x, d: int):
    return _np(_rolling(x, d).min())


def ts_max(x, d: int):
    return _np(_rolling(x, d).max())


def ts_rank(x, d: int):
    """Percentile rank of today's value within x[t-d+1..t]: r/d in (0, 1], average ties."""
    def fn(v):
        last = v[..., -1:]
        less = (v < last).sum(axis=-1)
        eq = (v == last).sum(axis=-1)
        return (less + (eq + 1) / 2.0) / d
    return _sliding_apply(x, d, fn)


def _argext(x, d, which, convention):
    if convention not in ARGMAX_CONVENTIONS:
        raise ValueError(f"argmax convention must be one of {ARGMAX_CONVENTIONS}")
    pick = np.argmax if which == "max" else np.argmin  # ties -> earliest (oldest) day
    idx = _sliding_apply(x, d, lambda v: pick(v, axis=-1).astype(float))  # 0 = oldest
    if convention == "oldest_1based":
        return idx + 1.0          # oldest day = 1, today = d
    return (d - 1.0) - idx        # days ago: today = 0, oldest = d-1


def ts_argmax(x, d: int, convention="oldest_1based"):
    return _argext(x, d, "max", convention)


def ts_argmin(x, d: int, convention="oldest_1based"):
    return _argext(x, d, "min", convention)


def decay_linear(x, d: int):
    """Weighted mean with weights d (today), d-1, ..., 1 (oldest), normalised to sum to 1."""
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return _sliding_apply(x, d, lambda v: v @ w)


def covariance(x, y, d: int):
    """Sample covariance (ddof=1) over the window; NaN if either input has a NaN in it."""
    x, y = _joint(x, y)
    return _np(_rolling(x, d).cov(pd.DataFrame(y), ddof=1))


def correlation(x, y, d: int):
    """Pearson correlation over the window.

    Incomplete window (any NaN in x or y) -> NaN. Complete window where either side is
    (numerically) constant -> 0.0: the data were observed and show no linear co-movement.
    Returning NaN there would, under the strict NaN policy, wipe out every enclosing window op
    (alpha 96 is entirely NaN that way, because correlations of discrete ts_rank values are
    often flat over 3-day windows).
    """
    x, y = _joint(x, y)
    rx, ry = _rolling(x, d), _rolling(y, d)
    cov = rx.cov(pd.DataFrame(y), ddof=1).to_numpy(dtype=float, copy=True)
    vx = rx.var(ddof=1).to_numpy(dtype=float, copy=True)
    vy = ry.var(ddof=1).to_numpy(dtype=float, copy=True)
    mx2 = rx.mean().to_numpy(dtype=float, copy=True) ** 2
    my2 = ry.mean().to_numpy(dtype=float, copy=True) ** 2
    flat = (vx <= 1e-12 * mx2) | (vy <= 1e-12 * my2) | (vx <= 0) | (vy <= 0)
    with np.errstate(all="ignore"):
        r = cov / np.sqrt(vx * vy)
    r = clean(r)
    r[flat & ~np.isnan(cov)] = 0.0
    return np.clip(r, -1.0, 1.0)


def _joint(x, y):
    x, y = np.broadcast_arrays(_f(x), _f(y))
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    bad = np.isnan(x) | np.isnan(y)
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y
