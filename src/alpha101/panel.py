"""Panel: the wide (dates x symbols) data container the evaluator reads from.

Raw fields (split-adjusted, NOT dividend-adjusted, as Yahoo serves them with auto_adjust=False):
open, high, low, close, volume, adj_close; optional shares_outstanding or cap.
Derived fields (docs/design.md section "Data layer"):
  returns  = adj_close / adj_close.shift(1) - 1      (total return, dividends included)
  vwap     = vwap_proxy(open, high, low, close)      (default: typical price (H+L+C)/3)
  adv{d}   = d-day mean of volume ("shares") or of close*volume ("dollar"); see EvalOptions
  cap      = given cap, else close * shares_outstanding
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

RAW_FIELDS = ("open", "high", "low", "close", "volume", "adj_close")
PRICE_FIELDS = ("open", "high", "low", "close", "vwap")
GROUP_CHAIN = {  # finest level first; "market" (everyone in one group) always last
    "subindustry": ("subindustry", "industry", "sector", "market"),
    "industry": ("industry", "sector", "market"),
    "sector": ("sector", "market"),
    "market": ("market",),
}


def vwap_typical(o, h, l, c):
    """(high + low + close) / 3 -- the default vwap proxy."""
    return (h + l + c) / 3.0


def vwap_ohlc4(o, h, l, c):
    """(open + high + low + close) / 4 -- alternative vwap proxy."""
    return (o + h + l + c) / 4.0


VWAP_PROXIES: dict[str, Callable] = {"typical": vwap_typical, "ohlc4": vwap_ohlc4}


class MissingField(KeyError):
    pass


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    fields: dict[str, np.ndarray]
    universe: np.ndarray | None = None            # bool (T, N); None = everyone, always
    groups: dict[str, np.ndarray] = field(default_factory=dict)  # level -> int codes (N,) or (T, N)
    vwap_proxy: str = "typical"
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        T, N = len(self.dates), len(self.symbols)
        if not self.dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted ascending")
        for k, v in self.fields.items():
            if v.shape != (T, N):
                raise ValueError(f"field {k} has shape {v.shape}, expected {(T, N)}")
        if self.universe is not None and self.universe.shape != (T, N):
            raise ValueError("universe mask shape mismatch")

    @property
    def basis(self) -> str:
        return "raw" if "split_age" in self.fields else "adjusted"

    def to_raw_basis(self) -> "Panel":
        """As-traded prices/volume rebuilt from the split column (docs/design.md, Data layer).

        Yahoo serves history split-adjusted to today's share basis. A split of ratio r on day u
        (Yahoo `split` = r, e.g. 4.0 for 4:1) means every raw price before u is r times the
        adjusted one. So raw[t] = adjusted[t] * prod_{u > t} r_u and volume divides by the same
        factor. adj_close is left alone (it only feeds `returns` and forward returns).
        Adds `split_age` = trading days since the symbol's last split (large if none); the
        evaluator masks cells whose lookback window straddles a split (split_age < lookback).
        """
        if self.basis == "raw":
            return self
        if "split" not in self.fields:
            raise MissingField("raw basis needs the 'split' field (0 or NaN = no split)")
        sp = np.array(self.fields["split"], dtype=float, copy=True)
        is_split = np.isfinite(sp) & (sp > 0) & (sp != 1.0)
        r = np.where(is_split, sp, 1.0)
        # F[t] = prod_{u > t} r[u]: reverse cumulative product, shifted by one row
        rev = np.cumprod(r[::-1], axis=0)[::-1]
        F = np.ones_like(r)
        F[:-1] = rev[1:]
        fields = dict(self.fields)
        for k in PRICE_FIELDS:
            if k in fields:
                fields[k] = fields[k] * F
        fields["volume"] = fields["volume"] / F
        if "shares_outstanding" in fields:
            fields["shares_outstanding"] = fields["shares_outstanding"] / F
        T = len(self.dates)
        idx = np.where(is_split, np.arange(T)[:, None], -10**9)
        last = np.maximum.accumulate(idx, axis=0)
        fields["split_age"] = (np.arange(T)[:, None] - last).astype(float)
        return Panel(dates=self.dates, symbols=self.symbols, fields=fields, universe=self.universe,
                     groups=self.groups, vwap_proxy=self.vwap_proxy)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.symbols)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], **kw) -> "Panel":
        """Build from wide DataFrames (index = dates, columns = symbols), aligned on the union."""
        dates = sorted(set().union(*[set(f.index) for f in frames.values()]))
        symbols = sorted(set().union(*[set(f.columns) for f in frames.values()]))
        dates = pd.DatetimeIndex(dates)
        symbols = pd.Index(symbols)
        arrs = {k: f.reindex(index=dates, columns=symbols).to_numpy(dtype=float, copy=True)
                for k, f in frames.items()}
        return cls(dates=dates, symbols=symbols, fields=arrs, **kw)

    @classmethod
    def from_long(cls, df: pd.DataFrame, **kw) -> "Panel":
        """Build from a long frame with columns symbol, date and any of RAW_FIELDS."""
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        cols = [c for c in d.columns if c not in ("symbol", "date")]
        frames = {c: d.pivot(index="date", columns="symbol", values=c).astype(float) for c in cols}
        return cls.from_frames(frames, **kw)

    def reindex_dates(self, dates: pd.DatetimeIndex) -> "Panel":
        """Same panel on another (sorted) date index; new dates are NaN, split is 0."""
        dates = pd.DatetimeIndex(sorted(dates))
        fields = {}
        for k, v in self.fields.items():
            f = pd.DataFrame(v, index=self.dates).reindex(dates)
            fields[k] = f.fillna(0.0).to_numpy(dtype=float, copy=True) if k == "split" \
                else f.to_numpy(dtype=float, copy=True)
        uni = None
        if self.universe is not None:
            uni = pd.DataFrame(self.universe, index=self.dates).reindex(dates).fillna(False) \
                .to_numpy(dtype=bool)
        return Panel(dates=dates, symbols=self.symbols, fields=fields, universe=uni,
                     groups=self.groups, vwap_proxy=self.vwap_proxy)

    def frame(self, x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, index=self.dates, columns=self.symbols)

    # ------------------------------------------------------------------ field access
    def get(self, name: str, adv_units: str = "shares") -> np.ndarray:
        key = (name, adv_units if name.startswith("adv") else None)
        if key in self._cache:
            return self._cache[key]
        val = self._derive(name, adv_units)
        val.setflags(write=False)  # shared by every expression; never mutate in place
        self._cache[key] = val
        return val

    def _raw(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise MissingField(f"panel has no field {name!r}")
        return self.fields[name]

    def _derive(self, name: str, adv_units: str) -> np.ndarray:
        from . import operators as op
        if name in ("open", "high", "low", "close", "volume", "adj_close"):
            return np.array(self._raw(name), dtype=float, copy=True)
        if name == "returns":
            p = self._raw("adj_close")
            return op.clean(p / op.delay(p, 1) - 1.0)
        if name == "vwap":
            if "vwap" in self.fields:
                return np.array(self.fields["vwap"], dtype=float, copy=True)
            f = VWAP_PROXIES[self.vwap_proxy]
            return f(*(self._raw(k) for k in ("open", "high", "low", "close")))
        if name == "cap":
            if "cap" in self.fields:
                return np.array(self.fields["cap"], dtype=float, copy=True)
            if "shares_outstanding" in self.fields:
                return self._raw("close") * self._raw("shares_outstanding")
            raise MissingField("cap needs a 'cap' or 'shares_outstanding' field")
        if name.startswith("adv"):
            d = int(name[3:])
            if adv_units == "shares":
                base = self._raw("volume")
            elif adv_units == "dollar":
                base = self._raw("close") * self._raw("volume")
            else:
                raise ValueError("adv_units must be 'shares' or 'dollar'")
            return op.ts_sum(base, d) / d
        raise MissingField(f"unknown field {name!r}")

    def group_chain(self, level: str) -> list[np.ndarray]:
        """Label arrays for `level` and its coarser fallbacks (see GROUP_CHAIN)."""
        chain = []
        for lv in GROUP_CHAIN[level]:
            if lv == "market":
                chain.append(np.zeros(len(self.symbols), dtype=np.int64))
            elif lv in self.groups:
                chain.append(np.asarray(self.groups[lv], dtype=np.int64))
        if len(chain) == 1 and level != "market":
            raise MissingField(f"no industry labels for {level!r} (or any coarser level) in panel")
        return chain

    @staticmethod
    def encode_groups(labels: pd.Series, symbols: pd.Index) -> np.ndarray:
        """Map a symbol -> label Series to int codes aligned to `symbols` (-1 = missing)."""
        s = labels.reindex(symbols)
        codes, _ = pd.factorize(s, use_na_sentinel=True)
        return codes.astype(np.int64)
