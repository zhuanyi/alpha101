"""Evaluate an expression AST over a Panel, with per-panel memoisation of sub-expressions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import operators as op
from ..panel import Panel
from .analysis import WINDOW_OPS, lookback, window
from .ast import Binary, Call, Group, Node, Num, Ternary, Unary, Var
from .parser import parse


@dataclass(frozen=True)
class EvalOptions:
    """The ambiguous-in-the-paper choices, in one place (docs/operators.md section 3).

    argmax_convention: "oldest_1based" (oldest day = 1, today = d; matches the common public
        implementations) or "days_ago" (today = 0; WorldQuant BRAIN's later ts_arg_max).
    adv_units: "shares" (mean share volume; dimensionally consistent with `volume / adv20`
        in alphas 7, 17, 21, 43, 47, 100) or "dollar" (paper A.2 text: "dollar volume").
    min_group_size: smallest group indneutralize will demean within before falling back.
    """
    argmax_convention: str = "oldest_1based"
    adv_units: str = "shares"
    min_group_size: int = 5


def pit_valid(panel: Panel, L: int) -> np.ndarray:
    """Bool (T, N): cells where a value with lookback L is point-in-time computable.

    A cell is valid when it is in the universe, the symbol has at least L rows of history after
    its first finite close (warm-up; without this a ternary can emit early values from its short
    branch, e.g. alpha 1: `returns < 0 ? stddev(returns, 20) : close`), and, on the raw
    (as-traded) basis, its L-day window does not straddle the symbol's own split (that would mix
    share bases). Used both for the final output mask and as the mask of the cross-sectional
    operators, so invalid cells never enter a rank / scale sum / group mean. A symbol with no
    finite close is invalid on every row.
    """
    ok = np.ones(panel.shape, dtype=bool) if panel.universe is None \
        else np.array(panel.universe, dtype=bool, copy=True)
    if L > 0 and "close" in panel.fields:
        finite = np.isfinite(panel.fields["close"])
        # argmax of an all-False column is 0; such a symbol has no first close and is never valid
        first = np.where(finite.any(axis=0), np.argmax(finite, axis=0), panel.shape[0])
        rows = np.arange(panel.shape[0])[:, None]
        ok &= rows >= first[None, :] + L
    if "split_age" in panel.fields:
        ok &= ~(panel.fields["split_age"] < L)
    return ok


_BIN = {"+": op.add, "-": op.sub, "*": op.mul, "/": op.div, "^": op.power}
_CMP = {"<", ">", "<=", ">=", "==", "!="}


class Evaluator:
    def __init__(self, panel: Panel, options: EvalOptions | None = None):
        self.panel = panel
        self.opt = options or EvalOptions()
        self._memo: dict[Node, object] = {}
        self._valid_memo: dict[Node, np.ndarray] = {}

    def clear(self) -> None:
        """Drop memoised sub-expressions (bounds memory when one evaluator runs many alphas)."""
        self._memo.clear()
        self._valid_memo.clear()

    def evaluate(self, expr: str | Node, mask_output: bool = True) -> pd.DataFrame:
        """Evaluate and return a dates x symbols DataFrame (NaN outside the universe)."""
        node = parse(expr) if isinstance(expr, str) else expr
        val = self._eval(node)
        arr = np.broadcast_to(np.asarray(val, dtype=float), self.panel.shape)
        arr = np.array(arr, dtype=float, copy=True)
        if mask_output:
            arr[~self._valid(node)] = np.nan
        return self.panel.frame(arr)

    def _valid(self, node: Node) -> np.ndarray:
        """Bool (T, N): True where `node`'s value is point-in-time computable (pit_valid)."""
        hit = self._valid_memo.get(node)
        if hit is None:
            hit = pit_valid(self.panel, lookback(node))
            hit.setflags(write=False)
            self._valid_memo[node] = hit
        return hit

    # ------------------------------------------------------------------ recursion
    def _eval(self, node: Node):
        if isinstance(node, Num):
            return node.value
        hit = self._memo.get(node)
        if hit is not None:
            return hit
        val = self._compute(node)
        if isinstance(val, np.ndarray):
            val.setflags(write=False)  # memoised values are shared between parents
        self._memo[node] = val
        return val

    def _compute(self, node: Node):
        p, o = self.panel, self.opt
        if isinstance(node, Var):
            return p.get(node.name, adv_units=o.adv_units)
        if isinstance(node, Unary):
            return op.neg(self._eval(node.operand))
        if isinstance(node, Binary):
            a, b = self._eval(node.left), self._eval(node.right)
            if node.op in _BIN:
                return _BIN[node.op](a, b)
            if node.op in _CMP:
                return op.compare(node.op, a, b)
            return op.logical(node.op, a, b)
        if isinstance(node, Ternary):
            return op.where(self._eval(node.cond), self._eval(node.if_true), self._eval(node.if_false))
        if isinstance(node, Group):
            raise TypeError("IndClass.<level> is only valid as indneutralize's 2nd argument")
        if isinstance(node, Call):
            return self._call(node)
        raise TypeError(type(node))

    def _arr(self, node: Node) -> np.ndarray:
        """Evaluate to a full (T, N) array (broadcasting scalars)."""
        v = self._eval(node)
        return np.broadcast_to(np.asarray(v, dtype=float), self.panel.shape)

    def _call(self, node: Call):
        n, args = node.name, node.args
        if n in WINDOW_OPS:
            *data, w = args
            if not isinstance(w, Num):
                raise ValueError(f"{n}: window must be a numeric literal, got {w.to_str()}")
            d = window(w.value)
            xs = [self._arr(a) for a in data]
            if n == "delay":
                return op.delay(xs[0], d)
            if n == "delta":
                return op.delta(xs[0], d)
            if n in ("correlation", "covariance"):
                return getattr(op, n)(xs[0], xs[1], d)
            if n in ("ts_argmax", "ts_argmin"):
                return getattr(op, n)(xs[0], d, convention=self.opt.argmax_convention)
            fn = {"sum": op.ts_sum, "product": op.ts_product, "stddev": op.ts_stddev}.get(n) \
                or getattr(op, n)
            return fn(xs[0], d)
        if n == "rank":
            return op.rank(self._arr(args[0]), self._valid(args[0]))
        if n == "scale":
            a = 1.0
            if len(args) == 2:
                if not isinstance(args[1], Num):
                    raise ValueError("scale: a must be a numeric literal")
                a = args[1].value
            return op.scale(self._arr(args[0]), a, self._valid(args[0]))
        if n == "indneutralize":
            if not isinstance(args[1], Group):
                raise ValueError("indneutralize: 2nd argument must be IndClass.<level>")
            chain = self.panel.group_chain(args[1].level)
            return op.indneutralize(self._arr(args[0]), chain, self._valid(args[0]),
                                    self.opt.min_group_size)
        if n == "abs":
            return op.absolute(self._eval(args[0]))
        if n == "log":
            return op.log(self._eval(args[0]))
        if n == "sign":
            return op.sign(self._eval(args[0]))
        if n == "signedpower":
            return op.signedpower(self._eval(args[0]), self._eval(args[1]))
        if n == "elementwise_min":
            return op.elementwise_min(self._eval(args[0]), self._eval(args[1]))
        if n == "elementwise_max":
            return op.elementwise_max(self._eval(args[0]), self._eval(args[1]))
        raise ValueError(f"no implementation for {n}")


def evaluate(panel: Panel, expr: str | Node, options: EvalOptions | None = None) -> pd.DataFrame:
    """Convenience: evaluate one expression on a panel."""
    return Evaluator(panel, options).evaluate(expr)
