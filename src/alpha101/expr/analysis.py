"""Static analysis of parsed expressions: inputs, industry need, lookback."""
from __future__ import annotations

import math

from .ast import Binary, Call, Group, Node, Num, Ternary, Unary, Var, walk

# operators whose last argument is a window length d (floored, see docs/operators.md)
WINDOW_OPS = {"delay", "delta", "correlation", "covariance", "decay_linear", "ts_min", "ts_max",
              "ts_argmin", "ts_argmax", "ts_rank", "sum", "product", "stddev"}
# lookback added by an op with window d, on top of its inputs' lookback
_SHIFT_OPS = {"delay", "delta"}  # need x[t-d]          -> +d
# every other window op uses x[t-d+1 .. t]               -> +(d-1)


def window(value: float) -> int:
    """The ONE window-rounding rule: floor(d) (paper A.2), never below 1."""
    d = int(math.floor(value + 1e-9))
    if d < 1:
        raise ValueError(f"window {value} floors to {d} < 1")
    return d


def inputs(node: Node) -> list[str]:
    """Sorted data fields an expression reads."""
    return sorted({n.name for n in walk(node) if isinstance(n, Var)})


def needs_industry(node: Node) -> bool:
    return any(isinstance(n, Group) for n in walk(node))


def group_levels(node: Node) -> list[str]:
    return sorted({n.level for n in walk(node) if isinstance(n, Group)})


def field_lookback(name: str) -> int:
    if name == "returns":
        return 1
    if name.startswith("adv"):
        return int(name[3:]) - 1
    return 0


def lookback(node: Node) -> int:
    """Trading days of history before t needed for a non-NaN value at t (strict NaN policy)."""
    if isinstance(node, (Num, Group)):
        return 0
    if isinstance(node, Var):
        return field_lookback(node.name)
    if isinstance(node, Unary):
        return lookback(node.operand)
    if isinstance(node, Binary):
        return max(lookback(node.left), lookback(node.right))
    if isinstance(node, Ternary):
        return max(lookback(node.cond), lookback(node.if_true), lookback(node.if_false))
    if isinstance(node, Call):
        if node.name in WINDOW_OPS:
            *data_args, w = node.args
            if not isinstance(w, Num):
                raise ValueError(f"{node.name}: window must be a numeric literal, got {w.to_str()}")
            d = window(w.value)
            base = max(lookback(a) for a in data_args)
            return base + (d if node.name in _SHIFT_OPS else d - 1)
        return max((lookback(a) for a in node.args), default=0)
    raise TypeError(type(node))
