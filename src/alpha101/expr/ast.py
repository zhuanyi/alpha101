"""AST node types for alpha expressions.

Nodes are frozen dataclasses so they are hashable and can be used as memoisation keys.
`to_str()` renders a canonical, fully parenthesised string: parsing it again yields an equal AST.
"""
from __future__ import annotations

from dataclasses import dataclass


class Node:
    def to_str(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_str()


def _fmt_num(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


@dataclass(frozen=True)
class Num(Node):
    value: float

    def to_str(self) -> str:
        return _fmt_num(self.value)


@dataclass(frozen=True)
class Var(Node):
    """A data field: open, high, low, close, volume, returns, vwap, cap, adv<d>."""
    name: str

    def to_str(self) -> str:
        return self.name


@dataclass(frozen=True)
class Group(Node):
    """An industry-classification level, written `IndClass.<level>` in the paper."""
    level: str

    def to_str(self) -> str:
        return f"indclass.{self.level}"


@dataclass(frozen=True)
class Unary(Node):
    op: str  # only "-" survives parsing; unary "+" is dropped
    operand: Node

    def to_str(self) -> str:
        return f"(-{self.operand.to_str()})"


@dataclass(frozen=True)
class Binary(Node):
    op: str  # + - * / ^ < > <= >= == != || &&
    left: Node
    right: Node

    def to_str(self) -> str:
        return f"({self.left.to_str()} {self.op} {self.right.to_str()})"


@dataclass(frozen=True)
class Ternary(Node):
    cond: Node
    if_true: Node
    if_false: Node

    def to_str(self) -> str:
        return f"({self.cond.to_str()} ? {self.if_true.to_str()} : {self.if_false.to_str()})"


@dataclass(frozen=True)
class Call(Node):
    name: str  # canonical lower-case operator name
    args: tuple[Node, ...]

    def to_str(self) -> str:
        return f"{self.name}({', '.join(a.to_str() for a in self.args)})"


def walk(node: Node):
    """Yield every node of the tree, parents before children."""
    yield node
    if isinstance(node, Unary):
        yield from walk(node.operand)
    elif isinstance(node, Binary):
        yield from walk(node.left)
        yield from walk(node.right)
    elif isinstance(node, Ternary):
        yield from walk(node.cond)
        yield from walk(node.if_true)
        yield from walk(node.if_false)
    elif isinstance(node, Call):
        for a in node.args:
            yield from walk(a)
