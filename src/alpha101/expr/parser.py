"""Tokeniser and Pratt parser for the "101 Formulaic Alphas" expression syntax.

Grammar (lowest to highest precedence; all binary operators left-associative except `^`):

    expr     := or_expr [ "?" expr ":" expr ]          (right-associative ternary)
    or_expr  := and_expr { "||" and_expr }
    and_expr := eq_expr  { "&&" eq_expr }
    eq_expr  := rel_expr { ("==" | "!=") rel_expr }
    rel_expr := add_expr { ("<" | ">" | "<=" | ">=") add_expr }
    add_expr := mul_expr { ("+" | "-") mul_expr }
    mul_expr := unary    { ("*" | "/") unary }
    unary    := ("-" | "+") unary | power
    power    := primary [ "^" unary ]                  (right-associative; -x^2 == -(x^2))
    primary  := NUMBER | NAME | NAME "(" [expr {"," expr}] ")" | "IndClass." NAME | "(" expr ")"

Names are case-insensitive (paper, Appendix A.2). Function aliases are normalised to one
canonical name (e.g. `Ts_ArgMax` -> `ts_argmax`, `SignedPower` -> `signedpower`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ast import Binary, Call, Group, Node, Num, Ternary, Unary, Var

FIELDS = {"open", "high", "low", "close", "volume", "returns", "vwap", "cap"}
ADV_RE = re.compile(r"^adv(\d+)$")
GROUP_LEVELS = {"market", "sector", "industry", "subindustry"}

# canonical function name -> (min_args, max_args)
FUNCTIONS: dict[str, tuple[int, int]] = {
    "abs": (1, 1), "log": (1, 1), "sign": (1, 1),
    "rank": (1, 1), "scale": (1, 2), "indneutralize": (2, 2),
    "delay": (2, 2), "delta": (2, 2),
    "correlation": (3, 3), "covariance": (3, 3),
    "signedpower": (2, 2), "decay_linear": (2, 2),
    "ts_min": (2, 2), "ts_max": (2, 2), "ts_argmin": (2, 2), "ts_argmax": (2, 2),
    "ts_rank": (2, 2), "sum": (2, 2), "product": (2, 2), "stddev": (2, 2),
    # min/max are resolved after parsing: numeric 2nd arg -> ts_min/ts_max, else elementwise
    "min": (2, 2), "max": (2, 2),
    "elementwise_min": (2, 2), "elementwise_max": (2, 2),
}
ALIASES = {"ts_sum": "sum", "ts_product": "product", "ts_stddev": "stddev",
           "corr": "correlation", "cov": "covariance"}

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)
  | (?P<op>\|\||&&|<=|>=|==|!=|[-+*/^<>?:(),])
""", re.VERBOSE)


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str  # num | name | op | eof
    text: str
    pos: int


def tokenize(src: str) -> list[Token]:
    out: list[Token] = []
    i = 0
    while i < len(src):
        m = TOKEN_RE.match(src, i)
        if not m:
            raise ParseError(f"unexpected character {src[i]!r} at {i} in {src!r}")
        kind = m.lastgroup
        if kind != "ws":
            out.append(Token(kind, m.group(), i))
        i = m.end()
    out.append(Token("eof", "", len(src)))
    return out


# binding powers for infix operators
_INFIX = {
    "||": 10, "&&": 20,
    "==": 30, "!=": 30,
    "<": 40, ">": 40, "<=": 40, ">=": 40,
    "+": 50, "-": 50,
    "*": 60, "/": 60,
}
_TERNARY_BP = 5
_UNARY_BP = 70


class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Token:
        t = self.next()
        if t.text != text:
            raise ParseError(f"expected {text!r} at {t.pos}, got {t.text!r} in {self.src!r}")
        return t

    def parse(self) -> Node:
        node = self.expr(0)
        t = self.peek()
        if t.kind != "eof":
            raise ParseError(f"trailing input {t.text!r} at {t.pos} in {self.src!r}")
        return node

    def expr(self, min_bp: int) -> Node:
        left = self.prefix()
        while True:
            t = self.peek()
            if t.kind != "op":
                break
            if t.text == "?":
                if _TERNARY_BP < min_bp:
                    break
                self.next()
                a = self.expr(0)
                self.expect(":")
                b = self.expr(_TERNARY_BP)  # right-associative
                left = Ternary(left, a, b)
                continue
            bp = _INFIX.get(t.text)
            if bp is None or bp <= min_bp:
                break
            self.next()
            right = self.expr(bp)
            left = Binary(t.text, left, right)
        return left

    def prefix(self) -> Node:
        t = self.peek()
        if t.kind == "op" and t.text in "-+":
            self.next()
            operand = self.expr(_UNARY_BP)
            if t.text == "+":
                return operand
            if isinstance(operand, Num):
                return Num(-operand.value)
            return Unary("-", operand)
        node = self.primary()
        if self.peek().text == "^":
            self.next()
            # prefix() parses a (possibly negated) primary and recurses on a further "^",
            # which makes "^" right-associative and lets the exponent carry a unary minus.
            node = Binary("^", node, self.prefix())
        return node

    def primary(self) -> Node:
        t = self.next()
        if t.kind == "num":
            return Num(float(t.text))
        if t.kind == "op" and t.text == "(":
            node = self.expr(0)
            self.expect(")")
            return node
        if t.kind == "name":
            low = t.text.lower()
            if low.startswith("indclass."):
                level = low.split(".", 1)[1]
                if level not in GROUP_LEVELS:
                    raise ParseError(f"unknown IndClass level {level!r} in {self.src!r}")
                return Group(level)
            if self.peek().text == "(":
                self.next()
                args: list[Node] = []
                if self.peek().text != ")":
                    args.append(self.expr(0))
                    while self.peek().text == ",":
                        self.next()
                        args.append(self.expr(0))
                self.expect(")")
                return _make_call(low, args, self.src)
            if low in FIELDS or ADV_RE.match(low):
                return Var(low)
            raise ParseError(f"unknown field {t.text!r} at {t.pos} in {self.src!r}")
        raise ParseError(f"unexpected token {t.text!r} at {t.pos} in {self.src!r}")


def _make_call(name: str, args: list[Node], src: str) -> Call:
    name = ALIASES.get(name, name)
    if name not in FUNCTIONS:
        raise ParseError(f"unknown function {name!r} in {src!r}")
    lo, hi = FUNCTIONS[name]
    if not lo <= len(args) <= hi:
        raise ParseError(f"{name} takes {lo}..{hi} args, got {len(args)} in {src!r}")
    if name in ("min", "max"):
        # Paper A.2: min(x, d) = ts_min(x, d). Alphas 71/73/76/77/82/87/88/92/96 call
        # min/max on two expressions, which can only mean elementwise min/max.
        if isinstance(args[1], Num):
            name = "ts_" + name
        else:
            name = "elementwise_" + name
    return Call(name, tuple(args))


def parse(src: str) -> Node:
    """Parse an alpha expression into an AST. Raises ParseError on malformed input."""
    return _Parser(src).parse()


def canonical(src: str) -> str:
    """Canonical string of an expression (parse -> to_str)."""
    return parse(src).to_str()
