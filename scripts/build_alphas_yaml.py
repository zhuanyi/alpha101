#!/usr/bin/env python3
"""Rebuild src/alpha101/formulas/alphas.yaml from the paper text.

Usage (the PDF is downloaded into its own empty temp dir, never into the repo):
    mkdir -p /tmp/alpha101_src && curl -sL https://arxiv.org/pdf/1601.00991 -o /tmp/alpha101_src/paper.pdf
    pdftotext -layout /tmp/alpha101_src/paper.pdf /tmp/alpha101_src/paper.txt
    PYTHONPATH=src python3 scripts/build_alphas_yaml.py /tmp/alpha101_src/paper.txt

`expression` is the paper text with whitespace normalised and nothing else. Every other field
except `notes` is computed from the parsed expression, and tests/test_formulas.py checks that the
committed YAML still agrees with that computation.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from alpha101.expr.analysis import WINDOW_OPS, group_levels, inputs, lookback, needs_industry
from alpha101.expr.ast import Call, Num, walk
from alpha101.expr.evaluator import Evaluator
from alpha101.expr.parser import parse
from alpha101.synthetic import random_panel, rescaled

OUT = Path(__file__).resolve().parents[1] / "src/alpha101/formulas/alphas.yaml"
DELAY0 = {42, 48, 53, 54}  # paper footnote 11

ADV_RATIO = ("adv_units matters: compares share `volume` with adv{d}; under the paper's literal "
             "'dollar volume' reading the ratio is ~1/price and the condition degenerates.")
LEVEL = ("price-level sensitive (output ranks change under a per-symbol price rescale): on "
         "split-adjusted history past cross-sections leak future splits; evaluate on the raw basis.")
NOTES = {
    1: "ts_argmax base matters: under oldest_1based a recent max ranks HIGH; under days_ago it "
       "ranks LOW (sign flip). The ternary mixes stddev(returns) (~0.02) with close (~100), so "
       "signedpower(.,2) is dominated by close on up days; literal reading kept.",
    7: ADV_RATIO + " `(-1 * 1)` is the constant -1.",
    12: "delta(close,1) is a dollar change; " + LEVEL,
    17: "volume/adv20 inside ts_rank: invariant to units over time, but see adv_units.",
    21: ADV_RATIO + " `(volume / adv20) == 1` is a float equality that is almost never true; "
        "kept literally. `||` returns NaN if either side is NaN.",
    23: "else-branch is the literal 0.",
    24: "`== 0.05` float equality kept literally.",
    25: "product of adv20 and vwap: " + LEVEL,
    29: "min(x, 5) with a numeric 2nd argument is ts_min(x, 5) (paper A.2). product(.,1) and "
        "sum(.,1) are identities; delta((close - 1), 5) == delta(close, 5).",
    32: "correlation window 230.",
    33: "`^1` is a no-op power.",
    41: "no rank: raw dollar quantity; " + LEVEL,
    42: "delay-0 alpha (paper footnote 11): intended to trade at the same close. Default lag=1 "
        "evaluates it as delay-1.",
    43: ADV_RATIO,
    47: "rank(1/close) and volume/adv20: " + LEVEL + " " + ADV_RATIO,
    48: "delay-0 alpha. IndClass.subindustry -> industry (yfinance has no sub-industry). "
        "Window 250 needs ~1 year of history.",
    53: "delay-0 alpha. Division by (close - low) is +/-inf when close == low -> NaN.",
    54: "delay-0 alpha. Division by (low - high) is inf on zero-range days -> NaN.",
    56: "needs cap (only alpha that does). cap is not point-in-time from Yahoo; see design.md.",
    59: "(vwap * w) + (vwap * (1 - w)) == vwap: the blend reuses one field (paper as printed).",
    61: "output is a 0/1 comparison (no sign flip).",
    62: "rank() of a 0/1 comparison; output is 0/-1.",
    66: "(low * w) + (low * (1 - w)) == low (paper as printed).",
    67: "rank ^ rank power; indneutralize(adv20, subindustry) -> industry.",
    71: "max(expr, expr) with two expressions -> elementwise max.",
    73: "max(expr, expr) -> elementwise max.",
    76: "max(expr, expr) -> elementwise max; adv81.",
    77: "min(expr, expr) -> elementwise min.",
    82: "min(expr, expr) -> elementwise min. (open * w) + (open * (1 - w)) == open.",
    84: "SignedPower with an ARRAY exponent delta(close, 4): base in (0,1], exponent in dollars, "
        "so values span many orders of magnitude; " + LEVEL,
    87: "max(expr, expr) -> elementwise max; adv81.",
    88: "min(expr, expr) -> elementwise min.",
    89: "(low * w) + (low * (1 - w)) == low (paper as printed).",
    92: "min(expr, expr) -> elementwise min; decay_linear of a 0/1 comparison.",
    93: "adv81.",
    96: "max(expr, expr) -> elementwise max; ts_argmax base matters.",
    98: "ts_argmin base matters.",
    100: "indneutralize applied twice at the same level (second pass is a no-op except for "
         "fallback cells); ts_argmin base matters; volume/adv20: " + ADV_RATIO,
    101: "the 0.001 in the denominator makes it slightly price-level sensitive for low-priced names.",
}


def extract(paper_text: str) -> dict[int, str]:
    start = paper_text.index("A.1. Formulaic Expressions for Alphas")
    end = paper_text.index("A.1. Functions and Operators")
    lines = [l for l in paper_text[start:end].splitlines()
             if l.strip() and not re.fullmatch(r"\s*\d+\s*", l)]  # drop blank + page-number lines
    parts = re.split(r"^Alpha#(\d+):", "\n".join(lines[1:]), flags=re.M)
    out = {}
    for i in range(1, len(parts), 2):
        out[int(parts[i])] = re.sub(r"\s+", " ", parts[i + 1]).strip()
    return out


def rounding_note(node) -> str | None:
    """List windows whose floor (paper) differs from round-to-nearest (public implementations)."""
    diff = []
    for n in walk(node):
        if isinstance(n, Call) and n.name in WINDOW_OPS and isinstance(n.args[-1], Num):
            v = n.args[-1].value
            if math.floor(v) != round(v):
                diff.append(f"{v:g}")
    if not diff:
        return None
    return ("floor vs round-to-nearest differs for windows " + ", ".join(sorted(set(diff), key=float))
            + " (we floor, per paper A.2; public implementations round).")


def scale_invariant(node) -> bool:
    """True if a per-symbol price rescale (split-basis change) leaves output ranks unchanged."""
    base = random_panel(T=420, N=30, seed=7)
    factors = np.random.default_rng(11).choice([0.1, 0.5, 2.0, 10.0], size=30)
    a = Evaluator(base).evaluate(node).to_numpy()[-120:]
    b = Evaluator(rescaled(base, factors)).evaluate(node).to_numpy()[-120:]
    import pandas as pd
    ra = pd.DataFrame(a).rank(axis=1).to_numpy()
    rb = pd.DataFrame(b).rank(axis=1).to_numpy()
    same_nan = np.array_equal(np.isnan(ra), np.isnan(rb))
    ok = ~np.isnan(ra) & ~np.isnan(rb)
    if not ok.any():
        raise RuntimeError(f"no finite values for {node.to_str()[:80]}")
    return bool(same_nan and np.allclose(ra[ok], rb[ok], atol=1e-6))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_text")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    exprs = extract(Path(args.paper_text).read_text())
    if sorted(exprs) != list(range(1, 102)):
        raise SystemExit(f"expected alphas 1..101, got {sorted(exprs)[:5]}..")
    entries = []
    for i in range(1, 102):
        node = parse(exprs[i])
        notes = [NOTES.get(i)]
        notes.append(rounding_note(node))
        inv = scale_invariant(node)
        if not inv and i not in NOTES:
            notes.append(LEVEL)
        entries.append({
            "id": i,
            "expression": exprs[i],
            "inputs": inputs(node),
            "needs_industry": needs_industry(node),
            "group_levels": group_levels(node),
            "max_lookback_days": lookback(node),
            "delay": 0 if i in DELAY0 else 1,
            "price_scale_invariant": inv,
            "notes": " ".join(n for n in notes if n),
        })
    header = ("# 101 Formulaic Alphas -- Kakushadze (2016), arXiv:1601.00991, Appendix A.1.\n"
              "# GENERATED by scripts/build_alphas_yaml.py from the paper text; edit NOTES there,\n"
              "# not here. `expression` is verbatim (whitespace normalised). Formulas are\n"
              "# proprietary to WorldQuant LLC and reproduced in the paper with its permission.\n")
    body = yaml.safe_dump({"source": "Kakushadze, 101 Formulaic Alphas, arXiv:1601.00991 (PDF dated 2015-12-09, fetched 2026-10-18)",
                           "alphas": entries}, sort_keys=False, width=100, allow_unicode=False)
    Path(args.out).write_text(header + body)
    n_inv = sum(e["price_scale_invariant"] for e in entries)
    print(f"wrote {args.out}: {len(entries)} alphas, {sum(e['needs_industry'] for e in entries)} "
          f"need industry, {n_inv} price-scale invariant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
