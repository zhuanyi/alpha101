"""The 101 alpha definitions (alphas.yaml) and a loader."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml

from ..expr.ast import Node
from ..expr.parser import parse


@dataclass(frozen=True)
class AlphaSpec:
    id: int
    expression: str
    inputs: tuple[str, ...]
    needs_industry: bool
    group_levels: tuple[str, ...]
    max_lookback_days: int
    delay: int
    price_scale_invariant: bool
    notes: str

    @property
    def name(self) -> str:
        return f"alpha{self.id:03d}"

    def ast(self) -> Node:
        return parse(self.expression)


@lru_cache(maxsize=None)
def load_alphas() -> dict[int, AlphaSpec]:
    """All alphas keyed by id, from the packaged alphas.yaml."""
    text = resources.files(__package__).joinpath("alphas.yaml").read_text(encoding="utf-8")
    out = {}
    for e in yaml.safe_load(text)["alphas"]:
        out[int(e["id"])] = AlphaSpec(
            id=int(e["id"]), expression=e["expression"], inputs=tuple(e["inputs"]),
            needs_industry=bool(e["needs_industry"]), group_levels=tuple(e.get("group_levels", [])),
            max_lookback_days=int(e["max_lookback_days"]), delay=int(e["delay"]),
            price_scale_invariant=bool(e["price_scale_invariant"]), notes=e.get("notes", "") or "")
    return out


def get_alpha(alpha_id: int) -> AlphaSpec:
    return load_alphas()[alpha_id]
