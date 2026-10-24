"""alpha101: an expression-driven implementation of Kakushadze (2016), "101 Formulaic Alphas"."""
from .expr.evaluator import EvalOptions, Evaluator, evaluate
from .expr.parser import ParseError, canonical, parse
from .formulas import AlphaSpec, get_alpha, load_alphas
from .lag import forward_returns
from .panel import Panel

__all__ = ["EvalOptions", "Evaluator", "evaluate", "ParseError", "canonical", "parse",
           "AlphaSpec", "get_alpha", "load_alphas", "forward_returns", "Panel"]
__version__ = "0.1.0"
