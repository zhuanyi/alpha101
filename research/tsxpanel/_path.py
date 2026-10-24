"""Puts research/taxloss (ishares, placeholders, prices, taxcal) and research/tsxpanel on sys.path."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "taxloss", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = HERE / "data"
RAW = DATA / "raw" / "ishares"
