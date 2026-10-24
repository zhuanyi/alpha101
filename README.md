# alpha101

A formulaic-alpha research engine, retail trading-cost models, and a set of documented research studies for a
Canadian retail investor with registered accounts. Most studies end negative after costs, and the repository keeps
those results on purpose.

- **Engine.** An expression-driven Python implementation of Kakushadze (2016), *101 Formulaic Alphas*
  ([arXiv:1601.00991](https://arxiv.org/abs/1601.00991)) for point-in-time-honest research on US and TSX equities. An alpha
  is a string; the 101 paper formulas ship as data (`src/alpha101/formulas/alphas.yaml`) and your own alphas use the same syntax.
- **Cost models.** Commission, spread, borrow and FX-conversion costs for a discount-broker investor
  (`src/alpha101/costs.py`). Every cost parameter used anywhere in the repository is tabulated in
  [`docs/cost_assumptions.md`](docs/cost_assumptions.md).
- **Studies.** Pre-registered where possible, holdout periods sealed in code, negative results kept (table below).

```python
from alpha101 import Evaluator, EvalOptions, forward_returns, get_alpha
from alpha101.stats import rank_ic, ic_summary
from alpha101.synthetic import random_panel

panel = random_panel(T=500, N=50)
ev = Evaluator(panel, EvalOptions())                  # conventions: docs/operators.md section 3
a12 = ev.evaluate(get_alpha(12).expression)           # paper alpha #12
mine = ev.evaluate("-1 * correlation(rank(open), rank(volume), 10) * sign(delta(close, 5))")
ic = rank_ic(a12, forward_returns(panel, horizon=5))  # the ONE lag function
print(ic_summary(ic["ic"], horizon=5))
```

## Running the tests

Tests use synthetic data only: no network, no database.

```bash
pip install -e '.[dev,report]'
python3 -m pytest -q
```

Tests that need a stored vendor price snapshot (Yahoo Finance, Bank of Canada) skip cleanly when it is absent. Raw
vendor data is not distributed: the study scripts fetch their own snapshot (`research/*/snapshot_*` and similar
directories are git-ignored). Scripts that touch a MySQL-compatible database read `ALPHA101_DB_*` variables; see
`config.example`.

## Studies

| study | where | verdict |
|---|---|---|
| Slow long-only book of the strongest 101 alphas (20 names, weekly/monthly/quarterly, flat commission) | `docs/slow_long_only.md`, `scripts/slow_long_only.py` | **Negative.** 0 of 132 backtests have positive active return with Newey-West t > 2 after costs; 15 of 132 are positive at all |
| Sector-momentum paper-traded rule (5 of 34 ETFs, 126-day momentum, min-variance weights, C$250k) | `research/sector_momentum/RESULTS.md` | **No edge.** CAGR 1.0% vs SPY 8.2% (CAD), excess -6.8%/yr (NW t -2.68); the momentum picks are indistinguishable from random picks gross of costs, and daily-churn rebalancing then costs about 4.6 points a year |
| Phase 1 conditional overnight premium (calendar, FOMC/CPI/NFP, VIX conditions on index ETFs) | `research/overnight/` | Stage A discovery code with fixed ADVANCE / REJECT / UNDERPOWERED rules and a sealed 2019+ holdout. Result tables are not distributed here; the deletion-rebound draft below records the track record as negative after retail costs |
| TSX tax-loss rebound (pre-registered; XIC/XCS holdings) | `research/taxloss/` | Pass-1 pipeline (calendar, price panel, split detection, data gates, placebos, decision rule). Decision tables are not distributed; the 2007-2011 series stays sealed |
| TSX point-in-time panel and index deletion events | `research/tsxpanel/`, `docs/research_notes/2026-10-23_tsx_pit_panel_coverage.md` | **Data study, no returns computed.** Counts and date spans only; coverage and feasibility of the deletion study (announcement dates found for 7 of 12 probed reviews, all from 2020 on) |
| TSX index-deletion rebound and low-volatility tilt | `docs/research_notes/2026-10-23_tsx_deletion_rebound_spec_DRAFT.md`, `..._tsx_lowvol_tilt_spec_DRAFT.md` | **Draft specifications**, not hash-locked, no outcomes computed. Priors: about 10% ADVANCE |
| Norbert's gambit (DLR.TO / DLR-U.TO) vs the broker's FX desk | `docs/research_notes/2026-10-21_norbert_gambit_results.md`, `scripts/cost_edge_norbert.py` | **Positive cost edge**, based on a ~3-month spread sample: the gambit is cheaper than the desk at every tested size from C$10k to C$500k (base case); under combined stress on the CAD-basis tier the desk wins one of ten cells (USD to CAD at C$500k, by 1.4 bp) |
| Dual-listed venue choice (TSX line vs US line, 40 names) | `docs/research_notes/2026-10-23_duallisting_venue_results.md`, `scripts/cost_edge_duallisting.py` | **Trade the line in the currency you hold.** Holding CAD the TSX line is cheaper at 4 of 4 sizes; holding USD the US line is cheaper; the FX leg (11 to 34 bp per conversion) dominates the half-spread difference |
| Options proxy (covered calls, put-writes) | `docs/research_notes/2026-10-24_options_proxy.md`, `scripts/options_proxy_study.py` | **Negative.** BXM-minus-SPX CAGR -715 bp/yr post-2012; retail covered-call cost 99-143 bp/yr; the index premium does not cover costs at any tested size. Buying option-data is not justified on this evidence |
| VWAP proxy and transcription cross-check | `docs/vwap_proxy_check.md`, `docs/transcription_crosscheck.md` | Method checks for the engine (typical-price VWAP proxy; formula transcription vs three public implementations) |

The options-proxy study was still under review upstream when this copy was made.

## Layout

| path | content |
|---|---|
| `src/alpha101/` | parser, evaluator, operators, panel and lag handling, statistics, cost model, evaluation, persistence, CLI |
| `docs/operators.md`, `docs/design.md` | exact operator semantics and conventions; architecture, data layer, evaluation, risks |
| `docs/research_plan.md` | pre-registered hypotheses H1-H4 and falsification rules |
| `docs/cost_assumptions.md` | every commission, spread, FX and financing parameter, with where it lives in the code |
| `docs/research_notes/` | generated study notes and draft specifications |
| `research/` | study code: overnight, sector momentum, tax-loss rebound, TSX panel |
| `scripts/` | study runners, benchmarks, generators |
| `tests/` | synthetic-data tests for the engine and every study |

Rules for contributors: one lag function; never edit operator semantics without a failing test; pure `%s` SQL
placeholders; no DROP/TRUNCATE; the package only touches its own schema.

## Licence and data notes

MIT (see `LICENSE`). The formulas are from Kakushadze (2016), published with WorldQuant LLC's permission in the paper;
check their terms before redistributing `src/alpha101/formulas/alphas.yaml`. Holdings files from index-fund providers and
price data from vendors are licensed to their users and are not part of this repository. Nothing here is investment advice.
