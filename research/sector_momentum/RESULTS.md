# Sector-momentum paper-traded rule: backtest of the exact rules (CAD, retail costs)

**Verdict: NO EDGE.** Exact rules, C$250k, 1999-07-01 → 2026-10-16: CAGR 1.0% vs SPY 8.2%
(CAD), Sharpe 0.14 vs 0.48, excess vs SPY −6.8%/yr (Newey-West t = −2.68, lag 12). The reference rules
rank below all 1,000 random-5 placebo runs net of costs (percentile 0.0 on both CAGR and Sharpe), but that
gap is a trading-frequency effect: gross of costs the reference rules sit at the 44th (CAGR) / 47th (Sharpe) percentile of a cost-free placebo, i.e. the
momentum selection adds nothing over random picks and the daily-churn rebalancing then costs ~4.6 points a year. A short
2026-06 → 2026-10 stretch of paper-traded outperformance was driven by a single energy position (USO), which is a bet that happened to work, not a repeatable edge.

The verdict rule was fixed in `verdict()` in the script before the first full run:
EDGE needs NW t ≥ 2, placebo percentile ≥ 95 on CAGR and Sharpe, and a CAGR above SPY in ≥ 2 of
3 historical sub-periods. NO EDGE is NW t < 1 and percentile < 90 on CAGR or Sharpe. Anything else is INCONCLUSIVE.
The headline is the exact reference configuration only (126d lookback, top 5, constrained min-variance
weights); the robustness grid below is not used to choose a better cell.

## Headline (full period, C$250k, net of retail costs, CAD)

| strategy | CAGR | vol | Sharpe | maxDD | worst 12m | turnover/yr | trades/yr | cost drag/yr | excess vs SPY (ann.) | NW t |
|---|---|---|---|---|---|---|---|---|---|---|
| **Reference rules** | 1.01% | 15.61% | 0.14 | −48.9% | −36.4% | 35.6x | 404 | 4.44% | −6.78% | −2.68 |
| 1. SPY buy & hold | 8.18% | 20.81% | 0.48 | −54.6% | −30.6% | 0.01x | 0 | 0.00% | – | – |
| 2. Equal-weight all eligible, monthly | 5.69% | 15.47% | 0.44 | −36.9% | −29.8% | 0.22x | 291 | 0.77% | −2.63% | −2.11 |
| 3. Equal-weight same top 5 | 5.51% | 18.12% | 0.39 | −47.4% | −36.7% | 22.2x | 233 | 2.23% | −2.17% | −0.85 |
| 4. Random-5 placebo (mean of 1,000) | 4.64% | – | 0.37 | – | – | – | – | – | – | – |

Gross of costs (same C$250k): reference rules 5.66% CAGR / Sharpe 0.43 / excess −2.30% (t −0.94);
equal-weight all 6.65%; equal-weight top 5 8.09% (excess +0.23%, t 0.09). Costs, driven by the
trigger design, explain about 4.6 points of CAGR drag; before costs the rules are roughly
a SPY-like return with lower vol, not an edge.

Turnover is one-way (half of buys + sells) over average NAV. Cost drag = commissions + half-spreads over average NAV.

### Random-5 placebo (1,000 runs, C$250k, same weighting/costs, monthly picks)

CAGR mean 4.64% (sd 1.23%, min 1.25%, max 8.62%); Sharpe mean 0.37 (sd 0.08, range 0.16 to 0.63).
Reference rules: percentile **0.0** for CAGR and for Sharpe (lower than every run).
The placebo re-picks monthly, so its turnover is far below the reference rules' daily top-5 churn, and the net percentile of 0.0 mixes selection with trading frequency.
**Cost-free comparison** (`results/placebo_distribution_gross.csv`, `results/placebo_gross.json`; 1,000 runs, no commissions or spreads for either side):
reference rules 5.67% CAGR / Sharpe 0.43 vs placebo mean 5.85% / 0.44 → percentile **44.3** (CAGR) / **46.9** (Sharpe). Selection is indistinguishable from random;
the net shortfall is the cost of the trigger design. The verdict rule is unchanged and was applied as written on the net numbers.
To separate selection from trading frequency, the same momentum picks refreshed **monthly only**
(variant, C$250k): CAGR 6.0%, Sharpe 0.44, percentile 86.7 (CAGR) / 81.1 (Sharpe), excess vs SPY
−2.0%/yr (t −0.76): inside the placebo's range, not distinguishable from luck. It is not a significant edge either.

### Book size

| book | rule CAGR | Sharpe | cost drag/yr | excess vs SPY | NW t |
|---|---|---|---|---|---|
| C$100k | −21.7% | −1.43 | 12.8% | −30.7% | −3.19 |
| C$250k | 1.0% | 0.14 | 4.4% | −6.8% | −2.68 |
| C$500k | 2.2% | 0.22 | 3.3% | −5.7% | −2.25 |

SPY: 8.18% CAGR, Sharpe 0.48 in every case. At C$100k the fixed C$9.99 per order on ~400 orders/yr is ruinous.
The C$100k row is a death spiral, not a steady-state rate: NAV falls from C$100k to ~C$48k by 2003, ~C$21k by 2019 and ~C$120 by 2026-01
(`monthly_returns_C100k.csv`), and the simulated account keeps paying C$9.99 per order on a shrinking balance (the MIN_TRADE floor scales down with
equity, so tiny trades keep passing). Its 12.8% "cost drag" is measured on the average NAV, which is dominated by those last years, so read the C$100k row as
"fixed fees are fatal at this size", not as a CAGR forecast. It does not enter the verdict (headline = C$250k). A real account would stop trading far earlier.

### Sub-periods (C$250k)

| period | rule CAGR | SPY CAGR | rule Sharpe | SPY Sharpe | excess/yr | NW t |
|---|---|---|---|---|---|---|
| pre-2010 (1999-07 → 2009) | −3.1% | −3.5% | −0.09 | −0.01 | +0.5% | 0.11 |
| 2010-2019 | 3.9% | 15.8% | 0.36 | 1.12 | −10.7% | −4.33 |
| 2020-present | 2.8% | 16.4% | 0.26 | 0.88 | −12.4% | −3.19 |

Only the pre-2010 period (which includes the 2000-02 and 2008 bear markets, with a 10-26 ETF universe) is on par with SPY.

## Robustness grid (C$250k, full period; **not** used to select a rule)

| lookback | top-N | weights | CAGR | Sharpe | maxDD | trades/yr | excess vs SPY | NW t |
|---|---|---|---|---|---|---|---|---|
| 63 | 3 | rp | 0.1% | 0.11 | −66.7% | 291 | −7.0% | −1.96 |
| 63 | 3 | ew | −0.3% | 0.09 | −69.3% | 227 | −7.1% | −1.89 |
| 63 | 5 | rp | −8.6% | −0.51 | −93.1% | 530 | −16.7% | −5.66 |
| 63 | 5 | ew | −1.6% | 0.00 | −67.7% | 316 | −9.1% | −3.05 |
| 63 | 8 | rp | −5.8% | −0.19 | −83.2% | 672 | −14.0% | −5.11 |
| 63 | 8 | ew | 1.0% | 0.14 | −44.2% | 403 | −7.0% | −3.09 |
| 126 | 3 | rp | 5.2% | 0.35 | −58.1% | 213 | −2.0% | −0.59 |
| 126 | 3 | ew | 5.4% | 0.35 | −56.9% | 167 | −1.6% | −0.47 |
| **126** | **5** | **rp (reference)** | **1.0%** | **0.14** | **−48.9%** | **404** | **−6.8%** | **−2.67** |
| 126 | 5 | ew | 5.5% | 0.39 | −47.4% | 233 | −2.2% | −0.85 |
| 126 | 8 | rp | −5.9% | −0.18 | −86.1% | 498 | −12.4% | −3.58 |
| 126 | 8 | ew | 5.3% | 0.40 | −40.4% | 283 | −2.7% | −1.25 |
| 252 | 3 | rp | 4.8% | 0.32 | −53.4% | 158 | −2.5% | −0.73 |
| 252 | 3 | ew | 3.8% | 0.28 | −52.4% | 120 | −3.2% | −0.93 |
| 252 | 5 | rp | 6.0% | 0.41 | −40.7% | 293 | −2.0% | −0.79 |
| 252 | 5 | ew | 5.8% | 0.38 | −41.5% | 167 | −1.9% | −0.69 |
| 252 | 8 | rp | 4.3% | 0.33 | −44.7% | 385 | −3.8% | −1.74 |
| 252 | 8 | ew | 5.8% | 0.39 | −41.5% | 215 | −2.2% | −1.12 |

No cell has a positive, significant excess return over SPY. Within this grid the ranking of the constrained-min-variance
weights versus equal weight is not stable (it hurts at 63d and top-8, helps slightly at 252d top-5).
The 5-name 126d min-variance configuration is among the weaker cells, because every top-5 change triggers a full re-weighting.

## MIN_TRADE_NOTIONAL sensitivity (reference rules, exact-QP solver)

The reference floor is US$2,000 on a ≈US$100k book. The headline scales it as 2% of current equity; absolute floors give (`results/min_trade_sensitivity.csv`):

| book | floor | CAGR | Sharpe | trades/yr | cost drag/yr | excess vs SPY | NW t |
|---|---|---|---|---|---|---|---|
| C$250k | 2% of equity (headline) | 1.03% | 0.14 | 404 | 4.43% | −6.76% | −2.67 |
| C$250k | absolute US$2,000 | 0.88% | 0.13 | 426 | 4.58% | −6.91% | −2.72 |
| C$250k | absolute US$5,000 | 1.37% | 0.17 | 383 | 4.23% | −6.43% | −2.54 |
| C$250k | none | −0.96% | 0.02 | 629 | 6.38% | −8.75% | −3.40 |
| C$136k (≈ US$100k) | 2% of equity | −2.29% | −0.07 | 404 | 7.61% | −10.09% | −3.78 |
| C$136k (≈ US$100k) | absolute US$2,000 | −1.55% | −0.02 | 381 | 7.07% | −9.33% | −3.55 |

The floor choice moves CAGR by -0.15 to +0.34 points at C$250k (excluding "none"), with the same sign and significance in every variant, so the verdict does not depend on it.
The scaled floor is the original ratio (2% of book), not a stricter rule: at a US$100k book the two conventions are the same. A book of C$136k does worse than C$250k because the fixed C$9.99 weighs more.

## Implementation facts that matter

* **What the rule actually does** (differs from a "monthly + drift" description): it runs every day, and rebalances
  immediately whenever the held set differs from today's top 5, in addition to the 5% drift test and a once-per-month rebalance.
  Over the full history this fired 2,789 times (holdings), 32 (drift), 197 (monthly): ≈100 rebalances/yr. That churn is the source of the
  35x turnover and 4.4%/yr cost drag. The "monthly picks only" variant above isolates it.
* **"Risk parity" is constrained minimum variance** (`cp.quad_form` minimised, 5% floor, 40% cap, 1y daily covariance, equal-weight fallback).
* **cvxpy imprecision reproduced.** The original call `problem.solve()` (OSQP, tiny variance scale) occasionally returns an essentially
  equal-weight vector flagged "optimal" (e.g. 2015-05-26: 20% each; true optimum 40/6/6/40/8). The headline run uses the verbatim cvxpy call
  (cvxpy 1.9.3; 1.9.2 gives the same vector on the probed date). A separately validated exact active-set solver is used
  for the grid, placebo and variants. Across all 6,866 distinct (date, picks) weight vectors, max |Δw| = 0.20, p99 = 0.15,
  and the headline CAGR differs by 0.014 points (1.011% vs 1.025%), so this does not change any conclusion.
* **Momentum lag**: `prices.iloc[-126]` is 125 rows back; replicated. A ticker is eligible when both window ends exist (≥ 126 rows).
* **Universe**: 34 tickers in the script; VNQ had no data in the price source used for the published run (33 used).
  Effective start 1999-07-01 (first month with ≥ 10 eligible ETFs; 12 eligible at 1999 year-end, 19 in 2003, 26 in 2007, 30 in 2013, 33 from 2018;
  `results/universe_evolution.csv`). No back-filled or simulated history.
* **Execution**: signal on close t, fill at the t+1 open when the stored open is a real Yahoo open (10,962 open fills vs 70 next-close fallbacks
  in the headline run); the fallback applies to flat `open=high=low=close` stub rows (the last months of the series, plus a few hundred rows in 2013-16).
  Whole shares sized at the signal close; cash can dip slightly negative within a day (margin), no interest on cash.
* **Costs**: C$9.99 per order converted at the trade-day USDCAD, plus half-spread per side of 2 bp for SPY and the 11 SPDR sector ETFs, 5 bp for all other ETFs.
* **MIN_TRADE_NOTIONAL** = US$2,000 on a book of ≈US$100k, applied as a constant 2% of current equity,
  so it scales with the C$100k/250k/500k books. Sells of dropped positions are not filtered. The passive benchmarks (SPY, equal-weight all) have no filter.
* **FX**: FRED DEXCAUS (noon rate) to 2026-06-26; extended with Yahoo CAD=X closes to 2026-10-20 (251-day overlap: mean abs diff 0.0026, max 0.0114 CAD).
  CAD return = (1+r_USD)(1+r_USDCAD) − 1 via a USD NAV valued daily at USDCAD.
* **Prices**: adjusted closes (dividends reinvested); adjusted open = open × adj_close/close. Rows after 2026-06-26 are close-only with adj_close = close, so dividends after that date are not adjusted (< 0.5% effect).
* **Risk-free**: no T-bill series is used, so rf = 0 in all Sharpe ratios (the SPY-relative statistics are unaffected).
* **Window end**: 2026-10-16, the end date of the published run.
* Excess statistics use monthly CAD returns vs SPY with a Newey-West (Bartlett, lag 12) standard error.

## Tests

`pytest research/sector_momentum/tests -q` (13 tests, synthetic data, no network): signal date < trade date for every order; perturbing all prices after day T leaves NAV through T
unchanged (causality); shifting the signal prices by one day changes results; a planted breakout is bought only after it is visible at a close; cost accounting and per-ETF P&L reconcile to NAV;
SPY buy-and-hold equals the direct series; exact active-set solver equals SLSQP on 300 random covariance matrices; momentum lag matches `iloc[-126]`; Newey-West sanity.

## Files

`backtest_sector_momentum.py` (run: `python backtest_sector_momentum.py [--refresh-data] [--placebo-runs N]`; needs numpy, pandas, scipy, cvxpy, pyarrow, yfinance, pytest),
`results/summary.csv` (every strategy × book × sub-period), `results/monthly_returns_C{100,250,500}k.csv`, `results/robustness_grid.csv`,
`results/placebo_distribution.csv`, `results/placebo_distribution_gross.csv`, `results/placebo_gross.json`, `results/min_trade_sensitivity.csv`, `results/universe_evolution.csv`,
`results/solver_agreement.json`, `results/run_meta.json`, `results/tables.md`.

## Reproducibility note

The tables above were produced from a stored copy of Yahoo Finance data (plus FRED DEXCAUS for USDCAD up to 2026-06-26). `backtest_sector_momentum.py` now downloads from Yahoo directly (`--refresh-data`), so a fresh run will differ slightly, mostly in the last months where the stored copy was close-only. The raw price data are not distributed.
