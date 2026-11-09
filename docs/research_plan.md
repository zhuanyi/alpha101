# Research plan: what the 101 alphas can and cannot do for a retail book

Status: pre-registration draft, 2026-10-18. Freeze this file, with any edits, **before** the
first holdout run. The CLI stores its SHA-256 in `alpha_runs.prereg_sha256` and refuses to
touch 2019–2026 without it (`design.md` §6).

## 1. Honest priors

- These are WorldQuant production alphas from about 2010–2013. They were published in 2015,
  replicated in every open-source factor library since, and sold as course material. Their
  documented holding period is 0.6–6.4 days, and the paper's Sharpe ratios exclude all costs.
  For **US large caps in 2019–2026**, the prior is that most stand-alone rank-ICs are
  indistinguishable from zero after correction for 101 tests. The ones that survive are probably
  short-horizon reversal or volume-conditioned reversal effects with too much turnover for a
  $9.99/trade account.
- **Where value is plausible:**
  1. The **library as infrastructure**: a tested expression engine, PIT data handling, and an
     evaluation harness that your own ideas plug into as strings.
  2. The **TSX**: fewer names, less quant capacity, wider spreads, and index-heavy ownership.
     Price-volume anomalies have historically decayed more slowly in smaller markets. The
     smoke test hints at something here, but on 23 symbols including ETFs it is not evidence of
     anything.
  3. **Combinations at lower turnover**: a weekly composite of several weakly correlated alphas
     (the paper reports an average pairwise correlation of 15.9%), traded as a 20-name book. This
     is where the paper's "mega-alpha" argument meets retail costs.
- **Where value is implausible:** daily-rebalanced single alphas, the delay-0 alphas (42, 48, 53,
  54) at retail, and anything US-listed that has to round-trip through the desk's CAD/USD FX spread.

## 2. Pre-registered hypotheses

Common settings: universe per `design.md` §5 (top 500 US / top 200 CA by trailing dollar volume
from the candidate pool; a symbol is eligible only with >= 20 daily price rows in the price table, so single-row Yahoo stubs such as EA, WBD, AVB are excluded at universe construction, not at ingest); raw basis; default `EvalOptions`; vwap proxy per `vwap_proxy_check.md`;
`forward_returns(lag=1, execution="close")`. IS = 2005–2018 (with eras 2005–2012 and 2013–2018);
OOS holdout = 2019-01-01 .. latest. Significance bar: |NW t| above the Šidák threshold at
N_eff, where N_eff is computed on the IS IC-correlation matrix (`stats.effective_number_of_tests`,
`stats.adjusted_threshold`). N_eff measures how many effectively independent trials were run during
selection, so it is estimated on the selection (in-sample) period; estimating it from holdout
returns would consume holdout information before the holdout test. Bonferroni(101) and HLZ t > 3
are also reported. Effects are reported with 95% CIs, not just t-stats.

### H1: some alphas survive out of sample (US and CA separately)
- **Claim:** at least **k = 3** of the 101 alphas have OOS rank-IC NW-t above the Šidák(N_eff) bar
  at the **5-day** horizon **and** the same IC sign as IS.
- **Test:** `alpha101 run --alphas all --start 2005-01-01 --horizon 5 --unlock-holdout --prereg docs/research_plan.md`
  computes IS and OOS IC; count alphas meeting both conditions, per market.
- **Falsified if** fewer than 3 qualify in a market. Report that as a result, not as a failure.
  A null here is the prior.
- **Also reported:** the expected false-positive count at the chosen bar (N_eff × p), so that
  "3 survivors" can be compared with what chance alone produces.

### H2: price-volume alphas work better on the TSX than in the US
- **Claim:** across the 82 non-industry, non-cap alphas, OOS 5-day IC that keeps its in-sample
  sign is larger in CA than in US. The test uses sign-consistent IC rather than |IC|, because a
  large |IC| with a flipped sign is not "working".
- **Test:** for each alpha, take s = sign(IS IC) in that market and compute OOS IC × s (positive =
  persisted). Take the cross-alpha median per market. The test statistic is the per-alpha
  difference (CA − US) of IC × s, with a sign test and a Wilcoxon signed-rank test across the 82
  alphas (alphas are the units; their dependence is handled by also reporting N_eff).
- **Falsified if** the Wilcoxon p > 0.05 or the median difference is ≤ 0.
- **Confounders, stated in advance:** the CA universe is smaller (200 vs 500 names, so noisier
  IC), has a different sector mix (financials, energy and materials are heavy), and has lower
  liquidity (which raises gross IC but also costs). Mitigation: repeat with a **US-200** universe
  (top 200 US by dollar volume) as the like-for-like comparison. That repeat is the headline.

### H3: a weekly 20-name composite survives retail costs
- **Claim:** an equal-weight z-score composite of the alphas that pass the **IS** screen (IS
  NW-t > 3 at h=5, sign-aligned, pairwise IC correlation < 0.5 by greedy selection), rebalanced
  **weekly**, run as a 10-long / 10-short book at **$100k per side** ($10k per name), has OOS net
  annualised return > 0 with a 95% CI excluding zero. Costs: $9.99 per trade, half-spread
  3/8 bps US/CA, borrow 0.5%/yr, and the FX spread on US names when the account holds no USD.
- **Test:** `evaluate.composite_backtest(...)` on the OOS era only. The composite's weights
  and members are fixed from IS data. Report gross, net, turnover, the break-even cost in bps, and
  trades per year.
- **Falsified if** the net CI includes zero, or if break-even cost < the assumed cost.
- **Variant (long-only, for TFSA/RRSP):** top-20 long-only vs the market's equal-weight universe
  return, measured by active return and its CI. Same falsification rule.

### H4: robustness of conventions (pre-registered variants, not a hunt)
- **Claim:** H1's survivors keep the same sign and stay above the raw 1.96 bar under each
  single-convention variant: `argmax_convention="days_ago"`, `adv_units="dollar"`,
  `basis="adjusted"`, `execution="open"`.
- **Test:** a paired NW t on the daily IC difference (variant − default) per survivor.
- **Falsified for an alpha if** its sign flips under any variant other than argmax for the
  argmax-using alphas, where a flip is *expected* and only documents the convention. Such alphas
  are reported as "convention-fragile" and excluded from H3's composite.
- **Bias measurement:** raw vs adjusted basis IC differences, aggregated over the 81 level-sensitive
  alphas, measure the split look-ahead bias that a naive (split-adjusted) replication would carry.

## 3. How results feed a portfolio

| outcome | action | account type |
|---|---|---|
| H3 long-short passes after costs, with <= 50 round trips/yr per name | small sleeve beside a core allocation, sized by the CI's lower bound, not its point estimate | margin account only (shorting) |
| H3 long-only variant passes | a tilt within the equity part of the core | a registered account if turnover stays low enough to avoid "carrying on a business" questions; withholding-tax treatment of US names differs by account type |
| H2 passes, H3 fails | TSX alphas become **features** for a lower-turnover model (monthly), not trades | n/a |
| everything fails | keep the library. Its value is the harness for new ideas | n/a |

**Prerequisite for any TSX allocation (known limitation / look-ahead):** the CA candidate pool is
today's S&P/TSX Composite constituents, which is survivorship-biased (`design.md` §5). Every TSX
result in §2 (H1-CA, H2, the CA legs of H3) is indicative only and must not drive an allocation
until a point-in-time TSX constituent history is sourced and the CA runs are repeated on it. The
US pool (current S&P 500 plus the Wikipedia "Selected changes" list) carries a residual bias of
the same kind, quantified in `design.md` §5.

Whatever survives is combined only through the paired-test discipline: a sleeve is added to the
core only if the **paired** test of (core + sleeve) − core on after-tax monthly returns has t > 2.

## 4. What is explicitly not done

- No parameter search over windows or coefficients. The paper's numbers are used as printed.
  Changing them would turn 101 tests into thousands and invalidate N_eff.
- No choosing between conventions by OOS performance.
- No look at 2019–2026 before this file is frozen.
- No delay-0 trading assumptions for retail.

## Amendments

The text above is the original pre-registration draft and is not edited. Each amendment below is dated and states what it changes. They are part of the pre-registration that is frozen and hashed at the gate.

### A1 (2026-10-19): H2 like-for-like universe is size-matched to CA every month
Replaces "repeat with a US-200 universe (top 200 US by dollar volume)" in H2. CA has fewer than 200 members in most
in-sample years (122-176 in 2007-2018) because the pool is today's constituents. The US comparison universe for H2 is
therefore capped, for each calendar month m, to K_m = the number of members of the final CA universe in that month
(after every CA eligibility rule, including A4, and the top-200 cap), using the same US liquidity rank and the same US
eligibility rules (stored as source `liquidity_cacap_us@<version>`). This size-matched run is the headline H2 test;
full US is reported alongside, together with the K_m series. Residual caveat: size matching does not match
composition (US top-N is liquidity-truncated, CA is all survivors); a gap between capped and full US is the size effect
and is reported. The IS sign s of H2 for US is taken from an in-sample run on the capped universe.

### A2 (2026-10-19): delisting-return convention
A name whose price data end exits at its last traded close: a forward return whose exit date falls after the name's
last row runs to that close instead of being NaN (data ending within 5 trading days of the panel end are treated as a
data lag, not a delisting). Performance delistings cannot be told apart from mergers with our data,
so the Shumway (1997) -30% delisting return is an off-by-default sensitivity option, not part of any test.
Measured footprint: no name with price data ends inside 2005-2018, so this convention changes no in-sample return.
Delisted names are missing from the pool altogether (no Yahoo history), which is where the survivorship bias enters
(see A5).

### A3 (2026-10-19): versioned universe sources
Every universe rebuild is stored under a versioned source name `<source>@YYYYMMDD` (e.g.
`liquidity_top500_us@20261019`); stored rows are never deleted to make room for a rebuild. Each run records the exact
versioned source it read; readers pick the latest version explicitly. The universe definition in section 2 is unchanged
by this; only the storage name is.

### A4 (2026-10-19): spread screen
Liquidity is a hard requirement: no thin names with wide spreads. In addition to the minimum price and the
dollar-volume top-N of section 2, a candidate is eligible in month m only if its Abdi-Ranaldo (2017) high-low-close
spread estimate over the 63 trading days ending at the prior month-end is at most 2% (US) or 3% (CA); a missing
estimate fails. The screen is applied before the top-N ranking, so a screened name is replaced by the next-ranked
eligible name. Thresholds and window are a recommendation. The estimator measures the high-low range
and is not a bid-ask half-spread; it does not validate the 3/8 bps cost assumption, it only removes the wide-range tail.

### A5 (2026-10-19): survivorship caveat in every report
Both candidate pools are built from today's index constituents. Every run report states this at the top, next to the
pool coverage figures (US pool tickers without Yahoo data, reused tickers, CA without change history), and says that IC
is conditioned on survivors, so the bias is expected to be upward on alpha, not symmetric.

### A6 (2026-10-19, wording erratum, no change in meaning)
"20 daily price rows" in section 2 means 20 rows in the `yahoo_ohlcv` price table.
