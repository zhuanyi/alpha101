# Operator semantics

This is the contract `src/alpha101/operators.py` implements and `tests/test_operators.py` checks.
Source: Kakushadze (2016), *101 Formulaic Alphas*, arXiv:1601.00991, **Appendix A.2 "Functions and
Operators"** and **A.3 "Input Data"**. The section numbers are printed wrongly in the PDF: both
subsections are headed "A.1", and Appendix B is only a legal disclaimer. Where the paper is silent,
the choice made here is marked **[choice]**, with the reason, and the alternative is noted when one
exists.

## 1. Conventions shared by all operators

| topic | rule |
|---|---|
| Data layout | 2-D float64 array `(T dates, N symbols)`, ascending dates. A `Panel` holds one market on its own exchange calendar (section 5). |
| Window | "past d days" = rows `t-d+1 .. t`, **today included**. `delay`/`delta` read row `t-d`. |
| Non-integer d | **`floor(d)`**, as the paper states for `ts_{O}` ("non-integer number of days d is converted to floor(d)"). We apply the same rule to every windowed operator, including `correlation`, `decay_linear`, `sum` and the rest. A 1e-9 guard absorbs float representation error (`2.9999999999999996 -> 3`). d < 1 after flooring is an error. **[choice vs practice]** Every public implementation we checked rounds to nearest instead (`docs/transcription_crosscheck.md`). The windows where the two rules differ are listed in each alpha's `notes`. `alpha101.expr.analysis.window` is the single rounding function. |
| Window argument type | Must be a numeric literal. An expression in a window slot is a parse-time/eval-time error. `signedpower` and `^` exponents may be arrays (alpha 84). |
| NaN in a window | **Strict**: any NaN in the window gives NaN. **[choice]** This keeps "missing" distinct from "zero". The public implementations zero-fill or forward-fill, which manufactures signal on halted or illiquid days. |
| Warm-up | A cell is defined only once its symbol has `max_lookback_days` rows of history after its first close. The evaluator masks the earlier cells. Without this, a ternary can emit early values from its short branch (alpha 1). |
| +/-inf | Every operator converts +/-inf to NaN (`x/0`, `log(0)`, overflow). **[choice]** Ranking an artefact as "largest" is worse than dropping it. |
| Booleans | Comparisons return 1.0/0.0 and **NaN if either side is NaN**. `||` and `&&` use truthiness (non-zero) and the same NaN rule. `c ? a : b` is NaN where `c` is NaN; NaN in the branch not taken does not propagate. |
| Case | Names are case-insensitive (`Ts_Rank` = `ts_rank`, `Log` = `log`). |
| Universe | Cross-sectional operators only see cells that are in the universe, past warm-up and outside a raw-basis split window (the same mask as the final output) and return NaN elsewhere. Time-series operators run on the whole history, membership or not, because a name's past matters on the day it enters. The final alpha is masked to the universe. |

## 2. Operator table

`x`, `y` are arrays; `d` is a floored window; `a` is a scalar or array.

| operator | definition used | NaN / edge cases | test |
|---|---|---|---|
| `abs(x)`, `sign(x)` | numpy; `sign(0)=0` | `sign(NaN)=NaN` | `test_sign_and_abs` |
| `log(x)` | natural log | `x<=0 -> NaN` | `test_log_nonpositive_is_nan` |
| `+ - * /` | elementwise | `/0 -> NaN` | `test_division_by_zero_is_nan` |
| `x ^ y` | power (`^` is **power**, not xor). Binds tighter than unary minus (`-x^2 = -(x^2)`); right-associative | negative base with a non-integer exponent -> NaN | `test_power_*`, parser tests |
| `< > <= >= == !=` | 1.0/0.0 | NaN in -> NaN out | `test_compare_propagates_nan` |
| `\|\|`, `&&` | logical on non-zero | NaN in -> NaN out | `test_logical_or_and` |
| `c ? a : b` | `np.where(c != 0, a, b)` | NaN `c` -> NaN | `test_where_nan_condition` |
| `rank(x)` | cross-sectional percentile **r/n in (0, 1]**, ties get the average rank, NaN and non-universe cells excluded from n | all-NaN row -> NaN | `test_rank_*` |
| `delay(x, d)` | `x[t-d]` | first d rows NaN | `test_delay*` |
| `delta(x, d)` | `x[t] - x[t-d]` | | `test_delta` |
| `correlation(x, y, d)` | sample Pearson over the window (pandas rolling cov/var, ddof cancels) | any NaN in x or y -> NaN. **Complete but flat window (zero variance on either side) -> 0.0** [choice: otherwise alpha 96 is all-NaN]. Clipped to [-1, 1] | `test_correlation_*` |
| `covariance(x, y, d)` | sample covariance, **ddof = 1** | strict NaN | `test_covariance_sample` |
| `scale(x, a=1)` | `x * a / sum(abs(x))` per row over universe cells | row sum 0 -> NaN | `test_scale_default_and_a` |
| `signedpower(x, a)` | `sign(x) * abs(x)^a`. The paper writes `x^a`, but the name and WorldQuant usage mean sign-preserving; for the paper's uses (`a=2` on a non-negative base, or a positive Ts_Rank base) the two agree except alpha 84's array exponent | | `test_signedpower_preserves_sign` |
| `decay_linear(x, d)` | `sum_k w_k x[t-k]`, with `w_k = (d-k) / (d(d+1)/2)` for `k = 0..d-1`: today weight d, oldest weight 1 | strict NaN (no renormalising over the valid subset) | `test_decay_linear_*` |
| `indneutralize(x, IndClass.level)` | `x - mean(x within group)` per row, universe cells only | see section 3.4 for fallbacks | `test_indneutralize_*` |
| `ts_min / ts_max(x, d)` | rolling min/max | strict NaN | `test_ts_min_max` |
| `min(x, d) / max(x, d)` | **numeric 2nd argument -> `ts_min/ts_max`** (paper A.2); **expression 2nd argument -> elementwise** (alphas 71, 73, 76, 77, 82, 87, 88, 92, 96). Resolved at parse time to `ts_min`/`elementwise_min` | elementwise: NaN propagates | parser + evaluator tests |
| `ts_argmax / ts_argmin(x, d)` | position of the max/min in the window. **Default `oldest_1based`: oldest day = 1, today = d.** Option `days_ago`: today = 0. Ties go to the **oldest** occurrence | strict NaN | `test_ts_argmax_*` |
| `ts_rank(x, d)` | percentile rank of today's value within its window: `(#less + (#equal+1)/2) / d`, in (0, 1] | strict NaN | `test_ts_rank_*` |
| `sum(x, d)`, `product(x, d)` | rolling sum / product | strict NaN | `test_ts_sum*`, `test_ts_product` |
| `stddev(x, d)` | rolling **sample** std, ddof = 1. **[choice]** pandas default. The paper is silent; ratios like `stddev(r,2)/stddev(r,5)` (alpha 34) change only by a constant, which ranks ignore | d = 1 -> NaN | `test_ts_stddev_is_sample` |

## 3. The ambiguous choices, and why

### 3.1 `rank` and `ts_rank` scale
The paper says only "cross-sectional rank". Alpha 35 (`1 - Ts_Rank(...)`) and alpha 27
(`0.5 < rank(...)`) only make sense if ranks are normalised to [0, 1]. We use pandas' percentile
r/n with average ties for both. The other candidate, (r-1)/(n-1), differs by O(1/n): rank ICs are
unchanged, and alpha 1's `- 0.5` is a constant shift either way. r/n also keeps n = 1 defined.

### 3.2 `ts_argmax` / `ts_argmin` base (sign-flip risk)
"Which day ts_max occurred on" fixes neither an origin nor a direction. Under `oldest_1based`, a
recent maximum gives a **high** value; under `days_ago`, a **low** one. Wrapped in `rank`, the
two conventions give opposite signs for alpha 1 and different results for alphas 57, 60, 96, 98
and 100. The public implementations count from the oldest day (1-based or 0-based; the offset
is rank-invariant). Third-party BRAIN descriptions disagree, and the official BRAIN docs could not
be read (login wall). Default: `oldest_1based`. `EvalOptions(argmax_convention="days_ago")` flips
it. The research plan treats the convention as a pre-registered robustness variant, not as
something to choose by IC.

### 3.3 `adv{d}` units
Paper A.3: "average daily **dollar** volume for the past d days". But alphas 7 (`adv20 < volume`),
21 (`volume / adv20 == 1`, `1 < volume/adv20`), 17, 43, 47 and 100 (`volume / adv20`) compare
adv with **share** volume. Read literally, the ratio is about 1/price, and alpha 21's branch
degenerates. Default: **`adv_units="shares"`**, because it is dimensionally consistent and matches
two of three public implementations. `EvalOptions(adv_units="dollar")` gives the literal paper
reading. In correlations and ranks of adv alone (alphas 28, 31, 36 and others), the difference is
a per-symbol price factor.

### 3.4 `indneutralize` and the IndClass levels
Demean within the group on each date, over universe cells with non-NaN x. We have yfinance
`sector` (11, GICS-like) and `industry` (~145) only, so `IndClass.subindustry` resolves to
`industry`. Fallback chain: subindustry -> industry -> sector -> market. A cell is demeaned
within the finest level whose group has **at least `min_group_size` (default 5)** non-NaN members
that day; `market` (everyone) always applies. A symbol with no label uses the next coarser label.
A panel with no labels at all raises `MissingField` rather than silently demeaning by market.
Labels are static per symbol today (look-ahead: see design.md); the operator also accepts
`(T, N)` labels, so point-in-time classifications plug in without code changes.

### 3.5 Formula quirks kept literally
- `(-1 * 1)` is the constant -1. Alpha 23 returns the literal 0 in its else-branch.
- Float equalities (`== 0.05` in alpha 24, `== 1` in alpha 21) are kept, though almost never true.
- Blends of one field with itself, e.g. `(vwap * w) + (vwap * (1 - w))` in alphas 59, 66, 82 and
  89, are as printed. They reduce to the field, and every public listing has them too.
- Alpha 29: `min(..., 5)` is `ts_min(..., 5)` (numeric argument). `product(., 1)` and `sum(., 1)`
  are identities.
- Alpha 84: `SignedPower(Ts_Rank(...), delta(close, 4))` raises a base in (0, 1] to a dollar-valued
  exponent. Kept literally; its values span many orders of magnitude and it is price-level
  sensitive.

## 4. The ONE lag convention

`alpha101.lag.forward_returns(panel, horizon, execution="close", lag=1)` is the only code that
looks forward in time. A test fails if any other module uses a negative shift.

- An alpha value dated **t** uses data through the **close of t**.
- Default (`execution="close", lag=1`): trade at the close of **t+1** and hold h days.
  The return is `adj_close[t+1+h] / adj_close[t+1] - 1`. The one-day implementation lag covers
  "compute after the close, trade the next day". It is conservative even for delay-1 alphas, which
  WorldQuant traded from the open of t+1.
- Option `execution="open"`: trade at the open of **t+lag**. The return is
  `P_open[t+lag+h] / P_open[t+lag] - 1`, where `P_open = open * adj_close / close` puts the
  open on the dividend-adjusted basis so that dividends inside the holding period count.
- Delay-0 alphas (42, 48, 53, 54; paper footnote 11) were meant to trade at the same close.
  `lag=0` needs `allow_delay0=True` and `execution="close"`. Retail cannot trade a close it has
  not seen, so research runs keep lag = 1 for all 101 alphas and report the delay-0 four as
  delay-1.
- The three-lag rule: prices and volume are real-time observables
  (no publication lag). The implementation lag is the `lag` argument. Static yfinance sector and
  market-cap snapshots have an *unknown* publication lag; that is a look-ahead, flagged in
  design.md, not something to fix with a shift.

## 5. Corporate actions and price basis

Yahoo (`auto_adjust=False`) serves open/high/low/close and volume **split-adjusted to the share
basis on the fetch date**, but not dividend-adjusted. `adj_close` is split- and
dividend-adjusted. `split` holds the ratio on its effective date; spin-offs show up as fractional
"splits", e.g. BN.TO 1.237, TRP.TO 1.097.

- `returns` = `adj_close[t] / adj_close[t-1] - 1`: a total return, as the paper's "daily
  close-to-close returns" requires (its eq. 2 note says prices are "adjusted for any splits and
  dividends").
- Only **20 of 101** alphas are invariant to a per-symbol price rescale (`price_scale_invariant`
  in alphas.yaml, measured on a synthetic panel by `scripts/build_alphas_yaml.py`). The other 81
  rank price levels cross-sectionally, e.g. `rank(open)`, `rank(1/close)` or `delta(close, 1)` in
  dollars. On split-adjusted history, a stock that splits 10:1 in 2024 looks cheap in every past
  cross-section. That is look-ahead, and it correlates with past winners.
- Therefore the default is the **raw (as-traded) basis**: `Panel.to_raw_basis()` rebuilds
  as-traded prices as `adjusted * prod(splits after t)`, with volume divided by the same factor.
  It is verified against the public AAPL 2020-08-28 ($499.23) and NVDA 2024-06-07 ($1,208.88)
  closes in `docs/smoke_test.md`. Any cell whose lookback window straddles **its own symbol's**
  split is set to NaN (`split_age < max_lookback_days`). Every remaining value is then exactly
  what a trader computing it on day t would have seen.
- Dividends: OHLC are not dividend-adjusted, so `delta(close, 1)` includes ex-dividend drops
  (around 0.25-1% a quarter). Back-adjusting OHLC for dividends would make past *levels* depend on
  future dividends (look-ahead again), so we do not.
- `basis="adjusted"` stays available. Running both bases and differencing the ICs measures the
  split look-ahead bias directly (smoke test table "Raw vs split-adjusted").

## 6. Edge-case table

| case | behaviour | where |
|---|---|---|
| window contains NaN (halt, missing volume, pre-IPO) | NaN for d rows | all `ts_*` |
| zero volume | stored as NaN at load (it is a Yahoo placeholder) | `data/yahoo.py` |
| `log(volume)` with volume 0 | not reachable (NaN at load); otherwise NaN | `log` |
| constant series in `correlation` | 0.0 when the window is complete | `correlation` |
| `high == low` (alpha 101 adds .001; 53, 54, 60 do not) | inf -> NaN | `div` |
| `close == low` in alpha 53 | inf -> NaN | `div` |
| ties in `ts_argmax` | oldest occurrence | `ts_argmax` |
| ties in `rank` / `ts_rank` | average rank | `rank`, `ts_rank` |
| single member in a group | falls back to the coarser level | `indneutralize` |
| no group labels at all | `MissingField` raised | `Panel.group_chain` |
| `scale` of an all-zero row | NaN | `scale` |
| split inside the lookback window (raw basis) | NaN | `Evaluator` mask (output and cross-sectional) |
| symbol younger than the lookback | NaN (warm-up mask) | `Evaluator` mask (output and cross-sectional) |
| US vs TSX holidays | separate panels per market calendar | `data/yahoo.panel_from_long(calendar=...)` |
| delay-0 alpha with lag=0 | refused unless `allow_delay0=True` | `forward_returns` |
| window literal floors to 0 | `ValueError` | `window` |
