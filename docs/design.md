# alpha101 — library design

Operator semantics are in `operators.md`, the research questions in `research_plan.md`, and
pipeline proof in `smoke_test.md`.

## 1. Goals and non-goals

**Goals.**
1. A reusable Python library in which an alpha is a *string*. All 101 Kakushadze formulas ship as
   data (`formulas/alphas.yaml`), and your own alphas use the same syntax.
2. Semantics that are written down, tested by hand-computed examples, and point-in-time honest:
   one lag function, an as-traded price basis, a per-market calendar, and the universe mask in
   every cross-sectional operator.
3. An evaluation layer that answers "is there any signal left, where, at what horizon, after desk
   costs and multiple testing", and writes results to `alpha101` so they can be regenerated.

**Non-goals (for now).** Portfolio optimisation, live trading, intraday data, fundamentals,
a general backtester. The evaluation is cross-sectional IC plus a decile long-short, which is what
decides whether anything deserves a portfolio-construction stage.

## 2. Architecture

```
alpha101.yahoo_ohlcv ──► data/yahoo.py ──► Parquet cache (data/cache/) ──► Panel (one per market)
alpha101.security_master ─► groups (sector/industry codes) ─┘                 │
alpha101.universe_membership ─► universe mask (T×N bool) ───┘                 │
                                                                                     ▼
alphas.yaml / user strings ──► parser (Pratt) ──► AST ──► Evaluator (memoised) ──► alpha (T×N)
                                                                                     │
                     lag.forward_returns(panel, h)  ◄── the ONLY forward-looking code │
                                   │                                                  ▼
                                   └──────────────► evaluation (IC, deciles, turnover, costs)
                                                         │                     │
                                     alpha101.alpha_ic / alpha_runs    reports/*.md + PNG
                                     backtest_results + research_log         data/alpha_values/*.parquet
```

Package layout:

```
src/alpha101/
  expr/ast.py          frozen dataclass nodes; canonical to_str()
  expr/parser.py       tokenizer + Pratt parser, aliases, min/max resolution
  expr/analysis.py     inputs(), needs_industry(), lookback(), window() — the ONE rounding rule
  expr/evaluator.py    EvalOptions, Evaluator (memo, warm-up mask, split mask, universe mask)
  operators.py         every operator on (T, N) float arrays
  panel.py             Panel, derived fields, raw basis, group chains
  lag.py               forward_returns — the ONE lag function
  stats.py             rank_ic, Newey-West t, ic_summary, effective N, adjusted threshold
  db.py                reader()/writer(), pinned to alpha101
  data/yahoo.py        read yahoo_ohlcv -> Panel (per market calendar)
  synthetic.py         deterministic synthetic panels (tests, benchmarks, invariance checks)
  formulas/alphas.yaml generated from the paper by scripts/build_alphas_yaml.py
  data/ingest.py, data/universe.py, data/cache.py, evaluate.py, costs.py, report.py, persist.py,
  cli.py (+ __main__.py)
```

## 3. Expression engine

### 3.1 Why expressions rather than 101 hand-written functions
- **Transcription is the failure mode.** The three public implementations we read
  (`transcription_crosscheck.md`) copy the formula text *correctly* into comments, and then get the
  code wrong in at least 17 places: covariance for correlation, a window of 220 for 20, a missing
  `* -1`, a time-series "rank", sma/7 divided by 7 again, `open` for `close`. When the formula
  *is* the code, that class of error is gone. What remains is the operator semantics, and there are
  ~25 operators to test by hand, not 101 functions.
- **Testability.** Each operator has hand-computed tests (`test_operators.py`). Each formula has a
  parse/round-trip test, a lookback-consistency test, a coverage test (no NaN-propagation
  collapse), and a "nothing before lookback" test, all on a synthetic panel.
- **Extensibility.** You add an alpha by writing a string. Guotai Junan's 191 alphas use
  the same grammar style plus a handful of operators (`sma(x,n,m)`, `wma`, `count`, `sumif`,
  `highday`/`lowday`, `regbeta`, `regresi`, `mean`, `tsmax`/`tsmin` aliases), each a new entry in
  `FUNCTIONS` plus one function in `operators.py`. No evaluator change is needed.
- **Free static analysis.** Inputs, industry need and maximum lookback come from the AST. The
  lookback drives how much history to load, the warm-up mask, and the split-window mask.
- **Memoisation.** Sub-expressions are hashable frozen dataclasses, so `delta(close, 1)`, which
  alpha 9 uses four times, is computed once per evaluator.

### 3.2 Grammar
Pratt parser over the paper's syntax. Precedence, low to high: ternary `?:` (right-assoc) <
`||` < `&&` < `== !=` < `< > <= >=` < `+ -` < `* /` < unary `-` < `^` (power, right-assoc; so
`-x^2 = -(x^2)`). The full grammar is in the `parser.py` docstring. Fields are `open high low
close volume returns vwap cap adv<d>`. `IndClass.<sector|industry|subindustry|market>` is a group
literal. Function names are case-insensitive and aliases are normalised (`Ts_ArgMax` ->
`ts_argmax`). `min`/`max` resolve at parse time: a numeric second argument gives `ts_min`/`ts_max`
(paper A.2); an expression gives `elementwise_min`/`elementwise_max` (alphas 71, 73, 76, 77, 82,
87, 88, 92, 96). The canonical form is fully parenthesised and round-trips:
`parse(to_str(ast)) == ast` for all 101 alphas.

### 3.3 Evaluation model
- Values are `(T, N)` float64 numpy arrays, or Python floats for literals. pandas appears only
  inside rolling kernels and at the API boundary (`DataFrame` in, `DataFrame` out).
- Every memoised array is set read-only (`setflags(write=False)`). Combined with pandas 3
  copy-on-write, where `to_numpy()` can return read-only views, every in-place write must work on
  an explicit `np.array(..., copy=True)`. The tests check this.
- `EvalOptions` holds the three ambiguous choices: `argmax_convention`, `adv_units` and
  `min_group_size` (`operators.md` §3). Pass a different options object to run a variant; never
  edit operator code to change a convention.
- One mask, warm-up (`max_lookback_days` after a symbol's first close) AND the raw-basis split
  window AND the universe, is applied both to the output and inside cross-sectional operators
  (rank, scale, indneutralize), so excluded cells never enter a rank or scale sum. A symbol with
  no finite close is excluded on every row.

## 4. Data layer

### 4.1 Stores
| store | role | owner |
|---|---|---|
| `alpha101.yahoo_ohlcv` | raw daily OHLCV + adj_close + dividend + split per symbol | ingestion |
| `alpha101.security_master` | name, market, sector, industry, market-cap snapshot, shares outstanding | ingestion |
| `alpha101.universe_membership` | membership events per symbol/source | universe builder |
| `alpha101.alpha_runs`, `alpha_ic` | run registry and daily IC series | persistence |
| `backtest_results`, `research_log` | one summary row per (alpha, market) at the primary horizon; one row per hypothesis | persistence |
| `data/cache/*.parquet` (local, git-ignored) | fast reload of the long OHLCV frame per market | cache |
| `data/alpha_values/<run_id>/<market>/alpha_NNN.parquet` (local) | raw alpha values (float32) | evaluation |

DDL for the new tables is in §8. The writer account may hold rights on other schemas, so
`db.writer()` pins `database=alpha101` and asserts `SELECT DATABASE()` before returning.

### 4.2 Corporate actions and the price basis
This is covered in full in `operators.md` §5. In short:
- Yahoo OHLC and volume are split-adjusted to the fetch date's share basis. That is verified in the
  stored data: AAPL is continuous across its 4:1 split of 2020-08-31, with volume scaled up 4x.
- 81 of 101 alphas are **price-level sensitive** (`price_scale_invariant: false`), so the default
  is `Panel.to_raw_basis()`. As-traded prices are rebuilt from the split column, and cells whose
  window straddles their own symbol's split are masked. `returns` and forward returns always use
  `adj_close`, which includes dividends.
- Per field: `open`/`high`/`low`/`close` are as-traded, which keeps cross-sectional levels
  point-in-time (PIT). `vwap` is a proxy of those. `volume` is as-traded shares. `adv{d}` uses that
  volume, or `close*volume` in dollar mode. `returns` uses `adj_close` so dividends are not
  mistaken for price drops. `cap` = `close * shares_outstanding`, both as-traded, so their product
  is split-invariant.
- **Re-fetch rule .** Yahoo re-adjusts the whole history after every new split. An
  incremental append after a split would mix share bases inside one symbol's history. The ingester
  must re-download a symbol's full history whenever the new data contain a split **or a dividend**
  dated after the stored max(date), whenever the overlap day's `close` differs from the stored value
  by >0.5%, or whenever the overlap day's `adj_close/close` ratio differs from the stored ratio by more
  than 1e-4 (relative). Dividends need their own trigger: `close` is not dividend-adjusted, but Yahoo
  rescales `adj_close` of the whole history at every ex-date, so an incremental append would mix two
  `adj_close` bases in one symbol. It upserts the whole history in that case, then deletes that symbol's
  stored dates that Yahoo no longer returns (a DELETE scoped to the symbol).

### 4.3 Calendars
One Panel per market, on that exchange's calendar: the `^GSPC` dates for US, `^GSPTSE` for CA.
US and TSX holidays differ (Canada Day, Civic Holiday, Canadian Thanksgiving, Boxing Day vs July 4,
US Thanksgiving, MLK, Juneteenth). On a union calendar, each foreign holiday would be a NaN row,
and the strict NaN policy would blank d days of every window op. Cross-market comparison happens on
the IC series, never inside one cross-section.

### 4.4 Proxies for fields we do not have
| field | proxy | look-ahead / bias | pluggable |
|---|---|---|---|
| `vwap` | **typical price (H+L+C)/3** (default); alternative (O+H+L+C)/4 | none (same-day OHLC) | `Panel(vwap_proxy=...)`, `VWAP_PROXIES`; a real `vwap` field overrides |
| `cap` | `close * shares_outstanding`, where shares come from yfinance `get_shares_full()` history if available. Otherwise use the static `sharesOutstanding` snapshot, flagged | the snapshot is **today's** share count: a look-ahead for buyback/issuance names. Alpha 56 is the only consumer, so it is reported separately with a flag | `cap` or `shares_outstanding` field |
| `adv{d}` | d-day mean of share volume (default) or of `close*volume` | none | `EvalOptions.adv_units` |
| IndClass | yfinance `sector` (sector level) and `industry` (industry and subindustry levels), static | **today's** classification: names that changed sector (GICS 2018 communication services, 2023 reshuffle) are mislabelled in the past. Only the 18 industry alphas are affected | `Panel.groups`, static `(N,)` or PIT `(T, N)` |

On the vwap choice: `vwap - close` is what most vwap alphas consume (11, 25, 42, 47, 57, 83). With
(H+L+C)/3, this becomes `(H+L-2C)/3`, a location-in-range measure. (O+H+L+C)/4 drags in the open,
which several alphas (5, 65, 98) already use separately, inflating their collinearity.
`scripts/vwap_proxy_check.py` compares both proxies against a VWAP built from Yahoo 60-minute
bars. Results are in `vwap_proxy_check.md` (generated). Typical price has the lower error and
the higher cross-sectional agreement of `(proxy - close)` with `(vwap - close)`, so it stays the
default. Caveat: the bar-built reference itself uses per-bar typical prices, which favours
`typical` somewhat. The cross-sectional agreement is the figure that matters for the alphas.

### 4.5 Data hygiene rules for the loader
- Drop rows with `close <= 0` or a missing close. Set zero volume to NaN.
- Exclude `^` index symbols from universes. They are used only as calendars and benchmarks.
- Keep ETFs out of the single-stock universe. They belong in the smoke test, not the research run.
- Flag stale rows: `open == high == low == close` with volume 0 (1,132 such rows since 2005 in the
  current table). The ingester keeps them, and `panel_from_long` treats volume 0 as NaN.

## 5. Universe

The aim is to be as point-in-time honest as free data allow, and to **quantify** what is left.

**Candidate pool** (`security_master`):
- US: current S&P 500 constituents (Wikipedia "List of S&P 500 companies", first table), plus
  every ticker that appears in that page's "Selected changes" table as added or removed since
  2005-01-01. Yahoo symbol mapping: `.` becomes `-` (`BRK.B` -> `BRK-B`). Symbols must match
  `^[A-Z0-9]+(-[A-Z0-9]+)*(\.TO)?$`; anything else is dropped. The changes table now lives on
  `Historical_components_of_the_S%26P_500` (fallback URL).
  **Identity check:** a ticker found only in the changes table is accepted only if Yahoo's first-trade date
  is not more than 7 days after the earliest change date and Yahoo's name matches a Wikipedia Security name.
  Otherwise Yahoo has given the ticker to a different issuer and it is recorded as `reused_ticker`
  (not ingested, not in the pool).
- CA: current S&P/TSX Composite constituents (Wikipedia "S&P/TSX Composite Index", ~220 names).
  Yahoo symbol = ticker with `.` -> `-`, plus `.TO` (`BBD.B` -> `BBD-B.TO`, `RCI.B` -> `RCI-B.TO`).
- Removed or delisted names are often missing on Yahoo. The builder records, per source, how many
  tickers it attempted and how many returned data. That count is the first survivorship number in
  the report. `reused_ticker` names are counted separately from names with no data, and "US symbols
  with data" counts distinct identity-checked issuers.
- **Known limitation / look-ahead.** Both pools start from *today's* constituents, so they are
  survivorship-biased. The US pool is only partly repaired by the Wikipedia "Selected changes"
  list, which is not guaranteed to be a complete change history. The CA pool has no change history at
  all: names that left the S&P/TSX Composite before today are missing. TSX results are therefore
  **indicative only** until a point-in-time TSX constituent history is sourced (none is
  identified yet). Sourcing it is a **prerequisite before any TSX result is used for
  allocation** (`research_plan.md` §3).

**Minimum history.** A symbol is universe-eligible only with at least 20 daily price rows in `yahoo_ohlcv`.
Single-row Yahoo stubs (for example EA, WBD, AVB; 5 US and 5 CA symbols today) are kept in the pool
and in `yahoo_ohlcv`, and are excluded at universe construction in, not at ingest.

**Daily universe** (`universe.py`), the honest part: on the first trading day of each month,
take candidates with at least 252 days of history and a close above $5 (US) or C$2 (CA). Rank them
by trailing 63-day median dollar volume (as-traded `close*volume`), computed only from data up to
the prior month-end. Keep the top **500 US** and top **200 CA**. Membership holds for that month.
Liquidity ranking uses only past data, so the universe adds no look-ahead *given the pool*. The
residual bias lives in the pool, which leans toward today's survivors. S&P index membership itself
(`sp500_wiki_*` sources) is stored for reference and as an alternative mask, reconstructed by
walking the changes table backwards from today's list.

*Amended 2026-10-19 (`research_plan.md` A1-A4):* a candidate must also pass a
spread screen at the prior month-end, the Abdi-Ranaldo (2017) high-low-close spread estimate over 63 days at most 2% US /
3% CA, applied before the top-N ranking. The H2 comparison universe `liquidity_cacap_us` keeps, in each month, as many
US names as the final CA universe has. Every rebuild is stored under a versioned source name `<source>@YYYYMMDD`, and
readers resolve the latest version explicitly. A name whose data end exits at its last traded close
(`lag.forward_returns`, `delisting="last_close"`).

**Survivorship quantification** (report section):
1. Coverage: share of removed-since-2005 S&P tickers with Yahoo data.
2. Return gap: monthly equal-weight total return of the US universe minus **RSP** (Invesco S&P 500
   Equal Weight ETF, since 2003, a PIT-honest EW S&P 500), annualised, per era. For CA,
   cap-weighted universe vs **XIC.TO**, using cap snapshots, labelled indicative.
3. Interpretation: survivorship mostly inflates the *level* of long-only returns. For dollar-neutral
   decile spreads it biases the IC only if the alpha loads on future-delisting names; we state
   that as an assumption.

## 6. Evaluation layer

All of it consumes `alpha (T×N)` and `forward_returns(panel, h)`. Nothing else shifts data.

| metric | definition |
|---|---|
| daily rank IC | Spearman across universe names with alpha and forward return both finite; NaN if fewer than 30 names (US) or 20 (CA) |
| IC summary | mean, std, IC-IR = mean/std (daily), NW t with lags = h, hit rate, days; full sample and per era |
| eras | 2005–2012, 2013–2018 (in-sample, IS), **2019–2026 holdout** (out-of-sample, OOS). The CLI refuses to compute anything on the holdout unless `--unlock-holdout` is passed **and** a pre-registration file (hypotheses + thresholds, `research_plan.md` §2) is given. Its SHA-256 is stored in `alpha_runs` |
| decile long-short | at each rebalance date, equal-weight long the top decile and short the bottom decile by alpha. h staggered sub-books (one starts each day, each held h days) averaged, so every day contributes. Daily returns come from adj_close via `forward_returns(panel, 1)` composition |
| turnover | one-way = 0.5 × Σ\|w_t − w_{t−1}·(1+r)/(1+r_p)\| per rebalance: Σ\|dw\| counts both sides. Report mean, and the share of rebalances with >50% one-way turnover |
| costs (`costs.py`) | per-trade commission **$9.99** (option `$7.00` high-volume tier) divided by the per-name notional (default $5,000 = $100k per side / 20 names), plus a half-spread of 3 bps US / 8 bps TSX (stated assumptions), plus an optional FX-desk spread (tiered 1.26%/0.78%/0.46% + 1/3 surcharge) on US legs **when no USD account exists**. Short borrow 0.5%/yr general collateral. All parameters are in one dataclass, printed into every report and stored in `params` JSON |
| account fit | long-short requires shorting, so **Margin only**: TFSA and RRSP cannot short. Registered accounts get the long-only top-decile-minus-benchmark variant. Any configuration with >50 round trips/yr per account is flagged "CRA business-income risk" for TFSA |
| multiple testing | correlation matrix of the 101 daily IC series (**in-sample** era 2005–2018, pairwise-complete). N_eff = (Σλ)²/Σλ². N_eff measures how many effectively independent trials were run during selection, so it is estimated on the selection (in-sample) period; estimating it from holdout returns would consume holdout information before the holdout test. Report the raw bar 1.96, Šidák(N_eff), Bonferroni(101) and the Harvey–Liu–Zhu 3.0, and count alphas clearing each |
| paired comparisons | variant vs baseline (e.g. argmax convention, raw vs adjusted, US vs CA) is always a **paired** NW t on the daily IC difference, never a comparison of two standalone t-stats |
| CIs | 95% CI on mean IC (NW) and on annualised decile-spread return. "Years needed for significance" = (1.96 / IR_annual)² is reported per alpha |

`report()` writes `reports/<run_id>/report.md` plus PNGs: cumulative IC per era, the
IC-by-horizon heatmap, the decile bar chart, and the N_eff scree plot. Every number in the
markdown is formatted from the run's own objects; no literals.

## 7. Persistence decision

Storing alpha values in MariaDB as `(date, symbol, alpha_id, value)` would take 700 names ×
~5,500 days × 101 alphas ≈ **390M rows** per variant. That is too large for a research table on
the database server and is never queried relationally. Decision:
- Raw alpha values go to **local Parquet**, one wide float32 file per alpha per market per run
  (≈ 15 MB uncompressed each). They are regenerable from code + `yahoo_ohlcv`, so they are a
  cache, not a dataset, and the reproducibility rule holds.
- **Daily IC series** go to `alpha_ic`: 101 alphas × 4 horizons × 2 markets × ~5,500 days
  ≈ 4.4M rows per run. They are small and queryable. `run_id` is part of the key because a
  convention variant (argmax, adv units, basis) is a different series of the "same" alpha.
- `alpha_runs` records each run's options, git SHA, panel fingerprint (symbol count, date range,
  `max(fetched_at)`), and pre-registration hash.
- `backtest_results` gets one row per (alpha, market) at the primary horizon (5d), for the
  net-of-cost decile long-short, with `strategy_name = alpha101_<id>_<market>_h5`,
  `params` = JSON {run_id, options, costs, script, git_sha}, and
  `validation_status = 'pending'`. `research_log` gets one row per hypothesis (`research_plan.md`)
  linked to the run's summary row.

## 8. DDL (new tables; `CREATE TABLE IF NOT EXISTS` only; never DROP/TRUNCATE)

```sql
CREATE TABLE IF NOT EXISTS security_master (
  symbol              VARCHAR(20)  NOT NULL,           -- Yahoo symbol (BRK-B, RCI-B.TO)
  name                VARCHAR(200) NULL,
  market              CHAR(2)      NOT NULL,           -- US | CA
  exchange            VARCHAR(20)  NULL,
  quote_type          VARCHAR(20)  NULL,               -- EQUITY | ETF | INDEX
  sector              VARCHAR(80)  NULL,
  industry            VARCHAR(120) NULL,
  market_cap_snapshot DOUBLE       NULL,               -- yfinance marketCap on fetched_at (NOT PIT)
  shares_outstanding  DOUBLE       NULL,               -- yfinance sharesOutstanding on fetched_at
  currency            CHAR(3)      NULL,
  source              VARCHAR(30)  NOT NULL DEFAULT 'yfinance',
  fetched_at          DATETIME     NOT NULL,
  PRIMARY KEY (symbol),
  KEY idx_market (market)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS universe_membership (
  symbol      VARCHAR(20) NOT NULL,
  market      CHAR(2)     NOT NULL,
  source      VARCHAR(40) NOT NULL,   -- sp500_wiki_current | sp500_wiki_changes | tsx_wiki_current
                                      -- | liquidity_top500_us | liquidity_top200_ca
  as_of       DATE        NOT NULL,   -- state holds from as_of until the next row for (symbol, source)
  in_universe TINYINT(1)  NOT NULL,
  created_at  TIMESTAMP   NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, source, as_of),
  KEY idx_source_date (source, as_of)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS alpha_runs (
  run_id            VARCHAR(40)  NOT NULL,   -- e.g. 20261020T0300Z-ab12cd3
  created_at        TIMESTAMP    NULL DEFAULT CURRENT_TIMESTAMP,
  git_sha           CHAR(40)     NOT NULL,
  script            VARCHAR(200) NOT NULL,
  options_json      TEXT         NOT NULL,   -- EvalOptions + basis + vwap proxy + costs
  panel_fingerprint TEXT         NOT NULL,
  prereg_sha256     CHAR(64)     NULL,       -- required when the holdout is unlocked
  holdout_unlocked  TINYINT(1)   NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS alpha_ic (
  run_id   VARCHAR(40) NOT NULL,
  alpha_id VARCHAR(40) NOT NULL,             -- alpha001..alpha101, or a user alpha name
  market   CHAR(2)     NOT NULL,
  horizon  SMALLINT    NOT NULL,
  date     DATE        NOT NULL,
  ic       DOUBLE      NULL,
  n        SMALLINT    NOT NULL,
  PRIMARY KEY (run_id, alpha_id, market, horizon, date),
  KEY idx_alpha (alpha_id, horizon, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 9. Performance

Sizing: 700 names × 5,000 days = 3.5M cells, 28 MB per float64 array. Measured on a synthetic
panel of exactly that size (`scripts/bench_operators.py`; numbers in `performance.md`, generated):
- Measured order of cost: `correlation` is the most expensive operator: five pandas rolling
  passes (cov, two variances, two means for the flat-window guard), then `ts_argmax`/`ts_argmin`
  and `ts_rank` with long windows. Those are numpy `sliding_window_view` scans, O(T·N·d), in
  column chunks capped at 256 MB. Next come `covariance` and the cross-sectional `rank` (pandas
  `rank(axis=1)`). Rolling sum/std/min/max, `decay_linear`, `ts_product` and `indneutralize`
  (O(T·N) per level via `bincount`) are a few tenths of a second each.
- All 101 alphas evaluate in roughly six and a half minutes single-process at 700 × 5,000.
  Alpha 36 is the slowest, and peak RSS stays under 2 GB (exact figures in `performance.md`). A
  research run is 2 markets (500 + 200 names, about the same total as the benchmark) × 5
  convention variants, so about half an hour of evaluation plus IC/deciles. No numba/polars is
  needed. If speed ever matters, the cheapest wins are (i) a cumulative-sum correlation kernel
  with the same flat-window guard and (ii) a numpy argsort-based cross-sectional rank, each a
  drop-in `ndarray -> ndarray` replacement behind the existing tests.
- Use a fresh `Evaluator` per alpha, or call `clear()`, to bound memo memory. Sharing one evaluator
  across alphas saves time only on common sub-expressions such as `adv20` and `rank(volume)`.

## 10. Risks and open issues (ranked)

1. **Conventions with sign-level impact** (ts_argmax origin, adv units) cannot be settled from
   the paper or official docs. Mitigation: pre-register the defaults, run the alternative as a
   counted variant, and report paired differences. Never pick by IC.
2. **Survivorship in the candidate pool**, plus the static sector/cap look-ahead. Mitigation:
   §5 quantification; industry and cap alphas are reported separately.
3. **Alpha decay.** These alphas were in production in 2010–2013, published in 2015, and are
   widely replicated. The prior for US large-cap IC in 2019–2026 is close to zero
   (`research_plan.md`).
4. **Yahoo data quality** (stale opens, re-adjusted history, missing delisted names). Mitigation:
   the hygiene rules in §4.5 and the re-fetch rule in §4.2.
5. **Delay.** Most alphas were designed for delay-1 (trade at the next open). Our default trades
   at the next close (lag 1, close execution), which is more conservative. `execution="open"` is
   the secondary run.
