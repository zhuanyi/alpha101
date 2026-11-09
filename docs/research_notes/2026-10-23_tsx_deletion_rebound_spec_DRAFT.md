# TSX index-deletion rebound: pre-registration spec (DRAFT, not hash-locked)

**Status:** DRAFT v0, 2026-10-23. Design only: the author of this draft computed no strategy return, no post-deletion price ratio and no outcome of any kind. This file is to be reviewed and hash-locked before any backtest runs. Anything marked **[OPEN]** is a choice left open on purpose.
**Data:** the PIT panel and event table of this study (`research/tsxpanel/`), iShares XIC and XCS holdings files (BlackRock licence: personal, non-commercial; raw files, panel and `events.csv` stay git-ignored and are never redistributed). Companion draft: `2026-10-23_tsx_lowvol_tilt_spec_DRAFT.md`. Structure copied from the tax-loss spec (`2026-10-19_tsx_taxloss_rebound_spec.md`, v2).
**Coverage and timing facts this spec relies on:** `2026-10-23_tsx_pit_panel_coverage.md` (event counts per year, where a deletion first disappears from the files relative to the review Friday, price coverage after deletion, announcement-date feasibility).

---

## 0. Hypothesis and prior

**Mechanism.** The S&P/TSX Composite is tracked by index funds (XIC is one). A name deleted at a quarterly review is sold by trackers at the close of the last day it is in the index, whatever its price. That is price pressure from a non-informational, scheduled flow. If liquidity providers are paid to absorb it, the deleted names should be cheap at the effective-date close and recover afterwards. The US literature on S&P 500 deletions finds a decline into the effective date and a large reversal afterwards, a gap that is not symmetric with additions (Harris & Gurel 1986; Shleifer 1986; Chen, Noronha & Singal 2004; Petajisto 2011 on the inclusion premium. All cited from memory, not re-read; verify before quoting). I know of no Canadian study and have not searched for one **[OPEN: search before lock]**.

**Why it may not work here.** (i) A Composite deletion is usually a small name that stays listed, often moving to the SmallCap index (XCS), so part of the passive holder base stays. (ii) Deleted names are past losers, so a short-term reversal is a competing explanation; the benchmark must control for it (§3). (iii) The deletion is announced about two weeks before the effective date, so the price pressure may sit before our entry. (iv) Free data cannot see the exit of names that leave both ETFs and are not on Yahoo (the loser-tail blindness of the tax-loss study; coverage table in the coverage note).

**Prior (judgement, not a measurement):** about 10% ADVANCE, 45% UNDERPOWERED, 45% REJECT. The track record in this repo is negative after retail costs (conditional overnight, slow long-only 101-alpha book, sector momentum); this is a flow-based effect, not a signal, which is the only reason the prior is not lower.

## 1. Universe, events and data

- **File timing (measured, see the coverage note).** The XIC and XCS file dated T (the review Friday) already shows the post-review membership; spot checks against the S&P DJI release counts for 2023-03 and 2024-12 match. The last file that holds a deleted name is therefore the one before T.
- **Review date T** = the third Friday of Mar/Jun/Sep/Dec (the S&P/TSX quarterly review takes effect after that close; the Monday after is the first day without the deleted names). The code takes T as the last trading day on or before that Friday.
- **Event = one quarterly review** (the unit of inference, §5). Basket = the names deleted from XIC at that review.
- **Deleted name (eligible)** = a row of `events.csv` with `etf = XIC`, `direction = delete`, `class ∈ {to_xcs, left_both}`, whose first-out file date is T, T+1 or T+2 trading days after a review Friday (measured in the coverage note: the iShares file as of the review Friday T already shows the post-review composition, so a review deletion is absent from the T file in the large majority of cases and its last holdings price is the T-1 file; in 2012-2013 and part of 2014 the files lagged one to two days, so the name is still in the T file and absent from T+1 or T+2; the offset-2 acceptance can admit an unrelated off-cycle deletion, one case in the 2022-12 window, which the S&P release would exclude in live use), not flagged `unconfirmed`, `wide_bracket`, `spans_2017_gap` or `scheme_switch_bracket`. `ticker_change` rows are renames, never deletions. Acquisitions are not inferred; a name that leaves because of a takeover can only enter the basket through the exit hierarchy of §2.4 (takeover requires a logged source).
- **Point-in-time rule:** deletions are knowable at T (they are announced before it). The file diff is only used to identify them after the fact; the live version of the trade uses the S&P DJI announcement. The announcement date is NOT an input of the primary (it could be found for only part of the sample, see the coverage note).
- **Daily-resolution reviews only.** Daily files: the 2012-03 review has stubs only, 2012-06 and 2012-12 are present, 2012-09 has 4 stub dates in its window; the 2017-03 and 2017-06 reviews fall in the Jan-Jun 2017 gap and have no usable events. A review is usable only if the T-1 file (XIC) and a file at T+10 (XIC, or XCS for moved names) exist. Of the 47 reviews with at least one review-aligned XIC deletion, **44 are usable**: 2014-12-19 and 2021-06-18 lack T+10 files and 2025-06-20 lacks the T-1 file (a stub on 19 June 2025). Excluded reviews are listed with the reason; no event is dropped for any other reason.
- **Secondary groups** (reported, never promoted): XCS deletions (names leaving the SmallCap index); deletions split by class (`to_xcs` versus `left_both`); XIC additions (the mirror effect, expected to be a pre-effective run-up, so entry at T is wrong for them: reported for completeness only).

## 2. Signal, entry, exit

### 2.1 Primary specification P (frozen at lock)

- **Entry:** buy every eligible deleted name at the **close of the last trading day before T (T-1)**, priced from the last holdings file that still holds the name (the file as of T-1; `Price` or `Market Value / Shares`). Using T-1 in every era keeps the rule uniform (in the lagged era the T file would also hold the name, but a rule that changed with the era would be a forking path). This is the one entry price every deleted name has: on T itself the name is already gone from the XIC file, and a `left_both` name has no holdings price at all. It is also conservative for a V-shaped effect: the T-1 buyer sits through the selling into the T close. The T-1 file must exist (a missing T-1 file makes the event unusable and is listed).
- **Exit:** sell at the **close of T+10 trading days**.
- **Weights:** equal weight within the basket; basket return = mean of name returns. Book C$100,000 split equally across the basket (a basket of 2-6 names gives C$17k-50k tickets).
- **Horizon rationale:** the files fetched cover T-10..T+10 trading days, and the reversal literature puts most of it inside two weeks. **[OPEN: confirm 10; variants below]**.

### 2.2 Variants (reported, not promoted)

Exit at T+5 and T+20 (T+20 needs Yahoo or the other ETF's file; coverage applies); entry at the T close where a price exists (the XCS file for `to_xcs` names, or an identity-checked Yahoo close; this tests whether the T-1 entry gives away the effective-day move); entry at T-6 (a proxy for "buy after the announcement"; needs no announcement date); `left_both` only; `to_xcs` only; exit at the first file day after T.

### 2.3 Returns

Price return from holdings prices, as-traded. **Corporate actions:** use the tax-loss detector (`prices.detect_splits`, share counts normalised by the fund-wide median ratio); a name with a split inside its window gets the split-adjusted ratio, and any name with |window return| above 100% is listed and manually cleared before any result is read (the `big_moves` gate of the tax-loss study, same file format). Dividends: ex-dividend inside the window from the Yahoo `dividend` column only when the Yahoo series passes the identity check; otherwise the price return stands and the share of names without dividend data is reported.

### 2.4 Exit-price hierarchy (copied from the tax-loss spec 1.4)

For a name whose exit-day price is absent from the XIC file: (a) the other ETF's file on the exit day; (b) an identity-checked Yahoo return applied to the T holdings price (never mixing Yahoo's split-adjusted level with holdings levels); (c) a cash takeover with a logged source URL; (d) otherwise the last holdings price × (1 − 0.30), tagged `unresolved`. **If `unresolved` exceeds 10% of basket weight (mean over reviews), the result is labelled data-limited**; bounds with `unresolved` at 0% and at -100% are always reported. **Expected from the coverage note (counts only, no returns): of the 185 review-aligned XIC deletions, 157 are `other_etf` (149 of them still in the XCS file at T+10), 9 are `yahoo`, 19 are `unresolved`; `takeover` is 0 because the source list holds no rows, so any acquired name sits inside `unresolved`. The unresolved share of basket weight is about 20% pooled (weights at the last holdings file), so the data-limited label is the expected outcome for the all-deletions basket.** The `to_xcs`-only arm has holdings prices for the exit and no blindness; see open question 6.

## 3. Benchmarks and placebos

1. **Matched-control benchmark (the selection benchmark).** For each deleted name, K = 3 controls drawn from XIC members at T that are NOT deleted at that review, nearest on (a) log ETF weight at T-1, (b) trailing price return over the 63 trading days before T-1 (from the holdings file closest to T-64 within 12 days; a name with none is not eligible as control), z-scored within the review; same sector breaks ties. The control window is the same as the name's (T-1 close to T+10 close). Controls stay in XIC, so they always have holdings prices (no exit blindness on the control side; this asymmetry biases AGAINST the deleted names and is reported). Excess of a name = its return minus the mean control return over the same window.
2. **Market benchmark:** XIC.TO total return (`adj_close`, `yahoo_ohlcv`) over the same window, for the tradability test.
3. **Random-basket placebo:** per review, draw as many names as the basket from XIC members at T, size-matched by weight bucket, no deleted names; 1,000 draws (`numpy.random.default_rng`, seed 20261023, without replacement within a draw, independent across reviews); same costs and exit rules. Report the percentile of the actual net mean.
4. **Placebo dates:** the same construction on the third Friday of the 8 months without a review (Jan, Feb, Apr, May, Jul, Aug, Oct, Nov), with pseudo-deleted names = the names with the lowest ETF weight among the members that are deleted at the NEXT review (a placebo for date, not for selection). The real review dates should stand out against the pseudo dates; report the real quarter's rank among the 9 (strictly 1st, ties are not 1st). **[OPEN: pseudo-deleted definition uses look-ahead on purpose and must never be traded; confirm it is acceptable as a pure placebo.]**
5. **Negative control:** XIC additions entered at T and held 10 days, expected NOT to rebound (they should run up before T). A positive effect there means the design picks up a general T-window effect.

## 4. Costs

Cost parameters are collected in [`docs/cost_assumptions.md`](../cost_assumptions.md).

Per name per side, in bp of the ticket (same form as the tax-loss spec 4): commission C$9.99 per order (standard; C$7.00 high-volume tier as a sensitivity) over book/N; half-spread h from the Abdi & Ranaldo (2017) close-high-low estimator on Yahoo OHLC over the 60 trading days before T-1 (median full spread / 2, floor half a tick over price), bucketed by {XIC member, XCS-only} × price band, with the tax-loss fallback table when a bucket has fewer than 5 names (assumptions: XIC ≥ C$20: 5 bp; XCS-only C$2-5: 60 bp, etc.); names without Yahoo OHLC take their bucket median. Timing slippage s_close 10 bp per side (sensitivities 0 and 25 bp). **Entry-side bounce:** index sellers hit the bid into the close of T, so the T close may print near the bid; run E2h (entry pays the full spread 2h) as a required robustness check, like the tax-loss study. A basket trades 4 times a year, so about 4 × (2 × N) tickets: commission is a minor cost, spread is the cost that matters. Account: TFSA or RRSP, no tax layer.

## 5. Statistics, power, decision rule

- **Unit:** one review (N = 44 usable reviews, per §1 and the coverage note; 185 review-aligned XIC deletions in 47 reviews, about 3.9 names per review). g_q = mean over the basket of (net name return - matched-control mean gross return) for review q; the net after-cost g_q is the decision statistic (same convention as the tax-loss spec 5.1). n_q = net basket return - XIC.TO total return (tradability).
- **Tests:** t_dec = min(plain one-sample t, Newey-West t with 1 lag) on {g_q}; the same on gross. Wilcoxon signed-rank, share of positive reviews, 10,000-rep block bootstrap CI (block = one calendar year). Pooled name-level regression is descriptive only (names within a review are correlated; its naive t overstates N). Dose-response by deletion size (descriptive): mean g_q by terciles of weight at T.
- **Variants:** n_eff and Sidak-adjusted bars from `alpha101.stats.effective_number_of_tests` / `adjusted_threshold`, reported under the label "two-sided normal quantile" next to the Student-t equivalent. The decision uses P alone.
- **Power (state it in the results):** with N = 44 reviews, MDE (net, t = 2) = 2 · sd / √44 = 0.30 · sd. If the review-level sd of g_q is 4%, the MDE is about 1.1% per review; a small-name basket of 2-6 names plausibly has sd of 4-8%, so the MDE is about 1.1-2.3% per 10-day window. Smaller effects come out UNDERPOWERED by construction. The sd is measured by the run, not assumed.
- **Decision rule (written before any data is seen), P at C$100k:**
  - **ADVANCE** if: (1) mean net g_q > 0 and t_dec ≥ 2.0; (2) random-basket placebo percentile of the net mean ≥ 95; (3) mean n_q > 0 and n_q > 0 in at least 55% of reviews; (4) both halves (2012-2018, 2019-2025) have a positive mean g_q; (5) the verdict is unchanged with `unresolved` at -100%, under E2h, and without the 2020 reviews; (6) the negative control (additions) shows no positive net mean with t ≥ 1.
  - **REJECT** if mean gross g_q ≤ 0, or the gross t < 1.0, or the placebo percentile < 75.
  - **WEAK** if (2) and (4) hold and the gross t ≥ 1.771 but (1) or (3) fails. Action: check C$250k; otherwise no trade.
  - **UNDERPOWERED** otherwise. Action: no capital; shadow-log the December 2026 review.
  - Outcomes are checked in the order REJECT, ADVANCE, WEAK, UNDERPOWERED.
- **Diversifier route (viability bar, second leg) [OPEN: decide before lock whether this is a pass route]:** the sleeve is invested about 10 of 63 trading days each quarter, so its correlation to anything is near zero by construction and "low correlation" cannot discriminate. Default here: reported only (CAPM alpha after costs, beta, Sharpe over the invested windows, correlation to XIC), not a pass route, for the same reason the tax-loss spec gives.

## 6. Holdout (to seal)

- **Discovery:** reviews from Dec 2012 through Dec 2025 (44 usable reviews), all in the P evaluation. A split inside this sample would leave about 11 reviews in a holdout with no power, so none is made.
- **Sealed:** the reviews of 2026-03-20, 2026-06-19 and 2026-09-18 (their files are NOT fetched or parsed before the lock; the code refuses to load panel dates after 2026-03-01 unless `--unlock-2026` is passed with the lock hash present) and the December 2026 review as a live shadow. Three sealed events cannot confirm anything; they can only contradict a large effect. State this in the results.
- **Variants are never promoted this year;** a variant that beats P becomes next year's pre-registration.

## 7. What would make this wrong

| Risk | Bias | Guard |
|---|---|---|
| Deleted names are past losers (short-term reversal) | rebound looks like an index effect | matched controls on trailing return and weight; negative control; placebo dates |
| Announcement pre-pressure | the price effect sits before T | entry at T-1 by design; the T-6 variant shows what entry earlier would add |
| Exit blindness (names leaving both ETFs with no Yahoo series) | omitting the worst exits inflates returns | §2.4 hierarchy, `unresolved` share, bounds at 0% and -100%, "data-limited" label |
| Fund implementation lag (the ETF may drop a name a day or two off the index date) | the file-based event date is not the index date | review-offset distribution measured in the coverage note (offset 0 dominates; offset 1 is the minority); T or T+1 accepted, the rest are off-cycle |
| Renames counted as deletions | fake events | `ticker_change` class by normalised name; self-check on the 2012-05 scheme switch; unmatched renames flagged `rename_candidate` for manual review |
| Splits and consolidations | false returns | tax-loss detector plus the big-move gate |
| Close prices not tradable (index sellers at the bid, no MOC guarantee) | overstates entry | E2h, s_close, calibration of quoted spreads before any live trade |
| Ticker reuse on Yahoo | wrong exit prices | identity check |
| Small N, clustered names | false positives and negatives | review as the unit; power paragraph; UNDERPOWERED outcome has an action |
| Forking paths after the first run | the reported P is no longer P | every run logs spec hash and code commit; later code changes are listed with a diff summary |
| Regime change (T+1 settlement since 2024; more passive AUM) | past effect may not hold | 2024-2025 reported separately (descriptive) |

## 8. Implementation outline (each step ends with an artifact)

1. `reviews.py` and `events.py` (this study): review dates, event table. 2. Basket builder: eligible deletions per review and the measured offset window. 3. Return layer: holdings prices, split detector, exit hierarchy, dividends. 4. Matched controls and placebos (pure functions, synthetic tests: no look-ahead in control matching; controls never include deleted names; K matches return the same set when the future is truncated). 5. Costs (reuse the tax-loss Abdi-Ranaldo and bucket code). 6. Stats and `decide()` coded literally in the §5 order. 7. Report generated from CSVs only.
Tests (pytest, synthetic fixtures, no network): control matching uses only data at or before T; truncating files after T+10 does not change the basket; the reorder of tickers does not change results; `decide()` returns each branch on constructed inputs.

## 9. Open questions

1. Exit horizon 10 versus 5 or 20 days. 2. Treat the diversifier leg as a pass route or not (default: not). 3. Pseudo-deleted definition for placebo dates (look-ahead by design). 4. Whether XCS deletions deserve a promoted arm (default: secondary). 5. Whether to pay for the announcement-date leg: the coverage note's probe found a release for 7 of 12 sampled reviews, all from 2020 on, so an announcement-date leg would cover only the recent third of the sample. 6. Because the all-deletions basket is expected to trip the data-limited gate, whether `to_xcs`-only (the clean-exit arm, about 85% of the names) becomes the primary and `left_both` the secondary, or the gate is kept as is.
