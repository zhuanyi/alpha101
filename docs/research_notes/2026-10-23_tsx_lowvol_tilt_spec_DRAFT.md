# TSX low-volatility tilt: pre-registration spec (DRAFT, not hash-locked)

**Status:** DRAFT v0, 2026-10-23. Design only: no volatility rank, return or outcome was computed to write this. It is to be reviewed and hash-locked before any backtest. **[OPEN]** marks an open choice.
**Data:** the PIT panel of this study (`research/tsxpanel/`: iShares XIC and XCS holdings; licence personal, non-commercial; raw files and derived tables git-ignored). Companion: `2026-10-23_tsx_deletion_rebound_spec_DRAFT.md`. Structure copied from the tax-loss spec (v2).

---

## 0. Hypothesis and prior

**Claim.** Within the S&P/TSX Composite, the lowest-volatility stocks earn at least the market's return with less risk, so a long-only, equal-weight basket of the lowest-volatility members, rebalanced slowly, gives a higher Sharpe ratio than XIC with a CAPM alpha above zero after retail costs. The US and international evidence on the low-volatility anomaly (Ang, Hodrick, Xing & Zhang 2006; Blitz & van Vliet 2007; Baker, Bradley & Wurgler 2011; Frazzini & Pedersen 2014 on betting against beta. Cited from memory, not re-read; verify before quoting) is a risk-adjusted effect, not a raw-return effect.

**What is at stake.** The real alternative is a cheap listed product: iShares MSCI Canada Min Vol (XMV.TO) and BMO Low Volatility Canadian Equity (ZLB.TO). A DIY basket is worth building only if it matches such a fund at lower cost or at a measurably better Sharpe. Both funds are benchmarks (§3), with their dates of inception noted. **Quality is explicitly OUT of scope:** there are no free point-in-time fundamentals, and a "low-volatility plus quality" design would need them.

**Prior (judgement):** about 10% ADVANCE, 55% UNDERPOWERED, 35% REJECT. A low-vol tilt is a small, slow effect: with about 12 years of data the test cannot see an alpha of 2%/yr (§5.3).

## 1. Universe and data

- **Universe at each rebalance date:** the names in the XIC holdings file of that date (about 220-250 names), point in time, including names that later died or were deleted. XCS-only names are excluded from the primary (illiquid, wide spreads, and a vol rank among them would pick up stale prices). Variant: XIC ∪ XCS.
- **Prices:** holdings prices only (`Price`, or `Market Value / Shares` when `-`). The panel is dense in the review windows and at month-ends, so **returns and volatility are computed from month-end files** (the last file of each month, with the 3-business-day fallback of the tax-loss study). Splits are removed with `prices.detect_splits` before returns are formed; a name with an unresolved |monthly return| above 100% is listed and manually cleared before any result is read (the big-move gate). Placeholder (`-` ticker) rows are recovered by name match as in the tax-loss study.
- **Dividends:** the basket tilts toward high-yield sectors (utilities, REITs, telecoms, banks), so price-only returns understate it. Dividends come from the Yahoo `dividend` column for names whose Yahoo series passes the identity check, entered as a yield on the ex-date; names without data are reported by share of basket weight and bounded (yield at 0 and at the median of the covered names). The benchmark gets the same treatment, so the comparison is like for like. XIC.TO, XMV.TO and ZLB.TO are compared on `adj_close` (total return).
- **Not used:** daily Yahoo prices for the volatility estimate (survivors only: the universe would be censored); any fundamental data.

## 2. Signal and portfolio

- **Volatility estimate:** standard deviation of the last **36 monthly price returns** (dividend-adjusted per §1) ending at the rebalance date; at least **24 valid monthly returns** required, else the name is ineligible at that date. Annualise by √12. The estimate has a relative standard error of about 12% (√(1/(2·36)) under normality): a coarse but adequate rank statistic when the cross-sectional spread runs from roughly 15% to over 60% a year. A monthly return needs the month-end files on both sides. The panel starts 2011-02 and has no month-end file for 2011-11, 2012-01 or 2017-01 to 2017-06, so by 2013-03 only 21 valid monthly returns exist; the 24th arrives at 2013-06, which is the **first rebalance (2013-06)**. After the 2017 gap the 36-month window holds 29-30 valid returns, still above the minimum. A return that spans a missing month-end is not a valid monthly return and is dropped from the volatility estimate (never stretched to a month). **[OPEN: confirm; history is the binding constraint here.]**
- **Rebalance:** quarter-end (Mar, Jun, Sep, Dec month-ends), executed at that month-end close (holdings price); the signal uses data up to and including that date (a one-day-lag variant, executing at the next file, is a robustness check, because the rank uses the close it trades at).
- **Portfolio P:** the **30 lowest-volatility eligible names**, equal weight. **Hysteresis:** an existing holding is kept while its rank is within the lowest 45 (this cuts turnover; without it the rank noise from the 12% estimation error churns the tail). Names that leave XIC between rebalances are held to the next rebalance and exited with the §1.4 hierarchy of the tax-loss spec (other ETF's file, identity-checked Yahoo return, logged takeover, else last price × 0.70 tagged `unresolved`); the `unresolved` share is reported.
- **Variants (reported, not promoted):** semi-annual rebalance; 20 and 50 names; sector-neutral (rank within sector, take the lowest third of each); a 50/50 blend of P and XIC; XIC ∪ XCS universe; 24-month and 60-month windows; the volatility-ranked highest-30 basket (negative control, §3).
- **Turnover and tickets:** reported per quarter; with a book of C$100,000 and 30 names, each ticket is about C$3,300.

### 2.1 The Jan-Jun 2017 gap (pre-registered handling)

There are no month-end files for 2017-01 to 2017-06. (a) The 2017-03 and 2017-06 rebalances cannot happen: the portfolio is held through and rebalanced at the first quarter-end with a file (2017-09). (b) The portfolio's return across the hole is booked as ONE multi-month period (2016-12 month-end to 2017-07 month-end, 7 months), computed from the holdings held through it with the §1 exit hierarchy for names that left. (c) That observation is excluded from the monthly t-statistics (it is not a monthly return) and reported separately; Newey-West lags are counted on the calendar-month index, so the hole is treated as missing observations and the gap is not closed up. (d) The cumulative-return and drawdown figures include the multi-month return. (e) A sensitivity run omits the whole of 2017 from all series. The 2017 volatility estimates lose 7 returns each (§2).

## 3. Benchmarks and placebos

1. **Same-universe benchmark:** equal-weight all eligible XIC names at the same dates (price plus dividends as §1). The selection claim is "low-vol beats the universe it is drawn from".
2. **XIC.TO total return** (`adj_close`): the investable passive alternative. **XMV.TO and ZLB.TO** total return: the investable low-vol alternatives (report their start dates; the comparison window is the overlap).
3. **Random-basket placebo:** 1,000 draws (seed 20261023, `numpy.random.default_rng`) of 30 names uniformly from the eligible universe at each rebalance, same hysteresis-free rule, same costs. Report the percentile of the net P statistic (Sharpe difference and CAPM alpha).
4. **Negative control:** the highest-volatility 30. The anomaly predicts it underperforms on a risk-adjusted basis; if it also beats the universe, the result is a size or small-cap effect, not volatility.
5. **Stale-signal placebo:** the same rule with the volatility window ending 12 months earlier. A real volatility signal is persistent, so this should do worse than P but still beat random; if it does as well as P, the rank is mostly a stable sector/size proxy.
6. **Beta control:** report the regression of P's excess return on the market and on the equal-weight universe; the claim of a "low-vol effect" requires alpha after that, not just a low beta.

## 4. Costs

Cost parameters are collected in [`docs/cost_assumptions.md`](../cost_assumptions.md).

Per ticket: commission C$9.99 (standard; C$7.00 high-volume-tier sensitivity); half-spread h from the tax-loss bucket table for XIC members (≥ C$20: 5 bp; C$5-20: 15 bp; C$2-5: 40 bp, assumptions, ×2 sensitivity), plus 10 bp timing slippage per side (0 and 25 bp sensitivities). Cost of a quarter = tickets × (commission/ticket size + h + s_close). Tax: TFSA or RRSP, no tax layer. Management-fee comparison with XMV/ZLB: report the DIY cost drag in bp per year next to each fund's published MER (fetched at run time, not assumed).

## 5. Statistics, power and decision rule

### 5.1 Observations and tests

- Monthly net excess returns of P over the same-universe benchmark and over XIC.TO, from the first rebalance (2013-06) to the last discovery month. The primary decision series is **monthly after-cost excess return over the same-universe benchmark** (selection), with the XIC.TO comparison for tradability.
- **t_dec** = the smaller of the plain t and the Newey-West t (3 lags) on that series. Report CAPM alpha and beta against XIC.TO (NW 3 lags), Sharpe of P and of XIC.TO with a bootstrap CI (block bootstrap, 12-month blocks, 10,000 reps), the Sharpe difference, maximum drawdown, the correlation to XIC.TO, annual turnover, and sector weights through time.
- Pooled cross-sectional regressions are descriptive only.

### 5.2 Variants and multiplicity

`alpha101.stats.effective_number_of_tests` / `adjusted_threshold` over the variant set, reported next to the P result with its label (two-sided normal-quantile Sidak bar and Student-t equivalent). The decision uses P alone.

### 5.3 Power (state it in the results)

With about 120 monthly observations and a monthly tracking error of 1.5-2% against the same-universe benchmark (an assumption; the run measures it), the minimum detectable monthly excess at t = 2 is 2 · sd / √120 ≈ 0.18 · sd ≈ 0.27-0.35% per month, i.e. 3.3-4.3% a year. Published low-vol alphas are smaller than that, so **UNDERPOWERED is the modal outcome by construction**. The Sharpe-difference test is no better. The deliverable of this study is therefore a measured tracking error, turnover and cost drag, and a bounded alpha estimate, as much as a verdict.

### 5.4 Decision rule (written before any data is seen), P at C$100k

- **ADVANCE** if all hold: (1) mean after-cost excess over the same-universe benchmark > 0 and t_dec ≥ 2.0; (2) random-basket placebo percentile ≥ 95 on the net CAPM alpha; (3) CAPM alpha (after costs) > 0 and Sharpe of P > Sharpe of XIC.TO, with P's maximum drawdown no worse than XIC.TO's; (4) the negative control (highest-vol 30) has a Sharpe below XIC.TO's; (5) the verdict is unchanged with `unresolved` exits at -100%, with the one-day-lag execution, and with dividends bounded at the low yield; (6) P's net Sharpe is not below ZLB.TO's and XMV.TO's over their overlap (the investable alternatives).
- **REJECT** if mean after-cost excess ≤ 0, or the Sharpe of P ≤ the Sharpe of XIC.TO, or the placebo percentile < 75.
- **UNDERPOWERED** in every other case. Action: log the measured tracking error and cost drag; no capital; keep the basket as a paper portfolio.
- Checked in the order REJECT, ADVANCE, UNDERPOWERED. **[OPEN] the diversifier leg of the viability bar** (positive CAPM alpha after costs, Sharpe above passive, low correlation) is encoded in (3): "low correlation" is not applied, because a long-only equity basket has a correlation to XIC.TO near 0.9 by construction. Confirm.

## 6. Holdout (to seal)

- **Discovery:** monthly returns from 2013-06 to 2023-12. **Sealed:** 2024-01 onward (the panel loader drops dates after 2023-12-31 unless `--unlock-holdout` is passed with the lock hash present), evaluated once with the frozen P. The sealed period has about 25 months: it can only contradict a large effect, and the results say so.
- Variants are never promoted this year; anything that beats P becomes next year's pre-registration.

## 7. What would make this wrong

| Risk | Bias | Guard |
|---|---|---|
| Monthly-return volatility is noisy (12% relative error) | rank churn and weak signal | hysteresis, semi-annual variant, stale-signal placebo, 24/60-month windows |
| Price-only returns | understates a high-yield basket | dividends from Yahoo with identity check, bounded where missing, same treatment for benchmarks |
| Sector concentration (utilities, REITs, telecoms, banks) | an interest-rate or sector bet looks like a volatility effect | sector weights reported, sector-neutral variant, beta control |
| Names leaving XIC during the quarter | exit blindness | §2 hierarchy, `unresolved` share, bounds |
| Look-ahead (rank uses the close it trades at) | slightly optimistic | one-day-lag variant; rank function takes only data at or before the date (truncation test) |
| Splits and consolidations in month-end prices | false volatility | `detect_splits` plus the big-move gate |
| Thin history (12 years, one rate cycle, the 2020 crash) | low power, regime dependence | power paragraph; subperiod table (2013-2019, 2020, 2021-2023) |
| Survivorship in the investable comparators (XMV, ZLB exist because they survived) | flatters the benchmark | caveat in the results |
| Forking paths after the first run | the reported P is not P | run log with spec hash and code commit; later changes listed with a diff summary |

## 8. Implementation outline

1. Month-end price table from the panel (split-adjusted, placeholders recovered). 2. `vol_rank(prices_le_date)`: the only function that sees ranking data. 3. Portfolio with hysteresis, costs, turnover. 4. Dividends with identity check. 5. Benchmarks, placebos, negative control, stale-signal placebo. 6. Stats and `decide()` coded literally. 7. Report generated from CSVs only.
Tests (synthetic, no network): the rank at date d is unchanged when all data after d is truncated; hysteresis keeps holdings inside rank 45 and replaces them outside; a 1:10 consolidation does not create a high-volatility name; the 24-return minimum drops short histories; `decide()` hits each branch on constructed inputs.

## 9. Open questions

1. First rebalance 2013-06 with 24 valid returns, or wait for 36 (2014-06 at the earliest, given the missing 2011-11 and 2012-01 month-ends)? 2. 30 names plus hysteresis 45, or a cap-aware weighting? 3. Whether XMV/ZLB parity (condition 6) belongs in ADVANCE or only in the report. 4. Whether the XIC ∪ XCS variant should be promoted (more cross-sectional spread, worse liquidity). 5. The diversifier leg as noted in §5.4.
