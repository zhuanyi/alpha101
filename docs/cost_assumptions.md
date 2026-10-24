# Cost assumptions

Every number below is read from a constant in this repository (file in the last column). They are retail-investor
assumptions for a Canadian investor trading listed stocks and ETFs through a discount broker; they are inputs, not
measurements, unless stated. Research notes refer to this file instead of naming a broker. "Desk" in the notes means
the broker's FX conversion desk.

## 1. Explicit commission

| item | value | where |
|---|---|---|
| Stock / ETF trade | 9.99 per trade, in the account currency (C$ on a TSX order, US$ on a US order) | `CostModel.commission` (`src/alpha101/costs.py`); `COSTS["commission"]` (`research/overnight`); `COMMISSION_CAD` (`research/sector_momentum`); `COMMISSION` (`research/taxloss/rebound.py`, `scripts/cost_edge_norbert.py`) |
| High-volume tier (sensitivity only) | 7.00 per trade | `commission_active_per_order` (`research/overnight`); comment in `costs.py`; studied as `commission_7` scenarios |
| Options | 9.99 per order + 1.25 per contract | `COMMISSION_BASE`, `COMMISSION_PER_CONTRACT` (`scripts/options_proxy_study.py`) |
| DLR-U.TO leg of Norbert's gambit | US$9.99 default, US$0 alternative (UNRESOLVED, reported both ways); the DLR.TO leg is C$9.99 | `DLRU_COMMISSION_USD`, `DLRU_ALT_USD` (`scripts/cost_edge_duallisting.py`) |
| How it is applied | `CostModel` spreads the flat fee pro rata over `notional_per_name` (default 5,000, i.e. 100k per side / 20 names); the slow long-only study uses `book / 20` (C$5k tickets at C$100k, C$12.5k at C$250k); the exact flat-fee accounting is cross-checked in `docs/slow_long_only.md` | `costs.py`, `scripts/slow_long_only.py` |

## 2. Implicit cost: half-spread and slippage (per side, in basis points of the traded value unless stated)

| study / module | US stocks and ETFs | Canadian stocks and ETFs | where |
|---|---|---|---|
| Generic cost model (alpha research, slow long-only) | 3 bp half-spread | 8 bp half-spread | `CostModel.half_spread_bps` |
| Overnight (Phase 0 defaults) | 0.5 bp (US ETF), 1.0 bp (US single names) | 2.0 bp (TSX ETF), 3.0 bp (TSX single names) | `DEFAULTS` (`research/overnight/overnight_intraday_test.py`) |
| Overnight (Phase 1 per-symbol) | SPY, QQQ, IWM 0.5; SSO 2.0; UPRO 2.0; TQQQ 1.0; QLD 2.0 | XIU.TO, XIC.TO 2.0 | `COSTS["half_spread_bp"]` (`research/overnight/overnight_conditional_test.py`) |
| Overnight open-fill slippage (one side, at the open) | 1.0 | 3.0 | `COSTS["open_slippage_bp"]` |
| Overnight timing slippage per side (no guaranteed MOC/MOO fills; entry near the close, exit near the open) | 1.0 (sensitivities 0 and 3) | 2.0 (sensitivities 0 and 3) | `COSTS["timing_slippage_bp_per_side"]`, scenarios `timing_0bp`, `timing_3bp` |
| Sector momentum | 2 bp for SPY and the 11 SPDR sector ETFs; 5 bp for every other ETF | not traded | `SPREAD_BP_SPDR`, `SPREAD_BP_OTHER` |
| Tax-loss rebound (TSX) | not traded | Estimated per name: Abdi-Ranaldo (2017) close-high-low spread over the last 60 sessions (at least 20 valid days), half of it, floored at half a tick (C$0.005) / price; bucket median (XIC member or XCS-only x price band) if a name has no estimate; fixed fallback table if a bucket has fewer than 5 names: XIC 40 bp (price < C$5), 15 bp (C$5-20), 5 bp (>= C$20); XCS 60 / 30 / 15 bp. Entry pays 2h in the `E2h` scenario; `h_x2` sensitivity doubles h | `rebound.py` (`HALF_TICK`, `SPREAD_WINDOW`, `MIN_SPREAD_DAYS`, `MIN_BUCKET`, `FALLBACK_H_BP`, `price_band`) |
| Tax-loss rebound timing slippage per side | n/a | 10 bp (sensitivities 0 and 25) | `S_CLOSE_BP` and scenarios `s_close_0`, `s_close_25` |
| Norbert's gambit / dual-listing venue choice | estimated per name and line | `used = max(Corwin-Schultz 2012 two-day high-low half-spread, Abdi-Ranaldo 2017 monthly estimate (at least 8 observations), half-tick)`, half-tick = 0.01 / 2 / median close | `cs_half_spread`, `ar_half_spread_month`, `spread_table` (`scripts/cost_edge_norbert.py`); same estimator reused in `cost_edge_duallisting.py` |
| Options proxy (monthly covered call on XIU) | n/a | C$0.03 per share half-spread on the option (ASSUMPTION; probability `close_frac` of a second commission and spread when the call is bought back, 0 in the base case) | `HALF_SPREAD`, `covered_call_cost_bp` (`scripts/options_proxy_study.py`) |

## 3. FX conversion (USD trade in a CAD account, or the reverse)

The desk quotes a listed spread by size of the converted amount, plus a surcharge of one third of the spread (all-in
= listed x 4/3). Per conversion; the margin of the desk's own rate over the interbank mid is NOT modelled
(`m_aff = 0`), so the figures are lower bounds on the desk cost.

| converted amount | listed spread | all-in (x 4/3) | source constant |
|---|---|---|---|
| below 25,000 | 1.26% | 1.68% | |
| 25,000 to 99,999 | 0.78% | 1.04% | |
| 100,000 to 249,999 | 0.46% | 0.6133% | |
| 250,000 to 999,999 | 0.27% | 0.36% | `FX_DESK_TIERS_PCT` in `costs.py` and `fx_tiers` in `research/overnight` (upper bounds 25k / 100k / 250k / 1M) |
| 1,000,000 and above | 0.19% | 0.2533% | |

**Known inconsistency in the code.** `scripts/cost_edge_norbert.py` (`FX_DESK_TIERS`, lower bounds 0 / 25k / 100k / 500k / 1M) puts the 0.46% tier over
100,000 to 499,999 and 0.27% from 500,000 to 999,999, i.e. it charges 0.46% (0.6133% all-in) for 250,000 to
499,999 where `costs.py` and the overnight study charge 0.27% (0.36% all-in). The cost-edge notes and
`tests/test_cost_edge_norbert.py` follow the second table; the generic `CostModel` and the overnight study follow the
first. The two disagree only between 250,000 and 499,999.

| variant | value | where |
|---|---|---|
| In-trade FX (UNVERIFIED reading of the broker's page) | amounts of 60,000 or more pay at least the 0.78% listed tier (all-in 1.04%) | `DESK_IN_TRADE_FROM`, `DESK_IN_TRADE_FLOOR` |
| Which currency sets the tier | both the USD amount (primary) and the CAD amount are shown (UNVERIFIED) | `basis` in `route_costs` |
| Conversions per round trip | two (in and out); a buy converts CAD to USD, a sell USD to CAD | `costs.py`, `cost_edge_duallisting.py` |

**Norbert's gambit (buy DLR.TO, journal to DLR-U.TO, sell in US$, or the reverse), one way, fraction of V:**
`c_NG(V) = 2k/V + h_DLR + h_DLRU + d`, with `k = 9.99` per order (two orders), `h` the estimated half-spreads from section 2,
and `d` the implied-rate deviation versus the Bank of Canada rate: 0 in the base case (the BoC rate is a session
mean, not a close), the observed 95th percentile as the adverse sensitivity. Journaling delay is modelled as risk only
(0, 1 and 3 business days). Whether DLR / DLR.U is on the broker's zero-commission ETF list is UNRESOLVED
(the sensitivity above covers it).

## 4. Other cost parameters

| item | value | where |
|---|---|---|
| Short borrow | 50 bp per year on the short weight (long-short books) | `CostModel.borrow_bps_per_year` |
| Margin financing of the settlement lag (overnight study) | 19.99% per year (US) / 22.99% per year (CA) on capital x settlement days / 365; settlement 1 day | `COSTS["margin_rate"]`, `COSTS["settlement_days"]` |
| Position sizes tested | overnight 25k / 50k / 100k / 250k (decision size 100k); dual listing and gambit 10k / 25k / 100k / 250k (/ 500k); sector momentum C$100k / 250k / 500k; options 25k / 100k / 250k | `COSTS["exposures"]`, `SIZES`, `NOTIONALS`, `BOOKS_CAD` |
| Sector momentum minimum trade | 2% of current equity (US$2,000 on a US$100k book); smaller non-exit trades are skipped | `MIN_TRADE_NOTIONAL` |
| Tax layer (overnight study, taxable accounts only) | capital gains 26.76%, income 53.53%, eligible-dividend marginal rates 53.53% (US) and 39.34% (CA), US withholding 15%; registered accounts: no tax layer | `TAX`, `DEFAULTS` |

## 5. Sources

Commission and FX tier figures were read from the broker's public fee pages on 2026-10-18 and were not re-verified by
the scripts (secondary-source claims are marked as such in the notes). Corwin and Schultz, "A Simple Way to Estimate
Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance (2012); Abdi and Ranaldo, "A Simple Estimation of
Bid-Ask Spreads from Daily Close, High, and Low Prices", Review of Financial Studies (2017).
