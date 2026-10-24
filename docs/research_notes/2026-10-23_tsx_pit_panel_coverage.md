# TSX point-in-time panel and index-event table: coverage report

**Date:** 2026-10-23 · **Code:** `research/tsxpanel/` (this study) · **Companion drafts:** `2026-10-23_tsx_deletion_rebound_spec_DRAFT.md`, `2026-10-23_tsx_lowvol_tilt_spec_DRAFT.md`.

**This report contains counts and date spans only.** No strategy return, post-deletion price ratio or outcome was computed. The panel, `events.csv` and every table behind this note are derived from iShares (BlackRock Canada) holdings files, licensed for personal, non-commercial use; they stay git-ignored under `research/tsxpanel/data/` and are not redistributed. The PR commits code, tests and this note only.

**Tax-loss study cache:** the daily-window fetch first ran with a symlink into the tax-loss study's raw cache. A review of my own run caught it within about two minutes: the run was stopped, the 94 files it had written were removed from that cache (file counts are back to the original 957 XIC and 956 XCS), and the cache was copied to this study's own directory (`research/tsxpanel/data/raw`, git-ignored). The remaining fetch wrote only there. The tax-loss study's inputs are unchanged, but the incident is recorded because for about two minutes they were not.

## 1. What was built

| Step | Module | Output (git-ignored) |
|---|---|---|
| Review dates and fetch plan | `reviews.py`, `fetch_windows.py` | `window_fetch_status.csv` |
| Panel | `panel.py` | `panel.parquet`, `panel_files.csv`, `placeholder_matches.csv`, `month_end_coverage.csv` |
| Events | `events.py`, `run_events.py` | `events.csv`, `events_per_year.csv`, `review_offsets.csv`, `off_cycle.csv` |
| Price coverage and exit classes | `coverage.py`, `run_coverage.py` | `price_coverage.csv`, `price_coverage_per_year.csv`, `review_aligned_exit.csv`, `exit_class` column of `events.csv` |
| Tables for this note | `tables.py` | `tables.md` |

Tax-loss code reused by import: `ishares` (fetch, parse, stub/mismatch classification, politeness and backoff), `placeholders` (name-match recovery of `-` tickers), `taxcal` (calendar), `yahootopup` (identity check, split un-adjusting, symbol loading). Tests: `tests/research_tsxpanel/` (23, synthetic, no network).

**Panel schema:** `date, etf, ticker, ysym, name, isin, weight, price, shares, sector, source`. `isin` is a null column: none of the BlackRock CA headers checked (2012, 2019, 2024, 2025) has an ISIN. `ysym` (Yahoo symbol) is the identity key; `source` is `parsed` or `placeholder` (recovered `-` rows). Pre-2012-06 files hold two identifier schemes and are de-duplicated by name using the post-switch ticker set, exactly as the tax-loss ingest does.

**Panel size:** 785,863 rows over 1,717 XIC and 1,716 XCS file dates, 2011-02-28 to 2026-02-17 (XIC 405,133 rows / 638 distinct `ysym`; XCS 380,730 rows / 1,001). 128 rows come from placeholder recovery (1,819 `-` rows scanned; the rest stay excluded, as in the tax-loss study).

## 2. Month-end coverage, 2011-2025 (last trading day, or up to 3 business days earlier)

| year | XCS | XIC |
|---|---|---|
| 2011 | 10/11 | 10/11 |
| 2012 | 11/12 | 11/12 |
| 2013 | 12/12 | 12/12 |
| 2014 | 12/12 | 12/12 |
| 2015 | 12/12 | 12/12 |
| 2016 | 12/12 | 12/12 |
| 2017 | 6/12 | 6/12 |
| 2018 | 12/12 | 12/12 |
| 2019 | 12/12 | 12/12 |
| 2020 | 12/12 | 12/12 |
| 2021 | 12/12 | 12/12 |
| 2022 | 12/12 | 12/12 |
| 2023 | 12/12 | 12/12 |
| 2024 | 12/12 | 12/12 |
| 2025 | 12/12 | 12/12 |

Missing month-ends (both ETFs identical): 2011-11 and 2012-01 (Jan 2011 is before the first file), and **2017-01 through 2017-06** (the known BlackRock gap). Everything else, 2013-2016 and 2018-2025, is complete. Files per calendar year:

| year | XCS | XIC |
|---|---|---|
| 2011 | 10 | 10 |
| 2012 | 71 | 71 |
| 2013 | 131 | 131 |
| 2014 | 132 | 132 |
| 2015 | 116 | 116 |
| 2016 | 132 | 132 |
| 2017 | 48 | 49 |
| 2018 | 130 | 130 |
| 2019 | 131 | 131 |
| 2020 | 133 | 133 |
| 2021 | 130 | 130 |
| 2022 | 132 | 132 |
| 2023 | 130 | 130 |
| 2024 | 130 | 130 |
| 2025 | 129 | 129 |
| 2026 | 31 | 31 |

## 3. Daily files around index reviews (2012-2025)

Plan: 56 quarterly reviews, review Friday T = third Friday of Mar/Jun/Sep/Dec, 10 trading days either side (21 dates per review), XIC and XCS: **2,352 requests, 553 already cached, 1 request per 1.1 s with backoff**. Result: **2,201 present, 151 stub (`Fund Holdings as of,"-"`), 0 hard misses (404), 0 blocked**. Every miss was logged at fetch time (`MISS <etf> <date> status=stub`).

| year | present | stub |
|---|---|---|
| 2012 | 112 | 44 |
| 2013 | 170 | 0 |
| 2014 | 170 | 0 |
| 2015 | 162 | 8 |
| 2016 | 170 | 0 |
| 2017 | 79 | 89 |
| 2018 | 160 | 0 |
| 2019 | 170 | 0 |
| 2020 | 172 | 0 |
| 2021 | 166 | 4 |
| 2022 | 168 | 0 |
| 2023 | 166 | 2 |
| 2024 | 162 | 0 |
| 2025 | 166 | 4 |
| 2026 | 8 | 0 |

Stubs by review (both ETFs, dates): 2012-03-16 (40), 2012-09-21 (4), 2014-12-19 (8), 2016-12-16 (4), 2017-03-17 (42) and 2017-06-16 (42, the 2017 gap), 2017-09-15 (1), 2021-06-18 (2), 2021-12-17 (2), 2023-06-16 (2), 2025-06-20 (4). **Daily resolution starts in 2012-06: the 2012-03 review has only stubs, 2012-06 and 2012-12 are usable.** 2012-09 has 4 stub dates inside its window. The 2017 Mar and Jun reviews have almost no files.

## 4. Index-change events

Method: for each ETF, consecutive present files are diffed on `ysym`; a deletion is a name in file a and not in the next file b (bracket: last_in = a, first_out = b). Guards: persistence (absent from the next 2 present files, else transient and dropped), re-entry after a gap is not an addition, and renames are separated by normalised-name match (class `ticker_change`). Classes: deletions `to_xcs` / `to_xic` (appears in the sibling ETF's file within 7 days), `left_both`, `ticker_change`; additions `from_xcs` / `from_xic`, `new`, `ticker_change`. Acquisitions are not inferred (the tax-loss `takeover_sources.csv` holds no rows). Counts by calendar year of the first-out file date:

**XIC deletions**

| year | left_both | ticker_change | to_xcs |
|---|---|---|---|
| 2011 | 13 | 2 | 5 |
| 2012 | 35 | 348 | 14 |
| 2013 | 17 | 5 | 15 |
| 2014 | 21 | 28 | 10 |
| 2015 | 8 | 0 | 22 |
| 2016 | 7 | 5 | 10 |
| 2017 | 11 | 3 | 7 |
| 2018 | 12 | 1 | 9 |
| 2019 | 7 | 2 | 18 |
| 2020 | 15 | 0 | 14 |
| 2021 | 21 | 2 | 4 |
| 2022 | 11 | 2 | 12 |
| 2023 | 12 | 3 | 9 |
| 2024 | 9 | 1 | 7 |
| 2025 | 13 | 0 | 10 |
| 2026 | 1 | 0 | 0 |

**XCS deletions**

| year | left_both | ticker_change | to_xic |
|---|---|---|---|
| 2011 | 28 | 1 | 23 |
| 2012 | 113 | 287 | 15 |
| 2013 | 52 | 14 | 14 |
| 2014 | 48 | 37 | 25 |
| 2015 | 76 | 0 | 7 |
| 2016 | 30 | 4 | 29 |
| 2017 | 43 | 1 | 14 |
| 2018 | 30 | 1 | 7 |
| 2019 | 35 | 2 | 12 |
| 2020 | 50 | 1 | 18 |
| 2021 | 34 | 2 | 30 |
| 2022 | 45 | 2 | 13 |
| 2023 | 42 | 6 | 1 |
| 2024 | 41 | 2 | 7 |
| 2025 | 36 | 2 | 11 |
| 2026 | 3 | 1 | 0 |

**Additions**

| year | XCS from_xic | XCS new | XCS ticker_change | XIC from_xcs | XIC new | XIC ticker_change |
|---|---|---|---|---|---|---|
| 2011 | 9 | 77 | 1 | 16 | 12 | 2 |
| 2012 | 11 | 57 | 178 | 10 | 20 | 211 |
| 2013 | 8 | 47 | 14 | 17 | 14 | 4 |
| 2014 | 6 | 72 | 36 | 18 | 22 | 26 |
| 2015 | 34 | 32 | 0 | 9 | 9 | 0 |
| 2016 | 8 | 31 | 4 | 15 | 11 | 5 |
| 2017 | 9 | 41 | 1 | 6 | 13 | 3 |
| 2018 | 14 | 33 | 1 | 5 | 6 | 1 |
| 2019 | 14 | 31 | 2 | 14 | 5 | 1 |
| 2020 | 15 | 53 | 1 | 9 | 7 | 0 |
| 2021 | 4 | 83 | 2 | 23 | 20 | 1 |
| 2022 | 13 | 45 | 2 | 8 | 11 | 2 |
| 2023 | 43 | 17 | 4 | 4 | 5 | 2 |
| 2024 | 18 | 28 | 2 | 10 | 3 | 1 |
| 2025 | 4 | 32 | 2 | 13 | 4 | 0 |
| 2026 | 0 | 1 | 1 | 0 | 0 | 0 |

Reading the tables:

- **2012 (348 XIC / 287 XCS `ticker_change` deletions) is the identifier-scheme switch of 2012-05-31**, not index activity; the name match classes them correctly, and the brackets that straddle the switch contain **no** events (self-check passed: no spike in the 2012-05 to 2012-06 bracket). 2014 shows a second, smaller burst of `ticker_change` (28 XIC): in some 2014 files BlackRock listed US-style ticker forms for a few days (for example 2014-04-25 and 2014-12-05), again caught by the name match.
- Real XIC removals run at roughly 12-14 `left_both` plus 10-12 `to_xcs` a year after 2012 (2015-2025: 7-21 and 4-22). XCS has about 30-50 `left_both` a year.
- 50 pairs (100 events) carry `rename_candidate` (a `left_both` deletion and a `new` addition in one bracket sharing the first name token: probably a company renamed); they are left in the table and flagged for manual review, not reclassified.

Flag counts (an event can carry several flags):

| etf | direction | f | n |
|---|---|---|---|
| XCS | add | rename_candidate | 31 |
| XCS | add | spans_2017_gap | 6 |
| XCS | add | unconfirmed | 1 |
| XCS | add | wide_bracket | 302 |
| XCS | delete | rename_candidate | 31 |
| XCS | delete | spans_2017_gap | 18 |
| XCS | delete | unconfirmed | 1 |
| XCS | delete | wide_bracket | 436 |
| XIC | add | rename_candidate | 19 |
| XIC | add | spans_2017_gap | 11 |
| XIC | add | wide_bracket | 245 |
| XIC | delete | rename_candidate | 19 |
| XIC | delete | spans_2017_gap | 13 |
| XIC | delete | wide_bracket | 353 |

`wide_bracket` = the bracket spans more than 4 trading days (month-end resolution between the dense windows); `spans_2017_gap` = a wide bracket overlapping Jan-Jun 2017; `unconfirmed` = fewer than 2 later files exist to confirm the event (2 events at the end of the sample).

## 5. When does a review deletion first disappear from the files? (measured)

Every daily-resolution XIC deletion (renames excluded) within 10 trading days of a review Friday T, by `first_out` minus the nearest T in trading days. The bracket does not have to contain T, so fund lags of two days are visible:

| first_out - nearest T (trading days) | 2012-2013 | 2014 | 2015-2025 |
|---|---|---|---|
| -10 | 2 | 0 | 5 |
| -9 | 1 | 0 | 1 |
| -8 | 0 | 0 | 3 |
| -7 | 1 | 0 | 2 |
| -6 | 1 | 0 | 1 |
| -5 | 0 | 0 | 2 |
| -4 | 0 | 1 | 0 |
| -3 | 1 | 1 | 0 |
| -2 | 1 | 0 | 0 |
| -1 | 4 | 0 | 0 |
| 0 | 1 | 7 | 138 |
| 1 | 21 | 3 | 2 |
| 2 | 12 | 0 | 1 |
| 4 | 1 | 0 | 1 |
| 6 | 4 | 1 | 1 |
| 7 | 1 | 1 | 0 |
| 8 | 3 | 1 | 3 |
| 9 | 0 | 3 | 2 |
| 10 | 0 | 1 | 5 |

- **2015-2025: offset 0 dominates (138 of 141 within 2 days).** The file as of the review Friday already shows the post-review membership; the last file that holds a deleted name is the one for T-1. A deleted name has no XIC-holdings price on T.
- **2012-2013: the funds lagged.** 21 events at offset 1 and 12 at offset 2, one at offset 0. 2014 is the transition (7 at offset 0, 3 at offset 1).
- Mass outside 0-2 (offsets of 4 days and more, and negative offsets) is spread thin: those are off-cycle deletions (takeovers, mid-quarter drops) that happen to fall near a review.
- **Accepted offsets: 0, 1, 2 trading days after T (`ACCEPT_OFFSETS` in `events.py`), all eras.** The offset-2 rule admits an occasional unrelated deletion (one `left_both` at offset 2 in the 2022-12 window, absent from the S&P release); in live use the S&P release is the filter.
- Spot checks against the S&P DJI releases (names from the public releases; counts from the files): **2023-03-17** release 2 additions and 3 deletions, files 2 `new` additions and 3 `to_xcs` deletions, all at offset 0. **2022-12-16** release 3 additions (Africa Oil, Pet Valu, StorageVault) and 3 deletions (Canaccord, Corus B, NFI): files show exactly those 3 additions and 3 deletions as `to_xcs`/`from_xcs`/`new`, plus one extra `left_both` at offset 2 that the release does not list. **2024-03-15** release 1 addition (MDA) and 2 deletions (the two Lithium Americas lines): the files show MDA as a `from_xcs` addition and the two lines as `to_xcs` deletions, all at offset 0. **2024-12-20**: the release names 4 additions and the files show 4 `from_xcs` additions at offset 0; whether that release also lists deletions was **not confirmed** (the PDF could not be fetched, HTTP 403), and the files show none.
- Consequence for the specs: the deletion draft enters at the T-1 close, which every deleted name has in the XIC file in every era.

## 6. Off-cycle deletions between reviews

Deletions that do not line up with a review Friday (renames excluded; `events` counts deletions):

| etf | events | certain_off_cycle | narrow_not_review | month_end_resolution |
|---|---|---|---|---|
| XCS | 932 | 330 | 135 | 215 |
| XIC | 379 | 180 | 78 | 116 |

Columns: `certain_off_cycle` = no review date inside the bracket at any resolution; `narrow_not_review` = daily-resolution bracket whose first-out is not at offsets 0-2 after a review date; `month_end_resolution` = bracket wider than 4 trading days.

For XIC `left_both` plus `to_xcs`: of 263 daily-resolution deletions, **185 are review-aligned (offsets 0-2) and 78 are off-cycle** (takeovers, mid-quarter drops, fund-side changes); a further 116 deletions are only visible at month-end granularity, of which **102 have no review date in their bracket and are certainly off-cycle**. The 185 review-aligned XIC deletions fall in **47 distinct reviews, 3.9 names per review**; **44 of those 47 reviews are usable** (the T-1 file and a T+10 file exist; 2014-12-19, 2021-06-18 and 2025-06-20 drop out).

## 7. Exit classes and price coverage after deletion

### 7.1 Exit classes (the tax-loss classes, `rebound.CLASSES`)

`events.csv` carries `exit_class` for deletions: **`other_etf`** (the name moves to the sibling ETF and keeps a holdings price there), **`yahoo`** (a `left_both` name with an identity-checked Yahoo series after deletion), **`unresolved`** (everything else). **`takeover` is 0**: the tax-loss `takeover_sources.csv` has no rows, and acquisitions are not inferred, so acquired names sit inside `unresolved`. Totals over all deletions: 392 `other_etf`, 178 `yahoo`, 741 `unresolved` (renames and additions carry no class).

### 7.2 Review-aligned XIC deletions only (the set the deletion draft would trade)

| year | deleted | other_etf | other_etf_in_xcs_at_T10 | yahoo | unresolved | unresolved_weight_pct | no_price_at_T10_weight_pct |
|---|---|---|---|---|---|---|---|
| 2012 | 18.0 | 14.0 | 13.0 | 0.0 | 4.0 | 85.2 | 85.2 |
| 2013 | 16.0 | 15.0 | 15.0 | 0.0 | 1.0 | 18.2 | 18.2 |
| 2014 | 10.0 | 10.0 | 3.0 | 0.0 | 0.0 | 0.0 | 64.7 |
| 2015 | 23.0 | 22.0 | 22.0 | 0.0 | 1.0 | 4.7 | 4.7 |
| 2016 | 11.0 | 10.0 | 10.0 | 0.0 | 1.0 | 16.7 | 16.7 |
| 2017 | 3.0 | 3.0 | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2018 | 10.0 | 9.0 | 9.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| 2019 | 20.0 | 18.0 | 18.0 | 0.0 | 2.0 | 13.6 | 13.6 |
| 2020 | 18.0 | 14.0 | 14.0 | 2.0 | 2.0 | 29.5 | 29.5 |
| 2021 | 9.0 | 4.0 | 4.0 | 2.0 | 3.0 | 18.4 | 18.4 |
| 2022 | 16.0 | 12.0 | 12.0 | 2.0 | 2.0 | 38.6 | 38.6 |
| 2023 | 11.0 | 9.0 | 9.0 | 1.0 | 1.0 | 4.4 | 4.4 |
| 2024 | 7.0 | 7.0 | 7.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2025 | 13.0 | 10.0 | 10.0 | 1.0 | 2.0 | 27.6 | 27.6 |
| ALL | 185.0 | 157.0 | 149.0 | 9.0 | 19.0 | 19.7 | 22.1 |

Reading: of 185 review-aligned XIC deletions, 157 are `other_etf` (149 verified in the XCS file at T+10 trading days), 9 are `yahoo`, 19 are `unresolved` (10% of names). Weighted by ETF weight at the last holdings file, `unresolved` is 19.7% pooled and names without an exit price at T+10 are 22.1%. The deletion spec's data-limited gate (10% of basket weight) is therefore expected to trip for the all-deletions basket; the `to_xcs`-only arm has an exit price for almost every name. The year columns with few names (2014, 2017) are small-number noise.

### 7.3 All `left_both` deletions (including off-cycle ones)

Targets: confirmed `left_both` deletions from XIC (213 events) and XCS (706). A name counts as covered only if its Yahoo series (a) passes the tax-loss identity check on the pre-deletion overlap (un-adjusted Yahoo close within 2% of the holdings price, series starting on or before the last holdings date) and (b) has at least 20 closes in the 90 calendar days from the first-out date. Sources, in order: the local `yahoo_ohlcv` table (read-only), then yfinance (read-only; cached under `data/yf_cache`; **nothing was written to the database**).

| etf | year | deleted | db | yfinance_only | uncovered | uncovered_pct | uncovered_weight_pct |
|---|---|---|---|---|---|---|---|
| XCS | 2011 | 28.0 | 0.0 | 3.0 | 25.0 | 89.3 | 92.8 |
| XCS | 2012 | 113.0 | 7.0 | 3.0 | 103.0 | 91.2 | 93.2 |
| XCS | 2013 | 52.0 | 4.0 | 1.0 | 47.0 | 90.4 | 97.3 |
| XCS | 2014 | 48.0 | 4.0 | 0.0 | 44.0 | 91.7 | 92.4 |
| XCS | 2015 | 76.0 | 9.0 | 2.0 | 65.0 | 85.5 | 86.1 |
| XCS | 2016 | 30.0 | 4.0 | 0.0 | 26.0 | 86.7 | 93.5 |
| XCS | 2017 | 43.0 | 7.0 | 0.0 | 36.0 | 83.7 | 92.7 |
| XCS | 2018 | 30.0 | 3.0 | 0.0 | 27.0 | 90.0 | 98.3 |
| XCS | 2019 | 35.0 | 9.0 | 1.0 | 25.0 | 71.4 | 88.3 |
| XCS | 2020 | 50.0 | 19.0 | 1.0 | 30.0 | 60.0 | 65.3 |
| XCS | 2021 | 34.0 | 3.0 | 5.0 | 26.0 | 76.5 | 77.6 |
| XCS | 2022 | 45.0 | 9.0 | 7.0 | 29.0 | 64.4 | 75.6 |
| XCS | 2023 | 42.0 | 13.0 | 5.0 | 24.0 | 57.1 | 80.9 |
| XCS | 2024 | 41.0 | 21.0 | 1.0 | 19.0 | 46.3 | 76.2 |
| XCS | 2025 | 36.0 | 11.0 | 2.0 | 23.0 | 63.9 | 84.3 |
| XCS | 2026 | 3.0 | 0.0 | 0.0 | 3.0 | 100.0 | 100.0 |
| XIC | 2011 | 13.0 | 1.0 | 1.0 | 11.0 | 84.6 | 86.2 |
| XIC | 2012 | 35.0 | 4.0 | 0.0 | 31.0 | 88.6 | 77.2 |
| XIC | 2013 | 17.0 | 1.0 | 0.0 | 16.0 | 94.1 | 87.9 |
| XIC | 2014 | 21.0 | 5.0 | 1.0 | 15.0 | 71.4 | 67.7 |
| XIC | 2015 | 8.0 | 0.0 | 0.0 | 8.0 | 100.0 | 100.0 |
| XIC | 2016 | 7.0 | 0.0 | 0.0 | 7.0 | 100.0 | 100.0 |
| XIC | 2017 | 11.0 | 0.0 | 0.0 | 11.0 | 100.0 | 100.0 |
| XIC | 2018 | 12.0 | 1.0 | 0.0 | 11.0 | 91.7 | 99.4 |
| XIC | 2019 | 7.0 | 0.0 | 0.0 | 7.0 | 100.0 | 100.0 |
| XIC | 2020 | 15.0 | 3.0 | 0.0 | 12.0 | 80.0 | 66.4 |
| XIC | 2021 | 21.0 | 2.0 | 0.0 | 19.0 | 90.5 | 93.4 |
| XIC | 2022 | 11.0 | 2.0 | 0.0 | 9.0 | 81.8 | 99.5 |
| XIC | 2023 | 12.0 | 1.0 | 0.0 | 11.0 | 91.7 | 73.3 |
| XIC | 2024 | 9.0 | 1.0 | 0.0 | 8.0 | 88.9 | 100.0 |
| XIC | 2025 | 13.0 | 1.0 | 0.0 | 12.0 | 92.3 | 98.3 |
| XIC | 2026 | 1.0 | 0.0 | 0.0 | 1.0 | 100.0 | 100.0 |
| XCS | ALL | 706.0 | 123.0 | 31.0 | 552.0 | 78.2 | 87.4 |
| XIC | ALL | 213.0 | 22.0 | 2.0 | 189.0 | 88.7 | 86.4 |

- Overall, **XIC `left_both` names are uncovered in 88.7% of cases (86.4% of ETF weight), XCS 78.2% (87.4% of weight)**. This pool is dominated by off-cycle exits (many look like acquisitions or delistings, which the deletion draft would never trade); 7.2 is the number that matters for the deletion study, this table for any strategy that must exit names leaving the indices.
- Of the 741 uncovered targets, **716 have no Yahoo series at all** under the holdings symbol (yfinance returns nothing: delisted, merged or renamed symbols); 15 start after the last holdings date, 7 fail the identity check (ticker reuse or a different issuer), 3 have too little overlap. A random retest of 15 cached "no data" symbols found 0 that now return data, so the misses are stable, not transient.
- The loser-tail blindness found in the tax-loss study repeats for names that leave both indices. Any strategy that needs a post-deletion exit for `left_both` names will be dominated by the `unresolved` class unless the data is bought (EODHD delisted history or Norgate, tax-loss spec 1.5) or the exit comes from the sibling ETF's file (`to_xcs` names).
- Yahoo symbols changed in corporate reorganisations (a renamed issuer keeps its history under the new symbol) are counted as uncovered; a symbol-history map would lift the figure somewhat but not change the picture.

## 8. Announcement-date feasibility (probe only, no scraper built)

Sample: 12 reviews drawn at random from the 44 reviews of 2015-2025 (seed 20261023). Query log: `research/tsxpanel/data/announcement_probe.csv` (query, hit or miss, URL). Method: web search for the S&P Dow Jones Indices release "Announces Changes to the S&P/TSX Composite Index" for each review; plus direct probes of newswire.ca search and the Wayback CDX API.

| Review | Found | Announcement date | Where |
|---|---|---|---|
| 2017-09-15, 2017-12-15, 2018-03-16, 2018-09-21, 2019-09-20 | no | n/a | no S&P/TSX release surfaced |
| 2020-06-19 | yes | 2020-06-12 | spglobal.com `spdji/en/documents/indexnews/announcements/...compositereview-pr.pdf` |
| 2022-12-16 | yes | 2022-12-02 | same pattern |
| 2023-03-17 | yes | 2023-03-03 | same pattern |
| 2024-03-15 | yes | 2024-03-01 | same pattern |
| 2024-09-20 | yes | 2024-09-06 | same pattern |
| 2024-12-20 | yes | 2024-12-06 | same pattern |
| 2025-06-20 | yes | 2025-06-06 | CNW release text surfaced in the results (PDF not separately confirmed) |

**7 of 12 found (58%), all from 2020 on; 0 of 5 for 2017-2019.** Releases come about two weeks before the effective Monday (typically the first or second Friday of the review month). The newswire.ca search endpoint answered HTTP 200 but returned "No results found" for the S&P/TSX queries, and the Wayback CDX API was "Temporarily Offline" on the day, so the pre-2020 misses mean "not found by this probe", not "does not exist". Feasibility verdict: an announcement-date leg could cover the recent third of the sample today; extending it back needs a different archive (CNW archive pages or Wayback snapshots of the S&P DJI announcements index) and is not worth building until the deletion study shows something at the effective date.

## 9. Limits

- Events come from ETF holdings, i.e. the fund's view of the index, not the index. Fund lags exist (offset 1 in 2012-2013).
- Between the dense windows only month-end files exist, so non-review events have month-end brackets.
- The 2017 gap removes the Mar and Jun 2017 reviews and weakens 2017.
- `ticker_change` is name-based; a company that changes name and ticker at once appears as a `left_both` plus a `new` (flagged `rename_candidate`, 50 pairs).
- Price coverage counts symbols as given by the holdings files; symbol histories are not reconstructed.

## 10. Reproduce

```
PYTHONPATH=src python3 research/tsxpanel/fetch_windows.py   # about 35 min; resumable
PYTHONPATH=src python3 research/tsxpanel/panel.py
PYTHONPATH=src python3 research/tsxpanel/run_events.py
PYTHONPATH=src python3 research/tsxpanel/run_coverage.py    # needs DB reader credentials; yfinance cache is resumable; rewrites events.csv with exit_class (rerun after run_events)
PYTHONPATH=src python3 research/tsxpanel/tables.py
```
