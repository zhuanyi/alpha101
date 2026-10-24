# Transcription cross-check (helper report, 2026-10-18)

> Produced by a research sub-agent reading (never executing) three public implementations downloaded to an isolated temp dir. Kept verbatim as evidence; the decisions taken from it are in `docs/operators.md` section 3 and in the `notes` of `src/alpha101/formulas/alphas.yaml`.


Our file: the transcription behind `src/alpha101/formulas/alphas.yaml`. Paper: Kakushadze (2016), text version.
Downloaded files live in /tmp/alpha101_xcheck/ and were only read as text. Nothing was executed or imported.

## Sources actually read

| id | file | URL | version | sha256 |
|---|---|---|---|---|
| S1 | yli188 `101Alpha_code_1.py` (pandas, one stock per DataFrame, 825 lines) | https://raw.githubusercontent.com/yli188/WorldQuant_alpha101_code/master/101Alpha_code_1.py | repo HEAD 3bb9918dd7b62039f41a585a9e37bfd67ce3719f (2019-03-07); blob 664fbb20 | e78cf3b3...ac3e |
| S2 | yli188 `101Alpha_code_2.py` (Quantopian/zipline CustomFactor, 77 alphas, 2591 lines) | https://raw.githubusercontent.com/yli188/WorldQuant_alpha101_code/master/101Alpha_code_2.py | same commit; blob 6091b6d3 | 3fe0f04a...105fd2 |
| S3 | DolphinDB `wq101alpha.dos` (module 1.0.0, header "last modified 2022.04.18"), formulas as `//` comment text plus implementation, all 101 alphas | https://raw.githubusercontent.com/dolphindb/DolphinDBModules/master/wq101alpha/1.0.0/src/wq101alpha.dos | dir last commit f2f2e4bbb2e534c95b502c37e85b8403031070ca (2026-10-12) | 1c2156bf...1ed |

Helper scripts, all in /tmp/a101_scripts/: xc_extract.py (S2 and S3 formula comments), xc_yli1.py (S1 comments), xc_texts.json and xc_yli1_texts.json (extracted texts).

Not found: no independent third source listing formula text was fetched. Searches returned only commentary pages. The official WorldQuant BRAIN docs are not machine-readable here: the learn page returned a 2.3 KB JS shell, the support page returned 403, and api.worldquantbrain.com/operators returned 401.

## 1. Formula-text comparison (S1, S2, S3 comment text vs ours)

Method: strip whitespace, lowercase, and normalise `2.` to `2` and `.001` to `0.001`.

- S1 comments: 101 of 101 identical to ours.
- S3 comments: 101 of 101 identical to ours. The only raw difference is alpha 1, where S3 drops the outer ternary parentheses (`rank(...)-0.5`, `(returns<0?...:close)`). That is semantically identical.
- S2 comments: all 77 alphas present are identical to ours. The 24 absent from S2 are 48, 58, 59, 63, 67, 69, 70, 71, 73, 76, 77, 79, 80, 82, 87, 88, 89, 90, 91, 92, 93, 96, 97, 100. Alphas 71, 73, 77, 88, 92 and 96 are the max/min-of-two-series alphas.

**Caveat on independence.** All three listings are copy-paste of the same paper text. They cannot detect a typo that exists in the paper itself, and they cannot detect an error in the PDF-to-text step if the same artefact was copied. They do confirm that our extraction has no stray tokens, dropped terms, or wrong or missing windows. This includes all the non-integer windows (16.1219, 17.9282 and so on) and the quirks that look suspicious but are in the paper, such as:

- alpha 29's `min(...,5)`
- alpha 66 `(low*0.96633)+(low*(1-0.96633))`
- alpha 59 and 82 reusing the same variable twice in the blend

Verdict: no transcription disagreement found between ours and any source.

## 2. Disagreements in the implementations

These are not formula-text disagreements. They matter if anyone reuses the code as a reference. Each row gives the alpha, their snippet, the classification and the verdict against the paper text.

### S3 (DolphinDB) implementation vs paper formula

| alpha | their code | class | verdict |
|---|---|---|---|
| 14 | `mcovar(open, vol, 10)` where the formula says correlation | their error | The paper says `correlation(open, volume, 10)`. S3 is wrong. |
| 39 | `msum(ratios(close - 1), 250)` | their error | The formula is `sum(returns,250)`. `ratios(close-1)` is the ratio of (close-1) to its previous value. S3 is wrong. |
| 52 | `msum(ret,240) - msum(ret,220)` | their error | The paper has `sum(returns,20)`. S3 typo (220 for 20). |
| 66 | `(low - vwap) \ (open - (high - low) \ 2)` | their error | The paper has `(high + low) / 2`. S3 uses `high - low`. |
| 66 | outer `mrank(..., true, 11)` | their error | The paper has `Ts_Rank(decay_linear(...,11.4157), 6.72611)`, so the window is 6.7 -> 7. S3 reuses the decay window (11). S1 and S2 use 7. |
| 80 | `return pow(rowRank(signdelta), trank)` | their error | The paper multiplies by `* -1`. S3 drops the sign. |
| 81 | `msum(mavg(vol,10), 49)` | implementation inconsistency | The paper has 49.6054. S3 rounds elsewhere (12.7054 -> 13, 16.6208 -> 17, 22.4101 -> 22, 9.91009 -> 10) but truncates here. S1 uses 50. The paper says floor, which gives 49. See section 3 for the rounding conflict. |
| 84 | `mrank(vwap - mmax(vwap,15), true, 20)` | implementation inconsistency | The paper has 20.7127. S3 truncates here but rounds elsewhere; S1 uses 21. The paper says floor, which gives 20. |
| 74 | `rowRank(rowRank(high*w + vwap*(1-w)))` | minor | The paper has a single `rank`. Double rank is monotone, so the result is identical up to ties. Harmless. |
| 56 | `rowRank(tmp1)` without `percent=true` | deliberate or sloppy | Raw 0-based ranks instead of 0..1 percentiles. This changes the scale of the product. Every other alpha uses percent=true. |
| 98 | `9 - mimin(..., 9)` | convention clash | Every other argmax/argmin in S3 (alphas 1, 57, 60, 96, 100) uses raw `mimax/mimin`, which is 0-based from the oldest day. Alpha 98 flips it to days-ago style. See section 3. |
| 31, 39, 57, 58 and every decay_linear alpha | `mavg(x, 1..d)` | UNKNOWN | I could not verify that `mavg` accepts a vector window and gives linear weights. I did not run DolphinDB. If it works as intended, the weights are oldest = 1 ... newest = d (the paper's orientation). If `mavg` treats `1..d` otherwise, every decay_linear alpha in S3 is affected. |
| 1, 57, 60, 96, 98, 100 | `mimax/mimin` | verified from DolphinDB docs | Relative position within the window, 0-based from the oldest day. Docs example: `x = 1.2 2 NULL 6 -1 6; mimax(x,3)` gives `[,,1,2,1,0]`. Ties go to the oldest. |

### S1 (yli188 code_1) implementation vs paper formula

The author notes in the file that alpha 27 "still [has] Some Error". The other rows below are my findings.

| alpha | their code | class | verdict |
|---|---|---|---|
| all | `rank(df) = df.rank(pct=True)` with the `axis=1` version commented out | convention (see section 3) | Each Alphas object holds one stock, so `rank` is a time-series percentile. It is not cross-sectional. All rank-based alphas are therefore not the paper's alphas when run this way. |
| 1 | `inner = self.close; inner[returns<0] = stddev(...)` | their error | Mutates `self.close` in place, which corrupts every later alpha. Also drops `- 0.5`. |
| 10 | no outer `rank(...)` | their error | The paper wraps the ternary in `rank`. |
| 21 | `cond_2 = sma(vol,20)/vol < 1 -> -1`, and the middle branch (`sma2 < sma8 - std8 -> 1`) is lost | their error | The paper gives `(1 < vol/adv20) \|\| (vol/adv20 == 1) -> 1`, else -1. S1 inverts the volume condition. |
| 23 | works through `alpha.at[cond,'close']`, hacky | fragile | The formula is the same. |
| 29 | `ts_min(rank(rank(scale(log(ts_sum(..., 2))))), 5)` | their error | The paper has `ts_min(..., 2)` then `sum(.,1)`. S1 uses `ts_sum` with window 2. The outer `product(.,1)` is dropped. |
| 32 | `scale(sma(close,7)/7 - close)` | their error | `sma` is already the mean, so it divides by 7 twice. The paper has `sum(close,7)/7`. |
| 36 | `sma(close,200)/200` | their error | Same double division. |
| 47 | `sma(high,5)/5` | their error | Same double division. |
| 38 | `ts_rank(open, 10)` | their error | The paper has `Ts_Rank(close,10)`. |
| 39 | `sma(returns,250)` for `sum` | benign | Mean vs sum is rank-invariant. |
| 28, 31, 32, 60 | `scale(df) = df.mul(k).div(np.abs(df).sum())` | convention | Sums over the time axis (column sum), not across stocks. |
| 7, 17, 25, 28, 31, 36 and others | `adv = sma(volume, d)` | convention | Share-volume mean; the paper says dollar volume. See section 3. |
| 64, 68 | `delta(x, 3.69741)`, `delta(x, 1.06157)` with float periods | implementation gap | Windows are not rounded here (S1 rounds them everywhere else). `df.diff(float)` behaviour is UNKNOWN; I did not run it. |

### S2 (yli188 code_2, zipline) implementation vs paper formula

| alpha | their code | class | verdict |
|---|---|---|---|
| 2, 3, 6 and many more (45 `.cov(` call sites in total, some alphas have several) | `pd.DataFrame(a).rolling(w).cov(pd.DataFrame(b))` where the formula says `correlation` | their error | I counted 45 `.cov(` calls and 2 `.corr(` calls in the file. Only alphas 13 and 16 are covariance in the paper. So nearly all correlation alphas use covariance. |
| 1 | `np.std(returns[-20:])` inside the 5-day loop | their error | The stddev window does not slide with each of the 5 days. All five days use the same value. |
| 1 | `stats.rankdata(argmax) - 0.5` | convention | `rankdata` gives raw ranks 1..n, so `- 0.5` is not on a 0..1 scale. |
| 7 | `adv20 = AverageDollarVolume(20)` compared to share `volume` | their error | Unit mismatch. |

## 3. Operator conventions per source

| convention | paper text (Appendix A) | S1 (yli188 code_1) | S2 (yli188 code_2) | S3 (DolphinDB) |
|---|---|---|---|---|
| cross-sectional `rank` | "cross-sectional rank", range unspecified | `df.rank(pct=True)`: percentile r/n in (0,1], average ties, but along the time axis for a single column (cross-sectional version commented out) | `stats.rankdata`: raw ranks 1..n, average ties, not normalised | `rowRank(percent=true)`: percent form. DolphinDB's exact formula ((r-1)/(n-1), r/n, or something else) UNKNOWN. |
| `ts_rank` output | "time-series rank in the past d days" | raw 1..d, `rankdata(window)[-1]`, average ties | raw 1..d, `DataFrame.rank().tail(1)`, average ties | `mrank(x, true, d)`: raw, 0-based 0..d-1 (percent defaults to false; verified from docs.dolphindb.cn mrank page) |
| `ts_argmax/argmin` base | "which day ts_max occurred on" | `np.argmax + 1`: 1 = oldest, d = today | `np.argmax`: 0 = oldest, d-1 = today | `mimax/mimin`: 0 = oldest, d-1 = today. Exception: alpha 98 uses `9 - mimin`, which flips it. |
| `decay_linear` | weights d, d-1, ..., 1, rescaled to 1 (newest = d) | weights 1..d / (d(d+1)/2) applied oldest-to-newest, so newest = d. First `period` rows copy raw values. NaNs are ffilled, bfilled, then zero-filled. | `arange(1,d+1)/sum` applied oldest-to-newest, newest = d | `mavg(x, 1..d)`: UNKNOWN (see section 2) |
| stddev ddof | unspecified | pandas `rolling.std`: ddof=1 | `np.std`: ddof=0 | `mstd`: ddof=1 per DolphinDB convention; I did not verify against docs, UNKNOWN |
| `adv{d}` units | **average daily dollar volume** | `sma(volume, d)`: share volume (`volume = S_DQ_VOLUME*100`) | `AverageDollarVolume(d)`: dollars, but alpha 7 compares it to share volume | `mavg(vol, d)`: share volume |
| non-integer windows | **floor(d)** | rounds (manual ints: 16.1219 -> 16, 17.9282 -> 18, 49.6054 -> 50, 20.7127 -> 21) | rounds (e.g. 13 for 12.7054) | rounds, except alphas 81 (49.6054 -> 49) and 84 (20.7127 -> 20) |
| inf / NaN | unspecified | correlation results: `replace([-inf, inf], 0).fillna(0)` in about a dozen alphas; ratios in alphas 34 and 38 use 1; zero denominators in 53, 54, 55 and 60 replaced with +-0.0001 | not handled | not handled |
| vwap proxy | "daily volume-weighted average price" | `amount*1000/(volume*100+1)` (Wind turnover/volume); `volume` multiplied by 100 | zipline `VWAP(window_length=2)` | `vwap` is a function argument; no proxy |
| `returns` | "daily close-to-close returns" | `S_DQ_PCTCHANGE` (Wind percentage change field) | zipline `Returns(window_length=2)` | `ratios(close) - 1` (simple return, unadjusted) |
| `min/max` with two args (29 vs 71/77/88/92/96) | `min(x,d) = ts_min(x,d)` and `max(x,d) = ts_max(x,d)`. Two-series use in 71/73/77/88/92/96 is outside the definition. | alpha 29: the outer `min(...,5)` is implemented as `ts_min(.., 5)`, matching the paper definition. 71/73/77/88/92/96 are element-wise via a p1/p2 DataFrame hack with `>=`. | alpha 29 present; 71/73/77/88/92/96 absent | alpha 29: `mmin(..., 5)`, matching the paper. 71/73/77/88/92/96: element-wise `max(a,b)` / `min(a,b)` |
| `IndNeutralize` with no industry data | defined only with IndClass | the 19 industry alphas (48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100) are omitted, others commented as unimplemented | the 19 industry alphas are omitted except alpha 56 (present in S2; its cap handling not inspected) | takes `indclass` as a parameter and uses `contextby demean` (or `avg` in 79, 80, 97). The comment says "levels of classification are ignored". Alpha 56 needs `cap`. |

Reading of the alpha 29 vs 71/77 question: the paper's own definition of `min(x, d)` as `ts_min(x, d)` supports reading alpha 29 as `ts_min(product(...,1), 5)`. All three sources agree on that. In alphas 71, 73, 77, 88, 92 and 96, the second argument is a full expression, not a window, so only the element-wise reading is possible. All sources that implement them agree.

## 4. Official WorldQuant BRAIN / WebSim definitions

**Could not verify from an official source.** The BRAIN documentation requires JS or login:

- https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions returns a 2.3 KB JS shell.
- The support.worldquantbrain.com operator list returns HTTP 403.
- https://api.worldquantbrain.com/operators returns HTTP 401.

Third-party statements found. These are not authoritative for BRAIN:

- https://coriva.eu.org/en/alpha101-overview/ (a commentary page about the paper, not BRAIN docs):
  - rank: "normalize to the 0-1 range. The stock with the lowest value gets 0, the highest gets 1."
  - ts_argmax: "Returns a 0-based index: 0 = the oldest day in the window, d-1 = today."
  - decay_linear: "most recent day gets the highest weight."
  - adv{d}: "d-day average daily dollar volume".
- https://www.scribd.com/document/964633851/WorldQuant-BRAIN (BRAIN operators Q&A): `ts_decay_linear(x, d, dense=false)` applies linear decay over the past d days, with `dense=false` treating NaN as 0. I did not open the document, only read the search snippet.
- https://polars-ta.readthedocs.io/en/latest/wq/time_series/: a polars library; per the search snippet its argmax counts back from today (most recent = 0). I did not open the page.

| item | status |
|---|---|
| BRAIN `adv20` (shares or dollars) | UNKNOWN. No source found that defines the BRAIN data field. The paper says dollars. |
| BRAIN `ts_rank` (raw or normalised) | UNKNOWN for BRAIN. A BRAIN overview snippet lists the signature `ts_rank(x, d, constant=0)`, but the description in that snippet belongs to ts_quantile. |
| BRAIN `ts_arg_max` (0 = today?) | UNKNOWN for BRAIN. Third-party libraries disagree: coriva says 0 = oldest; polars-ta says 0 = today. |
| BRAIN `rank` range | UNKNOWN for BRAIN. Third-party says 0..1. |
| BRAIN `decay_linear` | UNKNOWN for the exact weights; the paper's definition is weights d..1 with the newest weighted highest. |

## 5. Bottom line

- Our transcription is consistent with all three listings: 101 of 101 against S1 and S3, and 77 of 77 available against S2. The listings share the paper as their source, so they cannot expose paper typos.
- The implementations are unreliable as references. Known bugs: S3 alphas 14, 39, 52, 66 (two bugs) and 80; S1 alphas 1, 10, 21, 29, 32, 36, 38 and 47; S2's use of `.cov` for correlation (45 call sites, at most 2 legitimate) and its dollar-vs-share adv in alpha 7.
- The paper says floor(d) for non-integer windows, but every implementation rounds to nearest. This is a decision we have to make. Rounding is the common practice.
- `adv{d}`: the paper says dollar volume. S1 and S3 use share volume. S2 uses dollars but mixes units in alpha 7.
- Not verified: DolphinDB `mavg(x, 1..d)` semantics, `rowRank(percent=true)` formula, `mstd` ddof, and `pandas diff(float)`. Every official BRAIN definition is UNKNOWN.
