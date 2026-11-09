#!/usr/bin/env python3
"""
Phase 1 -- conditional overnight premium, Stage A (discovery).

Spec: pre-registered conditional-overnight spec (not distributed).
Imports from the verbatim Phase 0 script (overnight_intraday_test.py, same directory), never modifies it.

DEFINITIONS (spec sections 1-4, frozen)
  Night t = close of trading day t-1 -> open of trading day t, labelled by the date of the open.
  on_t = overnight_div_t (dividends accrue to the overnight holder); px_t = overnight_px_t; tot_t = total_t.
  Conditions (all computed ONLY in condition_flags(); data dated <= t-1, VIX <= t-2, plus calendars):
    H1  DOW         open of t on Tuesday or Wednesday            (falsified on D2 2010-2018 only)
    H2  Weekend     open of t on Monday or first open after a >=3 calendar day gap (control group: weekday nights)
    H3  Pre-FOMC    t = date of a SCHEDULED FOMC statement (statement after the open)
    H4  CPI night   08:30 ET CPI release falls inside night t (t = first open after the release timestamp)
    H5  NFP night   08:30 ET Employment Situation release falls inside night t
    H6  Turn-month  open of t is trading day -1, +1, +2 or +3 of the month (the symbol's own calendar)
    H7  Post-down   total_{t-1} <= -1.5 * sigma20, sigma20 = std(total over t-21..t-2)   ("near-real-time")
    H8  High-VIX    ^VIX close of day t-2 >= 20 (asof, backward only)
    X2  Attention   volume_{t-1} / mean(volume_{t-21..t-2}) >= 2                          (exploratory)
    X3  Calendar    H3 | H4 | H5 | H6                                                      (exploratory)
    X4a H3 nights, return = close t-1 -> close t (total_t)                                 (exploratory)
    X1 (pre-BoC) and X4b (13:55 ET bar) are NOT run in this phase.
  Tests per condition x unit x era: T-a OLS with Newey-West HAC (weekday controls unless H1/H2),
  T-b weekday-stratified permutation (10,000 draws; unstratified for H1/H2 because stratifying by weekday
  would pin the null for a weekday condition), T-c paired d_t = C_t*(on_t - cost_t(E)) - f*tot_t at E=$100k,
  plus power (SE, minimum detectable mean = 2*SE, years needed = (1.96/IR)^2).
  Units: single instruments SPY QQQ IWM XIU.TO XIC.TO, and equal-weight panels US={SPY,QQQ,IWM}, CA={XIU,XIC}
  (a panel night needs >= half of its members; panel C = at least half of the present members flagged).
  Eras: D1 first date -> 2009-12-31, D2 2010-01-01 -> 2018-12-31, HO 2019-01-01 -> last (LOCKED).

HOLDOUT LOCK: rows >= 2019-01-01 are dropped immediately after loading unless --unlock-holdout, which in turn
  refuses to run without --frozen <file>. The only use of post-2018 rows in Stage A is the weekday table
  (contaminated_holdout_weekday.csv) that Phase 0 already published. Stage B is NOT implemented here.

DECISION RULES (spec section 6, operationalised before any data was seen)
  ADVANCE: in D2 (and in D1 where the unit has >= 30 flagged nights) b > 0, t_b >= z* (H1: t_b >= 2, D2 only),
    permutation p < 0.05, T-c paired mean > 0 at E=$100k (1x, USD side held, idle cash, TFSA/RRSP tax) and
    round trips/yr <= 60.
  Otherwise REJECT if b <= 0 in D2, or the D2 95% CI upper bound of b is below the round-trip cost in bp at
    E=$100k, or round trips/yr > 60 (cannot meet the implementable-turnover rule whatever the statistics).
  Otherwise UNDERPOWERED (b > 0 but the interval still contains an economically relevant effect).
  Only the 16 hypothesis x panel rows (H1-H8 x US/CA) are decision-grade; the rest are exploratory.

COST SENSITIVITIES (spec section 3.2; --cost-sens). Decisions fixed BEFORE any sensitivity result was seen:
  Scenarios (one change at a time, everything else at the COSTS defaults):
    commission_7   commission $7.00/side (high-volume tier) instead of $9.99
    spread_x2      every half_spread_bp x 2 (spec: "x2")
    timing_0bp     timing_slippage_bp_per_side = 0 for both markets
    timing_3bp     timing_slippage_bp_per_side = 3 for both markets
  Reading of the timing rows: the spec 3.2 table has no timing row (it comes from "Brokerage constraints": model a
  timing-slippage haircut), so the literal reading is the code comment's "0 and 3 bp per side": ABSOLUTE values, same for
  US and TSX (defaults are 1.0 US / 2.0 TSX, so 3 bp is x3 for the US and x1.5 for the TSX). The spec 3.2 open_slippage
  sensitivities (0 and 3/6 bp), FX, financing and exposure variants are NOT part of this scenario set.
  Re-evaluated per scenario: the paired T-c series d_t (cost enters only there and in the reject-on-cost rule), then the same
  decide(). t_b, permutation p, n_eff and z* do not depend on costs and are reused from the default run (z* is NOT recomputed).
  The 'default' scenario is re-derived through the same code path and must equal the main decision table or the run aborts.

MULTIPLE TESTING: n_eff = (sum eig)^2 / sum eig^2 of the correlation matrix of the daily T-c series d_t of all
  reported cells in D1+D2; z* = Phi^-1(1 - p/2), p = 1 - 0.95^(1/n_eff) (Sidak).

Usage (see --help for every flag):
  --build-calendar [--write-db]   FOMC pages + FRED -> event_calendar_check.csv, data/event_calendar.csv
  --fetch-new [--write-db]        SSO UPRO TQQQ QLD ^VIX CAD=X -> raw_yahoo/ cache, yahoo_ohlcv
  --from-db [--write-db]          Stage A run -> CSVs/PNGs in --out-dir
  --write-report                  results markdown generated from the CSVs
"""
import argparse
import datetime as dt
import glob
import hashlib
import copy
import json
import math
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from overnight_intraday_test import (DEFAULTS, build_returns, fetch_yahoo, load_env,  # noqa: E402
                                     newey_west_t, read_yahoo, reader_connection, stats,
                                     write_yahoo, writer_connection)

SCRIPT_PATH = "research/overnight/overnight_conditional_test.py"
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_OUT = os.environ.get("OVERNIGHT_OUT", "phase1_conditional_overnight_results")
DEFAULT_PHASE0 = os.environ.get("OVERNIGHT_PHASE0_DIR", "overnight_intraday_results")

HO_START = pd.Timestamp("2019-01-01")
ERAS = {"D1": (pd.Timestamp("1900-01-01"), pd.Timestamp("2009-12-31")),
        "D2": (pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31")),
        "D1D2": (pd.Timestamp("1900-01-01"), pd.Timestamp("2018-12-31"))}
XIC_FIRST_VALID = pd.Timestamp("2010-01-01")  # Phase 0 data-quality note: XIC opens before 2010 unusable

HYP_SYMBOLS = ["SPY", "QQQ", "IWM", "XIU.TO", "XIC.TO"]
PANELS = {"US": ["SPY", "QQQ", "IWM"], "CA": ["XIU.TO", "XIC.TO"]}
NEW_SYMBOLS = ["SSO", "UPRO", "TQQQ", "QLD", "^VIX", "CAD=X"]
# symbol: (underlying, leverage L, group). Hypothesis instruments have L=1.
VEHICLES = {"SPY": ("SPY", 1, "us"), "QQQ": ("QQQ", 1, "us"), "IWM": ("IWM", 1, "us"),
            "XIU.TO": ("XIU.TO", 1, "ca"), "XIC.TO": ("XIC.TO", 1, "ca"),
            "SSO": ("SPY", 2, "us"), "UPRO": ("SPY", 3, "us"), "TQQQ": ("QQQ", 3, "us"), "QLD": ("QQQ", 2, "us")}
LOAD_SYMBOLS = sorted(set(VEHICLES) | {"^VIX"})

HYPOTHESES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]
EXPLORATORY = ["X2", "X3", "X4a"]
WEEKDAY_CONDS = ("H1", "H2")  # no weekday controls / no weekday stratification
HYP_TEXT = {
    "H1": "H1 DOW: E[overnight | night ends at a Tuesday or Wednesday open] > E[overnight | other nights]; falsified on D2 2010-2018 if b <= 0 or t_b < 2.",
    "H2": "H2 Weekend (Elm): E[overnight | weekend night] > E[overnight | weekday nights]; falsified if b <= 0 or t_b < 2 (significantly negative b reported as 'Elm reversed').",
    "H3": "H3 Pre-FOMC: E[overnight | night before a scheduled FOMC statement] > other nights.",
    "H4": "H4 CPI night: E[overnight | US CPI 08:30 ET release falls inside the night] > other nights.",
    "H5": "H5 NFP night: E[overnight | Employment Situation 08:30 ET release falls inside the night] > other nights.",
    "H6": "H6 Turn-of-month: E[overnight | open on trading day -1,+1,+2,+3 of the month] > other nights.",
    "H7": "H7 Post-down-day: E[overnight | previous day total return <= -1.5 sigma20] > other nights (near-real-time condition).",
    "H8": "H8 High-VIX: E[overnight | ^VIX close of t-2 >= 20] > other nights.",
}
SUCCESS_TEXT = ("Advance to Stage B if in D2 (and D1 where available) b>0 with t_b >= z* (H1: t_b>=2 on D2 alone), permutation p<0.05, "
                "T-c paired mean > 0 at E=$100k (1x, USD side held, idle cash, TFSA/RRSP tax) and round trips/yr <= 60.")

# ----------------------------------------------------------------------------
# Costs / tax (assumptions; every number is overridable here, none is hard-coded elsewhere)
# ----------------------------------------------------------------------------
COSTS = dict(
    commission=DEFAULTS["commission_per_order"],                 # $9.99 standard
    commission_active=DEFAULTS["commission_active_per_order"],   # $7.00 high-volume tier (sensitivity)
    half_spread_bp={"SPY": DEFAULTS["half_spread_bps_etf"], "QQQ": DEFAULTS["half_spread_bps_etf"],
                    "IWM": DEFAULTS["half_spread_bps_etf"], "XIU.TO": DEFAULTS["half_spread_bps_ca_etf"],
                    "XIC.TO": DEFAULTS["half_spread_bps_ca_etf"],
                    "SSO": 2.0, "UPRO": 2.0, "TQQQ": 1.0, "QLD": 2.0},   # SSO/UPRO/TQQQ per spec; QLD assumed = SSO
    open_slippage_bp={"us": 1.0, "ca": 3.0},              # one side: the sale at the open (assumption)
    # Brokerage constraint (the retail broker does not guarantee MOC/MOO): timing-slippage haircut per side (entry near the
    # close, exit near the open). Assumption, not measured; sensitivities 0 and 3 bp per side.
    timing_slippage_bp_per_side={"us": 1.0, "ca": 2.0},
    fx_tiers=[(25_000, 0.0126), (100_000, 0.0078), (250_000, 0.0046), (1_000_000, 0.0027), (float("inf"), 0.0019)],
    fx_surcharge=4.0 / 3.0,                               # spread + one third of spread
    margin_rate={"us": 0.1999, "ca": 0.2299}, settlement_days=1,
    exposures=[25_000, 50_000, 100_000, 250_000],
    decision_exposure=100_000,
)
# Spec 3.2 sensitivities, one change at a time (see the module docstring for the reading of the timing rows).
SCENARIOS = {
    "default": {},
    "commission_7": {"commission": COSTS["commission_active"]},
    "spread_x2": {"spread_mult": 2.0},
    "timing_0bp": {"timing_bp": 0.0},
    "timing_3bp": {"timing_bp": 3.0},
}
TAX = dict(cap_gains=0.2676, income=0.5353, us_div_margin=0.5353, ca_div_margin=0.3934, us_withholding=0.15)
ACCOUNTS = ["TFSA", "TFSA_business_downside", "RRSP", "MARGIN_capgains", "MARGIN_business"]
MAX_TRIPS = 60
CRA_TRIPS = 50
ANN = 252

# ----------------------------------------------------------------------------
# Calendar builder
# ----------------------------------------------------------------------------
FED = "https://www.federalreserve.gov"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
SHUTDOWN_WINDOWS = [("2013-10-17", "2013-11-10"), ("2025-10-15", "2025-12-31")]
CAL_COLS = ["event_type", "event_date", "event_time_et", "scheduled", "has_sep", "reference_period",
            "anomaly", "source", "source_url", "fetched_at"]

EVENT_DDL = """
CREATE TABLE IF NOT EXISTS event_calendar (
  event_type      VARCHAR(30)  NOT NULL,
  event_date      DATE         NOT NULL,
  event_time_et   TIME         NULL,
  scheduled       TINYINT(1)   NOT NULL DEFAULT 1,
  has_sep         TINYINT(1)   NULL,
  reference_period VARCHAR(20) NULL,
  anomaly         VARCHAR(100) NULL,
  source          VARCHAR(30)  NOT NULL,
  source_url      VARCHAR(500) NULL,
  fetched_at      DATETIME     NOT NULL,
  PRIMARY KEY (event_type, event_date),
  KEY idx_date (event_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def mask_secret(text, secret):
    return text.replace(secret, "****") if secret else text


def _get(url, params=None, secret=None, tries=3):
    import requests
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = type(e).__name__
    raise RuntimeError(f"GET {mask_secret(url, secret)} failed: {last}")


def _month_num(name):
    return dt.datetime.strptime(name[:3], "%b").month


def parse_fomc_historical(html, source_page):
    """Parse a fomchistoricalYYYY.htm page -> rows (event_date, scheduled, has_sep, statement_url, title)."""
    heads = list(re.finditer(r"<h5[^>]*>\s*([^<]+?)\s*</h5>", html))
    rows = []
    pat = re.compile(r"(?P<m1>[A-Za-z]+(?:/[A-Za-z]+)?)\s+(?P<d1>\d+)(?:-(?:(?P<m2>[A-Za-z]+)\s+)?(?P<d2>\d+))?\s*"
                     r"(?P<flag>\([a-z]+\))?\s*(?P<kind>Meeting|Conference Call)\s*-\s*(?P<y>\d{4})")
    for i, m in enumerate(heads):
        mm = pat.match(m.group(1))
        if not mm:
            continue
        seg = html[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(html)]
        flag = (mm.group("flag") or "").strip("()")
        if flag == "cancelled":
            continue
        month = _month_num((mm.group("m2") or mm.group("m1")).split("/")[-1])
        day = int(mm.group("d2") or mm.group("d1"))
        date = dt.date(int(mm.group("y")), month, day)
        link = re.search(r'<a[^>]*href="([^"]+)"[^>]*>\s*Statement\s*</a>', seg)
        url = None
        if link:
            url = link.group(1)
            url = url if url.startswith("http") else FED + url
        rows.append(dict(event_date=date, scheduled=int(mm.group("kind") == "Meeting" and flag != "unscheduled"),
                         has_sep=int(("Projection" in seg) or ("SEP" in seg)), source_url=url,
                         title=m.group(1), source_page=source_page))
    return rows


def parse_fomc_current(html):
    """Parse fomccalendars.htm (2021+): one row per meeting with a statement link."""
    rows = []
    years = list(re.finditer(r"(\d{4}) FOMC Meetings", html))
    for i, ym in enumerate(years):
        seg = html[ym.end(): years[i + 1].start() if i + 1 < len(years) else len(html)]
        year = int(ym.group(1))
        parts = re.split(r'<div class="(?:fomc-meeting--shaded )?row fomc-meeting"', seg)[1:]
        for p in parts:
            mon = re.search(r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>", p)
            dat = re.search(r"fomc-meeting__date[^>]*>([^<]+)<", p)
            if not (mon and dat):
                continue
            months = [x for x in mon.group(1).strip().split("/")]
            dtxt = dat.group(1).strip()
            days = re.findall(r"\d+", dtxt)
            month = _month_num(months[-1])
            day = int(days[-1])
            link = re.search(r'href="([^"]+monetary(\d{8})a\.htm)"', p[p.find("Statement:"):] if "Statement:" in p else "")
            if not link:
                continue  # future meeting, no statement yet
            sdate = dt.datetime.strptime(link.group(2), "%Y%m%d").date()
            rows.append(dict(event_date=sdate, scheduled=1, has_sep=int("*" in dtxt),
                             source_url=FED + link.group(1), title=f"{months[-1]} {dtxt} {year}",
                             source_page="fomccalendars"))
    return rows


def build_fomc(log):
    rows = []
    for y in range(1994, 2021):
        r = _get(f"{FED}/monetarypolicy/fomchistorical{y}.htm")
        rows += parse_fomc_historical(r.text, f"fomchistorical{y}")
    rows += parse_fomc_current(_get(f"{FED}/monetarypolicy/fomccalendars.htm").text)
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df["source_url"].notna()].copy()           # meetings without a statement (pre-1999 holds) are not events
    df["anomaly"] = None
    for i, r in df.iterrows():
        m = re.findall(r"(?<!\d)((?:19|20)\d{6})(?!\d)", r["source_url"])
        if m:
            sd = dt.datetime.strptime(m[0], "%Y%m%d").date()
            if sd != r["event_date"]:
                df.at[i, "anomaly"] = f"statement_date_mismatch:page={r['event_date']},link={sd}"
                if abs((sd - r["event_date"]).days) <= 1:   # a link >1 day off is a Fed page error: keep the page date
                    df.at[i, "event_date"] = sd
    df = df[df["event_date"] >= dt.date(1994, 2, 1)].drop_duplicates("event_date").sort_values("event_date").copy()
    # Before 1999-05-18 a statement was issued only when policy changed, so its existence was not knowable at entry.
    early = df["event_date"] < dt.date(1999, 5, 18)
    df.loc[early, "scheduled"] = 0
    df.loc[early, "anomaly"] = [("statement_only_on_policy_change" if pd.isna(a) else a + ";statement_only_on_policy_change")
                                for a in df.loc[early, "anomaly"]]
    log(f"FOMC: {n_all} meetings parsed, {len(df)} with a statement link, {int(df.scheduled.sum())} scheduled")
    out = pd.DataFrame(dict(
        event_type="FOMC_STATEMENT", event_date=df["event_date"].values,
        event_time_et=["14:15:00" if d < dt.date(2013, 1, 1) else "14:00:00" for d in df["event_date"]],
        scheduled=df["scheduled"].values, has_sep=df["has_sep"].values, reference_period=None,
        anomaly=df["anomaly"].values, source="fed_web", source_url=df["source_url"].values))
    return out


def build_fred_events(env, log, raw_dir):
    """CPI (release 10) and NFP (release 50). Headline date = first ALFRED vintage of CPIAUCNS / PAYEMS per
    reference month (bls.gov returns HTTP 403 to scripts); other FRED release dates are tagged duplicates."""
    key = env.get("FRED_API_KEY") or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set in the environment")
    os.makedirs(raw_dir, exist_ok=True)
    base = "https://api.stlouisfed.org/fred"
    out = []
    spec = [("US_CPI", 10, "CPIAUCNS"), ("US_NFP", 50, "PAYEMS")]
    for etype, rid, series in spec:
        rel = _get(f"{base}/release/dates", dict(release_id=rid, include_release_dates_with_no_data="true",
                   realtime_start="1900-01-01", file_type="json", limit=10000, api_key=key), key).json()
        obs = _get(f"{base}/series/observations", dict(series_id=series, realtime_start="1990-01-01",
                   realtime_end="9999-12-31", file_type="json", limit=100000, api_key=key), key).json()
        for nm, payload in (("release_%d" % rid, rel), ("vintages_%s" % series, obs)):
            with open(os.path.join(raw_dir, f"fred_{nm}.json"), "w") as f:
                json.dump(payload, f, sort_keys=True)
        rdates = pd.to_datetime([x["date"] for x in rel["release_dates"]])
        o = pd.DataFrame(obs["observations"])
        o = o[o["value"] != "."]
        first = o.groupby("date")["realtime_start"].min()
        first = first[first.index >= "1992-10-01"]
        chosen = {}
        for ref, rs in first.items():
            chosen.setdefault(pd.Timestamp(rs), []).append(ref[:7])
        last_vintage = max(chosen)
        for d in rdates[(rdates >= "1993-01-01")]:
            if d in chosen:
                ref = "/".join(chosen[d])
                dup = 0
            elif d > last_vintage:
                ref, dup = None, 0       # future scheduled date, not yet released
            else:
                ref, dup = None, 1
            anomalies = []
            if dup:
                anomalies.append("duplicate_release_date")
            if any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in SHUTDOWN_WINDOWS):
                anomalies.append("shutdown_delay")
            out.append(dict(event_type=etype, event_date=d.date(), event_time_et="08:30:00", scheduled=int(not dup),
                            has_sep=None, reference_period=ref, anomaly=";".join(anomalies) or None, source="fred_api",
                            source_url=f"{base}/release/dates?release_id={rid} (headline date: first ALFRED vintage of {series})"))
        log(f"{etype}: {sum(1 for r in out if r['event_type'] == etype)} dates "
            f"({sum(1 for r in out if r['event_type'] == etype and not r['scheduled'])} duplicates)")
    return pd.DataFrame(out)


def calendar_checks(cal, us_days, out_dir, log, seed=20261018, check_urls=True):
    """Holiday-release tags, per-year counts, anomaly list, FOMC URL spot check."""
    cal = cal.copy()
    ed = pd.to_datetime(cal["event_date"])
    last = us_days.max()
    hol = (~ed.isin(us_days)) & (ed <= last) & (cal["event_type"] != "FOMC_STATEMENT")
    cal.loc[hol, "anomaly"] = [("holiday_release" if pd.isna(a) else a + ";holiday_release") for a in cal.loc[hol, "anomaly"]]
    cal["year"] = ed.dt.year
    cnt = (cal[cal.scheduled == 1].pivot_table(index="year", columns="event_type", values="event_date", aggfunc="count")
           .fillna(0).astype(int))
    cnt.to_csv(os.path.join(out_dir, "event_calendar_check.csv"))
    exc = []
    for y in range(1994, 2026):
        n = int(cnt.loc[y, "FOMC_STATEMENT"]) if y in cnt.index and "FOMC_STATEMENT" in cnt else 0
        if n != 8:
            exc.append(f"FOMC {y}: {n} scheduled statements")
    for et in ("US_CPI", "US_NFP"):
        for y in range(1993, 2026):
            n = int(cnt.loc[y, et]) if y in cnt.index and et in cnt else 0
            if n != 12:
                exc.append(f"{et} {y}: {n} scheduled releases")
    an = cal[cal["anomaly"].notna()][["event_type", "event_date", "scheduled", "anomaly"]]
    an.to_csv(os.path.join(out_dir, "event_calendar_anomalies.csv"), index=False)
    with open(os.path.join(out_dir, "event_calendar_exceptions.txt"), "w") as f:
        f.write("\n".join(exc) + "\n")
    log(f"calendar check: {len(exc)} count exceptions, {len(an)} anomaly rows")
    if check_urls:
        fomc = cal[(cal.event_type == "FOMC_STATEMENT") & cal.source_url.str.startswith("https://")].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        picks = sorted(rng.choice(len(fomc), 3, replace=False))
        rows = []
        for i in picks:
            u = fomc.loc[i, "source_url"]
            try:
                code = _get(u).status_code
            except RuntimeError as e:
                code = str(e)[-40:]
            rows.append(dict(event_date=fomc.loc[i, "event_date"], url=u, status=code))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fomc_url_check.csv"), index=False)
        log("FOMC url spot check: " + "; ".join(f"{r['event_date']} -> {r['status']}" for r in rows))
    return cal.drop(columns="year")


def write_calendar_db(conn, cal, fetched_at, log):
    cur = conn.cursor()
    cur.execute(EVENT_DDL)
    sql = ("INSERT INTO event_calendar (event_type,event_date,event_time_et,scheduled,has_sep,reference_period,anomaly,source,source_url,fetched_at)"
           " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
           " ON DUPLICATE KEY UPDATE event_time_et=VALUES(event_time_et),scheduled=VALUES(scheduled),has_sep=VALUES(has_sep),"
           "reference_period=VALUES(reference_period),anomaly=VALUES(anomaly),source=VALUES(source),"
           "source_url=VALUES(source_url),fetched_at=VALUES(fetched_at)")

    def nn(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
    rows = [(r.event_type, r.event_date, nn(r.event_time_et), int(r.scheduled), None if nn(r.has_sep) is None else int(r.has_sep),
             nn(r.reference_period), nn(r.anomaly), r.source, (nn(r.source_url) or "")[:500], fetched_at) for r in cal.itertuples()]
    cur.executemany(sql, rows)
    conn.commit()
    log(f"event_calendar upsert: {len(rows)} rows")
    return len(rows)


def load_calendar_events(args, conn=None):
    if args.calendar_csv:
        df = pd.read_csv(args.calendar_csv, parse_dates=["event_date"])
    else:
        cur = conn.cursor()
        cur.execute("SELECT event_type,event_date,scheduled FROM event_calendar")
        df = pd.DataFrame(cur.fetchall(), columns=["event_type", "event_date", "scheduled"])
        df["event_date"] = pd.to_datetime(df["event_date"])
    return df[["event_type", "event_date", "scheduled"]]


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------

def clean_frames(frames):
    """Per-symbol hygiene that is not an era choice: XIC.TO opens before 2010-01-01 are unusable (Phase 0)."""
    out = dict(frames)
    if "XIC.TO" in out:
        out["XIC.TO"] = out["XIC.TO"][out["XIC.TO"].index >= XIC_FIRST_VALID]
    return out


def apply_era_mask(frames, unlock=False):
    """Holdout lock: drop every row >= 2019-01-01 right after loading."""
    if unlock:
        return dict(frames)
    return {s: df[df.index < HO_START] for s, df in frames.items()}


def trading_days(df):
    d = df[(df["open"] > 0) & (df["close"] > 0) & (df["adj_close"] > 0)]
    return pd.DatetimeIndex(sorted(set(d.index)))


def contaminated_weekday_table(frames_full, symbols=HYP_SYMBOLS):
    """2019+ weekday table (Phase 0 published it already). Same logic as Phase 0; the only use of post-2018 rows."""
    rows = []
    for s in symbols:
        if s not in frames_full:
            continue
        r = build_returns(frames_full[s])
        sub = r[r.index >= HO_START]
        if len(sub) < 200:
            continue
        for d in range(5):
            x = sub[sub.index.dayofweek == d]["overnight_div"]
            st = stats(x)
            rows.append(dict(symbol=s, open_weekday=["Mon", "Tue", "Wed", "Thu", "Fri"][d], n=st["n"],
                             mean_bps=x.mean() * 1e4, t_nw=st["t_nw"], label="contaminated_holdout"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Conditions: the ONE place flags are computed
# ----------------------------------------------------------------------------

class Calendar:
    """Calendar inputs. trading_days[s] is the exchange schedule of symbol s (known in advance, therefore not
    truncated in the look-ahead self-test); events has columns event_type,event_date,scheduled."""

    def __init__(self, events, trading_days_by_symbol):
        self.events = events
        self.trading_days = trading_days_by_symbol

    def event_dates(self, etype):
        e = self.events
        return pd.DatetimeIndex(e[(e["event_type"] == etype) & (e["scheduled"] == 1)]["event_date"])


def _turn_of_month(tdays):
    """Flag (-1,+1,+2,+3) per trading day of schedule tdays; the first month is skipped when the schedule starts mid-month."""
    ym = pd.Series(tdays.year * 100 + tdays.month, index=tdays)
    pos = ym.groupby(ym).cumcount().values              # 0-based position from the start of the month
    last = tdays.values == pd.Series(tdays, index=tdays).groupby(ym).transform("max").values
    full = (ym != ym.iloc[0]).values if tdays[0].day > 5 else np.ones(len(tdays), dtype=bool)
    return pd.Series((last | (pos <= 2)) & full, index=tdays)


def condition_flags(rets_by_symbol, calendar, vix):
    """date x (symbol, condition) frame of 0/1 flags (NaN on dates the symbol does not trade).
    Reads only data dated <= t-1 (VIX <= t-2) plus calendars; no forward-looking shifts."""
    fomc = calendar.event_dates("FOMC_STATEMENT")
    cpi = calendar.event_dates("US_CPI")
    nfp = calendar.event_dates("US_NFP")
    frames = {}
    for s, r in rets_by_symbol.items():
        idx = r.index
        tdays = calendar.trading_days[s]
        dow = idx.dayofweek
        pos = tdays.get_indexer(idx)
        assert (pos >= 0).all(), f"{s}: return dates missing from its trading-day schedule"
        prev = pd.DatetimeIndex([tdays[p - 1] if p > 0 else pd.NaT for p in pos])
        gap = (idx - prev).days
        f = pd.DataFrame(index=idx)
        f["H1"] = np.isin(dow, [1, 2])
        f["H2"] = (dow == 0) | (np.asarray(gap) >= 3)
        f["H3"] = idx.isin(fomc)

        def release_nights(rel):
            rel = rel[rel >= tdays[0]]
            k = tdays.searchsorted(rel, side="left")      # first open at/after the release day (08:30 < 09:30 open)
            k = k[k < len(tdays)]
            return tdays[k]
        f["H4"] = idx.isin(release_nights(cpi))
        f["H5"] = idx.isin(release_nights(nfp))
        f["H6"] = _turn_of_month(tdays).reindex(idx).fillna(False).values
        tot = r["total"]
        sigma = tot.shift(2).rolling(20).std()
        f["H7"] = (tot.shift(1) <= -1.5 * sigma).values
        f.loc[sigma.isna().values, "H7"] = False
        if vix is not None:
            v = vix.reindex(vix.index.union(idx)).ffill().reindex(idx)   # backward asof only
            f["H8"] = (v.shift(2) >= 20).values
            f.loc[v.shift(2).isna().values, "H8"] = False
            f["H8_t1"] = (v.shift(1) >= 20).values                        # labelled sensitivity (t-1 close not known at entry)
            f.loc[v.shift(1).isna().values, "H8_t1"] = False
            f["VIX_t2"] = v.shift(2).values
            f["VIX_t1"] = v.shift(1).values
        if "volume" in r:
            vol = r["volume"]
            ratio = vol.shift(1) / vol.shift(2).rolling(20).mean()
            f["X2"] = (ratio >= 2).fillna(False).values
        f["X3"] = f["H3"] | f["H4"] | f["H5"] | f["H6"]
        f["X4a"] = f["H3"]
        for c in f.columns:
            if c not in ("VIX_t2", "VIX_t1"):
                f[c] = f[c].astype(float)
        frames[s] = f
    return pd.concat(frames, axis=1, sort=True)


def flag_series(flags, sym, cond, idx=None):
    s = flags[(sym, cond)].dropna()
    return s if idx is None else s.reindex(idx).fillna(0.0)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def nw_lags(n):
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))   # same rule as newey_west_t


def ols_hac(y, X, lags=None):
    """OLS with Newey-West (Bartlett) HAC covariance; returns beta, se."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n = len(y)
    lags = nw_lags(n) if lags is None else lags
    xtxi = np.linalg.inv(X.T @ X)
    beta = xtxi @ (X.T @ y)
    e = y - X @ beta
    Z = X * e[:, None]
    S = Z.T @ Z
    for L in range(1, lags + 1):
        G = Z[L:].T @ Z[:-L]
        S += (1.0 - L / (lags + 1.0)) * (G + G.T)
    V = xtxi @ S @ xtxi
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def design(C, dow, with_dow):
    cols = [np.ones(len(C)), C]
    if with_dow:
        for k in (1, 2, 3, 4):
            cols.append((dow == k).astype(float))
    return np.column_stack(cols)


def t_a(on_bp, C, dow, cond):
    """T-a: b (bp/night), HAC SE."""
    X = design(C, dow, cond not in WEEKDAY_CONDS)
    beta, se = ols_hac(on_bp, X)
    b, s = beta[1], se[1]
    return dict(b_bp=b, se_b_bp=s, t_b=(b / s if s > 0 else np.nan), ci_lo_bp=b - 1.96 * s, ci_hi_bp=b + 1.96 * s)


def permutation_p(on, C, strata, B, rng, chunk=400):
    """One-sided p of mean(on | C) vs random condition sets with the same count (per stratum if strata given)."""
    on = np.asarray(on, float)
    C = np.asarray(C, bool)
    obs = on[C].mean()
    groups = [np.arange(len(on))] if strata is None else [np.where(strata == g)[0] for g in np.unique(strata)]
    ge = 0
    done = 0
    k_tot = int(C.sum())
    while done < B:
        m = min(chunk, B - done)
        sums = np.zeros(m)
        for g in groups:
            kg = int(C[g].sum())
            if kg == 0:
                continue
            if kg == len(g):
                sums += on[g].sum()
                continue
            R = rng.random((m, len(g)))
            sel = np.argpartition(R, kg - 1, axis=1)[:, :kg]
            sums += on[g][sel].sum(axis=1)
        ge += int(((sums / k_tot) >= obs - 1e-15).sum())
        done += m
    return obs, (1 + ge) / (B + 1)


def cell_rng(seed, key):
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


# ----------------------------------------------------------------------------
# Costs and tax
# ----------------------------------------------------------------------------

def market_of(sym):
    return "ca" if sym.endswith(".TO") else "us"


def fx_spread(capital):
    for lim, sp in COSTS["fx_tiers"]:
        if capital < lim:
            return sp
    return COSTS["fx_tiers"][-1][1]


def scenario_costs(name):
    """COSTS copy with the named SCENARIOS override applied; the module-level COSTS is never mutated."""
    ov = SCENARIOS[name]
    c = copy.deepcopy(COSTS)
    if "commission" in ov:
        c["commission"] = ov["commission"]
    if "spread_mult" in ov:
        c["half_spread_bp"] = {k: v * ov["spread_mult"] for k, v in c["half_spread_bp"].items()}
    if "timing_bp" in ov:
        c["timing_slippage_bp_per_side"] = {k: ov["timing_bp"] for k in c["timing_slippage_bp_per_side"]}
    return c


def cost_per_trip(symbol, E, L=1, costs=None, fx=False, margin=False, commission=None):
    """Dollar cost of one round trip for E dollars of underlying exposure with an L-x vehicle:
    2*commission + capital*(2*half_spread + open_slippage + 2*timing slippage) [+ FX] [+ margin financing]."""
    c = costs or COSTS
    mk = market_of(symbol)
    capital = E / L
    comm = c["commission"] if commission is None else commission
    bp = (2 * c["half_spread_bp"][symbol] + c["open_slippage_bp"][mk]
          + 2 * c["timing_slippage_bp_per_side"][mk])
    usd = 2 * comm + capital * bp / 1e4
    if fx and mk == "us":
        usd += capital * 2 * fx_spread(capital) * c["fx_surcharge"]
    if margin:
        usd += c["margin_rate"][mk] / 365.0 * capital * c["settlement_days"]
    return usd


def annual_tax(pnl_by_year, account, market, tax=None):
    """pnl_by_year: DataFrame index=year, columns net (total net P&L $), div (dividend component $).
    Tax computed per calendar year, never per night. Simplified: no loss carry-forward."""
    t = tax or TAX
    net, div = pnl_by_year["net"], pnl_by_year["div"]
    price = net - div
    if account == "TFSA":
        return div * (t["us_withholding"] if market == "us" else 0.0)
    if account == "RRSP":
        return div * 0.0
    if account == "TFSA_business_downside":
        return (net.clip(lower=0)) * t["income"]
    if account == "MARGIN_business":
        return (net.clip(lower=0)) * t["income"]
    if account == "MARGIN_capgains":
        dv = t["us_div_margin"] if market == "us" else t["ca_div_margin"]   # US: 15% withholding is creditable
        return price.clip(lower=0) * t["cap_gains"] + div * dv
    raise ValueError(account)


# ----------------------------------------------------------------------------
# Units (single instrument or equal-weight panel) and the per-cell computation
# ----------------------------------------------------------------------------

def unit_members():
    u = {s: [s] for s in HYP_SYMBOLS}
    u.update({f"panel_{p}": m for p, m in PANELS.items()})
    return u


def make_unit(members, rets, flags, cond):
    """DataFrame[on, px, tot, C, dow] for one unit. Panel night needs >= half of the members; C = >= half flagged."""
    have = [m for m in members if m in rets]
    cols = {"on": [], "px": [], "tot": [], "C": []}
    for m in have:
        r = rets[m]
        cols["on"].append(r["overnight_div"].rename(m))
        cols["px"].append(r["overnight_px"].rename(m))
        cols["tot"].append(r["total"].rename(m))
        cols["C"].append(flags[(m, cond)].dropna().rename(m))
    if not have:
        return None
    on = pd.concat(cols["on"], axis=1, sort=True)
    ok = on.notna().sum(axis=1) >= max(1, math.ceil(len(members) / 2))
    out = pd.DataFrame(index=on.index[ok])
    out["on"] = on[ok].mean(axis=1)
    out["px"] = pd.concat(cols["px"], axis=1, sort=True)[ok].mean(axis=1)
    out["tot"] = pd.concat(cols["tot"], axis=1, sort=True)[ok].mean(axis=1)
    cc = pd.concat(cols["C"], axis=1, sort=True)[ok]
    out["C"] = (cc.mean(axis=1) >= 0.5).astype(float)
    out["dow"] = out.index.dayofweek
    return out


def in_era(df, era):
    lo, hi = ERAS[era]
    return df[(df.index >= lo) & (df.index <= hi)]


def cost_frac_for(unit, E=None, costs=None):
    E = E or COSTS["decision_exposure"]
    syms = unit_members()[unit]
    return float(np.mean([cost_per_trip(s, E, costs=costs) / E for s in syms]))


def cell_frame(rets, flags, cond, unit):
    """(unit frame over all eras, return column) for one condition x unit; None if the unit has no data."""
    u_all = make_unit(unit_members()[unit], rets, flags, "H3" if cond == "X4a" else cond)
    return u_all, ("tot" if cond == "X4a" else "on")


def paired_row(u, cond, unit, era, cf, ret_col="on"):
    """T-c paired test d_t = C_t*(r_t - cf) - f*tot_t for one era frame; cf is the round-trip cost as a fraction."""
    C = u["C"].values.astype(bool)
    f = C.mean()
    d = C * (u[ret_col].values - cf) - f * u["tot"].values
    tn = newey_west_t(d)
    mean_ann = d.mean() * ANN
    se_ann = abs(mean_ann / tn) if tn and not np.isnan(tn) and tn != 0 else np.nan
    sd_d = d.std(ddof=1)
    paired = dict(condition=cond, unit=unit, era=era, f=f, cost_bp=cf * 1e4, mean_d_ann=mean_ann, t_nw_d=tn,
                  ci_lo=mean_ann - 1.96 * se_ann, ci_hi=mean_ann + 1.96 * se_ann,
                  sharpe_d=(mean_ann / (sd_d * math.sqrt(ANN)) if sd_d > 0 else np.nan),
                  hit_C=float((u[ret_col].values[C] > 0).mean()))
    return paired, pd.Series(d, index=u.index)


def cell_stats(u, cond, unit, era, seed, B, ret_col="on"):
    """All statistics of one (condition, unit, era) cell. u is the era-restricted unit frame."""
    n = len(u)
    C = u["C"].values.astype(bool)
    nC = int(C.sum())
    row = dict(condition=cond, unit=unit, era=era, n=n, n_C=nC)
    if n < 200 or nC < 10 or nC == n:
        return row, None, None
    on_bp = u[ret_col].values * 1e4
    years = n / ANN
    ta = t_a(on_bp, C.astype(float), u["dow"].values, cond)
    mC, mN = on_bp[C].mean(), on_bp[~C].mean()
    sd = on_bp[C].std(ddof=1)
    se_mean = sd / math.sqrt(nC)
    trips = nC / years
    ir = (mC / sd) * math.sqrt(trips) if sd > 0 else np.nan
    row.update(mean_on_C_bp=mC, mean_on_notC_bp=mN, mean_px_C_bp=u["px"].values[C].mean() * 1e4,
               mean_div_C_bp=(u["on"].values[C] - u["px"].values[C]).mean() * 1e4, **ta,
               se_mean_C_bp=se_mean, mde_bp=2 * se_mean,
               years_needed=((1.96 / ir) ** 2 if ir and ir > 0 else np.inf), trips_per_year=trips,
               note=("elm_reversed" if cond == "H2" and ta["t_b"] <= -2 else ""))
    strata = None if cond in WEEKDAY_CONDS else u["dow"].values
    obs, p = permutation_p(on_bp, C, strata, B, cell_rng(seed, f"{cond}|{unit}|{era}"))
    perm = dict(condition=cond, unit=unit, era=era, n_C=nC, obs_mean_on_C_bp=obs, perm_p=p, n_perm=B,
                stratified=strata is not None)
    paired, d = paired_row(u, cond, unit, era, cost_frac_for(unit), ret_col)
    return row, perm, (paired, d)


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def effective_number_of_tests(corr):
    """Copied from alpha101.stats.effective_number_of_tests (src/alpha101/stats.py): (sum eig)^2 / sum(eig^2)."""
    c = np.asarray(corr, dtype=float)
    eig = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def adjusted_threshold(n_eff, alpha=0.05):
    """Copied from alpha101.stats.adjusted_threshold: two-sided |t| bar after a Sidak correction."""
    p = 1 - (1 - alpha) ** (1 / max(n_eff, 1.0))
    return float(_st.norm.ppf(1 - p / 2))


# ----------------------------------------------------------------------------
# Self-tests and acceptance helpers
# ----------------------------------------------------------------------------

def lookahead_selftest(rets, calendar, vix, flags_full, seed, k=5):
    """Truncate all price inputs at k random dates X; flags for dates <= X must equal the full-sample flags."""
    rng = np.random.default_rng(seed)
    spy = rets[HYP_SYMBOLS[0]].index
    cut = sorted(rng.choice(np.arange(300, len(spy) - 1), k, replace=False))
    lines, ok_all = [], True
    for i in cut:
        X = spy[i]
        rt = {s: r[r.index <= X] for s, r in rets.items()}
        vt = None if vix is None else vix[vix.index <= X]
        ft = condition_flags(rt, calendar, vt)
        full = flags_full[flags_full.index <= X].reindex(ft.index)
        a = ft.fillna(-1).astype(float)
        b = full[ft.columns].fillna(-1).astype(float)
        same = bool(np.array_equal(a.values, b.values))
        ok_all &= same
        lines.append(f"{X.date()}  cells={ft.shape[1]} rows={len(ft)}  {'PASS' if same else 'FAIL'}")
    return lines, ok_all


def source_has_forward_shift(paths):
    pat = re.compile(r"\.shift\(\s*-")
    hits = []
    for p in paths:
        with open(p) as f:
            for i, line in enumerate(f, 1):
                if pat.search(line):
                    hits.append(f"{os.path.basename(p)}:{i}")
    return hits


def placebo_calibration(on_bp, dow, n_true, n_draws, seed):
    """Random conditions at the H3 frequency; share of draws with |t_b| > 2."""
    rng = np.random.default_rng(seed)
    n = len(on_bp)
    hits = 0
    ts = []
    for _ in range(n_draws):
        C = np.zeros(n)
        C[rng.choice(n, n_true, replace=False)] = 1.0
        beta, se = ols_hac(on_bp, design(C, dow, True))
        t = beta[1] / se[1]
        ts.append(t)
        hits += abs(t) > 2
    return hits / n_draws, hits, float(np.std(ts)), nw_lags(n)


def leverage_tracking(rets):
    rows = []
    for v, (und, L, _) in VEHICLES.items():
        if L == 1 or v not in rets or und not in rets:
            continue
        j = pd.concat([rets[v]["overnight_div"].rename("f"), rets[und]["overnight_div"].rename("u")], axis=1, sort=True).dropna()
        for label, g in [("ALL", j)] + [(str(y), g) for y, g in j.groupby(j.index.year)]:
            if len(g) < 30:
                continue
            slope, icpt, r, _, _ = _st.linregress(g["u"].values, g["f"].values)
            rows.append(dict(vehicle=v, underlying=und, L=L, year=label, n=len(g), slope=slope, r2=r * r,
                             mean_diff_bp=(g["f"] - L * g["u"]).mean() * 1e4,
                             slope_within_10pct=abs(slope / L - 1) <= 0.10))
    return pd.DataFrame(rows)


def _reproduce(rets, contam, ps, wk, source):
    rows = []
    for s in ("SPY", "XIU.TO"):
        r = in_era(rets[s], "D2")
        u = pd.DataFrame({"on": r["overnight_div"], "C": 1.0})
        mine = u["on"][u["C"].astype(bool)].mean() * ANN
        ref = float(ps[(ps.symbol == s) & (ps.period == "2010-2018")]["on_mean"].iloc[0])
        rows.append(dict(source=source, check=f"{s} D2 annualised overnight_div mean (C==1)", mine=mine, phase0=ref,
                         abs_diff=abs(mine - ref)))
    m = contam.merge(wk, on=["symbol", "open_weekday"], suffixes=("", "_p0"))
    for col, col0 in (("n", "n_p0"), ("mean_bps", "mean_bps_p0"), ("t_nw", "t_nw_p0")):
        rows.append(dict(source=source, check=f"weekday table {col} (contaminated_holdout, {len(m)} cells)", mine=float("nan"),
                         phase0=float("nan"), abs_diff=float((m[col] - m[col0]).abs().max())))
    return rows


def phase0_reproduction(rets, contam, phase0_dir):
    """Criterion 1: C==1 annualised overnight_div means (2010-2018) and the weekday table vs the Phase 0 CSVs.
    Two sources: the `yahoo_ohlcv` table (prices stored as DECIMAL(18,6), so ~5e-7 absolute rounding vs the
    float64 Phase 0 used) and the Phase 0 raw_yahoo CSV cache (bit-identical inputs)."""
    ps = pd.read_csv(os.path.join(phase0_dir, "period_summary.csv"))
    wk = pd.read_csv(os.path.join(phase0_dir, "overnight_by_weekday_recent.csv"))
    rows = _reproduce(rets, contam, ps, wk, "db")
    raw = os.path.join(phase0_dir, "raw_yahoo")
    if os.path.isdir(raw):
        fr = {}
        for s in HYP_SYMBOLS:
            fp = os.path.join(raw, s + ".csv")
            if os.path.exists(fp):
                fr[s] = pd.read_csv(fp, parse_dates=["date"]).set_index("date")
        fr = clean_frames(fr)
        full = {s: build_returns(df) for s, df in fr.items()}
        rows += _reproduce({s: r[r.index < HO_START] for s, r in full.items()}, contaminated_weekday_table(fr), ps, wk,
                           "phase0_raw_cache")
    out = pd.DataFrame(rows)
    out["pass_1e-9"] = out["abs_diff"] <= 1e-9
    return out


# ----------------------------------------------------------------------------
# Stage A pipeline
# ----------------------------------------------------------------------------

def run_stage_a(frames_full, events, out_dir, seed, n_perm, n_placebo, phase0_dir, log, n_selftest=5):
    os.makedirs(out_dir, exist_ok=True)
    frames_full = clean_frames(frames_full)
    contam = contaminated_weekday_table(frames_full)
    contam.to_csv(os.path.join(out_dir, "contaminated_holdout_weekday.csv"), index=False)
    frames = apply_era_mask(frames_full)           # holdout lock: nothing below sees a row >= 2019-01-01
    del frames_full
    assert all(df.index.max() < HO_START for df in frames.values())

    vix = frames["^VIX"]["close"].astype(float) if "^VIX" in frames else None
    rets, tds = {}, {}
    for s in LOAD_SYMBOLS:
        if s not in frames or s == "^VIX":
            continue
        r = build_returns(frames[s])
        r["volume"] = frames[s]["volume"].reindex(r.index)
        rets[s] = r
        tds[s] = trading_days(frames[s])
    cal = Calendar(events[events["event_date"] < HO_START], tds)
    flags = condition_flags(rets, cal, vix)
    log(f"flags: {flags.shape[1]} columns over {len(flags)} dates")

    # ------------------------------------------------------------ acceptance helpers
    p0 = None
    if phase0_dir and os.path.exists(os.path.join(phase0_dir, "period_summary.csv")):
        p0 = phase0_reproduction(rets, contam, phase0_dir)
        p0.to_csv(os.path.join(out_dir, "phase0_reproduction.csv"), index=False)
    lines, ok = lookahead_selftest(rets, cal, vix, flags, seed, n_selftest)
    hits = source_has_forward_shift([os.path.abspath(__file__)])
    with open(os.path.join(out_dir, "lookahead_selftest.txt"), "w") as f:
        f.write("truncate-and-recompute: flags for dates <= X must equal full-sample flags\n")
        f.write("\n".join(lines) + "\n")
        f.write(f"forward shift(-k) in module source: {'NONE' if not hits else ', '.join(hits)}\n")
        f.write(f"OVERALL {'PASS' if ok and not hits else 'FAIL'}\n")
    log(f"lookahead self-test: {'PASS' if ok and not hits else 'FAIL'}")
    lt = leverage_tracking(rets)
    lt.to_csv(os.path.join(out_dir, "leverage_tracking.csv"), index=False)

    # ------------------------------------------------------------ condition counts
    cc = []
    for (s, c), col in flags.items():
        if c.startswith("VIX_"):
            continue
        col = col.dropna()
        g = col.groupby(col.index.year).sum().astype(int)
        for y, v in g.items():
            cc.append(dict(symbol=s, condition=c, year=y, nights_flagged=v, nights=int((col.index.year == y).sum())))
    pd.DataFrame(cc).to_csv(os.path.join(out_dir, "condition_counts.csv"), index=False)

    # ------------------------------------------------------------ the cells
    conds = HYPOTHESES + EXPLORATORY
    units = unit_members()
    tests, perms, paired, dser = [], [], [], {}
    for cond in conds:
        for unit, members in units.items():
            if not all((m, cond) in flags.columns for m in members if m in rets):
                continue
            u_all, ret_col = cell_frame(rets, flags, cond, unit)
            if u_all is None:
                continue
            for era in ("D1", "D2", "D1D2"):
                u = in_era(u_all, era)
                tr, pr, pd_ = cell_stats(u, cond, unit, era, seed, n_perm if era != "D1D2" else 1, ret_col)
                if era == "D1D2":
                    pr = None           # permutation p-values are reported per discovery era only
                tests.append(tr)
                if pr:
                    perms.append(pr)
                if pd_:
                    paired.append(pd_[0])
                    dser[(cond, unit, era)] = pd_[1]
    T = pd.DataFrame(tests)
    P = pd.DataFrame(perms)
    PD = pd.DataFrame(paired)
    log(f"cells: tests={len(T)} perms={len(P)} paired={len(PD)}")

    # ------------------------------------------------------------ n_eff, z*
    d_mat = pd.DataFrame({f"{c}|{u}": s for (c, u, e), s in dser.items() if e == "D1D2"})
    corr = d_mat.corr(min_periods=100).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    n_eff = effective_number_of_tests(corr)
    z_star = adjusted_threshold(n_eff)
    m_cells = d_mat.shape[1]
    neff = dict(n_cells=m_cells, n_eff=n_eff, z_star_sidak=z_star,
                z_bonferroni=float(_st.norm.ppf(1 - 0.05 / m_cells / 2)),
                z_sidak_raw_count=adjusted_threshold(float(m_cells)),
                decision_grade_cells=len(HYPOTHESES) * len(PANELS))
    with open(os.path.join(out_dir, "neff.json"), "w") as f:
        json.dump(neff, f, indent=1, sort_keys=True, default=_json_default)
    log(f"n_eff={n_eff:.2f} of {m_cells} cells; z*={z_star:.3f}")

    # ------------------------------------------------------------ per-era files
    for era in ("D1", "D2", "D1D2"):
        T[T.era == era].to_csv(os.path.join(out_dir, f"tests_{era}.csv"), index=False)
        if era != "D1D2":
            P[P.era == era].to_csv(os.path.join(out_dir, f"placebo_{era}.csv"), index=False)
        PD[PD.era == era].to_csv(os.path.join(out_dir, f"paired_vs_bh_{era}.csv"), index=False)

    # ------------------------------------------------------------ VIX buckets
    vb = []
    for unit, members in units.items():
        for lag in ("VIX_t2", "VIX_t1"):
            vs = pd.concat([flags[(m, lag)].dropna().rename(m) for m in members if m in rets], axis=1, sort=True).mean(axis=1)
            u = make_unit(members, rets, flags, "H1")
            if u is None:
                continue
            u = u.join(vs.rename("v"), how="inner").dropna(subset=["v"])
            for era in ("D1", "D2"):
                ue = in_era(u, era)
                for name, lo, hi in (("<15", -np.inf, 15), ("15-20", 15, 20), ("20-30", 20, 30), (">=30", 30, np.inf)):
                    x = ue[(ue.v >= lo) & (ue.v < hi)]["on"]
                    vb.append(dict(unit=unit, era=era, vix_lag=lag, bucket=name, n=len(x),
                                   mean_on_bp=x.mean() * 1e4 if len(x) else np.nan,
                                   t_nw=newey_west_t(x.values) if len(x) >= 10 else np.nan))
    VB = pd.DataFrame(vb)
    for era in ("D1", "D2"):
        VB[VB.era == era].to_csv(os.path.join(out_dir, f"vix_buckets_{era}.csv"), index=False)

    # ------------------------------------------------------------ costs, tax, break-even
    ct, be = cost_tax_tables(rets, flags, conds)
    for era in ("D1", "D2"):
        ct[ct.era == era].to_csv(os.path.join(out_dir, f"cost_tax_{era}.csv"), index=False)
        be[be.era == era].to_csv(os.path.join(out_dir, f"breakeven_{era}.csv"), index=False)

    # ------------------------------------------------------------ placebo calibration (criterion 4)
    spy_all = in_era(make_unit(["SPY"], rets, flags, "H3"), "D1D2")
    n_h3 = int(spy_all["C"].sum())
    rate, hits_n, sd_t, lags = placebo_calibration(spy_all["on"].values * 1e4, spy_all["dow"].values, n_h3,
                                                   n_placebo, seed + 1)
    pc = dict(n_draws=n_placebo, n_nights=len(spy_all), n_condition_nights=n_h3, nw_lags=lags,
              rate_abs_t_gt_2=rate, hits=hits_n, sd_of_t=sd_t, in_range_3_7pct=bool(0.03 <= rate <= 0.07))
    with open(os.path.join(out_dir, "placebo_calibration.json"), "w") as f:
        json.dump(pc, f, indent=1, sort_keys=True, default=_json_default)
    log(f"placebo calibration: |t|>2 in {rate:.4f} of {n_placebo} draws (n={len(spy_all)}, H3 nights={n_h3}, lags={lags})")

    # ------------------------------------------------------------ decisions
    DEC = decide(T, P, PD, z_star)
    for era in ("D1", "D2"):
        DEC[DEC.era == era].to_csv(os.path.join(out_dir, f"decision_table_{era}.csv"), index=False)
    final = DEC[DEC.era == "FINAL"].drop(columns=["era"])
    final.to_csv(os.path.join(out_dir, "decision_table_D1D2.csv"), index=False)
    return dict(tests=T, perms=P, paired=PD, decisions=final, neff=neff, placebo=pc, dseries=dser, rets=rets,
                flags=flags, cost_tax=ct, breakeven=be, vix=VB, lookahead_ok=ok and not hits, phase0=p0, leverage=lt)


def cost_tax_tables(rets, flags, conds):
    """Net economics by condition x vehicle x exposure x FX x financing x account, per era (D1, D2).
    Gross per night on a flagged night = on(vehicle)/L per dollar of exposure."""
    rows, brows = [], []
    for cond in conds:
        if cond == "X4a":
            continue
        for veh, (und, L, grp) in VEHICLES.items():
            if veh not in rets or (und, cond) not in flags.columns:
                continue
            r = rets[veh]
            C = flag_series(flags, und, cond).reindex(r.index).fillna(0).astype(bool)
            for era in ("D1", "D2"):
                sel = (r.index >= ERAS[era][0]) & (r.index <= ERAS[era][1])
                rr, CC = r[sel], C[sel]
                if CC.sum() < 10 or len(rr) < 200:
                    continue
                years = len(rr) / ANN
                trips = CC.sum() / years
                g_on = (rr["overnight_div"][CC] / L)
                g_div = ((rr["overnight_div"] - rr["overnight_px"])[CC] / L)
                mk = market_of(veh)
                for E in COSTS["exposures"]:
                    for fx in (False, True):
                        if fx and mk == "ca":
                            continue
                        for margin in (False, True):
                            if margin and L > 1:
                                continue
                            cf = cost_per_trip(veh, E, L, fx=fx, margin=margin) / E
                            net = g_on - cf
                            by_y = pd.DataFrame({"net": net * E, "div": g_div * E}).groupby(net.index.year).sum()
                            for acct in ACCOUNTS:
                                tx = annual_tax(by_y, acct, mk)
                                after = (by_y["net"] - tx).sum() / years
                                rows.append(dict(
                                    condition=cond, vehicle=veh, underlying=und, L=L, era=era, exposure=E,
                                    fx="convert_each_trade" if fx else "usd_side_held",
                                    financing="margin" if margin else "idle_cash", account=acct,
                                    nights=int(CC.sum()), trips_per_year=trips,
                                    gross_bp_per_night=g_on.mean() * 1e4, cost_bp_per_trip=cf * 1e4,
                                    net_pre_tax_per_year=by_y["net"].sum() / years, tax_per_year=tx.sum() / years,
                                    net_after_tax_per_year=after, cra_business_income_risk=bool(trips > CRA_TRIPS)))
                gm = g_on.mean()
                var = (COSTS["half_spread_bp"][veh] * 2 + COSTS["open_slippage_bp"][mk]
                       + 2 * COSTS["timing_slippage_bp_per_side"][mk]) / L / 1e4
                brows.append(dict(condition=cond, vehicle=veh, underlying=und, L=L, era=era,
                                  mean_on_C_bp_per_exposure=gm * 1e4, variable_cost_bp=var * 1e4,
                                  breakeven_exposure=(2 * COSTS["commission"] / (gm - var) if gm > var else float("inf"))))
    return pd.DataFrame(rows), pd.DataFrame(brows)


def decide(T, P, PD, z_star):
    """Operationalised decision rules (module docstring). Returns per-era criteria rows and FINAL rows."""
    out = []
    key = ["condition", "unit"]
    for cond in HYPOTHESES + EXPLORATORY:
        for unit in unit_members():
            rows = {}
            for era in ("D1", "D2"):
                t = T[(T.condition == cond) & (T.unit == unit) & (T.era == era)]
                p = P[(P.condition == cond) & (P.unit == unit) & (P.era == era)]
                pdr = PD[(PD.condition == cond) & (PD.unit == unit) & (PD.era == era)]
                if t.empty or "b_bp" not in t or pd.isna(t["b_bp"].iloc[0]) or p.empty or pdr.empty:
                    continue
                t, p, pdr = t.iloc[0], p.iloc[0], pdr.iloc[0]
                bar = 2.0 if cond == "H1" else z_star
                crit = dict(b_pos=bool(t.b_bp > 0), t_ok=bool(t.t_b >= bar), perm_ok=bool(p.perm_p < 0.05),
                            paired_ok=bool(pdr.mean_d_ann > 0), trips_ok=bool(t.trips_per_year <= MAX_TRIPS))
                rows[era] = (t, p, pdr, crit, bar)
                out.append(dict(condition=cond, unit=unit, era=era, n_C=int(t.n_C), b_bp=t.b_bp, t_b=t.t_b, bar=bar,
                                perm_p=p.perm_p, mean_d_ann=pdr.mean_d_ann, trips_per_year=t.trips_per_year,
                                ci_hi_bp=t.ci_hi_bp, mde_bp=t.mde_bp, **crit, all_pass=all(crit.values())))
            if "D2" not in rows:
                continue
            t2, _, pd2, _, _ = rows["D2"]
            eras_needed = ["D2"] if cond == "H1" else ["D2"] + (["D1"] if ("D1" in rows and rows["D1"][0].n_C >= 30) else [])
            adv = all(rows[e][3] and all(rows[e][3].values()) for e in eras_needed)
            cost_bp = pd2.cost_bp
            if adv:
                verdict, why = "advance", "all criteria pass in " + "+".join(eras_needed)
            elif t2.b_bp <= 0:
                verdict, why = "reject", "b <= 0 in D2"
            elif t2.ci_hi_bp < cost_bp:
                verdict, why = "reject", "D2 CI upper bound below round-trip cost"
            elif t2.trips_per_year > MAX_TRIPS:
                verdict, why = "reject", f"round trips/yr {t2.trips_per_year:.0f} > {MAX_TRIPS}"
            else:
                verdict, why = "underpowered", "b > 0 but CI still contains an economically relevant effect"
            grade = cond in HYPOTHESES and unit.startswith("panel_")
            out.append(dict(condition=cond, unit=unit, era="FINAL", n_C=int(t2.n_C), b_bp=t2.b_bp, t_b=t2.t_b,
                            bar=rows["D2"][4], perm_p=rows["D2"][1].perm_p, mean_d_ann=pd2.mean_d_ann,
                            trips_per_year=t2.trips_per_year, ci_hi_bp=t2.ci_hi_bp, mde_bp=t2.mde_bp,
                            cost_bp=cost_bp, eras_required="+".join(eras_needed), verdict=verdict, reason=why,
                            decision_grade=grade, z_star=z_star))
    return pd.DataFrame(out)


def scenario_decisions(res, name):
    """(FINAL decision rows, paired table) for one SCENARIOS entry. Only the cost-dependent inputs (T-c series, cost_bp) change."""
    costs = scenario_costs(name)
    T = res["tests"]
    paired = []
    for cond in HYPOTHESES + EXPLORATORY:
        for unit in unit_members():
            u_all, ret_col = cell_frame(res["rets"], res["flags"], cond, unit)
            if u_all is None:
                continue
            for era in ("D1", "D2", "D1D2"):
                t = T[(T.condition == cond) & (T.unit == unit) & (T.era == era)]
                if t.empty or "b_bp" not in t or pd.isna(t["b_bp"].iloc[0]):
                    continue                     # cell_stats returned no paired row for these either
                paired.append(paired_row(in_era(u_all, era), cond, unit, era, cost_frac_for(unit, costs=costs), ret_col)[0])
    PD = pd.DataFrame(paired)
    dec = decide(T, res["perms"], PD, res["neff"]["z_star_sidak"])
    return dec[dec.era == "FINAL"].drop(columns=["era"]).reset_index(drop=True), PD


def cost_sensitivity(res, out_dir, log):
    """Run every SCENARIOS entry, write cost_sensitivity.csv (one row per scenario x condition x unit) and return it.
    The 'default' scenario must reproduce the main decision table exactly, otherwise this raises."""
    base = res["decisions"].reset_index(drop=True)
    frames = []
    for name in SCENARIOS:
        dec, PD = scenario_decisions(res, name)
        if name == "default":
            pd.testing.assert_frame_equal(dec, base, check_exact=True)
        d = dec.merge(base[["condition", "unit", "verdict"]].rename(columns={"verdict": "default_verdict"}), on=["condition", "unit"])
        d["scenario"] = name
        d = d.merge(res["tests"][res["tests"].era == "D2"][["condition", "unit", "mean_on_C_bp"]], on=["condition", "unit"], how="left")
        d = d.merge(PD[PD.era == "D2"][["condition", "unit", "t_nw_d"]], on=["condition", "unit"], how="left")
        d["net_bp_D2"] = d["mean_on_C_bp"] - d["cost_bp"]
        d["flipped"] = d["verdict"] != d["default_verdict"]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "cost_sensitivity.csv"), index=False)
    n_flip = int(out[out.scenario != "default"].flipped.sum())
    log(f"cost sensitivities: {len(SCENARIOS)} scenarios x {len(base)} cells; verdict flips vs default: {n_flip}")
    return out


# ----------------------------------------------------------------------------
# Figures, report, DB rows
# ----------------------------------------------------------------------------

def make_figures(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = res["tests"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    conds = HYPOTHESES + EXPLORATORY
    for ax, unit in zip(axes, ("panel_US", "panel_CA")):
        for j, era in enumerate(("D1", "D2")):
            t = T[(T.unit == unit) & (T.era == era)].set_index("condition").reindex(conds)
            x = np.arange(len(conds)) + (j - 0.5) * 0.35
            ok = t["b_bp"].notna()
            ax.errorbar(x[ok], t["b_bp"][ok], yerr=1.96 * t["se_b_bp"][ok], fmt="o", capsize=3, label=era)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45)
        ax.set_title(f"{unit}: b (bp/night) +/- 95% HAC CI")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b_by_condition.png"), dpi=110)
    plt.close(fig)

    dec = res["decisions"]
    adv = dec[(dec.verdict == "advance") & dec.decision_grade]
    fig, ax = plt.subplots(figsize=(10, 5))
    if adv.empty:
        ax.text(0.5, 0.5, "no condition advances (Stage A)", ha="center", va="center")
        ax.axis("off")
    else:
        for r in adv.itertuples():
            d = res["dseries"][(r.condition, r.unit, "D1D2")]
            on = make_unit(unit_members()[r.unit], res["rets"], res["flags"], r.condition)
            on = in_era(on, "D1D2")
            net = on["C"] * (on["on"] - cost_frac_for(r.unit)) * COSTS["decision_exposure"]
            ax.plot(net.index, net.cumsum(), label=f"{r.condition} {r.unit}")
        ax.legend()
        ax.set_ylabel("cumulative net P&L, $ (E = $100k)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "net_pnl_cum.png"), dpi=110)
    plt.close(fig)


def _fmt(x, nd=2):
    if isinstance(x, (float, np.floating)):
        return "nan" if np.isnan(x) else ("inf" if np.isinf(x) else f"{x:.{nd}f}")
    return str(x)


def md_table(df, cols=None, nd=2):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir, name):
    rd = lambda f: pd.read_csv(os.path.join(out_dir, f))   # noqa: E731
    dec = rd("decision_table_D1D2.csv")
    neff = json.load(open(os.path.join(out_dir, "neff.json")))
    pc = json.load(open(os.path.join(out_dir, "placebo_calibration.json")))
    T2, T1 = rd("tests_D2.csv"), rd("tests_D1.csv")
    lev = rd("leverage_tracking.csv")
    lev_all = lev[lev.year == "ALL"]
    p0 = rd("phase0_reproduction.csv") if os.path.exists(os.path.join(out_dir, "phase0_reproduction.csv")) else None
    contam = rd("contaminated_holdout_weekday.csv")
    selftest = open(os.path.join(out_dir, "lookahead_selftest.txt")).read()
    cal = rd("event_calendar_check.csv") if os.path.exists(os.path.join(out_dir, "event_calendar_check.csv")) else None
    L = [f"# Phase 1 Stage A -- conditional overnight premium (generated by `{SCRIPT_PATH} --write-report`)", "",
         "All numbers below are read from the CSV/JSON files in this directory. Stage A uses D1 (to 2009-12-31) and D2 "
         "(2010-01-01 to 2018-12-31); the 2019+ holdout is locked.", "",
         "## Multiple testing", "",
         f"- cells in the n_eff matrix: {neff['n_cells']}; n_eff = {neff['n_eff']:.2f}; z* (Sidak) = {neff['z_star_sidak']:.3f}; "
         f"Bonferroni z = {neff['z_bonferroni']:.3f}; Sidak at the raw count = {neff['z_sidak_raw_count']:.3f}", "",
         "## Stage A decision table (hypothesis x panel; H1-H8 are decision-grade)", ""]
    d = dec[dec.condition.isin(HYPOTHESES) & dec.unit.str.startswith("panel_")]
    L.append(md_table(d, ["condition", "unit", "n_C", "b_bp", "t_b", "bar", "perm_p", "mean_d_ann", "trips_per_year", "ci_hi_bp",
                          "mde_bp", "verdict", "reason"]))
    L += ["", "### Exploratory and single-instrument cells (not decision-grade)", ""]
    e = dec[~(dec.condition.isin(HYPOTHESES) & dec.unit.str.startswith("panel_"))]
    L.append(md_table(e, ["condition", "unit", "n_C", "b_bp", "t_b", "perm_p", "mean_d_ann", "trips_per_year", "verdict"]))
    for nm, T in (("D2 (2010-2018)", T2), ("D1 (to 2009)", T1)):
        L += ["", f"## T-a tests and power, {nm}", "",
              md_table(T[T.unit.str.startswith("panel_")], ["condition", "unit", "n", "n_C", "mean_on_C_bp", "mean_on_notC_bp", "b_bp",
                                                           "t_b", "ci_lo_bp", "ci_hi_bp", "se_mean_C_bp", "mde_bp", "years_needed", "note"])]
    if os.path.exists(os.path.join(out_dir, "cost_sensitivity.csv")):
        cs = rd("cost_sensitivity.csv")
        g = cs[cs.condition.isin(HYPOTHESES) & cs.unit.str.startswith("panel_")]
        L += ["", "## Cost sensitivities (spec 3.2; decision-grade cells, D2 statistics)", "",
              "One change at a time versus the default costs: " + "; ".join(f"`{k}` = {v}" for k, v in SCENARIOS.items() if v) + ". "
              "t_b, permutation p and z* are cost-independent and unchanged; net_bp_D2 = mean(on | C) - round-trip cost at E=$100k; "
              "t_nw_d is the NW t of the paired T-c series.", ""]
        flips = g[g.flipped]
        L += [f"Verdict flips versus default: **{len(flips)}**.", ""]
        if len(flips):
            L += [md_table(flips, ["scenario", "condition", "unit", "default_verdict", "verdict", "reason"]), ""]
        L.append(md_table(g, ["scenario", "condition", "unit", "mean_on_C_bp", "cost_bp", "net_bp_D2", "t_b", "t_nw_d", "mean_d_ann",
                              "verdict", "default_verdict", "flipped"]))
    L += ["", "## Contaminated holdout weekday table (2019+, Phase 0 already published it; informational only)", "",
          md_table(contam, ["symbol", "open_weekday", "n", "mean_bps", "t_nw"])]
    L += ["", "## Engineering checks", ""]
    if p0 is not None:
        L += ["### Phase 0 reproduction (criterion 1)", "", md_table(p0, nd=12), ""]
    L += ["### Placebo calibration (criterion 4)", "",
          f"{pc['n_draws']} random conditions with {pc['n_condition_nights']} nights each on {pc['n_nights']} SPY nights, NW lags = {pc['nw_lags']}: "
          f"|t_b| > 2 in {pc['rate_abs_t_gt_2'] * 100:.2f}% of draws (target 3-7%: {'in range' if pc['in_range_3_7pct'] else 'OUT OF RANGE'}).", "",
          "### Leverage tracking (criterion 5)", "", md_table(lev_all, ["vehicle", "underlying", "L", "n", "slope", "r2", "mean_diff_bp", "slope_within_10pct"], nd=3), "",
          "### Look-ahead self-test (criterion 2)", "", "```", selftest.strip(), "```"]
    if cal is not None:
        L += ["", "### Calendar (criterion 7): scheduled events per year", "", md_table(cal)]
    with open(os.path.join(out_dir, name), "w") as f:
        f.write("\n".join(L) + "\n")


def _dec(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)


def _stage_metrics(res, h, unit, cf):
    """Per-row backtest numbers for one hypothesis x panel at round-trip cost fraction cf."""
    u = in_era(make_unit(unit_members()[unit], res["rets"], res["flags"], h), "D1D2")
    C = u["C"].values
    net = C * (u["on"].values - cf)
    f = C.mean()
    bh = f * u["tot"].values
    d = C * (u["on"].values - cf) - f * u["tot"].values       # the T-c series at this cost
    eq = np.cumprod(1 + net)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    years = len(u) / ANN
    deployed = max(C.sum(), 1)
    ann_dep = net.sum() / deployed * (C.sum() / years)
    sh = lambda x: float(x.mean() * ANN / (x.std(ddof=1) * math.sqrt(ANN))) if x.std(ddof=1) > 0 else None  # noqa: E731
    return u, C, eq, mdd, ann_dep, d, bh, sh


def write_stage_a_db(conn, res, out_dir, args, log, scenario="default", dec=None):
    """Insert backtest_results/research_log rows ('pending', conclusion NULL). scenario != 'default' writes the spec 3.2
    cost-sensitivity rows (strategy_name suffix _cs_<scenario>) at that scenario's costs and decisions `dec`."""
    cur = conn.cursor()
    sfx = "" if scenario == "default" else f"_cs_{scenario}"
    cur.execute("SELECT COUNT(*) FROM backtest_results WHERE strategy_name LIKE %s", (f"overnight_cond_%_stageA{sfx}",))
    if cur.fetchone()[0] and not args.force_db:
        raise RuntimeError(f"Stage A rows (scenario {scenario}) already exist in backtest_results; refusing to duplicate (use --force-db)")
    costs = scenario_costs(scenario)
    dec = res["decisions"] if dec is None else dec
    sql_b = ("INSERT INTO backtest_results (strategy_name,start_date,end_date,universe,params,total_return,annualized_return,"
             "sharpe_ratio,max_drawdown,win_rate,total_trades,benchmark_return,benchmark_sharpe,validation_status,notes,result_file_path)"
             " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    sql_r = ("INSERT INTO research_log (hypothesis,success_criteria,test_method,result_summary,conclusion,backtest_result_id)"
             " VALUES (%s,%s,%s,%s,%s,%s)")
    ids = {}
    path = os.path.join(out_dir, "decision_table_D1D2.csv" if scenario == "default" else "cost_sensitivity.csv")
    htag = "" if scenario == "default" else f"[cost scenario {scenario}] "
    cf_all = {u: cost_frac_for(u, costs=costs) for u in unit_members()}
    for h in HYPOTHESES:
        for panel in PANELS:
            unit = f"panel_{panel}"
            row = dec[(dec.condition == h) & (dec.unit == unit)]
            if row.empty:
                continue
            row = row.iloc[0]
            u, C, eq, mdd, ann_dep, d, bh, sh = _stage_metrics(res, h, unit, cf_all[unit])
            params = dict(script=SCRIPT_PATH, hypothesis=h, panel=panel, members=PANELS[panel], stage="A",
                          era="D1D2", seed=args.seed, n_perm=args.n_perm, scenario=scenario, costs=costs, tax=TAX,
                          exposure=COSTS["decision_exposure"], frozen_file_hash=None, verdict=row.verdict,
                          z_star=res["neff"]["z_star_sidak"], n_eff=res["neff"]["n_eff"])
            cur.execute(sql_b, (f"overnight_cond_{h}_{panel}_stageA{sfx}", u.index.min().date(), u.index.max().date(),
                                ",".join(PANELS[panel]), json.dumps(params, default=str)[:60000],
                                _dec(float(eq[-1] - 1)), _dec(ann_dep), _dec(sh(pd.Series(d))), _dec(mdd),
                                _dec(float((u["on"].values[C.astype(bool)] > 0).mean())), int(2 * C.sum()),
                                _dec(float(np.prod(1 + bh) - 1)), _dec(sh(pd.Series(bh))), "pending",
                                None, path))
            ids[(h, panel)] = cur.lastrowid
    for h in HYPOTHESES:
        sub = dec[(dec.condition == h) & dec.unit.str.startswith("panel_")]
        summ = json.dumps(sub.to_dict("records"), default=str)[:60000]
        cur.execute(sql_r, (htag + HYP_TEXT[h], SUCCESS_TEXT, "overnight_conditional_test.py", summ, None,
                            ids.get((h, "US"))))
    ex = dec[dec.condition.isin(EXPLORATORY)]
    cur.execute(sql_r, (htag + "Exploratory X2 (attention), X3 (calendar union), X4a (full announcement day); X1 (pre-BoC) and X4b not run in Stage A.",
                        "Reported and counted in n_eff; not decision-grade.", "overnight_conditional_test.py",
                        json.dumps(ex.to_dict("records"), default=str)[:60000], None, None))
    conn.commit()
    log(f"DB[{scenario}]: backtest_results ids {sorted(ids.values())}; research_log rows {len(HYPOTHESES) + 1}")
    return ids


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="output directory (CSVs, PNGs, run.log, raw_yahoo/)")
    ap.add_argument("--build-calendar", action="store_true", help="build FOMC/CPI/NFP calendar; writes data/fomc_statements.csv, data/event_calendar.csv, event_calendar_check.csv")
    ap.add_argument("--fetch-new", action="store_true", help="fetch SSO UPRO TQQQ QLD ^VIX CAD=X from Yahoo into <out-dir>/raw_yahoo/")
    ap.add_argument("--from-db", action="store_true", help="run Stage A from the yahoo_ohlcv and event_calendar tables")
    ap.add_argument("--write-db", action="store_true", help="with --build-calendar: upsert event_calendar; --fetch-new: upsert yahoo_ohlcv; --from-db: insert backtest_results/research_log (pending)")
    ap.add_argument("--cost-sens", action="store_true", help="with --from-db: run the spec 3.2 cost scenarios, write cost_sensitivity.csv")
    ap.add_argument("--write-cost-sens-db", action="store_true", help="with --cost-sens: insert the non-default scenario rows (pending) into backtest_results/research_log")
    ap.add_argument("--force-db", action="store_true", help="insert Stage A DB rows even if they already exist")
    ap.add_argument("--write-report", action="store_true", help="generate the results markdown from the CSVs in --out-dir")
    ap.add_argument("--report-name", default="2026-10-19_phase1_conditional_overnight_results.md")
    ap.add_argument("--unlock-holdout", action="store_true", help="Stage B only: requires --frozen (NOT implemented in this change)")
    ap.add_argument("--frozen", default=None, help="phase1_frozen_conditions.json for Stage B")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--n-perm", type=int, default=10000, help="permutation draws per cell (T-b)")
    ap.add_argument("--n-placebo", type=int, default=1000, help="draws for the placebo calibration (criterion 4)")
    ap.add_argument("--calendar-csv", default=None, help="read events from this CSV instead of the event_calendar table")
    ap.add_argument("--phase0-dir", default=DEFAULT_PHASE0, help="Phase 0 results directory (period_summary.csv, overnight_by_weekday_recent.csv)")
    ap.add_argument("--no-plots", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.unlock_holdout and not args.frozen:
        ap.error("--unlock-holdout requires --frozen <phase1_frozen_conditions.json> (Stage B refuses to run without it)")
    if args.frozen and not os.path.exists(args.frozen):
        ap.error(f"--frozen file not found: {args.frozen}")
    if args.unlock_holdout:
        sys.exit("Stage B (holdout) is not implemented in this change: Stage A only.")
    if (args.cost_sens or args.write_cost_sens_db) and not args.from_db:
        ap.error("--cost-sens / --write-cost-sens-db need --from-db")
    if args.write_cost_sens_db and not args.cost_sens:
        ap.error("--write-cost-sens-db needs --cost-sens")
    if not (args.build_calendar or args.fetch_new or args.from_db or args.write_report):
        ap.error("nothing to do: give one of --build-calendar --fetch-new --from-db --write-report")

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "run.log"), "a")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    argd = {k: v for k, v in vars(args).items()}
    log(f"=== overnight_conditional_test.py run {dt.datetime.now().strftime('%Y%m%dT%H%M%S')} args={argd}")
    env = load_env()

    if args.build_calendar:
        fetched = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
        fomc = build_fomc(log)
        fred = build_fred_events(env, log, os.path.join(args.out_dir, "raw_calendar"))
        conn = reader_connection(env)
        cur = conn.cursor()
        cur.execute("SELECT date FROM yahoo_ohlcv WHERE symbol='SPY' ORDER BY date")
        us_days = pd.DatetimeIndex([r[0] for r in cur.fetchall()])
        conn.close()
        cal = pd.concat([fomc, fred], ignore_index=True)
        cal["fetched_at"] = fetched
        cal = calendar_checks(cal, us_days, args.out_dir, log)
        os.makedirs(DATA_DIR, exist_ok=True)
        fomc[["event_date", "scheduled", "has_sep", "source_url"]].to_csv(os.path.join(DATA_DIR, "fomc_statements.csv"), index=False)
        cal[CAL_COLS].drop(columns=["fetched_at"]).to_csv(os.path.join(DATA_DIR, "event_calendar.csv"), index=False)
        if args.write_db:
            w = writer_connection(env)
            write_calendar_db(w, cal[CAL_COLS], fetched, log)
            w.close()

    if args.fetch_new:
        frames = fetch_yahoo(NEW_SYMBOLS, log)
        raw_dir = os.path.join(args.out_dir, "raw_yahoo")
        os.makedirs(raw_dir, exist_ok=True)
        for s, df in frames.items():
            df.to_csv(os.path.join(raw_dir, s.replace("^", "_idx_").replace("=", "_") + ".csv"))
        if args.write_db:
            w = writer_connection(env)
            write_yahoo(w, frames, log)
            w.close()

    if args.from_db:
        conn = reader_connection(env)
        frames = read_yahoo(conn, LOAD_SYMBOLS)
        events = load_calendar_events(args, conn)
        conn.close()
        missing = [s for s in LOAD_SYMBOLS if s not in frames]
        if missing:
            raise RuntimeError(f"symbols missing from yahoo_ohlcv: {missing}")
        res = run_stage_a(frames, events, args.out_dir, args.seed, args.n_perm, args.n_placebo, args.phase0_dir, log)
        if not args.no_plots:
            make_figures(res, args.out_dir)
        if args.write_db:
            w = writer_connection(env)
            write_stage_a_db(w, res, args.out_dir, args, log)
            w.close()
        if args.cost_sens:
            cs = cost_sensitivity(res, args.out_dir, log)
            if args.write_cost_sens_db:
                w = writer_connection(env)
                for name in SCENARIOS:
                    if name != "default":
                        write_stage_a_db(w, res, args.out_dir, args, log, scenario=name, dec=cs[cs.scenario == name])
                w.close()

    if args.write_report:
        write_report(args.out_dir, args.report_name)
        log(f"report written: {args.report_name}")
    logf.close()


if __name__ == "__main__":
    main()
