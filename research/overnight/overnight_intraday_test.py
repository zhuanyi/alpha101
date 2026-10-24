#!/usr/bin/env python3
"""
Overnight vs intraday return anomaly test (Knuteson / Cooper-Cliff-Gulen / Lou-Polk-Skouras).

Does the close-to-open (overnight) return still dominate the open-to-close (intraday)
return for US and Canadian index ETFs, index levels and large-cap single names, and
does any of it survive retail costs?

Data: Yahoo Finance via yfinance (max history, auto_adjust=False). Close-only vendor series cannot be used
for an open/close split, so raw Yahoo OHLCV is stored in a `yahoo_ohlcv` table (PK symbol+date) and the test
is reproducible without re-downloading.

Return definitions (log returns, day t):
  intraday_t          = ln(Close_t / Open_t)
  overnight_px_t      = ln(Open_t / Close_{t-1})              price-only (no dividends)
  total_t             = ln(AdjClose_t / AdjClose_{t-1})        split+dividend adjusted
  overnight_div_t     = total_t - intraday_t                   dividends accrue to the overnight holder
Yahoo's Open/Close/High/Low are split-adjusted but not dividend-adjusted, so overnight_px
and intraday are internally consistent across splits. Known Yahoo artefact: index Open
before the early 1980s (^GSPC) equals the prior Close; we drop days where Open == prior
Close for all series ("stale open" filter) and report how many were dropped.

Credentials: ALPHA101_DB_* environment variables (see config.example); nothing is hardcoded.

Usage:
  python3 overnight_intraday_test.py --help
  python3 overnight_intraday_test.py --fetch --write-db --out-dir <dir>
  python3 overnight_intraday_test.py --from-db --out-dir <dir>   # re-run from stored data
"""
import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Universe (fixed; no survivorship choice is made on returns)
# ----------------------------------------------------------------------------
UNIVERSE = {
    "us_etf":     ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "TLT", "GLD"],
    "ca_etf":     ["XIU.TO", "XIC.TO", "ZCN.TO", "XSP.TO", "XFN.TO", "XEG.TO", "ZEB.TO"],
    "index":      ["^GSPC", "^GSPTSE"],
    "us_single":  ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "GOOGL", "BRK-B", "JNJ",
                   "PG", "V", "UNH", "HD", "COST", "META", "TSLA"],
    "ca_single":  ["RY.TO", "TD.TO", "SHOP.TO", "CNQ.TO", "ENB.TO", "BN.TO", "BMO.TO",
                   "BNS.TO", "CP.TO", "CNR.TO", "SU.TO", "TRP.TO", "MFC.TO", "ATD.TO",
                   "WCN.TO", "CSU.TO"],
}
ALL_SYMBOLS = [s for grp in UNIVERSE.values() for s in grp]
GROUP_OF = {s: g for g, syms in UNIVERSE.items() for s in syms}

# ----------------------------------------------------------------------------
# Cost / tax assumptions (all documented in the report; override on CLI)
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    commission_per_order=9.99,        # CAD or USD per trade, the retail broker standard, verified 2026-10-18
    commission_active_per_order=7.00, # >=150 trades/quarter
    half_spread_bps_etf=0.5,   # SPY/QQQ-class US ETF: ~1 cent on $500-700 => ~0.1-0.2 bp; 0.5 bp conservative
    half_spread_bps_ca_etf=2.0,# XIU-class TSX ETF: 1 cent on ~$40-55 => ~1-1.5 bp; 2 bp conservative
    half_spread_bps_single=1.0,# US mega-cap single names
    half_spread_bps_ca_single=3.0,
    cap_gains_rate=0.2676,     # Ontario top combined 2026, capital gains (KPMG)
    income_rate=0.5353,        # Ontario top combined 2026, interest/business income (KPMG)
    positions=[50_000, 100_000, 250_000],
)

# ----------------------------------------------------------------------------
# Env / DB helpers
# ----------------------------------------------------------------------------

def load_env():
    """Process environment only (no env-file lookup); kept as a function so callers pass `env` around."""
    return dict(os.environ)


def writer_connection(env):
    """Writer connection from ALPHA101_DB_WRITE_USER / ALPHA101_DB_WRITE_PASSWORD (see config.example)."""
    from alpha101 import db
    return db.writer()


def reader_connection(env):
    """Reader connection from ALPHA101_DB_USER / ALPHA101_DB_PASSWORD (see config.example)."""
    from alpha101 import db
    return db.reader()


YAHOO_DDL = """
CREATE TABLE IF NOT EXISTS yahoo_ohlcv (
  symbol     VARCHAR(20)   NOT NULL,
  date       DATE          NOT NULL,
  open       DECIMAL(18,6) NULL,
  high       DECIMAL(18,6) NULL,
  low        DECIMAL(18,6) NULL,
  close      DECIMAL(18,6) NULL,
  adj_close  DECIMAL(18,6) NULL,
  volume     BIGINT        NULL,
  dividend   DECIMAL(18,6) NULL,
  split      DECIMAL(18,6) NULL,
  source     VARCHAR(30)   NOT NULL DEFAULT 'yfinance',
  fetched_at DATETIME      NOT NULL,
  PRIMARY KEY (symbol, date),
  KEY idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def fetch_yahoo(symbols, log):
    import yfinance as yf
    frames = {}
    fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
    for s in symbols:
        try:
            df = yf.download(s, period="max", auto_adjust=False, actions=True,
                             progress=False, threads=False)
        except Exception as e:  # noqa
            log(f"  {s}: download failed: {e}")
            continue
        if df is None or df.empty:
            log(f"  {s}: empty")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Adj Close": "adj_close", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume",
                                "Dividends": "dividend", "Stock Splits": "split"})
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df["symbol"] = s
        df["fetched_at"] = fetched_at
        frames[s] = df
        log(f"  {s}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}")
    return frames


def write_yahoo(conn, frames, log):
    cur = conn.cursor()
    cur.execute(YAHOO_DDL)
    # NOTE: every VALUES element must be a %s placeholder, otherwise pymysql's executemany cannot
    # rewrite the statement as one multi-row INSERT and falls back to one round trip per row.
    sql = ("INSERT INTO yahoo_ohlcv (symbol,date,open,high,low,close,adj_close,volume,dividend,split,source,fetched_at)"
           " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
           " ON DUPLICATE KEY UPDATE open=VALUES(open),high=VALUES(high),low=VALUES(low),close=VALUES(close),"
           " adj_close=VALUES(adj_close),volume=VALUES(volume),dividend=VALUES(dividend),split=VALUES(split),"
           " fetched_at=VALUES(fetched_at)")
    total = 0
    for s, df in frames.items():
        rows = []
        for d, r in df.iterrows():
            def f(x):
                return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
            vol = r.get("volume")
            vol = None if (vol is None or (isinstance(vol, float) and math.isnan(vol))) else int(vol)
            rows.append((s, d.date(), f(r.get("open")), f(r.get("high")), f(r.get("low")), f(r.get("close")),
                         f(r.get("adj_close")), vol, f(r.get("dividend", 0.0)), f(r.get("split", 0.0)),
                         "yfinance", r["fetched_at"]))
        for i in range(0, len(rows), 2000):
            cur.executemany(sql, rows[i:i + 2000])
        conn.commit()
        total += len(rows)
    log(f"  yahoo_ohlcv upsert: {total} rows")
    return total


def read_yahoo(conn, symbols):
    cur = conn.cursor()
    frames = {}
    for s in symbols:
        cur.execute("SELECT date,open,high,low,close,adj_close,volume,dividend,split FROM yahoo_ohlcv "
                    "WHERE symbol=%s ORDER BY date", (s,))
        rows = cur.fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "adj_close", "volume", "dividend", "split"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").astype(float)
        frames[s] = df
    return frames

# ----------------------------------------------------------------------------
# Returns and statistics
# ----------------------------------------------------------------------------

def build_returns(df):
    """Return DataFrame with intraday, overnight_px, overnight_div, total log returns, stale-open flag."""
    d = df[["open", "close", "adj_close"]].astype(float).copy()
    d = d[(d["open"] > 0) & (d["close"] > 0) & (d["adj_close"] > 0)]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    prev_close = d["close"].shift(1)
    out = pd.DataFrame(index=d.index)
    out["intraday"] = np.log(d["close"] / d["open"])
    out["overnight_px"] = np.log(d["open"] / prev_close)
    out["total"] = np.log(d["adj_close"] / d["adj_close"].shift(1))
    out["overnight_div"] = out["total"] - out["intraday"]
    out["stale_open"] = np.isclose(d["open"], prev_close, rtol=0, atol=1e-9)
    out = out.dropna()
    return out


def newey_west_t(x, lags=None):
    """t-stat of the mean with Newey-West (Bartlett) HAC standard error."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    m = x.mean()
    e = x - m
    s = np.dot(e, e) / n
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * np.dot(e[L:], e[:-L]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    return m / se


def stats(x, ann=252):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 10:
        return dict(n=n, ann_mean=np.nan, ann_vol=np.nan, sharpe=np.nan, t_nw=np.nan, cum=np.nan, hit=np.nan)
    mu = x.mean() * ann
    vol = x.std(ddof=1) * math.sqrt(ann)
    return dict(n=n, ann_mean=mu, ann_vol=vol, sharpe=(mu / vol if vol > 0 else np.nan),
                t_nw=newey_west_t(x.values), cum=float(np.expm1(x.sum())), hit=float((x > 0).mean()))


def paired_stats(a, b):
    """Paired difference test a-b (same days)."""
    d = (pd.Series(a) - pd.Series(b)).dropna()
    return dict(diff_ann=d.mean() * 252, t_nw_diff=newey_west_t(d.values), n=len(d))


def period_table(r, label_fn):
    rows = []
    for name, (lo, hi) in label_fn.items():
        sub = r[(r.index >= lo) & (r.index <= hi)]
        if len(sub) < 60:
            continue
        s_on = stats(sub["overnight_div"]); s_in = stats(sub["intraday"]); s_px = stats(sub["overnight_px"])
        pr = paired_stats(sub["overnight_div"], sub["intraday"])
        rows.append(dict(period=name, start=sub.index.min().date(), end=sub.index.max().date(), n=len(sub),
                         on_mean=s_on["ann_mean"], on_vol=s_on["ann_vol"], on_sharpe=s_on["sharpe"], on_t=s_on["t_nw"],
                         onpx_mean=s_px["ann_mean"], onpx_t=s_px["t_nw"],
                         in_mean=s_in["ann_mean"], in_vol=s_in["ann_vol"], in_sharpe=s_in["sharpe"], in_t=s_in["t_nw"],
                         diff_ann=pr["diff_ann"], diff_t=pr["t_nw_diff"],
                         on_cum=s_on["cum"], in_cum=s_in["cum"]))
    return pd.DataFrame(rows)


def rolling_3y(r, step_months=1):
    """Rolling 3-year (756 trading day) windows, monthly step, overnight-minus-intraday paired t."""
    rows = []
    idx = r.index
    starts = pd.date_range(idx.min(), idx.max() - pd.DateOffset(years=3), freq=f"{step_months}MS")
    for s in starts:
        e = s + pd.DateOffset(years=3)
        sub = r[(idx >= s) & (idx < e)]
        if len(sub) < 600:
            continue
        rows.append(dict(window_end=e.date(), on_mean=sub["overnight_div"].mean() * 252,
                         in_mean=sub["intraday"].mean() * 252,
                         on_sharpe=stats(sub["overnight_div"])["sharpe"], in_sharpe=stats(sub["intraday"])["sharpe"],
                         diff_t=paired_stats(sub["overnight_div"], sub["intraday"])["t_nw_diff"]))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# Cost / tax layer
# ----------------------------------------------------------------------------

def cost_layer(r_overnight, group, price_series, notional, p):
    """After-cost overnight-only strategy: buy at close, sell at next open, every day.
    Round trip = 2 commissions + 2 half-spreads. Returns dict of gross/net annualised stats and break-even."""
    if group in ("us_etf",):
        hs = p["half_spread_bps_etf"]
    elif group == "ca_etf":
        hs = p["half_spread_bps_ca_etf"]
    elif group == "us_single":
        hs = p["half_spread_bps_single"]
    elif group == "ca_single":
        hs = p["half_spread_bps_ca_single"]
    else:
        hs = 0.0  # index: not tradable, cost of nearest ETF applies; report gross only
    comm_frac = 2.0 * p["commission_per_order"] / notional
    spread_frac = 2.0 * hs / 1e4
    per_day_cost = comm_frac + spread_frac
    net = r_overnight - np.log1p(per_day_cost) if per_day_cost > 0 else r_overnight
    g = stats(r_overnight); n = stats(net)
    gross_daily = r_overnight.mean()
    be_notional = (2.0 * p["commission_per_order"]) / max(gross_daily - spread_frac, 1e-12) if gross_daily > spread_frac else float("inf")
    return dict(notional=notional, half_spread_bps=hs, per_day_cost_bps=per_day_cost * 1e4,
                gross_ann=g["ann_mean"], net_ann=n["ann_mean"], net_sharpe=n["sharpe"], net_t=n["t_nw"],
                breakeven_notional=be_notional, trades_per_year=2 * 252,
                commissions_per_year=2 * 252 * p["commission_per_order"])


def tax_layer(net_ann, p):
    """TFSA: untaxed (but 252 round trips/yr is textbook 'carrying on a business' risk).
    Margin: capital gains (26.76%) or business income (53.53%)."""
    return dict(tfsa=net_ann, margin_capgains=net_ann * (1 - p["cap_gains_rate"]) if net_ann > 0 else net_ann,
                margin_business=net_ann * (1 - p["income_rate"]) if net_ann > 0 else net_ann)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download from Yahoo (default if --from-db not set)")
    ap.add_argument("--from-db", action="store_true", help="read stored yahoo_ohlcv instead of downloading")
    ap.add_argument("--from-cache", action="store_true", help="read raw CSVs from <out-dir>/raw_yahoo instead of downloading")
    ap.add_argument("--write-db", action="store_true", help="upsert raw OHLCV to the yahoo_ohlcv table and log results")
    ap.add_argument("--out-dir", required=True, help="where CSV/PNG/markdown fragments go")
    ap.add_argument("--symbols", default=None, help="comma list to restrict universe (testing)")
    ap.add_argument("--commission", type=float, default=DEFAULTS["commission_per_order"])
    ap.add_argument("--recent-start", default="2019-01-01")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "run.log"), "a")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    log(f"=== overnight_intraday_test.py run {dt.datetime.now().isoformat(timespec='seconds')} args={vars(args)}")
    p = dict(DEFAULTS); p["commission_per_order"] = args.commission
    env = load_env()
    symbols = args.symbols.split(",") if args.symbols else ALL_SYMBOLS

    raw_dir = os.path.join(args.out_dir, "raw_yahoo")
    if args.from_db:
        conn = reader_connection(env)
        frames = read_yahoo(conn, symbols); conn.close()
        log(f"read {len(frames)} symbols from yahoo_ohlcv")
    elif args.from_cache:
        frames = {}
        for s in symbols:
            fp = os.path.join(raw_dir, s.replace("^", "_idx_") + ".csv")
            if os.path.exists(fp):
                df = pd.read_csv(fp, parse_dates=["date"]).set_index("date")
                df["symbol"] = s; frames[s] = df
        log(f"read {len(frames)} symbols from {raw_dir}")
        if args.write_db:
            wconn = writer_connection(env)
            write_yahoo(wconn, frames, log)
            wconn.close()
    else:
        log("fetching from Yahoo Finance (yfinance, period=max, auto_adjust=False)")
        frames = fetch_yahoo(symbols, log)
        os.makedirs(raw_dir, exist_ok=True)
        for s, df in frames.items():
            df.to_csv(os.path.join(raw_dir, s.replace("^", "_idx_") + ".csv"))
        log(f"raw CSV cache written to {raw_dir}")
        if args.write_db:
            wconn = writer_connection(env)
            write_yahoo(wconn, frames, log)
            wconn.close()

    # ------------------------------------------------------------------ returns
    rets = {}
    stale_report = []
    for s, df in frames.items():
        r = build_returns(df)
        n_stale = int(r["stale_open"].sum())
        stale_report.append((s, len(r), n_stale))
        # Stale-open filter applies to INDEX series only (Yahoo fills index Open = prior Close before ~1982
        # and on stale-print days). For ETFs/stocks an open equal to the prior close is a legitimate zero
        # overnight return and is kept; the count is still reported.
        if s.startswith("^"):
            r = r[~r["stale_open"]]
        rets[s] = r
    pd.DataFrame(stale_report, columns=["symbol", "days", "stale_open_days"]).to_csv(
        os.path.join(args.out_dir, "stale_open_report.csv"), index=False)

    # ------------------------------------------------------------------ period tables
    recent = pd.Timestamp(args.recent_start)
    periods = {
        "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
        "1993-1999": (pd.Timestamp("1993-01-01"), pd.Timestamp("1999-12-31")),
        "2000-2009": (pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
        "2010-2018": (pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31")),
        f"{recent.year}-2026": (recent, pd.Timestamp("2100-01-01")),
        "2022-2026": (pd.Timestamp("2022-01-01"), pd.Timestamp("2100-01-01")),
    }
    all_rows = []
    rolling = {}
    for s, r in rets.items():
        t = period_table(r, periods)
        t.insert(0, "group", GROUP_OF.get(s, "?")); t.insert(0, "symbol", s)
        all_rows.append(t)
        rolling[s] = rolling_3y(r)
    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(os.path.join(args.out_dir, "period_summary.csv"), index=False)
    pd.concat([v.assign(symbol=k) for k, v in rolling.items()], ignore_index=True).to_csv(
        os.path.join(args.out_dir, "rolling_3y.csv"), index=False)

    # ------------------------------------------------------------------ group aggregates (equal-weight across names, paired)
    grp_rows = []
    for g, syms in UNIVERSE.items():
        have = [s for s in syms if s in rets]
        if not have:
            continue
        for pname, (lo, hi) in periods.items():
            panel_on = pd.concat([rets[s]["overnight_div"].rename(s) for s in have], axis=1)
            panel_in = pd.concat([rets[s]["intraday"].rename(s) for s in have], axis=1)
            panel_on = panel_on[(panel_on.index >= lo) & (panel_on.index <= hi)]
            panel_in = panel_in[(panel_in.index >= lo) & (panel_in.index <= hi)]
            # require at least half the names present each day
            ok = panel_on.notna().sum(axis=1) >= max(1, len(have) // 2)
            ew_on = panel_on[ok].mean(axis=1); ew_in = panel_in[ok].mean(axis=1)
            if len(ew_on) < 60:
                continue
            so, si = stats(ew_on), stats(ew_in); pr = paired_stats(ew_on, ew_in)
            # count of names with positive overnight-minus-intraday difference and t>2
            sub = summary[(summary.group == g) & (summary.period == pname)]
            grp_rows.append(dict(group=g, period=pname, n_names=len(have), days=len(ew_on),
                                 on_mean=so["ann_mean"], on_sharpe=so["sharpe"], on_t=so["t_nw"],
                                 in_mean=si["ann_mean"], in_sharpe=si["sharpe"], in_t=si["t_nw"],
                                 diff_ann=pr["diff_ann"], diff_t=pr["t_nw_diff"],
                                 names_diff_pos=int((sub["diff_ann"] > 0).sum()), names_diff_t_gt2=int((sub["diff_t"] > 2).sum())))
    groups = pd.DataFrame(grp_rows)
    groups.to_csv(os.path.join(args.out_dir, "group_summary.csv"), index=False)

    # ------------------------------------------------------------------ costs and taxes (recent window, overnight-only strategy)
    cost_rows = []
    for s, r in rets.items():
        sub = r[r.index >= recent]
        if len(sub) < 200:
            continue
        g = GROUP_OF.get(s, "index")
        for notional in p["positions"]:
            c = cost_layer(sub["overnight_div"], g, None, notional, p)
            tx = tax_layer(c["net_ann"], p)
            cost_rows.append(dict(symbol=s, group=g, period=f"{recent.year}-2026", **c,
                                  net_tfsa=tx["tfsa"], net_margin_capgains=tx["margin_capgains"],
                                  net_margin_business=tx["margin_business"]))
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(os.path.join(args.out_dir, "cost_tax_recent.csv"), index=False)

    # Weekday / day-of-week split for the overnight return (which nights carry it?)
    dow_rows = []
    for s, r in rets.items():
        sub = r[r.index >= recent]
        if len(sub) < 200:
            continue
        for d in range(5):
            x = sub[sub.index.dayofweek == d]["overnight_div"]
            st = stats(x)
            dow_rows.append(dict(symbol=s, group=GROUP_OF.get(s), open_weekday=["Mon", "Tue", "Wed", "Thu", "Fri"][d],
                                 n=st["n"], mean_bps=x.mean() * 1e4, t_nw=st["t_nw"]))
    pd.DataFrame(dow_rows).to_csv(os.path.join(args.out_dir, "overnight_by_weekday_recent.csv"), index=False)

    # ------------------------------------------------------------------ plots
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_syms = [s for s in ["SPY", "QQQ", "IWM", "XIU.TO", "XIC.TO", "^GSPC", "^GSPTSE", "AAPL", "NVDA", "RY.TO", "TD.TO", "SHOP.TO"] if s in rets]
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        for ax, s in zip(axes.flat, plot_syms):
            r = rets[s]
            ax.plot(r.index, np.exp(r["overnight_div"].cumsum()), color="#1f77b4", lw=1.2, label="overnight (close→open, incl. div)")
            ax.plot(r.index, np.exp(r["intraday"].cumsum()), color="#2ca02c", lw=1.2, label="intraday (open→close)")
            ax.set_yscale("log"); ax.set_title(s, fontsize=11); ax.grid(alpha=0.3)
            ax.axhline(1.0, color="grey", lw=0.6)
        axes.flat[0].legend(fontsize=8, loc="upper left")
        for ax in axes.flat[len(plot_syms):]:
            ax.axis("off")
        fig.suptitle("Value of 1 unit, overnight-only vs intraday-only (log scale, Yahoo unadjusted open/close; dividends to overnight)", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "overnight_intraday_cumulative.png"), dpi=110); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=False)
        for ax, s in zip(axes.flat, [x for x in ["SPY", "QQQ", "XIU.TO", "^GSPTSE"] if x in rolling]):
            rr = rolling[s]
            if rr.empty:
                continue
            ax.plot(pd.to_datetime(rr["window_end"]), rr["on_mean"] * 100, color="#1f77b4", label="overnight ann. mean %")
            ax.plot(pd.to_datetime(rr["window_end"]), rr["in_mean"] * 100, color="#2ca02c", label="intraday ann. mean %")
            ax2 = ax.twinx(); ax2.plot(pd.to_datetime(rr["window_end"]), rr["diff_t"], color="#d62728", lw=0.8, ls="--", label="paired NW t (on−in)")
            ax2.axhline(2, color="#d62728", lw=0.5, alpha=0.5); ax2.axhline(-2, color="#d62728", lw=0.5, alpha=0.5)
            ax.axhline(0, color="grey", lw=0.6); ax.set_title(f"{s}: rolling 3-year windows"); ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "rolling_3y.png"), dpi=110); plt.close(fig)

        # group bar chart recent window
        fig, ax = plt.subplots(figsize=(10, 5))
        gsub = groups[groups.period == f"{recent.year}-2026"]
        x = np.arange(len(gsub))
        ax.bar(x - 0.2, gsub["on_mean"] * 100, 0.4, color="#1f77b4", label="overnight")
        ax.bar(x + 0.2, gsub["in_mean"] * 100, 0.4, color="#2ca02c", label="intraday")
        ax.set_xticks(x); ax.set_xticklabels(gsub["group"]); ax.set_ylabel("annualised mean log return, %")
        ax.set_title(f"Equal-weight by group, {recent.year}-2026"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "group_recent.png"), dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ DB logging
    if args.write_db:
        wconn = writer_connection(env); cur = wconn.cursor()
        spy_recent = summary[(summary.symbol == "SPY") & (summary.period == f"{recent.year}-2026")]
        grp_recent = groups[groups.period == f"{recent.year}-2026"].to_dict("records")
        params = dict(script="research/quant-strategies/scripts/overnight_intraday_test.py", universe=UNIVERSE,
                      cost=DEFAULTS, recent_start=args.recent_start,
                      definitions="overnight_div=ln(adj_t/adj_t-1)-ln(close_t/open_t); intraday=ln(close/open); NW t-stats",
                      group_recent=grp_recent)
        for _, row in spy_recent.iterrows():
            cur.execute("INSERT INTO backtest_results (strategy_name,start_date,end_date,universe,params,total_return,annualized_return,"
                        "sharpe_ratio,max_drawdown,win_rate,total_trades,benchmark_return,benchmark_sharpe,validation_status,notes,result_file_path)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s)",
                        ("overnight_only_SPY_gross", row["start"], row["end"], ",".join(ALL_SYMBOLS), json.dumps(params, default=str)[:60000],
                         float(row["on_cum"]), float(row["on_mean"]), float(row["on_sharpe"]), None, None, int(row["n"]) * 2,
                         float(row["in_cum"]), float(row["in_sharpe"]),
                         f"Gross overnight-only SPY {recent.year}-2026 vs intraday-only (benchmark cols). Paired NW t(on-in)={row['diff_t']:.2f}. See results md.",
                         os.path.join(args.out_dir, "period_summary.csv")))
            bid = cur.lastrowid
            cur.execute("INSERT INTO research_log (hypothesis,success_criteria,test_method,result_summary,conclusion,backtest_result_id)"
                        " VALUES (%s,%s,%s,%s,%s,%s)",
                        ("Knuteson overnight/intraday anomaly: close-to-open returns dominate open-to-close returns for US/CA index ETFs, indices and large caps, and the effect persists in 2019-2026.",
                         "Paired Newey-West t(overnight - intraday) > 2 in 2019-2026 for equal-weight group panels; after retail costs (2 x $9.99 + spread per round trip) net overnight return > buy-and-hold at $100k position.",
                         "overnight_intraday_test.py", json.dumps(grp_recent, default=str)[:60000],
                         None, bid))
        wconn.commit(); wconn.close()
        log("backtest_results/research_log rows written")

    # ------------------------------------------------------------------ console summary
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    log("\n=== GROUP SUMMARY (equal-weight panels) ===")
    log(groups.round(3).to_string())
    log("\n=== PER-SYMBOL recent window ===")
    log(summary[summary.period == f"{recent.year}-2026"][["symbol", "group", "n", "on_mean", "on_t", "in_mean", "in_t", "diff_ann", "diff_t"]].round(3).to_string())
    log("\n=== COSTS (recent, $100k) ===")
    log(costs[costs.notional == 100_000][["symbol", "per_day_cost_bps", "gross_ann", "net_ann", "net_sharpe", "breakeven_notional", "net_tfsa", "net_margin_capgains"]].round(3).to_string())
    logf.close()


if __name__ == "__main__":
    main()
