"""Persistence (docs/design.md sections 7 and 8).

Run registry (`alpha_runs`), daily IC series (`alpha_ic`), per-(alpha, market) summaries
(`backtest_results`) and hypotheses (`research_log`). Only CREATE TABLE IF NOT EXISTS and plain INSERTs
are issued; every VALUES element is a `%s` placeholder.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import costs as costs_mod
from .costs import TRADING_DAYS
from .data import ingest, universe

SCRIPT = "src/alpha101/cli.py"
CHUNK = 5000
TEXT_MAX_BYTES = 65535  # MySQL TEXT

# design.md section 8, verbatim.
ALPHA_RUNS_DDL = """CREATE TABLE IF NOT EXISTS alpha_runs (
  run_id            VARCHAR(40)  NOT NULL,   -- e.g. 20261020T0300Z-ab12cd3
  created_at        TIMESTAMP    NULL DEFAULT CURRENT_TIMESTAMP,
  git_sha           CHAR(40)     NOT NULL,
  script            VARCHAR(200) NOT NULL,
  options_json      TEXT         NOT NULL,   -- EvalOptions + basis + vwap proxy + costs
  panel_fingerprint TEXT         NOT NULL,
  prereg_sha256     CHAR(64)     NULL,       -- required when the holdout is unlocked
  holdout_unlocked  TINYINT(1)   NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

ALPHA_IC_DDL = """CREATE TABLE IF NOT EXISTS alpha_ic (
  run_id   VARCHAR(40) NOT NULL,
  alpha_id VARCHAR(40) NOT NULL,             -- alpha001..alpha101, or a user alpha name
  market   CHAR(2)     NOT NULL,
  horizon  SMALLINT    NOT NULL,
  date     DATE        NOT NULL,
  ic       DOUBLE      NULL,
  n        SMALLINT    NOT NULL,
  PRIMARY KEY (run_id, alpha_id, market, horizon, date),
  KEY idx_alpha (alpha_id, horizon, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

DDL: dict[str, str] = {"security_master": ingest.SECURITY_MASTER_DDL,
                       "universe_membership": universe.UNIVERSE_MEMBERSHIP_DDL,
                       "alpha_runs": ALPHA_RUNS_DDL, "alpha_ic": ALPHA_IC_DDL}
CONCLUSIONS = ("supported", "rejected", "inconclusive", "needs_more_data")

# NOTE: every VALUES element is a %s placeholder.
INSERT_RUN_SQL = ("INSERT INTO alpha_runs (run_id,git_sha,script,options_json,panel_fingerprint,"
                  "prereg_sha256,holdout_unlocked) VALUES (%s,%s,%s,%s,%s,%s,%s)")
INSERT_IC_SQL = ("INSERT INTO alpha_ic (run_id,alpha_id,market,horizon,date,ic,n) VALUES (%s,%s,%s,%s,%s,%s,%s)"
                 " ON DUPLICATE KEY UPDATE ic=VALUES(ic),n=VALUES(n)")
SELECT_RUN_SQL = "SELECT git_sha,holdout_unlocked FROM alpha_runs WHERE run_id=%s"
INSERT_SUMMARY_SQL = ("INSERT INTO backtest_results (strategy_name,start_date,end_date,universe,params,"
                      "annualized_return,sharpe_ratio,total_trades,validation_status,notes,"
                      "result_file_path) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
INSERT_HYPOTHESIS_SQL = ("INSERT INTO research_log (hypothesis,success_criteria,test_method,result_summary,"
                         "conclusion,backtest_result_id) VALUES (%s,%s,%s,%s,%s,%s)")


def ensure_tables(conn) -> None:
    """Create the four tables this project owns (CREATE TABLE IF NOT EXISTS only)."""
    with conn.cursor() as cur:
        for ddl in DDL.values():
            cur.execute(ddl)
    conn.commit()


def new_run_id(git_sha: str, now: datetime | None = None) -> str:
    """`<UTC yyyymmddThhmmZ>-<sha7>`. A naive `now` is interpreted as UTC."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return f"{now:%Y%m%dT%H%MZ}-{git_sha[:7]}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    """(HEAD sha, dirty) where dirty means uncommitted changes under src/ or scripts/."""
    repo = repo or _repo_root()
    sha = _git(repo, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repo, "status", "--porcelain", "--", "src", "scripts").strip())
    return sha, dirty


def _path_is_committed(path: Path, repo: Path | None = None) -> bool:
    """True iff `path` is tracked by git and has no modification and is not ignored."""
    repo = Path(repo or _repo_root())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(path)], cwd=repo,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        return False
    status = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", str(path)], cwd=repo,
                            capture_output=True, text=True)
    return status.returncode == 0 and not status.stdout.strip()


def _json(obj) -> str:
    """JSON text without NaN/Infinity: non-finite floats (Python and numpy) become null."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.bool_):  # not an np.integer, so it needs its own branch
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return float(o) if math.isfinite(float(o)) else None
        return o
    return json.dumps(clean(obj), sort_keys=True, default=str, allow_nan=False)


def _check_text(name: str, text: str) -> None:
    if len(text.encode("utf-8")) > TEXT_MAX_BYTES:
        raise ValueError(f"{name} exceeds {TEXT_MAX_BYTES} bytes (TEXT column)")


def register_run(conn, run_id, cfg, costs, panel_fps: dict, prereg_path: Path | None, *,
                 repo: Path | None = None) -> None:
    """Insert the alpha_runs row. Re-registering a run_id fails loudly (plain INSERT).

    The pre-registration file must be tracked by git and clean (not modified, not ignored).
    """
    sha, dirty = git_state(repo)
    if dirty:
        raise RuntimeError("uncommitted changes under src/ or scripts/; commit before registering a run")
    if cfg.unlock_holdout and prereg_path is None:
        raise ValueError("an unlocked holdout run requires a pre-registration file")
    prereg_sha = None
    if prereg_path is not None:
        p = Path(prereg_path).resolve()
        if not _path_is_committed(p, repo):
            raise RuntimeError(f"pre-registration file {p} is not committed (untracked, ignored, "
                               "modified, or outside the repo)")
        prereg_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if len(run_id) > 40:
        raise ValueError(f"run_id longer than 40 chars: {run_id!r}")
    options_json = _json({"run_config": dataclasses.asdict(cfg), "costs": dataclasses.asdict(costs)})
    fingerprint = _json(panel_fps)
    _check_text("options_json", options_json)
    _check_text("panel_fingerprint", fingerprint)
    with conn.cursor() as cur:
        cur.execute(INSERT_RUN_SQL, (run_id, sha, SCRIPT, options_json, fingerprint, prereg_sha,
                                     1 if cfg.unlock_holdout else 0))
    conn.commit()


def write_ic(conn, run_id, alpha_id: str, market: str, horizon: int, ic: pd.DataFrame) -> int:
    """Upsert one (alpha, market, horizon) daily IC series; returns the number of rows sent."""
    if len(alpha_id) > 40:
        raise ValueError(f"alpha_id longer than 40 chars: {alpha_id!r}")
    if str(market).lower() not in ("us", "ca"):
        raise ValueError(f"market must be 'us' or 'ca', got {market!r}")
    mk = str(market).upper()
    rows = []
    for d, ic_v, n_v in zip(ic.index, ic["ic"].to_numpy(dtype=float), ic["n"].to_numpy(dtype=float)):
        rows.append((run_id, alpha_id, mk, int(horizon), pd.Timestamp(d).date(),
                     float(ic_v) if math.isfinite(ic_v) else None,
                     int(n_v) if math.isfinite(n_v) else 0))
    with conn.cursor() as cur:
        for i in range(0, len(rows), CHUNK):
            cur.executemany(INSERT_IC_SQL, rows[i:i + CHUNK])
            conn.commit()
    return len(rows)


def _dec(x) -> float | None:
    """DECIMAL(10,4) value: rounded, or None when non-finite or outside the column range."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    r = round(x, 4)
    return None if abs(r) >= 1e6 else r


def _fmt(x, spec: str) -> str:
    x = float(x)
    return "nan" if not math.isfinite(x) else format(x, spec)


def write_summary(conn, run_id, alpha_id, market, result, cfg, costs, universe: str,
                  report_path: str | None = None) -> int:
    """Insert one backtest_results row (net-of-cost decile long-short at the primary horizon).

    The scored range is every date with a finite net return. In an unlocked run it spans in-sample and
    holdout rows; the CLI decides what to pass in `result`.

    In a locked run the end-of-in-sample liquidation tickets sit on holdout rows (NaN net), so they are
    not counted in `total_trades`, while their cost is charged on the last in-sample row.
    The registered `holdout_unlocked` flag must match `cfg.unlock_holdout`, else RuntimeError.
    """
    with conn.cursor() as cur:
        cur.execute(SELECT_RUN_SQL, (run_id,))
        found = cur.fetchall()
    if not found:
        raise RuntimeError(f"run {run_id!r} is not registered in alpha_runs; register_run first")
    git_sha, registered_unlocked = found[0][0], found[0][1]
    if bool(registered_unlocked) != bool(cfg.unlock_holdout):
        raise RuntimeError(f"run {run_id!r} was registered with holdout_unlocked={bool(registered_unlocked)} "
                           f"but cfg.unlock_holdout={bool(cfg.unlock_holdout)}")
    market = str(market).lower()
    h = cfg.primary_horizon
    net = costs_mod.apply_costs(result.deciles["long_short"], result.trades, market, costs)
    net = net[np.isfinite(net.to_numpy(dtype=float))]
    if len(net) == 0:
        raise ValueError("no scored rows: the net return series has no finite value")
    start_date, end_date = pd.Timestamp(net.index[0]).date(), pd.Timestamp(net.index[-1]).date()
    mean = float(net.mean())
    ann = mean * TRADING_DAYS
    sharpe = None
    if len(net) >= 2:
        sd = float(net.std(ddof=1))
        if sd > 0 and math.isfinite(sd):
            sharpe = mean / sd * math.sqrt(TRADING_DAYS)
    total_trades = int(result.trades["n_trades"].reindex(net.index).fillna(0).sum())

    if alpha_id.startswith("alpha") and alpha_id[5:].isdigit():
        strategy_name = f"alpha101_{int(alpha_id[5:]):03d}_{market}_h{h}"
    else:
        strategy_name = f"alpha101_{alpha_id}_{market}_h{h}"
    if len(strategy_name) > 100:
        raise ValueError(f"strategy_name longer than 100 chars: {strategy_name!r}")

    options = dataclasses.asdict(cfg.options)
    options.update({"basis": cfg.basis, "vwap_proxy": cfg.vwap_proxy, "execution": cfg.execution,
                    "horizons": cfg.horizons, "primary_horizon": cfg.primary_horizon, "start": cfg.start,
                    "holdout_start": cfg.holdout_start, "unlock_holdout": cfg.unlock_holdout})
    params = _json({"run_id": run_id, "git_sha": git_sha, "options": options,
                    "costs": dataclasses.asdict(costs), "script": SCRIPT})
    _check_text("params", params)

    s = result.summary[h]["is"]
    notes = (f"IS h={h}: mean IC {_fmt(s['mean_ic'], '.4f')}, NW t {_fmt(s['t_nw'], '.2f')}, "
             f"{int(s['n_days'])} days; net decile spread annualised {_fmt(ann, '.4f')}")

    with conn.cursor() as cur:
        cur.execute(INSERT_SUMMARY_SQL, (strategy_name, start_date, end_date, universe, params,
                                         _dec(ann), _dec(sharpe), total_trades, "pending", notes,
                                         report_path))
        rowid = cur.lastrowid
    conn.commit()
    return rowid


def write_hypothesis(conn, hypothesis: str, criteria: str, method: str, summary: dict,
                     conclusion: str, backtest_result_id: int | None) -> int:
    """Insert one research_log row; returns its id."""
    if conclusion not in CONCLUSIONS:
        raise ValueError(f"conclusion must be one of {CONCLUSIONS}, got {conclusion!r}")
    if len(method) > 100:
        raise ValueError("test_method longer than 100 chars")
    with conn.cursor() as cur:
        cur.execute(INSERT_HYPOTHESIS_SQL, (hypothesis, criteria, method,
                                            _json(summary),
                                            conclusion, backtest_result_id))
        rowid = cur.lastrowid
    conn.commit()
    return rowid
