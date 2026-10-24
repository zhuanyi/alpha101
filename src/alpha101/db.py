"""MySQL connections. Credentials are read from the environment and never printed; no host is baked in.

Reader: ALPHA101_DB_HOST, ALPHA101_DB_USER, ALPHA101_DB_PASSWORD (optional ALPHA101_DB_PORT).
Writer: ALPHA101_DB_WRITE_USER, ALPHA101_DB_WRITE_PASSWORD (same host and port as the reader).
Every connection is pinned to the schema named by ALPHA101_DB_NAME (default "alpha101"), and
`assert_database` re-checks it before any write, so a writer account that also has rights on other
schemas cannot touch them through this package. See `config.example`.
"""
from __future__ import annotations

import os

DATABASE = os.environ.get("ALPHA101_DB_NAME", "alpha101")


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set. Refusing to guess (see config.example).")
    return v


def _connect(user_var: str, pwd_var: str):
    import pymysql
    return pymysql.connect(host=_require("ALPHA101_DB_HOST"), port=int(os.environ.get("ALPHA101_DB_PORT") or 3306),
                           user=_require(user_var), password=_require(pwd_var), database=DATABASE,
                           charset="utf8mb4", connect_timeout=20)


def reader():
    return _connect("ALPHA101_DB_USER", "ALPHA101_DB_PASSWORD")


def writer():
    conn = _connect("ALPHA101_DB_WRITE_USER", "ALPHA101_DB_WRITE_PASSWORD")
    assert_database(conn)
    return conn


def assert_database(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE()")
        db = cur.fetchone()[0]
    if db != DATABASE:
        raise RuntimeError(f"connected to {db!r}, expected {DATABASE!r}; refusing to write")
