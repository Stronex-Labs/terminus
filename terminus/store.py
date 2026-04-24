"""Persistent simulation store — every backtest ever run is saved here.

Design principles:
  * Content-hashed keys — same inputs never re-compute
  * Trade-level detail preserved as JSON blob (reconstructable year-by-year
    without re-simulating)
  * Append-only log — nothing is deleted, corrections go on top
  * Queryable via SQL for ad-hoc analysis without re-running anything

Schema:
  sims               — one row per (config, pair, tf, date-range, slip-model)
  walk_forward_runs  — one row per yearly slice evaluated
  sensitivity_runs   — one row per parameter-neighborhood sweep
  pair_data_meta     — tracks the kline window each pair actually has
  manifest           — run-level metadata (git sha, timestamp, CLI args)

Anything written here can be re-analyzed, re-filtered, or exported without
burning compute. That is the entire point of this module.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


# Default store lives in the user's home dir so the package is portable.
# Override with ResearchStore(db_path=...) or via TERMINUS_HOME env var.
import os
_DEFAULT_ROOT = Path(os.environ.get(
    "TERMINUS_HOME",
    str(Path.home() / ".terminus"),
))
_RESEARCH_DIR = _DEFAULT_ROOT / "research"
_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = _RESEARCH_DIR / "sims.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS sims (
    hash                TEXT PRIMARY KEY,
    pair                TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    config_name         TEXT NOT NULL,
    family              TEXT NOT NULL,
    tp_pct              REAL NOT NULL,
    stop_pct            REAL NOT NULL,
    max_hold_bars       INTEGER NOT NULL,
    cooldown_bars       INTEGER NOT NULL,
    exit_method         TEXT NOT NULL DEFAULT 'fixed_tp_stop',
    regime_filter       TEXT DEFAULT NULL,
    date_start          TEXT NOT NULL,
    date_end            TEXT NOT NULL,
    fee_rate            REAL NOT NULL,
    entry_slip          REAL NOT NULL,
    stop_slip           REAL NOT NULL,
    tp_slip             REAL NOT NULL,
    timeout_slip        REAL NOT NULL,
    n_trades            INTEGER NOT NULL,
    win_rate_pct        REAL NOT NULL,
    avg_pnl_pct         REAL NOT NULL,
    net_avg_pnl_pct     REAL NOT NULL,
    total_return_pct    REAL NOT NULL,
    final_balance       REAL NOT NULL,
    max_drawdown_pct    REAL NOT NULL,
    trades_per_month    REAL NOT NULL,
    calmar              REAL NOT NULL,
    trades_json         TEXT DEFAULT NULL,  -- [[entry_ts_ms, pnl_pct, reason], ...]
    config_json         TEXT NOT NULL,      -- full resolved config dict
    manifest_id         INTEGER NOT NULL,
    created_at          REAL NOT NULL,
    FOREIGN KEY (manifest_id) REFERENCES manifest(id)
);

CREATE INDEX IF NOT EXISTS idx_sims_pair_tf ON sims(pair, timeframe);
CREATE INDEX IF NOT EXISTS idx_sims_family ON sims(family);
CREATE INDEX IF NOT EXISTS idx_sims_calmar ON sims(calmar DESC);
CREATE INDEX IF NOT EXISTS idx_sims_return ON sims(total_return_pct DESC);
CREATE INDEX IF NOT EXISTS idx_sims_pair_family_tf ON sims(pair, family, timeframe);

CREATE TABLE IF NOT EXISTS walk_forward_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    hash                TEXT NOT NULL,       -- hash of (parent_config, year_label, mode)
    parent_config_hash  TEXT NOT NULL,       -- the full-window sim hash
    pair                TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    mode                TEXT NOT NULL,       -- 'frozen' | 'reopt_anchored' | 'reopt_rolling' | 'train_test_75_25'
    year_label          TEXT NOT NULL,       -- 'Y-1', '2022', 'train', 'test', etc
    date_start          TEXT NOT NULL,
    date_end            TEXT NOT NULL,
    config_name         TEXT NOT NULL,       -- may differ from parent if re-optimized
    config_json         TEXT NOT NULL,
    n_trades            INTEGER NOT NULL,
    win_rate_pct        REAL NOT NULL,
    total_return_pct    REAL NOT NULL,
    max_drawdown_pct    REAL NOT NULL,
    trades_per_month    REAL NOT NULL,
    trades_json         TEXT DEFAULT NULL,
    manifest_id         INTEGER NOT NULL,
    created_at          REAL NOT NULL,
    UNIQUE(hash)
);
CREATE INDEX IF NOT EXISTS idx_wf_parent ON walk_forward_runs(parent_config_hash);
CREATE INDEX IF NOT EXISTS idx_wf_pair_tf_year ON walk_forward_runs(pair, timeframe, year_label);
CREATE INDEX IF NOT EXISTS idx_wf_mode ON walk_forward_runs(mode);

CREATE TABLE IF NOT EXISTS sensitivity_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_hash         TEXT NOT NULL,
    offset_key          TEXT NOT NULL,       -- e.g. 'tp+25,stop-10'
    neighbor_hash       TEXT NOT NULL,
    total_return_pct    REAL NOT NULL,
    calmar              REAL NOT NULL,
    manifest_id         INTEGER NOT NULL,
    UNIQUE(anchor_hash, offset_key)
);
CREATE INDEX IF NOT EXISTS idx_sensitivity_anchor ON sensitivity_runs(anchor_hash);

CREATE TABLE IF NOT EXISTS pair_data_meta (
    pair                TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    date_start          TEXT NOT NULL,
    date_end            TEXT NOT NULL,
    n_bars              INTEGER NOT NULL,
    years_covered       REAL NOT NULL,
    cache_path          TEXT NOT NULL,
    fetched_at          REAL NOT NULL,
    PRIMARY KEY (pair, timeframe)
);

CREATE TABLE IF NOT EXISTS manifest (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          REAL NOT NULL,
    ended_at            REAL DEFAULT NULL,
    git_sha             TEXT DEFAULT NULL,
    git_dirty           INTEGER DEFAULT 0,
    phase               TEXT NOT NULL,       -- 'full_sweep' | 'walk_forward' | 'sensitivity' | 'portfolio' | ...
    label               TEXT DEFAULT NULL,   -- free-text human label
    cli_args_json       TEXT DEFAULT NULL,
    summary_json        TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id         INTEGER NOT NULL,
    ts                  REAL NOT NULL,
    level               TEXT NOT NULL,       -- INFO | WARN | ERROR
    message             TEXT NOT NULL,
    context_json        TEXT DEFAULT NULL,
    FOREIGN KEY (manifest_id) REFERENCES manifest(id)
);
CREATE INDEX IF NOT EXISTS idx_log_manifest ON log(manifest_id);

CREATE TABLE IF NOT EXISTS portfolio_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    members_json        TEXT NOT NULL,       -- list of sim hashes + allocation weights
    weighted_annual_ret REAL NOT NULL,
    weighted_max_dd     REAL NOT NULL,
    correlation_json    TEXT DEFAULT NULL,
    bear_year_return    REAL DEFAULT NULL,
    years_profitable    INTEGER DEFAULT NULL,
    years_tested        INTEGER DEFAULT NULL,
    manifest_id         INTEGER NOT NULL,
    created_at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  REAL NOT NULL,
    event_type          TEXT NOT NULL,
    manifest_id         INTEGER DEFAULT NULL,
    payload_json        TEXT DEFAULT NULL,
    remote_sent         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_telemetry_type ON telemetry_events(event_type);

CREATE TABLE IF NOT EXISTS community_submissions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_hash            TEXT NOT NULL,
    family              TEXT NOT NULL,
    pair_tier           TEXT NOT NULL,       -- 'major' | 'mid' | 'small'
    timeframe           TEXT NOT NULL,
    calmar              REAL NOT NULL,
    total_return_pct    REAL NOT NULL,
    max_drawdown_pct    REAL NOT NULL,
    n_trades            INTEGER NOT NULL,
    bear_year_pct       REAL DEFAULT NULL,
    years_tested        INTEGER NOT NULL,
    years_profitable    INTEGER NOT NULL,
    submitted_at        REAL NOT NULL,
    hub_response_json   TEXT DEFAULT NULL,
    UNIQUE(sim_hash)
);
"""


def _git_info(cwd: Path | None = None) -> tuple[str | None, bool]:
    try:
        cwd_str = str(cwd) if cwd else None
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd_str,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd_str,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return sha, dirty
    except Exception:
        return None, False


def hash_config(
    pair: str, timeframe: str, config_name: str, family: str,
    tp_pct: float, stop_pct: float,
    max_hold_bars: int, cooldown_bars: int,
    exit_method: str, regime_filter: str | None,
    date_start: str, date_end: str,
    fee_rate: float, entry_slip: float, stop_slip: float,
    tp_slip: float, timeout_slip: float,
    extra: dict | None = None,
) -> str:
    """Deterministic content hash over every input that affects output."""
    payload = {
        "pair": pair, "tf": timeframe, "cfg": config_name, "fam": family,
        "tp": round(tp_pct, 6), "stop": round(stop_pct, 6),
        "hold": max_hold_bars, "cd": cooldown_bars,
        "exit": exit_method, "regime": regime_filter or "",
        "d0": date_start, "d1": date_end,
        "fee": round(fee_rate, 6), "es": round(entry_slip, 6),
        "ss": round(stop_slip, 6), "tps": round(tp_slip, 6),
        "os": round(timeout_slip, 6),
        "x": extra or {},
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


@dataclass
class SimRecord:
    hash: str
    pair: str
    timeframe: str
    config_name: str
    family: str
    tp_pct: float
    stop_pct: float
    max_hold_bars: int
    cooldown_bars: int
    exit_method: str
    regime_filter: str | None
    date_start: str
    date_end: str
    fee_rate: float
    entry_slip: float
    stop_slip: float
    tp_slip: float
    timeout_slip: float
    n_trades: int
    win_rate_pct: float
    avg_pnl_pct: float
    net_avg_pnl_pct: float
    total_return_pct: float
    final_balance: float
    max_drawdown_pct: float
    trades_per_month: float
    calmar: float
    trades_json: str | None
    config_json: str
    manifest_id: int
    created_at: float = field(default_factory=time.time)


class ResearchStore:
    """SQLite-backed persistent simulation store."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._current_manifest_id: int | None = None

    # --- Manifest / run tracking ---------------------------------------

    def open_manifest(
        self, phase: str, label: str | None = None,
        cli_args: dict | None = None,
    ) -> int:
        sha, dirty = _git_info(self.db_path.parent.parent.parent.parent)
        cur = self._conn.execute(
            "INSERT INTO manifest(started_at, git_sha, git_dirty, phase, label, cli_args_json)"
            " VALUES(?,?,?,?,?,?)",
            (time.time(), sha, 1 if dirty else 0, phase, label,
             json.dumps(cli_args or {}, default=str)),
        )
        self._conn.commit()
        self._current_manifest_id = cur.lastrowid
        self.log("INFO", f"manifest opened: phase={phase} label={label}")
        return cur.lastrowid

    def close_manifest(self, summary: dict | None = None) -> None:
        if self._current_manifest_id is None:
            return
        self._conn.execute(
            "UPDATE manifest SET ended_at=?, summary_json=? WHERE id=?",
            (time.time(), json.dumps(summary or {}, default=str),
             self._current_manifest_id),
        )
        self._conn.commit()
        self._current_manifest_id = None

    @property
    def manifest_id(self) -> int:
        if self._current_manifest_id is None:
            self.open_manifest("ad_hoc", "no-explicit-manifest")
        return self._current_manifest_id  # type: ignore

    def log(self, level: str, message: str, context: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO log(manifest_id, ts, level, message, context_json)"
            " VALUES(?,?,?,?,?)",
            (self.manifest_id, time.time(), level, message,
             json.dumps(context or {}, default=str)),
        )
        self._conn.commit()

    # --- Sim lookup / insert -------------------------------------------

    def lookup_sim(self, h: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM sims WHERE hash=?", (h,)
        ).fetchone()

    def have_sim(self, h: str) -> bool:
        return self.lookup_sim(h) is not None

    def put_sim(self, rec: SimRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sims("
            "hash, pair, timeframe, config_name, family, tp_pct, stop_pct,"
            " max_hold_bars, cooldown_bars, exit_method, regime_filter,"
            " date_start, date_end, fee_rate, entry_slip, stop_slip, tp_slip,"
            " timeout_slip, n_trades, win_rate_pct, avg_pnl_pct,"
            " net_avg_pnl_pct, total_return_pct, final_balance,"
            " max_drawdown_pct, trades_per_month, calmar, trades_json,"
            " config_json, manifest_id, created_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.hash, rec.pair, rec.timeframe, rec.config_name, rec.family,
             rec.tp_pct, rec.stop_pct, rec.max_hold_bars, rec.cooldown_bars,
             rec.exit_method, rec.regime_filter, rec.date_start, rec.date_end,
             rec.fee_rate, rec.entry_slip, rec.stop_slip, rec.tp_slip,
             rec.timeout_slip, rec.n_trades, rec.win_rate_pct, rec.avg_pnl_pct,
             rec.net_avg_pnl_pct, rec.total_return_pct, rec.final_balance,
             rec.max_drawdown_pct, rec.trades_per_month, rec.calmar,
             rec.trades_json, rec.config_json, rec.manifest_id, rec.created_at),
        )
        self._conn.commit()

    # --- Queries --------------------------------------------------------

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params).fetchall())

    def top_n_by_calmar(
        self, pair: str, timeframe: str | None = None,
        family: str | None = None, limit: int = 20,
    ) -> list[sqlite3.Row]:
        sql = ["SELECT * FROM sims WHERE pair=?"]
        args: list[Any] = [pair]
        if timeframe:
            sql.append("AND timeframe=?")
            args.append(timeframe)
        if family:
            sql.append("AND family=?")
            args.append(family)
        sql.append("ORDER BY calmar DESC LIMIT ?")
        args.append(limit)
        return self.query(" ".join(sql), args)

    def all_pairs(self) -> list[str]:
        return [r[0] for r in self.query(
            "SELECT DISTINCT pair FROM sims ORDER BY pair"
        )]

    def all_families(self) -> list[str]:
        return [r[0] for r in self.query(
            "SELECT DISTINCT family FROM sims ORDER BY family"
        )]

    def sim_count(self) -> int:
        return self.query("SELECT COUNT(*) FROM sims")[0][0]

    # --- Walk-forward ---------------------------------------------------

    def put_walk_forward(
        self, *, parent_config_hash: str, pair: str, timeframe: str,
        mode: str, year_label: str, date_start: str, date_end: str,
        config_name: str, config_json: str,
        n_trades: int, win_rate_pct: float,
        total_return_pct: float, max_drawdown_pct: float,
        trades_per_month: float, trades_json: str | None = None,
    ) -> str:
        h = hashlib.sha256(
            f"{parent_config_hash}|{mode}|{year_label}|{date_start}|{date_end}|{config_name}"
            .encode()
        ).hexdigest()[:24]
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO walk_forward_runs("
                "hash, parent_config_hash, pair, timeframe, mode, year_label,"
                " date_start, date_end, config_name, config_json, n_trades,"
                " win_rate_pct, total_return_pct, max_drawdown_pct,"
                " trades_per_month, trades_json, manifest_id, created_at"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (h, parent_config_hash, pair, timeframe, mode, year_label,
                 date_start, date_end, config_name, config_json, n_trades,
                 win_rate_pct, total_return_pct, max_drawdown_pct,
                 trades_per_month, trades_json, self.manifest_id, time.time()),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass
        return h

    def get_wf_for(self, parent_hash: str, mode: str = "frozen") -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM walk_forward_runs WHERE parent_config_hash=? AND mode=?"
            " ORDER BY year_label",
            (parent_hash, mode),
        )

    # --- Pair-data meta -------------------------------------------------

    def record_pair_data(
        self, pair: str, timeframe: str, date_start: str, date_end: str,
        n_bars: int, years_covered: float, cache_path: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pair_data_meta("
            "pair, timeframe, date_start, date_end, n_bars, years_covered,"
            " cache_path, fetched_at"
            ") VALUES (?,?,?,?,?,?,?,?)",
            (pair, timeframe, date_start, date_end, n_bars, years_covered,
             cache_path, time.time()),
        )
        self._conn.commit()

    # --- Portfolio candidates ------------------------------------------

    def put_portfolio(
        self, name: str, members: list[dict],
        weighted_annual_ret: float, weighted_max_dd: float,
        correlation: dict | None = None,
        bear_year_return: float | None = None,
        years_profitable: int | None = None,
        years_tested: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO portfolio_candidates("
            "name, members_json, weighted_annual_ret, weighted_max_dd,"
            " correlation_json, bear_year_return, years_profitable,"
            " years_tested, manifest_id, created_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, json.dumps(members, default=str), weighted_annual_ret,
             weighted_max_dd,
             json.dumps(correlation, default=str) if correlation else None,
             bear_year_return, years_profitable, years_tested,
             self.manifest_id, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    # --- Lifecycle ------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @contextmanager
    def manifest_scope(
        self, phase: str, label: str | None = None,
        cli_args: dict | None = None,
    ):
        mid = self.open_manifest(phase, label, cli_args)
        try:
            yield mid
        finally:
            self.close_manifest()


# --- Convenience wrapper ---------------------------------------------

_STORE: ResearchStore | None = None


def get_store(db_path: Path | str = DEFAULT_DB_PATH) -> ResearchStore:
    global _STORE
    if _STORE is None or Path(db_path) != _STORE.db_path:
        _STORE = ResearchStore(db_path)
    return _STORE
