"""Opt-in telemetry and community contribution hub for Terminus.

Local telemetry is always-on (SQLite only — stays on your machine).
Remote telemetry requires explicit opt-in via env var or CLI flag.

Opt-in:  TERMINUS_TELEMETRY=1  (or --telemetry flag)
Opt-out: TERMINUS_TELEMETRY=0  (default)

What is collected locally:
  - Which CLI command ran, how long it took
  - Aggregate sim counts, cache-hit rates, sims/sec
  - Survivor counts and score distributions (no strategy params)

What is sent remotely (only when opted in):
  - Same aggregate stats — no pair names, no trade data, no personal info
  - Community contributions: family name, tier (major/mid/small), tf,
    Calmar, bear-year return, year coverage — no individual trade records

Hub endpoint: https://hub.terminuslab.io/api/v1  (not yet live — stubs send to
  a local no-op until the endpoint is deployed)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger("terminus.telemetry")

HUB_BASE_URL = os.environ.get(
    "TERMINUS_HUB_URL",
    "https://hub.terminuslab.io/api/v1",
)
_REMOTE_ENABLED: bool | None = None  # lazy-resolve from env


# ---------------------------------------------------------------------------
# Tier classification (for anonymising pair names in remote payloads)
# ---------------------------------------------------------------------------

_TIER_MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"}
_TIER_MIDS = {
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT",
    "LTCUSDT", "ATOMUSDT", "NEARUSDT", "TRXUSDT",
}


def pair_tier(pair: str) -> str:
    if pair in _TIER_MAJORS:
        return "major"
    if pair in _TIER_MIDS:
        return "mid"
    return "small"


# ---------------------------------------------------------------------------
# Remote opt-in check
# ---------------------------------------------------------------------------

def remote_enabled() -> bool:
    global _REMOTE_ENABLED
    if _REMOTE_ENABLED is None:
        _REMOTE_ENABLED = os.environ.get("TERMINUS_TELEMETRY", "0").strip() == "1"
    return _REMOTE_ENABLED


def set_remote_enabled(enabled: bool) -> None:
    global _REMOTE_ENABLED
    _REMOTE_ENABLED = enabled
    os.environ["TERMINUS_TELEMETRY"] = "1" if enabled else "0"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    event_type: str
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Local recorder (SQLite via store)
# ---------------------------------------------------------------------------

def record_local(store, event: TelemetryEvent) -> None:
    """Write event to telemetry_events table. Silently skips on any error."""
    try:
        store._conn.execute(
            "INSERT INTO telemetry_events(ts, event_type, manifest_id, payload_json)"
            " VALUES (?,?,?,?)",
            (event.ts, event.event_type,
             store._current_manifest_id,
             json.dumps(event.payload, default=str)),
        )
        store._conn.commit()
    except Exception as e:
        logger.debug(f"telemetry local write skipped: {e}")


# ---------------------------------------------------------------------------
# Remote sender (stub — hub endpoint not yet live)
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict:
    """POST to hub. Returns response dict or error dict."""
    try:
        import httpx
        resp = httpx.post(
            f"{HUB_BASE_URL}/{endpoint}",
            json=payload,
            timeout=8.0,
            headers={"User-Agent": "terminus-lab/0.1.0"},
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def emit(store, event_type: str, payload: dict) -> None:
    """Record event locally; optionally send to hub if opted in."""
    ev = TelemetryEvent(event_type=event_type, payload=payload)
    record_local(store, ev)
    if remote_enabled():
        resp = _post("events", {"event_type": event_type, **payload})
        if "error" in resp:
            logger.debug(f"telemetry remote skipped: {resp['error']}")


# ---------------------------------------------------------------------------
# Named event helpers — called by sweep / walk_forward / portfolio / fetch
# ---------------------------------------------------------------------------

def sweep_complete(store, *, pairs: list[str], tfs: list[str], configs_run: int,
                   cached_hits: int, persisted: int, elapsed_sec: float) -> None:
    emit(store, "sweep_complete", {
        "n_pairs": len(pairs),
        "n_tfs": len(tfs),
        "configs_run": configs_run,
        "cached_hits": cached_hits,
        "persisted": persisted,
        "elapsed_sec": round(elapsed_sec, 1),
        "sims_per_sec": round(configs_run / max(elapsed_sec, 1), 1),
        "cache_hit_rate": round(cached_hits / max(configs_run, 1), 3),
    })


def walk_forward_complete(store, *, pair: str, timeframe: str,
                          n_configs: int, elapsed_sec: float) -> None:
    emit(store, "walk_forward_complete", {
        "pair_tier": pair_tier(pair),
        "timeframe": timeframe,
        "n_configs": n_configs,
        "elapsed_sec": round(elapsed_sec, 1),
    })


def portfolio_complete(store, *, n_legs: int, annual_ret: float,
                       max_dd: float, sharpe: float) -> None:
    emit(store, "portfolio_complete", {
        "n_legs": n_legs,
        "annual_ret": round(annual_ret, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
    })


def fetch_complete(store, *, n_pairs: int, n_tfs: int,
                   total_bars: int, elapsed_sec: float) -> None:
    emit(store, "fetch_complete", {
        "n_pairs": n_pairs,
        "n_tfs": n_tfs,
        "total_bars": total_bars,
        "elapsed_sec": round(elapsed_sec, 1),
    })


def filter_run(store, *, n_candidates: int, n_survivors: int,
               min_calmar: float) -> None:
    emit(store, "filter_run", {
        "n_candidates": n_candidates,
        "n_survivors": n_survivors,
        "min_calmar": min_calmar,
        "survival_rate": round(n_survivors / max(n_candidates, 1), 3),
    })


# ---------------------------------------------------------------------------
# Community contribution — submit anonymised strategy performance to hub
# ---------------------------------------------------------------------------

@dataclass
class ContribRecord:
    sim_hash: str
    family: str
    pair_tier: str    # 'major' | 'mid' | 'small'
    timeframe: str
    calmar: float
    total_return_pct: float
    max_drawdown_pct: float
    n_trades: int
    bear_year_pct: float | None
    years_tested: int
    years_profitable: int


def contribute_strategy(store, survivor) -> dict:
    """Submit one survivor's anonymised performance to the community hub.

    Returns the hub response dict. No-ops (returns error dict) if not opted in
    or if the hub is unreachable.
    """
    rec = ContribRecord(
        sim_hash=survivor.sim_hash,
        family=survivor.family,
        pair_tier=pair_tier(survivor.pair),
        timeframe=survivor.timeframe,
        calmar=round(survivor.calmar, 4),
        total_return_pct=round(survivor.total_return_pct, 2),
        max_drawdown_pct=round(survivor.max_drawdown_pct, 2),
        n_trades=survivor.n_trades,
        bear_year_pct=round(survivor.bear_year_pct, 2) if survivor.bear_year_pct else None,
        years_tested=survivor.years_tested,
        years_profitable=survivor.years_profitable,
    )
    payload = asdict(rec)

    # Always persist to local community_submissions table
    try:
        store._conn.execute(
            "INSERT OR IGNORE INTO community_submissions("
            "sim_hash, family, pair_tier, timeframe, calmar,"
            " total_return_pct, max_drawdown_pct, n_trades,"
            " bear_year_pct, years_tested, years_profitable, submitted_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.sim_hash, rec.family, rec.pair_tier, rec.timeframe,
             rec.calmar, rec.total_return_pct, rec.max_drawdown_pct,
             rec.n_trades, rec.bear_year_pct,
             rec.years_tested, rec.years_profitable, time.time()),
        )
        store._conn.commit()
    except Exception as e:
        logger.debug(f"community_submissions local write failed: {e}")

    if not remote_enabled():
        return {"status": "local_only", "note": "set TERMINUS_TELEMETRY=1 to share"}

    resp = _post("contribute", payload)
    if "hub_id" in resp:
        try:
            store._conn.execute(
                "UPDATE community_submissions SET hub_response_json=? WHERE sim_hash=?",
                (json.dumps(resp), rec.sim_hash),
            )
            store._conn.commit()
        except Exception:
            pass
    return resp


def contribute_batch(store, survivors: list, limit: int = 50) -> dict:
    """Submit up to `limit` survivors to the community hub."""
    submitted, errors = 0, 0
    for s in survivors[:limit]:
        r = contribute_strategy(store, s)
        if "error" in r:
            errors += 1
        else:
            submitted += 1
    return {"submitted": submitted, "errors": errors, "total": min(len(survivors), limit)}
