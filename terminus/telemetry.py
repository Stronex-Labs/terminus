"""Telemetry and community research hub for Terminus.

Terminus builds a shared knowledge base: every sim result, walk-forward
slice, portfolio, and engine performance metric is collected locally and
optionally contributed to the community hub at hub.terminuslab.io.

The hub is a federated backtesting database — when many users run Terminus
across different pairs and timeframes, the aggregate picture of what works
(and what doesn't) is far richer than any single user's runs.

Local telemetry:  always-on, written to SQLite (stays on your machine)
Remote telemetry: opt-in via TERMINUS_TELEMETRY=1 or `terminus contribute`

What is collected:
  Full sim results    — pair, tf, family, all params, all metrics, trades_json
  Walk-forward slices — per-year returns, DD, trades for every WF run
  Portfolio results   — legs, weights, per-year breakdown, Sharpe/Calmar/DD
  Engine perf         — sims/sec, cache hit rate, elapsed time per command

Hub endpoint: https://hub.terminuslab.io/api/v1  (stub until live)
Override:     TERMINUS_HUB_URL=http://localhost:8000/api/v1
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("terminus.telemetry")

HUB_BASE_URL = os.environ.get(
    "TERMINUS_HUB_URL",
    "https://terminus-hub.shatla-tech.workers.dev/api/v1",
)

_REMOTE_ENABLED: bool | None = None


def remote_enabled() -> bool:
    global _REMOTE_ENABLED
    if _REMOTE_ENABLED is None:
        _REMOTE_ENABLED = os.environ.get("TERMINUS_TELEMETRY", "1").strip() != "0"
    return _REMOTE_ENABLED


def set_remote_enabled(enabled: bool) -> None:
    global _REMOTE_ENABLED
    _REMOTE_ENABLED = enabled
    os.environ["TERMINUS_TELEMETRY"] = "1" if enabled else "0"


# ---------------------------------------------------------------------------
# Remote sender
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict:
    """POST JSON to hub. Returns response dict or {"error": ...} on failure."""
    try:
        import httpx
        resp = httpx.post(
            f"{HUB_BASE_URL}/{endpoint}",
            json=payload,
            timeout=15.0,
            headers={"User-Agent": "terminus-lab/0.1.0"},
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Local write helpers
# ---------------------------------------------------------------------------

def _write_event(store, event_type: str, payload: dict,
                 remote_sent: bool = False) -> None:
    try:
        store._conn.execute(
            "INSERT INTO telemetry_events"
            "(ts, event_type, manifest_id, payload_json, remote_sent)"
            " VALUES (?,?,?,?,?)",
            (time.time(), event_type, store._current_manifest_id,
             json.dumps(payload, default=str), 1 if remote_sent else 0),
        )
        store._conn.commit()
    except Exception as e:
        logger.debug(f"telemetry local write skipped: {e}")


def _mark_sent(store, event_id: int) -> None:
    try:
        store._conn.execute(
            "UPDATE telemetry_events SET remote_sent=1 WHERE id=?", (event_id,)
        )
        store._conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engine / command performance events
# ---------------------------------------------------------------------------

def sweep_complete(store, *, pairs: list[str], tfs: list[str],
                   configs_run: int, cached_hits: int, persisted: int,
                   elapsed_sec: float) -> None:
    payload = {
        "pairs": pairs,
        "tfs": tfs,
        "configs_run": configs_run,
        "cached_hits": cached_hits,
        "persisted": persisted,
        "elapsed_sec": round(elapsed_sec, 1),
        "sims_per_sec": round(configs_run / max(elapsed_sec, 1), 1),
        "cache_hit_rate": round(cached_hits / max(configs_run, 1), 3),
    }
    _write_event(store, "sweep_complete", payload)
    if remote_enabled():
        resp = _post("events/sweep", payload)
        if "error" in resp:
            logger.debug(f"telemetry remote failed: {resp['error']}")


def walk_forward_complete(store, *, pair: str, timeframe: str,
                          n_configs: int, elapsed_sec: float) -> None:
    payload = {
        "pair": pair,
        "timeframe": timeframe,
        "n_configs": n_configs,
        "elapsed_sec": round(elapsed_sec, 1),
    }
    _write_event(store, "walk_forward_complete", payload)
    if remote_enabled():
        _post("events/walk_forward", payload)


def portfolio_complete(store, *, n_legs: int, annual_ret: float,
                       max_dd: float, sharpe: float,
                       legs: list[dict] | None = None,
                       year_breakdown: dict | None = None) -> None:
    payload = {
        "n_legs": n_legs,
        "annual_ret": round(annual_ret, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "legs": legs or [],
        "year_breakdown": year_breakdown or {},
    }
    _write_event(store, "portfolio_complete", payload)
    if remote_enabled():
        _post("events/portfolio", payload)


def fetch_complete(store, *, n_pairs: int, n_tfs: int,
                   total_bars: int, elapsed_sec: float) -> None:
    payload = {
        "n_pairs": n_pairs,
        "n_tfs": n_tfs,
        "total_bars": total_bars,
        "elapsed_sec": round(elapsed_sec, 1),
    }
    _write_event(store, "fetch_complete", payload)
    if remote_enabled():
        _post("events/fetch", payload)


def filter_run(store, *, n_candidates: int, n_survivors: int,
               min_calmar: float) -> None:
    payload = {
        "n_candidates": n_candidates,
        "n_survivors": n_survivors,
        "min_calmar": min_calmar,
        "survival_rate": round(n_survivors / max(n_candidates, 1), 3),
    }
    _write_event(store, "filter_run", payload)
    if remote_enabled():
        _post("events/filter", payload)


# ---------------------------------------------------------------------------
# Full sim result contribution
# ---------------------------------------------------------------------------

def _sim_payload(sim_row: dict) -> dict:
    """Build the full contribution payload from a sims table row."""
    return {
        "hash":              sim_row.get("hash"),
        "pair":              sim_row.get("pair"),
        "timeframe":         sim_row.get("timeframe"),
        "config_name":       sim_row.get("config_name"),
        "family":            sim_row.get("family"),
        "tp_pct":            sim_row.get("tp_pct"),
        "stop_pct":          sim_row.get("stop_pct"),
        "max_hold_bars":     sim_row.get("max_hold_bars"),
        "cooldown_bars":     sim_row.get("cooldown_bars"),
        "exit_method":       sim_row.get("exit_method"),
        "regime_filter":     sim_row.get("regime_filter"),
        "date_start":        sim_row.get("date_start"),
        "date_end":          sim_row.get("date_end"),
        "fee_rate":          sim_row.get("fee_rate"),
        "entry_slip":        sim_row.get("entry_slip"),
        "stop_slip":         sim_row.get("stop_slip"),
        "tp_slip":           sim_row.get("tp_slip"),
        "timeout_slip":      sim_row.get("timeout_slip"),
        "n_trades":          sim_row.get("n_trades"),
        "win_rate_pct":      sim_row.get("win_rate_pct"),
        "avg_pnl_pct":       sim_row.get("avg_pnl_pct"),
        "net_avg_pnl_pct":   sim_row.get("net_avg_pnl_pct"),
        "total_return_pct":  sim_row.get("total_return_pct"),
        "final_balance":     sim_row.get("final_balance"),
        "max_drawdown_pct":  sim_row.get("max_drawdown_pct"),
        "trades_per_month":  sim_row.get("trades_per_month"),
        "calmar":            sim_row.get("calmar"),
        "trades_json":       sim_row.get("trades_json"),   # full trade list
        "config_json":       sim_row.get("config_json"),
    }


def _wf_payload(wf_rows: list[dict]) -> list[dict]:
    """Build walk-forward contribution payload — one entry per year slice."""
    return [
        {
            "hash":              r.get("hash"),
            "parent_config_hash": r.get("parent_config_hash"),
            "pair":              r.get("pair"),
            "timeframe":         r.get("timeframe"),
            "mode":              r.get("mode"),
            "year_label":        r.get("year_label"),
            "date_start":        r.get("date_start"),
            "date_end":          r.get("date_end"),
            "config_name":       r.get("config_name"),
            "config_json":       r.get("config_json"),
            "n_trades":          r.get("n_trades"),
            "win_rate_pct":      r.get("win_rate_pct"),
            "total_return_pct":  r.get("total_return_pct"),
            "max_drawdown_pct":  r.get("max_drawdown_pct"),
            "trades_per_month":  r.get("trades_per_month"),
            "trades_json":       r.get("trades_json"),
        }
        for r in wf_rows
    ]


def contribute_sim(store, sim_hash: str) -> dict:
    """Contribute one full sim result (+ its walk-forward slices) to the hub.

    Always writes to local community_submissions. Sends to hub if opted in.
    Returns the hub response or a local-only status dict.
    """
    sim_row = store.lookup_sim(sim_hash)
    if sim_row is None:
        return {"error": f"sim {sim_hash} not found in store"}
    sim_row = dict(sim_row)

    wf_rows = [dict(r) for r in store.get_wf_for(sim_hash, mode="frozen")]

    sim_data = _sim_payload(sim_row)
    wf_data = _wf_payload(wf_rows)

    # Always persist to local community_submissions
    try:
        store._conn.execute(
            "INSERT OR IGNORE INTO community_submissions("
            "sim_hash, family, pair_tier, timeframe, calmar,"
            " total_return_pct, max_drawdown_pct, n_trades,"
            " bear_year_pct, years_tested, years_profitable, submitted_at,"
            " full_sim_json, wf_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sim_hash,
                sim_row.get("family", ""),
                _pair_tier(sim_row.get("pair", "")),
                sim_row.get("timeframe", ""),
                sim_row.get("calmar", 0),
                sim_row.get("total_return_pct", 0),
                sim_row.get("max_drawdown_pct", 0),
                sim_row.get("n_trades", 0),
                _bear_year_from_wf(wf_rows),
                len(wf_rows),
                sum(1 for r in wf_rows if r.get("total_return_pct", 0) > 0),
                time.time(),
                json.dumps(sim_data, default=str),
                json.dumps(wf_data, default=str),
            ),
        )
        store._conn.commit()
    except Exception as e:
        logger.debug(f"community_submissions local write failed: {e}")

    if not remote_enabled():
        return {"status": "local_only"}

    payload = {"sim": sim_data, "walk_forward": wf_data}
    resp = _post("contribute/sim", payload)

    if "hub_id" in resp or resp.get("status") == "ok":
        try:
            store._conn.execute(
                "UPDATE community_submissions SET hub_response_json=?"
                " WHERE sim_hash=?",
                (json.dumps(resp), sim_hash),
            )
            store._conn.commit()
        except Exception:
            pass
    elif "error" in resp:
        logger.debug(f"hub contribute failed for {sim_hash}: {resp['error']}")

    return resp


def contribute_survivor(store, survivor) -> dict:
    """Contribute a Survivor (from filter_sims) — shorthand for contribute_sim."""
    return contribute_sim(store, survivor.sim_hash)


def contribute_batch(store, survivors: list, limit: int = 200) -> dict:
    """Contribute up to `limit` survivors to the community hub.

    Skips any sim already submitted (UNIQUE constraint on sim_hash).
    """
    submitted, skipped, errors = 0, 0, 0
    for s in survivors[:limit]:
        # Check if already submitted
        already = store._conn.execute(
            "SELECT 1 FROM community_submissions WHERE sim_hash=? AND hub_response_json IS NOT NULL",
            (s.sim_hash,),
        ).fetchone()
        if already:
            skipped += 1
            continue
        resp = contribute_survivor(store, s)
        if "error" in resp and resp.get("status") != "local_only":
            errors += 1
        else:
            submitted += 1
    return {
        "submitted": submitted,
        "skipped_already_sent": skipped,
        "errors": errors,
        "total": min(len(survivors), limit),
    }


def contribute_all_sims(store, min_calmar: float = 0.0,
                        limit: int = 10_000) -> dict:
    """Contribute every sim in the store above min_calmar to the hub.

    Useful for a first-time bulk upload of all local research results.
    Skips sims already submitted.
    """
    rows = store.query(
        "SELECT hash FROM sims WHERE calmar >= ? ORDER BY calmar DESC LIMIT ?",
        (min_calmar, limit),
    )
    submitted, skipped, errors = 0, 0, 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        h = row[0]
        already = store._conn.execute(
            "SELECT 1 FROM community_submissions WHERE sim_hash=?"
            " AND hub_response_json IS NOT NULL",
            (h,),
        ).fetchone()
        if already:
            skipped += 1
            continue
        resp = contribute_sim(store, h)
        if "error" in resp and resp.get("status") != "local_only":
            errors += 1
        else:
            submitted += 1
        if i % 100 == 0:
            logger.info(f"contribute_all_sims: {i}/{total} processed")
    return {"submitted": submitted, "skipped": skipped,
            "errors": errors, "total": total}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIER_MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"}
_TIER_MIDS = {
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT",
    "LTCUSDT", "ATOMUSDT", "NEARUSDT", "TRXUSDT",
}


def _pair_tier(pair: str) -> str:
    if pair in _TIER_MAJORS:
        return "major"
    if pair in _TIER_MIDS:
        return "mid"
    return "small"


def _bear_year_from_wf(wf_rows: list[dict],
                       bear_label: str = "2022") -> float | None:
    for r in wf_rows:
        if r.get("year_label") == bear_label:
            return r.get("total_return_pct")
    return None
