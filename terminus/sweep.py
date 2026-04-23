"""Full-sweep runner — iterates pairs × TFs × configs and persists every sim.

Content-hash cache means re-running is instant for anything already stored.
BTC daily regime is computed once and applied as a wrapper.

Usage:
  from lib.sweep_runner import run_full_sweep
  run_full_sweep(pairs=['BTCUSDT', 'ETHUSDT', ...], tfs=[...],
                 exit_methods=['fixed_tp_stop', 'breakeven_after_1r'])
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import rules as rv
from .registry import (
    build_all_configs, build_configs_with_regime, all_exit_methods,
)
from .store import ResearchStore, SimRecord, hash_config, get_store
from .simulate import simulate_fast, slip_for, tier_of
from .indicators import precompute_all, precompute_v2, build_btc_regime_series


logger = logging.getLogger("sweep_runner")


DEFAULT_CACHE_DIR = Path.home() / ".terminus" / "cache"


def _load_and_precompute(pair: str, tf: str, days: int,
                         cache_dir: Path | None = None) -> pd.DataFrame | None:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = cache_dir / f"{pair}_{tf}_{days}d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) < 400:
        return None
    return precompute_v2(precompute_all(df))


def _build_btc_regime(days: int = 2920,
                      cache_dir: Path | None = None) -> pd.Series | None:
    df = _load_and_precompute("BTCUSDT", "1d", days, cache_dir=cache_dir)
    if df is None:
        logger.warning("BTCUSDT_1d not cached - BTC-regime wrapping disabled")
        return None
    return build_btc_regime_series(df)


def _persist_one(
    store: ResearchStore, *,
    pair: str, timeframe: str, name: str, family: str,
    tp: float, stop: float, hold: int, cd: int,
    exit_method: str, regime_filter: str | None,
    date_start: str, date_end: str,
    fee_rate: float, slip: dict,
    sim_result: dict,
    full_config: dict,
) -> str:
    h = hash_config(
        pair=pair, timeframe=timeframe, config_name=name, family=family,
        tp_pct=tp, stop_pct=stop, max_hold_bars=hold, cooldown_bars=cd,
        exit_method=exit_method, regime_filter=regime_filter,
        date_start=date_start, date_end=date_end,
        fee_rate=fee_rate, entry_slip=slip["entry_slip"],
        stop_slip=slip["stop_slip"], tp_slip=slip["tp_slip"],
        timeout_slip=slip["timeout_slip"],
    )
    # trades compressed to tuples to save space
    trades_trimmed = [
        [t["entry_ts"], t["exit_ts"],
         round(t["pnl_pct"], 6), t["exit_reason"]]
        for t in sim_result.get("trades", [])
    ]
    calmar = (sim_result["total_net"] / sim_result["max_drawdown"]
              if sim_result["max_drawdown"] > 0 else 0.0)
    rec = SimRecord(
        hash=h, pair=pair, timeframe=timeframe, config_name=name, family=family,
        tp_pct=tp, stop_pct=stop, max_hold_bars=hold, cooldown_bars=cd,
        exit_method=exit_method, regime_filter=regime_filter,
        date_start=date_start, date_end=date_end,
        fee_rate=fee_rate, entry_slip=slip["entry_slip"],
        stop_slip=slip["stop_slip"], tp_slip=slip["tp_slip"],
        timeout_slip=slip["timeout_slip"],
        n_trades=sim_result["n"], win_rate_pct=sim_result["win_rate"],
        avg_pnl_pct=sim_result["avg_pnl"],
        net_avg_pnl_pct=sim_result["net_avg_pnl"],
        total_return_pct=sim_result["total_net"],
        final_balance=sim_result["final_balance"],
        max_drawdown_pct=sim_result["max_drawdown"],
        trades_per_month=sim_result["trades_per_month"],
        calmar=calmar,
        trades_json=json.dumps(trades_trimmed) if trades_trimmed else None,
        config_json=json.dumps(full_config),
        manifest_id=store.manifest_id,
    )
    store.put_sim(rec)
    return h


def run_full_sweep(
    pairs: Iterable[str], tfs: Iterable[str], *,
    days: int = 2920,
    exit_methods: list[str] | None = None,
    include_regime_wrap: bool = True,
    fee_rate: float = 0.00075,
    min_bars_required: int = 400,
    min_trades_to_persist: int = 3,
    label: str | None = None,
    progress_every: int = 500,
    store_path: Path | str | None = None,
) -> dict:
    """Run the full research sweep. Returns a summary dict."""
    pairs = list(pairs)
    tfs = list(tfs)
    if exit_methods is None:
        exit_methods = all_exit_methods()
    store = get_store(store_path) if store_path else get_store()

    base_configs = build_all_configs()
    btc_regime = _build_btc_regime(days) if include_regime_wrap else None
    if include_regime_wrap and btc_regime is not None:
        regime_configs = build_configs_with_regime(base_configs, btc_regime)
    else:
        regime_configs = []
    all_configs = base_configs + regime_configs

    total_est = len(pairs) * len(tfs) * len(all_configs) * len(exit_methods)
    logger.info(f"Sweep scope: {len(pairs)} pairs × {len(tfs)} TFs × "
                f"{len(all_configs)} configs × {len(exit_methods)} exit-methods "
                f"= {total_est:,} sims")

    with store.manifest_scope(
        phase="full_sweep",
        label=label or f"sweep-{len(pairs)}p-{len(tfs)}tf-{days}d",
        cli_args={"pairs": pairs, "tfs": tfs, "days": days,
                  "exit_methods": exit_methods,
                  "include_regime_wrap": include_regime_wrap,
                  "fee_rate": fee_rate},
    ):
        completed = 0
        cached_hits = 0
        persisted = 0
        skipped_thin = 0
        errors = 0
        t_start = time.time()

        for pair in pairs:
            slip = slip_for(pair)
            for tf in tfs:
                pdf = _load_and_precompute(pair, tf, days)
                if pdf is None or len(pdf) < min_bars_required:
                    logger.info(f"{pair} {tf}: insufficient data, skipping")
                    continue

                first_ts = pd.Timestamp(int(pdf["open_time"].iloc[0]),
                                         unit="ms", tz="UTC")
                last_ts = pd.Timestamp(int(pdf["open_time"].iloc[-1]),
                                        unit="ms", tz="UTC")
                date_start = first_ts.date().isoformat()
                date_end = last_ts.date().isoformat()

                for cfg in all_configs:
                    name, rule, tp, stop, hold, cd, family = cfg
                    regime_filter = ("btc_daily_ema50_gt_200"
                                      if "+BTCreg" in family else None)
                    for exit_method in exit_methods:
                        # content-hash cache check
                        h = hash_config(
                            pair=pair, timeframe=tf, config_name=name,
                            family=family, tp_pct=tp, stop_pct=stop,
                            max_hold_bars=hold, cooldown_bars=cd,
                            exit_method=exit_method,
                            regime_filter=regime_filter,
                            date_start=date_start, date_end=date_end,
                            fee_rate=fee_rate,
                            entry_slip=slip["entry_slip"],
                            stop_slip=slip["stop_slip"],
                            tp_slip=slip["tp_slip"],
                            timeout_slip=slip["timeout_slip"],
                        )
                        if store.have_sim(h):
                            cached_hits += 1
                            completed += 1
                            continue
                        try:
                            res = simulate_fast(
                                pdf, rule, tp, stop, hold, cd,
                                fee_rate=fee_rate,
                                exit_method=exit_method, **slip,
                            )
                        except Exception as e:
                            errors += 1
                            store.log("ERROR", f"sim failed {pair} {tf} {name}",
                                      {"err": str(e)})
                            completed += 1
                            continue

                        # Only persist if enough trades — thin configs waste space
                        if res["n"] < min_trades_to_persist:
                            skipped_thin += 1
                            completed += 1
                            continue

                        full_cfg = {
                            "tp_pct": tp, "stop_pct": stop,
                            "max_hold_bars": hold, "cooldown_bars": cd,
                            "exit_method": exit_method,
                            "fee_rate": fee_rate, "slip": slip,
                            "regime_filter": regime_filter,
                            "tier": tier_of(pair),
                        }
                        _persist_one(
                            store, pair=pair, timeframe=tf, name=name,
                            family=family, tp=tp, stop=stop, hold=hold, cd=cd,
                            exit_method=exit_method,
                            regime_filter=regime_filter,
                            date_start=date_start, date_end=date_end,
                            fee_rate=fee_rate, slip=slip,
                            sim_result=res, full_config=full_cfg,
                        )
                        persisted += 1
                        completed += 1

                        if completed % progress_every == 0:
                            elapsed = time.time() - t_start
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total_est - completed) / rate if rate > 0 else 0
                            logger.info(
                                f"[{completed:>7,}/{total_est:,}] "
                                f"{pair} {tf}  "
                                f"cached={cached_hits:,} "
                                f"persisted={persisted:,} "
                                f"thin={skipped_thin:,} "
                                f"err={errors}  "
                                f"{rate:.0f} sims/s  "
                                f"eta {eta/60:.1f}m"
                            )

        summary = {
            "total_est": total_est,
            "completed": completed,
            "cached_hits": cached_hits,
            "persisted": persisted,
            "skipped_thin": skipped_thin,
            "errors": errors,
            "elapsed_sec": round(time.time() - t_start, 1),
        }
        logger.info(f"Sweep done: {summary}")
        store.log("INFO", "sweep summary", summary)
        return summary
