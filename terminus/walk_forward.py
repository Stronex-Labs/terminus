"""Real walk-forward with per-year re-optimization and frozen-params modes.

MODES:
  frozen
      Take a config from the full-window sweep. For each year y in the data,
      run that EXACT config on year y alone. The strategy passes only if
      EVERY year is net-profitable. This is the honest test — you cannot
      re-optimize parameters live.

  reopt_anchored
      Walk-forward with growing training window. For year y:
        train = [earliest .. y-1]
        best  = argmax Calmar over the configs on train
        test  = run best on year y
      Each year's winner may be different. Realistic if the operator
      re-selects parameters yearly on historical data.

  train_test_75_25
      Baseline: train on first 75%, test on last 25%. Same as v1.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from .store import ResearchStore, get_store
from .simulate import simulate_fast, slip_for


logger = logging.getLogger("walk_forward_v2")


@dataclass
class YearSlice:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    i_start: int
    i_end: int


def year_slices(df: pd.DataFrame, ts_col: str = "ts") -> list[YearSlice]:
    """Split df into calendar-year windows (UTC). Returns list in chronological order."""
    ts = df[ts_col] if ts_col in df.columns else pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )
    years = sorted(set(ts.dt.year))
    out = []
    for y in years:
        mask = ts.dt.year == y
        idx = np.where(mask)[0]
        if len(idx) < 50:  # require at least ~50 bars to count a year
            continue
        out.append(YearSlice(
            label=str(y),
            start=ts.iloc[idx[0]],
            end=ts.iloc[idx[-1]],
            i_start=int(idx[0]),
            i_end=int(idx[-1]),
        ))
    return out


def _run_slice(
    df: pd.DataFrame, rule, tp: float, stop: float,
    max_hold: int, cooldown: int,
    fee_rate: float, slip: dict, exit_method: str,
    i_start: int, i_end: int,
) -> dict:
    """Simulate on a sub-slice of df. Entry rule still sees the full df so
    its indicators (which have lookback >200) stay valid."""
    # Mask: only count entries whose index is in [i_start..i_end]
    vec = getattr(rule, "vectorized_signal", None)
    if vec is None:
        # Build a masked wrapper
        def rule_masked(i, d):
            if i < i_start or i > i_end:
                return False
            return rule(i, d)
        return simulate_fast(
            df, rule_masked, tp, stop, max_hold, cooldown,
            fee_rate=fee_rate, exit_method=exit_method, **slip,
        )
    else:
        # Build a masked VRule
        base_sig = vec(df)
        masked_sig = base_sig.copy()
        masked_sig[:i_start] = False
        masked_sig[i_end + 1:] = False

        class _MaskedVRule:
            def vectorized_signal(self, d):
                return masked_sig
            def __call__(self, i, d):
                return bool(masked_sig[i])
        return simulate_fast(
            df, _MaskedVRule(), tp, stop, max_hold, cooldown,
            fee_rate=fee_rate, exit_method=exit_method, **slip,
        )


def walk_forward_frozen(
    df: pd.DataFrame, *,
    pair: str, timeframe: str, config_name: str, family: str,
    rule, tp: float, stop: float, max_hold: int, cooldown: int,
    exit_method: str, parent_hash: str,
    fee_rate: float = 0.00075,
    store: ResearchStore | None = None,
) -> list[dict]:
    """Run the SAME config on every year; persist per-year results."""
    if store is None:
        store = get_store()
    slip = slip_for(pair)
    slices = year_slices(df)
    results = []
    for ys in slices:
        res = _run_slice(
            df, rule, tp, stop, max_hold, cooldown,
            fee_rate=fee_rate, slip=slip, exit_method=exit_method,
            i_start=ys.i_start, i_end=ys.i_end,
        )
        cfg_dict = {"tp_pct": tp, "stop_pct": stop,
                    "max_hold_bars": max_hold, "cooldown_bars": cooldown,
                    "exit_method": exit_method}
        store.put_walk_forward(
            parent_config_hash=parent_hash, pair=pair, timeframe=timeframe,
            mode="frozen", year_label=ys.label,
            date_start=ys.start.date().isoformat(),
            date_end=ys.end.date().isoformat(),
            config_name=config_name,
            config_json=json.dumps(cfg_dict),
            n_trades=res["n"],
            win_rate_pct=res["win_rate"],
            total_return_pct=res["total_net"],
            max_drawdown_pct=res["max_drawdown"],
            trades_per_month=res["trades_per_month"],
            trades_json=None,  # save space — reconstructable from parent
        )
        results.append({
            "year": ys.label,
            "n_trades": res["n"],
            "total_return_pct": res["total_net"],
            "max_drawdown_pct": res["max_drawdown"],
            "win_rate_pct": res["win_rate"],
        })
    return results


def walk_forward_reopt_anchored(
    df: pd.DataFrame, *,
    pair: str, timeframe: str, configs: list[tuple],
    fee_rate: float = 0.00075, exit_method: str = "fixed_tp_stop",
    store: ResearchStore | None = None,
    min_training_years: int = 3,
) -> list[dict]:
    """For each year y:
      train = all bars before y (>= min_training_years)
      pick best-Calmar config on train
      evaluate that config on y
    Returns list of {year, winner_config_name, train_calmar, test_return_pct}.
    """
    if store is None:
        store = get_store()
    slip = slip_for(pair)
    slices = year_slices(df)
    results = []

    for y_idx, ys in enumerate(slices):
        if y_idx < min_training_years:
            continue  # not enough history to train on
        # Training window: everything before this year
        train_end_i = slices[y_idx - 1].i_end
        train_start_i = slices[0].i_start

        # Evaluate each config on the training window; pick best Calmar
        best = None
        best_calmar = -1e9
        for (name, rule, tp, stop, hold, cd, family) in configs:
            try:
                res = _run_slice(
                    df, rule, tp, stop, hold, cd,
                    fee_rate=fee_rate, slip=slip, exit_method=exit_method,
                    i_start=train_start_i, i_end=train_end_i,
                )
            except Exception:
                continue
            if res["n"] < 10:
                continue
            cal = (res["total_net"] / res["max_drawdown"]
                   if res["max_drawdown"] > 0 else 0.0)
            if cal > best_calmar:
                best_calmar = cal
                best = (name, rule, tp, stop, hold, cd, family, res)

        if best is None:
            results.append({"year": ys.label, "winner": None,
                             "reason": "no training winner"})
            continue

        # Evaluate winner on test year
        name, rule, tp, stop, hold, cd, family, train_res = best
        test_res = _run_slice(
            df, rule, tp, stop, hold, cd,
            fee_rate=fee_rate, slip=slip, exit_method=exit_method,
            i_start=ys.i_start, i_end=ys.i_end,
        )

        # Persist
        cfg_dict = {"tp_pct": tp, "stop_pct": stop,
                    "max_hold_bars": hold, "cooldown_bars": cd,
                    "exit_method": exit_method,
                    "train_calmar": round(best_calmar, 3)}
        from .research_store import hash_config as _h
        parent_hash = _h(
            pair=pair, timeframe=timeframe, config_name=name, family=family,
            tp_pct=tp, stop_pct=stop, max_hold_bars=hold, cooldown_bars=cd,
            exit_method=exit_method, regime_filter=None,
            date_start="full", date_end="full",
            fee_rate=fee_rate,
            entry_slip=slip["entry_slip"], stop_slip=slip["stop_slip"],
            tp_slip=slip["tp_slip"], timeout_slip=slip["timeout_slip"],
        )
        store.put_walk_forward(
            parent_config_hash=parent_hash, pair=pair, timeframe=timeframe,
            mode="reopt_anchored", year_label=ys.label,
            date_start=ys.start.date().isoformat(),
            date_end=ys.end.date().isoformat(),
            config_name=name,
            config_json=json.dumps(cfg_dict),
            n_trades=test_res["n"],
            win_rate_pct=test_res["win_rate"],
            total_return_pct=test_res["total_net"],
            max_drawdown_pct=test_res["max_drawdown"],
            trades_per_month=test_res["trades_per_month"],
        )
        results.append({
            "year": ys.label,
            "winner": name,
            "train_calmar": round(best_calmar, 3),
            "train_ret": round(train_res["total_net"], 2),
            "test_ret": round(test_res["total_net"], 2),
            "test_dd": round(test_res["max_drawdown"], 2),
            "test_n": test_res["n"],
        })
    return results
