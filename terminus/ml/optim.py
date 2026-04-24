"""Walk-forward aware parameter optimizer.

Replaces brute-force grid search with smarter random search validated
on out-of-sample yearly folds. Each fold trains on all prior years,
tests on the next year — same discipline as the frozen walk-forward.

Usage:
    from terminus.ml.optim import optimize_params

    result = optimize_params(
        df,
        rule_fn=lambda tp, stop, hold: rsi_cross(lo=30, hi=70),
        search_space={
            "tp_pct":         (2.0, 10.0),
            "stop_pct":       (1.0,  6.0),
            "max_hold_bars":  (20,  120),
            "cooldown_bars":  (3,    20),
        },
        n_iter=100,
    )
    print(result.best_params, result.cv_calmar)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("terminus.ml.optim")

MIN_FOLDS = 3          # minimum OOS folds required for a valid CV score
MIN_TRADES_PER_FOLD = 3  # folds with fewer trades are ignored in scoring


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimResult:
    best_params: dict
    cv_calmar: float               # mean Calmar across OOS folds
    cv_return: float               # mean annual return across OOS folds
    fold_results: list[dict] = field(default_factory=list)
    n_iters_run: int = 0
    n_valid_iters: int = 0         # iters with >= MIN_FOLDS informative folds

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        return (
            f"OptimResult(cv_calmar={self.cv_calmar:.2f}, "
            f"cv_return={self.cv_return:.1f}%, {p})"
        )


# ---------------------------------------------------------------------------
# Calendar-year fold splitter
# ---------------------------------------------------------------------------

def _year_folds(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return list of (train_df, test_df) pairs — one per calendar year.

    Requires df to have a datetime-like index or a 'ts' column.
    Each fold: train = all bars before that year, test = that year's bars.
    """
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True)
    else:
        ts = pd.to_datetime(df.index, utc=True)

    years = sorted(ts.dt.year.unique())
    folds = []
    for i, yr in enumerate(years):
        if i == 0:
            continue  # no training data before the first year
        train_mask = ts.dt.year < yr
        test_mask = ts.dt.year == yr
        train_df = df[train_mask.values]
        test_df = df[test_mask.values]
        if len(train_df) < 50 or len(test_df) < 20:
            continue
        folds.append((train_df, test_df))
    return folds


# ---------------------------------------------------------------------------
# Param sampler
# ---------------------------------------------------------------------------

def _sample_params(search_space: dict, rng: random.Random) -> dict:
    params = {}
    for key, spec in search_space.items():
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            params[key] = rng.randint(lo, hi)
        else:
            params[key] = round(rng.uniform(lo, hi), 4)
    return params


# ---------------------------------------------------------------------------
# Single fold evaluation
# ---------------------------------------------------------------------------

def _eval_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rule_fn: Callable,
    params: dict,
    fee_rate: float,
) -> dict | None:
    """Evaluate params on one OOS fold. Returns metrics dict or None if too few trades."""
    from ..simulate import simulate_fast, slip_for

    tp = params.get("tp_pct", 3.0)
    stop = params.get("stop_pct", 2.0)
    hold = int(params.get("max_hold_bars", 50))
    cd = int(params.get("cooldown_bars", 5))
    exit_method = params.get("exit_method", "fixed_tp_stop")

    try:
        rule = rule_fn(**{k: v for k, v in params.items()
                          if k not in ("tp_pct", "stop_pct", "max_hold_bars",
                                       "cooldown_bars", "exit_method")})
    except TypeError:
        rule = rule_fn()

    # Use full df for indicator context (signal looks back from test period)
    full_df = pd.concat([train_df, test_df])

    slip = slip_for("BTCUSDT")  # conservative estimate
    res = simulate_fast(
        full_df, rule,
        tp_pct=tp, stop_pct=stop,
        max_hold_bars=hold, cooldown_bars=cd,
        fee_rate=fee_rate,
        entry_slip=slip["entry"], stop_slip=slip["stop"],
        tp_slip=slip["tp"], timeout_slip=slip["timeout"],
        exit_method=exit_method,
        eval_window=(len(train_df), len(full_df)),
    )
    if res["n"] < MIN_TRADES_PER_FOLD:
        return None
    return {
        "n_trades": res["n"],
        "total_return_pct": res.get("total_net", 0.0),
        "max_drawdown_pct": res.get("max_drawdown", 1.0),
        "calmar": res.get("total_net", 0.0) / max(res.get("max_drawdown", 1.0), 0.01),
    }


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_params(
    df: pd.DataFrame,
    rule_fn: Callable,
    search_space: dict,
    n_iter: int = 100,
    fee_rate: float = 0.001,
    min_trades_per_fold: int = MIN_TRADES_PER_FOLD,
    seed: int = 42,
) -> OptimResult:
    """Random walk-forward parameter search.

    Args:
        df:           Full OHLCV + indicator DataFrame (precomputed).
        rule_fn:      Callable that accepts subset of search_space keys and
                      returns a VRule-compatible signal function.
        search_space: Dict of param_name → (lo, hi) bounds.
                      Integer bounds (int, int) sample ints; float bounds sample floats.
        n_iter:       Number of random parameter draws to evaluate.
        fee_rate:     Trading fee rate applied to each sim.
        seed:         RNG seed for reproducibility.

    Returns:
        OptimResult with best_params, cv_calmar, fold_results.
    """
    rng = random.Random(seed)
    folds = _year_folds(df)

    if len(folds) < MIN_FOLDS:
        raise ValueError(
            f"Need >= {MIN_FOLDS} year-folds; got {len(folds)}. "
            f"Fetch more history (--days 2920 or more)."
        )

    best_calmar = -np.inf
    best_params: dict = {}
    best_fold_results: list[dict] = []
    n_valid = 0

    logger.info(
        f"optimize_params: {n_iter} iters × {len(folds)} folds, "
        f"space={list(search_space.keys())}"
    )

    for iteration in range(n_iter):
        params = _sample_params(search_space, rng)
        fold_metrics: list[dict] = []

        for train_df, test_df in folds:
            m = _eval_fold(train_df, test_df, rule_fn, params, fee_rate)
            if m is not None:
                fold_metrics.append(m)

        if len(fold_metrics) < MIN_FOLDS:
            continue

        n_valid += 1
        mean_calmar = float(np.mean([m["calmar"] for m in fold_metrics]))

        if mean_calmar > best_calmar:
            best_calmar = mean_calmar
            best_params = params
            best_fold_results = fold_metrics

        if (iteration + 1) % 20 == 0:
            logger.info(
                f"  iter {iteration+1}/{n_iter}: best_calmar={best_calmar:.2f} "
                f"params={best_params}"
            )

    if not best_params:
        logger.warning("No valid parameter sets found — all folds had too few trades")
        return OptimResult(
            best_params={}, cv_calmar=0.0, cv_return=0.0,
            fold_results=[], n_iters_run=n_iter, n_valid_iters=n_valid,
        )

    mean_ret = float(np.mean([m["total_return_pct"] for m in best_fold_results]))
    logger.info(
        f"optimize_params done: best_calmar={best_calmar:.2f} "
        f"mean_ret={mean_ret:.1f}% params={best_params}"
    )
    return OptimResult(
        best_params=best_params,
        cv_calmar=round(best_calmar, 4),
        cv_return=round(mean_ret, 2),
        fold_results=best_fold_results,
        n_iters_run=n_iter,
        n_valid_iters=n_valid,
    )
