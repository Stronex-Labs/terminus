"""Training dataset builder for the Terminus ML screener.

Loads sim results from ResearchStore, reconstructs feature vectors
from trade entries (using cached klines + funding + F&G), and returns
X (feature matrix) and y (labels) for LightGBM training.

Label definition:
  Positive (y=1): calmar >= CALMAR_POSITIVE  (strong edge)
  Negative (y=0): calmar <= CALMAR_NEGATIVE  (no edge / losing)
  Dropped:        ambiguous middle band

The feature vector is built from the market conditions at each trade
entry point, averaged across all trades in a sim. This gives the model
a "what does the market look like when this strategy fires?" signal.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from terminus.ml.features import build_feature_vector, FEATURE_NAMES, vector_to_list

if TYPE_CHECKING:
    from terminus.store import ResearchStore

logger = logging.getLogger("terminus.ml.dataset")

CALMAR_POSITIVE = 1.5   # clear edge
CALMAR_NEGATIVE = 0.3   # clearly unprofitable (not just break-even)
MIN_TRADES = 10          # skip sims with too few entries to average over


async def _load_klines_for_sim(pair: str, timeframe: str, days: int = 500) -> pd.DataFrame | None:
    """Load cached klines for a pair/timeframe. Returns None on failure."""
    from terminus.kline_cache import KlineCache
    try:
        cache = KlineCache()
        df = await cache.load_or_fetch(pair, timeframe, days=days)
        return df
    except Exception as e:
        logger.warning(f"Could not load klines for {pair}/{timeframe}: {e}")
        return None


def _extract_entry_features(
    trades_json: str,
    klines_df: pd.DataFrame,
    sim_row: dict,
    funding_series: pd.Series | None = None,
    fng_series: pd.Series | None = None,
    btc_regime_series: pd.Series | None = None,
) -> list[float] | None:
    """Extract the mean feature vector across all entries in a sim.

    For each trade entry timestamp, look up the kline row and compute
    the feature vector. Return the mean across all trades as the sim's
    feature representation.
    """
    try:
        trades = json.loads(trades_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return None

    if not trades or len(trades) < MIN_TRADES:
        return None

    vectors: list[list[float]] = []

    for trade in trades:
        ts_ms = int(trade[0]) if isinstance(trade, list) else int(trade.get("entry_ts_ms", 0))
        if not ts_ms:
            continue

        # Find kline row closest to entry (merge_asof style)
        idx = klines_df["open_time"].searchsorted(ts_ms, side="right") - 1
        if idx < 0 or idx >= len(klines_df):
            continue
        row = klines_df.iloc[idx]

        # Funding rate at entry time
        funding_8h = 0.0
        if funding_series is not None and not funding_series.empty:
            fi = funding_series.index.searchsorted(ts_ms, side="right") - 1
            if 0 <= fi < len(funding_series):
                funding_8h = float(funding_series.iloc[fi])

        # F&G at entry time
        fng = 50
        if fng_series is not None and not fng_series.empty:
            fi = fng_series.index.searchsorted(ts_ms, side="right") - 1
            if 0 <= fi < len(fng_series):
                fng = int(fng_series.iloc[fi])

        # BTC regime at entry time
        btc_regime = 1
        if btc_regime_series is not None and not btc_regime_series.empty:
            fi = btc_regime_series.index.searchsorted(ts_ms, side="right") - 1
            if 0 <= fi < len(btc_regime_series):
                btc_regime = int(btc_regime_series.iloc[fi])

        fv = build_feature_vector(
            row,
            funding_8h=funding_8h,
            fng=fng,
            btc_regime=btc_regime,
            tp_pct=float(sim_row.get("tp_pct", 0.0)),
            stop_pct=float(sim_row.get("stop_pct", 0.0)),
            max_hold_bars=int(sim_row.get("max_hold_bars", 0)),
            ts_ms=ts_ms,
        )
        vectors.append(vector_to_list(fv))

    if not vectors:
        return None

    arr = np.array(vectors)
    return arr.mean(axis=0).tolist()


async def build_dataset(
    store: "ResearchStore",
    *,
    min_trades: int = MIN_TRADES,
    calmar_positive: float = CALMAR_POSITIVE,
    calmar_negative: float = CALMAR_NEGATIVE,
    max_sims: int = 0,  # 0 = no limit
    use_funding: bool = True,
    use_fng: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, feature_names) training arrays from the research store.

    Returns:
        X: float32 array (n_samples, n_features)
        y: int array (n_samples,) — 1=positive, 0=negative
        feature_names: list of feature name strings
    """
    from terminus.funding import load_or_fetch_funding
    from terminus.sentiment import load_or_fetch_fng
    from terminus.indicators import build_btc_regime_series

    logger.info("Loading training data from research store...")

    # Fetch all sims above negative threshold
    rows = store.query(
        "SELECT * FROM sims WHERE n_trades >= ? ORDER BY created_at DESC",
        (min_trades,),
    )
    if max_sims > 0:
        rows = rows[:max_sims]

    logger.info(f"Processing {len(rows)} sims (calmar threshold: +={calmar_positive}, -={calmar_negative})")

    # Pre-fetch shared context series
    funding_series: pd.Series | None = None
    fng_series: pd.Series | None = None
    btc_regime_series: pd.Series | None = None

    if use_funding:
        try:
            funding_series = await load_or_fetch_funding("BTCUSDT", days=500)
            logger.info(f"Loaded {len(funding_series)} funding rows")
        except Exception as e:
            logger.warning(f"Could not load funding data: {e}")

    if use_fng:
        try:
            fng_series = await load_or_fetch_fng(days=500)
            logger.info(f"Loaded {len(fng_series)} F&G rows")
        except Exception as e:
            logger.warning(f"Could not load F&G data: {e}")

    # BTC daily klines for regime series
    try:
        from terminus.kline_cache import KlineCache
        cache = KlineCache()
        btc_df = await cache.load_or_fetch("BTCUSDT", "1d", days=500)
        if btc_df is not None and not btc_df.empty:
            regime = build_btc_regime_series(btc_df)
            if hasattr(regime.index, "asi8"):
                btc_regime_series = regime
            else:
                btc_regime_series = regime
            logger.info(f"Loaded BTC regime series ({len(btc_regime_series)} rows)")
    except Exception as e:
        logger.warning(f"Could not build BTC regime series: {e}")

    # Cache klines by pair/tf to avoid redundant fetches
    klines_cache: dict[str, pd.DataFrame | None] = {}

    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    skipped = 0

    for sim in rows:
        sim = dict(sim)
        calmar = float(sim.get("calmar", 0.0))

        # Apply label gating
        if calmar >= calmar_positive:
            label = 1
        elif calmar <= calmar_negative:
            label = 0
        else:
            skipped += 1
            continue

        pair = sim["pair"]
        tf = sim["timeframe"]
        key = f"{pair}/{tf}"

        if key not in klines_cache:
            klines_cache[key] = await _load_klines_for_sim(pair, tf)

        klines_df = klines_cache[key]
        if klines_df is None or klines_df.empty:
            skipped += 1
            continue

        features = _extract_entry_features(
            sim.get("trades_json", "[]"),
            klines_df,
            sim,
            funding_series=funding_series,
            fng_series=fng_series,
            btc_regime_series=btc_regime_series,
        )
        if features is None:
            skipped += 1
            continue

        X_rows.append(features)
        y_vals.append(label)

    logger.info(
        f"Dataset: {len(X_rows)} samples "
        f"(+={sum(y_vals)}, -={len(y_vals)-sum(y_vals)}, skipped={skipped})"
    )

    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=int), FEATURE_NAMES

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_vals, dtype=int),
        FEATURE_NAMES,
    )
