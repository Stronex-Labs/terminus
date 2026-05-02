"""Feature vector builder for the Terminus ML screener.

build_feature_vector() extracts a fixed-length numeric feature vector
from a market snapshot (kline row + context). This same function must be
called identically in both:

  - Terminus training pipeline (train.py) — to label historical entries
  - Stronex live screener (ml_screener.py) — to score current pairs

IMPORTANT: Any change to this function breaks the model. Version the
feature spec alongside the model file.

Feature schema (v1, 18 features)
---------------------------------
Technical (11):
  rsi, adx, atr_pct, ema20_50_ratio, ema50_200_ratio, bb_width,
  vol_ratio, price_vs_ema200, cci, macd_hist_norm, roc10

Context (4):
  funding_8h, fng_norm, btc_regime, hour_of_day

Strategy config (3) — only used in training, zeroed in live scoring:
  tp_pct, stop_pct, max_hold_bars_norm
"""

from __future__ import annotations

import math
from typing import Any

# Feature names in exact order — must match training and inference
FEATURE_NAMES: list[str] = [
    # Technical
    "rsi",
    "adx",
    "atr_pct",
    "ema20_50_ratio",
    "ema50_200_ratio",
    "bb_width",
    "vol_ratio",
    "price_vs_ema200",
    "cci",
    "macd_hist_norm",
    "roc10",
    # Context
    "funding_8h",
    "fng_norm",
    "btc_regime",
    "hour_of_day_sin",
    "hour_of_day_cos",
    # Strategy config (zeroed in live scoring)
    "tp_pct",
    "stop_pct",
    "max_hold_bars_norm",
]

FEATURE_VERSION = "v1"


def _safe(val: Any, default: float = 0.0) -> float:
    """Return float or default if None/NaN/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_feature_vector(
    row: "pd.Series | dict",
    *,
    funding_8h: float = 0.0,
    fng: int = 50,
    btc_regime: int = 1,
    tp_pct: float = 0.0,
    stop_pct: float = 0.0,
    max_hold_bars: int = 0,
    ts_ms: int | None = None,
) -> dict[str, float]:
    """Build a feature vector from a kline row and context.

    Args:
        row:           Dict or pd.Series with indicator columns (from
                       terminus.indicators.precompute_all output).
                       Expected columns: close, rsi, adx, atr, ema20,
                       ema50, ema200, bb_upper, bb_lower, vol_ratio,
                       cci (optional), macd_h (optional), roc10 (optional).
        funding_8h:    BTC 8h funding rate (fraction, e.g. -0.001).
        fng:           Fear & Greed index 0-100.
        btc_regime:    1 if BTC EMA50 > EMA200, else 0.
        tp_pct:        Strategy take-profit % (training only; 0 in live).
        stop_pct:      Strategy stop-loss % (training only; 0 in live).
        max_hold_bars: Strategy max hold bars (training only; 0 in live).
        ts_ms:         Entry timestamp in UTC milliseconds (for time features).

    Returns:
        Ordered dict matching FEATURE_NAMES exactly.
    """
    close = _safe(row.get("close") if hasattr(row, "get") else row["close"], 1.0)
    ema20  = _safe(row.get("ema20",  close) if hasattr(row, "get") else row.get("ema20",  close))
    ema50  = _safe(row.get("ema50",  close) if hasattr(row, "get") else row.get("ema50",  close))
    ema200 = _safe(row.get("ema200", close) if hasattr(row, "get") else row.get("ema200", close))

    # Ratios (close to 1.0 = near equal)
    ema20_50   = (ema20  / ema50)  - 1.0 if ema50  > 0 else 0.0
    ema50_200  = (ema50  / ema200) - 1.0 if ema200 > 0 else 0.0
    price_ema200 = (close / ema200) - 1.0 if ema200 > 0 else 0.0

    # ATR as % of price
    atr = _safe(row.get("atr", 0) if hasattr(row, "get") else row.get("atr", 0))
    atr_pct = atr / close if close > 0 else 0.0

    # Bollinger band width as % of midband
    bb_upper = _safe(row.get("bb_upper", close) if hasattr(row, "get") else row.get("bb_upper", close))
    bb_lower = _safe(row.get("bb_lower", close) if hasattr(row, "get") else row.get("bb_lower", close))
    bb_mid   = (bb_upper + bb_lower) / 2 if (bb_upper + bb_lower) > 0 else close
    bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0.0

    # Time features (cyclic encoding of hour of day)
    hour = 12  # default midday
    if ts_ms is not None:
        try:
            import datetime
            dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
            hour = dt.hour
        except Exception:
            pass
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    def _get(key: str, default: float = 0.0) -> float:
        if hasattr(row, "get"):
            return _safe(row.get(key, default), default)
        try:
            return _safe(row[key], default)
        except (KeyError, IndexError):
            return default

    return {
        "rsi":              _safe(_get("rsi", 50.0)),
        "adx":              _safe(_get("adx", 20.0)),
        "atr_pct":          _safe(atr_pct),
        "ema20_50_ratio":   _safe(ema20_50),
        "ema50_200_ratio":  _safe(ema50_200),
        "bb_width":         _safe(bb_width),
        "vol_ratio":        _safe(_get("vol_ratio", 1.0)),
        "price_vs_ema200":  _safe(price_ema200),
        "cci":              _safe(_get("cci", 0.0)),
        "macd_hist_norm":   _safe(_get("macd_h", 0.0)),
        "roc10":            _safe(_get("roc10", 0.0)),
        "funding_8h":       _safe(funding_8h),
        "fng_norm":         _safe((fng - 50) / 50),   # normalize to [-1, 1]
        "btc_regime":       float(int(btc_regime)),
        "hour_of_day_sin":  _safe(hour_sin),
        "hour_of_day_cos":  _safe(hour_cos),
        "tp_pct":           _safe(tp_pct),
        "stop_pct":         _safe(stop_pct),
        "max_hold_bars_norm": _safe(max_hold_bars / 100.0),
    }


def vector_to_list(fv: dict[str, float]) -> list[float]:
    """Convert feature dict to ordered list matching FEATURE_NAMES."""
    return [fv[name] for name in FEATURE_NAMES]
