"""Regime classifier — labels each bar as bull / bear / chop.

Replaces the simple BTC EMA50 > EMA200 binary filter with a 3-class
XGBoost model trained on rolling features. The model is trained
walk-forward style: always on prior data only, never on future bars.

Label generation (auto, from realized forward returns):
  bull  — forward 20-bar return >= +5%
  bear  — forward 20-bar return <= -5%
  chop  — otherwise

Features (all from the input OHLCV DataFrame):
  ret_5, ret_20, ret_60          — rolling log returns
  rsi_14                         — RSI
  atr_pct                        — ATR / close
  vol_ratio                      — volume / 20-bar vol SMA
  ema_ratio                      — EMA50 / EMA200 - 1
  close_vs_sma200                — close / SMA200 - 1
  bb_width                       — Bollinger band width / mid
  drawdown_20                    — rolling 20-bar max drawdown

Usage:
    clf = train_regime_classifier(btc_daily_df)
    regime_series = clf.predict(btc_daily_df)  # 'bull'/'bear'/'chop' per bar
    clf.save(Path("~/.terminus/regime_model.json"))

    # Load pre-trained
    clf2 = RegimeClassifier.load(Path("~/.terminus/regime_model.json"))
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("terminus.ml.regime")

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_CHOP = "chop"

_LABEL_MAP = {0: REGIME_BEAR, 1: REGIME_CHOP, 2: REGIME_BULL}
_INT_MAP = {REGIME_BEAR: 0, REGIME_CHOP: 1, REGIME_BULL: 2}

FORWARD_BARS = 20          # bars ahead used to auto-label
BULL_THRESHOLD = 0.05      # +5% forward return → bull
BEAR_THRESHOLD = -0.05     # -5% forward return → bear
MIN_TRAIN_BARS = 252       # ~1 year of daily data minimum to train


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime features from OHLCV DataFrame. Returns aligned DataFrame."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # Rolling returns
    feat["ret_5"] = close.pct_change(5).clip(-1, 1)
    feat["ret_20"] = close.pct_change(20).clip(-1, 1)
    feat["ret_60"] = close.pct_change(60).clip(-1, 1)

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = (100 - 100 / (1 + rs)) / 100  # normalised 0-1

    # ATR as % of price
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    feat["atr_pct"] = tr.rolling(14).mean() / close

    # Volume ratio
    vol_sma = volume.rolling(20).mean()
    feat["vol_ratio"] = (volume / vol_sma.replace(0, np.nan)).clip(0, 5)

    # EMA ratio (EMA50 / EMA200 - 1)
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    feat["ema_ratio"] = (ema50 / ema200 - 1).clip(-0.5, 0.5)

    # Close vs SMA200
    sma200 = close.rolling(200).mean()
    feat["close_vs_sma200"] = (close / sma200 - 1).clip(-0.5, 0.5)

    # Bollinger band width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_width"] = (2 * std20 / sma20.replace(0, np.nan)).clip(0, 1)

    # 20-bar rolling drawdown
    roll_max = close.rolling(20).max()
    feat["drawdown_20"] = ((close - roll_max) / roll_max.replace(0, np.nan)).clip(-1, 0)

    return feat


def _auto_labels(df: pd.DataFrame, forward_bars: int = FORWARD_BARS) -> pd.Series:
    """Generate labels from realized forward returns."""
    close = df["close"].astype(float)
    fwd_ret = close.shift(-forward_bars) / close - 1
    labels = pd.Series(REGIME_CHOP, index=df.index, dtype=object)
    labels[fwd_ret >= BULL_THRESHOLD] = REGIME_BULL
    labels[fwd_ret <= BEAR_THRESHOLD] = REGIME_BEAR
    return labels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class RegimeClassifier:
    """XGBoost-backed 3-class regime classifier."""

    _model: Any = field(default=None, repr=False)
    feature_cols: list[str] = field(default_factory=list)
    trained_on_bars: int = 0
    train_accuracy: float = 0.0

    # --- Training ----------------------------------------------------------

    def train(self, df: pd.DataFrame) -> "RegimeClassifier":
        """Train on full DataFrame. Labels auto-generated from forward returns.

        Last FORWARD_BARS rows are dropped (no future label available).
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install terminus-lab[ml]  # needs xgboost")

        feat = _features(df)
        labels = _auto_labels(df)

        # Drop rows with NaN features or unlabelled tail
        valid = feat.dropna().index.intersection(
            labels.dropna().index
        )[:-FORWARD_BARS]

        if len(valid) < MIN_TRAIN_BARS:
            raise ValueError(
                f"Need >= {MIN_TRAIN_BARS} valid bars to train; got {len(valid)}"
            )

        X = feat.loc[valid].values.astype(np.float32)
        y = np.array([_INT_MAP[labels.loc[i]] for i in valid], dtype=np.int32)

        self.feature_cols = list(feat.columns)
        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        self._model.fit(X, y)
        self.trained_on_bars = len(valid)
        self.train_accuracy = float((self._model.predict(X) == y).mean())
        logger.info(
            f"Regime model trained on {len(valid)} bars; "
            f"train accuracy={self.train_accuracy:.1%}"
        )
        return self

    # --- Inference ---------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of 'bull'/'bear'/'chop' labels, same index as df."""
        if self._model is None:
            raise RuntimeError("Call .train() or .load() first")
        feat = _features(df)[self.feature_cols]
        nan_mask = feat.isna().any(axis=1)
        result = pd.Series(REGIME_CHOP, index=df.index, dtype=object)
        valid_idx = feat.index[~nan_mask]
        if len(valid_idx):
            X = feat.loc[valid_idx].values.astype(np.float32)
            preds = self._model.predict(X)
            result.loc[valid_idx] = [_LABEL_MAP[p] for p in preds]
        return result

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns [bear, chop, bull] probabilities."""
        if self._model is None:
            raise RuntimeError("Call .train() or .load() first")
        feat = _features(df)[self.feature_cols]
        nan_mask = feat.isna().any(axis=1)
        proba = pd.DataFrame(
            0.0, index=df.index,
            columns=[REGIME_BEAR, REGIME_CHOP, REGIME_BULL],
        )
        proba.loc[nan_mask, REGIME_CHOP] = 1.0
        valid_idx = feat.index[~nan_mask]
        if len(valid_idx):
            X = feat.loc[valid_idx].values.astype(np.float32)
            p = self._model.predict_proba(X)
            proba.loc[valid_idx, REGIME_BEAR] = p[:, 0]
            proba.loc[valid_idx, REGIME_CHOP] = p[:, 1]
            proba.loc[valid_idx, REGIME_BULL] = p[:, 2]
        return proba

    def bull_mask(self, df: pd.DataFrame, min_bull_prob: float = 0.5) -> pd.Series:
        """Boolean mask — True where regime is confidently bull."""
        proba = self.predict_proba(df)
        return proba[REGIME_BULL] >= min_bull_prob

    # --- Persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install terminus-lab[ml]")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path.with_suffix(".xgb")))
        meta = {
            "feature_cols": self.feature_cols,
            "trained_on_bars": self.trained_on_bars,
            "train_accuracy": self.train_accuracy,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Regime model saved to {path.with_suffix('.xgb')}")

    @classmethod
    def load(cls, path: Path | str) -> "RegimeClassifier":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install terminus-lab[ml]")
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        obj = cls(
            feature_cols=meta["feature_cols"],
            trained_on_bars=meta["trained_on_bars"],
            train_accuracy=meta["train_accuracy"],
        )
        obj._model = xgb.XGBClassifier()
        obj._model.load_model(str(path.with_suffix(".xgb")))
        return obj


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def train_regime_classifier(btc_daily_df: pd.DataFrame) -> RegimeClassifier:
    """Train a fresh regime classifier on BTC daily data."""
    return RegimeClassifier().train(btc_daily_df)
