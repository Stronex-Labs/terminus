"""Indicator precompute — one-shot, persists on the DataFrame for all rules.

Two layers:
  precompute_all(df)  — core indicators used by v1 rules (RSI, EMA, MACD,
                        Bollinger, ATR, Donchian, volume, Stochastic,
                        Williams %R, ROC)
  precompute_v2(df)   — v2 indicators on top (Supertrend, Keltner, Ichimoku,
                        VWAP, Heikin Ashi, ATR burst, opening-range proxy,
                        ROC)

Call precompute_all first, then precompute_v2. All v1 column names are
stable so results are reproducible across tool versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def precompute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Core indicators for v1 rule families.

    Requires columns: open, high, low, close, volume, open_time.
    A `ts` column (UTC datetime) is added if missing.
    Returns a new DataFrame; input is not mutated.
    """
    import pandas_ta as ta

    df = df.copy()
    for n in (9, 20, 50, 100, 200):
        df[f"ema{n}"] = ta.ema(df["close"], length=n)
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["rsi7"] = ta.rsi(df["close"], length=7)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_sig"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    bb = ta.bbands(df["close"], length=20, std=2.0)
    bbl = next(c for c in bb.columns if c.startswith("BBL_"))
    bbm = next(c for c in bb.columns if c.startswith("BBM_"))
    bbu = next(c for c in bb.columns if c.startswith("BBU_"))
    df["bb_lo"] = bb[bbl]
    df["bb_mid"] = bb[bbm]
    df["bb_up"] = bb[bbu]
    df["bb_width"] = (bb[bbu] - bb[bbl]) / bb[bbm]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["donch_hi20"] = df["high"].rolling(20).max()
    df["donch_hi50"] = df["high"].rolling(50).max()
    df["donch_lo20"] = df["low"].rolling(20).min()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["roc"] = df["close"].pct_change(periods=10) * 100

    if "ts" not in df.columns and "open_time" in df.columns:
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def precompute_v2(df: pd.DataFrame) -> pd.DataFrame:
    """v2 indicators on top of precompute_all. All columns prefixed `v2_`."""
    import pandas_ta as ta

    df = df.copy()

    try:
        st = ta.supertrend(df["high"], df["low"], df["close"],
                           length=10, multiplier=3.0)
        st_col = next((c for c in st.columns if c.startswith("SUPERT_")), None)
        std_col = next((c for c in st.columns if c.startswith("SUPERTd_")), None)
        if st_col and std_col:
            df["v2_supertrend"] = st[st_col]
            df["v2_supertrend_dir"] = st[std_col]
        else:
            df["v2_supertrend"] = np.nan
            df["v2_supertrend_dir"] = np.nan
    except Exception:
        df["v2_supertrend"] = np.nan
        df["v2_supertrend_dir"] = np.nan

    df["v2_kelt_mid"] = ta.ema(df["close"], length=20)
    df["v2_kelt_up"] = df["v2_kelt_mid"] + 2 * df["atr"]
    df["v2_kelt_lo"] = df["v2_kelt_mid"] - 2 * df["atr"]

    try:
        ichi = ta.ichimoku(df["high"], df["low"], df["close"],
                           tenkan=9, kijun=26, senkou=52)
        if isinstance(ichi, tuple):
            ichi = ichi[0]
        for col, mine in [
            ("ITS_9", "v2_ichi_tenkan"),
            ("IKS_26", "v2_ichi_kijun"),
            ("ISA_9", "v2_ichi_senkou_a"),
            ("ISB_26", "v2_ichi_senkou_b"),
        ]:
            df[mine] = ichi[col] if col in ichi.columns else np.nan
    except Exception:
        for k in ("v2_ichi_tenkan", "v2_ichi_kijun",
                  "v2_ichi_senkou_a", "v2_ichi_senkou_b"):
            df[k] = np.nan

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    df["v2_vwap24"] = pv.rolling(24).sum() / df["volume"].rolling(24).sum()
    df["v2_vwap96"] = pv.rolling(96).sum() / df["volume"].rolling(96).sum()

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for k in range(1, len(df)):
        ha_open.iloc[k] = (ha_open.iloc[k - 1] + ha_close.iloc[k - 1]) / 2.0
    df["v2_ha_close"] = ha_close
    df["v2_ha_open"] = ha_open
    df["v2_ha_green"] = (ha_close > ha_open).astype(int)
    g = df["v2_ha_green"].values
    streak = np.zeros(len(df), dtype=int)
    for k in range(3, len(df)):
        if g[k] == 1 and g[k - 1] == 1 and g[k - 2] == 1 and g[k - 3] == 0:
            streak[k] = 1
    df["v2_ha_3green_fresh"] = streak

    df["v2_atr_pct20"] = df["atr"].rolling(20).rank(pct=True)
    df["v2_atr_sma20"] = df["atr"].rolling(20).mean()
    df["v2_atr_burst"] = (df["atr"] > 1.5 * df["v2_atr_sma20"]).astype(int)

    bar_range = df["high"] - df["low"]
    df["v2_close_pos"] = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)

    for n in (5, 10, 20):
        df[f"v2_roc{n}"] = df["close"].pct_change(n) * 100

    if "ts" in df.columns:
        df["v2_utc_day"] = df["ts"].dt.date.astype(str)
        grp = df.groupby("v2_utc_day")
        orh = grp["high"].rolling(4).max().reset_index(level=0, drop=True)
        orl = grp["low"].rolling(4).min().reset_index(level=0, drop=True)
        df["v2_or_high"] = orh
        df["v2_or_low"] = orl

    return df


def build_btc_regime_series(btc_daily_df: pd.DataFrame) -> pd.Series:
    """Timestamp-indexed Series: 1 when BTC daily EMA50>EMA200, else 0.

    Used to wrap rules so they only fire when the overall crypto regime is
    bullish. Sitting in cash (stablecoin) during bear regimes is
    Shariah-compliant — stablecoins are not interest-bearing.
    """
    import pandas_ta as ta
    df = btc_daily_df.copy()
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    regime = (df["ema50"] > df["ema200"]).astype(int)
    if "ts" in df.columns:
        regime.index = pd.DatetimeIndex(df["ts"])
    return regime.dropna()
