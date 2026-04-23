"""Vectorized rule registry — every rule returns a bool ndarray over df.

These are numpy-vectorized rewrites of the scalar (i, df)->bool rules in
btc_strategy_lab.py and strategies_v2.py. Running them gives identical
signal indices to the scalar version but ~50-200x faster.

Each rule is a `VRule` instance:
  - .name                -> family prefix
  - .signal(df)          -> np.ndarray[bool]  (vectorized signal)
  - .__call__(i, df)     -> bool              (scalar fallback, unused by fast sim)
  - .vectorized_signal(df) -> np.ndarray[bool] (alias for signal, used by simulate_fast)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class VRule:
    family: str
    params: dict
    fn: Callable  # (df) -> np.ndarray[bool]

    def signal(self, df: pd.DataFrame) -> np.ndarray:
        return self.fn(df)

    vectorized_signal = signal

    def __call__(self, i: int, df: pd.DataFrame) -> bool:
        # Scalar fallback — re-evaluate once per call (slow but correct)
        if not hasattr(self, "_cache_id") or self._cache_id is not id(df):
            self._cache = self.signal(df)
            self._cache_id = id(df)
        if 0 <= i < len(self._cache):
            return bool(self._cache[i])
        return False


def _col(df, name):
    return df[name].values if name in df.columns else np.full(len(df), np.nan)


# ---------------------------------------------------------------------------
# v1 rules — vectorized
# ---------------------------------------------------------------------------
def rsi_cross(lo: float, hi: float, trend_filter: bool = True) -> VRule:
    def _f(df):
        rsi = _col(df, "rsi")
        prev = np.roll(rsi, 1); prev[0] = np.nan
        ema200 = _col(df, "ema200")
        close = _col(df, "close")
        in_zone = (rsi >= lo) & (rsi <= hi) & (prev < lo)
        if trend_filter:
            trend = close > ema200
        else:
            trend = np.ones_like(rsi, dtype=bool)
        valid = ~np.isnan(rsi) & ~np.isnan(prev) & ~np.isnan(ema200)
        return in_zone & trend & valid
    return VRule("RSI-cross", {"lo": lo, "hi": hi, "trend": trend_filter}, _f)


def rsi_in_zone(lo: float, hi: float) -> VRule:
    def _f(df):
        rsi = _col(df, "rsi")
        ema200 = _col(df, "ema200")
        close = _col(df, "close")
        valid = ~np.isnan(rsi) & ~np.isnan(ema200)
        return (rsi >= lo) & (rsi <= hi) & (close > ema200) & valid
    return VRule("RSI-in", {"lo": lo, "hi": hi}, _f)


def rsi7_oversold_cross(level: float = 30) -> VRule:
    def _f(df):
        r7 = _col(df, "rsi7")
        prev = np.roll(r7, 1); prev[0] = np.nan
        ema200 = _col(df, "ema200")
        close = _col(df, "close")
        valid = ~np.isnan(r7) & ~np.isnan(prev) & ~np.isnan(ema200)
        return (r7 > level) & (prev <= level) & (close > ema200) & valid
    return VRule("RSI7-cross", {"level": level}, _f)


def ema_cross(fast: int, slow: int) -> VRule:
    def _f(df):
        a = _col(df, f"ema{fast}")
        b = _col(df, f"ema{slow}")
        pa = np.roll(a, 1); pa[0] = np.nan
        pb = np.roll(b, 1); pb[0] = np.nan
        valid = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(pa) & ~np.isnan(pb)
        return (a > b) & (pa <= pb) & valid
    return VRule("EMA-cross", {"fast": fast, "slow": slow}, _f)


def price_cross_ema(n: int) -> VRule:
    def _f(df):
        close = _col(df, "close")
        pclose = np.roll(close, 1); pclose[0] = np.nan
        e = _col(df, f"ema{n}")
        pe = np.roll(e, 1); pe[0] = np.nan
        valid = ~np.isnan(e) & ~np.isnan(pe)
        return (close > e) & (pclose <= pe) & valid
    return VRule("Price-cross-EMA", {"n": n}, _f)


def bull_stack_fresh() -> VRule:
    def _f(df):
        e20 = _col(df, "ema20"); e50 = _col(df, "ema50"); e200 = _col(df, "ema200")
        pe20 = np.roll(e20, 1); pe20[0] = np.nan
        pe50 = np.roll(e50, 1); pe50[0] = np.nan
        pe200 = np.roll(e200, 1); pe200[0] = np.nan
        valid = (~np.isnan(e20) & ~np.isnan(e50) & ~np.isnan(e200)
                 & ~np.isnan(pe20) & ~np.isnan(pe50) & ~np.isnan(pe200))
        new_stack = (e20 > e50) & (e50 > e200)
        old_stack = (pe20 > pe50) & (pe50 > pe200)
        return new_stack & (~old_stack) & valid
    return VRule("BullStack-fresh", {}, _f)


def bb_lower_touch() -> VRule:
    def _f(df):
        low = _col(df, "low"); bb_lo = _col(df, "bb_lo")
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        rsi = _col(df, "rsi")
        valid = (~np.isnan(low) & ~np.isnan(bb_lo)
                 & ~np.isnan(ema200) & ~np.isnan(rsi))
        touched = low <= bb_lo * 1.002
        return touched & (close > ema200) & (rsi < 45) & valid
    return VRule("BB-lower-touch", {}, _f)


def bb_squeeze_breakout(width_pct: float = 0.05) -> VRule:
    def _f(df):
        w = _col(df, "bb_width")
        # rolling 20-bar min of width
        ser = pd.Series(w)
        p_low = ser.rolling(20, min_periods=20).min().shift(1).values
        close = _col(df, "close"); bb_up = _col(df, "bb_up")
        ema200 = _col(df, "ema200")
        valid = (~np.isnan(w) & ~np.isnan(p_low)
                 & ~np.isnan(bb_up) & ~np.isnan(ema200))
        was_squeeze = p_low < width_pct
        broke_out = close > bb_up
        return was_squeeze & broke_out & (close > ema200) & valid
    return VRule("BB-sqz", {"w": width_pct}, _f)


def macd_cross_above_zero() -> VRule:
    def _f(df):
        m = _col(df, "macd"); s = _col(df, "macd_sig")
        pm = np.roll(m, 1); pm[0] = np.nan
        ps = np.roll(s, 1); ps[0] = np.nan
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = (~np.isnan(m) & ~np.isnan(s) & ~np.isnan(pm)
                 & ~np.isnan(ps) & ~np.isnan(ema200))
        return ((m > s) & (pm <= ps) & (m > 0) & (close > ema200) & valid)
    return VRule("MACD-cross-0", {}, _f)


def donch_breakout(n: int) -> VRule:
    def _f(df):
        close = _col(df, "close")
        # prior N-bar high (shifted by 1 so we compare against the completed level)
        high = pd.Series(_col(df, "high"))
        prior = high.rolling(n, min_periods=n).max().shift(1).values
        ema200 = _col(df, "ema200")
        valid = ~np.isnan(prior) & ~np.isnan(ema200)
        return (close > prior) & (close > ema200) & valid
    return VRule("Donch-brk", {"n": n}, _f)


def volume_breakout(lookback: int, vol_mult: float) -> VRule:
    def _f(df):
        close = _col(df, "close")
        high = pd.Series(_col(df, "high"))
        prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1).values
        vr = _col(df, "vol_ratio"); ema200 = _col(df, "ema200")
        valid = ~np.isnan(prior_high) & ~np.isnan(vr) & ~np.isnan(ema200)
        return (close > prior_high) & (vr > vol_mult) & (close > ema200) & valid
    return VRule("Vol-brk", {"lb": lookback, "vm": vol_mult}, _f)


def pullback_ema(n: int, rsi_lo: float = 40, rsi_hi: float = 65) -> VRule:
    def _f(df):
        close = _col(df, "close"); low = _col(df, "low")
        e = _col(df, f"ema{n}")
        e20 = _col(df, "ema20"); e50 = _col(df, "ema50"); e200 = _col(df, "ema200")
        rsi = _col(df, "rsi")
        valid = (~np.isnan(e) & ~np.isnan(e20) & ~np.isnan(e50)
                 & ~np.isnan(e200) & ~np.isnan(rsi))
        bull_stack = (e20 > e50) & (e50 > e200)
        near = (low <= e * 1.005) & (close >= e * 0.995)
        healthy = (rsi > rsi_lo) & (rsi < rsi_hi)
        return bull_stack & near & healthy & valid
    return VRule("Pullback-EMA", {"n": n, "lo": rsi_lo, "hi": rsi_hi}, _f)


def atr_channel_break(k: float = 2.0) -> VRule:
    def _f(df):
        close = _col(df, "close")
        pclose = np.roll(close, 1); pclose[0] = np.nan
        atr = _col(df, "atr"); ema200 = _col(df, "ema200")
        valid = ~np.isnan(atr) & ~np.isnan(ema200) & ~np.isnan(pclose)
        return (close > pclose + k * atr) & (close > ema200) & valid
    return VRule("ATR-brk", {"k": k}, _f)


def stoch_cross(level: float = 20) -> VRule:
    def _f(df):
        k = _col(df, "stoch_k"); d = _col(df, "stoch_d")
        pk = np.roll(k, 1); pk[0] = np.nan
        pd_ = np.roll(d, 1); pd_[0] = np.nan
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = (~np.isnan(k) & ~np.isnan(d) & ~np.isnan(pk)
                 & ~np.isnan(pd_) & ~np.isnan(ema200))
        return ((k > d) & (pk <= pd_) & (k < level + 20)
                & (close > ema200) & valid)
    return VRule("Stoch-cross", {"level": level}, _f)


def willr_reversal(level: float = -80) -> VRule:
    def _f(df):
        w = _col(df, "willr")
        pw = np.roll(w, 1); pw[0] = np.nan
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = ~np.isnan(w) & ~np.isnan(pw) & ~np.isnan(ema200)
        return (w > level) & (pw <= level) & (close > ema200) & valid
    return VRule("WillR-rev", {"level": level}, _f)


def combo_rsi_vol(lo: float, hi: float, vol_mult: float) -> VRule:
    def _f(df):
        rsi = _col(df, "rsi")
        prev = np.roll(rsi, 1); prev[0] = np.nan
        vr = _col(df, "vol_ratio"); ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = (~np.isnan(rsi) & ~np.isnan(prev)
                 & ~np.isnan(vr) & ~np.isnan(ema200))
        fresh = (rsi >= lo) & (rsi <= hi) & (prev < lo)
        return fresh & (vr > vol_mult) & (close > ema200) & valid
    return VRule("RSI-Vol", {"lo": lo, "hi": hi, "vm": vol_mult}, _f)


# ---------------------------------------------------------------------------
# v2 rules — vectorized
# ---------------------------------------------------------------------------
def supertrend_flip() -> VRule:
    def _f(df):
        d = _col(df, "v2_supertrend_dir")
        pd_ = np.roll(d, 1); pd_[0] = np.nan
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = ~np.isnan(d) & ~np.isnan(pd_) & ~np.isnan(ema200)
        return (d == 1) & (pd_ == -1) & (close > ema200) & valid
    return VRule("Supertrend-flip", {}, _f)


def chandelier_entry(n: int = 22, atr_mult: float = 3.0) -> VRule:
    def _f(df):
        high = pd.Series(_col(df, "high"))
        prior_high = high.rolling(n, min_periods=n).max().shift(1).values
        atr = _col(df, "atr"); close = _col(df, "close"); ema200 = _col(df, "ema200")
        valid = ~np.isnan(prior_high) & ~np.isnan(atr) & ~np.isnan(ema200)
        chand = prior_high - atr_mult * atr
        return (close > chand) & (close > ema200) & valid
    return VRule("Chand", {"n": n, "k": atr_mult}, _f)


def keltner_break() -> VRule:
    def _f(df):
        close = _col(df, "close")
        pclose = np.roll(close, 1); pclose[0] = np.nan
        up = _col(df, "v2_kelt_up")
        pup = np.roll(up, 1); pup[0] = np.nan
        ema200 = _col(df, "ema200")
        valid = ~np.isnan(up) & ~np.isnan(pup) & ~np.isnan(ema200)
        return (close > up) & (pclose <= pup) & (close > ema200) & valid
    return VRule("Keltner-brk", {}, _f)


def ichimoku_bullish() -> VRule:
    def _f(df):
        t = _col(df, "v2_ichi_tenkan"); kj = _col(df, "v2_ichi_kijun")
        sa = _col(df, "v2_ichi_senkou_a"); sb = _col(df, "v2_ichi_senkou_b")
        pt = np.roll(t, 1); pt[0] = np.nan
        pkj = np.roll(kj, 1); pkj[0] = np.nan
        close = _col(df, "close")
        cloud_top = np.maximum(sa, sb)
        valid = (~np.isnan(t) & ~np.isnan(kj) & ~np.isnan(sa) & ~np.isnan(sb)
                 & ~np.isnan(pt) & ~np.isnan(pkj))
        return (t > kj) & (pt <= pkj) & (close > cloud_top) & valid
    return VRule("Ichi-bull", {}, _f)


def vwap_reclaim(window: str = "v2_vwap24") -> VRule:
    def _f(df):
        close = _col(df, "close")
        pclose = np.roll(close, 1); pclose[0] = np.nan
        v = _col(df, window)
        pv = np.roll(v, 1); pv[0] = np.nan
        ema200 = _col(df, "ema200")
        valid = ~np.isnan(v) & ~np.isnan(pv) & ~np.isnan(ema200)
        return (close > v) & (pclose <= pv) & (close > ema200) & valid
    return VRule("VWAP-reclaim", {"w": window}, _f)


def ha_three_green() -> VRule:
    def _f(df):
        fresh = _col(df, "v2_ha_3green_fresh")
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = ~np.isnan(fresh) & ~np.isnan(ema200)
        return (fresh == 1) & (close > ema200) & valid
    return VRule("HA-3green", {}, _f)


def orb_break() -> VRule:
    def _f(df):
        if "v2_or_high" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        close = _col(df, "close")
        pclose = np.roll(close, 1); pclose[0] = np.nan
        orh = _col(df, "v2_or_high"); ema200 = _col(df, "ema200")
        valid = ~np.isnan(orh) & ~np.isnan(ema200)
        return (close > orh) & (pclose <= orh) & (close > ema200) & valid
    return VRule("ORB", {}, _f)


def atr_burst() -> VRule:
    def _f(df):
        burst = _col(df, "v2_atr_burst")
        close_pos = _col(df, "v2_close_pos")
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = ~np.isnan(burst) & ~np.isnan(close_pos) & ~np.isnan(ema200)
        return (burst == 1) & (close_pos >= 0.70) & (close > ema200) & valid
    return VRule("ATR-burst", {}, _f)


def rsi_mr_fast(lo: float = 20, reclaim: float = 25) -> VRule:
    def _f(df):
        rsi = pd.Series(_col(df, "rsi"))
        prev = rsi.shift(1).values
        recent_min = rsi.rolling(5, min_periods=5).min().shift(1).values
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        rsi_v = rsi.values
        valid = (~np.isnan(rsi_v) & ~np.isnan(prev)
                 & ~np.isnan(recent_min) & ~np.isnan(ema200))
        return ((recent_min < lo) & (rsi_v >= reclaim) & (prev < reclaim)
                & (close > ema200 * 0.98) & valid)
    return VRule("RSI-MR", {"lo": lo, "rec": reclaim}, _f)


def momentum_bar(min_pct: float = 0.015) -> VRule:
    def _f(df):
        o = _col(df, "open"); c = _col(df, "close")
        h = _col(df, "high")
        ph = np.roll(h, 1); ph[0] = np.nan
        e50 = _col(df, "ema50"); e200 = _col(df, "ema200")
        valid = ~np.isnan(e50) & ~np.isnan(e200)
        pct = (c - o) / np.where(o == 0, np.nan, o)
        return ((pct >= min_pct) & (c > ph) & (c > e50)
                & (e50 > e200) & valid)
    return VRule("MomBar", {"p": min_pct}, _f)


def roc_momentum(n: int = 10, min_roc: float = 3.0) -> VRule:
    def _f(df):
        roc = _col(df, f"v2_roc{n}")
        prev = np.roll(roc, 1); prev[0] = np.nan
        ema200 = _col(df, "ema200"); close = _col(df, "close")
        valid = ~np.isnan(roc) & ~np.isnan(prev) & ~np.isnan(ema200)
        return (roc >= min_roc) & (prev < min_roc) & (close > ema200) & valid
    return VRule("ROC", {"n": n, "r": min_roc}, _f)


# ---------------------------------------------------------------------------
# Wrapper: BTC regime filter
# ---------------------------------------------------------------------------
def with_btc_regime(base: VRule, btc_regime_series: pd.Series,
                    ts_col: str = "ts") -> VRule:
    """Wrap a vectorized rule: also requires BTC daily regime == 1."""
    def _f(df):
        sig = base.signal(df)
        if ts_col not in df.columns:
            return sig
        ts = df[ts_col]
        # as-of join — for each bar's ts find BTC regime on-or-before
        idx = btc_regime_series.index.searchsorted(pd.DatetimeIndex(ts),
                                                    side="right") - 1
        idx_valid = idx >= 0
        regime = np.zeros(len(df), dtype=bool)
        regime[idx_valid] = btc_regime_series.values[idx[idx_valid]] == 1
        return sig & regime
    # augment family name so it's distinguishable
    return VRule(f"{base.family}+BTCreg", dict(base.params, btc_reg=True), _f)
