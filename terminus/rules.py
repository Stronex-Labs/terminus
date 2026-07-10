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


# ---------------------------------------------------------------------------
# v2+ rules — new strategy families
# ---------------------------------------------------------------------------

def mr_bb_rsi_divergence() -> VRule:
    """Mean reversion: BB lower touch + RSI making higher low (divergence)."""
    def _f(df):
        low = _col(df, "low")
        bb_lo = _col(df, "bb_lo")
        rsi = pd.Series(_col(df, "rsi"))
        ema200 = _col(df, "ema200")
        close = _col(df, "close")
        adx = _col(df, "adx")
        # RSI minimum over last 10 bars (shifted by 1 to exclude current)
        rsi_min10 = rsi.rolling(10, min_periods=5).min().shift(1).values
        rsi_v = rsi.values
        valid = (~np.isnan(low) & ~np.isnan(bb_lo) & ~np.isnan(rsi_v)
                 & ~np.isnan(ema200) & ~np.isnan(adx) & ~np.isnan(rsi_min10))
        touched = low <= bb_lo * 1.002
        rsi_low = rsi_v < 35
        rsi_recovering = rsi_v > rsi_min10  # higher low pattern
        not_falling = close > ema200 * 0.97
        ranging = adx < 30
        return touched & rsi_low & rsi_recovering & not_falling & ranging & valid
    return VRule("MR-BB-RSI-div", {}, _f)


def mr_ema_deviation() -> VRule:
    """Mean reversion: price >3% below EMA20, then recovery bar."""
    def _f(df):
        close = pd.Series(_col(df, "close"))
        ema20 = pd.Series(_col(df, "ema20"))
        rsi = _col(df, "rsi")
        vr = _col(df, "vol_ratio")
        # Deviation: close was >3% below EMA20 in last 3 bars
        deviation = (close / ema20 - 1)  # negative when below
        was_deviated = deviation.rolling(3, min_periods=1).min().values < -0.03
        # Recovery: current close > previous close
        pclose = close.shift(1).values
        close_v = close.values
        ema20_v = ema20.values
        recovery = close_v > pclose
        valid = (~np.isnan(close_v) & ~np.isnan(ema20_v) & ~np.isnan(pclose)
                 & ~np.isnan(rsi) & ~np.isnan(vr))
        return was_deviated & recovery & (rsi < 40) & (vr > 0.8) & valid
    return VRule("MR-EMA-dev", {}, _f)


def momentum_adx_surge() -> VRule:
    """Momentum: ADX rising + price trending in bull stack."""
    def _f(df):
        adx = pd.Series(_col(df, "adx"))
        adx_5ago = adx.shift(5).values
        adx_v = adx.values
        close = _col(df, "close")
        ema20 = _col(df, "ema20")
        ema50 = _col(df, "ema50")
        rsi = _col(df, "rsi")
        vr = _col(df, "vol_ratio")
        valid = (~np.isnan(adx_v) & ~np.isnan(adx_5ago) & ~np.isnan(close)
                 & ~np.isnan(ema20) & ~np.isnan(ema50) & ~np.isnan(rsi)
                 & ~np.isnan(vr))
        adx_strong = (adx_v > 25) & (adx_v > adx_5ago)
        bull_stack = (close > ema20) & (ema20 > ema50)
        rsi_zone = (rsi >= 55) & (rsi <= 75)
        vol_ok = vr > 1.2
        return adx_strong & bull_stack & rsi_zone & vol_ok & valid
    return VRule("Mom-ADX-surge", {}, _f)


def momentum_sniper(min_conditions: int = 3) -> VRule:
    """Mom-sniper — N-of-4 momentum vote (faithful port of Stronex's live
    rule_momentum_sniper, so terminus can validate the deployed family):
      1) ema9 > ema20   2) rsi in [50,70]   3) vol_ratio > 1.3   4) adx > 20.
    Fires when at least `min_conditions` of the 4 are true.
    """
    def _f(df):
        ema9 = _col(df, "ema9"); ema20 = _col(df, "ema20")
        rsi = _col(df, "rsi"); vr = _col(df, "vol_ratio"); adx = _col(df, "adx")
        valid = (~np.isnan(ema9) & ~np.isnan(ema20) & ~np.isnan(rsi)
                 & ~np.isnan(vr) & ~np.isnan(adx))
        conds = ((ema9 > ema20).astype(int)
                 + ((rsi >= 50) & (rsi <= 70)).astype(int)
                 + (vr > 1.3).astype(int)
                 + (adx > 20).astype(int))
        return (conds >= min_conditions) & valid
    return VRule("Mom-sniper", {"mc": min_conditions}, _f)


def momentum_ema_accel() -> VRule:
    """Momentum: EMA20-EMA50 gap widening (acceleration)."""
    def _f(df):
        ema20 = pd.Series(_col(df, "ema20"))
        ema50 = pd.Series(_col(df, "ema50"))
        ema200 = _col(df, "ema200")
        close = _col(df, "close")
        adx = _col(df, "adx")
        gap = ema20 - ema50
        prev_gap = gap.shift(1).values
        gap_v = gap.values
        ema20_v = ema20.values
        ema50_v = ema50.values
        valid = (~np.isnan(gap_v) & ~np.isnan(prev_gap) & ~np.isnan(ema200)
                 & ~np.isnan(close) & ~np.isnan(adx))
        widening = gap_v > prev_gap
        full_stack = (close > ema20_v) & (ema20_v > ema50_v) & (ema50_v > ema200)
        adx_ok = adx > 20
        return widening & full_stack & adx_ok & valid
    return VRule("Mom-EMA-accel", {}, _f)


def breakout_range_expansion() -> VRule:
    """Breakout: consolidation then volatility expansion with volume."""
    def _f(df):
        atr = pd.Series(_col(df, "atr"))
        atr_sma20 = atr.rolling(20, min_periods=20).mean().values
        atr_v = atr.values
        close = _col(df, "close")
        high = pd.Series(_col(df, "high"))
        hh20 = high.rolling(20, min_periods=20).max().shift(1).values
        vr = _col(df, "vol_ratio")
        ema200 = _col(df, "ema200")
        valid = (~np.isnan(atr_v) & ~np.isnan(atr_sma20) & ~np.isnan(hh20)
                 & ~np.isnan(vr) & ~np.isnan(ema200))
        expansion = atr_v > 1.5 * atr_sma20
        breakout = close > hh20
        vol_confirm = vr > 1.5
        trend = close > ema200
        return expansion & breakout & vol_confirm & trend & valid
    return VRule("Brk-RangeExp", {}, _f)


def pullback_fib_zone() -> VRule:
    """Pullback: price retraces to 38-62% fib zone of recent swing in uptrend."""
    def _f(df):
        close = _col(df, "close")
        high = pd.Series(_col(df, "high"))
        low = pd.Series(_col(df, "low"))
        ema20 = _col(df, "ema20")
        ema50 = _col(df, "ema50")
        ema200 = _col(df, "ema200")
        rsi = _col(df, "rsi")
        # Swing high = max of last 20 bars
        swing_high = high.rolling(20, min_periods=10).max().values
        # Swing low = min of bars 20-40 ago
        swing_low = low.rolling(20, min_periods=10).min().shift(20).values
        # Fib levels
        swing_range = swing_high - swing_low
        fib_382 = swing_high - 0.618 * swing_range  # 38.2% retrace from high
        fib_618 = swing_high - 0.382 * swing_range  # 61.8% retrace from high
        valid = (~np.isnan(close) & ~np.isnan(ema20) & ~np.isnan(ema50)
                 & ~np.isnan(ema200) & ~np.isnan(rsi) & ~np.isnan(swing_high)
                 & ~np.isnan(swing_low) & (swing_range > 0))
        uptrend = (ema20 > ema50) & (ema50 > ema200)
        in_fib = (close >= fib_382) & (close <= fib_618)
        rsi_ok = (rsi >= 35) & (rsi <= 55)
        return uptrend & in_fib & rsi_ok & valid
    return VRule("PB-Fib-zone", {}, _f)


def pullback_vwap_bounce() -> VRule:
    """Pullback: price touches VWAP24 and bounces in uptrend."""
    def _f(df):
        close = _col(df, "close")
        low = _col(df, "low")
        vwap = _col(df, "v2_vwap24")
        ema200 = _col(df, "ema200")
        rsi = _col(df, "rsi")
        valid = (~np.isnan(close) & ~np.isnan(low) & ~np.isnan(vwap)
                 & ~np.isnan(ema200) & ~np.isnan(rsi))
        uptrend = close > ema200
        touched = low <= vwap * 1.003  # within 0.3%
        bounced = close > vwap
        rsi_ok = rsi < 55
        return uptrend & touched & bounced & rsi_ok & valid
    return VRule("PB-VWAP-bounce", {}, _f)


def funding_fade() -> VRule:
    """Funding rate fade: negative funding + technical support."""
    def _f(df):
        if "funding_8h" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        funding = _col(df, "funding_8h")
        close = _col(df, "close")
        ema200 = _col(df, "ema200")
        rsi = _col(df, "rsi")
        vr = _col(df, "vol_ratio")
        valid = (~np.isnan(funding) & ~np.isnan(close) & ~np.isnan(ema200)
                 & ~np.isnan(rsi) & ~np.isnan(vr))
        neg_funding = funding < -0.0001  # -0.01%
        not_collapse = close > ema200 * 0.95
        rsi_ok = rsi < 45
        vol_ok = vr > 0.5
        return neg_funding & not_collapse & rsi_ok & vol_ok & valid
    return VRule("Funding-fade", {}, _f)


def vol_squeeze_keltner() -> VRule:
    """Volatility squeeze: BB inside Keltner, then breakout on release."""
    def _f(df):
        bb_up = pd.Series(_col(df, "bb_up"))
        kelt_up = pd.Series(_col(df, "v2_kelt_up"))
        close = _col(df, "close")
        ema200 = _col(df, "ema200")
        macd_hist = _col(df, "macd_hist")
        bb_up_v = bb_up.values
        kelt_up_v = kelt_up.values
        # Was in squeeze in any of last 5 bars
        squeeze = bb_up < kelt_up  # BB inside Keltner = squeeze
        was_squeezed = squeeze.rolling(5, min_periods=1).max().shift(1).values.astype(bool)
        # Current: squeeze released (BB now outside Keltner)
        released = bb_up_v > kelt_up_v
        # Breakout direction up
        broke_up = close > bb_up_v
        valid = (~np.isnan(bb_up_v) & ~np.isnan(kelt_up_v) & ~np.isnan(close)
                 & ~np.isnan(ema200) & ~np.isnan(macd_hist)
                 & ~np.isnan(was_squeezed))
        trend = close > ema200
        mom_pos = macd_hist > 0
        return was_squeezed & released & broke_up & trend & mom_pos & valid
    return VRule("Vol-Sqz-Kelt", {}, _f)
