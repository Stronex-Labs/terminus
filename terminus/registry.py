"""Unified config registry — returns every (v1 + v2) config as a VRule tuple.

Signature of each tuple:
    (name, vrule, tp_pct, stop_pct, max_hold_bars, cooldown_bars, family_key)

The registry is parameterized so callers can request:
  - only v1 families
  - only v2 families
  - v1 + v2 (default for the big sweep)
  - BTC-regime-wrapped variants (adds a BTC regime bool filter to each base rule)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import rules as rv


def build_v1_configs() -> list[tuple]:
    """16 families from the original lab, now vectorized."""
    out = []

    # RSI-cross: zones × TP/stop
    for lo, hi in [(30, 70), (40, 70), (45, 70), (47, 70), (50, 65), (45, 75)]:
        for tp, stop, hold in [(0.015, 0.020, 18), (0.025, 0.030, 24),
                                (0.035, 0.040, 30), (0.050, 0.050, 36)]:
            out.append((f"RSI-cross[{lo}-{hi}] TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.rsi_cross(lo, hi), tp, stop, hold, 2, "RSI-cross"))

    # RSI in-zone
    for lo, hi in [(47, 70), (50, 65), (45, 75)]:
        for tp, stop, hold in [(0.020, 0.025, 24), (0.030, 0.035, 30),
                                (0.040, 0.045, 36)]:
            out.append((f"RSI-in[{lo}-{hi}] TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.rsi_in_zone(lo, hi), tp, stop, hold, 6, "RSI-in"))

    # RSI7 oversold cross
    for level in (25, 30, 35):
        for tp, stop, hold in [(0.015, 0.020, 18), (0.025, 0.030, 24)]:
            out.append((f"RSI7-cross{level} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.rsi7_oversold_cross(level), tp, stop, hold, 3, "RSI7-cross"))

    # EMA crossovers
    for fast, slow in [(20, 50), (50, 200), (9, 50)]:
        for tp, stop, hold in [(0.020, 0.030, 24), (0.035, 0.045, 36),
                                (0.060, 0.060, 48)]:
            out.append((f"EMA{fast}/{slow}-cross TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.ema_cross(fast, slow), tp, stop, hold, 4, "EMA-cross"))

    # Price crosses EMA
    for n in (50, 100, 200):
        for tp, stop, hold in [(0.025, 0.030, 24), (0.040, 0.045, 36)]:
            out.append((f"Price-cross-EMA{n} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.price_cross_ema(n), tp, stop, hold, 4, "Price-cross-EMA"))

    # Bull stack fresh
    for tp, stop, hold in [(0.030, 0.035, 36), (0.050, 0.050, 48)]:
        out.append((f"BullStack-fresh TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.bull_stack_fresh(), tp, stop, hold, 8, "BullStack-fresh"))

    # BB lower touch
    for tp, stop, hold in [(0.015, 0.025, 18), (0.025, 0.035, 24),
                            (0.040, 0.040, 30)]:
        out.append((f"BB-lower-touch TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.bb_lower_touch(), tp, stop, hold, 4, "BB-lower-touch"))

    # BB squeeze breakout
    for w in (0.04, 0.06):
        for tp, stop, hold in [(0.030, 0.040, 36), (0.050, 0.050, 48)]:
            out.append((f"BB-sqz{w:.2f} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.bb_squeeze_breakout(w), tp, stop, hold, 6, "BB-sqz"))

    # MACD cross above zero
    for tp, stop, hold in [(0.025, 0.030, 24), (0.040, 0.045, 36),
                            (0.060, 0.060, 48)]:
        out.append((f"MACD-cross-0 TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.macd_cross_above_zero(), tp, stop, hold, 4, "MACD-cross"))

    # Donchian breakout
    for n in (10, 20, 50):
        for tp, stop, hold in [(0.025, 0.035, 24), (0.040, 0.045, 36),
                                (0.060, 0.060, 48)]:
            out.append((f"Donch{n}-brk TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.donch_breakout(n), tp, stop, hold, 4, "Donch-brk"))

    # Volume breakout
    for lb, vm in [(20, 1.3), (20, 1.5), (50, 1.5)]:
        for tp, stop, hold in [(0.030, 0.040, 24), (0.050, 0.050, 36)]:
            out.append((f"Vol-brk{lb}/{vm} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.volume_breakout(lb, vm), tp, stop, hold, 4, "Vol-brk"))

    # Pullback
    for n, lo, hi in [(20, 40, 65), (50, 40, 60)]:
        for tp, stop, hold in [(0.020, 0.025, 24), (0.030, 0.035, 30),
                                (0.045, 0.045, 36)]:
            out.append((f"Pullback-EMA{n}[{lo}-{hi}] TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.pullback_ema(n, lo, hi), tp, stop, hold, 4, "Pullback-EMA"))

    # ATR channel break
    for k in (1.5, 2.0, 3.0):
        for tp, stop, hold in [(0.030, 0.040, 30), (0.050, 0.050, 36)]:
            out.append((f"ATR-brk{k} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.atr_channel_break(k), tp, stop, hold, 4, "ATR-brk"))

    # Stoch
    for tp, stop, hold in [(0.020, 0.025, 24), (0.030, 0.035, 30)]:
        out.append((f"Stoch-cross TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.stoch_cross(), tp, stop, hold, 3, "Stoch-cross"))

    # Williams %R
    for tp, stop, hold in [(0.020, 0.025, 24), (0.030, 0.035, 30)]:
        out.append((f"WillR-rev TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.willr_reversal(), tp, stop, hold, 3, "WillR-rev"))

    # Combo RSI + Volume
    for lo, hi in [(40, 70), (47, 70)]:
        for vm in (1.3, 1.5):
            for tp, stop, hold in [(0.025, 0.030, 24), (0.040, 0.045, 36)]:
                out.append((f"RSI[{lo}-{hi}]+Vol{vm} TP{tp*100:.1f}/S{stop*100:.1f}",
                            rv.combo_rsi_vol(lo, hi, vm), tp, stop, hold, 4, "RSI-Vol"))

    return out


def build_v2_configs() -> list[tuple]:
    out = []

    # Supertrend flip
    for tp, stop, hold in [(0.040, 0.040, 36), (0.060, 0.050, 48),
                            (0.080, 0.060, 60)]:
        out.append((f"Supertrend-flip TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.supertrend_flip(), tp, stop, hold, 4, "Supertrend-flip"))

    # Chandelier entry
    for n in (22, 44):
        for k in (2.0, 3.0):
            for tp, stop, hold in [(0.040, 0.045, 36), (0.060, 0.050, 48)]:
                out.append((f"Chand{n}k{k} TP{tp*100:.1f}/S{stop*100:.1f}",
                            rv.chandelier_entry(n, k), tp, stop, hold, 4, "Chand"))

    # Keltner
    for tp, stop, hold in [(0.030, 0.035, 24), (0.050, 0.045, 36),
                            (0.070, 0.055, 48)]:
        out.append((f"Keltner-brk TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.keltner_break(), tp, stop, hold, 4, "Keltner-brk"))

    # Ichimoku
    for tp, stop, hold in [(0.040, 0.045, 36), (0.060, 0.055, 48),
                            (0.080, 0.060, 60)]:
        out.append((f"Ichi-bull TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.ichimoku_bullish(), tp, stop, hold, 8, "Ichi-bull"))

    # VWAP reclaim — two windows
    for w, label in [("v2_vwap24", "VWAP24"), ("v2_vwap96", "VWAP96")]:
        for tp, stop, hold in [(0.020, 0.025, 18), (0.030, 0.035, 24),
                                (0.050, 0.045, 36)]:
            out.append((f"{label}-reclaim TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.vwap_reclaim(w), tp, stop, hold, 3, f"{label}-reclaim"))

    # Heikin Ashi 3-green
    for tp, stop, hold in [(0.030, 0.035, 24), (0.050, 0.045, 36)]:
        out.append((f"HA-3green TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.ha_three_green(), tp, stop, hold, 4, "HA-3green"))

    # Opening range break
    for tp, stop, hold in [(0.020, 0.025, 18), (0.035, 0.035, 24),
                            (0.050, 0.045, 36)]:
        out.append((f"ORB TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.orb_break(), tp, stop, hold, 2, "ORB"))

    # ATR burst
    for tp, stop, hold in [(0.025, 0.030, 18), (0.040, 0.040, 24),
                            (0.060, 0.050, 36)]:
        out.append((f"ATR-burst TP{tp*100:.1f}/S{stop*100:.1f}",
                    rv.atr_burst(), tp, stop, hold, 3, "ATR-burst"))

    # Fast RSI mean reversion
    for lo, reclaim in [(18, 25), (22, 28), (25, 32)]:
        for tp, stop, hold in [(0.010, 0.015, 8), (0.015, 0.020, 12),
                                (0.025, 0.025, 18)]:
            out.append((f"RSI-MR[{lo}/{reclaim}] TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.rsi_mr_fast(lo, reclaim), tp, stop, hold, 2, "RSI-MR"))

    # Single-bar momentum
    for mp in (0.010, 0.015, 0.020):
        for tp, stop, hold in [(0.015, 0.020, 8), (0.025, 0.025, 12),
                                (0.040, 0.035, 18)]:
            out.append((f"MomBar{mp*100:.1f} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.momentum_bar(mp), tp, stop, hold, 2, "MomBar"))

    # ROC momentum
    for n, mr in [(5, 3.0), (10, 5.0), (10, 7.0), (20, 10.0)]:
        for tp, stop, hold in [(0.025, 0.030, 18), (0.040, 0.040, 30),
                                (0.060, 0.050, 42)]:
            out.append((f"ROC{n}@{mr} TP{tp*100:.1f}/S{stop*100:.1f}",
                        rv.roc_momentum(n, mr), tp, stop, hold, 4,
                        f"ROC{n}"))

    return out


def build_all_configs() -> list[tuple]:
    return build_v1_configs() + build_v2_configs()


def build_configs_with_regime(
    base_configs: list[tuple], btc_regime_series: pd.Series,
) -> list[tuple]:
    """Return regime-wrapped variants of each base config.

    Output tuples carry the `+BTCreg` suffix in name and family.
    """
    out = []
    for name, rule, tp, stop, hold, cd, fam in base_configs:
        wrapped = rv.with_btc_regime(rule, btc_regime_series)
        out.append((f"{name} +BTCreg", wrapped, tp, stop, hold, cd,
                    f"{fam}+BTCreg"))
    return out


def all_exit_methods() -> list[str]:
    return [
        "fixed_tp_stop",
        "breakeven_after_1r",
        "scale_out_half_at_1r",
        # Trail-based kept out of the mass sweep (they add much cost);
        # we'll re-run winners with trail methods in a targeted pass.
    ]


def count_configs() -> dict:
    v1 = build_v1_configs()
    v2 = build_v2_configs()
    return {
        "v1": len(v1),
        "v2": len(v2),
        "total_base": len(v1) + len(v2),
        "with_regime_wrapped": (len(v1) + len(v2)) * 2,
    }


if __name__ == "__main__":
    counts = count_configs()
    print(counts)
