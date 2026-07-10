"""exit_check fidelity: path vs discrete must diverge in the known direction.

The #1 backtest trap (strategy-lever-lab): a path-replay exit sees intra-bar
wicks the live discrete-per-cycle closer never sees. This test pins the
behavioural difference so the two modes can't silently converge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from terminus.simulate import simulate_fast, LOOKBACK_BARS


class _SignalAt:
    """Entry rule that fires on exactly one bar (vectorized fast path)."""
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def vectorized_signal(self, df: pd.DataFrame) -> np.ndarray:
        a = np.zeros(len(df), dtype=bool)
        a[self.idx] = True
        return a


def _flat_df(n: int) -> pd.DataFrame:
    """Flat 100.0 OHLC — no stop/TP triggers unless we inject a wick."""
    return pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.5),
        "low": np.full(n, 99.5),
        "close": np.full(n, 100.0),
        "open_time": np.arange(n, dtype=np.int64) * 60000,
    })


def _one_trade_df():
    n = LOOKBACK_BARS + 12
    df = _flat_df(n)
    sig_i = LOOKBACK_BARS          # entry resolves at sig_i + 1
    wick_bar = sig_i + 2           # within the max_hold window
    # Intra-bar LOW pierces the 5% stop, but the bar CLOSES back at 100.
    df.loc[wick_bar, "low"] = 94.0
    return df, _SignalAt(sig_i)


def test_path_catches_stop_wick_discrete_misses_it():
    df, rule = _one_trade_df()
    kw = dict(tp_pct=0.05, stop_pct=0.05, max_hold_bars=6, cooldown_bars=0)

    path = simulate_fast(df, rule, exit_check="path", **kw)
    disc = simulate_fast(df, rule, exit_check="discrete", **kw)

    # Exactly one trade each
    assert len(path["trades"]) == 1
    assert len(disc["trades"]) == 1
    pt, dt = path["trades"][0], disc["trades"][0]

    # PATH sees the 94.0 wick -> STOP, a ~5% loss.
    assert pt["exit_reason"] == "STOP"
    assert pt["pnl_pct"] < -0.04

    # DISCRETE only sees the 100.0 close -> never stops on the wick; rides to
    # TIMEOUT near flat. The loss is an order of magnitude smaller.
    assert dt["exit_reason"] == "TIMEOUT"
    assert dt["pnl_pct"] > -0.01

    # The whole point: same entry, materially different outcome.
    assert pt["pnl_pct"] < dt["pnl_pct"] - 0.03


def test_path_catches_tp_wick_discrete_misses_it():
    n = LOOKBACK_BARS + 12
    df = _flat_df(n)
    sig_i = LOOKBACK_BARS
    df.loc[sig_i + 2, "high"] = 106.0   # spikes through the 5% TP, closes at 100
    rule = _SignalAt(sig_i)
    kw = dict(tp_pct=0.05, stop_pct=0.05, max_hold_bars=6, cooldown_bars=0)

    path = simulate_fast(df, rule, exit_check="path", **kw)
    disc = simulate_fast(df, rule, exit_check="discrete", **kw)

    assert path["trades"][0]["exit_reason"] == "TP"
    assert path["trades"][0]["pnl_pct"] > 0.04
    assert disc["trades"][0]["exit_reason"] == "TIMEOUT"   # close never reached TP
    assert disc["trades"][0]["pnl_pct"] < 0.01


def test_default_is_path_unchanged():
    """Default (no exit_check) must equal explicit path — no silent regression."""
    df, rule = _one_trade_df()
    kw = dict(tp_pct=0.05, stop_pct=0.05, max_hold_bars=6, cooldown_bars=0)
    default = simulate_fast(df, rule, **kw)
    path = simulate_fast(df, rule, exit_check="path", **kw)
    assert default["trades"] == path["trades"]


def test_path_gap_through_stop_no_phantom_fill():
    """A bar that gaps ENTIRELY below the stop must fill at the bar's own high,
    never at the untouched stop level (that would be a phantom fill above the
    traded range). gap-realism clamp: exit_price <= this bar's high * (1-slip)."""
    n = LOOKBACK_BARS + 12
    df = _flat_df(n)
    sig_i = LOOKBACK_BARS
    gap_bar = sig_i + 2
    # Whole bar gaps below the ~5% stop (~95.2): high 90 < stop, low 88.
    df.loc[gap_bar, "high"] = 90.0
    df.loc[gap_bar, "low"] = 88.0
    df.loc[gap_bar, "open"] = 89.0
    df.loc[gap_bar, "close"] = 89.0
    rule = _SignalAt(sig_i)
    kw = dict(tp_pct=0.05, stop_pct=0.05, max_hold_bars=6, cooldown_bars=0)

    path = simulate_fast(df, rule, exit_check="path", **kw)
    t = path["trades"][0]
    assert t["exit_reason"] == "STOP"
    # No phantom: the fill cannot be above the bar's high (90.0).
    assert t["exit_price"] <= 90.0
    # Loss is worse than the nominal ~5% stop (it gapped through).
    assert t["pnl_pct"] < -0.09


def test_path_worst_case_ordering_high_cannot_dodge_own_low():
    """A bar's HIGH must not ratchet the stop up in time to convert that SAME
    bar's stop-piercing LOW into a profitable exit. Worst-case ordering: the low
    is tested against the PRIOR stop first, so a bar that both spikes to +1R and
    wicks below the initial stop exits as a STOP (loss), not a breakeven win."""
    n = LOOKBACK_BARS + 12
    df = _flat_df(n)
    sig_i = LOOKBACK_BARS
    bar = sig_i + 2
    # Same bar: high pierces +1R (would arm breakeven), low pierces the -5% stop.
    df.loc[bar, "high"] = 106.0   # >= one_r_up (~105) -> would-be breakeven arm
    df.loc[bar, "low"] = 94.0     # <  initial_stop (~95.2)
    rule = _SignalAt(sig_i)
    kw = dict(tp_pct=0.20, stop_pct=0.05, max_hold_bars=6, cooldown_bars=0,
              exit_method="breakeven_after_1r")

    path = simulate_fast(df, rule, exit_check="path", **kw)
    t = path["trades"][0]
    # Optimistic ratchet-then-check would arm BE off the 106 high and exit ~flat;
    # worst-case exits at the initial stop for a real loss.
    assert t["exit_reason"] == "STOP"
    assert t["pnl_pct"] < -0.04
