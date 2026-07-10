"""Vectorized realistic-execution simulator.

Same semantics as simulate_v2 but rewritten with numpy arrays and a
two-phase design:

  Phase A — scan entries:
    Call `entry_rule_vectorized(df) -> bool ndarray` ONCE to get all signal
    indices. If a rule is only available as a scalar (i, df) -> bool
    callable, we fall back to evaluating once per bar using a cached
    dataframe — still faster than simulate_v2 because the exit loop is
    vectorized.

  Phase B — resolve each entry:
    For each signal index, locate the exit bar via vectorized numpy
    comparisons on the high/low arrays from entry to entry+max_hold.

Typical speedup: 10-50x on 1h+ data, 50-200x on 15m/30m where the bar
count matters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


LOOKBACK_BARS = 200


def _compile_signals(entry_rule, df, lookback=LOOKBACK_BARS) -> np.ndarray:
    """Evaluate the entry rule across every bar. Returns bool array."""
    # Fast path — if the rule has a `.vectorized_signal(df)` method
    vec = getattr(entry_rule, "vectorized_signal", None)
    if callable(vec):
        sig = vec(df)
        arr = np.asarray(sig, dtype=bool)
        if arr.shape[0] != len(df):
            raise ValueError("vectorized signal length mismatch")
        return arr
    # Scalar fallback
    n = len(df)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        try:
            out[i] = bool(entry_rule(i, df))
        except Exception:
            out[i] = False
    return out


def simulate_fast(
    df: pd.DataFrame, entry_rule: Callable,
    tp_pct: float, stop_pct: float,
    max_hold_bars: int, cooldown_bars: int,
    *,
    fee_rate: float = 0.00075,
    entry_slip: float = 0.0005,
    stop_slip: float = 0.0010,
    tp_slip: float = 0.0002,
    timeout_slip: float = 0.0005,
    latency_bars: int = 0,
    exit_method: str = "fixed_tp_stop",
    atr_mult_trail: float = 3.0,
    exit_check: str = "path",
) -> dict:
    """Vectorized simulator. Same output shape as simulate_v2.

    exit_check controls the FIDELITY of the exit-check model — it must match how
    the live engine samples price, or conclusions can invert (the #1 backtest
    trap, see strategy-lever-lab):

      "path"     — intra-bar path replay: the exit sees the bar's high AND low
                   (a stop/TP fills at its level when the wick touches it). Use
                   when the live closer reconstructs the intra-bar path
                   (e.g. a tick-tight 1m trail).
      "discrete" — discrete per-cycle sampling: the exit only ever sees the
                   CLOSE of each bar (the price a scan-loop closer observes), and
                   fills AT that close. Intra-bar wicks are invisible. Use when
                   the live closer samples one current price per cycle.

    Same entries, different exit-check model => legitimately different results.
    """
    n = len(df)
    if n < LOOKBACK_BARS + max_hold_bars + 2:
        return _empty_result()

    # Extract numpy arrays ONCE (avoids per-bar .iloc[] overhead)
    open_arr = df["open"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    ot_arr = df["open_time"].values.astype(np.int64)
    atr_arr = (df["atr"].values.astype(np.float64)
               if "atr" in df.columns else np.full(n, np.nan))

    # Phase A — compile entries
    sig = _compile_signals(entry_rule, df)

    # Phase B — resolve exits, honoring cooldown + open-trade window
    signals = []
    in_open_until = -1
    last_exit_i = -cooldown_bars
    last_i = n - max_hold_bars

    # Iterate only over signal indices
    sig_idx = np.where(sig[:last_i])[0]

    for i in sig_idx:
        if i < LOOKBACK_BARS:
            continue
        if i < in_open_until or (i - last_exit_i) < cooldown_bars:
            continue

        entry_i = i + 1 + latency_bars
        if entry_i >= n:
            break
        fill_open = open_arr[entry_i]
        entry_price = fill_open * (1 + entry_slip)
        initial_stop = entry_price * (1 - stop_pct)
        initial_tp = entry_price * (1 + tp_pct)
        one_r_up = entry_price * (1 + stop_pct)
        half_r_up = entry_price * (1 + 0.5 * stop_pct)

        # Window of bars to scan
        end = min(entry_i + max_hold_bars, n)
        hs = high_arr[entry_i:end]
        ls = low_arr[entry_i:end]
        cs = close_arr[entry_i:end]
        ats = atr_arr[entry_i:end]

        # Exit-check fidelity: in discrete mode the closer only ever sees the
        # CLOSE each cycle (no intra-bar wick), and fills at that close.
        discrete = (exit_check == "discrete")
        hi_chk = cs if discrete else hs
        lo_chk = cs if discrete else ls

        # --- Vectorized exit resolution ---
        # For simple fixed_tp_stop we can find the first bar whose low<=stop
        # or high>=tp in a single pass. For path-dependent trailing exits,
        # we still need a tiny python loop, but only over the trade's window,
        # not the whole df.
        exit_offset = None
        exit_price = None
        exit_reason = None
        scale_out_triggered = False
        scale_out_pnl = 0.0

        if exit_method == "fixed_tp_stop":
            # Either bar first touches stop OR tp. Stop takes precedence if both
            # in same bar (conservative). Discrete fills at the observed close;
            # path fills at the stop/tp level the wick traded through.
            stop_hits = np.where(lo_chk <= initial_stop)[0]
            tp_hits = np.where(hi_chk >= initial_tp)[0]
            if len(stop_hits) == 0 and len(tp_hits) == 0:
                pass
            else:
                stop_first = stop_hits[0] if len(stop_hits) else len(ls)
                tp_first = tp_hits[0] if len(tp_hits) else len(ls)
                if stop_first <= tp_first:
                    exit_offset = stop_first
                    # gap-realism: a stop never fills above the bar's own high —
                    # if the bar gapped entirely below the stop you get the worse
                    # gapped price, not the untouched level (else a phantom fill).
                    _fill = cs[stop_first] if discrete else min(initial_stop, hs[stop_first])
                    exit_price = _fill * (1 - stop_slip)
                    exit_reason = "STOP"
                else:
                    exit_offset = tp_first
                    exit_price = (cs[tp_first] if discrete else initial_tp) * (1 - tp_slip)
                    exit_reason = "TP"

        else:
            # Path-dependent: loop bars in the window (still fast — small window)
            current_stop = initial_stop
            running_high = entry_price
            breakeven_armed = False

            for k in range(len(hs)):
                high = hi_chk[k]
                low = lo_chk[k]

                # WORST-CASE intra-bar ordering: test this bar's LOW against the
                # stop carried from PRIOR bars BEFORE this bar's HIGH can ratchet
                # the trail (the ratchet is deferred to the bottom of the loop, so
                # it only protects the NEXT bar). A real stop can't use a bar's own
                # high to dodge its own low — ratchet-then-check is the classic
                # optimistic path-replay bug (strategy-lever-lab closer-fidelity).
                if low <= current_stop:
                    exit_offset = k
                    # gap-realism: never fill above THIS bar's high — a gap-through
                    # fills at the (worse) gapped price, not the untouched level.
                    _fill = cs[k] if discrete else min(current_stop, high)
                    exit_price = _fill * (1 - stop_slip)
                    exit_reason = ("STOP" if current_stop <= initial_stop * 1.0001
                                   else "TRAIL")
                    break
                # fixed TP check (level is order-independent; stop takes precedence)
                if exit_method in ("fixed_tp_stop", "breakeven_after_1r",
                                    "fixed_with_breakeven") and high >= initial_tp:
                    exit_offset = k
                    exit_price = (cs[k] if discrete else initial_tp) * (1 - tp_slip)
                    exit_reason = "TP"
                    break

                # --- advance peak + trailing stop, for the NEXT bar only ---
                running_high = max(running_high, high)
                if exit_method in ("atr_trail", "chandelier_trail"):
                    if not np.isnan(ats[k]):
                        trail = running_high - atr_mult_trail * ats[k]
                        if trail > current_stop:
                            current_stop = trail
                elif exit_method == "breakeven_after_1r":
                    if not breakeven_armed and high >= one_r_up:
                        be = entry_price * (1 + fee_rate * 2)
                        if be > current_stop:
                            current_stop = be
                        breakeven_armed = True
                elif exit_method == "fixed_with_breakeven":
                    if not breakeven_armed and high >= half_r_up:
                        be = entry_price * (1 + fee_rate * 2)
                        if be > current_stop:
                            current_stop = be
                        breakeven_armed = True
                elif exit_method == "scale_out_half_at_1r":
                    if not scale_out_triggered and high >= one_r_up:
                        half_exit = one_r_up * (1 - tp_slip)
                        scale_out_pnl = 0.5 * ((half_exit - entry_price) / entry_price)
                        scale_out_triggered = True
                        be = entry_price * (1 + fee_rate * 2)
                        if be > current_stop:
                            current_stop = be
                    if scale_out_triggered and not np.isnan(ats[k]):
                        trail = running_high - atr_mult_trail * ats[k]
                        if trail > current_stop:
                            current_stop = trail

        if exit_offset is None:
            exit_offset = len(hs) - 1
            exit_price = cs[exit_offset] * (1 - timeout_slip)
            exit_reason = "TIMEOUT"

        exit_i = entry_i + exit_offset
        remaining = 1.0 - (0.5 if scale_out_triggered else 0.0)
        position_pnl = remaining * ((exit_price - entry_price) / entry_price)
        pnl_pct = scale_out_pnl + position_pnl

        signals.append({
            "entry_ts": int(ot_arr[entry_i]),
            "exit_ts": int(ot_arr[exit_i]),
            "entry_price": round(entry_price, 8),
            "exit_price": round(exit_price, 8),
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "scale_out": scale_out_triggered,
        })
        in_open_until = exit_i + 1
        last_exit_i = exit_i

    return _finalize(signals, fee_rate)


def _finalize(signals, fee_rate):
    if not signals:
        return _empty_result()

    wins = [s for s in signals if s["pnl_pct"] > 0]
    total_pnl = sum(s["pnl_pct"] for s in signals)
    avg_pnl = total_pnl / len(signals)
    net_avg = avg_pnl - 2 * fee_rate
    total_net = total_pnl - 2 * fee_rate * len(signals)

    START_BALANCE = 1000.0
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    for s in signals:
        balance *= (1 + s["pnl_pct"] - 2 * fee_rate)
        peak = max(peak, balance)
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd

    first_ts = signals[0]["entry_ts"]
    last_ts = signals[-1]["entry_ts"]
    span_days = max((last_ts - first_ts) / (1000 * 86400), 1)
    trades_per_month = len(signals) / span_days * 30

    return {
        "n": len(signals),
        "win_rate": len(wins) / len(signals) * 100,
        "avg_pnl": avg_pnl * 100,
        "net_avg_pnl": net_avg * 100,
        "total_net": total_net * 100,
        "trades_per_month": trades_per_month,
        "final_balance": balance,
        "max_drawdown": max_dd * 100,
        "trades": signals,
    }


def _empty_result():
    return {
        "n": 0, "win_rate": 0.0, "avg_pnl": 0.0,
        "net_avg_pnl": 0.0, "total_net": 0.0, "trades_per_month": 0.0,
        "final_balance": 1000.0, "max_drawdown": 0.0, "trades": [],
    }


# --- Tiered slip model ---------------------------------------------------
# Majors get tight slip; small-caps get punished. Override by passing a
# custom dict to simulate_fast's slip_for hook, or extend these sets.
TIER_MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"}
TIER_MIDS = {"ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
             "MATICUSDT", "TRXUSDT", "DOGEUSDT"}
TIER_SMALLS = {"SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "ATOMUSDT",
               "TIAUSDT", "INJUSDT", "NEARUSDT"}


def slip_for(pair: str) -> dict:
    """Return a slip dict {entry_slip, stop_slip, tp_slip, timeout_slip}.

    Conservative retail estimates at ~$10k trade size. Majors get tight
    slip (0.05% entry), mids get 2x (0.10%), smalls get 4x (0.20%).
    Unknown pairs are treated as mid.
    """
    p = pair.upper()
    if p in TIER_MAJORS:
        return dict(entry_slip=0.0005, stop_slip=0.0010,
                    tp_slip=0.0002, timeout_slip=0.0005)
    if p in TIER_MIDS:
        return dict(entry_slip=0.0010, stop_slip=0.0015,
                    tp_slip=0.0003, timeout_slip=0.0010)
    if p in TIER_SMALLS:
        return dict(entry_slip=0.0020, stop_slip=0.0030,
                    tp_slip=0.0005, timeout_slip=0.0020)
    return dict(entry_slip=0.0010, stop_slip=0.0015,
                tp_slip=0.0003, timeout_slip=0.0010)


def tier_of(pair: str) -> str:
    p = pair.upper()
    if p in TIER_MAJORS:
        return "major"
    if p in TIER_MIDS:
        return "mid"
    if p in TIER_SMALLS:
        return "small"
    return "unknown"
