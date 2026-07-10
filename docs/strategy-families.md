# Strategy Families

Terminus sweeps a registry of rule families across every configured timeframe. Each family expands into many parameter configs (TP / stop / hold / cooldown), and each also has a `+BTCreg` variant that gates entries on the BTC regime classifier.

| Category | Families | Examples |
|----------|----------|----------|
| **Trend** | EMA crosses, MACD, Supertrend, Ichimoku | `EMA9/21-cross`, `Ichi-bull+BTCreg` |
| **Momentum** | RSI, ROC, Stochastic, Williams %R, ADX-surge, N-of-4 sniper | `RSI-cross-30`, `Mom-sniper[3of4]` |
| **Volatility** | Bollinger Bands, ATR breakout, Keltner | `ATR-brk-1.5`, `Keltner-brk` |
| **Channel** | Donchian, VWAP reclaim | `Donch20-brk`, `VWAP-reclaim` |
| **Price Action** | Heikin Ashi, pullback | `HA-reversal`, `EMA-pullback` |
| **Composite** | Multi-indicator combos | `RSI+BB+MACD`, `EMA+Vol-confirm` |

## Mom-sniper (N-of-4 momentum vote)

`momentum_sniper(min_conditions=3)` fires when at least `min_conditions` of four conditions hold on the bar:

1. `ema9 > ema20` — short-term uptrend
2. `rsi` in `[50, 70]` — momentum, not overbought
3. `vol_ratio > 1.3` — volume expansion vs its 20-bar average
4. `adx > 20` — trend strength

It ships in `3-of-4` and `4-of-4` configs. This family requires the `adx` column, which `precompute_all` computes as ADX(14).

## Exit methods

Each config can be simulated under any of six exit methods (`fixed_tp_stop`, `atr_trail`, `chandelier_trail`, `breakeven_after_1r`, `fixed_with_breakeven`, `scale_out_half_at_1r`). Orthogonally, the **[exit-model fidelity](exit-check-fidelity.md)** (`exit_check=path|discrete`) controls how the exit samples intra-bar price. Fidelity and method are independent choices — set both to match your live engine.

## Adding a family

Rules live in `terminus/rules.py` as vectorized `VRule` objects; configs are registered in `terminus/registry.py`. Any indicator a rule reads must be produced by `precompute_all` in `terminus/indicators.py` — a rule that reads a missing column silently never fires.
