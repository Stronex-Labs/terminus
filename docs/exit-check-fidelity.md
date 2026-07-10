# Exit-Model Fidelity

The single biggest reason a backtest lies to you is not the entry signal — it's the **exit model**. If your simulator resolves exits differently from how your live engine actually samples price, your conclusions can *invert*: a losing strategy can backtest as a winner, and vice-versa.

Terminus makes this explicit with one argument.

## `exit_check`: `path` vs `discrete`

```python
from terminus.simulate import simulate_fast

# intra-bar path replay (default)
res = simulate_fast(df, rule, tp_pct=0.04, stop_pct=0.04,
                    max_hold_bars=36, exit_check="path")

# discrete per-cycle sampling
res = simulate_fast(df, rule, tp_pct=0.04, stop_pct=0.04,
                    max_hold_bars=36, exit_check="discrete")
```

| Mode | The exit sees | Use when your live closer… |
|------|---------------|-----------------------------|
| `path` *(default)* | the bar's **high and low** — a stop/TP fills when the intra-bar wick touches its level | reconstructs the intra-bar path (e.g. a tick-tight 1-minute trailing stop) |
| `discrete` | only each bar's **close**, and fills at that close — intra-bar wicks are invisible | samples one current price per scan cycle (a loop that reads the latest price every N seconds) |

The same entries under the two modes legitimately produce different results. **Pick the one that matches your live execution.** A tight trailing stop that your live bot enforces intra-bar is `path`; a slow scan-loop that only ever sees the last close is `discrete`.

## `path` mode is modelled worst-case

Path mode does **not** give the strategy the benefit of the doubt on intra-bar ordering. Two guarantees:

### 1. A bar's high cannot dodge its own low

Within a single bar you do not know whether the high or the low came first. Terminus assumes the **adverse** order: a bar's **low is tested against the stop carried from prior bars before that same bar's high can ratchet the trailing stop up.** A real stop cannot use a bar's own high to escape its own low — the naïve "ratchet-then-check" loop is the classic optimistic path-replay bug, and it is exactly what Terminus avoids.

### 2. No fill above the bar's own high (gap realism)

When price gaps entirely below a stop, a real order fills at the gapped price, not at the untouched stop level. Terminus clamps the fill to `min(stop, bar_high)` so a stop **never** prints an exit above the traded range of the bar it fired on. This applies to both the vectorized `fixed_tp_stop` path and the trailing loop.

Together these make `path` mode *conservative* — it will not manufacture exits that a live venue could not have given you.

## Why this matters in practice

A strategy validated under an optimistic exit model can show tens of basis points of edge per trade that simply does not exist once fills are honest. If you tune knobs (trail distance, stop width, timeout) against a flattering exit model, you optimize for an artifact. Match the model to your live closer first; tune second.

!!! tip "Rule of thumb"
    If you are unsure which mode matches your live setup, run both. If the strategy only survives under `path` *and* your live engine samples discretely (or vice-versa), you have found a fidelity gap — not an edge.
