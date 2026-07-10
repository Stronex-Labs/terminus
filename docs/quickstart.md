# Quickstart

## Install

```bash
pip install terminus-lab
# with the ML screener / regime classifier:
pip install "terminus-lab[ml]"
```

Requires Python 3.10+.

## Verify

```bash
terminus --help
```

## The five-command pipeline

```bash
# 1. Fetch history (pairs x timeframes, cached under kline_cache/ as parquet)
terminus fetch --days 2920

# 2. Run the full parameter sweep across every family x timeframe
terminus sweep

# 3. Walk-forward the top candidates year by year (frozen params)
terminus walk-forward

# 4. Generate the survivor report
terminus report

# 5. Build a correlation-capped portfolio from the survivors
terminus portfolio
```

## Minimal single-pair run in Python

```python
import pandas as pd
from terminus.indicators import precompute_all
from terminus import rules as rv
from terminus.simulate import simulate_fast

df = pd.read_parquet("BTCUSDT_4h.parquet")   # columns: open, high, low, close, volume, open_time
df = precompute_all(df)

rule = rv.momentum_sniper(min_conditions=3)  # N-of-4 momentum vote
res = simulate_fast(
    df, rule,
    tp_pct=0.04, stop_pct=0.04, max_hold_bars=36, cooldown_bars=2,
    exit_check="path",        # match your live closer — see Exit-Model Fidelity
)

print(len(res["trades"]), "trades")
print(res["trades"][0])
```

!!! warning "Exit model first"
    Before you trust any result, make sure `exit_check` matches how your live engine samples price. See [Exit-Model Fidelity](exit-check-fidelity.md) — mismatching it is how a losing strategy backtests as a winner.

## Next

- [Strategy Families](strategy-families.md) — what gets swept.
- [CLI Reference](cli-reference.md) — every command and flag.
