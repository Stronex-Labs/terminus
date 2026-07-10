# CLI Reference

All commands are subcommands of `terminus`. Run `terminus <command> --help` for the full, authoritative flag list — this page is a map, and the `README` on GitHub carries the exhaustive per-flag tables.

| Command | Purpose |
|---------|---------|
| `terminus fetch` | Download and cache klines (parquet under `kline_cache/`). Repeat runs are instant. |
| `terminus sweep` | Run every family × timeframe × parameter config through the realistic-execution simulator. |
| `terminus walk-forward` | Re-test the top candidates year by year with frozen parameters. |
| `terminus report` | Generate the survivor report from sweep + walk-forward results. |
| `terminus portfolio` | Build a correlation-capped, Sharpe-ranked portfolio from survivors. |
| `terminus contribute` | Package a validated result for the community hub. |
| `terminus ml regime` | Train / apply the BULL/BEAR/CHOP regime classifier (requires `[ml]`). |
| `terminus ml train` | Train the LightGBM config screener (requires `[ml]`). |

## Common flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--days` | `2920` | History window to fetch (8 years). Newer pairs auto-degrade to available history. |
| `--exit-methods` | `fixed_tp_stop` | Comma-separated exit methods to sweep. |
| `--exit-check` | `path` | Exit-model fidelity — `path` or `discrete`. See [Exit-Model Fidelity](exit-check-fidelity.md). |

## Typical presets

```bash
# Full discovery sweep (default exit method)
terminus sweep

# Enable all trailing / scale-out exit methods (5x the configs)
terminus sweep --exit-methods fixed_tp_stop,atr_trail,chandelier_trail,breakeven_after_1r,scale_out_half_at_1r

# Match a discrete scan-loop live engine
terminus sweep --exit-check discrete
```

!!! note
    `--exit-check` and `--exit-methods` are independent. The method is *what* the exit does; the check is *how faithfully* intra-bar price is sampled. Set both to mirror your live closer.
