# Terminus

> **End of All Trades.** A ruthless backtesting lab for long-only spot crypto strategies — where strategies prove themselves, or die.

Terminus is a Python backtesting and walk-forward validation engine for **long-only, spot-only** cryptocurrency strategies. It is built around one principle: **make it hard to cheat.** Most backtesting tools make overfitting easy; Terminus makes it expensive.

```bash
pip install terminus-lab
```

## Why Terminus

- **Walk-forward required.** Frozen parameters tested year by year — at most two losing years tolerated, none worse than −10%.
- **Bear years count.** 2022 was −64% on BTC. A strategy survives that leg or it doesn't ship.
- **Multi-pair generalization.** Works on one pair only? That's a curve fit — Terminus requires success across ≥5 pairs.
- **Realistic execution.** Tiered slippage by market cap, maker/taker fees, cooldowns, max-hold timeouts.
- **Exit-model fidelity.** The #1 silent backtest trap — an exit that fills on intra-bar wicks your live engine never sees. Terminus makes it explicit and models it worst-case. See [Exit-Model Fidelity](exit-check-fidelity.md).
- **Halal-first.** Spot-only, no leverage, no shorts, no interest. Cash is a valid position.

## Where to go next

- **[Quickstart](quickstart.md)** — from `pip install` to a survivor report in five commands.
- **[Exit-Model Fidelity](exit-check-fidelity.md)** — the `exit_check` model and why matching your live closer matters.
- **[Strategy Families](strategy-families.md)** — the built-in rule families and how they're swept.
- **[CLI Reference](cli-reference.md)** — every `terminus` command and flag.

## Links

- **PyPI:** <https://pypi.org/project/terminus-lab/>
- **Source:** <https://github.com/Stronex-Labs/terminus>
- **License:** MIT
