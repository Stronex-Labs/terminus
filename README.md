# Terminus
**End of All Trades**

*Where strategies go to prove themselves — or die.*

Terminus runs your strategies through the gauntlet: 8 years of market data,
year-by-year walk-forward with frozen parameters, realistic fees and
slippage, and a content-hashed simulation store so you never burn compute
twice.

## Why Terminus

Most backtesting tools let you cheat. You tune parameters on the full
window, claim "75/25 train/test", and call it robust. Terminus doesn't.

- **Every active year must be profitable** — frozen parameters walked
  forward year by year. If your strategy had one losing year between 2018
  and 2025, it's dropped.
- **Bear years count** — 2022 was −64% on BTC. Your strategy has to survive
  that with realistic fees and slippage.
- **Multi-pair generalization required** — a strategy that only works on
  one pair is a curve fit. Terminus requires the family to succeed on
  multiple pairs before it's a survivor.
- **Realistic execution** — fees, entry slippage, stop slippage, TP
  slippage, timeout slippage, cooldowns, max-hold. Tiered by market cap so
  small caps aren't flattered.
- **Shariah-compliant by design** — spot-only, no leverage, no shorts, no
  interest-bearing instruments, no lending. Cash (stablecoin) is a valid
  position in bear regimes.
- **Content-hashed cache** — every simulation is keyed by `hash(pair, tf,
  config, date-range, slippage, fee)`. Re-running the same experiment is
  instant. Running a slightly different experiment costs only the delta.

## What's in the box

- **30+ strategy families** — RSI, EMA/MACD crosses, Bollinger, Donchian,
  ATR breakouts, Ichimoku, Supertrend, Keltner, VWAP reclaim, Heikin Ashi,
  ROC momentum, ATR burst, Chandelier, Williams %R, Stochastic, combos
- **6 exit methods** — fixed TP/stop, trailing ATR, Chandelier trailing,
  breakeven-after-1R, scale-out-at-1R, fixed-with-breakeven
- **3 walk-forward modes** — frozen (honest), anchored re-optimization,
  rolling re-optimization
- **Greedy portfolio construction** — correlation-capped leg selection,
  per-year breakdown, realized Sharpe / Calmar / max-DD
- **Vectorized simulator** — numpy-first, 300-700× faster than the scalar
  reference
- **SQLite research store** — 6-table schema: sims, walk_forward_runs,
  sensitivity_runs, pair_data_meta, manifests, logs, portfolio_candidates
- **Binance and pluggable data fetchers** — start with Binance, swap in any
  source with an OHLCV function

## Quickstart

```bash
pip install terminus-lab

# Fetch 8 years of data for 21 pairs across 8 timeframes (~20 min of API time)
terminus fetch --pairs BTCUSDT,ETHUSDT,SOLUSDT --tfs 1h,4h,1d --days 2920

# Run the full sweep (21 pairs × 8 TFs × 332 configs = 55,776 sims, ~40 min)
terminus sweep

# Walk-forward the top candidates year-by-year
terminus walk-forward --top 15

# Generate the report with per-pair ship/no-ship verdicts
terminus report --min-calmar 1.5 --min-bear-return -5
```

## Example output

```
=== TOP 15 SURVIVORS by ANNUALIZED return ===
 Rank  Pair       TF    Family                     Yrs   TotalRet   AnnRet   Calmar   Bear22
 ----- ---------- ----  ------------------------   ----  --------   ------   ------   ------
   1   TIAUSDT    2h    Ichi-bull+BTCreg           3/3    +90.7%    +24.0%    2.77     -
   2   XRPUSDT    2h    Ichi-bull+BTCreg           6/6   +213.5%    +21.0%    5.68    +0.0%
   3   LTCUSDT    12h   Ichi-bull+BTCreg           3/3    +70.1%    +19.4%    4.28    +0.0%
   4   BNBUSDT    1h    ROC10+BTCreg               5/5   +133.2%    +18.4%    4.93    +0.0%
   5   SOLUSDT    4h    ATR-brk                    5/5   +130.6%    +18.2%    7.86   +13.2%
```

## Philosophy

Terminus is **pessimistic by default**. It will tell you your strategy
doesn't work before it tells you it does. That's the point. A strategy
that survives Terminus has a higher probability of surviving live markets.

Terminus is **reproducible by design**. The content-hash cache means any
claim in a report can be traced back to the exact inputs that produced it.
`sims.db` travels with your conclusions.

Terminus is **spot-only, halal-first**. Built by someone who can't use
futures or margin. If you have those tools, you have more alpha available
— but if you don't, Terminus fits your constraints natively instead of
apologizing for them.

## Status

**Alpha**. Used in production (one paper-trading bot trading 7 pairs). API
may shift. Results format is stable. The research data model is stable.

## Contributing

PRs welcome for:
- New strategy families (any rule that can be expressed as a numpy
  vectorized signal)
- Alternative data fetchers (Kraken, Coinbase, OKX, Bybit spot)
- Portfolio construction methods
- Parameter sensitivity analysis
- New walk-forward modes

See `CONTRIBUTING.md`.

## License

MIT. Use it, fork it, ship it. If you find something useful, open a PR —
the community sim database grows stronger with every contributor.

## Credits

Born out of a live spot-trading bot after the author got tired of backtest
frameworks that flatter strategies into production.
