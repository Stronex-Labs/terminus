<p align="center">
  <img src="https://raw.githubusercontent.com/Stronex-Labs/terminus/main/assets/logo.png" alt="Terminus" width="320"/>
</p>

<h1 align="center">Terminus</h1>
<p align="center"><em>End of All Trades</em></p>

<p align="center">
  <a href="https://pypi.org/project/terminus-lab/"><img src="https://img.shields.io/pypi/v/terminus-lab?style=flat-square&color=blue" /></a>
  <a href="https://pypi.org/project/terminus-lab/"><img src="https://img.shields.io/pypi/dm/terminus-lab?style=flat-square&color=blue&label=installs" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/halal--first-spot%20only-gold?style=flat-square" />
  <img src="https://img.shields.io/github/stars/Stronex-Labs/terminus?style=flat-square" />
</p>

<p align="center"><strong>Where strategies go to prove themselves. Or die.</strong></p>

---

Terminus is a pure Python backtesting engine built for one thing: **honest results**.

8 years of market data. Year-by-year walk-forward with frozen parameters. Realistic fees and slippage. A content-hashed simulation store so you never burn compute twice.

Most backtesting tools let you cheat. Terminus doesn't.

---

## Why Terminus

- **Near-perfect walk-forward required.** Frozen parameters tested year by year. At most 2 losing years tolerated, each no worse than -10%.
- **Bear years count.** 2022 was -64% on BTC. Your strategy survives that or it doesn't ship.
- **Multi-pair generalization required.** Works on one pair? That's a curve fit. Terminus requires success across multiple pairs.
- **Realistic execution.** Fees, entry slippage, stop slippage, TP slippage, timeout slippage, cooldowns, max-hold. Tiered by market cap.
- **Halal-first.** Spot-only, no leverage, no shorts, no interest. Cash (stablecoin) is a valid position in bear regimes.
- **Content-hashed cache.** Every sim keyed by `hash(pair + tf + config + date_range + slippage + fee)`. Same inputs = instant cache hit.

---

## Quickstart

```bash
pip install terminus-lab

# Fetch 8 years of data
terminus fetch --pairs BTCUSDT,ETHUSDT,SOLUSDT --tfs 1h,4h,1d --days 2920

# Run the full sweep
terminus sweep

# Walk-forward the top candidates year-by-year
terminus walk-forward --top 15

# Generate the report
terminus report --min-calmar 1.5 --min-bear-return -5
```

---

## Example Output

```
=== TOP 15 SURVIVORS by ANNUALIZED return ===
 Rank  Pair       TF    Family                   Yrs   TotalRet   AnnRet   Calmar   Bear22
 ----- ---------- ----  -----------------------  ----  --------   ------   ------   ------
   1   TIAUSDT    2h    Ichi-bull+BTCreg          3/3    +90.7%    +24.0%    2.77     -
   2   XRPUSDT    2h    Ichi-bull+BTCreg          6/6   +213.5%    +21.0%    5.68    +0.0%
   3   LTCUSDT    12h   Ichi-bull+BTCreg          3/3    +70.1%    +19.4%    4.28    +0.0%
   4   BNBUSDT    1h    ROC10+BTCreg              5/5   +133.2%    +18.4%    4.93    +0.0%
   5   SOLUSDT    4h    ATR-brk                   5/5   +130.6%    +18.2%    7.86   +13.2%
```

---

## What's Inside

| Module | What it does |
|--------|-------------|
| **Sweep engine** | Runs thousands of configs in parallel, skips cache hits |
| **Walk-forward** | Frozen / anchored / rolling, calendar-year folds |
| **Portfolio builder** | Correlation-capped leg selection, blended Sharpe/Calmar |
| **ML module** | XGBoost regime classifier (BULL/BEAR/CHOP) + WF-aware optimizer |
| **Community hub** | Contribute sim results, query the global leaderboard |
| **SQLite store** | Local 6-table schema, travels with your research |

**30+ strategy families:** RSI, EMA/MACD crosses, Bollinger, Donchian, ATR breakouts, Ichimoku, Supertrend, Keltner, VWAP reclaim, Heikin Ashi, ROC momentum, and more.

---

## Philosophy

Terminus is **pessimistic by default**. It will tell you your strategy doesn't work before it tells you it does. That's the point.

Terminus is **reproducible by design**. The content-hash cache means any claim in a report traces back to the exact inputs that produced it. `sims.db` travels with your conclusions.

Terminus is **spot-only, halal-first**. Built by someone who can't use futures or margin. If you have those tools, you have more alpha. If you don't, Terminus fits your constraints natively.

---

## Community Hub

**Leaderboard:** https://terminus-hub.shatla-tech.workers.dev

After every `terminus sweep` and `terminus walk-forward`, Terminus automatically contributes your survivors to the community hub. No manual step needed. Sims are always stored locally first; remote send is retried on the next run if the network was down.

To opt out:
```bash
export TERMINUS_TELEMETRY=0
```

To force-upload everything you've ever run:
```bash
terminus contribute --all
```

---

## Status

**Alpha.** Used in production on a paper-trading bot running 7 pairs. API may shift. Results format and research data model are stable.

---

## Contributing

PRs welcome for:
- New strategy families (numpy-vectorized signals)
- Alternative data fetchers (Kraken, Coinbase, OKX, Bybit spot)
- Portfolio construction methods
- New walk-forward modes

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT. Use it, fork it, ship it. If you find something useful, open a PR. The community sim database grows stronger with every contributor.
