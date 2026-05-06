<p align="center">
  <img src="https://raw.githubusercontent.com/Stronex-Labs/terminus/main/assets/logo.png" width="180" alt="Terminus Logo"/>
</p>

<h1 align="center">Terminus</h1>
<p align="center"><em>End of All Trades</em></p>

<p align="center">
  <b>Ruthless backtesting lab for long-only spot strategies. Where strategies prove themselves, or die.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/terminus-lab/"><img src="https://img.shields.io/pypi/v/terminus-lab?style=flat&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/terminus-lab/"><img src="https://img.shields.io/pypi/dm/terminus-lab?style=flat&color=blue&label=installs" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"></a>
  <img src="https://img.shields.io/badge/status-alpha-orange?style=flat" alt="Status">
  <br>
  <img src="https://img.shields.io/badge/Strategies-30%2B-E67E22" alt="Strategies">
  <img src="https://img.shields.io/badge/Exit_Methods-6-7C3AED" alt="Exit Methods">
  <img src="https://img.shields.io/badge/Timeframes-6-0F766E" alt="Timeframes">
  <img src="https://img.shields.io/badge/Halal_First-Spot_Only-DAA520" alt="Halal First">
  <br>
  <a href="https://github.com/Stronex-Labs/terminus"><img src="https://img.shields.io/github/stars/Stronex-Labs/terminus?style=flat" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-why-terminus">Why</a> &nbsp;&middot;&nbsp;
  <a href="#-key-features">Features</a> &nbsp;&middot;&nbsp;
  <a href="#-quickstart">Quickstart</a> &nbsp;&middot;&nbsp;
  <a href="#-cli-reference">CLI</a> &nbsp;&middot;&nbsp;
  <a href="#-strategy-families">Strategies</a> &nbsp;&middot;&nbsp;
  <a href="#-execution-model">Execution</a> &nbsp;&middot;&nbsp;
  <a href="#-ml-module">ML</a> &nbsp;&middot;&nbsp;
  <a href="#-community-hub">Hub</a> &nbsp;&middot;&nbsp;
  <a href="#-project-structure">Structure</a> &nbsp;&middot;&nbsp;
  <a href="#-roadmap">Roadmap</a> &nbsp;&middot;&nbsp;
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <code>pip install terminus-lab</code>
</p>

---

## 💡 Why Terminus

Most backtesting tools make it easy to overfit. Terminus makes it **hard to cheat**.

- **Walk-forward required.** Frozen parameters tested year by year. At most 2 losing years tolerated, each no worse than -10%.
- **Bear years count.** 2022 was -64% on BTC. Your strategy survives that or it doesn't ship.
- **Multi-pair generalization.** Works on one pair? That's a curve fit. Terminus requires success across multiple pairs.
- **Realistic execution.** Tiered slippage by market cap, maker/taker fees, cooldowns, max-hold timeouts.
- **Halal-first.** Spot-only, no leverage, no shorts, no interest. Cash is a valid position.
- **Content-hashed cache.** Every sim keyed by `SHA-256(pair + tf + config + dates + slippage + fee)`. Same inputs = instant hit.

---

## ✨ Key Features

<table>
<tr>
<td width="25%" valign="top">

### Sweep Engine
Runs thousands of configs in parallel. Content-hash dedup skips cache hits automatically. Never burn compute twice.

</td>
<td width="25%" valign="top">

### Walk-Forward
Frozen, anchored, and 75/25 split modes. Calendar-year folds. Bear year 2022 is the litmus test.

</td>
<td width="25%" valign="top">

### Portfolio Builder
Greedy correlation-capped leg selection. Blended Sharpe/Calmar ranking. Daily P&L reconstruction.

</td>
<td width="25%" valign="top">

### ML Module
XGBoost regime classifier (BULL/BEAR/CHOP). LightGBM screener. Walk-forward-aware optimizer.

</td>
</tr>
</table>

### 30+ Strategy Families

| Category | Families | Examples |
|----------|----------|----------|
| **Trend** | EMA crosses, MACD, Supertrend, Ichimoku | `EMA9/21-cross`, `Ichi-bull+BTCreg` |
| **Momentum** | RSI, ROC, Stochastic, Williams %R | `RSI-cross-30`, `ROC10+BTCreg` |
| **Volatility** | Bollinger Bands, ATR breakout, Keltner | `ATR-brk-1.5`, `Keltner-brk` |
| **Channel** | Donchian, VWAP reclaim | `Donch20-brk`, `VWAP-reclaim` |
| **Price Action** | Heikin Ashi, pullback | `HA-reversal`, `EMA-pullback` |
| **Composite** | Multi-indicator combos | `RSI+BB+MACD`, `EMA+Vol-confirm` |

### 6 Exit Methods

| Method | Description |
|--------|-------------|
| `fixed_tp_stop` | Fixed take-profit and stop-loss levels |
| `atr_trail` | ATR-based trailing stop |
| `chandelier_trail` | Chandelier exit (highest high minus ATR) |
| `breakeven_after_1r` | Move stop to entry after 1R profit |
| `fixed_with_breakeven` | Fixed TP/SL with breakeven at 1R |
| `scale_out_half_at_1r` | Close 50% at 1R, trail the rest |

---

## 🚀 Quickstart

```bash
pip install terminus-lab
```

```bash
# 1. Fetch 8 years of data
terminus fetch --pairs BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT --tfs 1h,4h,1d --days 2920

# 2. Run the full parameter sweep
terminus sweep

# 3. Walk-forward the top candidates year-by-year
terminus walk-forward --top 15

# 4. Generate the survivor report
terminus report --min-calmar 1.5 --min-bear-return -5

# 5. Build a portfolio from survivors
terminus portfolio
```

### Example Output

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

## 🖥 CLI Reference

| Command | Description |
|---------|-------------|
| `terminus fetch` | Download and cache Binance klines (Parquet) |
| `terminus sweep` | Run full parameter sweep across all strategy families |
| `terminus walk-forward` | Calendar-year walk-forward validation |
| `terminus report` | Filter survivors and generate report |
| `terminus portfolio` | Build correlation-capped portfolio from survivors |
| `terminus contribute` | Upload sim results to the community hub |
| `terminus ml` | Train regime classifier or LightGBM screener |

<details>
<summary><b>Common flags</b></summary>

```bash
# Fetch
terminus fetch --pairs BTCUSDT,ETHUSDT --tfs 1h,4h,1d --days 2920

# Sweep with specific exit methods
terminus sweep --pairs BTCUSDT --exit-methods fixed_tp_stop,atr_trail

# Walk-forward top N by Calmar
terminus walk-forward --top 15

# Report with custom filters
terminus report --min-calmar 1.5 --min-bear-return -5 --min-trades-per-year 5

# Portfolio with correlation cap
terminus portfolio --max-corr 0.6 --target-ann-return 25
```

</details>

---

## ⚙ Execution Model

Terminus models **realistic execution**, not idealized fills:

| Parameter | Majors | Mid-caps | Small-caps |
|-----------|--------|----------|------------|
| Entry slippage | 0.05% | 0.10% | 0.20% |
| Stop slippage | 0.10% | 0.20% | 0.40% |
| TP slippage | 0.02% | 0.04% | 0.08% |
| Timeout slippage | 0.05% | 0.10% | 0.20% |
| Fee (per side) | 0.075% | 0.075% | 0.075% |

**Majors:** BTC, ETH, SOL, XRP, BNB &nbsp; | &nbsp; **Mid-caps:** top 20-50 by market cap &nbsp; | &nbsp; **Small-caps:** everything else

Entry fills at next bar's open. Stops fill at stop price with adverse slippage. TPs fill at limit price with favorable slippage. Timeouts fill at close with slippage.

---

## 🧠 ML Module

<details>
<summary><b>Regime Classifier</b></summary>

XGBoost 3-class classifier trained on rolling features:

- **BULL**: forward returns > +5%
- **BEAR**: forward returns < -5%
- **CHOP**: everything else

10 features: RSI, EMA ratios, volatility, volume momentum, trend strength. Auto-labels from forward returns. Skips entries in wrong regime when enabled.

```bash
terminus ml --train-regime --pair BTCUSDT --tf 1d
```

</details>

<details>
<summary><b>LightGBM Screener</b></summary>

Trained on sweep results to predict which configs will have high Calmar. Reduces sweep time by pre-filtering low-probability configs.

```bash
terminus ml --train-screener
```

</details>

<details>
<summary><b>Walk-Forward-Aware Optimizer</b></summary>

Random parameter search that respects walk-forward boundaries. Never optimizes on test data.

```bash
terminus ml --optimize --pair BTCUSDT --tf 4h --family ATR-brk
```

</details>

---

## 🌐 Community Hub

**Leaderboard:** [terminus-hub.shatla-tech.workers.dev](https://terminus-hub.shatla-tech.workers.dev)

After every `terminus sweep` and `terminus walk-forward`, Terminus automatically contributes your survivors to the community hub. Sims are stored locally first; remote send retries on the next run if the network was down.

- Results are deduplicated by content hash
- Contributor count increments when multiple users discover the same strategy
- Rare pairs get coverage from users who care about them

```bash
# Opt out
export TERMINUS_TELEMETRY=0

# Force-upload everything
terminus contribute --all
```

---

## 🧪 Survivor Filters

A strategy must pass **all** of these to be promoted:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Calmar ratio | ≥ 1.5 | Return must justify the drawdown |
| Bear year (2022) | ≥ -10% | Survive -64% BTC without catastrophic loss |
| Losing years | ≤ 2 | Most years must be profitable |
| Worst losing year | ≥ -10% | No single year wipes the account |
| Trades per year | ≥ 5 | Enough sample size to be meaningful |
| Multi-pair | ≥ 2 pairs | Single-pair success = curve fit |
| Single-year outlier | < 60% of total | No one-year windfall disguised as edge |

---

## 📁 Project Structure

<details>
<summary><b>Expand</b></summary>

```
terminus/
├── cli.py              # Entry point — all subcommands
├── fetch.py            # Async Binance kline fetcher (Parquet cache)
├── simulate.py         # Vectorized trade simulator, 6 exit methods
├── sweep.py            # Parallel parameter sweep engine
├── walk_forward.py     # Frozen / anchored / 75-25 modes
├── filter.py           # Survivor filtering + report
├── portfolio.py        # Greedy correlation-capped portfolio builder
├── registry.py         # Strategy family × parameter grid permutations
├── rules.py            # Vectorized entry signal rules (VRule)
├── indicators.py       # Two-layer indicator precompute (30+ indicators)
├── store.py            # SQLite research DB, content-hashed caching
├── telemetry.py        # Local telemetry + community hub submission
├── funding.py          # Funding rate data
├── sentiment.py        # Fear & Greed index
├── risk/
│   ├── factor_model.py # Risk factor analytics
│   └── metrics.py      # Sharpe, Calmar, max drawdown, etc.
└── ml/
    ├── regime.py       # XGBoost 3-class regime classifier
    ├── optim.py        # Walk-forward-aware random optimizer
    ├── features.py     # Feature engineering
    ├── dataset.py      # Dataset builder from sweep results
    └── train.py        # LightGBM screener training
```

</details>

---

## 🏛 Philosophy

<table>
<tr>
<td width="33%" valign="top">

### Pessimistic by Default
Terminus will tell you your strategy **doesn't work** before it tells you it does. That's the point. Every filter exists to kill bad strategies, not validate good ones.

</td>
<td width="33%" valign="top">

### Reproducible by Design
The content-hash cache means any claim traces back to exact inputs via SHA-256. `sims.db` travels with your conclusions. No "it worked on my machine."

</td>
<td width="33%" valign="top">

### Halal-First
Spot-only, no leverage, no shorts, no interest. Built by someone who can't use futures or margin. Cash (stablecoin) is always a valid position in bear regimes.

</td>
</tr>
</table>

---

## 🗺 Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| **v0.1** | Core sweep + walk-forward + report | ✅ Done |
| **v0.2** | ML regime classifier + community hub | ✅ Done |
| **v0.3** | Portfolio optimization + risk analytics | 🔄 In progress |
| **v0.4** | Alternative data sources (Kraken, OKX, Bybit) | 📋 Planned |
| **v0.5** | Web dashboard for results visualization | 📋 Planned |
| **v1.0** | Stable API + comprehensive test suite | 📋 Planned |

---

## 🤝 Contributing

PRs welcome for:
- New strategy families (numpy-vectorized signals)
- Alternative data fetchers (Kraken, Coinbase, OKX, Bybit spot)
- Portfolio construction methods
- New walk-forward modes
- Exit method variants

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## ⚠ Disclaimer

Terminus is a research tool. Past backtest performance does not guarantee future results. Always paper-trade validated strategies before risking real capital. The authors are not responsible for any financial losses.

---

## License

MIT. Use it, fork it, ship it. If you find something useful, open a PR. The community sim database grows stronger with every contributor.

---

<p align="center">
  <a href="https://pypi.org/project/terminus-lab/">
    <img src="https://img.shields.io/badge/PyPI-terminus--lab-blue?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/>
  </a>
</p>

<p align="center">
  <sub>Built by <a href="https://github.com/Stronex-Labs">Stronex Labs</a></sub>
</p>
