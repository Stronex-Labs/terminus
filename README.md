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
  <img src="https://img.shields.io/badge/Strategies-30-E67E22" alt="Strategies">
  <img src="https://img.shields.io/badge/Exit_Methods-6-7C3AED" alt="Exit Methods">
  <img src="https://img.shields.io/badge/Timeframes-8-0F766E" alt="Timeframes">
  <img src="https://img.shields.io/badge/Halal_First-Spot_Only-DAA520" alt="Halal First">
  <br>
  <a href="https://github.com/Stronex-Labs/terminus"><img src="https://img.shields.io/github/stars/Stronex-Labs/terminus?style=flat" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-why-terminus">Why</a> &nbsp;&middot;&nbsp;
  <a href="#-key-features">Features</a> &nbsp;&middot;&nbsp;
  <a href="#-installation">Install</a> &nbsp;&middot;&nbsp;
  <a href="#-quickstart">Quickstart</a> &nbsp;&middot;&nbsp;
  <a href="#-sweep-presets">Presets</a> &nbsp;&middot;&nbsp;
  <a href="#-cli-reference">CLI</a> &nbsp;&middot;&nbsp;
  <a href="#-strategy-families">Strategies</a> &nbsp;&middot;&nbsp;
  <a href="#-execution-model">Execution</a> &nbsp;&middot;&nbsp;
  <a href="#-ml-module">ML</a> &nbsp;&middot;&nbsp;
  <a href="#-environment-variables">Env</a> &nbsp;&middot;&nbsp;
  <a href="#-community-hub">Hub</a> &nbsp;&middot;&nbsp;
  <a href="#-project-structure">Structure</a> &nbsp;&middot;&nbsp;
  <a href="#-api--plugins">API</a> &nbsp;&middot;&nbsp;
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
- **Multi-pair generalization.** Works on one pair? That's a curve fit. Terminus requires success across at least 5 pairs.
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
Frozen and anchored modes with calendar-year folds. WFE ratio scores overfit risk. Bear year 2022 is the litmus test.

</td>
<td width="25%" valign="top">

### Portfolio Builder
Greedy correlation-capped leg selection. Sharpe-ranked with Calmar pre-filter. Daily P&L reconstruction.

</td>
<td width="25%" valign="top">

### ML Module
LightGBM screener for config pre-filtering. LightGBM regime classifier (BULL/BEAR/CHOP). Walk-forward-aware optimizer.

</td>
</tr>
</table>

### 30 Strategy Families

| Category | Families | Examples |
|----------|----------|----------|
| **Trend** | EMA crosses, MACD, Supertrend, Ichimoku | `EMA9/21-cross`, `Ichi-bull+BTCreg` |
| **Momentum** | RSI, ROC, Stochastic, Williams %R, ADX-surge, N-of-4 sniper | `RSI-cross-30`, `Mom-sniper[3of4]` |
| **Volatility** | Bollinger Bands, ATR breakout, Keltner | `ATR-brk-1.5`, `Keltner-brk` |
| **Channel** | Donchian, VWAP reclaim | `Donch20-brk`, `VWAP-reclaim` |
| **Price Action** | Heikin Ashi, pullback | `HA-reversal`, `EMA-pullback` |
| **Composite** | Multi-indicator combos | `RSI+BB+MACD`, `EMA+Vol-confirm` |

Each family also has a `+BTCreg` variant that gates entries on the BTC regime classifier, effectively doubling the config space to 60.

### 6 Exit Methods

| Method | Description | Default sweep |
|--------|-------------|:---:|
| `fixed_tp_stop` | Fixed take-profit and stop-loss levels | ✅ |
| `atr_trail` | ATR-based trailing stop | — |
| `chandelier_trail` | Chandelier exit (highest high minus ATR) | — |
| `breakeven_after_1r` | Move stop to entry after 1R profit | ✅ |
| `fixed_with_breakeven` | Fixed TP/SL with breakeven at 1R | — |
| `scale_out_half_at_1r` | Close 50% at 1R, trail the rest | ✅ |

All 6 are implemented in the simulator. The default sweep uses the 3 marked above; trail-based methods can be enabled with `--exit-methods`.

### Exit-check fidelity (`exit_check`)

The **single biggest backtest trap** is an exit model that doesn't match how the live engine samples price — it can invert your conclusions. `simulate_fast(..., exit_check=...)` (and `run_full_sweep(..., exit_check=...)`) makes the fidelity explicit:

| Mode | The exit sees | Use when the live closer… |
|------|---------------|---------------------------|
| `path` *(default)* | the bar's **high and low** — a stop/TP fills when the intra-bar wick touches its level | reconstructs the intra-bar path (e.g. a tick-tight 1m trail) |
| `discrete` | only each bar's **close**, and fills at that close — intra-bar wicks are invisible | samples one current price per scan cycle |

`path` mode is modelled **worst-case**: a bar's low is tested against the stop carried from *prior* bars **before** that same bar's high can ratchet the trail (a real stop can't use a bar's own high to dodge its own low), and a stop never fills above the bar's own high (gap-through fills at the worse gapped price, not the untouched level). Pick the mode that matches your live closer — mismatching it is how a losing strategy backtests as a winner.

---

## 📦 Installation

### From PyPI (recommended)

```bash
pip install terminus-lab
```

### With ML support

```bash
pip install terminus-lab[ml]
```

This adds LightGBM and scikit-learn — everything needed for both the screener and regime classifier.

### From source (development)

```bash
git clone https://github.com/Stronex-Labs/terminus.git
cd terminus
pip install -e ".[dev,ml]"
```

### Requirements

- **Python 3.10+** (tested on 3.10, 3.11, 3.12, 3.13)
- **Core deps:** numpy, pandas, pandas-ta, httpx
- **ML deps (optional):** lightgbm, scikit-learn, joblib
- **Storage:** ~2 GB for full 21-pair 8-year Parquet cache
- **RAM:** 4 GB minimum, 8 GB recommended for full sweeps

### Verify installation

```bash
terminus --help
```

---

## 🚀 Quickstart

```bash
# 1. Fetch 8 years of data (21 pairs, 8 timeframes)
terminus fetch

# 2. Run the full parameter sweep
terminus sweep

# 3. Walk-forward the top candidates year-by-year
terminus walk-forward --top 15

# 4. Generate the survivor report
terminus report --min-calmar 1.5 --min-bear-return -5

# 5. Build a portfolio from survivors
terminus portfolio
```

### Minimal quick-run (single pair)

```bash
terminus fetch --pairs BTCUSDT --tfs 4h,1d --days 2920
terminus sweep --pairs BTCUSDT --tfs 4h,1d
terminus walk-forward --pairs BTCUSDT --top 5
terminus report
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

## 🎯 Sweep Presets

Common sweep configurations for different use cases:

### Full discovery (default)

```bash
terminus sweep
```

Runs all 21 pairs, all 8 timeframes, `fixed_tp_stop` exit method. ~10,000+ configs tested. Takes 30-60 minutes on a modern machine.

### Extended exit methods

```bash
terminus sweep --exit-methods fixed_tp_stop,atr_trail,chandelier_trail,breakeven_after_1r,scale_out_half_at_1r
```

All exit methods enabled. 5x the configs. Multi-hour run.

### Regime-gated only

```bash
terminus sweep --pairs BTCUSDT,ETHUSDT,SOLUSDT --tfs 4h,1d
```

Focus on majors with fewer timeframes for faster iteration.

### No regime filter

```bash
terminus sweep --no-regime
```

Skips the `+BTCreg` variants. Half the config space, useful for pairs with limited BTC correlation.

### Labeled runs

```bash
terminus sweep --label "q2-2026-majors"
terminus walk-forward --label "q2-2026-wf"
```

Label manifests for organization and comparison between runs.

---

## 🖥 CLI Reference

### `terminus fetch`

Download and cache Binance klines as Parquet files.

```bash
terminus fetch [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | All 21 pairs | Comma-separated pair list |
| `--tfs` | `15m,30m,1h,2h,4h,6h,12h,1d` | Comma-separated timeframes |
| `--days` | `2920` (8 years) | Days of history to fetch |
| `--concurrency` | `3` | Parallel download workers |
| `--force` | off | Re-download even if cached |

### `terminus sweep`

Run full parameter sweep across all strategy families.

```bash
terminus sweep [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | All 21 pairs | Comma-separated pair list |
| `--tfs` | All 8 timeframes | Comma-separated timeframes |
| `--days` | `2920` | Data window (days) |
| `--exit-methods` | `fixed_tp_stop` | Comma-separated exit methods |
| `--no-regime` | off | Skip `+BTCreg` variants |
| `--label` | auto | Manifest label for this run |

### `terminus walk-forward`

Calendar-year walk-forward validation of top candidates.

```bash
terminus walk-forward [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top` | `15` | Top N candidates per pair/tf group |
| `--min-calmar` | `1.5` | Minimum Calmar to qualify |
| `--min-trades` | `25` | Minimum total trades to qualify |
| `--days` | `2920` | Data window (days) |
| `--pairs` | all | Filter to specific pairs |
| `--label` | auto | Manifest label |

### `terminus report`

Filter survivors and print ranked report.

```bash
terminus report [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--min-calmar` | `1.5` | Minimum Calmar ratio |
| `--min-trades-total` | `25` | Minimum total trades |
| `--min-trades-per-year` | `4` | Minimum trades per year |
| `--min-bear-return` | `-5.0` | Max allowed loss in 2022 (%) |
| `--min-pairs-generalization` | `3` | Min pairs with same family winning |
| `--top` | `100` | Show top N survivors |

### `terminus portfolio`

Build a correlation-capped portfolio from survivors.

```bash
terminus portfolio [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--min-calmar` | `1.5` | Filter threshold |
| `--min-bear-return` | `-5.0` | Bear year filter |
| `--max-legs` | `6` | Maximum portfolio legs |
| `--max-corr` | `0.65` | Max pairwise correlation allowed |
| `--pool` | `60` | Top-N pool before greedy select |
| `--target` | `25.0` | Target annualized return (%) |

### `terminus contribute`

Upload sim results to the community hub.

```bash
terminus contribute [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--all` | off | Submit all sims (not just survivors) |
| `--min-calmar` | `0.0` | Include sims above this Calmar |
| `--limit` | `10000` | Max sims per submission |
| `--enable` | off | Enable remote sharing for this session |

### `terminus ml regime`

Train the LightGBM regime classifier on BTC daily data.

```bash
terminus ml regime [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | `2920` | Training data window |
| `--output` | `~/.terminus/regime_model` | Output path (writes `.lgb` + `.json`) |

### `terminus ml train`

Train LightGBM screener from sweep results.

```bash
terminus ml train [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--calmar-threshold` | `1.5` | Calmar for positive label |
| `--calmar-negative` | `0.3` | Calmar for negative label |
| `--min-trades` | `10` | Minimum trades per sim |
| `--test-fraction` | `0.25` | Test set split |
| `--output` | `~/.terminus/models/` | Output directory |
| `--max-sims` | `0` (all) | Limit training samples |
| `--no-funding` | off | Disable funding rate features |
| `--no-fng` | off | Disable Fear & Greed features |

### Global flags

| Flag | Description |
|------|-------------|
| `-v` / `--verbose` | Enable debug logging |

---

## ⚙ Execution Model

Terminus models **realistic execution**, not idealized fills:

| Parameter | Majors | Mid-caps | Small-caps |
|-----------|--------|----------|------------|
| Entry slippage | 0.05% | 0.10% | 0.20% |
| Stop slippage | 0.10% | 0.15% | 0.30% |
| TP slippage | 0.02% | 0.03% | 0.05% |
| Timeout slippage | 0.05% | 0.10% | 0.20% |
| Fee (per side) | 0.075% | 0.075% | 0.075% |

**Majors:** BTC, ETH, SOL, XRP, BNB &nbsp; | &nbsp; **Mid-caps:** top 20-50 by market cap &nbsp; | &nbsp; **Small-caps:** everything else

Entry fills at next bar's open. Stops fill at stop price with adverse slippage. TPs fill at limit price with favorable slippage. Timeouts fill at close with slippage.

---

## 🧠 ML Module

> **Dependencies:** Install with `pip install terminus-lab[ml]` to get LightGBM and scikit-learn. All ML features (screener + regime classifier) are covered by this single extra.

<details>
<summary><b>LightGBM Screener</b></summary>

Binary classifier trained on sweep results to predict which configs will have high Calmar. Reduces sweep time by pre-filtering low-probability configs.

Features include: indicator parameters, timeframe encoding, pair volatility profile, funding rates, and Fear & Greed index.

```bash
terminus ml train
terminus ml train --calmar-threshold 2.0 --no-funding
```

</details>

<details>
<summary><b>LightGBM Regime Classifier</b></summary>

3-class classifier (BULL/BEAR/CHOP) trained on rolling features:

- **BULL**: forward returns > +5%
- **BEAR**: forward returns < -5%
- **CHOP**: everything else

10 features: RSI, EMA ratios, volatility, volume momentum, trend strength. Auto-labels from forward returns. Skips entries in wrong regime when enabled via `+BTCreg` strategy variants.

```bash
terminus ml regime
terminus ml regime --days 3650 --output ./models/regime_v2
```

</details>

<details>
<summary><b>Walk-Forward-Aware Optimizer</b></summary>

Random parameter search that respects walk-forward boundaries. Never optimizes on test data. Available as a library module (`terminus.ml.optim`) — not yet exposed via CLI.

</details>

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMINUS_TELEMETRY` | `1` | Set to `0` to disable community hub submissions |
| `TERMINUS_NO_UPDATE` | `0` | Set to `1` to disable auto-update checks on PyPI |
| `TERMINUS_HOME` | `~/.terminus` | Base directory for models, cache metadata, manifests |
| `TERMINUS_HUB_URL` | *(built-in)* | Override the community hub endpoint URL |

---

## 🖥 Recommended Setup

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores (parallelized sweep) |
| RAM | 4 GB | 8-16 GB |
| Disk | 2 GB (cache) | 5 GB (full data + sims.db) |
| Network | Needed for fetch | Stable for initial data download |

### Data coverage

For robust walk-forward results, use **at least 4 years** of data (`--days 1460`). The default 8 years (`--days 2920`) ensures coverage of both bull (2021) and bear (2022) regimes.

### Default pairs (21)

```
BTCUSDT  ETHUSDT  SOLUSDT  XRPUSDT  BNBUSDT  TRXUSDT  DOGEUSDT
ADAUSDT  AVAXUSDT LINKUSDT MATICUSDT DOTUSDT  LTCUSDT  ATOMUSDT
NEARUSDT SUIUSDT  APTUSDT  ARBUSDT  OPUSDT   TIAUSDT  INJUSDT
```

### Default timeframes (8)

```
15m  30m  1h  2h  4h  6h  12h  1d
```

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

# Enable sharing explicitly
terminus contribute --enable
```

---

## 🧪 Survivor Filters

A strategy must pass **all** of these to be promoted:

| Filter | Default | Rationale |
|--------|---------|-----------|
| Calmar ratio | >= 1.5 | Return must justify the drawdown |
| Bear year (2022) | >= -2% | Survive -64% BTC without meaningful loss |
| Losing years | <= 2 | Most years must be profitable |
| Worst losing year | >= -10% | No single year wipes the account |
| Trades per year | >= 5 | Enough sample size to be meaningful |
| Multi-pair generalization | >= 5 pairs | Single-pair success = curve fit |
| CVaR (95%) | <= 8% | Tail risk must be bounded |

All thresholds are configurable via CLI flags (e.g. `--min-calmar`, `--min-bear-return`).

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
├── walk_forward.py     # Frozen / anchored modes
├── filter.py           # Survivor filtering + report
├── portfolio.py        # Greedy correlation-capped portfolio builder
├── registry.py         # Strategy family x parameter grid permutations
├── rules.py            # Vectorized entry signal rules (VRule)
├── indicators.py       # Two-layer indicator precompute (30+ indicators)
├── store.py            # SQLite research DB, content-hashed caching
├── telemetry.py        # Local telemetry + community hub submission
├── funding.py          # Funding rate data
├── sentiment.py        # Fear & Greed index
├── risk/
│   ├── factor_model.py # Risk factor analytics
│   ├── metrics.py      # Sharpe, Calmar, max drawdown, etc.
│   └── rademacher.py   # Rademacher-adjusted Sharpe (multiple-testing deflation)
└── ml/
    ├── regime.py       # LightGBM 3-class regime classifier
    ├── optim.py        # Walk-forward-aware random optimizer
    ├── features.py     # Feature engineering
    ├── dataset.py      # Dataset builder from sweep results
    └── train.py        # LightGBM screener training
```

</details>

---

## 🔌 API & Plugins

> **Status: Planned for v0.5+**

### REST API (planned)

A local HTTP server for programmatic access to sweep results, walk-forward data, and portfolio state. Will serve the web dashboard and expose JSON endpoints.

### Plugin system (planned)

Extensibility for:
- Custom data sources (beyond Binance)
- Custom exit methods
- Custom strategy families
- Custom portfolio construction algorithms
- Result exporters (CSV, Notion, Telegram)

Currently, new strategies are added by editing `terminus/registry.py` and `terminus/rules.py` directly. Plugin architecture will formalize this in v1.0.

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
| **v0.2** | ML regime classifier + community hub + LightGBM screener | ✅ Done |
| **v0.3** | Portfolio optimization + risk analytics | 🔄 In progress |
| **v0.4** | Alternative data sources (Kraken, OKX, Bybit) | 📋 Planned |
| **v0.5** | Web dashboard + REST API | 📋 Planned |
| **v0.6** | Plugin system + custom strategy loader | 📋 Planned |
| **v1.0** | Stable public API + comprehensive test suite + docs | 📋 Planned |

---

## 🤝 Contributing

PRs welcome for:
- New strategy families (numpy-vectorized signals)
- Alternative data fetchers (Kraken, Coinbase, OKX, Bybit spot)
- Portfolio construction methods
- New walk-forward modes
- Exit method variants
- ML feature engineering

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## ⚠ Disclaimer

Terminus is a **research tool**. It is not financial advice.

- Past backtest performance does not guarantee future results
- Walk-forward validation reduces but does not eliminate overfitting risk
- Always paper-trade validated strategies before risking real capital
- The authors are not financial advisors and bear no responsibility for trading decisions made using this software
- Crypto markets are volatile; you can lose your entire investment
- This tool is provided "as-is" without warranty of any kind

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
