# The Terminus Story

## Why it exists

Most backtesting tools are built to make you feel good about your strategy.
Terminus is built to tell you it doesn't work.

The project started from a constraint: no futures, no margin, no shorts.
Halal-only. That rules out 90% of quant tooling, which is built around leverage
and hedging. The alternatives were either toy libraries or heavyweight platforms
that assume you're a hedge fund.

So Terminus was built from scratch. The goal was simple: take a strategy idea,
throw 8 years of real market data at it including 2022 (BTC -64%), run it
year-by-year with frozen parameters, and find out if it actually survives.

Most don't.

---

## What makes it different

**Walk-forward is not optional.** Every strategy gets tested year-by-year with
parameters frozen at the start. In-sample fitting followed by out-of-sample
validation isn't a feature you toggle on — it's the only mode.

**Bear years are not excluded.** 2022 happened. Your strategy either made it
or it didn't. Terminus shows you that number without rounding.

**The cache is content-addressed.** Every simulation is keyed by a hash of its
inputs: pair, timeframe, config, date range, slippage, and fees. Run the same
sim twice and the second run is instant. The `sims.db` file travels with your
research — it is your research.

**Execution is pessimistic.** Slippage is tiered by market cap. Entry slippage,
stop slippage, TP slippage, and timeout slippage are all modeled separately.
Cooldowns and max-hold periods prevent the engine from pretending you can trade
every signal at ideal prices.

---

## The community angle

Every `terminus sweep` and `terminus walk-forward` automatically contributes
survivors to a shared leaderboard at `terminus-hub.shatla-tech.workers.dev`.
The logic is simple: if you found a strategy that works, there's no reason to
keep that information private. The database grows stronger with every researcher
who runs it.

Submissions are stored locally first. If the network is down, they get retried
on the next run. No data is ever lost.

---

## Version history

### 0.1.0 — First public release
Core engine: sweep, walk-forward, portfolio, report. 30+ strategy families
across RSI, EMA/MACD, Bollinger, Donchian, ATR breakouts, Ichimoku, Supertrend,
Keltner, VWAP, Heikin Ashi, ROC momentum. Content-hashed SQLite store.
Published to PyPI.

### 0.1.1 — Logo fix
PyPI README was rendering a broken image. Root cause: relative asset paths don't
resolve on PyPI. Fixed by switching to an absolute GitHub raw URL.

### 0.1.2 — Copy cleanup
Removed em dashes from all user-facing text. Freshened the PyPI description.
Logo renders correctly on the project page.

### 0.1.3 — Auto-update + release tooling
Added startup version check: on the first `terminus` invocation each day,
the CLI silently checks PyPI for a newer version and upgrades itself, then
re-runs the original command. Users always get the latest without lifting a finger.

Added `release.py`: one command (`python release.py patch`) bumps the version,
commits, tags, pushes to GitHub, builds, and uploads to PyPI.

### 0.1.4 — Dependency reduction
Dropped `pyarrow` and `python-dotenv`. Price cache switched from Parquet to CSV
(stdlib, no engine dependency). `python-dotenv` was unused.
Core dependencies: numpy, pandas, pandas-ta, httpx.

---

## What's next

- More strategy families (contributed by the community)
- Additional data fetchers: Kraken, OKX, Bybit spot
- Portfolio construction improvements
- Richer walk-forward modes

The engine is stable. The research data model is stable. The strategy space
is not — every new pair and family expands what's discoverable.
