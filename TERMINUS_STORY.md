# Terminus — The Build Story
*A timeline of how we built an open-source backtesting lab from scratch*

---

## The Name

The tool was called **Crucible** during development — a crucible being the vessel where metals are subjected to extreme heat to prove their purity. The name fit: strategies enter, heat gets applied (walk-forward years, bear markets, realistic fees), and only the ones that survive come out the other side.

When it came time to go public, Khaled asked: *"what goes with end of all trades as one word like crucible?"*

The brainstorm that came back covered every angle:

| Name | Why |
|------|-----|
| **terminus** | Latin for "the end", final stop, where things conclude |
| **reckoning** | Day of reckoning — final accounting |
| **verdict** | The final decision handed down |
| **gauntlet** | Run the gauntlet, survive or don't |
| **anvil / forge / kiln / smelter** | The forge/fire family — things harden or shatter |
| **nihaya** (نهاية) | Arabic — "the end / finale" |
| **mizan** (ميزان) | Arabic — "the scale / balance / judgment" |
| **hisab** (حساب) | Arabic — "the reckoning / final accounting" |

`terminus` was ranked first: *clean, one word, means "the end", easy to import.*

Khaled picked it immediately: *"i like terminus lets use it with end of all trades"*

And that's how it happened. **"End of All Trades"** didn't get dropped — it became the subtitle. In Roman mythology, Terminus is the god of boundaries and territorial markers — the final point beyond which nothing passes. In transit systems, the terminus is the last station: the end of the line.

One word that holds the whole meaning.

> *Where strategies go to prove themselves — or die.*

The internal name (Crucible) stayed in the first commit as environment variables (`CRUCIBLE_HOME`) and was cleaned up on the way to open-source — replaced with `TERMINUS_HOME`. The rename wasn't just cosmetic: it marked the moment the tool stopped being a private research instrument and became something with a defined identity and a public mission.

---

## The Problem

It started with a frustration: running a trading strategy through every pair, every timeframe, every parameter combination takes forever — and most backtesting tools either cost money, lock you into their ecosystem, or just aren't built for the kind of systematic, large-scale parameter sweep that serious quant work demands.

The goal was simple in theory: build something that could run **thousands of simulations fast**, cache the results so you never run the same thing twice, validate strategies honestly with walk-forward testing, and do it all locally, open-source, no cloud lock-in.

What came out of that was **Terminus**.

---

## Phase 1 — The Core Engine

### The Simulation Store
The foundation was a local **SQLite database** — one file on disk that stores every single sim result. No server, no cloud, just a `.db` file you own. Every simulation gets a **content hash** as its primary key: if you run the same config twice, it hits the cache and returns instantly. This became the backbone of everything.

### The Parameter Sweep
The first real feature: a sweep engine that takes a list of pairs (e.g. `BTCUSDT`, `ETHUSDT`, `SOLUSDT`) × timeframes (`1h`, `4h`, `1d`) × strategy families × parameter grids and runs them all. Not sequentially — it batches them, skips cache hits, and persists results as they come in. The speed difference between a cache hit and a live sim meant you could iterate on ideas in seconds instead of hours.

### Entry Families
Terminus supports multiple **entry signal families** — each representing a different edge hypothesis. The engine is designed so you define the family, the parameter space, and the exit logic, and the sweep handles the rest. Walk-forward slicing per calendar year is built in so every strategy gets honest out-of-sample validation before it ever shows up in your results.

---

## Phase 2 — Filtering & Walk-Forward

### The Filter Bug (and the Fix)
One of the most important moments in the build: we found that the filter defaults were **too strict**. `require_every_year_profitable=True` was silently killing profitable strategies — like a TRX configuration that made money overall but had one losing year. The filter was treating it as worthless.

The fix: `require_every_year_profitable=False`, `max_losing_years=2`. Now the filter respects the actual quality of a strategy rather than penalizing anything that touched a single bad year. This unblocked a lot of real edge that was being hidden.

### Walk-Forward Validation
Every strategy that survives the filter has been tested with **frozen walk-forward**: train on historical years, test on the year you didn't touch. Results are stored per-year so you can see exactly how a config performed in 2021 (bull), 2022 (crash), 2023 (recovery), and so on. The bear-year return (2022) became a key signal for robustness.

---

## Phase 3 — Open Sourcing

### Sanitizing the Repo
When we decided to open-source Terminus, the codebase had internal fingerprints everywhere — a personal email in `pyproject.toml`, an internal project name ("Crucible") still showing up in `LICENSE` and environment variable names (`CRUCIBLE_HOME`), even baked into git commit history.

We rewrote the git history using `git filter-branch` to scrub the author email from every commit. Renamed `CRUCIBLE_HOME` to `TERMINUS_HOME`. Fixed the LICENSE. Replaced internal references. The repo came out clean — nothing that ties it to internal infra or personal identity.

### What Terminus Ships With
- CLI (`terminus sweep`, `terminus filter`, `terminus report`, `terminus portfolio`, `terminus contribute`)
- SQLite store with content-hash caching
- Walk-forward validation (frozen mode)
- Portfolio builder (combine legs, compute blended Sharpe/Calmar)
- Regime filter (skip entries in bear regimes)
- Full `.env.example` with documented config keys
- ML module (added later — see Phase 5)

---

## Phase 4 — The Community Hub

### The Idea
Backtesting results are most valuable when they're **aggregated across many users**. If 100 people run Terminus across different pairs and timeframes, the collective picture of what works and what doesn't is far richer than any single user's runs. This became the idea for the **community hub** — a federated backtesting database.

### What Gets Contributed
Not anonymized aggregates — **everything**. Full sim results: pair, timeframe, strategy family, all parameters, all metrics. Full trade lists (entry/exit prices, timestamps, P&L per trade). Walk-forward slices per calendar year. Portfolio compositions. Engine performance metrics.

The philosophy: if you're going to build a shared knowledge base, it needs to be detailed enough to actually be useful for research.

### Telemetry by Default
Contribution is **opt-in by default** (`TERMINUS_TELEMETRY=1`). Users who want privacy can set `TERMINUS_TELEMETRY=0` and nothing leaves their machine. Local telemetry (the SQLite `telemetry_events` table) always runs — that's for your own records. Remote contribution is what goes to the hub.

### The Backend — Cloudflare Workers + D1

We tried Railway first — OAuth token issues in a non-interactive shell killed it. Then we looked at what made sense: the hub is a simple API that ingests sim results and serves leaderboard queries. It doesn't need a VM. It needs edge latency and zero maintenance.

**Cloudflare Workers + D1** (SQLite at the edge) was the answer. Built with **Hono** (TypeScript). Deployed in minutes. The worker lives at:

```
https://terminus-hub.shatla-tech.workers.dev/api/v1
```

Endpoints:
- `POST /contribute/sim` — full sim + walk-forward contribution
- `GET /leaderboard` — top configs ranked by Calmar across all contributors
- `POST /events/sweep`, `/events/walk_forward`, `/events/portfolio`, `/events/fetch`, `/events/filter` — engine performance telemetry
- `GET /health` — heartbeat

The D1 database has three tables: `sims`, `walk_forward_runs`, `events`. Duplicate submissions (same hash from different users) increment a `contributor_count` rather than creating duplicates — so the leaderboard shows how many independent users validated each config.

---

## Phase 5 — ML Module

### Regime Classifier
Added an **XGBoost 3-class regime classifier** (BULL / BEAR / CHOP). Features: 5/20/60-bar returns, RSI-14, ATR%, volume ratio, EMA ratio, close vs SMA200, Bollinger Band width, 20-bar drawdown. Auto-labels from forward returns with configurable thresholds.

The idea: don't enter a mean-reversion trade in a trending regime. The classifier runs on your price data and tags each bar, so the engine can skip entries that are in the wrong context.

### Parameter Optimizer
A **walk-forward-aware random parameter search** — not grid search, not Bayesian, just intelligent random sampling across the parameter space with calendar-year folds as the validation structure. Scores by mean out-of-sample Calmar. Fast to run, resistant to overfitting because the folds are time-based.

Both live under `terminus/terminus/ml/`:
- `regime.py` — `RegimeClassifier` class
- `optim.py` — `OptimResult`, random WF search

CLI: `terminus ml regime --pair BTCUSDT --tf 1h`

---

## The Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Simulation store | SQLite (local), D1 (hub) |
| Cache key | SHA-256 content hash of config |
| CLI | Click |
| Hub backend | Cloudflare Workers (Hono / TypeScript) |
| ML | XGBoost, scikit-learn |
| Validation | Walk-forward, calendar-year folds |
| Distribution | PyPI (planned), GitHub open source |

---

## Key Numbers (at launch)

- Supported pairs: any USDT perpetual (Binance format)
- Timeframes: `1m` through `1w`
- Strategy families: configurable (pluggable entry signals)
- Cache hit: ~0ms (SQLite lookup)
- Live sim: varies by data size, typically <1s per config on modern hardware
- Hub: zero cold start (Cloudflare edge), SQLite D1

---

## What This Unlocks

1. **Solo researcher** — run 50,000 configs overnight, wake up to a filtered leaderboard of what actually worked
2. **Team** — share a hub instance, contribute findings, avoid duplicating runs
3. **Community** — every user's results feed the aggregate leaderboard; rare pairs and exotic timeframes get covered by users who care about them
4. **Content** — every sim result tells a story: this config made 340% in 2021, lost 12% in 2022, recovered in 2023. That's a narrative.

---

## What's Next

- PyPI package (`pip install terminus-lab`)
- Hub going fully live with real contributor data
- Regime-aware sweep (auto-skip bad-regime entries)
- Portfolio optimizer that respects walk-forward constraints
- Public leaderboard UI on the web

---

*Built by Khaled Mansour. Open sourced under the Terminus license.*
*Hub: terminus-hub.shatla-tech.workers.dev*
*Repo: github.com/Shatla-tech/terminus*
