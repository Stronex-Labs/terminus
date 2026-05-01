"""Terminus quickstart — single-pair sweep in under 60 seconds.

Usage:
    pip install terminus-lab
    python quickstart.py

What it does:
    1. Fetches 2 years of BTCUSDT 4h data from Binance (cached to CSV)
    2. Runs the default v2 strategy family sweep (~100 configs)
    3. Walk-forward validates the top 5 by Calmar ratio
    4. Prints a summary table and saves results to ./results/
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Fetch data
# ---------------------------------------------------------------------------
from terminus.fetch import BinanceFetcher

PAIR = "BTCUSDT"
TF   = "4h"
DAYS = 730  # 2 years


async def fetch_data():
    fetcher = BinanceFetcher()
    df = await fetcher.fetch(PAIR, TF, DAYS)
    if df is None or len(df) < 200:
        print(f"ERROR: could not fetch {PAIR} {TF} data.", file=sys.stderr)
        sys.exit(1)
    print(f"Fetched {len(df):,} bars  ({df.shape[1]} cols)  for {PAIR} {TF}")
    return df


# ---------------------------------------------------------------------------
# 2. Pre-compute indicators
# ---------------------------------------------------------------------------
from terminus.indicators import precompute_v2


def prep(df):
    df = precompute_v2(df)
    print(f"Indicators added: {[c for c in df.columns if c not in ('open','high','low','close','volume','open_time')]}")
    return df


# ---------------------------------------------------------------------------
# 3. Build strategy configs
# ---------------------------------------------------------------------------
from terminus import build_v2_configs


def make_configs():
    configs = build_v2_configs(
        pairs=[PAIR],
        timeframes=[TF],
        tp_list=[0.04, 0.06, 0.08],
        stop_list=[0.03, 0.04, 0.05],
        max_hold_list=[20, 30],
        cooldown_list=[2],
    )
    print(f"Testing {len(configs)} configs …")
    return configs


# ---------------------------------------------------------------------------
# 4. Sweep
# ---------------------------------------------------------------------------
from terminus import simulate, ResearchStore, get_store


def sweep(df, configs):
    store: ResearchStore = get_store(Path(".terminus_quickstart"))
    results = []

    for cfg in configs:
        entry_rule = cfg["entry_rule"]
        trades = simulate(
            df, entry_rule,
            tp_pct=cfg["tp_pct"],
            stop_pct=cfg["stop_pct"],
            max_hold_bars=cfg["max_hold_bars"],
            cooldown_bars=cfg["cooldown_bars"],
        )
        if len(trades) < 10:
            continue
        sim = store.record(cfg, trades)
        results.append(sim)

    results.sort(key=lambda s: s.calmar, reverse=True)
    print(f"\nTop 5 by Calmar:\n{'─'*65}")
    print(f"{'Rule':<22} {'TF':<5} {'TP':>5} {'SL':>5} {'WR':>6} {'Return':>8} {'MaxDD':>7} {'Calmar':>7}")
    print(f"{'─'*65}")
    for s in results[:5]:
        c = s.config
        print(
            f"{c.get('entry_rule_name','?'):<22} {c['timeframe']:<5} "
            f"{c['tp_pct']*100:>4.0f}% {c['stop_pct']*100:>4.0f}% "
            f"{s.win_rate*100:>5.0f}% {s.total_return_pct:>7.1f}% "
            f"{s.max_drawdown_pct:>6.1f}% {s.calmar:>7.2f}"
        )
    return results


# ---------------------------------------------------------------------------
# 5. Walk-forward validate the top 5
# ---------------------------------------------------------------------------
from terminus import walk_forward_frozen


def validate(df, results):
    print(f"\nWalk-forward (75/25 split) on top 5:\n{'─'*50}")
    for sim in results[:5]:
        cfg = sim.config
        wf = walk_forward_frozen(
            df, cfg["entry_rule"],
            tp_pct=cfg["tp_pct"],
            stop_pct=cfg["stop_pct"],
            max_hold_bars=cfg["max_hold_bars"],
            cooldown_bars=cfg["cooldown_bars"],
            train_frac=0.75,
        )
        verdict = "✅ PASS" if wf.test_net_return_pct > 0 else "❌ FAIL"
        print(
            f"  {cfg.get('entry_rule_name','?'):<20} "
            f"train={wf.train_net_return_pct:+.1f}%  "
            f"test={wf.test_net_return_pct:+.1f}%  {verdict}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 65)
    print(f"  Terminus Quickstart  —  {PAIR} {TF}  ({DAYS}d)")
    print("=" * 65)

    df_raw  = await fetch_data()
    df      = prep(df_raw)
    configs = make_configs()
    results = sweep(df, configs)

    if not results:
        print("\nNo configs produced ≥10 trades. Try longer history or wider parameters.")
        return

    validate(df, results)

    print("\nDone. Results stored in .terminus_quickstart/")
    print("Next: try `terminus sweep --pairs BTCUSDT,ETHUSDT --tfs 4h,6h` for a full run.")


if __name__ == "__main__":
    asyncio.run(main())
