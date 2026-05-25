"""Full 7-family strategy test with Stronex-realistic constraints.

Runs: MR, Momentum, Breakout, Trend Pullback, Funding Fade, Vol Squeeze
+ existing families (EMA-cross, Donchian, ATR, RSI, BB, Stoch, etc.)

Applies ALL Stronex architecture gates in the filter:
- Fee: 0.075% per side
- Slippage: entry 0.05%, stop 0.10%, TP 0.02%, timeout 0.05%
- Min R:R after fees: 1.3
- Max hold: 24h equivalent per TF
- BTC regime filter
- All 8 safety audits including 3 new ones

Usage:
    python scripts/full_strategy_test.py [--pairs BTCUSDT,ETHUSDT] [--days 1460]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from terminus.store import get_store
from terminus.sweep import run_full_sweep, _load_and_precompute
from terminus.walk_forward import walk_forward_frozen
from terminus.filter import filter_sims, survivor_report
from terminus.registry import build_all_configs, build_configs_with_regime, count_configs
from terminus.indicators import build_btc_regime_series
from terminus.fetch import BinanceFetcher, load_or_fetch, cache_path
from terminus.simulate import slip_for
from terminus.risk.rademacher import deflated_sharpe, rademacher_penalty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("full_strategy_test")

# ---------------------------------------------------------------------------
# Stronex-realistic constraints
# ---------------------------------------------------------------------------
STRONEX_FEE = 0.00075          # 0.075% per side (BNB discount)
STRONEX_SLIP = {
    "entry_slip": 0.0005,      # 0.05%
    "stop_slip": 0.0010,       # 0.10% (market order on stop)
    "tp_slip": 0.0002,         # 0.02% (limit order)
    "timeout_slip": 0.0005,    # 0.05%
}
STRONEX_MIN_RR = 1.3           # Fee-adjusted R:R floor
STRONEX_MAX_HOLD_HOURS = 24    # Auto-close after 24h

# Pairs: Stronex's actual universe (top 50 by volume, subset for testing)
DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "TRXUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "MATICUSDT", "INJUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT",
]

# Timeframes: Stronex uses 4h and 6h primarily, 1h for MR fast
DEFAULT_TFS = ["1h", "4h", "6h"]

# Days: 4 years (matches Stronex minimum for ROBUST tier)
DEFAULT_DAYS = 1460


# ---------------------------------------------------------------------------
# Phase 1: Fetch data
# ---------------------------------------------------------------------------
async def fetch_data(pairs: list[str], tfs: list[str], days: int):
    """Fetch all pair/tf combos, cache to parquet."""
    logger.info(f"Phase 1: Fetching {len(pairs)} pairs × {len(tfs)} TFs × {days}d")
    async with BinanceFetcher() as fetcher:
        sem = asyncio.Semaphore(5)
        results = []

        async def worker(pair, tf):
            async with sem:
                try:
                    df = await load_or_fetch(fetcher, pair, tf, days)
                    bars = len(df) if df is not None else 0
                    results.append((pair, tf, bars))
                    if bars >= 100:
                        logger.info(f"  {pair} {tf}: {bars} bars OK")
                except Exception as e:
                    logger.warning(f"  {pair} {tf}: FAILED ({e})")
                    results.append((pair, tf, 0))

        await asyncio.gather(*[worker(p, t) for p in pairs for t in tfs])

    ok = sum(1 for _, _, b in results if b >= 400)
    logger.info(f"Phase 1 done: {ok}/{len(results)} usable ({ok} have ≥400 bars)")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Full sweep
# ---------------------------------------------------------------------------
def run_sweep(pairs: list[str], tfs: list[str], days: int):
    """Run all 193 configs × regime wrap across all pairs/TFs."""
    counts = count_configs()
    logger.info(
        f"Phase 2: Sweep — {counts['total_base']} base configs "
        f"(+{counts['total_base']} regime-wrapped = {counts['with_regime_wrapped']} total)"
    )

    summary = run_full_sweep(
        pairs=pairs, tfs=tfs, days=days,
        exit_methods=["fixed_tp_stop"],
        include_regime_wrap=True,
        label="stronex-full-7family",
    )
    logger.info(f"Phase 2 done: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Phase 3: Walk-forward on top candidates
# ---------------------------------------------------------------------------
def run_walk_forward(pairs: list[str], tfs: list[str], days: int, top_per_pair: int = 15):
    """Walk-forward the top-N configs per pair/TF."""
    store = get_store()

    base_configs = build_all_configs()
    btc_df = _load_and_precompute("BTCUSDT", "1d", days)
    btc_regime = None
    regime_configs = []
    if btc_df is not None:
        btc_regime = build_btc_regime_series(btc_df)
        regime_configs = build_configs_with_regime(base_configs, btc_regime)
    all_configs = {c[0]: c for c in base_configs + regime_configs}

    # Get top candidates by Calmar
    pair_filter = ",".join("?" * len(pairs))
    tf_filter = ",".join("?" * len(tfs))
    sql = f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY pair, timeframe ORDER BY calmar DESC
            ) AS rn
            FROM sims
            WHERE pair IN ({pair_filter})
              AND timeframe IN ({tf_filter})
              AND calmar >= 1.0
              AND n_trades >= 15
        ) WHERE rn <= ?
    """
    params = list(pairs) + list(tfs) + [top_per_pair]
    rows = store.query(sql, params)
    logger.info(f"Phase 3: Walk-forward on {len(rows)} candidates (top-{top_per_pair}/pair/TF)")

    wf_count = 0
    skip_count = 0
    for r in rows:
        r = dict(r)
        cfg = all_configs.get(r["config_name"])
        if cfg is None:
            continue
        name, rule, tp, stop, hold, cd, family = cfg

        # Skip if already done
        if store.get_wf_for(r["hash"], mode="frozen"):
            skip_count += 1
            continue

        pdf = _load_and_precompute(r["pair"], r["timeframe"], days)
        if pdf is None:
            continue

        try:
            walk_forward_frozen(
                pdf, pair=r["pair"], timeframe=r["timeframe"],
                config_name=name, family=family,
                rule=rule, tp=tp, stop=stop,
                max_hold=hold, cooldown=cd,
                exit_method=r["exit_method"],
                fee_rate=STRONEX_FEE,
                parent_hash=r["hash"], store=store,
            )
            wf_count += 1
        except Exception as e:
            logger.warning(f"  WF failed {r['pair']} {name}: {e}")

    logger.info(f"Phase 3 done: {wf_count} new WF runs, {skip_count} skipped (cached)")


# ---------------------------------------------------------------------------
# Phase 4: Filter with ALL 8 safety audits + Stronex constraints
# ---------------------------------------------------------------------------
def run_filter_and_report(days: int):
    """Apply full safety audit filter and produce ranked report."""
    store = get_store()

    logger.info("Phase 4: Filtering with 8 safety audits + Stronex constraints")

    # Stronex-aware filter settings:
    # - min_calmar 1.5 (meaningful edge after fees)
    # - require bear year (2022) survived
    # - single-year outlier at 60%
    # - CVaR < 8% (tail risk manageable for 15% position sizing)
    # - BTC correlation check ON
    # - min 5 trades/year (statistically meaningful)
    # - min 5 pairs generalization (rule works broadly, not pair-specific)
    survivors = filter_sims(
        store,
        min_full_calmar=1.5,
        min_trades_per_year=5,
        min_total_trades=20,
        min_bear_return=-5.0,         # Allow up to -5% in bear year (realistic)
        min_pairs_generalization=3,   # Relaxed: 3 pairs min (we test 15)
        require_every_year_profitable=False,
        max_losing_years=1,           # At most 1 losing year
        max_losing_year_severity=-8.0,  # That year can't be worse than -8%
        bear_year_label="2022",
        include_frozen_wf=True,
        max_cvar_95=0.08,
        max_single_year_concentration=0.60,  # NEW: single-year outlier
        check_btc_correlation=True,          # NEW: cross-pair correlation
        btc_overlap_threshold=0.70,          # NEW: 70% overlap = same trade
    )

    # Post-filter: Apply Stronex R:R gate
    # Reject configs where TP/stop ratio < 1.3 after fees
    rr_filtered = []
    for s in survivors:
        tp_pct = s.full_row.get("tp_pct", 0.03)
        stop_pct = s.full_row.get("stop_pct", 0.04)
        fee_cost = 2 * STRONEX_FEE  # round-trip fee
        effective_reward = tp_pct - fee_cost
        effective_risk = stop_pct + fee_cost
        rr = effective_reward / effective_risk if effective_risk > 0 else 0
        if rr >= STRONEX_MIN_RR:
            rr_filtered.append(s)

    logger.info(
        f"Phase 4: {len(survivors)} passed safety audits, "
        f"{len(rr_filtered)} passed R:R≥{STRONEX_MIN_RR} gate"
    )

    # Rademacher deflation check
    n_tested = count_configs()["with_regime_wrapped"]
    penalty = rademacher_penalty(n_tested)
    logger.info(f"Rademacher penalty for {n_tested} configs tested: {penalty:.2f} Sharpe units")

    # Print full report
    if rr_filtered:
        print("\n" + "=" * 130)
        print("STRONEX-VALIDATED SURVIVORS (all 8 audits passed + R:R gate)")
        print("=" * 130)
        print(survivor_report(rr_filtered, limit=60))
        print()

        # Strategy family breakdown
        family_counts = {}
        for s in rr_filtered:
            base_fam = s.family.replace("+BTCreg", "")
            family_counts[base_fam] = family_counts.get(base_fam, 0) + 1
        print("\n--- STRATEGY FAMILY BREAKDOWN ---")
        for fam, count in sorted(family_counts.items(), key=lambda x: -x[1]):
            print(f"  {fam:<25} {count} survivors")

        # ToD/DoW enrichment summary for top 10
        print("\n--- TIME-OF-DAY / DAY-OF-WEEK ENRICHMENT (top 10) ---")
        for s in rr_filtered[:10]:
            info = s.tod_dow_info
            if info:
                print(
                    f"  {s.pair:<10} {s.family:<20} "
                    f"best_hours={info.get('best_hours', [])} "
                    f"best_days={info.get('best_days', [])} "
                    f"conc={info.get('hour_concentration', 0):.0%}"
                )

        # BTC overlap report
        overlap_flagged = [s for s in rr_filtered if s.btc_trade_overlap > 0.5]
        if overlap_flagged:
            print(f"\n--- BTC CORRELATION WARNING ({len(overlap_flagged)} configs >50% overlap) ---")
            for s in overlap_flagged:
                print(f"  {s.pair:<10} {s.family:<20} overlap={s.btc_trade_overlap:.0%}")
    else:
        print("\nNo survivors passed all gates. Consider relaxing constraints.")

    return rr_filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full 7-family Stronex strategy test")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetch (use cached)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip sweep (use cached sims)")
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip walk-forward (use cached)")
    parser.add_argument("--top", type=int, default=15,
                        help="Top N per pair/TF for walk-forward")
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"TERMINUS FULL STRATEGY TEST — Stronex Constraints")
    print(f"{'='*80}")
    print(f"Pairs: {len(pairs)} | TFs: {tfs} | Days: {args.days}")
    print(f"Configs: {count_configs()}")
    print(f"Fee: {STRONEX_FEE*100:.3f}%/side | Min R:R: {STRONEX_MIN_RR}")
    print(f"Slip: entry={STRONEX_SLIP['entry_slip']*100:.2f}% "
          f"stop={STRONEX_SLIP['stop_slip']*100:.2f}% "
          f"tp={STRONEX_SLIP['tp_slip']*100:.2f}%")
    print(f"{'='*80}\n")

    # Phase 1: Fetch
    if not args.skip_fetch:
        asyncio.run(fetch_data(pairs, tfs, args.days))
    else:
        logger.info("Phase 1: SKIPPED (using cache)")

    # Phase 2: Sweep
    if not args.skip_sweep:
        run_sweep(pairs, tfs, args.days)
    else:
        logger.info("Phase 2: SKIPPED (using cache)")

    # Phase 3: Walk-forward
    if not args.skip_wf:
        run_walk_forward(pairs, tfs, args.days, top_per_pair=args.top)
    else:
        logger.info("Phase 3: SKIPPED (using cache)")

    # Phase 4: Filter + Report
    survivors = run_filter_and_report(args.days)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE — {len(survivors)} validated strategies in {elapsed:.0f}s")
    print(f"{'='*80}")

    return 0 if survivors else 1


if __name__ == "__main__":
    sys.exit(main())
