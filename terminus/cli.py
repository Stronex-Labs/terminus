"""Terminus CLI — subcommands: fetch, sweep, walk-forward, report, portfolio.

Usage:
    terminus fetch --pairs BTCUSDT,ETHUSDT --tfs 1h,4h,1d
    terminus sweep --pairs BTCUSDT --exit-methods fixed_tp_stop
    terminus walk-forward --top 15
    terminus report
    terminus portfolio
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from .store import get_store
from .sweep import run_full_sweep, _load_and_precompute
from .walk_forward import walk_forward_frozen
from .filter import filter_sims, survivor_report
from .portfolio import greedy_portfolio
from .registry import build_all_configs, build_configs_with_regime
from .indicators import build_btc_regime_series
from .fetch import BinanceFetcher, load_or_fetch, cache_path, is_fresh
from .simulate import slip_for
from . import telemetry


DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "TRXUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "MATICUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
    "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "TIAUSDT", "INJUSDT",
]
DEFAULT_TFS = ["15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]


def _setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )


# --- fetch -----------------------------------------------------------------

async def _fetch_all(pairs, tfs, days, concurrency, force):
    store = get_store()
    with store.manifest_scope("prefetch",
                              label=f"prefetch-{days}d-{len(pairs)}p-{len(tfs)}tf"):
        async with BinanceFetcher() as fetcher:
            sem = asyncio.Semaphore(concurrency)
            results = []

            async def worker(pair, tf):
                async with sem:
                    t0 = time.time()
                    try:
                        df = await load_or_fetch(fetcher, pair, tf, days,
                                                  force=force)
                    except Exception as e:
                        logging.error(f"{pair} {tf}: {e}")
                        results.append((pair, tf, "ERROR", 0))
                        return
                    bars = len(df) if df is not None else 0
                    status = "OK" if bars >= 100 else "THIN"
                    logging.info(
                        f"{pair:<10} {tf:<4} {bars:>8} bars  {time.time()-t0:.1f}s"
                    )
                    if df is not None and bars >= 100:
                        first_ts = df["open_time"].iloc[0]
                        last_ts = df["open_time"].iloc[-1]
                        years = (last_ts - first_ts) / 1000 / 86400 / 365.25
                        path = cache_path(pair, tf, days)
                        store.record_pair_data(
                            pair, tf,
                            str(df["ts"].iloc[0].date()),
                            str(df["ts"].iloc[-1].date()),
                            bars, round(years, 3), str(path),
                        )
                    results.append((pair, tf, status, bars))

            await asyncio.gather(*[worker(p, t) for p in pairs for t in tfs])
    return results


def cmd_fetch(args):
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] or DEFAULT_PAIRS
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()] or DEFAULT_TFS
    results = asyncio.run(_fetch_all(pairs, tfs, args.days,
                                      args.concurrency, args.force))
    ok = sum(1 for _, _, s, _ in results if s == "OK")
    total_bars = sum(b for _, _, _, b in results)
    print(f"\n{ok}/{len(results)} fetches OK; {total_bars:,} total bars cached")


# --- sweep -----------------------------------------------------------------

def cmd_sweep(args):
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] or DEFAULT_PAIRS
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()] or DEFAULT_TFS
    exit_methods = [m.strip() for m in args.exit_methods.split(",") if m.strip()]

    summary = run_full_sweep(
        pairs=pairs, tfs=tfs, days=args.days,
        exit_methods=exit_methods,
        include_regime_wrap=not args.no_regime,
        label=args.label,
    )
    print(f"\nSweep done: {summary}")


# --- walk-forward ----------------------------------------------------------

def cmd_walk_forward(args):
    store = get_store()

    base_configs = build_all_configs()
    btc_df = _load_and_precompute("BTCUSDT", "1d", args.days)
    if btc_df is not None:
        btc_regime = build_btc_regime_series(btc_df)
        regime_configs = build_configs_with_regime(base_configs, btc_regime)
    else:
        regime_configs = []
    all_configs = {c[0]: c for c in base_configs + regime_configs}

    where = ["calmar >= ?", "n_trades >= ?"]
    params = [args.min_calmar, args.min_trades]
    if args.pairs:
        pair_filter = [p.strip() for p in args.pairs.split(",") if p.strip()]
        ph = ",".join("?" * len(pair_filter))
        where.append(f"pair IN ({ph})")
        params.extend(pair_filter)
    sql = f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY pair, timeframe ORDER BY calmar DESC) AS rn
            FROM sims WHERE {' AND '.join(where)}
        ) WHERE rn <= ?
    """
    params.append(args.top)
    rows = store.query(sql, params)
    print(f"WF candidates: {len(rows)}")

    by_pair_tf = {}
    for r in rows:
        by_pair_tf.setdefault((r["pair"], r["timeframe"]), []).append(dict(r))

    with store.manifest_scope("walk_forward", label=args.label or f"wf-top{args.top}"):
        for (pair, tf), group in by_pair_tf.items():
            pdf = _load_and_precompute(pair, tf, args.days)
            if pdf is None:
                continue
            for r in group:
                cfg = all_configs.get(r["config_name"])
                if cfg is None:
                    continue
                name, rule, tp, stop, hold, cd, family = cfg
                if store.get_wf_for(r["hash"], mode="frozen"):
                    continue
                try:
                    walk_forward_frozen(
                        pdf, pair=pair, timeframe=tf,
                        config_name=name, family=family,
                        rule=rule, tp=tp, stop=stop,
                        max_hold=hold, cooldown=cd,
                        exit_method=r["exit_method"],
                        parent_hash=r["hash"],
                    )
                except Exception as e:
                    logging.error(f"WF failed {pair} {tf} {name}: {e}")
    print("Walk-forward done.")


# --- report ----------------------------------------------------------------

def cmd_report(args):
    store = get_store()
    survivors = filter_sims(
        store,
        min_full_calmar=args.min_calmar,
        min_trades_per_year=args.min_trades_per_year,
        min_total_trades=args.min_trades_total,
        min_bear_return=args.min_bear_return,
        min_pairs_generalization=args.min_pairs_generalization,
        include_frozen_wf=True,
    )
    telemetry.filter_run(store, n_candidates=store.sim_count(),
                         n_survivors=len(survivors), min_calmar=args.min_calmar)
    print(f"\n{len(survivors)} survivors\n")
    print(survivor_report(survivors, limit=args.top))


# --- portfolio -------------------------------------------------------------

def cmd_portfolio(args):
    store = get_store()
    survivors = filter_sims(
        store,
        min_full_calmar=args.min_calmar,
        min_trades_per_year=4,
        min_total_trades=25,
        min_bear_return=args.min_bear_return,
        min_pairs_generalization=3,
    )
    r = greedy_portfolio(survivors[:args.pool], max_legs=args.max_legs,
                         max_corr=args.max_corr, target_annual_pct=args.target)
    m = r.get("metrics", {})
    telemetry.portfolio_complete(
        store, n_legs=len(r.get("legs", [])),
        annual_ret=m.get("annual_ret", 0),
        max_dd=m.get("max_dd", 0),
        sharpe=m.get("sharpe", 0),
    )
    print(f"\nPortfolio ({len(r.get('legs', []))} legs):")
    print(f"  Annualized: {m.get('annual_ret', 0):.2f}%")
    print(f"  Sharpe: {m.get('sharpe', 0)}")
    print(f"  MaxDD: {m.get('max_dd', 0)}%")
    print(f"  Max leg-to-leg correlation: {r.get('max_corr_observed', 0)}")
    print(f"\nLegs:")
    for leg in r.get("legs", []):
        print(f"  {leg['pair']:<12} {leg['timeframe']:<4} "
              f"{leg['family']:<26} w={leg['weight']}")
    yrs = r.get("year_breakdown", {})
    if yrs:
        print(f"\nPer-year:")
        for y, ret in sorted(yrs.items()):
            print(f"  {y}: {ret:+.2f}%")


# --- contribute ------------------------------------------------------------

def cmd_contribute(args):
    store = get_store()
    if args.enable:
        telemetry.set_remote_enabled(True)
        print("Remote telemetry enabled for this session (TERMINUS_TELEMETRY=1).")
        print("Add to your shell profile to persist.")
    survivors = filter_sims(
        store,
        min_full_calmar=args.min_calmar,
        min_trades_per_year=4,
        min_total_trades=25,
    )
    if not survivors:
        print("No survivors to contribute. Run walk-forward first.")
        return
    result = telemetry.contribute_batch(store, survivors, limit=args.limit)
    print(
        f"\nContribution: {result['submitted']} submitted, "
        f"{result['errors']} errors, {result['total']} total."
    )
    if not telemetry.remote_enabled():
        print("\nNote: saved locally only. Set TERMINUS_TELEMETRY=1 or use "
              "--enable to share with the community hub.")


# --- ml --------------------------------------------------------------------

def cmd_ml_regime(args):
    print("Training regime classifier on BTC daily data...")
    t0 = time.time()
    try:
        from .ml.regime import train_regime_classifier
    except ImportError:
        print("ML deps missing. Run: pip install terminus-lab[ml]")
        return

    df = _load_and_precompute("BTCUSDT", "1d", args.days)
    if df is None:
        print("No BTC daily data found. Run: terminus fetch --pairs BTCUSDT --tfs 1d")
        return

    clf = train_regime_classifier(df)
    model_path = Path(args.output)
    clf.save(model_path)
    elapsed = time.time() - t0
    print(
        f"Regime model trained in {elapsed:.1f}s\n"
        f"  Bars: {clf.trained_on_bars}\n"
        f"  Train accuracy: {clf.train_accuracy:.1%}\n"
        f"  Saved to: {model_path.with_suffix('.xgb')}"
    )

    # Preview the current regime
    regime = clf.predict(df)
    last_10 = regime.iloc[-10:]
    print(f"\nLast 10 bars regime:")
    for i, r in zip(df["ts"].iloc[-10:], last_10):
        print(f"  {str(i)[:10]}  {r}")


# --- main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="terminus",
        description="Terminus — ruthless backtesting lab for long-only spot strategies.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    subp = p.add_subparsers(dest="cmd", required=True)

    # fetch
    f = subp.add_parser("fetch", help="Fetch & cache kline data.")
    f.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    f.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    f.add_argument("--days", type=int, default=2920)
    f.add_argument("--concurrency", type=int, default=3)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    # sweep
    s = subp.add_parser("sweep", help="Run the full strategy sweep.")
    s.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    s.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    s.add_argument("--days", type=int, default=2920)
    s.add_argument("--exit-methods", default="fixed_tp_stop")
    s.add_argument("--no-regime", action="store_true")
    s.add_argument("--label", default=None)
    s.set_defaults(func=cmd_sweep)

    # walk-forward
    w = subp.add_parser("walk-forward", help="Walk-forward top candidates.")
    w.add_argument("--top", type=int, default=15)
    w.add_argument("--min-calmar", type=float, default=1.5)
    w.add_argument("--min-trades", type=int, default=25)
    w.add_argument("--days", type=int, default=2920)
    w.add_argument("--pairs", default="")
    w.add_argument("--label", default=None)
    w.set_defaults(func=cmd_walk_forward)

    # report
    r = subp.add_parser("report", help="Filter survivors and print ranked report.")
    r.add_argument("--min-calmar", type=float, default=1.5)
    r.add_argument("--min-trades-total", type=int, default=25)
    r.add_argument("--min-trades-per-year", type=int, default=4)
    r.add_argument("--min-bear-return", type=float, default=-5.0)
    r.add_argument("--min-pairs-generalization", type=int, default=3)
    r.add_argument("--top", type=int, default=100)
    r.set_defaults(func=cmd_report)

    # portfolio
    po = subp.add_parser("portfolio", help="Construct a portfolio from survivors.")
    po.add_argument("--min-calmar", type=float, default=1.5)
    po.add_argument("--min-bear-return", type=float, default=-5.0)
    po.add_argument("--max-legs", type=int, default=6)
    po.add_argument("--max-corr", type=float, default=0.65)
    po.add_argument("--pool", type=int, default=60, help="Top-N pool before greedy select")
    po.add_argument("--target", type=float, default=25.0)
    po.set_defaults(func=cmd_portfolio)

    # contribute
    co = subp.add_parser(
        "contribute",
        help="Submit anonymised strategy performance to the community hub.",
    )
    co.add_argument("--min-calmar", type=float, default=1.5)
    co.add_argument("--limit", type=int, default=50,
                    help="Max survivors to submit per run.")
    co.add_argument("--enable", action="store_true",
                    help="Enable remote sharing (sets TERMINUS_TELEMETRY=1 this session).")
    co.set_defaults(func=cmd_contribute)

    # ml
    ml_p = subp.add_parser("ml", help="Machine learning utilities.")
    ml_sub = ml_p.add_subparsers(dest="ml_cmd", required=True)

    ml_regime = ml_sub.add_parser("regime", help="Train regime classifier on BTC daily data.")
    ml_regime.add_argument("--days", type=int, default=2920)
    ml_regime.add_argument(
        "--output",
        default=str(Path.home() / ".terminus" / "regime_model"),
        help="Output path (without extension; .xgb and .json written).",
    )
    ml_regime.set_defaults(func=cmd_ml_regime)

    args = p.parse_args()
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
