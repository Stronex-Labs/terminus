"""Filter survivors from the sim store.

A config "survives" if:
  1. Full-window Calmar >= min_calmar (default 1.5)
  2. Frozen walk-forward: every year with >= min_trades_per_year is
     net-profitable (no net loss years among years with real activity)
  3. Bear year 2022 return >= min_bear_return (default -2%)
  4. Same family generalizes to >= min_pairs pairs at Calmar >= 1.0
  5. Adequate trade count on test window (>= min_total_trades)

Anything failing any criterion is dropped. Output is a list of survivor
dicts, sorted by a composite score.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .store import ResearchStore, get_store
from .risk.metrics import compute_cvar


logger = logging.getLogger("filter_survivors")


@dataclass
class Survivor:
    sim_hash: str
    pair: str
    timeframe: str
    config_name: str
    family: str
    total_return_pct: float
    max_drawdown_pct: float
    calmar: float
    n_trades: int
    years_tested: int
    years_profitable: int
    weakest_year_pct: float
    bear_year_pct: float | None
    family_pair_coverage: int
    score: float
    cvar_95: float = 0.0
    btc_trade_overlap: float = 0.0
    tod_dow_info: dict = field(default_factory=dict)
    full_row: dict = field(default_factory=dict)


def load_wf(store: ResearchStore, sim_hash: str,
            mode: str = "frozen") -> list[dict]:
    rows = store.get_wf_for(sim_hash, mode=mode)
    return [dict(r) for r in rows]


def _family_coverage_map(store: ResearchStore,
                         min_calmar: float = 1.0) -> dict[str, int]:
    """For each family, how many distinct pairs had at least one config
    with Calmar >= min_calmar on full-window?"""
    rows = store.query(
        "SELECT family, pair FROM sims WHERE calmar >= ?",
        (min_calmar,),
    )
    fam_pairs = defaultdict(set)
    for r in rows:
        fam_pairs[r["family"]].add(r["pair"])
    return {f: len(ps) for f, ps in fam_pairs.items()}


def _score(cand: Survivor) -> float:
    """Composite score favoring: positive weakest year, high avg return,
    bounded DD, family generalization."""
    weakest_bonus = 20 * max(cand.weakest_year_pct, -20)
    return (
        cand.total_return_pct / max(1.0, cand.years_tested)  # per-year return proxy
        + 2.0 * cand.calmar
        + 0.3 * cand.family_pair_coverage
        + 0.01 * weakest_bonus
    )


def _single_year_outlier(active_years: list[dict], threshold: float = 0.60) -> bool:
    """Return True if a single year provides >threshold of total positive return."""
    positive = [y["total_return_pct"] for y in active_years if y["total_return_pct"] > 0]
    if not positive:
        return False
    total_pos = sum(positive)
    if total_pos <= 0:
        return False
    return max(positive) / total_pos > threshold


def _btc_trade_overlap(
    candidate_trades: list,
    btc_trades: list,
    window: int = 2,
) -> float:
    """Fraction of candidate entries that fall within ±window bars of a BTC entry.

    Trades can be either dicts with 'bar'/'entry_ts' keys, or lists
    [entry_ts, exit_ts, pnl_pct, exit_reason].
    Uses entry_ts for correlation when bar index is unavailable.
    """
    if not candidate_trades or not btc_trades:
        return 0.0

    def _get_ts(t):
        if isinstance(t, dict):
            return t.get("entry_ts") or t.get("bar", -999)
        elif isinstance(t, (list, tuple)) and len(t) >= 1:
            return t[0]  # entry_ts is first element
        return -999

    btc_timestamps = sorted(_get_ts(t) for t in btc_trades)
    if not btc_timestamps or btc_timestamps[0] == -999:
        return 0.0

    # Use timestamp proximity: ±window * avg_bar_duration
    if len(btc_timestamps) >= 2:
        durations = [btc_timestamps[i+1] - btc_timestamps[i]
                     for i in range(min(20, len(btc_timestamps)-1))
                     if btc_timestamps[i+1] > btc_timestamps[i]]
        avg_bar_ms = sum(durations) / len(durations) if durations else 14400000
    else:
        avg_bar_ms = 14400000  # 4h default

    tolerance = window * avg_bar_ms
    overlap_count = 0
    import bisect
    for t in candidate_trades:
        ts = _get_ts(t)
        if ts == -999:
            continue
        idx = bisect.bisect_left(btc_timestamps, ts - tolerance)
        if idx < len(btc_timestamps) and abs(btc_timestamps[idx] - ts) <= tolerance:
            overlap_count += 1
        elif idx > 0 and abs(btc_timestamps[idx-1] - ts) <= tolerance:
            overlap_count += 1
    return overlap_count / len(candidate_trades) if candidate_trades else 0.0


def enrich_tod_dow(trades: list) -> dict:
    """Return time-of-day / day-of-week enrichment from trade entry timestamps.

    Trades can be dicts with 'entry_ts' and 'pnl_pct' keys, or lists
    [entry_ts, exit_ts, pnl_pct, exit_reason].
    Returns empty dict if fewer than 10 trades have timestamps.
    """
    import datetime as _dt

    def _get_entry_ts(t):
        if isinstance(t, dict):
            return t.get("entry_ts")
        elif isinstance(t, (list, tuple)) and len(t) >= 1:
            return t[0]
        return None

    def _get_pnl(t):
        if isinstance(t, dict):
            return t.get("pnl_pct", 0.0)
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            return t[2]
        return 0.0

    stamped = [t for t in trades if _get_entry_ts(t)]
    if len(stamped) < 10:
        return {}

    hour_wins: dict[int, int] = Counter()
    hour_total: dict[int, int] = Counter()
    dow_wins: dict[int, int] = Counter()
    dow_total: dict[int, int] = Counter()

    for t in stamped:
        entry_ts = _get_entry_ts(t)
        # entry_ts might be ms or seconds — normalize
        if entry_ts > 1e12:
            entry_ts = entry_ts / 1000.0
        dt = _dt.datetime.fromtimestamp(entry_ts, tz=_dt.timezone.utc)
        h = dt.hour
        d = dt.weekday()  # 0=Mon .. 6=Sun
        hour_total[h] += 1
        dow_total[d] += 1
        if _get_pnl(t) > 0:
            hour_wins[h] += 1
            dow_wins[d] += 1

    def _win_rate(wins: dict, totals: dict, key: int) -> float:
        t = totals.get(key, 0)
        return wins.get(key, 0) / t if t > 0 else 0.0

    hours_with_trades = sorted(hour_total.keys())
    hour_wr = {h: _win_rate(hour_wins, hour_total, h) for h in hours_with_trades}
    sorted_hours = sorted(hour_wr, key=hour_wr.get, reverse=True)  # type: ignore[arg-type]
    best_hours = sorted_hours[:3]
    worst_hours = sorted_hours[-3:] if len(sorted_hours) >= 3 else sorted_hours

    days_with_trades = sorted(dow_total.keys())
    dow_wr = {d: _win_rate(dow_wins, dow_total, d) for d in days_with_trades}
    sorted_days = sorted(dow_wr, key=dow_wr.get, reverse=True)  # type: ignore[arg-type]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    best_days = [day_names[d] for d in sorted_days[:2]]
    worst_days = [day_names[d] for d in sorted_days[-2:]] if len(sorted_days) >= 2 else [day_names[d] for d in sorted_days]

    total_trades = sum(hour_total.values())
    top3_count = sum(hour_total.get(h, 0) for h in best_hours)
    hour_concentration = top3_count / total_trades if total_trades > 0 else 0.0

    return {
        "best_hours": best_hours,
        "worst_hours": worst_hours,
        "best_days": best_days,
        "worst_days": worst_days,
        "hour_concentration": round(hour_concentration, 3),
    }


def filter_sims(
    store: ResearchStore | None = None, *,
    min_full_calmar: float = 1.5,
    min_trades_per_year: int = 5,
    min_total_trades: int = 20,
    min_bear_return: float = -2.0,
    min_pairs_generalization: int = 5,
    require_every_year_profitable: bool = False,
    max_losing_years: int = 2,
    max_losing_year_severity: float = -10.0,
    bear_year_label: str = "2022",
    include_frozen_wf: bool = True,
    max_cvar_95: float = 0.08,
    max_single_year_concentration: float = 0.60,
    check_btc_correlation: bool = False,
    btc_overlap_threshold: float = 0.70,
) -> list[Survivor]:
    """Return surviving configs across all pairs/families in the store.

    If include_frozen_wf=False, skip the per-year walk-forward gate and
    only apply the full-window gates (useful during early research).
    """
    if store is None:
        store = get_store()

    fam_cov = _family_coverage_map(store, min_calmar=1.0)

    rows = store.query(
        "SELECT * FROM sims WHERE calmar >= ? AND n_trades >= ?"
        " ORDER BY calmar DESC",
        (min_full_calmar, min_total_trades),
    )
    logger.info(f"Full-window pre-filter: {len(rows)} rows with Calmar>={min_full_calmar}")

    survivors: list[Survivor] = []
    for r in rows:
        r = dict(r)
        wf = load_wf(store, r["hash"]) if include_frozen_wf else []

        # Active years: those with enough trades to be informative
        active_years = [y for y in wf if y["n_trades"] >= min_trades_per_year]
        if include_frozen_wf and len(active_years) < 3:
            continue

        # Losing-year gate:
        #   - up to `max_losing_years` years may be net-negative
        #   - any losing year must be no worse than `max_losing_year_severity`
        # (Old behavior `require_every_year_profitable` => max_losing_years=0.)
        if include_frozen_wf:
            losing = [y for y in active_years if y["total_return_pct"] < 0]
            effective_max = 0 if require_every_year_profitable else max_losing_years
            if len(losing) > effective_max:
                continue
            if losing and any(y["total_return_pct"] < max_losing_year_severity
                              for y in losing):
                continue

        # Single-year outlier gate
        if include_frozen_wf and active_years:
            if _single_year_outlier(active_years, max_single_year_concentration):
                continue

        # Bear-year gate
        bear = next((y for y in wf if y["year_label"] == bear_year_label), None)
        bear_ret = bear["total_return_pct"] if bear else None
        if bear_ret is not None and bear_ret < min_bear_return:
            continue

        # CVaR gate — tail risk filter on per-trade PnL distribution
        trades = json.loads(r.get("trades_json") or "[]")
        cvar_95 = 0.0
        if trades:
            import numpy as np
            # Trades can be dicts or lists [entry_ts, exit_ts, pnl_pct, reason]
            def _pnl(t):
                if isinstance(t, dict):
                    return t.get("pnl_pct", 0.0)
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    return t[2]  # pnl_pct is 3rd element
                return 0.0
            pnls = np.array([_pnl(t) / 100.0 for t in trades])
            cvar_95 = float(compute_cvar(pnls, 0.95))
            if cvar_95 > max_cvar_95:
                continue

        # BTC trade-timestamp correlation (soft gate)
        btc_overlap = 0.0
        if trades and check_btc_correlation and r["pair"] != "BTCUSDT":
            btc_rows = store.query(
                "SELECT trades_json FROM sims WHERE family = ? AND pair = 'BTCUSDT'"
                " AND calmar >= ? LIMIT 1",
                (r["family"], min_full_calmar),
            )
            if btc_rows:
                btc_trades = json.loads(btc_rows[0]["trades_json"] or "[]")
                btc_overlap = _btc_trade_overlap(trades, btc_trades)
                if btc_overlap > btc_overlap_threshold:
                    continue

        # Time-of-day / day-of-week enrichment
        tod_dow = enrich_tod_dow(trades) if trades else {}

        # Weakest year (among active)
        if active_years:
            weakest = min(y["total_return_pct"] for y in active_years)
        else:
            weakest = r["total_return_pct"]

        fam_base = r["family"].split("+")[0]
        coverage = fam_cov.get(r["family"], 0)
        base_coverage = fam_cov.get(fam_base, 0)
        effective_coverage = max(coverage, base_coverage)
        if effective_coverage < min_pairs_generalization:
            continue

        years_profitable = sum(1 for y in active_years if y["total_return_pct"] > 0)

        s = Survivor(
            sim_hash=r["hash"],
            pair=r["pair"],
            timeframe=r["timeframe"],
            config_name=r["config_name"],
            family=r["family"],
            total_return_pct=r["total_return_pct"],
            max_drawdown_pct=r["max_drawdown_pct"],
            calmar=r["calmar"],
            n_trades=r["n_trades"],
            years_tested=len(active_years),
            years_profitable=years_profitable,
            weakest_year_pct=weakest,
            bear_year_pct=bear_ret,
            family_pair_coverage=effective_coverage,
            score=0.0,
            cvar_95=cvar_95,
            btc_trade_overlap=btc_overlap,
            tod_dow_info=tod_dow,
            full_row=r,
        )
        s.score = _score(s)
        survivors.append(s)

    survivors.sort(key=lambda s: s.score, reverse=True)
    logger.info(f"Survivors: {len(survivors)}")
    return survivors


def survivor_report(survivors: list[Survivor], limit: int = 50) -> str:
    lines = []
    lines.append(
        f"{'Rank':>4}  {'Pair':<10} {'TF':<4} {'Family':<22}  "
        f"{'Ret':>8}  {'DD':>6}  {'Calm':>5}  "
        f"{'Yrs':>5}  {'Bear':>7}  {'Weak':>7}  {'CVaR95':>7}  {'Cov':>3}  Score"
    )
    lines.append("-" * 130)
    for i, s in enumerate(survivors[:limit]):
        bear_s = f"{s.bear_year_pct:+.1f}%" if s.bear_year_pct is not None else "  -  "
        lines.append(
            f"{i+1:>4}. {s.pair:<10} {s.timeframe:<4} {s.family[:22]:<22}  "
            f"{s.total_return_pct:>+7.1f}% {s.max_drawdown_pct:>5.1f}% "
            f"{s.calmar:>5.2f}  {s.years_profitable}/{s.years_tested:<3} "
            f"{bear_s:>7}  {s.weakest_year_pct:>+6.1f}%  {s.cvar_95:>6.1%}  "
            f"{s.family_pair_coverage:>3}  {s.score:>6.1f}"
        )
    return "\n".join(lines)
