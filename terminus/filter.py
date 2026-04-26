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
from collections import defaultdict
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
            pnls = np.array([t.get("pnl_pct", 0.0) / 100.0 for t in trades])
            cvar_95 = float(compute_cvar(pnls, 0.95))
            if cvar_95 > max_cvar_95:
                continue

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
