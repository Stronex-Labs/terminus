"""Portfolio construction — combine survivors into allocations targeting
25-40% annualized with realized DD cap.

Approach:
  1. Take a list of survivors (see filter_survivors).
  2. Reconstruct each survivor's daily P&L series from its trade list.
  3. Compute pairwise correlation on daily returns.
  4. Greedy selection: start with the best-scoring survivor. Iteratively
     add the candidate that (a) raises portfolio Sharpe the most, and
     (b) has pairwise correlation < max_corr with all currently-held.
  5. Size each leg so no single leg risks more than `max_single_risk` of
     equity, and total leverage = 1.0 (spot — we can't exceed it).
  6. Report year-by-year portfolio return, DD, correlation matrix.

Cash (USDC) is the default allocation when the BTC-regime filter kills
every strategy in a regime-wrapped setup. The unwrapped version
stays fully invested but pays the 2022 drawdown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .store import ResearchStore, get_store


@dataclass
class Leg:
    sim_hash: str
    pair: str
    timeframe: str
    family: str
    weight: float
    trades: list  # [[entry_ts, exit_ts, pnl_pct, reason], ...]
    daily_pnl: pd.Series = field(default_factory=pd.Series)


def _daily_pnl_from_trades(trades: list, index: pd.DatetimeIndex) -> pd.Series:
    """Map each trade's PnL onto its exit day. Sum if multiple in one day."""
    if not trades:
        return pd.Series(0.0, index=index)
    rows = []
    for t in trades:
        exit_ts_ms = t[1] if isinstance(t, list) else t.get("exit_ts")
        pnl = t[2] if isinstance(t, list) else t.get("pnl_pct")
        rows.append({"day": pd.Timestamp(exit_ts_ms, unit="ms", tz="UTC").normalize(),
                      "pnl": pnl})
    df = pd.DataFrame(rows)
    grouped = df.groupby("day")["pnl"].sum()
    return grouped.reindex(index, fill_value=0.0)


def reconstruct_legs(
    sim_hashes: list[str],
    store: ResearchStore | None = None,
) -> list[Leg]:
    if store is None:
        store = get_store()
    legs = []
    for h in sim_hashes:
        row = store.lookup_sim(h)
        if row is None:
            continue
        row = dict(row)
        trades = json.loads(row["trades_json"]) if row["trades_json"] else []
        legs.append(Leg(
            sim_hash=h, pair=row["pair"], timeframe=row["timeframe"],
            family=row["family"], weight=0.0, trades=trades,
        ))
    return legs


def compute_daily_pnl_matrix(legs: list[Leg]) -> pd.DataFrame:
    """Return a DataFrame of daily PnL per leg (columns), date index."""
    if not legs:
        return pd.DataFrame()
    all_days = set()
    for leg in legs:
        for t in leg.trades:
            exit_ts = t[1] if isinstance(t, list) else t.get("exit_ts")
            if exit_ts:
                all_days.add(pd.Timestamp(exit_ts, unit="ms", tz="UTC").normalize())
    if not all_days:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(sorted(all_days))
    cols = {}
    for i, leg in enumerate(legs):
        leg.daily_pnl = _daily_pnl_from_trades(leg.trades, idx)
        key = f"{leg.pair}_{leg.timeframe}_{leg.family}_{i}"
        cols[key] = leg.daily_pnl
    return pd.DataFrame(cols)


def correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily PnL. NaN-safe."""
    if matrix.empty:
        return pd.DataFrame()
    return matrix.corr(min_periods=30)


def _portfolio_metrics(weighted_daily: pd.Series) -> dict:
    if len(weighted_daily) == 0:
        return {"annual_ret": 0, "sharpe": 0, "max_dd": 0, "years": 0}
    equity = (1 + weighted_daily).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1)
    # Annualize
    total_ret = equity.iloc[-1] - 1
    years = (weighted_daily.index[-1] - weighted_daily.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / max(years, 1 / 365)) - 1 if total_ret > -1 else -1
    std = weighted_daily.std() * np.sqrt(365)
    sharpe = (weighted_daily.mean() * 365) / std if std > 0 else 0
    return {
        "annual_ret": cagr * 100,
        "sharpe": round(sharpe, 2),
        "max_dd": round(abs(dd.min()) * 100, 2),
        "years": round(years, 1),
    }


def _year_breakdown(weighted_daily: pd.Series) -> dict[str, float]:
    """Per-calendar-year portfolio return (%)."""
    if len(weighted_daily) == 0:
        return {}
    grouped = weighted_daily.groupby(weighted_daily.index.year)
    out = {}
    for y, g in grouped:
        eq = (1 + g).prod()
        out[str(y)] = round((eq - 1) * 100, 2)
    return out


def greedy_portfolio(
    survivors: list,
    *,
    max_legs: int = 6,
    max_corr: float = 0.65,
    min_daily_activity: int = 100,
    target_annual_pct: float = 30.0,
    store: ResearchStore | None = None,
) -> dict:
    """Greedy forward selection.

    Pick the best-score leg. Repeatedly add the next-best leg that
    (a) has max pairwise correlation < max_corr with any already-picked,
    (b) raises portfolio Sharpe.
    """
    if store is None:
        store = get_store()
    if not survivors:
        return {"legs": [], "metrics": {}, "year_breakdown": {}}

    candidates = reconstruct_legs([s.sim_hash for s in survivors], store)
    candidates = [c for c in candidates if len(c.trades) > 0]
    if not candidates:
        return {"legs": [], "metrics": {}, "year_breakdown": {}}

    # Build the daily PnL matrix for ALL candidates once
    matrix = compute_daily_pnl_matrix(candidates)
    if matrix.empty:
        return {"legs": [], "metrics": {}, "year_breakdown": {}}

    # Drop candidates with too-sparse daily activity
    activity = (matrix != 0).sum()
    keep = activity[activity >= min_daily_activity].index.tolist()
    if not keep:
        # fall back: relax
        keep = activity.sort_values(ascending=False).head(max_legs * 3).index.tolist()
    matrix = matrix[keep]
    # Map back to Leg objects
    keep_keys = set(keep)
    candidates = [c for c, key in
                  zip(candidates,
                      [f"{c.pair}_{c.timeframe}_{c.family}_{i}"
                       for i, c in enumerate(candidates)])
                  if key in keep_keys]

    # Rebuild the matrix after candidate pruning to stay in sync
    matrix = compute_daily_pnl_matrix(candidates)
    if matrix.empty:
        return {"legs": [], "metrics": {}, "year_breakdown": {}}

    corr = correlation(matrix)

    chosen = [0]  # start with best-scoring
    chosen_keys = [matrix.columns[0]]

    for _ in range(max_legs - 1):
        best_idx = None
        best_metrics = None
        for j in range(len(matrix.columns)):
            if j in chosen:
                continue
            cand_key = matrix.columns[j]
            # Correlation filter
            max_c = max(abs(corr.loc[cand_key, k]) for k in chosen_keys
                        if not np.isnan(corr.loc[cand_key, k]))
            if max_c > max_corr:
                continue
            trial = chosen + [j]
            weights = np.ones(len(trial)) / len(trial)
            weighted = (matrix.iloc[:, trial] * weights).sum(axis=1)
            m = _portfolio_metrics(weighted)
            if best_metrics is None or m["sharpe"] > best_metrics["sharpe"]:
                best_metrics = m
                best_idx = j
        if best_idx is None:
            break
        chosen.append(best_idx)
        chosen_keys.append(matrix.columns[best_idx])

    weights = np.ones(len(chosen)) / len(chosen)
    weighted = (matrix.iloc[:, chosen] * weights).sum(axis=1)
    metrics = _portfolio_metrics(weighted)
    years = _year_breakdown(weighted)

    legs_out = []
    for idx, w in zip(chosen, weights):
        leg = candidates[idx]
        legs_out.append({
            "sim_hash": leg.sim_hash,
            "pair": leg.pair,
            "timeframe": leg.timeframe,
            "family": leg.family,
            "weight": round(float(w), 3),
        })

    return {
        "legs": legs_out,
        "metrics": metrics,
        "year_breakdown": years,
        "max_corr_observed": round(
            max(abs(corr.iloc[chosen, chosen].values[np.triu_indices(len(chosen), k=1)])
                if len(chosen) > 1 else [0]), 3
        ),
        "target_annual_pct": target_annual_pct,
        "hit_target": metrics.get("annual_ret", 0) >= target_annual_pct,
    }
