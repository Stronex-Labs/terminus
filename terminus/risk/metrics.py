"""Risk metrics — CVaR, VaR, drawdown analytics.

Extends the basic Sharpe/Calmar metrics already in Terminus with
tail-risk measures used in institutional risk management.

All functions operate on a list/array of per-trade PnL percentages
or a pd.Series of daily returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Full risk metric set for a backtest result."""
    sharpe: float
    calmar: float
    max_drawdown_pct: float
    var_95: float          # Value at Risk at 95% confidence (per-trade)
    var_99: float          # Value at Risk at 99% confidence
    cvar_95: float         # Conditional VaR / Expected Shortfall at 95%
    cvar_99: float         # Conditional VaR at 99%
    tail_ratio: float      # 95th percentile gain / 5th percentile loss
    omega_ratio: float     # probability-weighted gains / losses above threshold
    n_trades: int
    win_rate: float


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk — worst loss at confidence level (positive = loss).

    Args:
        returns: array of per-trade returns as fractions (e.g. 0.02 = 2%)
        confidence: confidence level (0.95 = 5th percentile)

    Returns:
        VaR as positive number (e.g. 0.03 = 3% expected max loss)
    """
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level.

    CVaR = mean of returns below the VaR threshold.
    More informative than VaR: captures average severity of tail losses.

    Args:
        returns: array of per-trade returns as fractions
        confidence: confidence level

    Returns:
        CVaR as positive number
    """
    if len(returns) == 0:
        return 0.0
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return float(-cutoff)
    return float(-tail.mean())


def compute_tail_ratio(returns: np.ndarray) -> float:
    """Tail ratio = 95th percentile gain / abs(5th percentile loss).

    > 1.0 means right tail is fatter than left tail — favorable asymmetry.
    """
    if len(returns) == 0:
        return 1.0
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    if p5 == 0:
        return float(p95) if p95 > 0 else 1.0
    return float(p95 / p5)


def compute_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega ratio — probability-weighted gains above threshold / losses below.

    Omega > 1.0 means the strategy generates more probability-weighted
    gain than loss relative to the threshold.
    """
    if len(returns) == 0:
        return 1.0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    gain_sum = gains.sum()
    loss_sum = losses.sum()
    if loss_sum == 0:
        return float("inf") if gain_sum > 0 else 1.0
    return float(gain_sum / loss_sum)


def compute_full_metrics(
    trades: list[dict],
    total_return_pct: float,
    max_drawdown_pct: float,
) -> RiskMetrics:
    """Compute full risk metric set from a list of trade dicts.

    Args:
        trades: list of dicts with 'pnl_pct' key (per-trade return as %)
        total_return_pct: total strategy return in %
        max_drawdown_pct: max drawdown in % (positive number)

    Returns:
        RiskMetrics dataclass
    """
    if not trades:
        return RiskMetrics(
            sharpe=0.0, calmar=0.0, max_drawdown_pct=max_drawdown_pct,
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            tail_ratio=1.0, omega_ratio=1.0, n_trades=0, win_rate=0.0,
        )

    pnls = np.array([t.get("pnl_pct", 0.0) / 100.0 for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0.0

    # Sharpe on per-trade returns (annualized assuming ~2 trades/week)
    mean_ret = float(pnls.mean())
    std_ret = float(pnls.std())
    sharpe = float(mean_ret / std_ret * np.sqrt(52)) if std_ret > 0 else 0.0

    # Calmar
    calmar = 0.0
    if max_drawdown_pct > 0:
        calmar = float(total_return_pct / max_drawdown_pct)

    return RiskMetrics(
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown_pct=max_drawdown_pct,
        var_95=compute_var(pnls, 0.95),
        var_99=compute_var(pnls, 0.99),
        cvar_95=compute_cvar(pnls, 0.95),
        cvar_99=compute_cvar(pnls, 0.99),
        tail_ratio=compute_tail_ratio(pnls),
        omega_ratio=compute_omega_ratio(pnls),
        n_trades=len(trades),
        win_rate=win_rate,
    )


def cvar_filter(
    trades: list[dict],
    max_cvar_95: float = 0.05,
) -> bool:
    """Return True if the strategy passes the CVaR gate.

    Args:
        trades: list of trade dicts with 'pnl_pct'
        max_cvar_95: maximum acceptable CVaR at 95% (e.g. 0.05 = 5% per trade)

    Returns:
        True if CVaR is within acceptable bounds
    """
    if not trades:
        return False
    pnls = np.array([t.get("pnl_pct", 0.0) / 100.0 for t in trades])
    cvar = compute_cvar(pnls, 0.95)
    return cvar <= max_cvar_95


def incremental_var(
    existing_returns: np.ndarray,
    new_position_returns: np.ndarray,
    weight: float = 0.1,
    confidence: float = 0.95,
) -> dict:
    """Compute incremental VaR of adding a new position to a portfolio.

    Args:
        existing_returns: current portfolio daily returns
        new_position_returns: new position's daily returns (same index)
        weight: fraction of portfolio to allocate to new position
        confidence: VaR confidence level

    Returns:
        dict with standalone_var, portfolio_var_before, portfolio_var_after,
        incremental_var, diversification_benefit
    """
    if len(existing_returns) == 0 or len(new_position_returns) == 0:
        return {
            "standalone_var": 0.0,
            "portfolio_var_before": 0.0,
            "portfolio_var_after": 0.0,
            "incremental_var": 0.0,
            "diversification_benefit": 0.0,
        }

    min_len = min(len(existing_returns), len(new_position_returns))
    port = existing_returns[-min_len:]
    new = new_position_returns[-min_len:]

    var_before = compute_var(port, confidence)
    combined = (1 - weight) * port + weight * new
    var_after = compute_var(combined, confidence)
    standalone = compute_var(new, confidence)
    incremental = var_after - var_before
    diversification = standalone * weight - incremental

    return {
        "standalone_var": float(standalone),
        "portfolio_var_before": float(var_before),
        "portfolio_var_after": float(var_after),
        "incremental_var": float(incremental),
        "diversification_benefit": float(diversification),
    }
