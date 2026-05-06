"""Rademacher-adjusted Sharpe ratio — deflates Sharpe based on strategy-space complexity.

When you test N strategies and pick the best one, the expected maximum Sharpe
of random noise strategies grows with sqrt(2 * ln(N)). The Rademacher penalty
estimates this "free lunch from multiple testing" and subtracts it.

If your Sharpe barely survives deflation, your edge is likely an artifact of
search-space size rather than a real signal.

References:
    - Harvey, Liu, Zhu (2016) "... and the Cross-Section of Expected Returns"
    - Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
    - Adapted from rademacher-anti-serum (GitHub)

Usage:
    from terminus.risk.rademacher import deflated_sharpe, rademacher_penalty

    # After a sweep that tested 5000 configs:
    penalty = rademacher_penalty(n_strategies=5000)
    deflated = deflated_sharpe(sharpe=1.8, n_strategies=5000,
                                n_trades=200, skewness=-0.2, kurtosis=4.1)
    # deflated < sharpe; if deflated < 0 → likely overfit
"""
from __future__ import annotations

import math

import numpy as np


def rademacher_penalty(n_strategies: int) -> float:
    """Expected maximum Sharpe of N random strategies (Bonferroni-style bound).

    Based on the asymptotic result: E[max(Z_1..Z_N)] ~ sqrt(2 * ln(N))
    for N iid standard normal Z_i.

    Args:
        n_strategies: number of strategies tested in the search space

    Returns:
        Expected Sharpe "haircut" — subtract from observed Sharpe
    """
    if n_strategies <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(n_strategies))


def sharpe_std_error(
    sharpe: float,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of the Sharpe ratio estimate.

    Accounts for non-normality of return distributions (crypto is fat-tailed).

    Args:
        sharpe: observed Sharpe ratio
        n_trades: number of independent trades
        skewness: skewness of trade returns (negative = left tail)
        kurtosis: kurtosis of trade returns (>3 = fat tails)

    Returns:
        Standard error of the Sharpe estimate
    """
    if n_trades < 2:
        return float("inf")
    # Lo (2002) + Opdyke (2007) adjustment for non-normal returns
    excess_kurt = kurtosis - 3.0
    se_sq = (
        1.0
        + 0.5 * sharpe**2
        - skewness * sharpe
        + (excess_kurt / 4.0) * sharpe**2
    ) / n_trades
    return math.sqrt(max(se_sq, 0.0))


def deflated_sharpe(
    sharpe: float,
    n_strategies: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio — adjusts for multiple testing and non-normality.

    Subtracts the expected maximum Sharpe from random search, then normalizes
    by the standard error. A deflated Sharpe > 0 suggests the strategy has
    genuine edge beyond what luck from searching N configs could explain.

    Args:
        sharpe: observed annualized Sharpe ratio
        n_strategies: total configs tested in the sweep
        n_trades: number of trades in the backtest
        skewness: skewness of per-trade returns
        kurtosis: kurtosis of per-trade returns

    Returns:
        Deflated Sharpe ratio (can be negative → likely overfit)
    """
    penalty = rademacher_penalty(n_strategies)
    se = sharpe_std_error(sharpe, n_trades, skewness, kurtosis)

    if se == 0 or se == float("inf"):
        return 0.0

    # Deflated = (observed - expected_max_random) / SE
    return (sharpe - penalty) / se


def deflated_sharpe_simple(
    sharpe: float,
    n_strategies: int,
) -> float:
    """Simplified deflated Sharpe — just subtract the Rademacher penalty.

    For quick screening: if sharpe - penalty <= 0, likely overfit.

    Args:
        sharpe: observed Sharpe ratio
        n_strategies: number of strategies tested

    Returns:
        sharpe - sqrt(2 * ln(N))
    """
    return sharpe - rademacher_penalty(n_strategies)


def batch_deflate(
    sharpes: np.ndarray,
    n_strategies: int,
    n_trades: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Rademacher deflation to an array of Sharpe ratios.

    Args:
        sharpes: array of observed Sharpe ratios
        n_strategies: total configs tested
        n_trades: array of trade counts per strategy (optional,
                  uses simplified method if not provided)

    Returns:
        Array of deflated Sharpe ratios
    """
    penalty = rademacher_penalty(n_strategies)
    if n_trades is None:
        return sharpes - penalty
    # Full deflation per strategy
    result = np.zeros_like(sharpes, dtype=float)
    for i in range(len(sharpes)):
        se = sharpe_std_error(float(sharpes[i]), int(n_trades[i]))
        if se > 0 and se != float("inf"):
            result[i] = (float(sharpes[i]) - penalty) / se
        else:
            result[i] = 0.0
    return result
