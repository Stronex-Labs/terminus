"""Crypto factor risk model — E * C * E^T decomposition.

Implements a lightweight multi-factor risk model adapted from the
Barra/Aladdin approach for crypto spot universes.

Factors (crypto-adapted):
  btc_beta        — rolling 60-bar beta to BTC
  momentum_1m     — 20-bar log return
  momentum_3m     — 60-bar log return
  volatility      — 20-bar ATR / close
  volume          — 20-bar relative volume vs 60-bar avg
  size            — log(avg_daily_volume_usdt) — proxy for market cap tier

Usage:
    model = CryptoFactorModel()
    model.fit(price_dict)          # {pair: OHLCV_df}
    report = model.decompose(weights)  # {pair: weight_fraction}
    stressed = model.stress_test(weights, scenario="crisis_2022")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("terminus.risk.factor_model")

ScenarioName = Literal["normal", "crisis_2022", "crisis_2018"]

_FACTOR_NAMES = [
    "btc_beta", "momentum_1m", "momentum_3m",
    "volatility", "volume", "size",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(df: pd.DataFrame, btc_returns: pd.Series) -> pd.Series:
    """Extract a single factor exposure vector for one pair."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ret = close.pct_change()

    # BTC beta — OLS over last 60 bars
    aligned = ret.align(btc_returns, join="inner")[0]
    btc_aligned = btc_returns.reindex(aligned.index)
    valid = aligned.dropna().index.intersection(btc_aligned.dropna().index)
    valid = valid[-60:]  # use last 60 bars
    if len(valid) >= 20:
        x = btc_aligned.loc[valid].values
        y = aligned.loc[valid].values
        cov = np.cov(x, y)
        btc_beta = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else 1.0
    else:
        btc_beta = 1.0

    # Momentum factors
    n = len(close)
    momentum_1m = float(close.iloc[-1] / close.iloc[max(0, n - 20)] - 1) if n >= 20 else 0.0
    momentum_3m = float(close.iloc[-1] / close.iloc[max(0, n - 60)] - 1) if n >= 60 else 0.0

    # Volatility
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    volatility = float((atr / close).iloc[-1]) if len(atr.dropna()) > 0 else 0.02

    # Volume factor
    vol_20 = volume.rolling(20).mean()
    vol_60 = volume.rolling(60).mean()
    volume_factor = float((vol_20 / vol_60.replace(0, np.nan)).iloc[-1]) if len(vol_60.dropna()) > 0 else 1.0

    # Size proxy — log avg daily volume
    avg_vol_usdt = float((volume * close).rolling(20).mean().iloc[-1]) if n >= 20 else 1.0
    size = float(np.log1p(max(avg_vol_usdt, 1)))

    return pd.Series({
        "btc_beta": btc_beta,
        "momentum_1m": momentum_1m,
        "momentum_3m": momentum_3m,
        "volatility": volatility,
        "volume": volume_factor,
        "size": size,
    })


# ---------------------------------------------------------------------------
# Factor covariance scenarios
# ---------------------------------------------------------------------------

# Pre-defined stress covariance matrices (scaled relative to normal).
# These approximate correlation-breakdown conditions in 2018 and 2022.
# Normal: unit diagonal, moderate off-diagonal correlations.
# Crisis: amplified variance, correlations spike toward 1 (all sell together).

def _normal_covariance() -> np.ndarray:
    """Baseline factor covariance matrix."""
    # Correlation-based — factors are moderately correlated
    corr = np.array([
        # btc_b  mom1m  mom3m  vol    volume size
        [1.00,   0.60,  0.50,  0.30,  0.20,  0.10],  # btc_beta
        [0.60,   1.00,  0.75,  0.20,  0.25,  0.05],  # momentum_1m
        [0.50,   0.75,  1.00,  0.25,  0.20,  0.10],  # momentum_3m
        [0.30,   0.20,  0.25,  1.00,  0.15,  0.05],  # volatility
        [0.20,   0.25,  0.20,  0.15,  1.00,  0.30],  # volume
        [0.10,   0.05,  0.10,  0.05,  0.30,  1.00],  # size
    ])
    # Diagonal variances (typical factor annualized std)
    std = np.array([0.30, 0.25, 0.35, 0.20, 0.15, 0.10])
    D = np.diag(std)
    return D @ corr @ D


def _crisis_covariance(severity: float = 1.5) -> np.ndarray:
    """Crisis factor covariance — correlations spike, variances increase."""
    # During crypto crises all factors spike toward BTC beta
    corr = np.array([
        [1.00,   0.90,  0.85,  0.65,  0.50,  0.20],  # btc_beta
        [0.90,   1.00,  0.90,  0.55,  0.45,  0.15],  # momentum_1m
        [0.85,   0.90,  1.00,  0.60,  0.40,  0.15],  # momentum_3m
        [0.65,   0.55,  0.60,  1.00,  0.35,  0.10],  # volatility
        [0.50,   0.45,  0.40,  0.35,  1.00,  0.25],  # volume
        [0.20,   0.15,  0.15,  0.10,  0.25,  1.00],  # size
    ])
    std = np.array([0.30, 0.25, 0.35, 0.20, 0.15, 0.10]) * severity
    D = np.diag(std)
    return D @ corr @ D


_COVARIANCE_SCENARIOS: dict[ScenarioName, np.ndarray] = {
    "normal": _normal_covariance(),
    "crisis_2022": _crisis_covariance(severity=1.8),
    "crisis_2018": _crisis_covariance(severity=2.2),
}


# ---------------------------------------------------------------------------
# Factor timing — attractiveness scores
# ---------------------------------------------------------------------------

@dataclass
class FactorAttractiveness:
    """Composite attractiveness score per factor (−1 to +1)."""
    momentum_score: float = 0.0   # positive = momentum is working
    volatility_score: float = 0.0  # positive = vol expanding (potential breakouts)
    size_score: float = 0.0        # positive = large-cap leading (risk-on)
    composite: float = 0.0         # equal-weight blend

    def tilt(self) -> dict[str, float]:
        """Return factor weight adjustments (multiply base weights by 1 + tilt)."""
        return {
            "btc_beta": 1.0 + 0.3 * self.momentum_score,
            "momentum_1m": 1.0 + 0.5 * self.momentum_score,
            "momentum_3m": 1.0 + 0.4 * self.momentum_score,
            "volatility": 1.0 + 0.3 * self.volatility_score,
            "volume": 1.0,
            "size": 1.0 + 0.2 * self.size_score,
        }


def compute_factor_attractiveness(
    returns_history: pd.DataFrame,  # columns = pairs, rows = bars
    lookback: int = 60,
    momentum_window: int = 20,
) -> FactorAttractiveness:
    """Compute factor attractiveness scores from recent returns history.

    Args:
        returns_history: DataFrame of daily returns, columns are pairs
        lookback: bars of history to use for scoring
        momentum_window: rolling window for momentum scoring

    Returns:
        FactorAttractiveness with composite score and per-factor components
    """
    if len(returns_history) < max(lookback, momentum_window + 10):
        return FactorAttractiveness()

    recent = returns_history.iloc[-lookback:].copy()

    # Momentum score: cross-sectional momentum working?
    # Compare top-quintile performers vs bottom over the lookback window
    cum_returns = (1 + recent).prod() - 1
    if len(cum_returns) >= 4:
        top_q = cum_returns.quantile(0.75)
        bot_q = cum_returns.quantile(0.25)
        spread = float(top_q - bot_q)
        # Normalize: typical spread is ~0.2, score in [-1, 1]
        momentum_score = float(np.clip(spread / 0.20, -1.0, 1.0))
    else:
        momentum_score = 0.0

    # Volatility score: is realized volatility expanding or contracting?
    # Expanding vol = positive score (potential for breakouts)
    early_vol = recent.iloc[:lookback // 2].std(axis=0).mean()
    late_vol = recent.iloc[lookback // 2:].std(axis=0).mean()
    if early_vol > 0:
        vol_ratio = late_vol / early_vol
        volatility_score = float(np.clip((vol_ratio - 1.0) / 0.5, -1.0, 1.0))
    else:
        volatility_score = 0.0

    # Size score: large-cap crypto (BTC+ETH) leading small caps?
    btc_cols = [c for c in recent.columns if "BTC" in c.upper()]
    eth_cols = [c for c in recent.columns if "ETH" in c.upper()]
    large_cap_cols = btc_cols + eth_cols
    small_cap_cols = [c for c in recent.columns if c not in large_cap_cols]

    if large_cap_cols and small_cap_cols:
        large_cum = float((1 + recent[large_cap_cols]).prod().mean() - 1)
        small_cum = float((1 + recent[small_cap_cols]).prod().mean() - 1)
        size_score = float(np.clip((large_cum - small_cum) / 0.15, -1.0, 1.0))
    else:
        size_score = 0.0

    composite = float(np.mean([momentum_score, volatility_score, size_score]))

    return FactorAttractiveness(
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        size_score=size_score,
        composite=composite,
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@dataclass
class FactorDecompositionResult:
    """Output of E * C * E^T decomposition for a portfolio."""
    portfolio_variance: float
    portfolio_vol_annualized: float
    factor_contributions: dict[str, float]   # factor -> % of variance explained
    marginal_contributions: dict[str, float]  # pair -> marginal variance contribution
    scenario: str
    cvat_95: float = 0.0  # CVaR at 95% from factor model (approximate)


@dataclass
class StressResult:
    """Comparison of normal vs crisis decomposition."""
    normal: FactorDecompositionResult
    crisis: FactorDecompositionResult
    variance_increase_pct: float
    vol_increase_pct: float


class CryptoFactorModel:
    """Multi-factor risk model for a crypto spot portfolio."""

    def __init__(self) -> None:
        self._exposure: pd.DataFrame | None = None   # (pairs × factors)
        self._pairs: list[str] = []

    def fit(self, price_dict: dict[str, pd.DataFrame]) -> "CryptoFactorModel":
        """Build factor exposure matrix from OHLCV data dict.

        Args:
            price_dict: {pair: OHLCV_df} — each df must have open/high/low/close/volume
        """
        btc_key = next(
            (k for k in price_dict if "BTC" in k.upper() and "USDT" in k.upper()),
            next(iter(price_dict), None),
        )
        if btc_key is None:
            raise ValueError("price_dict is empty")

        btc_returns = price_dict[btc_key]["close"].astype(float).pct_change()

        rows = {}
        for pair, df in price_dict.items():
            try:
                rows[pair] = _extract_features(df, btc_returns)
            except Exception as e:
                logger.warning(f"Factor extraction failed for {pair}: {e}")

        self._exposure = pd.DataFrame(rows).T[_FACTOR_NAMES]
        self._pairs = list(self._exposure.index)
        logger.info(f"Factor model fit on {len(self._pairs)} pairs")
        return self

    def decompose(
        self,
        weights: dict[str, float],
        scenario: ScenarioName = "normal",
    ) -> FactorDecompositionResult:
        """Run E * C * E^T decomposition for a weighted portfolio.

        Args:
            weights: {pair: fraction} — fractions should sum to ~1.0
            scenario: covariance scenario to use

        Returns:
            FactorDecompositionResult with variance attribution
        """
        if self._exposure is None:
            raise RuntimeError("Call .fit() first")

        pairs = [p for p in weights if p in self._exposure.index]
        if not pairs:
            raise ValueError("No weight pairs found in fitted exposure matrix")

        w = np.array([weights.get(p, 0.0) for p in pairs])
        E = self._exposure.loc[pairs].values.astype(np.float64)  # (n_assets × n_factors)
        C = _COVARIANCE_SCENARIOS[scenario]                        # (n_factors × n_factors)

        # Portfolio factor exposures: e_p = w^T * E  → (1 × n_factors)
        e_p = w @ E

        # Portfolio variance = e_p @ C @ e_p^T
        port_var = float(e_p @ C @ e_p.T)
        port_var = max(port_var, 1e-12)

        # Factor variance contributions
        factor_var = e_p * (C @ e_p)  # element-wise: (n_factors,)
        factor_total = float(factor_var.sum())
        factor_contributions = {
            name: float(factor_var[i] / port_var)
            for i, name in enumerate(_FACTOR_NAMES)
        }

        # Marginal variance contributions per pair
        mrc_numerator = E @ C @ e_p  # (n_assets,)
        marginal_contributions = {
            pair: float(w[i] * mrc_numerator[i] / port_var)
            for i, pair in enumerate(pairs)
        }

        port_vol_annual = float(np.sqrt(port_var * 252))

        # Approximate CVaR at 95% from portfolio vol (Gaussian assumption)
        # CVaR_95 ≈ vol * 2.063 (normal distribution, 1-day)
        cvat_95 = port_vol_annual / np.sqrt(252) * 2.063

        return FactorDecompositionResult(
            portfolio_variance=port_var,
            portfolio_vol_annualized=port_vol_annual,
            factor_contributions=factor_contributions,
            marginal_contributions=marginal_contributions,
            scenario=scenario,
            cvat_95=cvat_95,
        )

    def stress_test(
        self,
        weights: dict[str, float],
        crisis_scenario: ScenarioName = "crisis_2022",
    ) -> StressResult:
        """Run normal vs crisis decomposition and return comparison."""
        normal = self.decompose(weights, scenario="normal")
        crisis = self.decompose(weights, scenario=crisis_scenario)
        var_increase = (crisis.portfolio_variance - normal.portfolio_variance) / normal.portfolio_variance
        vol_increase = (crisis.portfolio_vol_annualized - normal.portfolio_vol_annualized) / normal.portfolio_vol_annualized
        return StressResult(
            normal=normal,
            crisis=crisis,
            variance_increase_pct=float(var_increase * 100),
            vol_increase_pct=float(vol_increase * 100),
        )

    def top_risk_contributors(
        self, weights: dict[str, float], n: int = 5
    ) -> list[tuple[str, float]]:
        """Return top-n pairs by marginal variance contribution."""
        result = self.decompose(weights)
        sorted_contribs = sorted(
            result.marginal_contributions.items(),
            key=lambda x: abs(x[1]), reverse=True,
        )
        return sorted_contribs[:n]
