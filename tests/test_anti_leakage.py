"""Anti-leakage unit tests — verify ML features contain zero future data.

These tests ensure the regime classifier and screener cannot accidentally
peek into future bars during feature computation or label generation.

Adapted from caiso-bess-arb anti-leakage testing discipline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high = close * (1 + rng.uniform(0, 0.03, n))
    low = close * (1 - rng.uniform(0, 0.03, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(1e6, 1e7, n)
    ts = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": opn, "high": high, "low": low,
        "close": close, "volume": volume,
        "open_time": ts.astype(np.int64) // 10**6,
    })


class TestRegimeFeatureLeakage:
    """Regime features must only use current and past bars, never future."""

    def test_features_dont_change_when_future_is_modified(self):
        """If we change bars [t+1:], features at bar t should be unchanged."""
        from terminus.ml.regime import _features

        df = _make_ohlcv(400)
        feat_full = _features(df)

        # Corrupt everything after bar 300
        df_corrupt = df.copy()
        df_corrupt.loc[301:, "close"] = 999999
        df_corrupt.loc[301:, "high"] = 999999
        df_corrupt.loc[301:, "low"] = 1
        df_corrupt.loc[301:, "volume"] = 0
        feat_corrupt = _features(df_corrupt)

        # Features at bar 300 and before must be identical
        target_idx = 300
        for col in feat_full.columns:
            orig = feat_full[col].iloc[target_idx]
            modified = feat_corrupt[col].iloc[target_idx]
            if pd.notna(orig):
                assert orig == pytest.approx(modified, rel=1e-10), (
                    f"Feature '{col}' at bar {target_idx} changed when future "
                    f"was modified: {orig} vs {modified}"
                )

    def test_features_at_bar_t_independent_of_future_bars(self):
        """Features computed on df[:t+1] must equal features on df[:] at bar t."""
        from terminus.ml.regime import _features

        df = _make_ohlcv(400)
        feat_full = _features(df)

        # Compute features on truncated df (up to bar 250 only)
        t = 250
        df_trunc = df.iloc[:t + 1].copy().reset_index(drop=True)
        feat_trunc = _features(df_trunc)

        # Last bar of truncated should match bar t of full
        for col in feat_full.columns:
            full_val = feat_full[col].iloc[t]
            trunc_val = feat_trunc[col].iloc[-1]
            if pd.notna(full_val) and pd.notna(trunc_val):
                assert full_val == pytest.approx(trunc_val, rel=1e-10), (
                    f"Feature '{col}' differs between full and truncated: "
                    f"{full_val} vs {trunc_val}"
                )

    def test_labels_use_forward_returns_only(self):
        """Auto-labels must use forward returns (future), but features must not."""
        from terminus.ml.regime import _auto_labels, FORWARD_BARS

        df = _make_ohlcv(300)
        labels = _auto_labels(df)

        # Last FORWARD_BARS labels should be NaN/chop (no future data)
        # Actually they become chop because fwd_ret is NaN → condition is False
        tail_labels = labels.iloc[-FORWARD_BARS:]
        # These should all be 'chop' since NaN comparisons are False
        assert all(l == "chop" for l in tail_labels), (
            "Labels in the last FORWARD_BARS should default to 'chop' "
            "(no future data available)"
        )

    def test_no_future_leakage_in_rolling_indicators(self):
        """Rolling windows (EMA, SMA, ATR) must use only past data."""
        from terminus.ml.regime import _features

        df = _make_ohlcv(500)

        # Set a "shock" at bar 400 — massive spike
        df_shock = df.copy()
        df_shock.loc[400, "close"] = df["close"].iloc[399] * 10
        df_shock.loc[400, "high"] = df["close"].iloc[399] * 10
        df_shock.loc[400, "volume"] = df["volume"].iloc[399] * 100

        feat_normal = _features(df)
        feat_shock = _features(df_shock)

        # Features BEFORE the shock (bar 399 and earlier) must be identical
        for col in feat_normal.columns:
            orig = feat_normal[col].iloc[399]
            shocked = feat_shock[col].iloc[399]
            if pd.notna(orig) and pd.notna(shocked):
                assert orig == pytest.approx(shocked, rel=1e-10), (
                    f"Feature '{col}' at bar 399 leaked future shock at 400: "
                    f"{orig} vs {shocked}"
                )


class TestRademacherSanity:
    """Basic sanity checks for the Rademacher deflation module."""

    def test_penalty_grows_with_n(self):
        from terminus.risk.rademacher import rademacher_penalty
        p10 = rademacher_penalty(10)
        p100 = rademacher_penalty(100)
        p1000 = rademacher_penalty(1000)
        assert 0 < p10 < p100 < p1000

    def test_deflated_sharpe_below_observed(self):
        from terminus.risk.rademacher import deflated_sharpe_simple
        observed = 2.0
        deflated = deflated_sharpe_simple(observed, n_strategies=5000)
        assert deflated < observed

    def test_single_strategy_no_penalty(self):
        from terminus.risk.rademacher import rademacher_penalty
        assert rademacher_penalty(1) == 0.0

    def test_deflated_sharpe_full(self):
        from terminus.risk.rademacher import deflated_sharpe
        # A Sharpe of 3.0 with only 10 strategies tested should survive deflation
        result = deflated_sharpe(
            sharpe=3.0, n_strategies=10, n_trades=200,
            skewness=0.0, kurtosis=3.0,
        )
        assert result > 0

        # A Sharpe of 1.0 with 10000 strategies tested should NOT survive
        result_weak = deflated_sharpe(
            sharpe=1.0, n_strategies=10000, n_trades=50,
            skewness=-0.5, kurtosis=5.0,
        )
        assert result_weak < result


class TestWFE:
    """Walk-Forward Efficiency ratio."""

    def test_wfe_perfect(self):
        from terminus.walk_forward import compute_wfe
        # If OOS matches IS perfectly, WFE = 1.0
        assert compute_wfe(20.0, 20.0) == pytest.approx(1.0)

    def test_wfe_overfit(self):
        from terminus.walk_forward import compute_wfe
        # IS: +30%, OOS: +3% → WFE = 0.1 (overfit)
        assert compute_wfe(30.0, 3.0) == pytest.approx(0.1)

    def test_wfe_negative_oos(self):
        from terminus.walk_forward import compute_wfe
        # IS: +20%, OOS: -5% → WFE = -0.25 (broken)
        assert compute_wfe(20.0, -5.0) == pytest.approx(-0.25)

    def test_wfe_zero_is(self):
        from terminus.walk_forward import compute_wfe
        # Zero IS return → WFE = 0 (avoid division by zero)
        assert compute_wfe(0.0, 5.0) == 0.0
