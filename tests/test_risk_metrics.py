"""Tests for terminus.risk.metrics — VaR, CVaR, tail metrics, incremental VaR."""
import numpy as np
import pytest

from terminus.risk.metrics import (
    compute_var,
    compute_cvar,
    compute_tail_ratio,
    compute_omega_ratio,
    compute_full_metrics,
    incremental_var,
    cvar_filter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def _normal_returns(n=200, mean=0.002, std=0.02):
    return RNG.normal(mean, std, n)

def _losing_returns(n=200):
    return RNG.normal(-0.01, 0.02, n)

def _right_skewed(n=500):
    """Many small losses, few large gains."""
    base = RNG.normal(-0.005, 0.01, n)
    base[RNG.integers(0, n, 20)] = 0.15  # fat right tail
    return base

def _left_skewed(n=500):
    """Many small gains, few large losses."""
    base = RNG.normal(0.005, 0.01, n)
    base[RNG.integers(0, n, 20)] = -0.15  # fat left tail
    return base

def _make_trades(pnl_pcts):
    return [{"pnl_pct": p} for p in pnl_pcts]


# ---------------------------------------------------------------------------
# compute_var
# ---------------------------------------------------------------------------

class TestComputeVar:
    def test_empty_returns_zero(self):
        assert compute_var(np.array([])) == 0.0

    def test_positive_result(self):
        r = _normal_returns()
        assert compute_var(r) > 0.0

    def test_confidence_95_vs_99(self):
        r = _normal_returns(1000)
        var95 = compute_var(r, 0.95)
        var99 = compute_var(r, 0.99)
        assert var99 >= var95

    def test_uniform_distribution(self):
        # Uniform on [-1, 1] → 5th percentile = -0.9 → VaR95 ≈ 0.9
        r = np.linspace(-1.0, 1.0, 1000)
        var = compute_var(r, 0.95)
        assert pytest.approx(var, abs=0.02) == 0.9

    def test_all_positive_returns_var_zero_or_negative(self):
        r = np.full(100, 0.01)  # all gains
        var = compute_var(r, 0.95)
        assert var <= 0.0  # no losses → VaR is non-positive


# ---------------------------------------------------------------------------
# compute_cvar
# ---------------------------------------------------------------------------

class TestComputeCVar:
    def test_empty_returns_zero(self):
        assert compute_cvar(np.array([])) == 0.0

    def test_cvar_gte_var(self):
        r = _normal_returns(1000)
        assert compute_cvar(r) >= compute_var(r)

    def test_cvar_gte_var_for_skewed(self):
        r = _left_skewed()
        assert compute_cvar(r) >= compute_var(r)

    def test_positive_result_for_losses(self):
        r = _losing_returns()
        assert compute_cvar(r) > 0.0

    def test_confidence_99_gte_95(self):
        r = _normal_returns(1000)
        cvar95 = compute_cvar(r, 0.95)
        cvar99 = compute_cvar(r, 0.99)
        assert cvar99 >= cvar95


# ---------------------------------------------------------------------------
# compute_tail_ratio
# ---------------------------------------------------------------------------

class TestComputeTailRatio:
    def test_empty_returns_one(self):
        assert compute_tail_ratio(np.array([])) == 1.0

    def test_right_skewed_gt_one(self):
        r = _right_skewed()
        assert compute_tail_ratio(r) > 1.0

    def test_left_skewed_lt_one(self):
        r = _left_skewed()
        assert compute_tail_ratio(r) < 1.0

    def test_symmetric_near_one(self):
        r = RNG.normal(0, 0.02, 5000)
        ratio = compute_tail_ratio(r)
        assert 0.7 < ratio < 1.3

    def test_no_losses_returns_positive(self):
        r = np.full(100, 0.01)
        assert compute_tail_ratio(r) >= 0.0


# ---------------------------------------------------------------------------
# compute_omega_ratio
# ---------------------------------------------------------------------------

class TestComputeOmegaRatio:
    def test_empty_returns_one(self):
        assert compute_omega_ratio(np.array([])) == 1.0

    def test_profitable_strategy_gt_one(self):
        r = _normal_returns(500, mean=0.005)
        assert compute_omega_ratio(r) > 1.0

    def test_losing_strategy_lt_one(self):
        r = _losing_returns()
        assert compute_omega_ratio(r) < 1.0

    def test_breakeven_near_one(self):
        r = np.concatenate([np.full(50, 0.01), np.full(50, -0.01)])
        ratio = compute_omega_ratio(r)
        assert pytest.approx(ratio, abs=0.1) == 1.0

    def test_no_losses_returns_inf_or_large(self):
        r = np.full(100, 0.01)
        ratio = compute_omega_ratio(r)
        assert ratio == float("inf") or ratio > 1.0


# ---------------------------------------------------------------------------
# compute_full_metrics
# ---------------------------------------------------------------------------

class TestComputeFullMetrics:
    def test_empty_trades(self):
        m = compute_full_metrics([], total_return_pct=0.0, max_drawdown_pct=10.0)
        assert m.n_trades == 0
        assert m.var_95 == 0.0
        assert m.cvar_95 == 0.0

    def test_100_mixed_trades(self):
        pnls = list(RNG.normal(0.5, 1.5, 100))  # pnl_pct in %
        trades = _make_trades(pnls)
        m = compute_full_metrics(trades, total_return_pct=30.0, max_drawdown_pct=15.0)
        assert m.n_trades == 100
        assert 0.0 <= m.win_rate <= 1.0
        assert m.cvar_95 >= m.var_95
        assert m.calmar == pytest.approx(2.0, abs=0.01)

    def test_calmar_zero_drawdown(self):
        trades = _make_trades([0.5] * 50)
        m = compute_full_metrics(trades, total_return_pct=25.0, max_drawdown_pct=0.0)
        assert m.calmar == 0.0  # no drawdown → calmar undefined, returns 0

    def test_win_rate_range(self):
        trades = _make_trades([1.0] * 70 + [-1.0] * 30)
        m = compute_full_metrics(trades, total_return_pct=40.0, max_drawdown_pct=10.0)
        assert pytest.approx(m.win_rate, abs=0.01) == 0.7

    def test_var_99_gte_var_95(self):
        pnls = list(RNG.normal(0.3, 2.0, 200))
        trades = _make_trades(pnls)
        m = compute_full_metrics(trades, 20.0, 10.0)
        assert m.var_99 >= m.var_95


# ---------------------------------------------------------------------------
# incremental_var
# ---------------------------------------------------------------------------

class TestIncrementalVar:
    def test_empty_inputs_return_zeros(self):
        result = incremental_var(np.array([]), np.array([]))
        assert result["incremental_var"] == 0.0
        assert result["portfolio_var_before"] == 0.0

    def test_keys_present(self):
        port = _normal_returns(200)
        new_pos = _normal_returns(200)
        result = incremental_var(port, new_pos)
        for key in ["standalone_var", "portfolio_var_before", "portfolio_var_after",
                    "incremental_var", "diversification_benefit"]:
            assert key in result

    def test_correlated_position_increases_var(self):
        port = _normal_returns(500, mean=0.001, std=0.02)
        # Positively correlated: same direction
        correlated = port + RNG.normal(0, 0.005, 500)
        result = incremental_var(port, correlated, weight=0.2)
        assert result["portfolio_var_after"] >= result["portfolio_var_before"] * 0.95

    def test_negatively_correlated_reduces_var(self):
        port = _normal_returns(500, mean=0.001, std=0.02)
        anti = -port + RNG.normal(0, 0.003, 500)
        result = incremental_var(port, anti, weight=0.1)
        assert result["portfolio_var_after"] < result["portfolio_var_before"]

    def test_weight_zero_no_change(self):
        port = _normal_returns(200)
        new_pos = _normal_returns(200)
        result = incremental_var(port, new_pos, weight=0.0)
        assert pytest.approx(result["portfolio_var_before"], abs=1e-9) == result["portfolio_var_after"]


# ---------------------------------------------------------------------------
# cvar_filter
# ---------------------------------------------------------------------------

class TestCVarFilter:
    def test_empty_trades_fails(self):
        assert cvar_filter([]) is False

    def test_small_losers_pass(self):
        trades = _make_trades([-0.3] * 50 + [1.0] * 50)  # pnl_pct %
        # CVaR of -0.3% trades = 0.003 fraction < 0.05 threshold
        assert cvar_filter(trades, max_cvar_95=0.05) is True

    def test_large_tail_losses_fail(self):
        # 5% of trades lose 80% → CVaR will exceed typical thresholds
        trades = _make_trades([-80.0] * 10 + [2.0] * 190)
        assert cvar_filter(trades, max_cvar_95=0.05) is False

    def test_tight_threshold(self):
        trades = _make_trades([-1.0] * 100)  # all -1% → CVaR = 0.01
        assert cvar_filter(trades, max_cvar_95=0.02) is True
        assert cvar_filter(trades, max_cvar_95=0.005) is False
