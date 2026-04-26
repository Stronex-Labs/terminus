"""Tests for terminus.risk.factor_model — CryptoFactorModel and factor attractiveness."""
import numpy as np
import pandas as pd
import pytest

from terminus.risk.factor_model import (
    CryptoFactorModel,
    FactorAttractiveness,
    StressResult,
    compute_factor_attractiveness,
)

# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)


def _make_ohlcv(n=150, start_price=100.0, vol=0.02, seed=None):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, vol, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + abs(rng.normal(0, 0.005, n)))
    low = close * (1 - abs(rng.normal(0, 0.005, n)))
    volume = rng.uniform(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _three_pair_dict():
    return {
        "BTCUSDT": _make_ohlcv(150, start_price=30000, seed=1),
        "ETHUSDT": _make_ohlcv(150, start_price=2000, seed=2),
        "SOLUSDT": _make_ohlcv(150, start_price=50, seed=3),
    }


def _fitted_model():
    model = CryptoFactorModel()
    model.fit(_three_pair_dict())
    return model


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fits_three_pairs(self):
        model = _fitted_model()
        assert len(model._pairs) == 3

    def test_exposure_matrix_shape(self):
        model = _fitted_model()
        assert model._exposure.shape == (3, 6)

    def test_exposure_columns_are_factor_names(self):
        model = _fitted_model()
        expected = ["btc_beta", "momentum_1m", "momentum_3m", "volatility", "volume", "size"]
        assert list(model._exposure.columns) == expected

    def test_pairs_in_exposure_index(self):
        model = _fitted_model()
        assert "BTCUSDT" in model._exposure.index
        assert "ETHUSDT" in model._exposure.index

    def test_empty_price_dict_raises(self):
        with pytest.raises(ValueError):
            CryptoFactorModel().fit({})

    def test_exposure_no_nans(self):
        model = _fitted_model()
        assert not model._exposure.isnull().any().any()

    def test_fit_returns_self(self):
        model = CryptoFactorModel()
        result = model.fit(_three_pair_dict())
        assert result is model

    def test_single_pair_fits(self):
        model = CryptoFactorModel()
        model.fit({"BTCUSDT": _make_ohlcv(150, seed=10)})
        assert len(model._pairs) == 1


# ---------------------------------------------------------------------------
# decompose()
# ---------------------------------------------------------------------------

class TestDecompose:
    def test_portfolio_variance_positive(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2}
        result = model.decompose(w)
        assert result.portfolio_variance > 0.0

    def test_portfolio_vol_annualized_positive(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2}
        result = model.decompose(w)
        assert result.portfolio_vol_annualized > 0.0

    def test_factor_contributions_sum_near_one(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2}
        result = model.decompose(w)
        total = sum(result.factor_contributions.values())
        assert pytest.approx(total, abs=0.05) == 1.0

    def test_factor_contributions_keys(self):
        model = _fitted_model()
        result = model.decompose({"BTCUSDT": 1.0})
        expected_factors = {"btc_beta", "momentum_1m", "momentum_3m", "volatility", "volume", "size"}
        assert set(result.factor_contributions.keys()) == expected_factors

    def test_marginal_contributions_pairs_present(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
        result = model.decompose(w)
        assert "BTCUSDT" in result.marginal_contributions
        assert "ETHUSDT" in result.marginal_contributions

    def test_marginal_contributions_sum_near_one(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2}
        result = model.decompose(w)
        total = sum(result.marginal_contributions.values())
        assert pytest.approx(total, abs=0.05) == 1.0

    def test_cvat_95_positive(self):
        model = _fitted_model()
        result = model.decompose({"BTCUSDT": 1.0})
        assert result.cvat_95 > 0.0

    def test_scenario_normal_lower_than_crisis(self):
        model = _fitted_model()
        w = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
        normal = model.decompose(w, scenario="normal")
        crisis = model.decompose(w, scenario="crisis_2022")
        assert crisis.portfolio_variance > normal.portfolio_variance

    def test_before_fit_raises_runtime_error(self):
        model = CryptoFactorModel()
        with pytest.raises(RuntimeError):
            model.decompose({"BTCUSDT": 1.0})

    def test_no_matching_weights_raises_value_error(self):
        model = _fitted_model()
        with pytest.raises(ValueError):
            model.decompose({"XRPUSDT": 1.0})

    def test_unknown_pairs_in_weights_ignored(self):
        model = _fitted_model()
        # Mix of known and unknown — should work with known pairs only
        result = model.decompose({"BTCUSDT": 0.6, "XRPUSDT": 0.4})
        assert result.portfolio_variance > 0.0

    def test_single_pair_concentration(self):
        model = _fitted_model()
        result = model.decompose({"BTCUSDT": 1.0})
        assert result.portfolio_variance > 0.0
        assert "BTCUSDT" in result.marginal_contributions

    def test_scenario_stored_in_result(self):
        model = _fitted_model()
        result = model.decompose({"BTCUSDT": 1.0}, scenario="crisis_2018")
        assert result.scenario == "crisis_2018"


# ---------------------------------------------------------------------------
# stress_test()
# ---------------------------------------------------------------------------

class TestStressTest:
    def test_returns_stress_result(self):
        model = _fitted_model()
        result = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        assert isinstance(result, StressResult)

    def test_crisis_variance_exceeds_normal(self):
        model = _fitted_model()
        result = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        assert result.crisis.portfolio_variance > result.normal.portfolio_variance

    def test_variance_increase_pct_positive(self):
        model = _fitted_model()
        result = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        assert result.variance_increase_pct > 0.0

    def test_vol_increase_pct_positive(self):
        model = _fitted_model()
        result = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        assert result.vol_increase_pct > 0.0

    def test_2018_more_severe_than_2022(self):
        model = _fitted_model()
        r2022 = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5}, crisis_scenario="crisis_2022")
        r2018 = model.stress_test({"BTCUSDT": 0.5, "ETHUSDT": 0.5}, crisis_scenario="crisis_2018")
        assert r2018.variance_increase_pct > r2022.variance_increase_pct


# ---------------------------------------------------------------------------
# top_risk_contributors()
# ---------------------------------------------------------------------------

class TestTopRiskContributors:
    def test_returns_list_of_tuples(self):
        model = _fitted_model()
        result = model.top_risk_contributors({"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2})
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_respects_n_limit(self):
        model = _fitted_model()
        result = model.top_risk_contributors({"BTCUSDT": 0.5, "ETHUSDT": 0.5}, n=1)
        assert len(result) == 1

    def test_n_larger_than_pairs_returns_all(self):
        model = _fitted_model()
        result = model.top_risk_contributors({"BTCUSDT": 0.5, "ETHUSDT": 0.5}, n=10)
        assert len(result) <= 2

    def test_pair_names_in_results(self):
        model = _fitted_model()
        result = model.top_risk_contributors({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        pairs = {r[0] for r in result}
        assert "BTCUSDT" in pairs or "ETHUSDT" in pairs


# ---------------------------------------------------------------------------
# compute_factor_attractiveness()
# ---------------------------------------------------------------------------

class TestComputeFactorAttractiveness:
    def _returns_df(self, n=100, pairs=None):
        pairs = pairs or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        data = {p: RNG.normal(0.001, 0.02, n) for p in pairs}
        return pd.DataFrame(data)

    def test_returns_factor_attractiveness_instance(self):
        df = self._returns_df(100)
        result = compute_factor_attractiveness(df)
        assert isinstance(result, FactorAttractiveness)

    def test_insufficient_data_returns_zeros(self):
        df = self._returns_df(10)  # less than lookback=60
        result = compute_factor_attractiveness(df, lookback=60)
        assert result.composite == 0.0
        assert result.momentum_score == 0.0
        assert result.volatility_score == 0.0
        assert result.size_score == 0.0

    def test_composite_within_bounds(self):
        df = self._returns_df(200)
        result = compute_factor_attractiveness(df)
        assert -1.0 <= result.composite <= 1.0

    def test_momentum_score_within_bounds(self):
        df = self._returns_df(200)
        result = compute_factor_attractiveness(df)
        assert -1.0 <= result.momentum_score <= 1.0

    def test_volatility_score_within_bounds(self):
        df = self._returns_df(200)
        result = compute_factor_attractiveness(df)
        assert -1.0 <= result.volatility_score <= 1.0

    def test_expanding_vol_positive_score(self):
        # Second half has much higher volatility → volatility_score should be positive
        n = 120
        pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        half = n // 2
        data = {
            p: np.concatenate([
                RNG.normal(0, 0.005, half),   # calm first half
                RNG.normal(0, 0.05, half),    # volatile second half
            ])
            for p in pairs
        }
        df = pd.DataFrame(data)
        result = compute_factor_attractiveness(df, lookback=n)
        assert result.volatility_score > 0.0

    def test_tilt_dict_has_factor_keys(self):
        df = self._returns_df(200)
        result = compute_factor_attractiveness(df)
        tilt = result.tilt()
        for key in ["btc_beta", "momentum_1m", "momentum_3m", "volatility", "volume", "size"]:
            assert key in tilt

    def test_tilt_values_near_one_for_zero_scores(self):
        result = FactorAttractiveness(
            momentum_score=0.0,
            volatility_score=0.0,
            size_score=0.0,
            composite=0.0,
        )
        tilt = result.tilt()
        for key, val in tilt.items():
            assert pytest.approx(val, abs=0.01) == 1.0

    def test_size_score_btc_leading_positive(self):
        n = 120
        # BTC outperforms alts strongly
        data = {
            "BTCUSDT": np.full(n, 0.015),   # large-cap: +1.5%/bar
            "SOLUSDT": np.full(n, -0.005),  # small-cap: -0.5%/bar
            "DOTUSDT": np.full(n, -0.005),
        }
        df = pd.DataFrame(data)
        result = compute_factor_attractiveness(df, lookback=n)
        assert result.size_score > 0.0
