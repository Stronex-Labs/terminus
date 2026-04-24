"""Terminus ML — regime classification and parameter optimization.

Requires: pip install terminus-lab[ml]
"""
from .regime import RegimeClassifier, train_regime_classifier, REGIME_BULL, REGIME_BEAR, REGIME_CHOP
from .optim import optimize_params, OptimResult

__all__ = [
    "RegimeClassifier",
    "train_regime_classifier",
    "optimize_params",
    "OptimResult",
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_CHOP",
]
