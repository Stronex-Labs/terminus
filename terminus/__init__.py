"""Terminus — ruthless backtesting lab for long-only spot strategies.

Public API:
    from terminus import (
        ResearchStore, simulate, VRule,
        build_all_configs, run_full_sweep,
        walk_forward_frozen, filter_sims,
        greedy_portfolio,
    )

ML (requires pip install terminus-lab[ml]):
    from terminus.ml import RegimeClassifier, optimize_params
"""
from __future__ import annotations

__version__ = "0.1.0"

from .store import ResearchStore, SimRecord, get_store, hash_config
from .simulate import simulate_fast as simulate
from .rules import VRule
from .indicators import precompute_all, precompute_v2, build_btc_regime_series
from .registry import (
    build_v1_configs, build_v2_configs, build_all_configs,
    build_configs_with_regime, all_exit_methods, count_configs,
)
from .walk_forward import walk_forward_frozen, walk_forward_reopt_anchored
from .filter import filter_sims, Survivor, survivor_report
from .portfolio import greedy_portfolio, reconstruct_legs, correlation
from .sweep import run_full_sweep
from . import telemetry

__all__ = [
    "__version__",
    "ResearchStore", "SimRecord", "get_store", "hash_config",
    "simulate", "VRule",
    "precompute_all", "precompute_v2", "build_btc_regime_series",
    "build_v1_configs", "build_v2_configs", "build_all_configs",
    "build_configs_with_regime", "all_exit_methods", "count_configs",
    "walk_forward_frozen", "walk_forward_reopt_anchored",
    "filter_sims", "Survivor", "survivor_report",
    "greedy_portfolio", "reconstruct_legs", "correlation",
    "run_full_sweep",
    "telemetry",
]
