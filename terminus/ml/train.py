"""LightGBM training pipeline for the Terminus ML screener.

Trains a binary classifier to predict whether a trade entry will be
profitable (label=1) or not (label=0) based on market conditions at
entry time. The model is exported alongside a feature spec JSON so
Stronex can load both and replicate the inference.

Usage
-----
    python -m terminus.ml.train
    # or via CLI:
    terminus ml train --calmar-threshold 1.5 --output ~/.terminus/models/

Output
------
    ~/.terminus/models/screener_lgbm_YYYYMMDD_HHMMSS.pkl  — model
    ~/.terminus/models/screener_lgbm_YYYYMMDD_HHMMSS_spec.json — feature spec
    ~/.terminus/models/latest.pkl       — symlink/copy to most recent model
    ~/.terminus/models/latest_spec.json — symlink/copy to most recent spec
    ~/.terminus/models/latest_metrics.json — out-of-sample evaluation
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("terminus.ml.train")

_MODELS_DIR = Path.home() / ".terminus" / "models"


def _chronological_split(
    X: np.ndarray, y: np.ndarray, test_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) chronologically — no shuffling to prevent leakage."""
    n = len(X)
    split = int(n * (1 - test_fraction))
    return X[:split], y[:split], X[split:], y[split:]


def train(
    store=None,
    *,
    calmar_positive: float = 1.5,
    calmar_negative: float = 0.3,
    min_trades: int = 10,
    test_fraction: float = 0.25,
    output_dir: Path | str | None = None,
    max_sims: int = 0,
    use_funding: bool = True,
    use_fng: bool = True,
) -> Path:
    """Train LightGBM classifier and export model + spec.

    Args:
        store:            ResearchStore (uses default if None)
        calmar_positive:  Calmar threshold for positive label
        calmar_negative:  Calmar threshold for negative label
        min_trades:       Minimum trades per sim to include
        test_fraction:    Held-out test set fraction (chronological)
        output_dir:       Directory to save model (default ~/.terminus/models/)
        max_sims:         Max sims to load (0 = all)
        use_funding:      Include funding rate features
        use_fng:          Include fear/greed features

    Returns:
        Path to the saved model file.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        import joblib
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install: pip install lightgbm scikit-learn joblib"
        ) from e

    from terminus.store import get_store
    from terminus.ml.dataset import build_dataset
    from terminus.ml.features import FEATURE_NAMES, FEATURE_VERSION

    if store is None:
        store = get_store()

    out_dir = Path(output_dir) if output_dir else _MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    logger.info("Building training dataset...")
    X, y, feature_names = asyncio.run(
        build_dataset(
            store,
            min_trades=min_trades,
            calmar_positive=calmar_positive,
            calmar_negative=calmar_negative,
            max_sims=max_sims,
            use_funding=use_funding,
            use_fng=use_fng,
        )
    )

    if len(X) < 50:
        raise ValueError(
            f"Insufficient training data: {len(X)} samples. "
            "Run more terminus sweeps to generate sim results first."
        )

    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features, "
                f"positive rate={y.mean():.1%}")

    # Chronological train/test split
    X_train, y_train, X_test, y_test = _chronological_split(X, y, test_fraction)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Class imbalance handling
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    # LightGBM training
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
    }

    logger.info("Training LightGBM...")
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_valid],
        callbacks=callbacks,
    )

    # Evaluate on held-out test set
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Top-decile precision (most useful for a screener)
    n_top = max(1, len(y_pred_proba) // 10)
    top_idx = np.argsort(y_pred_proba)[-n_top:]
    top_precision = y_test[top_idx].mean()

    metrics = {
        "auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "top_decile_precision": round(float(top_precision), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "best_iteration": int(model.best_iteration),
        "feature_version": FEATURE_VERSION,
        "calmar_positive": calmar_positive,
        "calmar_negative": calmar_negative,
    }

    logger.info(
        f"Model trained: AUC={auc:.3f}, precision={precision:.3f}, "
        f"top-decile precision={top_precision:.3f}, "
        f"best_iter={model.best_iteration}"
    )

    # Save model + spec
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"screener_lgbm_{timestamp}.pkl"
    spec_path = out_dir / f"screener_lgbm_{timestamp}_spec.json"
    metrics_path = out_dir / "latest_metrics.json"

    joblib.dump(model, model_path)

    spec = {
        "feature_names": feature_names,
        "feature_version": FEATURE_VERSION,
        "n_features": len(feature_names),
        "calmar_positive": calmar_positive,
        "calmar_negative": calmar_negative,
        "created_at": timestamp,
        "metrics": metrics,
    }
    spec_path.write_text(json.dumps(spec, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Update latest symlinks (copy on Windows where symlinks are restricted)
    latest_model = out_dir / "latest.pkl"
    latest_spec = out_dir / "latest_spec.json"
    shutil.copy2(model_path, latest_model)
    shutil.copy2(spec_path, latest_spec)

    logger.info(f"Model saved: {model_path}")
    logger.info(f"Latest: {latest_model}")
    logger.info(f"Metrics: {metrics}")

    return model_path


def main() -> None:
    """Entry point for `terminus ml train`."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train Terminus ML screener")
    parser.add_argument("--calmar-threshold", type=float, default=1.5,
                        help="Calmar threshold for positive label (default 1.5)")
    parser.add_argument("--calmar-negative", type=float, default=0.3,
                        help="Calmar threshold for negative label (default 0.3)")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum trades per sim (default 10)")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="Test set fraction (default 0.25)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default ~/.terminus/models/)")
    parser.add_argument("--max-sims", type=int, default=0,
                        help="Max sims to use (0=all)")
    parser.add_argument("--no-funding", action="store_true",
                        help="Disable funding rate features")
    parser.add_argument("--no-fng", action="store_true",
                        help="Disable fear/greed features")
    args = parser.parse_args()

    model_path = train(
        calmar_positive=args.calmar_threshold,
        calmar_negative=args.calmar_negative,
        min_trades=args.min_trades,
        test_fraction=args.test_fraction,
        output_dir=args.output,
        max_sims=args.max_sims,
        use_funding=not args.no_funding,
        use_fng=not args.no_fng,
    )
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
