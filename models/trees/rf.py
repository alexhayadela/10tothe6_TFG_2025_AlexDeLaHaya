"""
Random Forest trainer — binary stock direction classification.

Hyperparameters and metric rationale: decisions/rf_decisions.md
Feature set rationale:               decisions/features_decisions.md

Usage:
    python -m models.trees.rf                        # h=1, ft_type=cross
    python -m models.trees.rf --horizon 1 --ft-type cross

Output: artifacts/rf_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from config import ARTIFACTS_PATH
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers
from models.trees.features import ml_ready
from models.evaluate import evaluate_model, print_metrics


# -- hyperparameters (see decisions/rf_decisions.md) --------------------------

RF_PARAMS = {
    "n_estimators":    500,
    "max_depth":       5,
    "max_features":    0.3,    # ~9/31 features per split; better for correlated financials
    "min_samples_leaf": 50,   # stable leaf probabilities (SE ~0.07); spans multiple dates
    "class_weight":    None,  # 52/48 imbalance too mild to warrant reweighting
    "bootstrap":       True,
    "oob_score":       True,  # free sanity check — NOT a substitute for temporal CV
    "random_state":    42,
    "n_jobs":          -1,
}

WINDOW_DAYS = 750  # ~3 years of trading days per training window
STEP_DAYS   = 63   # ~quarterly steps between CV windows


# -- sliding window CV ---------------------------------------------------------

def sliding_windows(dates: np.ndarray, window: int, step: int,
                    embargo: int = 1, min_test: int = 21) -> list:
    """Generate (train_dates, test_dates) pairs from a sorted date array.

    - Train window: `window` trading days.
    - Embargo: `embargo` days gap between last train day and first test day
      to prevent leakage from overlapping features at the boundary.
    - Test window: next `step` days after embargo.
    - Skips windows where the test set is smaller than `min_test` days.
    """
    windows = []
    i = window
    while i + embargo + min_test <= len(dates):
        train_dates = dates[i - window : i]
        test_start  = i + embargo
        test_end    = min(i + embargo + step, len(dates))
        if test_end - test_start < min_test:
            break
        test_dates = dates[test_start : test_end]
        windows.append((train_dates, test_dates))
        i += step
    return windows


def _train_and_eval(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                    train_dates, test_dates) -> dict | None:
    """Train one RF window and return metrics. Returns None if sets are empty."""
    train_mask = dates.isin(train_dates)
    test_mask  = dates.isin(test_dates)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    return evaluate_model(y_test.values, preds, probas)


# -- main training pipeline ----------------------------------------------------

def train_rf(horizon: int = 1, ft_type: str = "cross") -> dict:
    """Full training pipeline for Random Forest.

    1. Load all IBEX35 OHLCV data from SQLite.
    2. Build features with ml_ready (horizon, ft_type).
    3. Sliding window CV: evaluate over multiple 3-year windows,
       reporting per-window and aggregate metrics.
    4. Train final model on the last `WINDOW_DAYS` trading days.
    5. Save artifact to artifacts/rf_h{horizon}.pkl.

    Returns the saved artifact dict.
    """
    print(f"\n{'='*55}")
    print(f"  Random Forest | h={horizon} | ft_type={ft_type}")
    print(f"{'='*55}\n")

    # 1. Load data
    tickers = get_ibex_tickers()
    with sqlite_connection() as conn:
        df_raw = fetch_ohlcv(tickers)
    df_raw = df_raw[df_raw["volume"] > 0].reset_index(drop=True)

    # 2. Build features
    df, X, y, mask = ml_ready(horizon, df_raw, df_macro=None, ft_type=ft_type)

    # dates aligned with X, y (same index)
    dates = df.loc[mask, "date"]
    unique_dates = np.sort(dates.unique())

    print(f"Usable rows   : {len(X)}")
    print(f"Unique dates  : {len(unique_dates)}")
    dist = y.value_counts(normalize=True)
    print(f"Class balance : down={dist.get(0, 0):.3f}  up={dist.get(1, 0):.3f}\n")

    # 3. Sliding window CV
    windows = sliding_windows(unique_dates, WINDOW_DAYS, STEP_DAYS)
    print(f"Sliding window CV: {len(windows)} windows "
          f"(train={WINDOW_DAYS}d, step={STEP_DAYS}d, embargo=1d)\n")

    all_metrics = []
    for i, (train_dates, test_dates) in enumerate(windows):
        metrics = _train_and_eval(X, y, dates, train_dates, test_dates)
        if metrics is None:
            continue
        all_metrics.append(metrics)
        print(
            f"  [{i+1:2d}/{len(windows)}] "
            f"test {str(test_dates[0])[:10]} -> {str(test_dates[-1])[:10]} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"auc={metrics['roc_auc']:.4f}  "
            f"mcc={metrics['mcc']:.4f}"
        )

    # Aggregate CV results
    cv_summary = {}
    print(f"\n{'-'*55}")
    print("CV aggregate (mean ± std):")
    for key in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "mcc"]:
        vals = [m[key] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[key] = {"mean": float(mean), "std": float(std)}
        marker = " <- primary" if key == "balanced_accuracy" else ""
        print(f"  {key:22s}: {mean:.4f} ± {std:.4f}{marker}")

    # 4. Train final model on last WINDOW_DAYS trading days
    print(f"\n{'-'*55}")
    print(f"Training final model on last {WINDOW_DAYS} trading days …")

    final_dates = unique_dates[-WINDOW_DAYS:]
    final_mask  = dates.isin(final_dates)
    X_final, y_final = X.loc[final_mask], y.loc[final_mask]

    final_model = RandomForestClassifier(**RF_PARAMS)
    final_model.fit(X_final, y_final)

    print(f"  Train rows : {len(X_final)}")
    print(f"  OOB score  : {final_model.oob_score_:.4f}")
    print(f"  Train start: {str(final_dates[0])[:10]}")
    print(f"  Train end  : {str(final_dates[-1])[:10]}")

    # Feature importances
    importances = (
        pd.Series(final_model.feature_importances_, index=X_final.columns)
        .sort_values(ascending=False)
    )
    print("\nTop 10 features:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:28s}: {imp:.4f}")

    # 5. Save artifact
    artifact = {
        "model":            final_model,
        "features":         list(X_final.columns),
        "params":           RF_PARAMS,
        "horizon":          horizon,
        "ft_type":          ft_type,
        "window_days":      WINDOW_DAYS,
        "train_start":      str(final_dates[0])[:10],
        "train_end":        str(final_dates[-1])[:10],
        "cv_metrics":       all_metrics,
        "cv_summary":       cv_summary,
        "feature_importances": importances.to_dict(),
    }

    out_path = ARTIFACTS_PATH / f"rf_h{horizon}.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nArtifact saved -> {out_path}")

    return artifact


# -- entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest direction classifier")
    parser.add_argument("--horizon",  type=int, default=1,       help="Prediction horizon (days)")
    parser.add_argument("--ft-type",  type=str, default="cross", help="Feature type: micro | cross | macro")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_rf(horizon=args.horizon, ft_type=args.ft_type)
