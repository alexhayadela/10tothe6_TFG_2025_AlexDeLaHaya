"""
XGBoost trainer -- binary stock direction classification.

Hyperparameter rationale: decisions/xgboost_decisions.md
Feature set rationale:    decisions/features_decisions.md
Metric rationale:         decisions/rf_decisions.md (shared across all models)

Key differences from RF:
  - Sequential boosting requires early stopping (essential, not optional).
  - Shallow trees (depth 3); complexity accumulates across rounds, not depth.
  - Temporal 80/20 inner split for early stopping within each train window.
  - Final model: best round found via early stopping, then retrained on full window.

Usage:
    python -m models.trees.xgb                         # h=1, ft_type=macro
    python -m models.trees.xgb --horizon 1 --ft-type macro

Output: artifacts/xgb_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from config import ARTIFACTS_PATH
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready
from models.evaluate import evaluate_model, print_metrics


# -- hyperparameters (see decisions/xgboost_decisions.md) ---------------------

XGB_PARAMS = {
    "n_estimators":     300,    # ceiling; actual rounds set by early stopping
    "learning_rate":    0.05,   # moderate shrinkage; balances speed and overfitting
    "max_depth":        3,      # shallow weak learners; boosting adds complexity across rounds
    "min_child_weight": 100,    # stricter than RF (50); boosting compounds leaf-level noise
    "subsample":        0.7,    # stochastic boosting; decorrelates sequential trees
    "colsample_bytree": 0.7,    # mild feature subsampling; depth-3 trees need feature quality
    "reg_lambda":       1.0,    # L2 on leaf weights; first lever if more regularization needed
    "reg_alpha":        0.0,    # L1 not needed at depth 3
    "scale_pos_weight": 1.0,    # 52/48 imbalance too mild to warrant reweighting
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",  # for early stopping monitor; model selection uses bal_acc
    "tree_method":      "hist",     # faster + mildly regularizing via feature binning
    "random_state":     42,
    "n_jobs":           -1,
}

EARLY_STOPPING_ROUNDS = 30   # patience: rounds without val improvement before stopping
VAL_FRACTION          = 0.2  # fraction of train window held out for early stopping
WINDOW_DAYS           = 750  # ~3 years of trading days per training window
STEP_DAYS             = 63   # ~quarterly steps between CV windows


# -- sliding window CV --------------------------------------------------------

def sliding_windows(dates: np.ndarray, window: int, step: int,
                    embargo: int = 1, min_test: int = 21) -> list:
    """Generate (train_dates, test_dates) pairs from a sorted date array.

    Identical to rf.py. Embargo of 1 day between last train day and first
    test day prevents leakage from overlapping rolling features at h=1.
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


def _temporal_inner_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                           val_fraction: float = VAL_FRACTION):
    """Split (X, y) into train/val using a temporal date boundary.

    Splits at a date boundary (not row boundary) so that all rows from
    the same date land in the same partition. This prevents cross-sectional
    leakage where ticker A from date t is in train and ticker B from date t
    is in validation (they share macro/breadth features).
    """
    sorted_dates = np.sort(dates.unique())
    split_idx    = int(len(sorted_dates) * (1 - val_fraction))
    inner_train_dates = sorted_dates[:split_idx]
    inner_val_dates   = sorted_dates[split_idx:]

    inner_train_mask = dates.isin(inner_train_dates)
    inner_val_mask   = dates.isin(inner_val_dates)

    return (
        X.loc[inner_train_mask], y.loc[inner_train_mask],
        X.loc[inner_val_mask],   y.loc[inner_val_mask],
    )


def _train_and_eval(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                    train_dates, test_dates) -> tuple[dict | None, int]:
    """Train one XGBoost window with early stopping; evaluate on test_dates.

    Returns (metrics_dict, best_iteration). best_iteration is tracked across
    windows to monitor whether n_estimators ceiling is ever binding.
    """
    window_mask = dates.isin(train_dates)
    test_mask   = dates.isin(test_dates)

    X_window = X.loc[window_mask]
    y_window = y.loc[window_mask]
    dates_window = dates.loc[window_mask]

    X_test  = X.loc[test_mask]
    y_test  = y.loc[test_mask]

    if len(X_window) == 0 or len(X_test) == 0:
        return None, -1

    # Temporal 80/20 inner split for early stopping (date-boundary aware)
    X_inner_train, y_inner_train, X_val, y_val = _temporal_inner_split(
        X_window, y_window, dates_window
    )

    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_inner_train, y_inner_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration  # 0-indexed round with lowest val logloss

    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    return evaluate_model(y_test.values, preds, probas), best_iter


# -- main training pipeline ---------------------------------------------------

def train_xgb(horizon: int = 1, ft_type: str = "macro") -> dict:
    """Full training pipeline for XGBoost.

    1. Load IBEX35 OHLCV (micro) and index data (macro) from SQLite.
    2. Build features with ml_ready (horizon, ft_type).
    3. Sliding window CV: for each window, split 80/20 temporally for early
       stopping, train on inner-train, evaluate on held-out test. Report
       per-window and aggregate metrics, and track best_iteration.
    4. Train final model:
       a. Find optimal n_rounds via early stopping on 80/20 split of full window.
       b. Retrain on 100% of the last WINDOW_DAYS trading days using n_rounds
          (no early stopping), so the final model uses all available training data.
    5. Save artifact to artifacts/xgb_h{horizon}.pkl.

    Returns the saved artifact dict.
    """
    print(f"\n{'='*55}")
    print(f"  XGBoost | h={horizon} | ft_type={ft_type}")
    print(f"{'='*55}\n")

    # 1. Load data
    ibex_tickers  = get_ibex_tickers()
    macro_tickers = get_macro_tickers()

    with sqlite_connection() as conn:
        df_micro_raw = fetch_ohlcv(ibex_tickers)
        df_macro_raw = fetch_ohlcv(macro_tickers)

    df_micro_raw = df_micro_raw[df_micro_raw["volume"] > 0].reset_index(drop=True)
    df_macro_raw = df_macro_raw.reset_index(drop=True)  # VIX/IBEX have no real volume

    # 2. Build features
    df_macro_arg = df_macro_raw if ft_type == "macro" else None
    df, X, y, mask = ml_ready(horizon, df_micro_raw, df_macro=df_macro_arg, ft_type=ft_type)

    dates        = df.loc[mask, "date"]
    unique_dates = np.sort(dates.unique())

    print(f"Usable rows   : {len(X)}")
    print(f"Unique dates  : {len(unique_dates)}")
    dist = y.value_counts(normalize=True)
    print(f"Class balance : down={dist.get(0, 0):.3f}  up={dist.get(1, 0):.3f}\n")

    # 3. Sliding window CV
    windows = sliding_windows(unique_dates, WINDOW_DAYS, STEP_DAYS)
    print(f"Sliding window CV: {len(windows)} windows "
          f"(train={WINDOW_DAYS}d / inner-val={int(WINDOW_DAYS*VAL_FRACTION)}d, "
          f"step={STEP_DAYS}d, embargo=1d)\n")

    all_metrics   = []
    best_iters_cv = []

    for i, (train_dates, test_dates) in enumerate(windows):
        metrics, best_iter = _train_and_eval(X, y, dates, train_dates, test_dates)
        if metrics is None:
            continue
        all_metrics.append(metrics)
        best_iters_cv.append(best_iter)
        print(
            f"  [{i+1:2d}/{len(windows)}] "
            f"test {str(test_dates[0])[:10]} -> {str(test_dates[-1])[:10]} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"auc={metrics['roc_auc']:.4f}  "
            f"mcc={metrics['mcc']:.4f}  "
            f"best_iter={best_iter+1}"
        )

    # Aggregate CV results
    cv_summary = {}
    print(f"\n{'-'*55}")
    print("CV aggregate (mean +/- std):")
    for key in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "mcc"]:
        vals = [m[key] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[key] = {"mean": float(mean), "std": float(std)}
        marker = " <- primary" if key == "balanced_accuracy" else ""
        print(f"  {key:22s}: {mean:.4f} +/- {std:.4f}{marker}")

    mean_best_iter = int(np.mean(best_iters_cv)) if best_iters_cv else 150
    print(f"\n  Early stopping — best_iter: "
          f"mean={mean_best_iter+1}  "
          f"min={min(best_iters_cv)+1}  "
          f"max={max(best_iters_cv)+1}")
    if mean_best_iter >= XGB_PARAMS["n_estimators"] - 10:
        print("  WARNING: best_iter near n_estimators ceiling — consider increasing n_estimators")

    # 4. Train final model
    print(f"\n{'-'*55}")
    print(f"Training final model on last {WINDOW_DAYS} trading days ...")

    final_dates = unique_dates[-WINDOW_DAYS:]
    final_mask  = dates.isin(final_dates)
    X_final, y_final = X.loc[final_mask], y.loc[final_mask]
    dates_final = dates.loc[final_mask]

    # Step 4a: find optimal n_rounds on 80/20 inner split of full window
    X_ft, y_ft, X_fv, y_fv = _temporal_inner_split(X_final, y_final, dates_final)
    probe_model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe_model.fit(X_ft, y_ft, eval_set=[(X_fv, y_fv)], verbose=False)
    n_rounds = probe_model.best_iteration + 1
    print(f"  Optimal rounds (via inner val): {n_rounds}")

    # Step 4b: retrain on 100% of final window with fixed n_rounds (no early stopping)
    final_params = {**XGB_PARAMS, "n_estimators": n_rounds}
    final_params.pop("eval_metric", None)   # not needed without early stopping
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_final, y_final, verbose=False)

    print(f"  Train rows  : {len(X_final)}")
    print(f"  Train start : {str(final_dates[0])[:10]}")
    print(f"  Train end   : {str(final_dates[-1])[:10]}")

    # Feature importances (gain-based)
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
        "params":           XGB_PARAMS,
        "n_rounds_final":   n_rounds,
        "horizon":          horizon,
        "ft_type":          ft_type,
        "window_days":      WINDOW_DAYS,
        "train_start":      str(final_dates[0])[:10],
        "train_end":        str(final_dates[-1])[:10],
        "cv_metrics":       all_metrics,
        "cv_summary":       cv_summary,
        "cv_best_iters":    [b + 1 for b in best_iters_cv],
        "feature_importances": importances.to_dict(),
    }

    out_path = ARTIFACTS_PATH / f"xgb_h{horizon}.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nArtifact saved -> {out_path}")

    return artifact


# -- entry point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost direction classifier")
    parser.add_argument("--horizon",  type=int, default=1,      help="Prediction horizon (days)")
    parser.add_argument("--ft-type",  type=str, default="macro", help="Feature type: micro | cross | macro")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_xgb(horizon=args.horizon, ft_type=args.ft_type)
