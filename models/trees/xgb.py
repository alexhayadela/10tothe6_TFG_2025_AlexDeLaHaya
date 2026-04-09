"""
XGBoost trainer -- binary stock direction classification.

Hyperparameter rationale: decisions/xgboost_decisions.md
Feature set rationale:    decisions/features_decisions.md
Metric rationale:         decisions/rf_decisions.md (shared)

Key differences from RF:
  - Sequential boosting requires early stopping (essential, not optional).
  - Shallow trees (depth 3); complexity accumulates across rounds, not depth.
  - Temporal 80/20 inner split for early stopping within each train window.
  - Final model: best round found via early stopping, then retrained on full window.

Usage (standalone):
    python -m models.trees.xgb                          # h=1, macro, sliding
    python -m models.trees.xgb --horizon 1 --mode expanding

Usage (via framework):
    python -m models.train --model xgb --horizon 1 --mode sliding

Output: artifacts/xgb_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from models.base import BaseTrainer
from models.evaluate import evaluate_model


# -- hyperparameters (see decisions/xgboost_decisions.md) ---------------------

XGB_PARAMS = {
    "n_estimators":     300,    # ceiling; actual rounds set by early stopping
    "learning_rate":    0.05,   # moderate shrinkage; balances speed and overfitting
    "max_depth":        3,      # shallow weak learners; boosting adds complexity across rounds
    "min_child_weight": 100,    # stricter than RF (50); boosting compounds leaf-level noise
    "subsample":        0.7,    # stochastic boosting; decorrelates sequential trees
    "colsample_bytree": 0.7,    # mild feature subsampling
    "reg_lambda":       1.0,    # L2 on leaf weights
    "reg_alpha":        0.0,    # L1 not needed at depth 3
    "scale_pos_weight": 1.0,    # 52/48 imbalance too mild to warrant reweighting
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
}

EARLY_STOPPING_ROUNDS = 30   # patience: rounds without val improvement
VAL_FRACTION          = 0.2  # fraction of train window held out for early stopping


# -- inner-split helper -------------------------------------------------------

def _temporal_inner_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                           val_fraction: float = VAL_FRACTION):
    """Split (X, y) at a temporal date boundary for XGBoost early stopping.

    Uses a date-boundary split (not a row split) so that all tickers from the
    same date land in the same partition. Row-based splits would put ticker A
    from date t in train and ticker B from date t in val -- leakage because they
    share macro and breadth features.
    """
    sorted_dates      = np.sort(dates.unique())
    split_idx         = int(len(sorted_dates) * (1 - val_fraction))
    inner_train_dates = sorted_dates[:split_idx]

    tr_mask  = dates.isin(inner_train_dates)
    val_mask = ~tr_mask
    return (
        X.loc[tr_mask],  y.loc[tr_mask],
        X.loc[val_mask], y.loc[val_mask],
    )


# -- trainer ------------------------------------------------------------------

class XGBTrainer(BaseTrainer):
    """XGBoost implementation of BaseTrainer.

    Two-step final model:
      1. Probe: find optimal n_rounds via early stopping on 80/20 inner split.
      2. Retrain: fit on 100% of the final window with n_estimators=n_rounds,
         no early stopping -- uses all training data for the deployed model.
    """

    @property
    def model_key(self) -> str:
        return "xgb"

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask = self.dates.isin(train_dates)
        te_mask = self.dates.isin(test_dates)

        X_window     = self.X.loc[tr_mask]
        y_window     = self.y.loc[tr_mask]
        dates_window = self.dates.loc[tr_mask]
        X_test       = self.X.loc[te_mask]
        y_test       = self.y.loc[te_mask]

        if len(X_window) == 0 or len(X_test) == 0:
            return None, -1

        X_it, y_it, X_iv, y_iv = _temporal_inner_split(X_window, y_window, dates_window)

        model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_it, y_it, eval_set=[(X_iv, y_iv)], verbose=False)
        best_iter = model.best_iteration

        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        return evaluate_model(y_test.values, preds, probas), best_iter

    def _meta_str(self, meta) -> str:
        return f"best_iter={meta + 1}" if meta is not None and meta >= 0 else ""

    def _aggregate_meta(self, all_meta, cv_summary):
        iters = [m for m in all_meta if m is not None and m >= 0]
        if not iters:
            return
        mean_iter = int(np.mean(iters))
        print(
            f"\n  Early stopping: best_iter "
            f"mean={mean_iter + 1}  min={min(iters) + 1}  max={max(iters) + 1}"
        )
        if mean_iter >= XGB_PARAMS["n_estimators"] - 10:
            print("  WARNING: best_iter near n_estimators ceiling -- consider raising n_estimators")
        cv_summary["cv_best_iters"] = [b + 1 for b in iters]

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        mask    = self.dates.isin(final_dates)
        X_f     = self.X.loc[mask]
        y_f     = self.y.loc[mask]
        dates_f = self.dates.loc[mask]

        # Step 1: probe 80/20 split to find optimal n_rounds
        X_ft, y_ft, X_fv, y_fv = _temporal_inner_split(X_f, y_f, dates_f)
        probe = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        probe.fit(X_ft, y_ft, eval_set=[(X_fv, y_fv)], verbose=False)
        n_rounds = probe.best_iteration + 1
        print(f"  Optimal rounds (via inner val): {n_rounds}")

        # Step 2: retrain on 100% of window with fixed n_rounds
        final_params = {**XGB_PARAMS, "n_estimators": n_rounds}
        final_params.pop("eval_metric", None)
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_f, y_f, verbose=False)

        print(f"  Train rows  : {len(X_f)}")

        importances = (
            pd.Series(model.feature_importances_, index=X_f.columns)
            .sort_values(ascending=False)
        )
        print("\nTop 10 features:")
        for feat, imp in importances.head(10).items():
            print(f"  {feat:28s}: {imp:.4f}")

        return {
            "model":               model,
            "features":            list(X_f.columns),
            "params":              XGB_PARAMS,
            "n_rounds_final":      n_rounds,
            "cv_best_iters":       cv_summary.get("cv_best_iters", []),
            "feature_importances": importances.to_dict(),
        }


# -- convenience wrapper + entry point ----------------------------------------

def train_xgb(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return XGBTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost direction classifier")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_xgb(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
