"""
Random Forest trainer -- binary stock direction classification.

Hyperparameter rationale: decisions/rf_decisions.md
Feature set rationale:    decisions/features_decisions.md
Metric rationale:         decisions/rf_decisions.md

Usage (standalone):
    python -m models.trees.rf                           # h=1, macro, sliding
    python -m models.trees.rf --horizon 1 --mode expanding --ft-type macro

Usage (via framework):
    python -m models.train --model rf --horizon 1 --mode sliding

Output: artifacts/rf_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)
=======
>>>>>>> development-cl

from models.base import BaseTrainer
from models.evaluate import evaluate_model


<<<<<<< HEAD
remove_cols = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target",
    "future_log_ret",
]

X = df.drop(columns=remove_cols)
X = X.replace([np.inf, -np.inf], np.nan)

mask = X.notna().all(axis=1)

X = X.loc[mask]
y = df.loc[mask, "target"]
dates = df.loc[mask, "date"]

print("X shape:", X.shape)
print("y shape:", y.shape)


unique_dates = np.sort(dates.unique())

tscv = TimeSeriesSplit(n_splits=5)
train_date_idx, test_date_idx = list(tscv.split(unique_dates))[-1]

train_dates = unique_dates[train_date_idx]
test_dates = unique_dates[test_date_idx]

train_mask = dates.isin(train_dates)
test_mask = dates.isin(test_dates)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


model = RandomForestClassifier(
    n_estimators=600,
    max_depth=5,
    max_features="sqrt",
    min_samples_leaf=25,
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
bal_acc = balanced_accuracy_score(y_test, preds)
roc = roc_auc_score(y_test, proba)
ll = log_loss(y_test, proba)
cd = y.value_counts(normalize=True)
mpc = proba.mean()
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("\n===== VALIDATION METRICS =====")
print("Accuracy:", acc)
print("Balanced Accuracy:", bal_acc)
print("ROC AUC:", roc)
print("Log Loss:", ll)
print("Class distribution:", cd)
print("Mean predicted probability:", mpc)
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("\nTop 10 Features:", importances.head(10))


metadata = {
    "features": list(X.columns),
    "n_features": X.shape[1],
    "train_start": str(train_dates.min()),
    "train_end": str(train_dates.max()),
    "test_start": str(test_dates.min()),
    "test_end": str(test_dates.max()),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "accuracy": acc,
    "balanced_accuracy": bal_acc,
    "roc_auc": roc,
    "log_loss": ll,
    "mean_predicted_prob": mpc,
    "class_distribution": cd,
    "params": model.get_params(),
    "random_state": 42,
    "top_features": importances.head(10).to_dict(),
=======
# -- hyperparameters (see decisions/rf_decisions.md) --------------------------

RF_PARAMS = {
    "n_estimators":     500,  # enough trees for stable OOB; diminishing returns after ~300
    "max_depth":        5,    # shallow trees; ensemble complexity via count not depth
    "max_features":     0.3,  # 30% of features per split; decorrelates trees on 41 features
    "min_samples_leaf": 50,   # prevents memorising micro-patterns; ~0.2% of 21K train rows
    "class_weight":     None, # 52/48 imbalance too mild for reweighting
    "bootstrap":        True,
    "oob_score":        True, # free generalisation estimate
    "random_state":     42,
    "n_jobs":           -1,
>>>>>>> development-cl
}


# -- trainer ------------------------------------------------------------------

class RFTrainer(BaseTrainer):
    """Random Forest implementation of BaseTrainer.

    Trees operate on the flat (n, features) feature matrix directly; no
    sequence building or scaling needed (_after_features is a no-op).
    """

    @property
    def model_key(self) -> str:
        return "rf"

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask = self.dates.isin(train_dates)
        te_mask = self.dates.isin(test_dates)
        X_tr, y_tr = self.X.loc[tr_mask], self.y.loc[tr_mask]
        X_te, y_te = self.X.loc[te_mask], self.y.loc[te_mask]

        if len(X_tr) == 0 or len(X_te) == 0:
            return None, None

        model  = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        preds  = model.predict(X_te)
        probas = model.predict_proba(X_te)[:, 1]
        return evaluate_model(y_te.values, preds, probas), None

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        mask = self.dates.isin(final_dates)
        X_f  = self.X.loc[mask]
        y_f  = self.y.loc[mask]

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_f, y_f)

        print(f"  OOB score   : {model.oob_score_:.4f}")
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
            "params":              RF_PARAMS,
            "feature_importances": importances.to_dict(),
        }


# -- convenience wrapper + entry point ----------------------------------------

<<<<<<< HEAD
joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "params": model.get_params(),
        "validation_metadata": metadata,
    },
    ARTIFACTS_PATH / f"rf_h{horizon}_full.pkl",
)

print("\nFull model retrained and saved.")
=======
def train_rf(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return RFTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest direction classifier")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_rf(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
>>>>>>> development-cl
