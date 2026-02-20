import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report
)

from models.trees.features import safe_build_features
from db.ohlcv.queries import load_ohlcv
from db.utils_ohlcv import get_ibex_tickers
from models.utils import get_artifacts_path

tickers = get_ibex_tickers()
df_micro = load_ohlcv(tickers)

df_micro = df_micro[df_micro["volume"] > 0]

horizon = 7
df = safe_build_features(df_micro, horizon)
df = df.sort_values("date").reset_index(drop=True)


remove_cols = [
    "ticker", "date", "open", "high", "low",
    "close", "volume", "target", "future_log_ret"
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
importances = (pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))
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
    "mean_predicted_prob":mpc,
    "class_distribution":cd,
    "params": model.get_params(),
    "random_state": 42,
    "top_features": importances.head(10).to_dict(),
}

"""
joblib.dump(
    {
        "model": model,
        "metadata": metadata,
    },
    "rf_ibex_h1_validated.pkl"
)
"""


model.fit(X, y)

joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "params": model.get_params(),
        "validation_metadata": metadata,
    },
    get_artifacts_path() / f"rf_h{horizon}_full.pkl"
)

print("\nFull model retrained and saved.")

 