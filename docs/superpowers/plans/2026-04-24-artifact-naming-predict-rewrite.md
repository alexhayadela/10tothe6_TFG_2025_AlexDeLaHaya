# Artifact Naming & predict.py Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change artifact filenames to encode all 4 training dimensions explicitly and rewrite `predict.py` to mirror `train.py`'s CLI with correct tree/RNN dispatch and per-ft_type row fetching.

**Architecture:** `BaseTrainer.run()` in `models/base.py` gets a one-line path change. `models/predict.py` is fully replaced: CLI args mirror `train.py`, artifact is loaded to read `ft_type`, row count is looked up from constants, and inference branches on `NEURAL_MODELS` set. No backwards compat with old artifact names.

**Tech Stack:** Python, joblib, PyTorch (for RNN inference), sklearn StandardScaler, argparse

---

## File Map

| File | Change |
|---|---|
| `models/base.py` | Line 282-283: replace suffix+path logic with new explicit naming |
| `models/predict.py` | Full rewrite |
| `.gitignore` | Update whitelisted artifact from `rf_h1_full.pkl` to `rf_h1_sliding_discrete.pkl` |

---

### Task 1: Update artifact naming in `models/base.py`

**Files:**
- Modify: `models/base.py:282-284`

- [ ] **Step 1: Replace the suffix + path lines**

In `models/base.py`, find lines 282-284:
```python
        suffix   = "_cont" if self.target_type == "continuous" else ""
        out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}{suffix}.pkl"
        joblib.dump(artifact, out_path)
```

Replace with:
```python
        out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}_{self.mode}_{self.target_type}.pkl"
        joblib.dump(artifact, out_path)
```

- [ ] **Step 2: Verify the change looks correct**

Run:
```bash
python -c "
from models.base import BaseTrainer
import inspect, textwrap
src = inspect.getsource(BaseTrainer.run)
# find the out_path line
for line in src.splitlines():
    if 'out_path' in line or 'joblib' in line:
        print(line)
"
```
Expected output (two lines):
```
        out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}_{self.mode}_{self.target_type}.pkl"
        joblib.dump(artifact, out_path)
```

- [ ] **Step 3: Commit**

```bash
git add models/base.py
git commit -m "feat: encode mode and target_type explicitly in artifact filename"
```

---

### Task 2: Update `.gitignore` whitelist

**Files:**
- Modify: `.gitignore:25`

- [ ] **Step 1: Replace the whitelisted artifact name**

In `.gitignore`, find line 25:
```
!artifacts/rf_h1_full.pkl
```

Replace with:
```
!artifacts/rf_h1_sliding_discrete.pkl
```

- [ ] **Step 2: Verify**

```bash
git check-ignore -v artifacts/rf_h1_sliding_discrete.pkl
```
Expected: no output (file is NOT ignored — the whitelist rule applies).

```bash
git check-ignore -v artifacts/rf_h1_full.pkl
```
Expected:
```
.gitignore:23:artifacts/*	artifacts/rf_h1_full.pkl
```
(old name is now ignored)

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore artifact whitelist to new naming convention"
```

---

### Task 3: Rewrite `models/predict.py`

**Files:**
- Modify: `models/predict.py` (full rewrite)

- [ ] **Step 1: Write the new predict.py**

Replace the entire contents of `models/predict.py` with:

```python
"""
Inference entry point for all trained models.

Loads a saved artifact by (model, horizon, mode, target_type), fetches the
right number of OHLCV rows for the artifact's ft_type, and returns one
prediction per ticker for the most recent available date.

Usage:
    python -m models.predict --model rf
    python -m models.predict --model xgb --horizon 1 --mode expanding --target-type discrete
    python -m models.predict --model lstm --horizon 1 --mode sliding --target-type discrete

ft_type is read from the artifact — no need to pass it manually.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch

from config import load_env, ARTIFACTS_PATH
from db.supabase.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready
from models.neural.lstm import (
    add_cyclic_dow,
    build_sequences,
    SequenceDataset,
    _scale,
    StockRNN,
    SEQ_LEN,
    BATCH_SIZE,
)
from models.neural.cnn_rnn import StockCNNRNN
from torch.utils.data import DataLoader


# -- row-fetch constants (one per ft_type, tunable independently) -------------

ROWS_MICRO = 260   # 252-day momentum is the longest rolling window
ROWS_CROSS = 260   # breadth adds only a 10-day rolling, same bottleneck
ROWS_MACRO = 260   # VIX 250-day percentile, but micro bottleneck dominates

_ROWS = {"micro": ROWS_MICRO, "cross": ROWS_CROSS, "macro": ROWS_MACRO}

# -- model sets ---------------------------------------------------------------

NEURAL_MODELS = {"gru", "lstm", "cnn_gru", "cnn_lstm"}


# -- artifact loading ---------------------------------------------------------

def load_artifact(model: str, horizon: int, mode: str, target_type: str) -> dict:
    name = f"{model}_h{horizon}_{mode}_{target_type}.pkl"
    path = ARTIFACTS_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


# -- tree inference -----------------------------------------------------------

def _predict_tree(artifact: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, probas) for tree/markov/meta models."""
    model = artifact["model"]
    feature_cols = artifact["features"]
    X_aligned = X[feature_cols]
    preds = model.predict(X_aligned)
    probas_2d = model.predict_proba(X_aligned)
    pred_proba = probas_2d[np.arange(len(preds)), preds]
    return preds, pred_proba


# -- RNN inference ------------------------------------------------------------

def _reconstruct_rnn(artifact: dict) -> torch.nn.Module:
    """Reconstruct StockRNN or StockCNNRNN from saved config + state dict."""
    cfg = artifact["model_config"]
    model_key = artifact["model_key"]
    if model_key.startswith("cnn_"):
        model = StockCNNRNN(
            input_size=cfg["input_size"],
            num_filters=cfg["num_filters"],
            kernel_size=cfg["kernel_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            cell=cfg["cell"],
        )
    else:
        model = StockRNN(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            cell=cfg["cell"],
        )
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model


def _predict_rnn(artifact: dict, X: pd.DataFrame,
                 tickers: pd.Series, dates: pd.Series,
                 target_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, probas, last_dates) for RNN models.

    Builds sequences per ticker, applies the stored scaler, runs the model.
    last_dates is returned so the caller can align predictions back to tickers.
    """
    feature_cols = artifact["features"]
    seq_len      = artifact["seq_len"]
    scaler       = artifact["scaler"]

    X_enc = add_cyclic_dow(X[feature_cols])

    # build_sequences returns (seqs, labels, last_dates) — labels are dummy zeros
    dummy_y = pd.Series(np.zeros(len(X_enc)), index=X_enc.index)
    seqs, _, last_dates = build_sequences(X_enc, dummy_y, tickers, dates, seq_len)

    if len(seqs) == 0:
        return np.array([]), np.array([]), np.array([])

    # Apply stored scaler (fit on training data)
    n, T, f = seqs.shape
    seqs_scaled = scaler.transform(seqs.reshape(-1, f)).reshape(n, T, f).astype(np.float32)

    loader = DataLoader(
        SequenceDataset(seqs_scaled, np.zeros(n, dtype=np.float32)),
        batch_size=BATCH_SIZE,
    )

    model = _reconstruct_rnn(artifact)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    outputs = []
    with torch.no_grad():
        for X_b, _ in loader:
            outputs.append(model(X_b.to(device)).cpu())
    outputs = torch.cat(outputs)

    if target_type == "continuous":
        raw = outputs.numpy()
        return raw, raw, last_dates

    probas = torch.sigmoid(outputs).numpy()
    preds  = (probas >= 0.5).astype(int)
    return preds, probas, last_dates


# -- main inference pipeline --------------------------------------------------

def get_predictions(model: str, horizon: int = 1,
                    mode: str = "sliding",
                    target_type: str = "discrete") -> pd.DataFrame:
    """Run full inference pipeline; return one prediction per ticker.

    Loads artifact, fetches the right number of OHLCV rows for the stored
    ft_type, builds features, runs inference, and keeps only the most recent
    prediction per ticker.

    Returns DataFrame with columns: ticker, date, pred, proba, model.
    """
    artifact = load_artifact(model, horizon, mode, target_type)
    ft_type  = artifact["ft_type"]
    rows     = _ROWS[ft_type]

    tickers      = get_ibex_tickers()
    macro_tickers = get_macro_tickers()

    df_micro = fetch_ohlcv(tickers, rows)
    df_macro = fetch_ohlcv(macro_tickers, rows) if ft_type == "macro" else None

    df, X, _, mask, _ = ml_ready(horizon, df_micro, df_macro=df_macro, ft_type=ft_type)

    df_meta  = df.loc[mask, ["ticker", "date"]].copy()
    tkr_col  = df.loc[mask, "ticker"]
    date_col = df.loc[mask, "date"]

    model_stem = f"{model}_h{horizon}_{mode}_{target_type}"

    if model in NEURAL_MODELS:
        preds, probas, last_dates = _predict_rnn(
            artifact, X, tkr_col, date_col, target_type
        )
        if len(preds) == 0:
            return pd.DataFrame(columns=["ticker", "date", "pred", "proba", "model"])

        df_pred = pd.DataFrame({
            "date":  last_dates,
            "pred":  preds,
            "proba": probas,
        })
        # join ticker back via last_date alignment
        date_to_tickers = (
            df_meta.groupby("date")["ticker"].apply(list).to_dict()
        )
        rows_list = []
        seen = {}
        for _, row in df_pred.iterrows():
            d = row["date"]
            if d not in date_to_tickers:
                continue
            for t in date_to_tickers[d]:
                if t not in seen or seen[t] < d:
                    seen[t] = d
                    rows_list.append({"ticker": t, "date": d,
                                      "pred": row["pred"], "proba": row["proba"]})
        df_out = pd.DataFrame(rows_list)
    else:
        preds, probas = _predict_tree(artifact, X)
        df_meta = df_meta.copy()
        df_meta["pred"]  = preds
        df_meta["proba"] = probas
        df_out = df_meta

    df_out = df_out.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)
    df_out["model"] = model_stem
    return df_out


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["rf", "xgb", "gru", "lstm", "cnn_gru", "cnn_lstm", "markov", "meta"],
        help="Model to run inference with",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in trading days (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["sliding", "expanding"],
        default="sliding",
        help="Training mode used when the model was trained (default: sliding)",
    )
    parser.add_argument(
        "--target-type",
        dest="target_type",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Target type used when the model was trained (default: discrete)",
    )
    args = parser.parse_args()

    load_env()

    df = get_predictions(
        model=args.model,
        horizon=args.horizon,
        mode=args.mode,
        target_type=args.target_type,
    )
    print(df.to_string(index=False))
```

- [ ] **Step 2: Smoke-test imports**

```bash
python -c "import models.predict; print('imports ok')"
```
Expected:
```
imports ok
```

- [ ] **Step 3: Verify CLI help renders correctly**

```bash
python -m models.predict --help
```
Expected: usage block listing `--model`, `--horizon`, `--mode`, `--target-type` with their choices.

- [ ] **Step 4: Commit**

```bash
git add models/predict.py
git commit -m "feat: rewrite predict.py with explicit args, tree/RNN dispatch, per-ft_type row fetch"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Artifact naming: 4-dimension explicit filename — Task 1
- [x] No backwards compat — no old names referenced anywhere
- [x] `.gitignore` whitelist updated — Task 2
- [x] `predict.py` CLI mirrors `train.py` — Task 3
- [x] `ROWS_MICRO/CROSS/MACRO` separate constants — Task 3
- [x] `ft_type` read from artifact — Task 3
- [x] Tree dispatch path — Task 3 `_predict_tree`
- [x] RNN dispatch path with model reconstruction — Task 3 `_predict_rnn` + `_reconstruct_rnn`
- [x] `ml_ready()` 5-value unpack — Task 3 (`df, X, _, mask, _`)
- [x] discrete: sigmoid threshold for pred, prob for proba — Task 3
- [x] continuous: raw output for both pred and proba — Task 3
- [x] `model` column = full artifact stem — Task 3

**Placeholder scan:** None found.

**Type consistency:**
- `load_artifact` returns `dict` — used in `get_predictions`, `_predict_tree`, `_predict_rnn`, `_reconstruct_rnn` — consistent.
- `_predict_tree` returns `(np.ndarray, np.ndarray)` — consumed in Task 3 as `preds, probas` — consistent.
- `_predict_rnn` returns `(np.ndarray, np.ndarray, np.ndarray)` — consumed as `preds, probas, last_dates` — consistent.
- `StockRNN` constructor: `input_size, hidden_size, num_layers, dropout, cell` — matches `lstm.py:146` — consistent.
- `StockCNNRNN` constructor: `input_size, num_filters, kernel_size, hidden_size, num_layers, dropout, cell` — matches `cnn_rnn.py:59` — consistent.
- `model_config` keys from `rnn_trainer.py:248-254` — `input_size, hidden_size, num_layers, dropout, cell` for RNN; `+ num_filters, kernel_size` for CNN — all accessed correctly in `_reconstruct_rnn`.
