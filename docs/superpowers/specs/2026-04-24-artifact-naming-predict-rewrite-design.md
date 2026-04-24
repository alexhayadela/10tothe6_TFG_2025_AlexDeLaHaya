---
title: Artifact Naming & predict.py Rewrite
date: 2026-04-24
status: approved
---

## Overview

Two related changes:
1. Rename artifact files to encode all training dimensions explicitly.
2. Rewrite `predict.py` to mirror `train.py`'s CLI, dispatch correctly for tree vs RNN models, and fetch the right number of rows per feature type.

No backwards compatibility with old artifact names.

---

## 1. Artifact Naming

### Current

`{model_key}_h{horizon}.pkl` or `{model_key}_h{horizon}_cont.pkl`

### New

`{model_key}_h{horizon}_{mode}_{target_type}.pkl`

**Examples:**
- `rf_h1_sliding_discrete.pkl`
- `xgb_h1_expanding_continuous.pkl`
- `lstm_h5_sliding_discrete.pkl`
- `cnn_gru_h1_sliding_discrete.pkl`

### Change in `models/base.py`

`BaseTrainer.run()` builds the output path. Replace:
```python
suffix = "_cont" if self.target_type == "continuous" else ""
out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}{suffix}.pkl"
```
With:
```python
out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}_{self.mode}_{self.target_type}.pkl"
```

### `.gitignore`

Update the whitelisted artifact entry from `rf_h1_full.pkl` to whichever single model is the production default (to be decided by user — update the whitelist entry accordingly).

---

## 2. `predict.py` Rewrite

### CLI Interface

Mirrors `train.py` exactly:

```
python -m models.predict --model rf --horizon 1 --mode sliding --target-type discrete
```

| Argument | Choices | Default |
|---|---|---|
| `--model` | rf, xgb, gru, lstm, cnn_gru, cnn_lstm, markov, meta | required |
| `--horizon` | int | 1 |
| `--mode` | sliding, expanding | sliding |
| `--target-type` | discrete, continuous | discrete |

### Row Fetching Constants

```python
ROWS_MICRO = 260
ROWS_CROSS = 260
ROWS_MACRO = 260
```

`ft_type` is read from the loaded artifact dict (stored there at training time). Row count is looked up from the constants above and passed to `fetch_ohlcv(tickers, rows)`.

The micro/cross/macro bottleneck is the 252-day momentum feature in per-stock indicators; 260 gives a safe buffer. Constants are kept separate for independent tuning.

### Model Dispatch

```python
NEURAL_MODELS = {"gru", "lstm", "cnn_gru", "cnn_lstm"}
```

Branching on `artifact["model_key"] in NEURAL_MODELS`:

**Tree path** (rf, xgb, markov, meta):
- Artifact has `model` (fitted sklearn/xgb object) and `features` (ordered column list)
- `model.predict(X)` and `model.predict_proba(X)`

**RNN path** (gru, lstm, cnn_gru, cnn_lstm):
- Artifact has `model_state`, `model_config`, `scaler`, `features`, `seq_len`
- Reconstruct model class from `model_config["cell"]`; load state dict
- Apply stored `scaler` to feature matrix
- Build sequences of length `seq_len` per ticker (same windowing logic as training)
- Run `model.eval()` + `torch.no_grad()` forward pass
- Discrete: sigmoid → threshold 0.5 for `pred`, raw sigmoid probability for `proba`
- Continuous: raw output is predicted `log_ret`

### Programmatic Entry Point

```python
def get_predictions(model: str, horizon: int = 1,
                    mode: str = "sliding", target_type: str = "discrete") -> pd.DataFrame:
    ...
```

Returns DataFrame with columns: `ticker`, `date`, `pred`, `proba`, `model`.

The `model` column value is the full artifact stem, e.g. `rf_h1_sliding_discrete`.

### `ml_ready()` Unpack Fix

`ml_ready()` returns 5 values `(df, X, y, mask, y_cont)`. Current predict.py unpacks only 4. The rewrite unpacks all 5 (using `_` for unused `y` and `y_cont` at inference time).

---

## Files Changed

| File | Change |
|---|---|
| `models/base.py` | Update artifact output path in `run()` |
| `models/predict.py` | Full rewrite |
| `.gitignore` | Update whitelisted artifact filename |
