# Notebooks v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create three updated notebook duplicates (`*_v2.ipynb`) that run cleanly against the current models API without touching the originals.

**Architecture:** Each v2 notebook is a full copy of the original with targeted fixes: (1) `ml_ready()` 5-value unpack, (2) `model_config` guard in `model_complexity()`, (3) `sort_values` column reference fixed in `model_comparison`. No changes to models/ code.

**Tech Stack:** Python 3.x, Jupyter notebooks (`.ipynb`), pandas, numpy, matplotlib, seaborn, PyTorch, scikit-learn, joblib.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `notebooks/confidence_analysis_v2.ipynb` | Create | Fixed `run_inference()`: 5-value `ml_ready` unpack |
| `notebooks/model_comparison_v2.ipynb` | Create | Fixed `model_complexity()`: `model_config` guard + sort fix |
| `notebooks/model_performance_v2.ipynb` | Create | Fixed `ml_ready()` 5-value unpack in cell 5 |

---

## Task 1: confidence_analysis_v2.ipynb

**Files:**
- Create: `notebooks/confidence_analysis_v2.ipynb`

The only functional changes from the original are in `run_inference()` (cell 9): change the 4-value `ml_ready` unpack to 5 values. Everything else is identical.

- [ ] **Step 1: Create the notebook**

Create `notebooks/confidence_analysis_v2.ipynb` as a copy of `confidence_analysis.ipynb` with cell 9 (`run_inference`) updated. The full corrected cell 9 source:

```python
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready

def _load_ohlcv():
    ibex_tickers  = get_ibex_tickers()
    macro_tickers = get_macro_tickers()
    with sqlite_connection() as conn:
        df_micro = fetch_ohlcv(ibex_tickers)
        df_macro = fetch_ohlcv(macro_tickers)
    df_micro = df_micro[df_micro['volume'] > 0].reset_index(drop=True)
    df_macro = df_macro.reset_index(drop=True)
    return df_micro, df_macro

def run_inference(artifact: dict, df_micro_raw, df_macro_raw) -> pd.DataFrame:
    """Run final model on data after train_end.  Returns (date, ticker, actual, proba, pred)."""
    import torch

    key        = artifact['model_key']
    horizon    = artifact['horizon']
    ft_type    = artifact['ft_type']
    train_end  = pd.Timestamp(artifact['train_end'])

    df_macro_arg = df_macro_raw if ft_type == 'macro' else None
    df, X, y, mask, y_cont = ml_ready(horizon, df_micro_raw,
                                       df_macro=df_macro_arg, ft_type=ft_type)

    dates   = df.loc[mask, 'date'].reset_index(drop=True)
    tickers = df.loc[mask, 'ticker'].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    eval_mask = pd.to_datetime(dates) > train_end
    if eval_mask.sum() == 0:
        print(f'  {key}: no held-out rows after {train_end.date()}')
        return pd.DataFrame()

    if key in ('rf', 'xgb'):
        X_ev = X.loc[eval_mask]
        probas = artifact['model'].predict_proba(X_ev)[:, 1]
        preds  = (probas >= 0.5).astype(int)
        return pd.DataFrame({
            'date':   dates.loc[eval_mask].values,
            'ticker': tickers.loc[eval_mask].values,
            'actual': y.loc[eval_mask].values,
            'proba':  probas,
            'pred':   preds,
        })

    # --- neural ---
    from models.neural.lstm import (add_cyclic_dow, build_sequences,
                                    SequenceDataset, SEQ_LEN)
    if 'dow' in X.columns:
        X = add_cyclic_dow(X.copy())

    seqs, labs, last_dates = build_sequences(X, y, tickers, dates, SEQ_LEN)
    eval_date_set = set(dates.loc[eval_mask].values)
    seq_mask = np.array([d in eval_date_set for d in last_dates])

    seqs_ev = seqs[seq_mask];  labs_ev = labs[seq_mask]
    last_ev = last_dates[seq_mask]
    if len(seqs_ev) == 0:
        print(f'  {key}: no eval sequences')
        return pd.DataFrame()

    sc  = artifact['scaler']
    n, T, f = seqs_ev.shape
    seqs_s  = sc.transform(seqs_ev.reshape(-1, f)).reshape(n, T, f)

    cfg = artifact['model_config']
    if 'num_filters' in cfg:
        from models.neural.cnn_rnn import StockCNNRNN
        model = StockCNNRNN(**cfg)
    else:
        from models.neural.lstm import StockRNN
        model = StockRNN(**cfg)
    model.load_state_dict(artifact['model_state'])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(seqs_s, dtype=torch.float32))
    probas = torch.sigmoid(logits).numpy()
    preds  = (probas >= 0.5).astype(int)

    return pd.DataFrame({
        'date':   last_ev,
        'actual': labs_ev.astype(int),
        'proba':  probas,
        'pred':   preds,
    })

# --- collect ---
from config import load_env
load_env()
df_micro_raw, df_macro_raw = _load_ohlcv()

pred_data = {}   # name -> DataFrame
for name, a in artifacts_discrete.items():
    print(f'Running inference: {name} ...')
    df_pred = run_inference(a, df_micro_raw, df_macro_raw)
    if len(df_pred) > 0:
        pred_data[name] = df_pred
        print(f'  -> {len(df_pred)} predictions, date range: '
              f'{df_pred["date"].min()} -- {df_pred["date"].max()}')

print(f'\nPredictions collected for: {list(pred_data.keys())}')
```

All other cells are copied verbatim from the original. The `run_inference_continuous` function in cell 25 already uses the correct 5-value unpack — copy it unchanged.

- [ ] **Step 2: Verify the notebook file was created**

Run: `ls notebooks/` and confirm `confidence_analysis_v2.ipynb` exists.

- [ ] **Step 3: Commit**

```bash
git add notebooks/confidence_analysis_v2.ipynb
git commit -m "feat: add confidence_analysis_v2 with ml_ready 5-value unpack fix"
```

---

## Task 2: model_comparison_v2.ipynb

**Files:**
- Create: `notebooks/model_comparison_v2.ipynb`

Two fixes in this notebook:
1. `model_complexity()` crashes with `KeyError: 'model_config'` for tree models — add guard so the function branches on `model_key` before touching `model_config`.
2. `sort_values(['horizon', 'balanced_accuracy_mean'], ...)` crashes because that column was overwritten by the `fmt()` loop — sort before formatting.

- [ ] **Step 1: Create the notebook with fixed model_complexity() and sort fix**

The corrected `model_complexity` function (cell 10, `add559a2`):

```python
def model_complexity(a: dict) -> int:
    """Rough complexity score comparable across model families."""
    key = a["model_key"]
    p   = a.get("params", {})
    if key == "rf":
        return p.get("n_estimators", 500) * (2 ** p.get("max_depth", 5) - 1)
    if key == "xgb":
        n_rounds = a.get("n_rounds_final", p.get("n_estimators", 150))
        return n_rounds * (2 ** p.get("max_depth", 3) - 1)
    if key == "markov":
        return a.get("params", {}).get("n_states", 3)
    # neural: gru, lstm, cnn_gru, cnn_lstm
    cfg  = a.get("model_config", {})
    if not cfg:
        return 0
    h, inp = cfg.get("hidden_size", 64), cfg.get("input_size", 1)
    mult = 4 if cfg.get("cell", "gru") == "lstm" else 3
    if "num_filters" in cfg:
        f, k = cfg["num_filters"], cfg["kernel_size"]
        conv = inp * f * k + f + 2 * f
        rnn  = mult * (f * h + h * h + h)
    else:
        conv = 0
        rnn  = mult * (inp * h + h * h + h)
    return conv + rnn + h + 1
```

The corrected CV Metrics Summary Table cell (`b82e4200`) — sort on the numeric column **before** overwriting it with formatted strings:

```python
def fmt(mean, std):
    return f'{mean:.4f} ± {std:.4f}'

display_cols = ['model', 'horizon', 'mode', 'ft_type', 'windows']

# Sort on numeric column before overwriting with formatted string
display_df = summary[display_cols + [f'{m}_mean' for m in METRICS] + [f'{m}_std' for m in METRICS]].copy()
display_df = display_df.sort_values(['horizon', 'balanced_accuracy_mean'], ascending=[True, False])

for m in METRICS:
    display_df[m] = display_df.apply(lambda r: fmt(r[f'{m}_mean'], r[f'{m}_std']), axis=1)

display_df[display_cols + METRICS].style \
    .set_caption('CV metrics  (mean ± std across sliding/expanding windows)') \
    .set_table_styles([{'selector': 'caption', 'props': 'font-size:13px; font-weight:bold;'}])
```

All other cells are copied verbatim from the original.

- [ ] **Step 2: Verify the notebook file was created**

Run: `ls notebooks/` and confirm `model_comparison_v2.ipynb` exists.

- [ ] **Step 3: Commit**

```bash
git add notebooks/model_comparison_v2.ipynb
git commit -m "feat: add model_comparison_v2 with model_config guard and sort fix"
```

---

## Task 3: model_performance_v2.ipynb

**Files:**
- Create: `notebooks/model_performance_v2.ipynb`

One fix: cell 5 (`cell-5`) calls `ml_ready()` with a 5-value unpack but the original already uses it correctly (`df_full, X_full, _, mask_full, _`). Check the actual cell source — if it already unpacks 5 values, copy verbatim. If it unpacks 4, apply the fix below.

The correct cell 5 source:

```python
# Build full macro feature set once; shared by all models
df_full, X_full, _, mask_full, _ = ml_ready(1, df_micro_raw, df_macro=df_macro_raw, ft_type='macro')
df_meta = df_full.loc[mask_full, ['ticker', 'date', 'open']].copy().reset_index(drop=True)
X_full  = X_full.reset_index(drop=True)

print(f'Usable rows : {len(X_full):,}')
print(f'Features    : {X_full.shape[1]}')
```

All other cells are copied verbatim from the original.

- [ ] **Step 1: Create the notebook**

Create `notebooks/model_performance_v2.ipynb` as a copy of `model_performance.ipynb` with cell 5 confirmed/fixed to use 5-value unpack as shown above.

- [ ] **Step 2: Verify the notebook file was created**

Run: `ls notebooks/` and confirm `model_performance_v2.ipynb` exists.

- [ ] **Step 3: Commit**

```bash
git add notebooks/model_performance_v2.ipynb
git commit -m "feat: add model_performance_v2 with ml_ready 5-value unpack fix"
```
