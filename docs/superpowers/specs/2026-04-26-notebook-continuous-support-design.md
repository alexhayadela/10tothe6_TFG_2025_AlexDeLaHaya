---
name: Notebook continuous support
description: Extend model_comparison.ipynb and confidence_analysis.ipynb to handle continuous (regression) artifacts alongside discrete (classification) ones in separate sections
type: spec
date: 2026-04-26
---

## Scope

Update two notebooks:
- `notebooks/model_comparison.ipynb` — add Section 2 for continuous artifacts (regression metrics)
- `notebooks/confidence_analysis.ipynb` — add Part C for continuous model confidence analysis

`notebooks/model_performance.ipynb` stays empty (reserved for backtesting).

The `--target-type` rename (previously `--continuous`) is CLI-only and does not affect notebook logic. Notebooks branch on `a["target_type"]` which was always the internal artifact field name.

---

## Artifact split pattern (shared by both notebooks)

At the artifact loading step, immediately split into two dicts:

```python
artifacts_discrete = {k: v for k, v in artifacts.items() if v["target_type"] == "discrete"}
artifacts_cont     = {k: v for k, v in artifacts.items() if v["target_type"] == "continuous"}
```

Each section operates on its own dict. If a dict is empty, a guard cell prints "no [discrete/continuous] artifacts found" and subsequent cells are wrapped in `if artifacts_X:` guards to no-op gracefully (no `raise SystemExit` — that kills the kernel).

---

## model_comparison.ipynb

### Section 1 — Discrete (existing behaviour, refactored)

All existing sections (1–10) are unchanged in structure and metrics. They now operate on `artifacts_discrete` instead of `artifacts`. Metrics set stays: `balanced_accuracy` (primary), `roc_auc`, `log_loss`, `mcc`, `accuracy`.

### Section 2 — Continuous (new)

Mirror structure of Section 1 but with regression metrics. Primary metric: `ic` (Spearman IC). Full metrics set: `ic`, `mae`, `rmse`, `r2`, `directional_accuracy`.

Sections to include (matching Section 1 where applicable):

| Subsection | Notes |
|---|---|
| 2.1 CV Metrics Summary Table | `ic_mean ± ic_std` etc., sorted by `ic_mean` desc |
| 2.2 Per-Window Distributions (boxplots) | IC boxplot per model |
| 2.3 IC over CV Windows (temporal trend) | Line plot, one line per model |
| 2.4 Head-to-Head Win Rate | Per (horizon, mode) pair, win = higher IC in a given window |
| 2.5 Metadata | Same as discrete section 7 |
| 2.6 Sliding vs Expanding comparison | Only shown when same model exists in both modes |
| 2.7 Complexity vs IC | Reuse `model_complexity()` helper from Section 1 |

**Not included in Section 2:** Feature Importances (tree regression artifacts have `feature_importances_` but it doesn't change between discrete/continuous RF — and RF doesn't support continuous anyway), Metric Correlation heatmap (5 regression metrics is small enough that correlation is not useful).

---

## confidence_analysis.ipynb

### Parts A and B — unchanged

Part A (CV window confidence proxy) and Part B (per-prediction confidence) continue to operate on `artifacts_discrete` only. `run_inference()` is unchanged — it uses `sigmoid` for neural models and `predict_proba` for trees, both correct for discrete only.

### Part C — Continuous Model Confidence (new)

**C1 — CV Window: IC vs Decisiveness**

Same structure as Part A. For each continuous artifact, iterate `cv_metrics` windows. Decisiveness proxy = `mean_abs_pred` (mean of |predicted log return| across the window — stored in the artifact's `cv_metrics` entry as produced by `evaluate_regression`). Scatter: decisiveness vs IC, one subplot per model. Correlation summary table.

> Note: `mean_abs_pred` must be present in the `evaluate_regression()` output. If it isn't, add it to `models/evaluate.py` as `float(np.mean(np.abs(y_pred)))`.

**C2 — Inference on held-out data (continuous)**

New function `run_inference_continuous(artifact, df_micro_raw, df_macro_raw)`:
- Same structure as `run_inference()` but returns `(date, ticker, actual_log_ret, pred_log_ret)`.
- For tree models (`rf`, `xgb`): call `artifact["model"].predict(X_ev)` (not `predict_proba`).
- For neural models: forward pass without sigmoid — raw output is the predicted log return. Load model with same `model_config`/`model_state` pattern as `run_inference()`, but remove the `torch.sigmoid()` call.
- Returns DataFrame with columns: `date`, `ticker`, `actual`, `pred`.

**C3 — |pred_return| as confidence filter**

Sweep threshold `delta` from 0 to the 90th percentile of `|pred_return|` in 20 steps. For each threshold:
- Filter to rows where `|pred| >= delta`
- Compute: directional accuracy, IC (Spearman), coverage (fraction of rows kept)

Plot: coverage–directional_accuracy curve and coverage–IC curve (mirrors B3/B4). One subplot per model.

**C4 — Summary metrics table**

Full held-out set per model: MAE, RMSE, IC, directional_accuracy. Printed as a styled DataFrame.

---

## What needs to change in evaluate.py

Add `mean_abs_pred` to `evaluate_regression()` return dict:

```python
"mean_abs_pred": float(np.mean(np.abs(y_pred))),
```

This is needed for C1. It is a non-breaking addition — existing callers just get an extra key.

---

## What does NOT change

- `model_performance.ipynb` — stays empty, reserved for backtesting
- Artifact format — no changes
- `evaluate_model()` — no changes
- `run_inference()` in confidence_analysis — no changes
- All model training code — no changes

---

## Backtesting assessment (separate work, not in this spec)

The backtesting decisions doc (`decisions/backtesting_decisions.md`) already defines the full framework. The pipeline has all required ingredients: `ml_ready()`, artifacts with `train_end`, and `run_inference` already implemented in `confidence_analysis.ipynb`. 

Estimated complexity: **medium**. The main missing piece is a `backtest.py` module (or `notebooks/model_performance.ipynb`) that:
1. Runs the walk-forward loop (signal generation per day using the artifact's final model)
2. Applies confidence filter
3. Computes open-to-open returns (requires open prices, currently only close is in OHLCV)
4. Deducts tiered transaction costs
5. Reports Sharpe, max drawdown, Calmar, CAGR vs IBEX35 benchmark

The open price gap is the most significant data dependency — `ingest_ohlcv.py` currently stores open/high/low/close/volume so open prices are in Supabase/SQLite already. The framework can reuse `run_inference` logic from `confidence_analysis.ipynb` directly. Estimated effort: 1-2 sessions.
