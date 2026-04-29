# Notebooks v2 Design

**Date:** 2026-04-29  
**Scope:** Create updated duplicates of all three notebooks (`*_v2.ipynb`) aligned to the current models API. Do not overwrite originals.

---

## Problem

Three notebooks in `notebooks/` break against the current codebase:

| Notebook | Breakage |
|---|---|
| `confidence_analysis.ipynb` | `ml_ready()` returns 5 values; cell 9 unpacks 4 → `ValueError` |
| `model_comparison.ipynb` | Accesses `artifact['model_config']` on all models; tree models lack this field → `KeyError` |
| `model_performance.ipynb` | Same `ml_ready()` 5-value unpack in `generate_signals()` |

---

## Approach

Fix all breakages and modernize all API calls throughout. Keep each notebook focused on its own concern. Produce new files (`*_v2.ipynb`) alongside originals.

---

## API Changes to Apply

### `ml_ready()` — now returns 5 values

```python
# OLD (broken)
df, X, y_discrete, mask = ml_ready(horizon, df_micro, df_macro=df_macro, ft_type=ft_type)

# NEW
df, X, y_discrete, mask, y_cont = ml_ready(horizon, df_micro, df_macro=df_macro, ft_type=ft_type)
```

Applies to: `confidence_analysis.ipynb` cell 9 and `run_inference_continuous()`, `model_performance.ipynb` `generate_signals()`.

### `model_config` guard — only neural models have this field

```python
# OLD (broken on tree models)
cfg = artifact['model_config']

# NEW
cfg = artifact.get('model_config', {})
```

`model_complexity()` branches on model type:
- Tree models (RF, XGB): use `n_estimators × max_depth` as param-count proxy
- Neural models: read `model_config` for actual architecture params

### Artifact loading — use canonical path

```python
# OLD
artifact = joblib.load(ARTIFACTS_PATH / f"{stem}.pkl")

# NEW
from models.predict import load_artifact
artifact = load_artifact(model, horizon, mode, target_type)
```

### `_predict_tree` / `_predict_rnn` — private imports confirmed OK

```python
from models.predict import _predict_tree, _predict_rnn
```

---

## Per-Notebook Changes

### `confidence_analysis_v2.ipynb`
- Fix `ml_ready()` unpack in `run_inference()` (cell 9) and `run_inference_continuous()`
- Use `load_artifact()` for artifact loading
- Verify `cv_summary` / `cv_metrics` field names match `BaseTrainer` output

### `model_comparison_v2.ipynb`
- Guard `model_config` access with `.get()`
- Fix `model_complexity()` to branch on tree vs neural
- Guard `feature_importances` access for models that may not have it
- Use `load_artifact()` for artifact loading

### `model_performance_v2.ipynb`
- Fix `ml_ready()` unpack in `generate_signals()`
- Import `_predict_tree`, `_predict_rnn` from `models.predict`
- Align artifact field names with current trainer output

---

## Error Handling

- Guard only where fields are genuinely absent on some model types
- No try/except around normal operations — let errors surface in cell output
- Setup cell at top of each notebook: verify `ARTIFACTS_PATH` exists and list available `.pkl` files

---

## Out of Scope

- Changes to any files under `models/`
- Merging notebooks into a unified file
- Implementing continuous inference in `predict.py`
