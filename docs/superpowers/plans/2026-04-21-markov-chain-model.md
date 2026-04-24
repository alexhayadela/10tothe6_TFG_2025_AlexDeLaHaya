# Markov Chain Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Markov Chain direction classifier (`markov`) that discretises return states from the existing 41-feature matrix, estimates transition probabilities per CV fold, and integrates cleanly into the `BaseTrainer` pipeline so it is accessible via `python -m models.train --model markov`.

**Architecture:** A `MarkovTrainer` inherits from `BaseTrainer` and implements `_train_window` / `_train_final`. At training time it bins a small set of lagged-return features into discrete states, estimates a transition-probability matrix, and predicts the next state's direction by looking up the probability of moving from the current state to an "up" outcome. No external ML libraries beyond NumPy/pandas are required; `sklearn` is used only for evaluation (already present).

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn (metrics only), joblib (artifacts, already used), existing `BaseTrainer` / `evaluate_model` / `ml_ready` pipeline.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `models/markov/__init__.py` | Empty package marker |
| Create | `models/markov/markov.py` | `MarkovChain` (model), `MarkovTrainer` (trainer), `train_markov` convenience wrapper |
| Modify | `models/train.py` | Add `"markov"` entry to `REGISTRY` and help text |

---

## Task 1: Package skeleton

**Files:**
- Create: `models/markov/__init__.py`

- [ ] **Step 1.1: Create the empty package file**

```python
# models/markov/__init__.py
```
(empty file — just needs to exist so Python treats the directory as a package)

- [ ] **Step 1.2: Commit**

```bash
git add models/markov/__init__.py
git commit -m "feat: add markov package skeleton"
```

---

## Task 2: `MarkovChain` model class (state discretisation + transition matrix)

**Files:**
- Create: `models/markov/markov.py`

### Context — what states we use

We discretise the **log return of the previous day** (`log_ret_1`, already in the feature matrix) into 3 quantile bins:
- `0` = bottom third (strong down)
- `1` = middle third (flat/neutral)
- `2` = top third (strong up)

The **target** is already binary (0=down, 1=up).  A "state" is therefore the discretised previous return; a "transition" is the probability of the target being 1 given that state.  With order-2 extension we concatenate the last two discretised returns into a joint state (9 combinations).

The model stores:
- `n_states`: number of discrete bins per lag (default 3)
- `order`: Markov order, 1 or 2 (default 1)
- `transition_matrix`: dict mapping state tuple → P(up)
- `default_prob`: fallback probability when an unseen state is encountered (prior = class mean)

Prediction:
1. Discretise the current `log_ret_1` (and `log_ret_2` / `log_ret_5` for order-2) into bins using the training-set quantile edges stored at fit time.
2. Look up P(up | current_state) from `transition_matrix`.
3. Hard predict: `1` if P(up) ≥ 0.5, else `0`.

- [ ] **Step 2.1: Write the `MarkovChain` class**

```python
"""
Markov Chain classifier for binary stock direction prediction.

State: discretised log_ret_1 (previous-day log return) binned into n_states
       quantile buckets. For order=2, the joint state (lag-1 bin, lag-2 bin)
       is used instead.

Transition matrix: P(up | current_state) estimated as the empirical fraction
       of up-moves following each observed state, with additive Laplace
       smoothing (alpha=1) to handle unseen states in small windows.
"""

import numpy as np
import pandas as pd


class MarkovChain:
    """Discrete Markov Chain direction classifier.

    Parameters
    ----------
    n_states : int
        Number of quantile bins for return discretisation (default 3).
    order : int
        Markov order — 1 uses only lag-1 return, 2 uses lags 1 and 2 (default 1).
    alpha : float
        Laplace smoothing count added to each (state, outcome) cell (default 1.0).
    """

    def __init__(self, n_states: int = 3, order: int = 1, alpha: float = 1.0):
        self.n_states  = n_states
        self.order     = order
        self.alpha     = alpha
        # set at fit time
        self.bin_edges_: list[np.ndarray] = []   # one array per lag used
        self.transition_: dict = {}               # state_tuple -> P(up)
        self.default_prob_: float = 0.5

    # ------------------------------------------------------------------
    def _lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract the lag columns used as state variables.

        For order=1 we use log_ret_1 (previous-day return already in X).
        For order=2 we additionally use log_ret_5 as a proxy for the 5-day
        momentum context (log_ret_2 is not in the feature set).
        """
        if self.order == 1:
            return X[["log_ret_1"]]
        return X[["log_ret_1", "log_ret_5"]]

    def _digitise(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Map continuous values to bin indices in [0, n_states-1]."""
        bins = np.digitize(values, edges[1:-1])   # edges[0]=-inf, edges[-1]=+inf
        return np.clip(bins, 0, self.n_states - 1)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MarkovChain":
        """Estimate transition probabilities from training data.

        Parameters
        ----------
        X : pd.DataFrame with columns including log_ret_1 (and log_ret_5 if order=2)
        y : np.ndarray of int, binary targets (0=down, 1=up)
        """
        lags = self._lag_features(X).values          # (n, order)
        y    = np.asarray(y)

        # compute quantile bin edges from training data, one set per lag
        self.bin_edges_ = []
        for col_idx in range(lags.shape[1]):
            quantiles = np.linspace(0, 100, self.n_states + 1)
            edges     = np.percentile(lags[:, col_idx], quantiles)
            edges[0]  = -np.inf
            edges[-1] = +np.inf
            self.bin_edges_.append(edges)

        # discretise all training rows
        digitised = np.stack(
            [self._digitise(lags[:, i], self.bin_edges_[i]) for i in range(lags.shape[1])],
            axis=1,
        )   # (n, order)

        # count transitions: for each state, count (n_total, n_up) with Laplace smoothing
        counts: dict = {}
        for row, label in zip(digitised, y):
            state = tuple(int(b) for b in row)
            if state not in counts:
                counts[state] = [0, 0]   # [total, up]
            counts[state][0] += 1
            counts[state][1] += int(label)

        # smoothed P(up | state) = (n_up + alpha) / (n_total + 2*alpha)
        self.transition_ = {
            state: (v[1] + self.alpha) / (v[0] + 2 * self.alpha)
            for state, v in counts.items()
        }
        self.default_prob_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row in X. Shape: (n,)."""
        lags = self._lag_features(X).values
        proba = np.empty(len(lags), dtype=float)
        for i, row in enumerate(lags):
            state = tuple(
                int(self._digitise(np.array([row[j]]), self.bin_edges_[j])[0])
                for j in range(lags.shape[1])
            )
            proba[i] = self.transition_.get(state, self.default_prob_)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard binary predictions (1=up, 0=down)."""
        return (self.predict_proba(X) >= 0.5).astype(int)
```

- [ ] **Step 2.2: Verify the class works in isolation (no framework needed)**

Open a Python REPL or a notebook cell and run:

```python
import numpy as np
import pandas as pd
from models.markov.markov import MarkovChain

rng = np.random.default_rng(0)
n   = 500
X_toy = pd.DataFrame({
    "log_ret_1": rng.normal(0, 0.01, n),
    "log_ret_5": rng.normal(0, 0.02, n),
})
y_toy = (X_toy["log_ret_1"].shift(-1).fillna(0) > 0).astype(int).values

mc = MarkovChain(n_states=3, order=1)
mc.fit(X_toy.iloc[:400], y_toy[:400])
proba = mc.predict_proba(X_toy.iloc[400:])
pred  = mc.predict(X_toy.iloc[400:])
print("proba range:", proba.min(), proba.max())
print("pred counts:", np.bincount(pred))
print("states seen:", len(mc.transition_))
# Expected: proba in (0,1), 3 states seen for order=1
```

Expected output:
```
proba range: ~0.3-0.7
pred counts: [n0 n1]  (both nonzero)
states seen: 3
```

---

## Task 3: `MarkovTrainer` — integrate with `BaseTrainer`

**Files:**
- Modify: `models/markov/markov.py` (append to the same file)

- [ ] **Step 3.1: Append `MarkovTrainer` to `models/markov/markov.py`**

```python
import argparse
from models.base import BaseTrainer
from models.evaluate import evaluate_model


# -- hyperparameters ----------------------------------------------------------

MARKOV_PARAMS = {
    "n_states": 3,   # quantile bins: 3 gives bottom/neutral/top thirds
    "order":    1,   # lag-1 only; order=2 adds log_ret_5 context
    "alpha":    1.0, # Laplace smoothing; prevents zero-probability states
}


# -- trainer ------------------------------------------------------------------

class MarkovTrainer(BaseTrainer):
    """Markov Chain implementation of BaseTrainer.

    Uses only log_ret_1 (order=1) or log_ret_1 + log_ret_5 (order=2) from the
    full feature matrix; all other features are ignored.  No scaling required
    (the model discretises returns into quantile bins at fit time).
    """

    @property
    def model_key(self) -> str:
        return "markov"

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask = self.dates.isin(train_dates)
        te_mask = self.dates.isin(test_dates)
        X_tr, y_tr = self.X.loc[tr_mask], self.y.loc[tr_mask]
        X_te, y_te = self.X.loc[te_mask], self.y.loc[te_mask]

        if len(X_tr) == 0 or len(X_te) == 0:
            return None, None

        model  = MarkovChain(**MARKOV_PARAMS)
        model.fit(X_tr, y_tr.values)
        preds  = model.predict(X_te)
        probas = model.predict_proba(X_te)
        n_states_seen = len(model.transition_)
        return evaluate_model(y_te.values, preds, probas), {"n_states_seen": n_states_seen}

    def _meta_str(self, meta) -> str:
        if meta is None:
            return ""
        return f"states={meta['n_states_seen']}"

    def _aggregate_meta(self, all_meta: list, cv_summary: dict):
        seen = [m["n_states_seen"] for m in all_meta if m is not None]
        if seen:
            print(f"  {'n_states_seen':22s}: {np.mean(seen):.1f} avg per fold")

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        mask = self.dates.isin(final_dates)
        X_f  = self.X.loc[mask]
        y_f  = self.y.loc[mask]

        model = MarkovChain(**MARKOV_PARAMS)
        model.fit(X_f, y_f.values)

        print(f"  States seen  : {len(model.transition_)}")
        print(f"  Train rows   : {len(X_f)}")
        print("\nTransition probabilities P(up | state):")
        for state, prob in sorted(model.transition_.items()):
            print(f"  state {state}: P(up)={prob:.4f}")

        return {
            "model":       model,
            "features":    ["log_ret_1", "log_ret_5"] if MARKOV_PARAMS["order"] == 2 else ["log_ret_1"],
            "params":      MARKOV_PARAMS,
            "transitions": model.transition_,
        }


# -- convenience wrapper + entry point ----------------------------------------

def train_markov(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return MarkovTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Markov Chain direction classifier")
    parser.add_argument("--horizon", type=int, default=1,        help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",  help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding", help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_markov(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
```

- [ ] **Step 3.2: Commit**

```bash
git add models/markov/markov.py
git commit -m "feat: implement MarkovChain model and MarkovTrainer"
```

---

## Task 4: Register `markov` in `models/train.py`

**Files:**
- Modify: `models/train.py`

- [ ] **Step 4.1: Add import at the top of `models/train.py`**

In `models/train.py`, after the existing neural import:

```python
from models.markov.markov      import MarkovTrainer
```

- [ ] **Step 4.2: Add entry to `REGISTRY`**

In the `REGISTRY` dict, after `"cnn_lstm"`:

```python
    "markov":   lambda **kw: MarkovTrainer(**kw),
```

- [ ] **Step 4.3: Add help text for `--model` argument**

In the `--model` help string, after the `cnn_lstm` line:

```python
"  markov    Markov Chain (transition probabilities on return states)\n"
```

- [ ] **Step 4.4: Verify the CLI shows `markov` in choices**

```bash
python -m models.train --help
```

Expected: `markov` appears in the `choices` list for `--model`.

- [ ] **Step 4.5: Commit**

```bash
git add models/train.py
git commit -m "feat: register markov model in train.py pipeline"
```

---

## Task 5: Smoke-test end-to-end

**Files:** none (verification only)

- [ ] **Step 5.1: Run a quick standalone smoke test**

```bash
python -m models.markov.markov --horizon 1 --ft-type macro --mode sliding
```

Expected output structure:
```
=======================================================
  MARKOV | h=1 | ft=macro | mode=sliding
=======================================================

Usable rows   : XXXX
Unique dates  : XXXX
Class balance : down=0.52X  up=0.47X

CV: XX windows | train=750d (fixed) | step=63d | embargo=1d

  [ 1/XX] test YYYY-MM-DD -> YYYY-MM-DD | bal_acc=0.XXXX  auc=0.XXXX  mcc=0.XXXX  states=3
  ...
-------------------------------------------------------
CV aggregate (mean +/- std):
  accuracy              : 0.XXXX +/- 0.XXXX
  balanced_accuracy     : 0.XXXX +/- 0.XXXX  <- primary
  ...
  n_states_seen         : 3.0 avg per fold

-------------------------------------------------------
Training final model (last 750d) ...
  States seen  : 3
  Train rows   : XXXX

Transition probabilities P(up | state):
  state (0,): P(up)=0.XXXX
  state (1,): P(up)=0.XXXX
  state (2,): P(up)=0.XXXX

Artifact saved -> artifacts/markov_h1.pkl
```

- [ ] **Step 5.2: Run via the unified entry point**

```bash
python -m models.train --model markov --horizon 1 --ft-type macro --mode sliding
```

Expected: identical output, artifact saved to `artifacts/markov_h1.pkl`.

- [ ] **Step 5.3: Verify the artifact loads correctly**

```python
import joblib
art = joblib.load("artifacts/markov_h1.pkl")
print(art.keys())
# Expected keys: model_key, horizon, ft_type, mode, window_days,
#                train_start, train_end, cv_metrics, cv_summary,
#                model, features, params, transitions
print(art["transitions"])
# Expected: {(0,): float, (1,): float, (2,): float}
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|-------------|------|
| Markov chain model implemented | Task 2 |
| Same structure as other models (BaseTrainer) | Task 3 |
| Integrated in train.py | Task 4 |
| Written in a `.md` plan first | This document |

### Placeholder scan

None found — all code blocks are complete and executable.

### Type consistency

- `MarkovChain.fit(X: pd.DataFrame, y: np.ndarray)` — used identically in `_train_window` and `_train_final` ✓
- `MarkovChain.predict_proba(X: pd.DataFrame) -> np.ndarray` — passed directly to `evaluate_model` ✓
- `MARKOV_PARAMS` dict referenced in `_train_window`, `_train_final`, and standalone `__main__` ✓
- `model_key = "markov"` matches artifact filename `markov_h{horizon}.pkl` ✓
- `_meta_str` / `_aggregate_meta` signatures match `BaseTrainer` hooks ✓
