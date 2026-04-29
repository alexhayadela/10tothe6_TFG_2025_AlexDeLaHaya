"""
Markov Chain classifier for binary stock direction prediction.

State: discretised log_ret_1 (previous-day log return) binned into n_states
       quantile buckets. For order=2, the joint state (lag-1 bin, lag-2 bin)
       is used instead, where lag-2 is proxied by log_ret_5.

Transition matrix: P(up | current_state) estimated as the empirical fraction
       of up-moves following each observed state, with additive Laplace
       smoothing (alpha=1) to handle unseen states in small windows.

Hyperparameter rationale: decisions/markov_decisions.md

Usage (standalone):
    python -m models.markov.markov                           # h=1, macro, sliding
    python -m models.markov.markov --horizon 1 --mode expanding --ft-type macro

Usage (via framework):
    python -m models.train --model markov --horizon 1 --mode sliding

Output: artifacts/markov_h{horizon}_{mode}_discrete.pkl
"""

import argparse
import numpy as np
import pandas as pd

from models.base import BaseTrainer
from models.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# MarkovChain model
# ---------------------------------------------------------------------------

class MarkovChain:
    """Discrete Markov Chain direction classifier.

    Parameters
    ----------
    n_states : int
        Number of quantile bins for return discretisation (default 3).
    order : int
        Markov order -- 1 uses only lag-1 return, 2 uses lags 1 and 2.
    alpha : float
        Laplace smoothing count added to each (state, outcome) cell (default 1.0).
    """

    def __init__(self, n_states: int = 3, order: int = 1, alpha: float = 1.0):
        self.n_states  = n_states
        self.order     = order
        self.alpha     = alpha
        self.bin_edges_: list = []      # one array per lag used, set at fit time
        self.transition_: dict = {}     # state_tuple -> P(up)
        self.default_prob_: float = 0.5

    def _lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract lag columns used as state variables.

        order=1 uses log_ret_1 (previous-day return already in X).
        order=2 additionally uses log_ret_5 as a medium-term context proxy.
        """
        if self.order == 1:
            return X[["log_ret_1"]]
        return X[["log_ret_1", "log_ret_5"]]

    def _digitise(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Map continuous values to bin indices in [0, n_states-1]."""
        bins = np.digitize(values, edges[1:-1])
        return np.clip(bins, 0, self.n_states - 1)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MarkovChain":
        """Estimate transition probabilities from training data.

        Parameters
        ----------
        X : pd.DataFrame with columns log_ret_1 (and log_ret_5 if order=2)
        y : np.ndarray of int, binary targets (0=down, 1=up)
        """
        lags = self._lag_features(X).values
        y    = np.asarray(y)

        self.bin_edges_ = []
        for col_idx in range(lags.shape[1]):
            quantiles = np.linspace(0, 100, self.n_states + 1)
            edges     = np.percentile(lags[:, col_idx], quantiles)
            edges[0]  = -np.inf
            edges[-1] = +np.inf
            self.bin_edges_.append(edges)

        digitised = np.stack(
            [self._digitise(lags[:, i], self.bin_edges_[i]) for i in range(lags.shape[1])],
            axis=1,
        )

        counts: dict = {}
        for row, label in zip(digitised, y):
            state = tuple(int(b) for b in row)
            if state not in counts:
                counts[state] = [0, 0]
            counts[state][0] += 1
            counts[state][1] += int(label)

        self.transition_ = {
            state: (v[1] + self.alpha) / (v[0] + 2 * self.alpha)
            for state, v in counts.items()
        }
        self.default_prob_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row in X. Shape: (n,)."""
        lags  = self._lag_features(X).values
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


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MARKOV_PARAMS = {
    "n_states": 3,    # quantile bins: bottom/neutral/top thirds
    "order":    1,    # lag-1 only; set to 2 to also use log_ret_5 context
    "alpha":    1.0,  # Laplace smoothing; prevents zero-probability states
}


# ---------------------------------------------------------------------------
# MarkovTrainer
# ---------------------------------------------------------------------------

class MarkovTrainer(BaseTrainer):
    """Markov Chain implementation of BaseTrainer.

    Uses only log_ret_1 (order=1) or log_ret_1 + log_ret_5 (order=2) from the
    full feature matrix.  No scaling required -- the model discretises returns
    into quantile bins at fit time.
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
        meta   = {"n_states_seen": len(model.transition_)}
        return evaluate_model(y_te.values, preds, probas), meta

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

        features = (
            ["log_ret_1", "log_ret_5"] if MARKOV_PARAMS["order"] == 2 else ["log_ret_1"]
        )
        return {
            "model":       model,
            "features":    features,
            "params":      MARKOV_PARAMS,
            "transitions": model.transition_,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper + standalone entry point
# ---------------------------------------------------------------------------

def train_markov(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return MarkovTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Markov Chain direction classifier")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_markov(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
