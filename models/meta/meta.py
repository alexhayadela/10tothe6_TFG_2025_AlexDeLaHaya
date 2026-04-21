"""
MetaTrainer -- ensemble de segundo nivel (stacking) sobre modelos base discretos.

Protocolo: out-of-fold stacking para evitar data leakage.
Meta-modelo: LogisticRegression(C=1.0) sobre probabilidades P(up) de los modelos base.
Modelos base: rf, xgb, gru, lstm, markov (configurables via BASE_MODELS).

Véase decisions/meta_model_decisions.md para la justificación completa.

Usage (standalone):
    python -m models.meta.meta                           # h=1, macro, sliding
    python -m models.meta.meta --horizon 1 --mode sliding --ft-type macro

Usage (via framework):
    python -m models.train --model meta --horizon 1

Output: artifacts/meta_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from models.base import BaseTrainer
from models.evaluate import evaluate_model
from models.neural.lstm import (
    SequenceDataset, _temporal_seq_split, _scale, SEQ_LEN, BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fixed order matters: determines column order in the meta-feature matrix.
BASE_MODELS = ["rf", "xgb", "gru", "lstm", "markov"]

META_PARAMS = {
    "C":            1.0,   # L2 regularisation; sufficient for 5 inputs
    "max_iter":     1000,
    "random_state": 42,
}

# Inner temporal split fraction used to generate meta-model training features.
META_VAL_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Helper: build per-model P(up) probabilities
# ---------------------------------------------------------------------------

def _base_probas_trees(trainer_cls, params, X_tr, y_tr, X_pr):
    """Train a tree-based model and return P(up) on prediction set."""
    m = trainer_cls(**params)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_pr)[:, 1]


def _base_probas_xgb(X_tr, y_tr, dates_tr, X_pr):
    """Train XGBoost with early stopping and return P(up) on prediction set."""
    import xgboost as xgb
    from models.trees.xgb import XGB_PARAMS, _temporal_inner_split, EARLY_STOPPING_ROUNDS

    X_it, y_it, X_iv, y_iv = _temporal_inner_split(X_tr, y_tr, dates_tr)
    m = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    m.fit(X_it, y_it, eval_set=[(X_iv, y_iv)], verbose=False)
    return m.predict_proba(X_pr)[:, 1]


def _base_probas_rnn(trainer, train_dates, pred_dates):
    """Train a GRU/LSTM on train_dates and return P(up) on pred_dates.

    Requires that trainer._after_features() has already been called so
    trainer.all_seqs / all_labels / all_last_dates are populated.
    """
    tr_seq_mask   = np.isin(trainer.all_last_dates, train_dates)
    pred_seq_mask = np.isin(trainer.all_last_dates, pred_dates)

    seqs_tr = trainer.all_seqs[tr_seq_mask]
    labs_tr = trainer.all_labels[tr_seq_mask]
    seqs_pr = trainer.all_seqs[pred_seq_mask]
    labs_pr = trainer.all_labels[pred_seq_mask]

    if len(seqs_tr) == 0 or len(seqs_pr) == 0:
        return np.full(len(seqs_pr), 0.5)

    X_it, y_it, X_iv, y_iv = _temporal_seq_split(
        seqs_tr, labs_tr, trainer.all_last_dates[tr_seq_mask]
    )
    X_it_s, X_iv_s, X_pr_s, _ = _scale(X_it, X_iv, seqs_pr)

    model, _ = trainer._fit(X_it_s, y_it, X_iv_s, y_iv)
    pr_dl = DataLoader(SequenceDataset(X_pr_s, labs_pr), batch_size=BATCH_SIZE)
    _, probas = trainer._predict(model, pr_dl)
    return probas


def _base_probas_markov(X_tr, y_tr, X_pr):
    """Train MarkovChain and return P(up) on prediction set."""
    from models.markov.markov import MarkovChain, MARKOV_PARAMS

    m = MarkovChain(**MARKOV_PARAMS)
    m.fit(X_tr, y_tr.values)
    return m.predict_proba(X_pr)


# ---------------------------------------------------------------------------
# MetaTrainer
# ---------------------------------------------------------------------------

class MetaTrainer(BaseTrainer):
    """Stacking ensemble (LogisticRegression) over discrete base models.

    _after_features builds sequences for the two neural trainers (GRU, LSTM)
    once so they can be reused across every CV fold without re-fetching data.

    _train_window uses an inner 80/20 temporal split within train_dates to
    generate out-of-fold base-model probabilities for training the meta-model,
    then evaluates on the true held-out test_dates.

    _train_final generates out-of-fold features over the final window's own
    80/20 split and fits the deployed meta-model.
    """

    def __init__(self, horizon: int = 1, ft_type: str = "macro",
                 mode: str = "sliding", target_type: str = "discrete"):
        super().__init__(horizon, ft_type, mode, target_type)
        # GRU and LSTM trainers are built once in _after_features and reused.
        self._gru_trainer  = None
        self._lstm_trainer = None

    @property
    def model_key(self) -> str:
        return "meta"

    def _print_header(self):
        print(f"  base_models={BASE_MODELS}")
        print(f"  meta={LogisticRegression.__name__}(C={META_PARAMS['C']})")

    # -- sequence preparation (hook) ------------------------------------------

    def _after_features(self):
        """Build GRU and LSTM sequence arrays once from the full dataset."""
        from models.neural.rnn_trainer import RNNTrainer
        from models.neural.lstm import add_cyclic_dow

        print("Building sequences for GRU/LSTM base models...")
        X_cyclic = add_cyclic_dow(self.X)

        for cell in ("gru", "lstm"):
            t = RNNTrainer(
                horizon=self.horizon, ft_type=self.ft_type,
                mode=self.mode, cell=cell,
            )
            t.X, t.y, t.dates, t.tickers, t.unique_dates = (
                X_cyclic, self.y, self.dates, self.tickers, self.unique_dates,
            )
            t._after_features()
            if cell == "gru":
                self._gru_trainer = t
            else:
                self._lstm_trainer = t

        print("Sequences ready.\n")

    # -- core feature builder -------------------------------------------------

    def _build_meta_features(self, train_dates, pred_dates) -> tuple:
        """Return (X_meta, y_meta) for a given train→pred split.

        X_meta : (n_pred, K)  — P(up) from each of the K base models
        y_meta : (n_pred,)    — true labels for pred_dates rows
        n_pred may be smaller than self.dates.isin(pred_dates).sum() for
        neural models (which lose SEQ_LEN rows per ticker); the minimum
        across all models determines the effective size.  Alignment is by
        pred_dates index — all models are called with the same pred_dates,
        so their outputs are date-aligned when pred_dates has no gaps.
        """
        from sklearn.ensemble import RandomForestClassifier
        from models.trees.rf import RF_PARAMS

        tr_mask   = self.dates.isin(train_dates)
        pred_mask = self.dates.isin(pred_dates)

        X_tr = self.X.loc[tr_mask]
        y_tr = self.y.loc[tr_mask]
        X_pr = self.X.loc[pred_mask]
        y_pr = self.y.loc[pred_mask]

        dates_tr = self.dates.loc[tr_mask]

        col_probas = {}

        # RF
        col_probas["rf"] = _base_probas_trees(
            RandomForestClassifier, RF_PARAMS, X_tr, y_tr, X_pr
        )

        # XGBoost
        col_probas["xgb"] = _base_probas_xgb(X_tr, y_tr, dates_tr, X_pr)

        # GRU
        col_probas["gru"] = _base_probas_rnn(self._gru_trainer, train_dates, pred_dates)

        # LSTM
        col_probas["lstm"] = _base_probas_rnn(self._lstm_trainer, train_dates, pred_dates)

        # Markov
        col_probas["markov"] = _base_probas_markov(X_tr, y_tr, X_pr)

        # Neural models may produce fewer rows (SEQ_LEN warmup per ticker).
        # Use the minimum length across all models to keep alignment safe.
        min_len = min(len(v) for v in col_probas.values())
        X_meta  = np.column_stack([col_probas[k][:min_len] for k in BASE_MODELS])
        y_meta  = y_pr.values[:min_len]

        return X_meta, y_meta

    # -- CV fold --------------------------------------------------------------

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask = self.dates.isin(train_dates)
        te_mask = self.dates.isin(test_dates)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            return None, None

        # Inner split to train the meta-model without seeing test_dates.
        sorted_tr  = np.sort(train_dates)
        split_idx  = int(len(sorted_tr) * (1 - META_VAL_FRACTION))
        inner_tr   = sorted_tr[:split_idx]
        inner_va   = sorted_tr[split_idx:]

        X_meta_va, y_meta_va = self._build_meta_features(inner_tr, inner_va)
        if len(X_meta_va) == 0:
            return None, None

        meta = LogisticRegression(**META_PARAMS)
        meta.fit(X_meta_va, y_meta_va)

        X_meta_te, y_meta_te = self._build_meta_features(train_dates, test_dates)
        if len(X_meta_te) == 0:
            return None, None

        preds  = meta.predict(X_meta_te)
        probas = meta.predict_proba(X_meta_te)[:, 1]
        return evaluate_model(y_meta_te, preds, probas), None

    # -- final model ----------------------------------------------------------

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        sorted_f   = np.sort(final_dates)
        split_idx  = int(len(sorted_f) * (1 - META_VAL_FRACTION))
        meta_tr    = sorted_f[:split_idx]
        meta_va    = sorted_f[split_idx:]

        X_meta, y_meta = self._build_meta_features(meta_tr, meta_va)

        meta = LogisticRegression(**META_PARAMS)
        meta.fit(X_meta, y_meta)

        print(f"  Train rows (meta): {len(X_meta)}")
        print("\nMeta-model coefficients (base model weights):")
        for name, coef in zip(BASE_MODELS, meta.coef_[0]):
            print(f"  {name:10s}: {coef:+.4f}")
        print(f"  intercept : {meta.intercept_[0]:+.4f}")

        return {
            "model":       meta,
            "base_models": BASE_MODELS,
            "features":    BASE_MODELS,
            "params":      META_PARAMS,
            "coef":        dict(zip(BASE_MODELS, meta.coef_[0].tolist())),
        }


# ---------------------------------------------------------------------------
# Convenience wrapper + standalone entry point
# ---------------------------------------------------------------------------

def train_meta(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return MetaTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train meta-model (stacking ensemble)")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_meta(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
