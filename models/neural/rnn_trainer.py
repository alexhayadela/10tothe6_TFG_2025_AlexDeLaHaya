"""
RNNTrainer / CNNRNNTrainer -- BaseTrainer subclasses for recurrent models.

RNNTrainer    : handles GRU and LSTM (StockRNN from lstm.py)
CNNRNNTrainer : extends RNNTrainer with the CNN front-end (StockCNNRNN from cnn_rnn.py)

Both trainers share the full training pipeline (sequence building, scaling,
early stopping, artifact format) and differ only in which model class is
instantiated via _build_model().

Architecture rationale:
  decisions/rnn_decisions.md     (LSTM / GRU)
  decisions/cnn_rnn_decisions.md (CNN + GRU/LSTM)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseTrainer
from models.evaluate import evaluate_model

# Sequence utilities + model class from lstm.py (no circular import: lstm.py
# only lazy-imports from this module inside train_lstm(), not at module level)
from models.neural.lstm import (
    add_cyclic_dow,
    build_sequences,
    SequenceDataset,
    _temporal_seq_split,
    _scale,
    StockRNN,
    SEQ_LEN,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    MAX_EPOCHS,
    LR,
    LR_PATIENCE,
    ES_PATIENCE,
    WEIGHT_DECAY,
    GRAD_CLIP,
)
from models.neural.cnn_rnn import StockCNNRNN, NUM_FILTERS, KERNEL_SIZE


# -- RNNTrainer ---------------------------------------------------------------

class RNNTrainer(BaseTrainer):
    """Trainer for standalone GRU or LSTM (StockRNN).

    _after_features applies cyclic-dow encoding and builds the full
    (n, T, F) sequence array once from the complete dataset.  Each CV fold
    then filters sequences by last_date without recomputing.
    """

    def __init__(self, horizon: int = 1, ft_type: str = "macro",
                 mode: str = "sliding", cell: str = "gru"):
        super().__init__(horizon, ft_type, mode)
        self.cell   = cell
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # populated by _after_features:
        self.all_seqs       = None
        self.all_labels     = None
        self.all_last_dates = None
        self.input_size     = None

    @property
    def model_key(self) -> str:
        return self.cell.lower()   # "gru" or "lstm"

    def _print_header(self):
        print(f"  cell={self.cell.upper()} | T={SEQ_LEN} | hidden={HIDDEN_SIZE} | device={self.device}")

    # -- sequence preparation (hook) ------------------------------------------

    def _after_features(self):
        """Encode cyclic dow and build all sequences from the full dataset."""
        self.X = add_cyclic_dow(self.X)
        self.input_size = self.X.shape[1]
        print(f"Input size    : {self.input_size} features (after cyclic dow)\n")
        print("Building sequences...")
        self.all_seqs, self.all_labels, self.all_last_dates = build_sequences(
            self.X, self.y, self.tickers, self.dates, SEQ_LEN
        )
        print(f"Total sequences: {len(self.all_seqs)} | shape: {self.all_seqs.shape}\n")

        n_params = sum(p.numel() for p in self._build_model().parameters())
        print(f"Model parameters: {n_params:,}\n")

    # -- model factory --------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Return a fresh StockRNN on the appropriate device."""
        return StockRNN(self.input_size, cell=self.cell).to(self.device)

    # -- training helpers -----------------------------------------------------

    def _fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> tuple:
        """Train with early stopping; return (model, best_epoch)."""
        tr_dl  = DataLoader(SequenceDataset(X_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model     = self._build_model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, verbose=False
        )

        best_val_loss = float("inf")
        best_state    = None
        best_epoch    = 0
        no_improve    = 0

        for epoch in range(MAX_EPOCHS):
            # --- train ---
            model.train()
            for X_b, y_b in tr_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            # --- validate ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    val_loss += criterion(model(X_b), y_b).item() * len(y_b)
            val_loss /= len(val_dl.dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(model.state_dict())
                best_epoch    = epoch
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= ES_PATIENCE:
                    break

        model.load_state_dict(best_state)
        return model, best_epoch + 1

    def _predict(self, model: nn.Module, loader: DataLoader) -> tuple:
        """Return (preds, probas) from a DataLoader."""
        model.eval()
        logits = []
        with torch.no_grad():
            for X_b, _ in loader:
                logits.append(model(X_b.to(self.device)).cpu())
        logits = torch.cat(logits)
        probas = torch.sigmoid(logits).numpy()
        return (probas >= 0.5).astype(int), probas

    # -- CV fold --------------------------------------------------------------

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask   = np.isin(self.all_last_dates, train_dates)
        test_mask = np.isin(self.all_last_dates, test_dates)

        seqs_tr   = self.all_seqs[tr_mask];    labs_tr   = self.all_labels[tr_mask]
        seqs_test = self.all_seqs[test_mask];   labs_test = self.all_labels[test_mask]

        if len(seqs_tr) == 0 or len(seqs_test) == 0:
            return None, -1

        X_itr, y_itr, X_ival, y_ival = _temporal_seq_split(
            seqs_tr, labs_tr, self.all_last_dates[tr_mask]
        )
        X_itr_s, X_ival_s, X_test_s, _ = _scale(X_itr, X_ival, seqs_test)

        model, best_ep = self._fit(X_itr_s, y_itr, X_ival_s, y_ival)

        test_dl = DataLoader(SequenceDataset(X_test_s, labs_test), batch_size=BATCH_SIZE)
        preds, probas = self._predict(model, test_dl)
        return evaluate_model(labs_test.astype(int), preds, probas), best_ep

    def _meta_str(self, meta) -> str:
        return f"best_ep={meta}" if meta is not None and meta > 0 else ""

    def _aggregate_meta(self, all_meta, cv_summary):
        epochs = [m for m in all_meta if m is not None and m > 0]
        if not epochs:
            return
        print(
            f"\n  Early stopping: best_epoch "
            f"mean={np.mean(epochs):.1f}  min={min(epochs)}  max={max(epochs)}"
        )
        cv_summary["cv_best_epochs"] = epochs

    # -- final model ----------------------------------------------------------

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        mask_seq   = np.isin(self.all_last_dates, final_dates)
        seqs_f     = self.all_seqs[mask_seq]
        labs_f     = self.all_labels[mask_seq]
        last_d_f   = self.all_last_dates[mask_seq]

        X_itr, y_itr, X_ival, y_ival = _temporal_seq_split(seqs_f, labs_f, last_d_f)
        X_itr_s, X_ival_s, scaler    = _scale(X_itr, X_ival)

        model, best_ep = self._fit(X_itr_s, y_itr, X_ival_s, y_ival)
        print(f"  Train seqs  : {len(seqs_f)}")
        print(f"  Best epoch  : {best_ep}")

        return {
            "model_state":  model.state_dict(),
            "model_config": self._model_config(),
            "scaler":       scaler,
            "features":     list(self.X.columns),
            "seq_len":      SEQ_LEN,
            "cv_best_epochs": cv_summary.get("cv_best_epochs", []),
        }

    def _model_config(self) -> dict:
        """Return the config dict needed to reconstruct the model at inference."""
        return {
            "input_size":  self.input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "cell":        self.cell,
        }


# -- CNNRNNTrainer ------------------------------------------------------------

class CNNRNNTrainer(RNNTrainer):
    """Trainer for CNN+GRU or CNN+LSTM (StockCNNRNN).

    Extends RNNTrainer by overriding _build_model to return StockCNNRNN and
    enriching model_config with the CNN hyperparameters.  All sequence
    building, scaling, and early-stopping logic is inherited unchanged.
    """

    @property
    def model_key(self) -> str:
        return f"cnn_{self.cell.lower()}"   # "cnn_gru" or "cnn_lstm"

    def _print_header(self):
        super()._print_header()
        print(f"  filters={NUM_FILTERS} | kernel={KERNEL_SIZE}")

    def _build_model(self) -> nn.Module:
        return StockCNNRNN(self.input_size, cell=self.cell).to(self.device)

    def _model_config(self) -> dict:
        config = super()._model_config()
        config.update({"num_filters": NUM_FILTERS, "kernel_size": KERNEL_SIZE})
        return config
