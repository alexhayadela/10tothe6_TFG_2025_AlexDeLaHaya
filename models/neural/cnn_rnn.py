"""
CNN+RNN (GRU/LSTM) trainer -- binary stock direction classification.

Architecture and hyperparameter rationale: decisions/cnn_rnn_decisions.md
Feature set rationale:                     decisions/features_decisions.md
Metric rationale:                          decisions/rf_decisions.md (shared)

Key design points vs pure LSTM/GRU:
  - 1D Conv block (Conv1d -> BatchNorm1d -> GELU) precedes the recurrent layer.
  - Conv acts as a local pattern detector across 3-day windows of all features.
  - CNN output (batch, seq_len=20, 32 channels) replaces raw (batch, 20, 41)
    as the RNN input; the GRU still processes a 20-step sequence.
  - No temporal pooling: T=20 is too short to lose resolution.
  - Dropout applied only after the GRU (not between CNN and GRU).
  - All other training settings (sliding window, early stopping, scaler,
    cyclic dow encoding) are identical to lstm.py.

Usage:
    python -m models.neural.cnn_rnn                         # h=1, cell=gru
    python -m models.neural.cnn_rnn --horizon 1 --cell lstm

Output: artifacts/cnn_gru_h{horizon}.pkl  (or cnn_lstm_h{horizon}.pkl)
"""

import argparse
import copy
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ARTIFACTS_PATH
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready
from models.evaluate import evaluate_model, print_metrics

# Reuse sequence/data utilities from lstm.py (identical pipeline)
from models.neural.lstm import (
    add_cyclic_dow,
    build_sequences,
    SequenceDataset,
    sliding_windows,
    _temporal_seq_split,
    _scale,
    SEQ_LEN,
    WINDOW_DAYS,
    STEP_DAYS,
    VAL_FRACTION,
    BATCH_SIZE,
    MAX_EPOCHS,
    LR,
    LR_PATIENCE,
    ES_PATIENCE,
    WEIGHT_DECAY,
    GRAD_CLIP,
)


# -- CNN+RNN-specific hyperparameters (see decisions/cnn_rnn_decisions.md) ----

NUM_FILTERS  = 32   # conv output channels; 0.75x GRU hidden -- mild bottleneck
KERNEL_SIZE  = 3    # 3-day local patterns; "same" padding preserves seq_len
HIDDEN_SIZE  = 64   # GRU hidden dimension; unchanged from baseline
NUM_LAYERS   = 1    # single recurrent layer
DROPOUT      = 0.3  # applied after GRU only (not between CNN and GRU)


# -- model --------------------------------------------------------------------

class StockCNNRNN(nn.Module):
    """1D-Conv feature extractor followed by GRU or LSTM classifier.

    Pipeline (per forward pass):
        1. Transpose input (batch, T, F) -> (batch, F, T) for Conv1d.
        2. Conv1d(F, num_filters, kernel) with same-padding -- detects local
           multi-feature co-occurrence patterns across 3 consecutive days.
        3. BatchNorm1d + GELU activation.
        4. Transpose back (batch, num_filters, T) -> (batch, T, num_filters).
        5. GRU/LSTM processes the CNN-transformed sequence.
        6. Dropout on the final hidden state, then linear projection to logit.

    No pooling is applied; temporal resolution is fully preserved so the
    recurrent layer receives all 20 timesteps.
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = NUM_FILTERS,
        kernel_size: int = KERNEL_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        cell: str = "gru",
    ):
        super().__init__()
        self.cell = cell

        # CNN block: (batch, F, T) -> (batch, num_filters, T)
        padding = kernel_size // 2  # "same" padding: output length == input length
        self.conv = nn.Conv1d(input_size, num_filters, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm1d(num_filters)
        self.act  = nn.GELU()

        # RNN block: input is now num_filters (not input_size)
        rnn_cls  = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(num_filters, hidden_size, num_layers, batch_first=True)

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # -- CNN block --
        x_c = x.transpose(1, 2)           # (batch, input_size, seq_len)
        x_c = self.conv(x_c)               # (batch, num_filters, seq_len)
        x_c = self.bn(x_c)
        x_c = self.act(x_c)
        x_c = x_c.transpose(1, 2)         # (batch, seq_len, num_filters)

        # -- RNN block --
        out = self.rnn(x_c)
        h_n = out[1][0] if self.cell == "lstm" else out[1]
        h_n = h_n[-1]                      # (batch, hidden_size)

        h_n = self.drop(h_n)
        return self.fc(h_n).squeeze(-1)    # (batch,)


# -- training helpers ---------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


def _eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            total_loss += criterion(model(X_b), y_b).item() * len(y_b)
    return total_loss / len(loader.dataset)


def _predict(model, loader, device):
    """Return (preds, probas) arrays from a DataLoader."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for X_b, _ in loader:
            all_logits.append(model(X_b.to(device)).cpu())
    logits = torch.cat(all_logits)
    probas = torch.sigmoid(logits).numpy()
    preds  = (probas >= 0.5).astype(int)
    return preds, probas


def _fit(X_tr, y_tr, X_val, y_val, input_size, device, cell="gru"):
    """Train StockCNNRNN with early stopping; return (model, best_epoch)."""
    tr_dl  = DataLoader(SequenceDataset(X_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model     = StockCNNRNN(input_size, cell=cell).to(device)
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
        _train_epoch(model, tr_dl, criterion, optimizer, device)
        val_loss = _eval_loss(model, val_dl, criterion, device)
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


def _train_and_eval(all_seqs, all_labels, all_last_dates,
                    train_dates, test_dates, input_size, device, cell):
    """Train one sliding-window fold; return (metrics, best_epoch)."""
    tr_mask   = np.isin(all_last_dates, train_dates)
    test_mask = np.isin(all_last_dates, test_dates)

    seqs_tr   = all_seqs[tr_mask];    labs_tr   = all_labels[tr_mask]
    seqs_test = all_seqs[test_mask];   labs_test = all_labels[test_mask]

    if len(seqs_tr) == 0 or len(seqs_test) == 0:
        return None, -1

    X_itr, y_itr, X_ival, y_ival = _temporal_seq_split(
        seqs_tr, labs_tr, all_last_dates[tr_mask]
    )
    X_itr_s, X_ival_s, X_test_s, _ = _scale(X_itr, X_ival, seqs_test)

    model, best_epoch = _fit(X_itr_s, y_itr, X_ival_s, y_ival, input_size, device, cell)

    test_dl = DataLoader(SequenceDataset(X_test_s, labs_test), batch_size=BATCH_SIZE)
    preds, probas = _predict(model, test_dl, device)
    metrics = evaluate_model(labs_test.astype(int), preds, probas)
    return metrics, best_epoch


# -- main training pipeline ---------------------------------------------------

def train_cnn_rnn(horizon: int = 1, ft_type: str = "macro", cell: str = "gru") -> dict:
    """Full training pipeline for CNN+GRU (or CNN+LSTM).

    Steps mirror lstm.py / rf.py exactly, with the CNN feature extractor
    added as a transparent preprocessing step inside the model:

    1. Load IBEX35 (micro) and index (macro) OHLCV from SQLite.
    2. Build features with ml_ready; apply cyclic dow encoding.
    3. Build all sequences (n, T=20, 41) from the full dataset once.
    4. Sliding window CV: scale -> CNN+RNN with early stopping -> evaluate.
    5. Final model: inner 80/20 temporal split of last WINDOW_DAYS, fit
       with early stopping, save artifact with scaler.

    Returns the saved artifact dict.
    """
    model_name = f"CNN+{cell.upper()}"
    print(f"\n{'='*55}")
    print(f"  {model_name} | h={horizon} | ft_type={ft_type} | T={SEQ_LEN}")
    print(f"  Filters={NUM_FILTERS}, kernel={KERNEL_SIZE}, hidden={HIDDEN_SIZE}")
    print(f"{'='*55}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Load data
    ibex_tickers  = get_ibex_tickers()
    macro_tickers = get_macro_tickers()

    with sqlite_connection() as conn:
        df_micro_raw = fetch_ohlcv(ibex_tickers)
        df_macro_raw = fetch_ohlcv(macro_tickers)

    df_micro_raw = df_micro_raw[df_micro_raw["volume"] > 0].reset_index(drop=True)
    df_macro_raw = df_macro_raw.reset_index(drop=True)

    # 2. Build features
    df_macro_arg = df_macro_raw if ft_type == "macro" else None
    df, X, y, mask = ml_ready(horizon, df_micro_raw, df_macro=df_macro_arg, ft_type=ft_type)
    X = add_cyclic_dow(X)

    tickers = df.loc[mask, "ticker"]
    dates   = df.loc[mask, "date"]

    input_size   = X.shape[1]   # 41
    unique_dates = np.sort(dates.unique())

    print(f"Usable rows   : {len(X)}")
    print(f"Unique dates  : {len(unique_dates)}")
    dist = y.value_counts(normalize=True)
    print(f"Class balance : down={dist.get(0, 0):.3f}  up={dist.get(1, 0):.3f}")
    print(f"Input size    : {input_size} features\n")

    # 3. Build all sequences once
    print("Building sequences...")
    all_seqs, all_labels, all_last_dates = build_sequences(X, y, tickers, dates, SEQ_LEN)
    print(f"Total sequences: {len(all_seqs)} | shape: {all_seqs.shape}\n")

    # Count parameters for one reference model (before training loop)
    _ref = StockCNNRNN(input_size, cell=cell)
    n_params = sum(p.numel() for p in _ref.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    del _ref

    # 4. Sliding window CV
    windows = sliding_windows(unique_dates, WINDOW_DAYS, STEP_DAYS)
    print(f"Sliding window CV: {len(windows)} windows "
          f"(train={WINDOW_DAYS}d, step={STEP_DAYS}d, embargo=1d)\n")

    all_metrics    = []
    best_epochs_cv = []

    for i, (train_dates, test_dates) in enumerate(windows):
        metrics, best_ep = _train_and_eval(
            all_seqs, all_labels, all_last_dates,
            train_dates, test_dates, input_size, device, cell
        )
        if metrics is None:
            continue
        all_metrics.append(metrics)
        best_epochs_cv.append(best_ep)
        print(
            f"  [{i+1:2d}/{len(windows)}] "
            f"test {str(test_dates[0])[:10]} -> {str(test_dates[-1])[:10]} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"auc={metrics['roc_auc']:.4f}  "
            f"mcc={metrics['mcc']:.4f}  "
            f"best_ep={best_ep}"
        )

    # Aggregate CV results
    cv_summary = {}
    print(f"\n{'-'*55}")
    print("CV aggregate (mean +/- std):")
    for key in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "mcc"]:
        vals = [m[key] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[key] = {"mean": float(mean), "std": float(std)}
        marker = " <- primary" if key == "balanced_accuracy" else ""
        print(f"  {key:22s}: {mean:.4f} +/- {std:.4f}{marker}")

    if best_epochs_cv:
        print(f"\n  Early stopping -- best_epoch: "
              f"mean={np.mean(best_epochs_cv):.1f}  "
              f"min={min(best_epochs_cv)}  max={max(best_epochs_cv)}")

    # 5. Train final model on last WINDOW_DAYS
    print(f"\n{'-'*55}")
    print(f"Training final model on last {WINDOW_DAYS} trading days ...")

    final_dates    = unique_dates[-WINDOW_DAYS:]
    final_mask_seq = np.isin(all_last_dates, final_dates)
    seqs_final     = all_seqs[final_mask_seq]
    labs_final     = all_labels[final_mask_seq]
    last_dates_fin = all_last_dates[final_mask_seq]

    X_itr, y_itr, X_ival, y_ival = _temporal_seq_split(seqs_final, labs_final, last_dates_fin)
    X_itr_s, X_ival_s, final_scaler = _scale(X_itr, X_ival)

    final_model, final_best_ep = _fit(X_itr_s, y_itr, X_ival_s, y_ival, input_size, device, cell)

    print(f"  Train seqs  : {len(seqs_final)}")
    print(f"  Best epoch  : {final_best_ep}")
    print(f"  Train start : {str(final_dates[0])[:10]}")
    print(f"  Train end   : {str(final_dates[-1])[:10]}")

    # 6. Save artifact
    artifact = {
        "model_state":  final_model.state_dict(),
        "model_config": {
            "input_size":  input_size,
            "num_filters": NUM_FILTERS,
            "kernel_size": KERNEL_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "cell":        cell,
        },
        "scaler":         final_scaler,
        "features":       list(X.columns),
        "horizon":        horizon,
        "ft_type":        ft_type,
        "seq_len":        SEQ_LEN,
        "window_days":    WINDOW_DAYS,
        "train_start":    str(final_dates[0])[:10],
        "train_end":      str(final_dates[-1])[:10],
        "cv_metrics":     all_metrics,
        "cv_summary":     cv_summary,
        "cv_best_epochs": best_epochs_cv,
    }

    model_key = f"cnn_{cell.lower()}"   # "cnn_gru" or "cnn_lstm"
    out_path  = ARTIFACTS_PATH / f"{model_key}_h{horizon}.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nArtifact saved -> {out_path}")

    return artifact


# -- entry point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+GRU/LSTM direction classifier")
    parser.add_argument("--horizon", type=int, default=1,      help="Prediction horizon (days)")
    parser.add_argument("--cell",    type=str, default="gru",  help="RNN cell type: gru | lstm")
    parser.add_argument("--ft-type", type=str, default="macro", help="Feature type: micro | cross | macro")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_cnn_rnn(horizon=args.horizon, ft_type=args.ft_type, cell=args.cell)
