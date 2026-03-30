"""
LSTM trainer -- binary stock direction classification.

Architecture and feature rationale: decisions/rnn_decisions.md
Feature set rationale:              decisions/features_decisions.md
Metric rationale:                   decisions/rf_decisions.md (shared)

Key design points vs tree models:
  - Input shape: (batch, seq_len=20, n_features=41) — sequential, not flat.
  - 'dow' is re-encoded cyclically (sin/cos) replacing the integer column.
  - All features z-scored using training-window statistics only.
  - One shared model across all 30 tickers (cross-sectional pooling).
  - Early stopping on temporal 80/20 inner-val split (patience=10 epochs).
  - GRU cell by default (fewer params than LSTM, comparable accuracy at T=20).

Usage:
    python -m models.neural.lstm                         # h=1, ft_type=macro
    python -m models.neural.lstm --horizon 1 --cell lstm --ft-type macro

Output: artifacts/lstm_h{horizon}.pkl  (or gru_h{horizon}.pkl with --cell gru)
"""

import argparse
import copy
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import ARTIFACTS_PATH
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready
from models.evaluate import evaluate_model, print_metrics


# -- hyperparameters (see decisions/rnn_decisions.md) -------------------------

SEQ_LEN       = 20     # trading-day lookback window (1 month)
HIDDEN_SIZE   = 64     # recurrent hidden state dimension (~20K total params)
NUM_LAYERS    = 1      # single recurrent layer — pre-computed features need no hierarchy
DROPOUT       = 0.3    # applied after RNN, before linear output
BATCH_SIZE    = 128
MAX_EPOCHS    = 100
LR_PATIENCE   = 5      # ReduceLROnPlateau patience (epochs)
ES_PATIENCE   = 10     # early stopping patience (epochs)
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
VAL_FRACTION  = 0.2    # temporal inner split for early stopping
WINDOW_DAYS   = 750    # ~3 years of training data per sliding window
STEP_DAYS     = 63     # ~quarterly steps between CV windows


# -- cyclic dow encoding ------------------------------------------------------

def add_cyclic_dow(X: pd.DataFrame) -> pd.DataFrame:
    """Replace integer 'dow' (0-4) with sin/cos cyclic encoding.

    Cyclic encoding preserves the circular nature (Friday ~ Monday) and
    gives the network two real-valued features that vary smoothly over the
    week. The integer 'dow' is then dropped. Feature count: 40 -> 41.
    """
    X = X.copy()
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 5)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 5)
    return X.drop(columns=["dow"])


# -- sequence construction ----------------------------------------------------

def build_sequences(X: pd.DataFrame, y: pd.Series,
                    tickers: pd.Series, dates: pd.Series,
                    seq_len: int = SEQ_LEN):
    """Slide a window of length seq_len over each ticker's feature history.

    For ticker with N rows, produces N - seq_len + 1 sequences.
    Each sequence: (seq_len, n_features) covering days [t-seq_len+1 .. t].
    The label is y[t] — the forward return target already encoded in y.

    Sequences from all tickers are pooled. The 'last_date' array tracks
    which date each sequence's prediction corresponds to, used downstream
    to assign sequences to train / val / test windows without leakage.

    Returns
    -------
    seqs       : float32 array (n, seq_len, n_features)
    labels     : float32 array (n,)
    last_dates : object array (n,) of date values (prediction date)
    """
    seqs, labs, last_dates = [], [], []

    for ticker in tickers.unique():
        mask    = tickers == ticker
        X_t     = X.loc[mask]
        y_t     = y.loc[mask]
        d_t     = dates.loc[mask]

        # Sort rows by date within this ticker
        order   = d_t.argsort()
        X_vals  = X_t.iloc[order.values].values          # (N, features)
        y_vals  = y_t.iloc[order.values].values          # (N,)
        d_vals  = d_t.iloc[order.values].values          # (N,)

        n = len(X_vals)
        for i in range(seq_len - 1, n):
            seqs.append(X_vals[i - seq_len + 1 : i + 1])  # (seq_len, features)
            labs.append(y_vals[i])
            last_dates.append(d_vals[i])

    return (
        np.array(seqs, dtype=np.float32),
        np.array(labs, dtype=np.float32),
        np.array(last_dates),
    )


# -- dataset ------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- model --------------------------------------------------------------------

class StockRNN(nn.Module):
    """Single-layer GRU or LSTM followed by dropout and a linear classifier.

    Receives sequences of shape (batch, seq_len, input_size).
    Returns raw logits (batch,) — apply sigmoid externally or use
    BCEWithLogitsLoss during training for numerical stability.
    """

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 cell: str = "lstm"):
        super().__init__()
        self.cell = cell
        rnn_cls   = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn  = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out = self.rnn(x)
        h_n = out[1][0] if self.cell == "lstm" else out[1]
        # h_n: (num_layers, batch, hidden) — take the last layer
        h_n = h_n[-1]           # (batch, hidden)
        h_n = self.drop(h_n)
        return self.fc(h_n).squeeze(-1)   # (batch,)


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
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item() * len(y_b)
    return total_loss / len(loader.dataset)


def _predict(model, loader, device):
    """Return (preds, probas) from a DataLoader."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for X_b, _ in loader:
            all_logits.append(model(X_b.to(device)).cpu())
    logits = torch.cat(all_logits)
    probas = torch.sigmoid(logits).numpy()
    preds  = (probas >= 0.5).astype(int)
    return preds, probas


def _fit(X_tr: np.ndarray, y_tr: np.ndarray,
         X_val: np.ndarray, y_val: np.ndarray,
         input_size: int, device, cell: str = "lstm") -> tuple:
    """Train StockRNN with early stopping; return (model, best_epoch).

    Fits on X_tr/y_tr, monitors val loss, restores best weights on exit.
    """
    tr_dl  = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model     = StockRNN(input_size, cell=cell).to(device)
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


# -- sliding window helpers ---------------------------------------------------

def sliding_windows(dates: np.ndarray, window: int, step: int,
                    embargo: int = 1, min_test: int = 21) -> list:
    """Identical to rf.py / xgb.py — generates (train_dates, test_dates) pairs."""
    windows = []
    i = window
    while i + embargo + min_test <= len(dates):
        train_dates = dates[i - window : i]
        test_start  = i + embargo
        test_end    = min(i + embargo + step, len(dates))
        if test_end - test_start < min_test:
            break
        test_dates = dates[test_start : test_end]
        windows.append((train_dates, test_dates))
        i += step
    return windows


def _temporal_seq_split(seqs, labels, last_dates, val_fraction=VAL_FRACTION):
    """Split sequences by last_date into inner train / val at a date boundary.

    Mirrors xgb._temporal_inner_split: all sequences whose prediction date
    falls in the first (1-val_fraction) of sorted dates go to train, the
    rest go to val.
    """
    sorted_dates  = np.sort(np.unique(last_dates))
    split_idx     = int(len(sorted_dates) * (1 - val_fraction))
    train_dates   = sorted_dates[:split_idx]

    tr_mask  = np.isin(last_dates, train_dates)
    val_mask = ~tr_mask
    return seqs[tr_mask], labels[tr_mask], seqs[val_mask], labels[val_mask]


def _scale(X_tr, X_val, X_test=None):
    """Z-score all features using training-set statistics only.

    Flattens (n, T, f) to (n*T, f) for fitting, then reshapes back.
    The scaler is returned so it can be reused for inference.
    """
    n_tr, T, f = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, f))

    X_tr_s  = scaler.transform(X_tr.reshape(-1, f)).reshape(n_tr, T, f)
    n_val   = X_val.shape[0]
    X_val_s = scaler.transform(X_val.reshape(-1, f)).reshape(n_val, T, f)

    if X_test is not None:
        n_test   = X_test.shape[0]
        X_test_s = scaler.transform(X_test.reshape(-1, f)).reshape(n_test, T, f)
        return X_tr_s, X_val_s, X_test_s, scaler

    return X_tr_s, X_val_s, scaler


def _train_and_eval(all_seqs, all_labels, all_last_dates,
                    train_dates, test_dates, input_size, device, cell):
    """Train one sliding-window fold; return (metrics, best_epoch)."""
    tr_mask   = np.isin(all_last_dates, train_dates)
    test_mask = np.isin(all_last_dates, test_dates)

    seqs_tr   = all_seqs[tr_mask];   labs_tr   = all_labels[tr_mask]
    seqs_test = all_seqs[test_mask];  labs_test = all_labels[test_mask]

    if len(seqs_tr) == 0 or len(seqs_test) == 0:
        return None, -1

    # 80/20 inner temporal split for early stopping
    X_itr, y_itr, X_ival, y_ival = _temporal_seq_split(seqs_tr, labs_tr, all_last_dates[tr_mask])

    # Scale using inner-train stats only
    X_itr_s, X_ival_s, X_test_s, _ = _scale(X_itr, X_ival, seqs_test)

    model, best_epoch = _fit(X_itr_s, y_itr, X_ival_s, y_ival, input_size, device, cell)

    test_dl    = DataLoader(SequenceDataset(X_test_s, labs_test), batch_size=BATCH_SIZE)
    preds, probas = _predict(model, test_dl, device)
    metrics = evaluate_model(labs_test.astype(int), preds, probas)
    return metrics, best_epoch


# -- main training pipeline ---------------------------------------------------

def train_lstm(horizon: int = 1, ft_type: str = "macro", cell: str = "lstm") -> dict:
    """Full training pipeline for LSTM/GRU.

    1. Load IBEX35 (micro) and index (macro) OHLCV from SQLite.
    2. Build features with ml_ready; apply cyclic dow encoding.
    3. Build all sequences (n, T=20, 41) from the full dataset once.
    4. Sliding window CV: for each window, scale, train with early stopping,
       evaluate on test sequences.
    5. Final model: inner 80/20 split of last WINDOW_DAYS, fit with early
       stopping, save artifact with scaler.

    Returns the saved artifact dict.
    """
    model_name = cell.upper()
    print(f"\n{'='*55}")
    print(f"  {model_name} | h={horizon} | ft_type={ft_type} | T={SEQ_LEN}")
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

    # Apply cyclic dow encoding (replaces integer 'dow' with sin/cos -> 41 features)
    X = add_cyclic_dow(X)

    tickers = df.loc[mask, "ticker"]
    dates   = df.loc[mask, "date"]

    input_size   = X.shape[1]  # 41
    unique_dates = np.sort(dates.unique())

    print(f"Usable rows   : {len(X)}")
    print(f"Unique dates  : {len(unique_dates)}")
    dist = y.value_counts(normalize=True)
    print(f"Class balance : down={dist.get(0, 0):.3f}  up={dist.get(1, 0):.3f}")
    print(f"Input size    : {input_size} features\n")

    # 3. Build all sequences once (avoids recomputing per window)
    print("Building sequences...")
    all_seqs, all_labels, all_last_dates = build_sequences(X, y, tickers, dates, SEQ_LEN)
    print(f"Total sequences: {len(all_seqs)} | shape: {all_seqs.shape}\n")

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

    model_key = cell.lower()  # "lstm" or "gru"
    out_path  = ARTIFACTS_PATH / f"{model_key}_h{horizon}.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nArtifact saved -> {out_path}")

    return artifact


# -- entry point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM/GRU direction classifier")
    parser.add_argument("--horizon", type=int, default=1,       help="Prediction horizon (days)")
    parser.add_argument("--cell",    type=str, default="lstm",  help="RNN cell type: lstm | gru")
    parser.add_argument("--ft-type", type=str, default="macro", help="Feature type: micro | cross | macro")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_lstm(horizon=args.horizon, ft_type=args.ft_type, cell=args.cell)
