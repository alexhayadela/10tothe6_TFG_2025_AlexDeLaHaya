"""
Inference entry point for all trained models.

Loads a saved artifact by (model, horizon, mode, target_type), fetches the
right number of OHLCV rows for the artifact's ft_type, and returns one
prediction per ticker for the most recent available date.

Usage:
    python -m models.predict --model rf
    python -m models.predict --model xgb --horizon 1 --mode expanding --target-type discrete
    python -m models.predict --model lstm --horizon 1 --mode sliding --target-type discrete

ft_type is read from the artifact — no need to pass it manually.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch

from config import load_env, ARTIFACTS_PATH
from db.supabase.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready
from models.neural.lstm import (
    add_cyclic_dow,
    SequenceDataset,
    StockRNN,
    BATCH_SIZE,
)
from models.neural.cnn_rnn import StockCNNRNN
from torch.utils.data import DataLoader


# -- row-fetch constants (one per ft_type, tunable independently) -------------

ROWS_MICRO = 260   # 252-day momentum is the longest rolling window
ROWS_CROSS = 260   # breadth adds only a 10-day rolling, same bottleneck
ROWS_MACRO = 260   # VIX 250-day percentile, but micro bottleneck dominates

_ROWS = {"micro": ROWS_MICRO, "cross": ROWS_CROSS, "macro": ROWS_MACRO}

# -- model sets ---------------------------------------------------------------

NEURAL_MODELS = {"gru", "lstm", "cnn_gru", "cnn_lstm"}


# -- artifact loading ---------------------------------------------------------

def load_artifact(model: str, horizon: int, mode: str, target_type: str) -> dict:
    name = f"{model}_h{horizon}_{mode}_{target_type}.pkl"
    path = ARTIFACTS_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


# -- tree inference -----------------------------------------------------------

def _predict_tree(artifact: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, probas) for tree/markov/meta models."""
    model        = artifact["model"]
    feature_cols = artifact["features"]
    X_aligned    = X[feature_cols]
    preds        = model.predict(X_aligned)
    if artifact.get("target_type") == "continuous":
        raw = preds.astype(float)
        return raw, raw
    probas_raw = model.predict_proba(X_aligned)
    if probas_raw.ndim == 2:
        pred_proba = probas_raw[np.arange(len(preds)), preds]
    else:
        pred_proba = probas_raw
    return preds, pred_proba


# -- RNN inference ------------------------------------------------------------

def _reconstruct_rnn(artifact: dict) -> torch.nn.Module:
    """Reconstruct StockRNN or StockCNNRNN from saved config + state dict."""
    cfg = artifact["model_config"]
    model_key = artifact["model_key"]
    if model_key.startswith("cnn_"):
        model = StockCNNRNN(
            input_size=cfg["input_size"],
            num_filters=cfg["num_filters"],
            kernel_size=cfg["kernel_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            cell=cfg["cell"],
        )
    else:
        model = StockRNN(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            cell=cfg["cell"],
        )
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model


def _predict_rnn(artifact: dict, X: pd.DataFrame,
                 tickers: pd.Series, dates: pd.Series,
                 target_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, probas, last_dates, last_tickers) for RNN models.

    Builds sequences per ticker while tracking ticker identity, applies the
    stored scaler, runs the model forward pass.
    """
    feature_cols = artifact["features"]
    seq_len      = artifact["seq_len"]
    scaler       = artifact["scaler"]

    X_enc = (add_cyclic_dow(X) if "dow" in X.columns else X)[feature_cols]

    all_seqs, all_last_dates, all_tickers_list = [], [], []
    for ticker in tickers.unique():
        tmask  = tickers == ticker
        X_t    = X_enc.loc[tmask]
        d_t    = dates.loc[tmask]
        order  = d_t.argsort()
        X_vals = X_t.iloc[order.values].values
        d_vals = d_t.iloc[order.values].values
        n = len(X_vals)
        for i in range(seq_len - 1, n):
            all_seqs.append(X_vals[i - seq_len + 1: i + 1])
            all_last_dates.append(d_vals[i])
            all_tickers_list.append(ticker)

    if not all_seqs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    seqs         = np.array(all_seqs, dtype=np.float32)
    last_dates   = np.array(all_last_dates)
    last_tickers = np.array(all_tickers_list)

    n, T, f = seqs.shape
    seqs_scaled = scaler.transform(seqs.reshape(-1, f)).reshape(n, T, f).astype(np.float32)

    loader = DataLoader(
        SequenceDataset(seqs_scaled, np.zeros(n, dtype=np.float32)),
        batch_size=BATCH_SIZE,
    )

    model = _reconstruct_rnn(artifact)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    outputs = []
    with torch.no_grad():
        for X_b, _ in loader:
            outputs.append(model(X_b.to(device)).cpu())
    outputs = torch.cat(outputs)

    if target_type == "continuous":
        raw = outputs.numpy()
        return raw, raw, last_dates, last_tickers

    probas = torch.sigmoid(outputs).numpy()
    preds  = (probas >= 0.5).astype(int)
    return preds, probas, last_dates, last_tickers


# -- main inference pipeline --------------------------------------------------

def get_predictions(model: str, horizon: int = 1,
                    mode: str = "sliding",
                    target_type: str = "discrete") -> pd.DataFrame:
    """Run full inference pipeline; return one prediction per ticker.

    Loads artifact, fetches the right number of OHLCV rows for the stored
    ft_type, builds features, runs inference, and keeps only the most recent
    prediction per ticker.

    Returns DataFrame with columns: ticker, date, pred, proba, model.
    """
    if target_type == "continuous":
        print("Feature will be added in the future.")
        return pd.DataFrame(columns=["ticker", "date", "pred", "proba", "model"])

    artifact = load_artifact(model, horizon, mode, target_type)
    ft_type  = artifact["ft_type"]
    rows     = _ROWS[ft_type]

    tickers      = get_ibex_tickers()
    macro_tickers = get_macro_tickers()

    df_micro = fetch_ohlcv(tickers, rows)
    df_macro = fetch_ohlcv(macro_tickers, rows) if ft_type == "macro" else None

    df, X_train, _, mask, _ = ml_ready(horizon, df_micro, df_macro=df_macro, ft_type=ft_type)

    # For inference we want all feature-complete rows, including the last `horizon`
    # rows whose targets are NaN (those are the rows we actually want to predict).
    _remove = ["ticker", "date", "open", "high", "low", "close",
               "volume", "target", "future_log_ret"]
    X_full = df.drop(columns=_remove).replace([np.inf, -np.inf], np.nan)
    feat_mask = X_full.notna().all(axis=1)
    X = X_full.loc[feat_mask]

    df_meta  = df.loc[feat_mask, ["ticker", "date"]].copy()
    tkr_col  = df.loc[feat_mask, "ticker"]
    date_col = df.loc[feat_mask, "date"]

    model_stem = f"{model}_h{horizon}_{mode}_{target_type}"

    if model in NEURAL_MODELS:
        preds, probas, last_dates, last_tickers = _predict_rnn(
            artifact, X, tkr_col, date_col, target_type
        )
        if len(preds) == 0:
            return pd.DataFrame(columns=["ticker", "date", "pred", "proba", "model"])
        df_out = pd.DataFrame({
            "ticker": last_tickers,
            "date":   last_dates,
            "pred":   preds,
            "proba":  probas,
        })
    else:
        preds, probas = _predict_tree(artifact, X)
        df_meta = df_meta.copy()
        df_meta["pred"]  = preds
        df_meta["proba"] = probas
        df_out = df_meta

    df_out = df_out.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)
    df_out["model"] = model_stem
    return df_out


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["rf", "xgb", "gru", "lstm", "cnn_gru", "cnn_lstm", "markov"],
        help="Model to run inference with",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in trading days (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["sliding", "expanding"],
        default="sliding",
        help="Training mode used when the model was trained (default: sliding)",
    )
    parser.add_argument(
        "--target-type",
        dest="target_type",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Target type used when the model was trained (default: discrete)",
    )
    args = parser.parse_args()

    load_env()

    df = get_predictions(
        model=args.model,
        horizon=args.horizon,
        mode=args.mode,
        target_type=args.target_type,
    )
    print(df.to_string(index=False))
