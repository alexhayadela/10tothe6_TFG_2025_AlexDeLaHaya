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

import torch
import torch.nn as nn

# Constants reused by rnn_trainer.py when building CNNRNNTrainer
from models.neural.lstm import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


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


# -- main training pipeline ---------------------------------------------------

def train_cnn_rnn(horizon: int = 1, ft_type: str = "macro", cell: str = "gru",
                  mode: str = "sliding") -> dict:
    """Delegate to CNNRNNTrainer (lazy import avoids circular dependency)."""
    from models.neural.rnn_trainer import CNNRNNTrainer  # noqa: PLC0415
    return CNNRNNTrainer(horizon=horizon, ft_type=ft_type, mode=mode, cell=cell).run()


# -- entry point --------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN+GRU/LSTM direction classifier")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--cell",    type=str, default="gru",     help="RNN cell type: gru | lstm")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_cnn_rnn(horizon=args.horizon, ft_type=args.ft_type, cell=args.cell, mode=args.mode)
