"""
GRU trainer -- binary stock direction classification.

This module is a thin entry point that delegates entirely to lstm.py with
cell="gru". GRU and LSTM share the same feature set, sequence length, and
training pipeline; the only difference is the recurrent cell architecture:

  - GRU: 2 gates (reset, update) — ~25 % fewer parameters than LSTM.
  - LSTM: 3 gates + cell state — slightly more expressive at same hidden size.

At T=20 and hidden=64 the empirical gap is negligible; GRU trains faster and
is less prone to overfitting on financial noise, making it the recommended
default (see decisions/rnn_decisions.md).

Usage:
    python -m models.neural.gru                          # h=1, ft_type=macro
    python -m models.neural.gru --horizon 1 --ft-type macro

Output: artifacts/gru_h{horizon}.pkl
"""

import argparse

from models.neural.lstm import train_lstm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU direction classifier")
    parser.add_argument("--horizon", type=int, default=1,       help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro", help="Feature type: micro | cross | macro")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_lstm(horizon=args.horizon, ft_type=args.ft_type, cell="gru")
