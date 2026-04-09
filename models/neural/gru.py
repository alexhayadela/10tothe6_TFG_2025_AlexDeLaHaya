"""
GRU trainer -- binary stock direction classification.

Thin entry point that delegates to RNNTrainer(cell="gru").

GRU and LSTM share the same feature set, sequence length, and training
pipeline; the only difference is the recurrent cell (2 gates vs 3 + cell
state). At T=20/hidden=64 the accuracy gap is negligible; GRU trains
faster. See decisions/rnn_decisions.md for the full rationale.

Usage:
    python -m models.neural.gru                          # h=1, ft_type=macro
    python -m models.neural.gru --horizon 1 --mode expanding

Usage (via framework):
    python -m models.train --model gru --horizon 1 --mode sliding

Output: artifacts/gru_h{horizon}.pkl
"""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU direction classifier")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    from models.neural.rnn_trainer import RNNTrainer
    RNNTrainer(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode, cell="gru").run()
