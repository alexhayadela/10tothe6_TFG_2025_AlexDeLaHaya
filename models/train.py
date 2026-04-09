"""
Unified training entry point for all models.

Selects model, horizon, training mode, and feature type from the command line.
All models share the same BaseTrainer pipeline (data loading, CV loop,
artifact saving); only the model-specific logic differs.

Usage:
    python -m models.train --model rf       --horizon 1
    python -m models.train --model xgb      --horizon 1 --mode expanding
    python -m models.train --model gru      --horizon 1 --mode sliding
    python -m models.train --model lstm     --horizon 1 --ft-type cross
    python -m models.train --model cnn_gru  --horizon 1
    python -m models.train --model cnn_lstm --horizon 1 --mode expanding
    python -m models.train --model all      --horizon 1  # runs every model

Training modes:
    sliding   : fixed WINDOW_DAYS-day training window, advances by STEP_DAYS
    expanding : window grows from WINDOW_DAYS to all available history

Feature types (ft_type):
    micro  : per-stock technical indicators only
    cross  : micro + IBEX35 breadth (leave-one-out)
    macro  : cross + VIX / S&P500 / relative-to-market  [default]
"""

import argparse

from models.trees.rf        import RFTrainer
from models.trees.xgb       import XGBTrainer
from models.neural.rnn_trainer import RNNTrainer, CNNRNNTrainer


# -- model registry -----------------------------------------------------------

REGISTRY: dict = {
    "rf":       lambda **kw: RFTrainer(**kw),
    "xgb":      lambda **kw: XGBTrainer(**kw),
    "gru":      lambda **kw: RNNTrainer(cell="gru",  **kw),
    "lstm":     lambda **kw: RNNTrainer(cell="lstm", **kw),
    "cnn_gru":  lambda **kw: CNNRNNTrainer(cell="gru",  **kw),
    "cnn_lstm": lambda **kw: CNNRNNTrainer(cell="lstm", **kw),
}


def train(model: str, horizon: int = 1, ft_type: str = "macro",
          mode: str = "sliding") -> dict | list:
    """Programmatic entry point.  Pass model='all' to run every model.

    Returns the artifact dict for a single model, or a list of dicts for 'all'.
    """
    kw = {"horizon": horizon, "ft_type": ft_type, "mode": mode}
    if model == "all":
        return [REGISTRY[m](**kw).run() for m in REGISTRY]
    if model not in REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(REGISTRY)} or 'all'")
    return REGISTRY[model](**kw).run()


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a direction-classification model on IBEX35 data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(REGISTRY) + ["all"],
        help=(
            "Model to train:\n"
            "  rf        Random Forest\n"
            "  xgb       XGBoost\n"
            "  gru       GRU (recurrent)\n"
            "  lstm      LSTM (recurrent)\n"
            "  cnn_gru   CNN + GRU hybrid\n"
            "  cnn_lstm  CNN + LSTM hybrid\n"
            "  all       Run every model sequentially"
        ),
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
        help=(
            "CV training mode (default: sliding):\n"
            "  sliding   Fixed WINDOW_DAYS-day window\n"
            "  expanding Window grows from WINDOW_DAYS to all history"
        ),
    )
    parser.add_argument(
        "--ft-type",
        dest="ft_type",
        choices=["micro", "cross", "macro"],
        default="macro",
        help=(
            "Feature set (default: macro):\n"
            "  micro  Per-stock technical indicators only\n"
            "  cross  micro + IBEX35 breadth (leave-one-out)\n"
            "  macro  cross + VIX / S&P500 / relative-to-market"
        ),
    )
    args = parser.parse_args()

    from config import load_env
    load_env()

    train(model=args.model, horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
