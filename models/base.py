"""
BaseTrainer -- shared training pipeline for all models.

All model trainers inherit from BaseTrainer and implement four methods:
  model_key       : str property  -- artifact stem, e.g. "rf", "gru", "cnn_gru"
  _after_features : hook          -- neural trainers build sequences here
  _train_window   : (train_dates, test_dates) -> (metrics | None, meta)
  _train_final    : (final_dates, cv_summary, all_metrics, all_meta) -> dict

run() orchestrates the full pipeline identically for all models:
  load data -> build features -> [hook] -> CV loop -> aggregate ->
  train final model -> assemble + save artifact

Window functions:
  sliding_windows   : fixed-size training window (WINDOW_DAYS days)
  expanding_windows : growing window starting from min_train days
"""

import abc
import numpy as np
import joblib

from config import ARTIFACTS_PATH
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers
from models.trees.features import ml_ready


# -- shared CV constants ------------------------------------------------------

WINDOW_DAYS = 750   # ~3 years of trading days
STEP_DAYS   = 63    # ~quarterly steps between windows


# -- window generators --------------------------------------------------------

def sliding_windows(dates: np.ndarray, window: int, step: int,
                    embargo: int = 1, min_test: int = 21) -> list:
    """Generate (train_dates, test_dates) pairs with a fixed-size training window.

    Each training window contains exactly `window` dates. An embargo of 1 day
    separates the last training date from the first test date to prevent feature
    leakage at overlapping rolling windows (especially relevant at h=1).
    Windows advance by `step` dates at a time.
    """
    windows = []
    i = window
    while i + embargo + min_test <= len(dates):
        train_dates = dates[i - window : i]
        test_start  = i + embargo
        test_end    = min(i + embargo + step, len(dates))
        if test_end - test_start < min_test:
            break
        test_dates = dates[test_start:test_end]
        windows.append((train_dates, test_dates))
        i += step
    return windows


def expanding_windows(dates: np.ndarray, min_train: int, step: int,
                      embargo: int = 1, min_test: int = 21) -> list:
    """Generate (train_dates, test_dates) pairs with a growing training window.

    Unlike sliding_windows, the training window starts at position 0 and grows
    by `step` dates with each fold. The first fold requires at least `min_train`
    training dates. Useful when you believe all historical data is informative
    and does not go stale (or when comparing against sliding to quantify staleness).
    """
    windows = []
    i = min_train
    while i + embargo + min_test <= len(dates):
        train_dates = dates[:i]           # expands from the beginning
        test_start  = i + embargo
        test_end    = min(i + embargo + step, len(dates))
        if test_end - test_start < min_test:
            break
        test_dates = dates[test_start:test_end]
        windows.append((train_dates, test_dates))
        i += step
    return windows


# -- base trainer -------------------------------------------------------------

class BaseTrainer(abc.ABC):
    """Template-method base class.  Subclasses implement _train_window and
    _train_final; run() handles everything else.

    Instance attributes set by run() (available inside _train_window /
    _train_final / _after_features):
        self.X            pd.DataFrame  -- flat feature matrix (all usable rows)
        self.y            pd.Series     -- binary target
        self.dates        pd.Series     -- date column aligned with X/y
        self.tickers      pd.Series     -- ticker column aligned with X/y
        self.unique_dates np.ndarray    -- sorted unique prediction dates
    """

    WINDOW_DAYS = WINDOW_DAYS
    STEP_DAYS   = STEP_DAYS

    def __init__(self, horizon: int = 1, ft_type: str = "macro", mode: str = "sliding"):
        self.horizon      = horizon
        self.ft_type      = ft_type
        self.mode         = mode
        # populated by run() before any abstract method is called
        self.X            = None
        self.y            = None
        self.dates        = None
        self.tickers      = None
        self.unique_dates = None

    # -- data -----------------------------------------------------------------

    def _load_raw(self):
        """Fetch OHLCV for IBEX35 stocks (micro) and index tickers (macro)."""
        ibex_tickers  = get_ibex_tickers()
        macro_tickers = get_macro_tickers()
        with sqlite_connection() as conn:
            df_micro_raw = fetch_ohlcv(ibex_tickers)
            df_macro_raw = fetch_ohlcv(macro_tickers)
        # volume filter applies only to individual stocks (indices have no real volume)
        df_micro_raw = df_micro_raw[df_micro_raw["volume"] > 0].reset_index(drop=True)
        df_macro_raw = df_macro_raw.reset_index(drop=True)
        return df_micro_raw, df_macro_raw

    def _build_features(self, df_micro_raw, df_macro_raw):
        """Run ml_ready pipeline; returns (df, X, y, mask)."""
        df_macro_arg = df_macro_raw if self.ft_type == "macro" else None
        return ml_ready(self.horizon, df_micro_raw, df_macro=df_macro_arg, ft_type=self.ft_type)

    # -- windows --------------------------------------------------------------

    def make_windows(self, unique_dates: np.ndarray) -> list:
        """Return (train_dates, test_dates) pairs for the chosen CV mode."""
        if self.mode == "sliding":
            return sliding_windows(unique_dates, self.WINDOW_DAYS, self.STEP_DAYS)
        elif self.mode == "expanding":
            return expanding_windows(unique_dates, self.WINDOW_DAYS, self.STEP_DAYS)
        raise ValueError(f"mode must be 'sliding' or 'expanding', got '{self.mode}'")

    # -- hooks ----------------------------------------------------------------

    def _after_features(self):
        """Called after self.X/y/dates/tickers/unique_dates are populated.

        Neural trainers override this to apply cyclic-dow encoding and build
        the (n, T, F) sequence arrays used by _train_window / _train_final.
        No-op for tree models.
        """
        pass

    def _print_header(self):
        """Override to print model-specific lines inside the header block."""
        pass

    def _meta_str(self, meta) -> str:
        """Override to append per-window metadata (e.g. best_iter) to the log."""
        return ""

    def _aggregate_meta(self, all_meta: list, cv_summary: dict):
        """Override to print / store aggregate CV metadata (e.g. mean best_epoch)."""
        pass

    # -- main pipeline --------------------------------------------------------

    def run(self) -> dict:
        """Orchestrate: load -> features -> [hook] -> CV -> final model -> save."""
        print(f"\n{'='*55}")
        print(f"  {self.model_key.upper()} | h={self.horizon} | ft={self.ft_type} | mode={self.mode}")
        self._print_header()
        print(f"{'='*55}\n")

        # 1. Load data
        df_micro_raw, df_macro_raw = self._load_raw()

        # 2. Build features
        df, X, y, mask = self._build_features(df_micro_raw, df_macro_raw)
        self.X            = X
        self.y            = y
        self.dates        = df.loc[mask, "date"]
        self.tickers      = df.loc[mask, "ticker"]
        self.unique_dates = np.sort(self.dates.unique())

        print(f"Usable rows   : {len(X)}")
        print(f"Unique dates  : {len(self.unique_dates)}")
        dist = y.value_counts(normalize=True)
        print(f"Class balance : down={dist.get(0, 0):.3f}  up={dist.get(1, 0):.3f}\n")

        # 3. Model-specific post-processing (sequence building for neural models)
        self._after_features()

        # 4. CV loop
        windows = self.make_windows(self.unique_dates)
        mode_info = (
            f"train={self.WINDOW_DAYS}d (fixed)"
            if self.mode == "sliding"
            else f"min_train={self.WINDOW_DAYS}d (expanding)"
        )
        print(f"CV: {len(windows)} windows | {mode_info} | step={self.STEP_DAYS}d | embargo=1d\n")

        all_metrics, all_meta = [], []
        for i, (train_dates, test_dates) in enumerate(windows):
            metrics, meta = self._train_window(train_dates, test_dates)
            if metrics is None:
                continue
            all_metrics.append(metrics)
            all_meta.append(meta)
            print(
                f"  [{i+1:2d}/{len(windows)}] "
                f"test {str(test_dates[0])[:10]} -> {str(test_dates[-1])[:10]} | "
                f"bal_acc={metrics['balanced_accuracy']:.4f}  "
                f"auc={metrics['roc_auc']:.4f}  "
                f"mcc={metrics['mcc']:.4f}  "
                + self._meta_str(meta)
            )

        # 5. Aggregate CV
        cv_summary = {}
        print(f"\n{'-'*55}")
        print("CV aggregate (mean +/- std):")
        for key in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "mcc"]:
            vals = [m[key] for m in all_metrics]
            mean, std = np.mean(vals), np.std(vals)
            cv_summary[key] = {"mean": float(mean), "std": float(std)}
            marker = " <- primary" if key == "balanced_accuracy" else ""
            print(f"  {key:22s}: {mean:.4f} +/- {std:.4f}{marker}")
        self._aggregate_meta(all_meta, cv_summary)

        # 6. Final training window
        #    sliding -> last WINDOW_DAYS;  expanding -> all available dates
        final_dates = (
            self.unique_dates[-self.WINDOW_DAYS:]
            if self.mode == "sliding"
            else self.unique_dates
        )
        label = f"last {self.WINDOW_DAYS}d" if self.mode == "sliding" else "all data"
        print(f"\n{'-'*55}")
        print(f"Training final model ({label}) ...")

        model_fields = self._train_final(final_dates, cv_summary, all_metrics, all_meta)

        # 7. Assemble + save artifact
        artifact = {
            "model_key":   self.model_key,
            "horizon":     self.horizon,
            "ft_type":     self.ft_type,
            "mode":        self.mode,
            "window_days": len(final_dates),
            "train_start": str(final_dates[0])[:10],
            "train_end":   str(final_dates[-1])[:10],
            "cv_metrics":  all_metrics,
            "cv_summary":  cv_summary,
            **model_fields,
        }

        out_path = ARTIFACTS_PATH / f"{self.model_key}_h{self.horizon}.pkl"
        joblib.dump(artifact, out_path)
        print(f"\nArtifact saved -> {out_path}")
        return artifact

    # -- abstract interface ---------------------------------------------------

    @property
    @abc.abstractmethod
    def model_key(self) -> str:
        """Artifact filename stem, e.g. 'rf', 'xgb', 'gru', 'cnn_gru'."""

    @abc.abstractmethod
    def _train_window(self, train_dates, test_dates) -> tuple:
        """Train one CV fold.  Return (metrics_dict | None, meta)."""

    @abc.abstractmethod
    def _train_final(self, final_dates, cv_summary: dict,
                     all_metrics: list, all_meta: list) -> dict:
        """Train the final model.  Return model-specific artifact fields as dict."""
