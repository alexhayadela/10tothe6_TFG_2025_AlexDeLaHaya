# Continuous Target (Return Regression) -- Design Decisions

**Date:** 2026-04-21
**Task:** Predicting `future_log_ret` (continuous) instead of the binary direction label
**Data:** 35 IBEX35 stocks, ~5000 trading days each, 41 features
**Training scheme:** Same 3-year sliding window / expanding CV as discrete models

---

## 1. What changes and what does not

The feature pipeline (`ml_ready`) already computes `future_log_ret = log(close[t+h] / close[t])`.
The discrete pipeline then binarises it: `target = (future_log_ret > 0).astype(int)`.

Adding continuous support means exposing `future_log_ret` directly as `y` instead of the binarised `target`, and switching each model's loss / objective function accordingly.  The feature matrix `X` is identical in both modes — no feature engineering changes are needed.

The `BaseTrainer.run()` pipeline is unchanged. The `mode` parameter (`target_type`) is passed at construction time and each trainer reads it to decide which `y` to use from the DataFrame.

---

## 2. Which models support continuous output — and why

### RF (`rf`) — **No**

Random Forest `RandomForestClassifier` maps one-to-one with the classification task.
`RandomForestRegressor` exists and could predict returns, but it has a fundamental structural problem for financial return regression: it cannot extrapolate beyond the range of training labels (it outputs averages within leaves).  Daily log returns during a tail event (e.g., -8% during a flash crash) will be predicted as the average of the most extreme training leaf — severely underestimating the actual move.  The practical consequence is that the model's risk signal collapses exactly when it matters most.

More importantly, the existing RF model already outputs `predict_proba` as a probability-like score that encodes return magnitude implicitly (a stock whose features strongly indicate momentum gets P(up) = 0.72, which is already a richer signal than a binary label).  Converting to a regressor loses the calibrated probability output used downstream for position sizing without gaining much useful return-magnitude information.

**Decision: RF stays classification-only.**

### XGBoost (`xgb`) — **Yes**

XGBoost natively supports regression via `objective="reg:squarederror"`.  Switching is a single parameter change.  The boosting framework, early stopping, and the two-step final-training procedure all carry over unchanged.

XGBoost regression on financial returns is well-studied.  Gu et al. (2020) use gradient-boosted trees with squared-error loss on monthly US equity returns and find it among the top performers for cross-sectional return prediction.  The key advantage over classification is that XGBoost regression preserves return magnitude in the loss signal: a prediction of -0.001 when the actual is -0.03 produces a larger gradient update than a prediction of +0.001, whereas both are classified identically as "wrong" under log-loss.

Evaluation for regression uses standard metrics (MAE, RMSE, R², directional accuracy) — the existing `evaluate_model` is replaced with `evaluate_regression`.

### GRU / LSTM (`gru`, `lstm`) — **Yes**

The only architectural change is replacing `BCEWithLogitsLoss` with `MSELoss` (or `HuberLoss`) and removing the sigmoid output.  The `StockRNN` model already outputs raw scalars from `nn.Linear(hidden_size, 1)` — for classification, a sigmoid is applied post-hoc; for regression, the raw output is the return prediction.

Neural return regression on equity sequences has clear precedent: Fischer & Krauss (2018) use LSTM for direction prediction but note that regression formulations are used interchangeably in the literature.  Sezer et al. (2020) review both formulations and find comparable results for daily predictions when the architecture is otherwise identical.

**Loss choice: MSELoss vs HuberLoss.**
MSE heavily penalises outliers (large return days) — which is actually desirable here since we want the model to be most cautious about extreme predictions.  However, financial returns have heavy tails (Cont, 2001): a single -5% day has 25x the MSE contribution of a -1% day, which can destabilise training if several extreme events fall in the same mini-batch.  HuberLoss (delta=0.01, approximately 1% daily return) transitions to MAE for returns beyond the threshold, preventing outliers from dominating gradients.

**Decision: HuberLoss (delta=0.01) for GRU/LSTM regression.**

### CNN+GRU / CNN+LSTM (`cnn_gru`, `cnn_lstm`) — **Yes**

Same reasoning as GRU/LSTM.  The CNN front-end is feature extraction only; the output head change is identical.

### Markov (`markov`) — **No**

The Markov model estimates `P(up | state)` — a probability derived from counting transitions.  There is no natural continuous extension without fundamentally redesigning the model (e.g., fitting a Gaussian emission per state, which would require at least 2 parameters per state and a different estimation procedure — essentially a Hidden Markov Model).

The existing Markov model can still produce a return-like signal by interpreting P(up) - 0.5 as a signed "expected return magnitude", but this is a monotone transformation of the existing output and does not add information.  A proper continuous Markov model is a separate architecture (see `decisions/markov_decisions.md`, Section 6).

**Decision: Markov stays classification-only.**

---

## 3. Summary table

| Model | Continuous? | Reason |
|-------|-------------|--------|
| `rf` | No | RF regressor cannot extrapolate; `predict_proba` already encodes magnitude |
| `xgb` | Yes | `objective="reg:squarederror"` is a one-line change; magnitude in loss signal |
| `gru` | Yes | `MSELoss`/`HuberLoss` replaces `BCEWithLogitsLoss`; no arch change needed |
| `lstm` | Yes | Same as GRU |
| `cnn_gru` | Yes | Same as GRU; CNN head is feature extraction only |
| `cnn_lstm` | Yes | Same as GRU |
| `markov` | No | Probabilistic counting; no natural regression extension without HMM redesign |

---

## 4. Regression evaluation metrics

Since `evaluate_model` is classification-specific, a parallel `evaluate_regression` is introduced in `models/evaluate.py`:

| Metric | Formula | Role |
|--------|---------|------|
| MAE | `mean(\|y-ŷ\|)` | Primary: robust, interpretable in return units |
| RMSE | `sqrt(mean((y-ŷ)²))` | Penalises large errors; watch for outlier sensitivity |
| R² | `1 - SS_res/SS_tot` | Fraction of variance explained; financial R² is typically 0.001-0.01 |
| Directional accuracy | `mean(sign(y) == sign(ŷ))` | Bridge to classification; is the model at least right on direction? |
| Information coefficient | Spearman rank corr(y, ŷ) | Standard in quantitative finance (Grinold & Kahn, 2000); rank-based, robust to outliers |

**Primary metric: IC (Information Coefficient).**

IC is the standard cross-sectional return predictability measure used in quantitative equity research (Grinold & Kahn, 2000; Gu et al., 2020).  It is threshold-independent, robust to the heavy-tailed return distribution, and directly interpretable: IC = 0.05 means the model's ranking of stocks by predicted return has a Spearman correlation of 5% with actual returns.  Even IC values of 0.03-0.06 are considered actionable in systematic equity strategies.

MAE is reported in basis points (1 bp = 0.0001 log return) to give a sense of prediction error in practical trading units.

---

## 5. Implementation: `target_type` parameter

A single new argument `target_type: str = "discrete"` is added to `BaseTrainer.__init__`.

- `"discrete"`: uses `y = df["target"]` (binary 0/1) — existing behaviour, default, all models
- `"continuous"`: uses `y = df["future_log_ret"]` (raw float) — new, only for XGB/GRU/LSTM/CNN variants

`BaseTrainer.run()` selects the correct `y` column from `df` based on `target_type`.  Everything downstream — `_train_window`, `_train_final`, metrics — is handled by each trainer individually.

The CLI gains `--target-type {discrete,continuous}`.

---

## 6. Artifact format for regression models

Regression artifacts follow the same structure as classification artifacts with three differences:

1. `target_type: "continuous"` is stored in the artifact.
2. `cv_metrics` and `cv_summary` contain regression metrics (MAE, RMSE, R², dir_acc, ic) instead of classification metrics.
3. No `predict_proba` — the model output is a float prediction of `future_log_ret`.

---

## References

- **Cont, R. (2001).** "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance*, 1(2), 223-236. Heavy-tailed return distribution motivates HuberLoss over MSELoss for neural regression.

- **Fischer, T. & Krauss, C. (2018).** "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669. LSTM applied to S&P 500 direction; continuous formulations discussed as natural extension.

- **Grinold, R.C. & Kahn, R.N. (2000).** *Active Portfolio Management*. McGraw-Hill. Defines the Information Coefficient (IC) as the correlation between predicted and realised returns; establishes IC as the primary metric for evaluating return-forecasting models in quantitative equity strategies.

- **Gu, S., Kelly, B. & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273. Largest-scale ML comparison for equity return regression; uses R² and IC as primary metrics; finds gradient-boosted trees among top performers for monthly return prediction.

- **Sezer, O.B., Gudelek, M.U. & Ozbayoglu, A.M. (2020).** "Financial time series forecasting with deep learning: a systematic literature review: 2005-2019." *Applied Soft Computing*, 90, 106181. Reviews both classification and regression formulations for equity prediction; finds comparable results when architecture is otherwise identical.
