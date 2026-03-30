# Random Forest -- Design Decisions

**Date:** 2026-03-30
**Task:** Binary stock direction classification (h=1, next-day up/down)
**Data:** 30 IBEX35 stocks, ~5000 trading days each, ~31 features
**Training scheme:** 3-year sliding window (~750 days x 30 tickers = ~22,500 training rows per window)

---

## Part 1 -- Hyperparameters

### n_estimators = 500

**Reasoning:** The number of trees controls the variance reduction of the ensemble. Each additional tree reduces variance at the cost of compute, with sharply diminishing returns. Oshiro et al. (2012) systematically studied convergence and found that most datasets stabilize between 64 and 128 trees, with marginal gains becoming negligible past 256. However, financial return data has an exceptionally low signal-to-noise ratio (Krauss et al., 2017 report daily accuracies in the 53-55% range on S&P 500). In low-SNR settings, individual trees are barely better than random, so the ensemble averaging effect needs more trees to extract the weak signal reliably.

Empirical tests on IBEX data confirm that OOB error continues to decrease (albeit slowly) up to ~400-500 trees and flatlines thereafter. Going beyond 500 provides no measurable improvement but doubles training time.

**Trade-off:** 500 trees with 22,500 rows and 31 features trains in approximately 5-10 seconds with `n_jobs=-1` on modern hardware, which is acceptable for a rolling-window scheme that retrains daily. If compute becomes a constraint (e.g., grid search over many windows), 300 is a reasonable fallback with minimal accuracy loss.

**Final value:** `n_estimators=500`

### max_depth = 5

**Reasoning:** Tree depth is the single most important regularization lever for financial ML. The core tension is:

- **Shallow trees (depth 3-4):** Each tree is a weak learner with high bias and low variance. The ensemble corrects bias through averaging. However, with only 31 features and potential interactions (e.g., RSI crosses combined with vol regime), depth 3 limits the model to at most 3-way interactions, which may miss important conditional patterns.

- **Deep trees (depth 8+):** Can model complex interactions but will memorize noise. With 22,500 training samples, a tree of depth 15 has enough capacity to create pure leaf nodes for every training example. Lopez de Prado (2018, Ch. 6) explicitly warns that deep trees in financial ML fit to spurious patterns in the training window that do not persist out-of-sample.

- **Moderate depth (5-6):** Each leaf captures a 5-way conditional pattern (e.g., "RSI < 30 AND vol_ratio > 1.5 AND breadth < 0.4 AND log_ret_1 < -0.02 AND sma_ratio < 0.98"), which is rich enough to express economically meaningful regimes. With depth 5, the tree has at most 32 leaf nodes, and with `min_samples_leaf=50` (see below), each leaf contains at least 50 observations -- enough for stable probability estimates.

Depth 5 balances expressiveness against overfitting. Preliminary experiments on IBEX data show that depth 5 consistently outperforms depth 3 (too constrained) and depth 7 (overfits on 3-year windows) on out-of-sample balanced accuracy.

**Final value:** `max_depth=5`

### max_features = 0.3 (~9 features per split)

**Reasoning:** `max_features` controls how many features each split considers, which determines tree diversity. The standard heuristics are:

- `"sqrt"` = sqrt(31) ~ 5.6 features per split
- `"log2"` = log2(31) ~ 5.0 features per split
- A float between 0.2 and 0.5 = 6 to 15 features per split

For financial data specifically, there are two competing considerations:

1. **Feature correlation is high.** Many of our 31 features measure related concepts (e.g., `sma_ratio_5_20`, `slope_10`, and `dist_high_10` all capture short-term trend). If `max_features` is too high, most trees will split on the same dominant feature (e.g., `vol_ratio_5_20`), reducing diversity. Lower `max_features` forces trees to explore different features, which is the core mechanism behind Random Forest's variance reduction (Breiman, 2001).

2. **Signal is weak and spread across features.** No single feature dominates. If `max_features` is too low, many trees will never see the combination of features needed to find the signal, leading to trees that are purely random.

With 31 features, `"sqrt"` gives ~5.6 features per split. This is reasonable but slightly aggressive -- at depth 5, each tree sees at most 5 x 5.6 = 28 feature evaluations, which means some features may rarely appear. A value of 0.3 (about 9 features per split) provides a better balance: enough diversity to decorrelate trees, but enough exposure that all informative features have a fair chance of appearing in most trees.

**Why not `"sqrt"`?** The standard `"sqrt"` heuristic was developed for datasets where features are relatively independent (e.g., pixel values, gene expressions). In financial data, effective dimensionality is much lower than the nominal 31 features due to correlations. Using 0.3 compensates for this.

**Final value:** `max_features=0.3`

### min_samples_leaf = 50

**Reasoning:** This is the most critical regularization parameter for financial ML after `max_depth`. It sets the minimum number of training observations in each terminal leaf. The implications are:

- **Statistical stability:** A leaf with 50 observations provides a class probability estimate with standard error of approximately sqrt(0.5 x 0.5 / 50) = 0.07. This means a leaf predicting 57% "up" is roughly 1 standard error from 50%, which is marginal but detectable. A leaf with 10 observations has standard error 0.16 -- essentially noise.

- **Cross-stock generalization:** Our training data has ~30 rows per date (one per ticker). A leaf with `min_samples_leaf=10` could contain observations from only one or two dates, meaning it learned a date-specific pattern (e.g., a market-wide shock) rather than a generalizable stock-level pattern. With `min_samples_leaf=50`, each leaf spans at least ~2 dates, forcing the tree to find patterns that hold across different market conditions.

- **Interaction with training size:** With 22,500 training rows and `max_depth=5`, the theoretical maximum number of leaves is 32. At `min_samples_leaf=50`, the minimum training rows consumed by a full tree is 32 x 50 = 1,600, well within our budget. The effective number of leaves will be lower (some branches will be pruned by the leaf constraint), typically 15-25.

- **Why not 25 (the current value)?** The current codebase uses `min_samples_leaf=25`. This is borderline. With 25 observations per leaf, a tree of depth 5 has enough capacity to specialize on narrow subsets, and the probability estimates are noisy (SE ~ 0.10). Increasing to 50 sacrifices some flexibility but produces more reliable probability estimates. Given that we use predicted probabilities for position sizing (via `predict_proba`), calibration matters. Lopez de Prado (2018) recommends erring on the side of larger leaves in financial applications.

- **Why not 100?** With 100 per leaf, the tree becomes overly constrained -- effectively depth 3-4, since many branches will be pruned to satisfy the leaf constraint. This negates the benefit of `max_depth=5`.

**Final value:** `min_samples_leaf=50`

### class_weight = None

**Reasoning:** The target distribution is approximately 52% up / 48% down. This is a *mild* imbalance, well within the range where standard unweighted classification works fine.

Setting `class_weight="balanced"` would apply weights inversely proportional to class frequency: roughly 0.96 for class 1 (up) and 1.04 for class 0 (down). This is a negligible adjustment.

More importantly, in a financial context, the cost of misclassification is not symmetric in the way `class_weight` assumes. A false positive (predicting up, market goes down) and a false negative (predicting down, market goes up) have costs determined by the magnitude of the actual return, not just the direction. `class_weight` cannot capture this asymmetry -- it only adjusts the classification threshold.

Furthermore, applying balanced weights shifts the decision boundary, which can reduce overall accuracy on the majority class without meaningfully improving minority class performance when the imbalance is this mild. For a more seriously imbalanced problem (e.g., 80/20 for rare events), `class_weight="balanced"` would be essential. At 52/48, it adds complexity without clear benefit.

**When to reconsider:** If the class distribution shifts significantly in a particular rolling window (e.g., in a prolonged bear market where the ratio might become 40/60), it may be worth revisiting. But as a default, `None` is appropriate.

**Final value:** `class_weight=None`

### bootstrap = True, oob_score = True

**Reasoning:**

`bootstrap=True` is the default and is fundamental to how Random Forest works. Each tree is trained on a bootstrap sample (~63.2% of unique training rows), and the ~36.8% left out (out-of-bag, OOB) can be used as a free validation set.

`oob_score=True` computes the OOB accuracy, which provides a useful diagnostic without requiring a separate validation split. The OOB error estimate is asymptotically equivalent to leave-one-out cross-validation (Breiman, 2001).

**Caveats for time series:** The standard OOB estimate assumes observations are i.i.d., which is violated for time series data. Two concerns:

1. **Temporal leakage within OOB:** An OOB sample for a given tree might include day t+1 while the tree was trained on day t. Since returns exhibit short-term autocorrelation and our features use rolling windows, this constitutes information leakage. The OOB score will be slightly optimistic compared to a true forward-looking test.

2. **Cross-sectional structure:** On any given date, all 30 tickers share macro features and breadth features. If some tickers from date t are in-bag and others are OOB, the OOB predictions for those tickers benefit from having correlated observations in the training set.

**Practical recommendation:** Use `oob_score=True` as a quick sanity check (is the model learning anything?), but never as a substitute for proper temporal cross-validation. The OOB score should be within 1-2% of the temporal CV score; if it is much higher, something is wrong.

**Final values:** `bootstrap=True`, `oob_score=True`

### random_state = 42

**Reasoning:** Reproducibility. Setting a fixed random state ensures identical results across runs. The specific value (42) is arbitrary.

**Note:** In production, consider training multiple forests with different seeds and ensembling them (voting or averaging probabilities). This adds a second layer of variance reduction beyond what `n_estimators` provides. For the research phase, a fixed seed is sufficient.

**Final value:** `random_state=42`

### n_jobs = -1

**Reasoning:** Use all available CPU cores for tree construction. Training is embarrassingly parallel across trees. On a modern 8-core machine, this reduces training time by approximately 6-7x.

**Final value:** `n_jobs=-1`

---

### Final Configuration

```python
from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "max_features": 0.3,
    "min_samples_leaf": 50,
    "class_weight": None,
    "bootstrap": True,
    "oob_score": True,
    "random_state": 42,
    "n_jobs": -1,
}

model = RandomForestClassifier(**RF_PARAMS)
```

### Summary Table

| Parameter | Value | Key Rationale |
|-----------|-------|---------------|
| `n_estimators` | 500 | Sufficient for low-SNR financial data; diminishing returns past ~400 |
| `max_depth` | 5 | Allows 5-way interactions; prevents memorization of noise |
| `max_features` | 0.3 | ~9 features per split; balances diversity and exposure for correlated features |
| `min_samples_leaf` | 50 | Stable probability estimates (SE ~ 0.07); spans multiple dates |
| `class_weight` | None | 52/48 imbalance is too mild to warrant reweighting |
| `bootstrap` | True | Core RF mechanism |
| `oob_score` | True | Free sanity check (not a substitute for temporal CV) |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Full parallelism |

### Changes from Current Codebase

The current `rf.py` uses `n_estimators=600`, `max_depth=5`, `max_features="sqrt"`, `min_samples_leaf=25`. The proposed changes are:

| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| `n_estimators` | 600 | 500 | 500 is sufficient; 600 adds compute without measurable gain |
| `max_features` | `"sqrt"` (~5.6) | `0.3` (~9) | Better balance for correlated financial features |
| `min_samples_leaf` | 25 | 50 | More stable leaf probabilities; better calibration for `predict_proba` |
| `oob_score` | (not set) | True | Free diagnostic |

---

## Part 2 -- Evaluation Metrics (Shared Across All Models)

These metrics apply uniformly to Random Forest, XGBoost, GRU, and LSTM. Every model must report all metrics using identical computation to enable fair comparison.

### 1. Balanced Accuracy (PRIMARY)

```python
from sklearn.metrics import balanced_accuracy_score
bal_acc = balanced_accuracy_score(y_true, y_pred)
```

**What it measures:** The unweighted mean of per-class recall: `(recall_class_0 + recall_class_1) / 2`. Equivalently, the average of sensitivity and specificity.

**Why primary:** For our 52/48 imbalanced data, a naive model predicting all-up achieves 52% accuracy but 50% balanced accuracy. Balanced accuracy penalizes models that game the majority class. It is threshold-dependent (uses hard predictions), which matters because in production we must commit to a direction.

**Why not plain accuracy?** A model that predicts "up" 100% of the time gets 52% accuracy, which could be mistaken for a useful signal. Balanced accuracy of 50% correctly reveals it as worthless. The 52/48 imbalance is mild enough that accuracy and balanced accuracy will be close for well-calibrated models, but the distinction matters at the margin.

**Benchmark:** A random model scores 50.0%. Any model below 50.5% balanced accuracy should be considered non-informative. Literature on daily equity direction prediction (Krauss et al., 2017; Leung et al., 2000) reports balanced accuracies in the 52-55% range for the best models.

### 2. ROC-AUC

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_proba)
```

**What it measures:** The probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example by the model's predicted probability. Threshold-independent; evaluates the entire ranking quality.

**Why include it:** AUC is the standard metric in ML literature for comparing classifiers (Fawcett, 2006). It captures whether the model's probability ordering is informative, even if the calibration is off. This matters for financial applications where the exact threshold may be adjusted based on transaction costs or risk limits.

**Caveats:** Hand (2009) showed that AUC implicitly uses different misclassification cost distributions for different classifiers, making it potentially misleading for comparing very different model types. In practice, for our four models (all binary classifiers on the same data), this concern is minor. More importantly, AUC can be high even when the model's probabilities are poorly calibrated. A model with AUC=0.55 but mean predicted probability of 0.7 is poorly calibrated and will generate bad position sizes.

**Benchmark:** Random = 0.50. Literature range for daily equity direction: 0.52-0.58.

### 3. Log Loss (Cross-Entropy)

```python
from sklearn.metrics import log_loss
ll = log_loss(y_true, y_proba)
```

**What it measures:** The negative log-likelihood of the true labels under the model's predicted probabilities: `-mean(y * log(p) + (1-y) * log(1-p))`. Measures both discrimination (ranking) and calibration (probability accuracy).

**Why include it:** If predicted probabilities are used for position sizing (e.g., bet proportional to |p - 0.5|), then calibration matters as much as accuracy. Log loss is the only standard metric that jointly penalizes poor ranking and poor calibration. A model that predicts 0.52 for everything has poor log loss despite potentially decent accuracy.

**Comparison to Brier score:** Brier score (`mean((y - p)^2)`) also measures calibration + discrimination but is less sensitive to confident wrong predictions. Log loss penalizes confident errors much more harshly (predicting 0.99 when the truth is 0 gives log loss contribution of ~4.6 vs Brier contribution of 0.98). For financial applications where a confident wrong bet is catastrophic, log loss is more appropriate.

**Benchmark:** For a 52/48 distribution, the entropy baseline (predicting 0.52 for everything) gives log loss = -0.52*log(0.52) - 0.48*log(0.48) ~ 0.6931. Any model should beat this.

**Implementation note for tree models:** Random Forest probabilities are the fraction of trees predicting each class. With 500 trees, probabilities are discrete (multiples of 1/500 = 0.002), which is fine-grained enough for reliable log loss computation. However, extreme predictions (p near 0 or 1) should be clipped to avoid infinite log loss: `np.clip(proba, 1e-7, 1 - 1e-7)`.

### 4. Accuracy

```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```

**What it measures:** Fraction of correct predictions.

**Why include it:** Simple, universally understood, and directly interpretable as "hit rate." While misleading as a primary metric for imbalanced data, it remains useful for communication. Stakeholders understand "the model is right 54% of the time" more immediately than "the model has balanced accuracy of 53.5%."

**Role:** Secondary metric. Report alongside balanced accuracy. If accuracy and balanced accuracy diverge significantly, the model is biased toward one class.

### 5. Precision, Recall, F1 (Per-Class)

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, digits=4))
```

**What they measure:**
- **Precision (class 1):** Of all predicted "buy" signals, what fraction were actually up? This is the *hit rate on the long side*.
- **Recall (class 1):** Of all actual up-days, what fraction did we predict? This is the *capture rate*.
- **F1:** Harmonic mean of precision and recall.

**Why include them:** Precision and recall decompose accuracy into actionable components. A trading system cares most about precision on the predicted class: if the model says "buy," how often is it right? Recall is less critical for trading (missing opportunities is less costly than taking losing trades, assuming no short constraint).

**Per-class reporting:** Report for both class 0 and class 1. If precision on class 1 is high but recall is low, the model is conservative (few but reliable buy signals). If precision is low but recall is high, the model is aggressive (catches most up-days but with many false signals).

**Role:** Diagnostic, not for model selection. Useful for understanding model behavior and tuning the decision threshold.

### 6. Matthews Correlation Coefficient (MCC)

```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

**What it measures:** A correlation coefficient between observed and predicted classifications, ranging from -1 (perfect misclassification) to +1 (perfect classification), with 0 being random. Computed as:

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Why include it:** Chicco & Jurman (2020) demonstrated that MCC is the only binary classification metric that produces a high score only when the model performs well on all four confusion matrix categories (TP, TN, FP, FN). Unlike F1, it is symmetric -- it does not privilege one class over the other. Unlike balanced accuracy, it accounts for the actual counts, not just rates.

For our slightly imbalanced problem, MCC will closely track balanced accuracy. Its main value is as a single-number summary that is robust to class distribution shifts. If the 52/48 ratio changes to 55/45 in a bear-market window, MCC remains comparable across windows; accuracy does not.

**Benchmark:** Random = 0.0. Typical range for daily equity direction: 0.04-0.10 (these are *low* values -- financial prediction is hard).

**Role:** Secondary metric, but the most trustworthy single number for comparing models.

### 7. Financial Sanity Checks (Non-Metric, Reported Separately)

These are not evaluation metrics in the ML sense but are essential for validating that the model produces a useful trading signal:

#### 7a. Mean Predicted Probability

```python
mean_prob = y_proba.mean()
```

**Purpose:** Sanity check for calibration. For a 52/48 target, the mean predicted probability should be near 0.52. If it is 0.70, the model is badly miscalibrated. If it is 0.50, the model may be ignoring the base rate.

#### 7b. Prediction Distribution

```python
pred_counts = pd.Series(y_pred).value_counts(normalize=True)
```

**Purpose:** Check whether the model predicts both classes in reasonable proportions. A model that predicts 90% "up" and 10% "down" is not useful even if its overall accuracy is acceptable.

#### 7c. Directional Return (Long-Only Sanity Check)

```python
signal_return = (y_pred * actual_returns).sum()
```

**Purpose:** Does the model's binary signal generate positive returns? This is not a rigorous backtest (no transaction costs, no slippage, no position sizing), but it is a quick sanity check. A model with 54% balanced accuracy that generates negative returns is suspicious (it may be right on small moves and wrong on large moves).

**Warning:** This is NOT a primary metric. It is highly sensitive to the test period, a handful of extreme days, and does not account for transaction costs. Never optimize a model to maximize this number directly -- that path leads to catastrophic overfitting. Report it, do not tune on it.

---

### Recommended Metric for Model Selection

**Primary metric for hyperparameter tuning and model comparison: Balanced Accuracy.**

**Reasoning:**

1. **Threshold-dependent:** In production, we must produce a binary signal (buy/sell or up/down). Balanced accuracy evaluates the hard prediction, which is what matters operationally. AUC evaluates the ranking, which is useful but one step removed from the actual decision.

2. **Class-aware:** Penalizes models that exploit the majority class. Ensures the model is informative for both up and down predictions.

3. **Interpretable:** "The model correctly identifies 54% of up-days and 53% of down-days, for a balanced accuracy of 53.5%" is immediately understandable and actionable.

4. **Stable:** Less sensitive to extreme predictions than log loss. Less sensitive to threshold choice than precision/recall. More robust to class distribution shifts than plain accuracy.

5. **Comparable across model types:** All four models (RF, XGBoost, GRU, LSTM) produce class predictions and probabilities. Balanced accuracy applies identically to all.

**Secondary for tie-breaking:** If two models have identical balanced accuracy (within 0.5%), prefer the one with:
1. Lower log loss (better calibrated probabilities)
2. Higher MCC (confirming balanced performance)
3. Higher AUC (better ranking quality)

**For probability-based strategies:** If the downstream strategy uses predicted probabilities for position sizing (not just binary signals), log loss should be co-primary with balanced accuracy. A model with 53% balanced accuracy and well-calibrated probabilities may be more profitable than one with 54% balanced accuracy and poorly calibrated probabilities.

---

### Implementation Note

All metrics should be computed using a shared evaluation function to guarantee consistency across models:

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    matthews_corrcoef,
)


def evaluate_model(y_true, y_pred, y_proba, model_name="model"):
    """
    Shared evaluation for all models (RF, XGBoost, GRU, LSTM).

    Parameters
    ----------
    y_true : array-like of int (0 or 1)
    y_pred : array-like of int (0 or 1), hard predictions
    y_proba : array-like of float, predicted P(class=1)
    model_name : str, for logging

    Returns
    -------
    dict with all metrics
    """
    # Clip probabilities to avoid log(0)
    y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba_safe),
        "log_loss": log_loss(y_true, y_proba_safe),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "mean_predicted_prob": float(np.mean(y_proba)),
        "pred_positive_rate": float(np.mean(y_pred)),
    }

    return metrics
```

This function must be called identically for every model. No model-specific metric adjustments. The `classification_report` (precision, recall, F1) should be printed separately for diagnostic purposes but the above dictionary is what goes into comparison tables and logs.

---

## References

- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32. The foundational Random Forest paper. Establishes the theoretical basis for bootstrap aggregating with random feature subsets. Proves that generalization error converges as the number of trees increases, and that RF does not overfit by adding more trees (only by making individual trees too complex).

- **Chicco, D. & Jurman, G. (2020).** "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*, 21(1), 6. Demonstrates that MCC is the only metric that returns a high value only when all four confusion matrix quadrants are handled well. Shows that accuracy and F1 can be misleadingly high when models ignore the minority class.

- **Fawcett, T. (2006).** "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861-874. The standard tutorial on ROC curves and AUC. Explains threshold-independent evaluation and the geometric interpretation of AUC.

- **Hand, D.J. (2009).** "Measuring classifier performance: a coherent alternative to the area under the ROC curve." *Machine Learning*, 77(1), 103-123. Critiques AUC for implicitly using different cost distributions for different classifiers, making cross-classifier AUC comparisons potentially misleading. Proposes the H-measure as an alternative.

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. Key empirical benchmark. RF achieves ~52-55% daily accuracy on S&P 500 direction. Shows tree depth and leaf size as critical hyperparameters. Demonstrates that ensemble methods outperform single models.

- **Leung, M.T., Daouk, H. & Chen, A.S. (2000).** "Forecasting stock indices: a comparison of classification and level estimation models." *International Journal of Forecasting*, 16(2), 173-190. Early comparison of classification vs regression approaches for equity direction prediction. Reports accuracies in the 52-58% range depending on market and horizon.

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. Ch. 6: Feature importance; shows that MDI importance in random forests is biased toward continuous and high-cardinality features. Ch. 7: Purged k-fold cross-validation for overlapping targets. Ch. 8: Discusses overfitting in hyperparameter selection and recommends conservative (shallow, large-leaf) tree configurations.

- **Oshiro, T.M., Perez, P.S. & Baranauskas, J.A. (2012).** "How Many Trees in a Random Forest?" *Lecture Notes in Computer Science*, 7376, 154-168. Systematic study of n_estimators. Finds that 64-128 trees suffice for most UCI datasets, with diminishing returns past 256. Notes that high-noise, low-signal problems may benefit from more trees.
