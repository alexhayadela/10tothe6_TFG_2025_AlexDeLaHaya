# XGBoost -- Design Decisions

**Date:** 2026-03-30
**Task:** Binary stock direction classification (h=1, next-day up/down)
**Data:** 30 IBEX35 stocks, ~5000 trading days each, ~39 features (micro + cross breadth + macro)
**Training scheme:** 3-year sliding window (~750 days x 30 tickers = ~22,500 training rows per window)
**Evaluation metrics:** Shared with RF -- see `decisions/rf_decisions.md`, Part 2

---

## Comparison to Random Forest

Random Forest and XGBoost are both tree ensembles, but they differ in how they combine trees and where their failure modes lie.

**Random Forest** builds trees independently (bagging). Each tree sees a bootstrap sample of the data and a random subset of features at each split. Trees are grown deep and are individually high-variance but low-bias. The ensemble averages them to reduce variance. The key property: *adding more trees cannot cause overfitting* (Breiman, 2001). The ensemble converges monotonically.

**XGBoost** builds trees sequentially (boosting). Each tree fits the *residual errors* of the current ensemble. Trees are deliberately kept shallow (weak learners), and the ensemble reduces bias by iteratively correcting mistakes. The key risk: *each new tree can overfit the residuals*, especially in later rounds when the true signal has been captured and only noise remains (Friedman, 2001).

### Where XGBoost has an advantage

1. **Bias reduction.** RF relies solely on tree depth to control bias. If the true decision boundary requires combining information in a way that a single tree of depth 5 cannot represent, RF cannot learn it no matter how many trees are added. XGBoost can learn it by accumulating corrections across rounds. For financial data with weak, distributed signals, this additive correction can extract marginally more signal.

2. **Explicit regularization.** XGBoost has L1 and L2 penalties on leaf weights, plus learning rate shrinkage. RF has no direct regularization on leaf values -- it relies entirely on structural constraints (depth, leaf size). This makes XGBoost more tunable for controlling overfitting.

3. **Feature interaction discovery.** Through sequential residual fitting, XGBoost can effectively learn higher-order interactions even with shallow trees. A depth-3 tree in round 1 might split on feature A, and a depth-3 tree in round 50 might split on feature B in the region where A was already partitioned. The cumulative effect is a deeper interaction than any individual tree encodes.

4. **Handling of heterogeneous signal strength.** If some features carry stronger signals than others, XGBoost will naturally focus on the strong features early (reducing the largest residuals first), then gradually incorporate weaker features. RF treats all features more democratically due to random subsampling.

### Where XGBoost is more fragile

1. **Overfitting to noise in late rounds.** After the genuine signal is captured (possibly by round 50-100), subsequent trees fit noise. This is the central failure mode for financial data. Early stopping is essential (see below).

2. **Sensitivity to hyperparameters.** RF is relatively robust to hyperparameter choices -- a depth-5 forest with 500 trees will perform reasonably across many datasets. XGBoost has more hyperparameters with stronger interactions (learning rate x n_estimators x depth), and poor combinations can catastrophically overfit or underfit.

3. **Non-stationarity.** Boosting assumes the training distribution is representative of the test distribution. In financial data with regime changes, the residuals being fit in later rounds may reflect patterns specific to the training period that reverse out-of-sample. RF's independent trees are more robust to this because no single tree dominates the ensemble.

4. **Probability calibration.** XGBoost's raw probabilities from `predict_proba` tend to be less well-calibrated than RF's (Niculescu-Mizil & Caruana, 2005), though the logistic objective helps. If probabilities are used for position sizing, post-hoc calibration (Platt scaling or isotonic regression) may be needed.

### Expected outcome

For daily equity direction prediction, Krauss et al. (2017) found gradient-boosted trees performed comparably to random forests on S&P 500 data, with neither consistently dominating. The advantage of XGBoost, when it exists, is typically 0.5-1.0 percentage points in accuracy. The risk is that a poorly regularized XGBoost will underperform a well-tuned RF by 2-3 points due to overfitting. Conservative regularization is therefore paramount.

---

## Hyperparameters

### n_estimators = 300 (with early stopping)

**Reasoning:** In boosting, `n_estimators` is an upper bound, not a target. Unlike RF, where adding more trees is always safe, each additional boosting round risks overfitting. The actual number of rounds used will be determined by early stopping (see below).

We set `n_estimators=300` as the ceiling. With `learning_rate=0.05`, 300 rounds gives a maximum total shrinkage contribution of 300 x 0.05 = 15 (in additive log-odds space). This is more than sufficient for a binary classification problem where the true signal is weak.

**Why not 500 or 1000?** Krauss et al. (2017) used 500 rounds on S&P 500 data with a larger training set. With our 22,500-row windows and low SNR, early stopping typically triggers between 80 and 200 rounds. Setting the ceiling at 300 provides headroom without wasting compute when early stopping fails to trigger (which would itself be a warning sign of an incorrect validation setup).

**Why not 100?** At `learning_rate=0.05`, 100 rounds may be insufficient to capture the full signal. The model may underfit, especially in regimes where the signal is slightly stronger than average.

**Final value:** `n_estimators=300` (with early stopping; see Early Stopping Strategy below)

### learning_rate (eta) = 0.05

**Reasoning:** The learning rate (eta) scales the contribution of each new tree. Smaller values mean each tree contributes less, requiring more rounds to reach the same model complexity, but with more granular control over when to stop.

The core trade-off is:
- **Large eta (0.1-0.3):** Each tree has a large effect. The model converges quickly but overshoots easily. Early stopping must trigger at exactly the right round, and a few extra rounds cause significant overfitting. This is dangerous for financial data where the validation signal is noisy.
- **Small eta (0.01-0.03):** Each tree barely changes the model. Convergence is slow (needs 500+ rounds), which wastes compute. The model is robust to stopping a few rounds late, but the training time in a rolling-window scheme becomes prohibitive.
- **Moderate eta (0.05):** A good balance. Requires ~100-200 rounds to converge, each round has a small but meaningful effect, and the early stopping curve is smooth enough to identify the optimum reliably.

Friedman (2001) showed that shrinkage (small learning rate) consistently improves generalization in gradient boosting, and recommended values in the 0.01-0.1 range. For tabular financial data, 0.05 is a standard choice in the literature (Krauss et al., 2017; Gu et al., 2020).

**Interaction with n_estimators:** The product `learning_rate x n_estimators` determines the model's effective capacity. At 0.05 x 300 = 15, we have sufficient capacity. If we used eta=0.01, we would need n_estimators=1500 for the same capacity, which is 5x slower.

**Final value:** `learning_rate=0.05`

### max_depth = 3

**Reasoning:** This is the most important difference from our RF configuration (which uses `max_depth=5`). Boosted trees should be shallower than RF trees for three reasons:

1. **Boosting reduces bias through iteration, not tree depth.** In RF, each tree must individually model as much of the signal as possible, because trees are averaged (not corrected). In XGBoost, a shallow tree in round 1 captures the main effect, and subsequent trees correct its errors. The cumulative model after 150 rounds of depth-3 trees can represent interactions far more complex than any single depth-3 tree.

2. **Shallow trees are weaker learners, which is what boosting needs.** Friedman (2001) showed that boosting works best when individual learners are "weak but not too weak." Depth 3 gives at most 8 leaf nodes with 3-way interactions. This is weak enough that each tree cannot memorize noise, but strong enough to capture meaningful conditional patterns (e.g., "RSI < 30 AND vol elevated AND breadth declining").

3. **Deeper trees amplify the overfitting risk in later rounds.** A depth-6 tree can create 64 leaf nodes, each potentially fitting noise in the residuals. With learning_rate=0.05, each leaf's contribution is small, but over 200 rounds, the accumulated noise from 200 x 64 leaf weights can overwhelm the signal. Depth 3 limits this to 200 x 8 = 1,600 leaf weights, a much more constrained model.

**Empirical support:** Chen & Guestrin (2016) report that depth 3-6 is typical for most applications, with depth 4-6 for complex tabular problems. For low-SNR financial data, depth 3 is the conservative and standard choice.

**Final value:** `max_depth=3`

### min_child_weight = 100

**Reasoning:** `min_child_weight` in XGBoost is the minimum sum of instance weights (hessians) in a child node. For binary logistic loss with uniform sample weights, the hessian for each observation is `p(1-p)`, which ranges from 0 to 0.25 (maximized at p=0.5). For early rounds where predictions are near 0.5, `min_child_weight=100` is approximately equivalent to requiring 100/0.25 = 400 observations per leaf. As the model becomes more confident in later rounds, the effective minimum sample count drops, but never below 100 observations (when p approaches 0 or 1, the hessian approaches 0, so more observations are needed to reach the weight threshold).

The rationale mirrors `min_samples_leaf=50` in our RF configuration, but is set higher for two reasons:

1. **Boosting amplifies leaf-level noise.** In RF, a noisy leaf estimate in one tree is diluted by 499 other independent trees. In XGBoost, a noisy leaf estimate in round k directly biases the residuals for rounds k+1, k+2, etc. The error compounds. Requiring more observations per leaf reduces the leaf-level noise that feeds into subsequent rounds.

2. **Cross-stock contamination.** With 30 tickers per date, a leaf with only 20 observations might contain data from a single date. The tree would be fitting a date-specific effect, which then gets baked into all subsequent trees' residuals. With min_child_weight=100, each leaf must span multiple dates.

**Why not 50?** min_child_weight=50 would give ~200 observations per leaf in early rounds, which is reasonable. But in later rounds when hessians are smaller, leaves could contain fewer than 50 observations, which is insufficient for financial data. 100 provides a comfortable floor.

**Why not 200?** This would over-constrain the depth-3 trees. With 22,500 training rows and min_child_weight=200, each leaf needs ~800 observations in early rounds, limiting the tree to effectively 22,500/800 ~ 28 leaves maximum -- but depth 3 only allows 8 leaves, so the constraint is not actually binding at 200. However, in later rounds when some regions have small hessians, the weight constraint would aggressively prune the trees, potentially preventing them from correcting real errors.

**Final value:** `min_child_weight=100`

### subsample = 0.7

**Reasoning:** Row subsampling (stochastic gradient boosting) trains each tree on a random fraction of the training data. This serves the same purpose as bootstrap sampling in RF: reducing correlation between trees and improving generalization.

Friedman (2002) introduced stochastic gradient boosting and found that subsampling 50-80% of the data typically improves generalization. The mechanism is straightforward: by training each tree on a different subset, the model is less likely to memorize patterns that only exist in specific observations.

For our setting:
- **subsample=0.7** means each tree sees 70% of the 22,500 rows = ~15,750 rows. The remaining 30% are held out per tree. This is aggressive enough to provide meaningful regularization.
- **Why not 0.5?** With only 15,750 rows, halving to ~11,250 per tree risks losing signal. Depth-3 trees with 8 leaves and min_child_weight=100 need at least ~3,200 rows to fill all leaves. At 0.5, the margin is thinner.
- **Why not 0.8 or 1.0?** Less regularization. With the other regularization mechanisms (learning rate, depth, L2 penalty), 0.8 is defensible but 0.7 provides a stronger guard against the specific failure mode of overfitting residuals from previous rounds.

**Final value:** `subsample=0.7`

### colsample_bytree = 0.7

**Reasoning:** Column subsampling per tree serves the same purpose as `max_features` in RF: forcing different trees to use different features, reducing the ensemble's reliance on any single feature.

With 39 features and `colsample_bytree=0.7`, each tree sees ~27 features. At depth 3, each tree uses at most 7 split features (one per internal node in a complete binary tree of depth 3, which has 2^3 - 1 = 7 internal nodes), so 27 candidate features per tree provides ample choice.

**Why bytree rather than bylevel?** `colsample_bylevel` resamples features at each depth level within a tree. For depth-3 trees with only 3 levels, the per-level sampling is too coarse to be effective. `colsample_bytree` (one random subset per tree) provides more meaningful diversification across the 300 trees.

**Why 0.7 rather than 0.3 (matching RF's max_features)?** In RF, low `max_features` is critical because each tree is deep and will overfit its feature subset. In XGBoost, trees are shallow (depth 3) and already regularized by learning rate and L2 penalty. Aggressive feature subsampling on top of this would handicap individual trees too much, making them too weak to provide useful corrections. 0.7 provides mild diversification without crippling individual tree quality.

**Final value:** `colsample_bytree=0.7`

### reg_lambda (L2) = 1.0

**Reasoning:** `reg_lambda` adds an L2 penalty to the leaf weights (the predicted values in each leaf). The loss function becomes:

```
L = sum(loss(y_i, hat{y}_i)) + lambda * sum(w_j^2)
```

where `w_j` are the leaf weights. This shrinks leaf weights toward zero, which has two effects:

1. **Reduces overfitting in individual trees.** A leaf with few observations (near the `min_child_weight` boundary) will have its weight pulled toward zero more strongly than a leaf with many observations. This is a form of Bayesian shrinkage.

2. **Smooths the ensemble's predictions.** Since each tree's contribution is smaller, the ensemble relies on agreement across many trees rather than a few trees with large weights.

The default in XGBoost is `reg_lambda=1.0`, which Chen & Guestrin (2016) found to work well across a wide range of problems. For financial data with low SNR, this default is appropriate -- it provides meaningful regularization without requiring tuning.

**Why not increase to 5 or 10?** Higher L2 penalty effectively reduces the learning rate further (since leaf weights are shrunk). With `learning_rate=0.05` already providing shrinkage, doubling the regularization through L2 would slow convergence without a clear benefit. If overfitting is still observed despite all other regularization (learning rate, depth, subsampling, early stopping), increasing `reg_lambda` to 3-5 is the first lever to try.

**Final value:** `reg_lambda=1.0`

### reg_alpha (L1) = 0.0

**Reasoning:** `reg_alpha` adds an L1 penalty to leaf weights, which promotes sparsity (some leaf weights become exactly zero, effectively pruning subtrees). This is useful when:

- The tree has many leaves with near-zero contribution (common with deep trees).
- You want the model to automatically deactivate irrelevant subtrees.

With `max_depth=3` (at most 8 leaves), sparsity among 8 leaf weights is not a meaningful objective. The trees are already sparse by construction. Adding L1 on top of L2 adds complexity to the optimization without a clear benefit.

**When to reconsider:** If `max_depth` is increased to 5-6 in future experiments, L1 regularization becomes more valuable for pruning the additional leaves.

**Final value:** `reg_alpha=0.0`

### scale_pos_weight = 1.0

**Reasoning:** Same logic as `class_weight=None` in the RF configuration. The class distribution is approximately 52/48, which is too mild to warrant reweighting. Setting `scale_pos_weight = n_negative / n_positive ~ 0.92` would barely adjust the loss function and could introduce instability in the gradient computation without meaningful benefit.

**Final value:** `scale_pos_weight=1.0` (default, no reweighting)

### objective = "binary:logistic"

**Reasoning:** This trains the model to minimize logistic loss (cross-entropy), outputting calibrated probabilities via the sigmoid function. This is the correct choice for binary classification when we need probability estimates for downstream position sizing.

**Why not "binary:hinge"?** Hinge loss (SVM-style) produces hard classifications without probability estimates. We need `predict_proba` for our evaluation metrics (log loss, AUC) and for probability-based position sizing.

**Final value:** `objective="binary:logistic"`

### eval_metric = "logloss"

**Reasoning:** This is the metric monitored during training for early stopping. We use `"logloss"` (binary cross-entropy on the validation set) rather than `"error"` (misclassification rate) because:

1. **Sensitivity.** Log loss detects overfitting earlier than accuracy. A model that starts producing overconfident probabilities (0.9 instead of 0.55) will show rising log loss before accuracy degrades, because accuracy only changes when the prediction crosses the 0.5 threshold.

2. **Consistency with objective.** The training objective is logistic loss. Monitoring the same loss on validation ensures we are tracking the same quantity. Using a different metric (e.g., AUC) for early stopping can lead to inconsistencies where training loss decreases but the monitored metric fluctuates.

3. **Alignment with our evaluation framework.** Log loss is one of our secondary metrics (see `rf_decisions.md`). Optimizing it during training ensures the model's probabilities are as well-calibrated as XGBoost can make them.

**Note:** The *model selection* metric across rolling windows remains balanced accuracy (see rf_decisions.md). `eval_metric` here is only for within-training early stopping.

**Final value:** `eval_metric="logloss"`

### tree_method = "hist"

**Reasoning:** The "hist" method uses histogram-based split finding, which buckets continuous features into discrete bins before finding optimal splits. This is:

1. **Faster.** O(n_bins x n_features) per split instead of O(n x n_features) for exact. With 22,500 rows and 39 features, the speedup is meaningful across 300 rounds.
2. **Regularizing.** Binning introduces a mild quantization that prevents splits on exact thresholds that may be noise-driven. This is a subtle but beneficial form of regularization for financial data.

Chen & Guestrin (2016) showed that histogram approximation introduces negligible accuracy loss compared to exact split finding, while providing 2-10x speedup.

**Final value:** `tree_method="hist"`

### random_state = 42

**Reasoning:** Reproducibility, consistent with the RF configuration.

**Final value:** `random_state=42`

### n_jobs = -1

**Reasoning:** XGBoost parallelizes across features (not trees, since boosting is sequential). With 39 features, parallelism across 8+ cores provides a meaningful speedup for split finding within each round.

**Final value:** `n_jobs=-1`

---

### Final Configuration

```python
import xgboost as xgb

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "scale_pos_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# Early stopping is applied at fit time, not in the constructor
EARLY_STOPPING_ROUNDS = 30

model = xgb.XGBClassifier(**XGB_PARAMS)

# At training time (within each rolling window):
# X_train, X_val = ... (see Early Stopping Strategy below)
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=False,
# )
```

### Summary Table

| Parameter | Value | Key Rationale |
|-----------|-------|---------------|
| `n_estimators` | 300 | Upper bound; actual rounds determined by early stopping |
| `learning_rate` | 0.05 | Moderate shrinkage; balances convergence speed and overfitting control |
| `max_depth` | 3 | Shallow weak learners; boosting accumulates complexity across rounds |
| `min_child_weight` | 100 | Stable leaf estimates; prevents date-specific fitting |
| `subsample` | 0.7 | Stochastic boosting; decorrelates sequential trees |
| `colsample_bytree` | 0.7 | Feature diversification; mild enough for shallow trees |
| `reg_lambda` (L2) | 1.0 | Default; shrinks leaf weights; first lever for additional regularization |
| `reg_alpha` (L1) | 0.0 | Not needed with depth 3; reconsider if depth increases |
| `scale_pos_weight` | 1.0 | 52/48 imbalance too mild to warrant adjustment |
| `objective` | `binary:logistic` | Outputs calibrated probabilities |
| `eval_metric` | `logloss` | Sensitive to overfitting; consistent with training objective |
| `tree_method` | `hist` | Faster and mildly regularizing |
| `early_stopping_rounds` | 30 | See Early Stopping Strategy |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Full parallelism |

### Comparison with RF Configuration

| Aspect | RF | XGBoost | Reason for Difference |
|--------|----|---------|-----------------------|
| n_estimators | 500 (all used) | 300 (ceiling, ~80-200 used) | Boosting overfits with too many rounds; RF does not |
| Tree depth | 5 | 3 | Boosting compensates for shallow trees via iteration |
| Leaf constraint | min_samples_leaf=50 | min_child_weight=100 | Boosting compounds leaf-level noise; needs stricter control |
| Feature subsampling | 0.3 (aggressive) | 0.7 (mild) | Deep RF trees need diversification; shallow XGB trees need quality |
| Explicit regularization | None | L2=1.0, learning_rate=0.05 | XGBoost advantage; RF relies on structural constraints |
| Early stopping | Not applicable | 30 rounds patience | Essential for boosting; unnecessary for bagging |
| Training time (est.) | ~5-10 sec per window | ~3-8 sec per window | XGBoost uses fewer effective rounds; both fast enough |

---

## Early Stopping Strategy

Early stopping is the most important regularization mechanism for XGBoost on financial data. It determines the actual number of boosting rounds by monitoring performance on a held-out validation set and stopping when performance degrades.

### Why early stopping is essential

Without early stopping, XGBoost will use all 300 rounds. In the first ~50-100 rounds, the model captures the genuine (weak) signal. In rounds 100-300, it fits noise in the training data, creating an increasingly complex model that performs worse out-of-sample. This is not a theoretical concern -- it is the dominant failure mode for gradient boosting on low-SNR financial data.

### Validation set design

**Approach: temporal split within the training window.**

Given a 3-year training window (days 1 to 750, ~22,500 rows), we split as follows:

- **Training set:** Days 1 to 600 (~18,000 rows, first ~2.4 years)
- **Validation set:** Days 601 to 750 (~4,500 rows, last ~0.6 years)

The validation set is the most recent portion of the training window, immediately preceding the test period. This preserves temporal ordering and avoids look-ahead bias.

**Why temporal split, not random split?**

A random split would scatter future observations into the training set and past observations into the validation set. This leaks future information into training (via cross-sectional correlations and shared macro features) and evaluates on stale data. For financial time series, any validation scheme must respect temporal ordering (Lopez de Prado, 2018, Ch. 7).

**Why not use the test set for early stopping?**

This would constitute information leakage. The test set (the next rolling window's out-of-sample period) must remain completely unseen during training and hyperparameter selection. Early stopping is a form of hyperparameter selection (choosing n_estimators), so it must use data strictly before the test period.

**Does the validation split introduce look-ahead bias?**

No, because the validation set (days 601-750) is always before the test set (days 751+). However, it does reduce the effective training data by 20%. This is a real cost: the model sees 2.4 years instead of 3 years of history. The regularization benefit of early stopping outweighs this cost, but if training windows are made shorter in the future (e.g., 1-year windows), the validation fraction should be reduced to 10-15%.

**Alternative considered: purged walk-forward within the window.** We could run multiple train/validation splits within the 3-year window (e.g., train on years 1-1.5, validate on 1.5-2; train on years 1-2, validate on 2-2.5; etc.) and average the optimal round. This is more robust but 3-4x slower and adds implementation complexity. For the research phase, a single temporal split is sufficient. The purged approach can be revisited if early stopping shows high variance across rolling windows.

### early_stopping_rounds = 30

**Reasoning:** This is the "patience" parameter -- how many consecutive rounds without improvement on the validation set before stopping.

- **Too low (5-10):** Validation log loss is noisy, especially with 4,500 observations where the true signal is weak. A run of 5 non-improving rounds is common even when the model is still learning. Premature stopping leaves signal on the table.
- **Too high (50-100):** Allows significant overfitting before stopping. If the true optimum is at round 80, patience of 100 means the model continues to round 180 before stopping, and the returned model is the round-80 model. This wastes compute but does not cause harm (XGBoost returns the best iteration, not the last). However, patience of 100 with n_estimators=300 means early stopping rarely triggers, defeating its purpose.
- **30 rounds at learning_rate=0.05:** Each round adds a small contribution (scaled by 0.05). Thirty rounds of non-improvement means 30 x 0.05 = 1.5 units of additive log-odds without validation improvement. This is long enough to ride through noise but short enough to stop well before serious overfitting.

**Implementation:**

```python
# Within the rolling-window training loop:

# 1. Split the training window temporally
split_idx = int(len(X_window) * 0.80)  # 80/20 temporal split
X_train = X_window[:split_idx]
y_train = y_window[:split_idx]
X_val = X_window[split_idx:]
y_val = y_window[split_idx:]

# 2. Fit with early stopping
model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# 3. Record the actual number of rounds used
actual_rounds = model.best_iteration + 1  # 0-indexed

# 4. Predict on the test set (next rolling window)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
```

### Diagnostics

Track and report across rolling windows:
- **Mean best_iteration:** If consistently near 300 (the ceiling), increase n_estimators.
- **Std of best_iteration:** High variance (e.g., std > 50) suggests the validation signal is too noisy; consider increasing the validation set or patience.
- **Validation logloss at best_iteration vs. at round 300:** The gap indicates how much overfitting early stopping prevented.

---

## Overfitting Guardrails -- Summary

XGBoost on financial data has six layers of defense against overfitting, listed from most to least important:

1. **Early stopping** (stops the model from fitting noise in later rounds)
2. **Learning rate = 0.05** (limits each tree's contribution, requiring consensus across many trees)
3. **max_depth = 3** (each tree is a weak learner that cannot memorize patterns)
4. **min_child_weight = 100** (each leaf is based on sufficient observations)
5. **subsample = 0.7 + colsample_bytree = 0.7** (stochastic regularization)
6. **reg_lambda = 1.0** (L2 shrinkage on leaf weights)

If the model still overfits (validation balanced accuracy >> test balanced accuracy by more than 3 points consistently), the escalation order is:

1. Reduce `n_estimators` ceiling (from 300 to 200)
2. Increase `min_child_weight` (from 100 to 200)
3. Reduce `learning_rate` (from 0.05 to 0.02, and increase ceiling to 500)
4. Increase `reg_lambda` (from 1.0 to 5.0)
5. Reduce `subsample` (from 0.7 to 0.5)

---

## References

- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32. Establishes that RF ensembles do not overfit with more trees -- a property that boosting does not share. Provides the key contrast for understanding why XGBoost requires early stopping.

- **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. The foundational XGBoost paper. Introduces the regularized objective (L1/L2 on leaf weights), the histogram-based split finding algorithm, and the column subsampling extension of stochastic gradient boosting. Reports state-of-the-art results across diverse tabular benchmarks.

- **Friedman, J.H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232. The theoretical foundation for gradient boosting. Introduces the framework of fitting residuals sequentially and proves convergence properties. Recommends small learning rates (shrinkage) for improved generalization.

- **Friedman, J.H. (2002).** "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367-378. Introduces row subsampling (the `subsample` parameter) to gradient boosting. Shows that training each tree on a random subset improves generalization, analogous to bootstrap sampling in bagging.

- **Gu, S., Kelly, B. & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273. Large-scale comparison of ML methods for financial prediction. Gradient-boosted trees are among the best performers for cross-sectional return prediction. Confirms that moderate depth (3-5) and conservative learning rates are optimal for financial applications.

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. Direct comparison of RF, gradient boosting, and deep networks on daily equity direction. Gradient-boosted trees perform comparably to RF, with neither consistently dominating. Uses 500 rounds with early stopping on S&P 500 data.

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. Ch. 6: Feature importance in tree ensembles. Ch. 7: Purged k-fold cross-validation -- essential reading for avoiding temporal leakage in financial ML. Ch. 8: Overfitting in hyperparameter selection. Recommends conservative tree configurations and warns against optimistic validation estimates.

- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting Good Probabilities with Supervised Learning." *Proceedings of the 22nd International Conference on Machine Learning*, 625-632. Compares probability calibration across classifiers. Boosted trees tend to produce less calibrated probabilities than bagged trees (RF), though logistic objective partially compensates. Recommends Platt scaling for post-hoc calibration of boosted models.
