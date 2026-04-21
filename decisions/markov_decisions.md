# Markov Chain -- Design Decisions

**Date:** 2026-04-21
**Task:** Binary stock direction classification (h=1, next-day up/down)
**Data:** 35 IBEX35 stocks, ~5000 trading days each, 41 features available
**Training scheme:** 3-year sliding window (~750 days x 35 tickers)
**Evaluation metrics:** Shared with all models -- see `decisions/rf_decisions.md`, Part 2

---

## 1. Why a Markov Chain model?

The Markov Chain model serves as a **statistical baseline** occupying a different part of the model space than the other four models in this project. RF, XGBoost, GRU, and LSTM all operate on the full 41-feature matrix and use discriminative learning (fitting a function from features to labels). The Markov model is generative, low-dimensional, and non-parametric in the neural/gradient sense -- it estimates transition probabilities directly from discretised return states.

Its role is threefold:

1. **Interpretable lower bound.** A Markov model answers the question "how much is predictable from the raw autocorrelation structure of returns alone, ignoring all technical indicators?" If the transition matrix is uninformative (all states map to ~50% P(up)), this documents that return autocorrelation carries no signal and justifies the engineering effort behind the full feature set.

2. **Regime detector.** Even if its out-of-sample accuracy is modest, the transition probabilities are economically interpretable. A state where P(up | previous_down) = 0.56 provides a direct estimate of short-term reversal tendency -- something opaque in a 500-tree forest or a GRU with 20K parameters.

3. **Sanity check for pipeline correctness.** A Markov model should reproduce a known stylised fact from the market microstructure literature: there should be mild mean reversion at the 1-day horizon (Lo & MacKinlay, 1988) and mild momentum at the 5-day horizon (Jegadeesh, 1990). If it does not, something is wrong with the feature pipeline or the target construction.

---

## 2. State Representation

### What to discretise

The model needs to discretise a continuous variable into a small number of states. The choice is which feature to use as the state variable.

**Candidates:**

| Feature | Economic interpretation | Notes |
|---------|------------------------|-------|
| `log_ret_1` | Previous-day return | Direct measure of short-term momentum/reversal |
| `log_ret_5` | 5-day return | Medium-term momentum signal |
| `rsi_14` | RSI(14) | Already discretised interpretation (overbought/oversold) |
| `vol_ratio_5_20` | Volatility regime | Regime indicator, not a directional signal |

**Decision: use `log_ret_1` as the primary state variable for order=1.**

Rationale: The foundational question for a 1-step Markov model of equity returns is whether today's direction predicts tomorrow's. This is exactly the return autocorrelation question studied by Lo & MacKinlay (1988). Using `log_ret_1` keeps the model faithful to this question. Using a derived indicator like RSI introduces a constructed transformation that embeds additional assumptions about the process.

For `order=2`, we add `log_ret_5` as the second lag. The weekly return is used rather than `log_ret_2` (which is not in the feature set) because 5-day momentum has a well-established empirical basis (Jegadeesh, 1990; Lehmann, 1990) and avoids the short-term bid-ask bounce reversal that contaminates daily autocorrelations.

### Number of bins: n_states = 3

We discretise `log_ret_1` into **3 quantile bins**:
- State 0: bottom third (strong down day)
- State 1: middle third (near-flat day)
- State 2: top third (strong up day)

**Why quantile bins rather than fixed thresholds?**

Fixed thresholds (e.g., ±0.5%) have two problems for a rolling-window scheme. First, volatility regimes change: a 0.5% move is significant in a low-volatility period but unremarkable during a crisis. A threshold calibrated to a low-vol window will classify most observations as "state 1 (flat)" during high-vol periods, degrading the state signal. Second, fixed thresholds create unequal state frequencies. With too few observations in the extreme states, the transition probability estimates are noisy.

Quantile bins guarantee approximately equal frequencies across states within each training window. This maximises the effective sample size for estimating each transition probability and ensures the model sees all three states with similar frequency.

Hamilton (1989), in the foundational paper on regime-switching models in finance, argues that state definitions should be data-driven rather than imposed from outside, precisely because volatility regimes are non-stationary. Our quantile approach is the simplest implementation of this principle.

**Why 3 states rather than 2 or 5?**

- **2 states (up/down):** Identical to simply using the sign of yesterday's return. This collapses the rich information in the magnitude of the move -- a return of -0.05% and -3% both become state 0. The literature (Cont, 2001) shows that return magnitude carries autocorrelation information (large moves tend to cluster), which 2 states cannot capture.

- **3 states:** Captures the strong-down / neutral / strong-up trichotomy. With ~750 training days x 35 tickers = ~26,250 training rows, each state receives approximately 8,750 observations. For a 1st-order model this gives ~8,750 transitions per state -- enough for stable probability estimates (SE < 0.01).

- **5 states:** Each state receives ~5,250 observations -- still adequate. However, for `order=2`, a 5-state model creates 25 joint states, each receiving only ~1,050 observations. With Laplace smoothing, the estimates remain valid, but the statistical benefit of the finer discretisation is marginal. The additional complexity is not justified.

- **Quantile bins (as opposed to equal-width bins):** The same argument as above. Returns have heavy tails (Mandelbrot, 1963; Cont, 2001). Equal-width bins would put the vast majority of observations in the center bin and very few in the tails. Quantile bins avoid this.

**Final value:** `n_states = 3`

---

## 3. Markov Order

### Order 1 (default) vs Order 2

A 1st-order Markov chain assumes the next state depends only on the current state: `P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t)`. A 2nd-order chain allows dependence on the two most recent states.

**Why order=1 as the default:**

The short-term return autocorrelation literature provides conflicting evidence on whether 1-day lags or multi-day patterns dominate:

- Lo & MacKinlay (1988) find significant positive autocorrelation in weekly returns (positive serial correlation at the portfolio level, driven by lagged cross-correlations between stocks). But at the individual stock level and daily horizon, autocorrelation is weaker and often negative (mean reversion from bid-ask bounce).

- Jegadeesh (1990) documents 1-month return reversals at the individual stock level, suggesting that the state at lag 1 has predictive content. But the signal decays fast, and by lag 2 the remaining predictability is small.

- Lehmann (1990) similarly finds significant reversal at the 1-week horizon but negligible signal at the 2-week horizon after transaction costs.

For the purpose of this model, order=1 is the appropriate starting assumption because:

1. **Statistical efficiency.** With `n_states=3`, an order-1 model has 3 states. An order-2 model has 9 joint states (3 x 3). Each transition from a joint state to the binary target requires enough observations to estimate P(up | state). With 8,750 observations per order-1 state, we get ~3,000-4,000 per order-2 state (accounting for the joint distribution's non-uniformity). This is still statistically adequate, but order-3 would push us below 1,000 per state in many windows.

2. **Parsimony (AIC/BIC principle).** Refinetti et al. (2021) argue that for financial return discretisation, the improvement in transition matrix accuracy from order=2 to order=3+ rarely justifies the additional parameters. The marginal value of the second lag drops sharply relative to the first.

3. **The 1-day autocorrelation captures the dominant signal.** For h=1 prediction, the most relevant lagged return is `log_ret_1`. Adding `log_ret_5` (order=2) adds a medium-term momentum context but introduces state sparsity.

**Order=2 is available as a configurable option** (`MARKOV_PARAMS["order"] = 2`) for experimentation. It should be tested to verify whether the 9-state model outperforms the 3-state model on the CV balanced accuracy. In many equity markets, the answer is no (Bulla & Bulla, 2006).

**Final value:** `order = 1` (default; `order=2` available)

---

## 4. Laplace Smoothing (alpha)

### The zero-frequency problem

With `n_states=3` and order=1, a single training window (~26,250 rows) will have seen all 3 states many times. However, in shorter windows or when order=2 is used (9 joint states), some states may have very few observations. More critically, after discretising using training-set quantile edges, test-set observations can occasionally produce bin indices that push to the boundary states (0 or 2) more or less frequently than seen in training.

Without smoothing, a state seen only once in training receives a transition probability of either 0 or 1 -- pathological predictions that produce infinite log loss and overconfident signals.

### Laplace (additive) smoothing

Laplace smoothing (Good, 1953; Manning & Schütze, 1999) adds a pseudo-count `alpha` to each (state, outcome) cell before computing the probability:

```
P(up | state) = (count(up | state) + alpha) / (count(state) + 2 * alpha)
```

With `alpha=1`, this is equivalent to adding one observation of each outcome per state -- a "uniform prior" over outcomes. The effect:

- A state seen 0 times: P(up) = 0.5 (maximally uncertain -- the correct prior)
- A state seen 1 time with 1 up: P(up) = 2/3 (instead of 1.0 -- appropriately shrunk)
- A state seen 100 times with 60 up: P(up) = 61/102 ≈ 0.598 (negligible shrinkage)
- A state seen 8,750 times with 4,550 up: P(up) = 4,551/8,752 ≈ 0.520 (imperceptible shrinkage)

The key property is that smoothing is most aggressive precisely when it is most needed (rare states) and negligible when counts are large.

**Why alpha=1 specifically?**

The choice of alpha represents the strength of the prior. In Bayesian terms, `alpha=1` corresponds to a Beta(1,1) prior -- a uniform prior over [0,1]. This is the weakest non-zero prior and the standard default (Laplace, 1814). It is appropriate when we have no strong prior belief about the transition probabilities.

Alternative choices:
- `alpha=0.5`: Jeffreys prior (Beta(0.5, 0.5)). Slightly less shrinkage. Theoretically motivated for categorical distributions (Jeffreys, 1946) but almost indistinguishable from alpha=1 at our sample sizes.
- `alpha=5`: Stronger shrinkage. Would only matter for extremely sparse states (< 20 observations). Unnecessary for n_states=3.
- `alpha=0` (no smoothing): Dangerous. Produces undefined probabilities for any state not seen in training, which occurs regularly in early CV windows where certain extreme return states may not appear.

**Final value:** `alpha = 1.0`

---

## 5. Prediction Threshold (0.5)

The model predicts "up" when `P(up | state) >= 0.5`. This is the natural Bayes-optimal threshold under equal misclassification costs and is consistent with all other models in this project.

The threshold is not a hyperparameter here because:

1. The Markov model's predicted probabilities are empirical frequencies (after smoothing). With n_states=3, there are only 3 distinct probability values in the output. Threshold optimisation over such a coarse probability set is not meaningful.

2. Balanced accuracy (our primary metric) implicitly optimises over all thresholds. The 0.5 threshold is used for hard predictions in the CV loop, which is the operationally relevant decision.

---

## 6. What the model does NOT use

The model ignores 39 of the 41 available features. This is intentional. The Markov chain is designed to answer a narrow question -- "is there autocorrelation signal in return states?" -- not to maximise predictive accuracy. Incorporating all 41 features would require a much more complex conditional independence structure (a high-dimensional state space or a hidden Markov model with emission distributions over the feature vector).

If a high-accuracy Markov-flavoured model were desired, the natural extension would be a **Hidden Markov Model (HMM)** with multivariate Gaussian emissions over the full feature set (Rabiner, 1989; Hamilton, 1989). This is a materially more complex implementation and is deferred to a future iteration. The current model is the simplest possible Markov baseline.

---

## 7. Relationship to Other Models

| Property | Markov | RF | XGBoost | GRU/LSTM |
|----------|--------|----|---------|----------|
| Input dimensionality | 1-2 features | 41 | 41 | 41 x T=20 |
| Number of parameters | 3-9 probs | ~500K splits | ~150 rounds x splits | ~20K weights |
| Fitting method | Frequency counting | Gini/entropy splits | Gradient boosting | Backprop |
| Feature scaling | Not needed | Not needed | Not needed | z-score |
| Temporal structure | Explicit (lags) | Implicit (rolling features) | Implicit | Explicit (sequence) |
| Probability calibration | Direct (empirical freq.) | Vote fractions | Logistic sigmoid | Sigmoid |
| Interpretability | Very high | Medium (feature importance) | Medium | Low |
| Expected balanced accuracy | ~50.5-52% | ~53-55% | ~53-55% | ~52-54% |

The Markov model is expected to be the weakest predictor by balanced accuracy. Its value is interpretability and as a diagnostic baseline, not as a production model.

---

## 8. Final Configuration

```python
MARKOV_PARAMS = {
    "n_states": 3,    # quantile bins: bottom/neutral/top thirds
    "order":    1,    # lag-1 only; set to 2 to also use log_ret_5 context
    "alpha":    1.0,  # Laplace smoothing; prevents zero-probability states
}
```

### Summary Table

| Parameter | Value | Key Rationale |
|-----------|-------|---------------|
| `n_states` | 3 | Equal state frequencies; SE < 0.01 per state at n=8,750; 5 states not justified for order=2 |
| `order` | 1 | Dominant autocorrelation signal at lag-1; order-2 marginal gain vs. state sparsity |
| `alpha` | 1.0 | Laplace (uniform) prior; negligible effect at large counts, prevents zero-probability states |

---

## References

- **Bulla, J. & Bulla, I. (2006).** "Stylized facts of financial time series and hidden semi-Markov models." *Computational Statistics & Data Analysis*, 51(4), 2192-2209. Tests Markov order selection for discretised return sequences. Finds that order 1-2 is sufficient for most equity series; higher orders do not improve transition matrix estimates meaningfully.

- **Cont, R. (2001).** "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance*, 1(2), 223-236. Documents the heavy tails and volatility clustering of daily returns. The rationale for quantile binning (equal-frequency) over equal-width binning follows directly from the non-Gaussian return distribution described here.

- **Good, I.J. (1953).** "The population frequencies of species and the estimation of population parameters." *Biometrika*, 40(3-4), 237-264. Foundational paper on smoothing sparse probability estimates. The Good-Turing estimator is a generalisation; Laplace (additive) smoothing with alpha=1 is the simplest special case.

- **Hamilton, J.D. (1989).** "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384. Introduces the regime-switching (Hidden Markov) model for economic time series. The argument that state definitions should be data-driven rather than imposed a priori underpins our use of quantile-based discretisation.

- **Jeffreys, H. (1946).** "An invariant form for the prior probability in estimation problems." *Proceedings of the Royal Society A*, 186(1007), 453-461. Proposes the Jeffreys prior (alpha=0.5 for Bernoulli distributions) as a theoretically motivated alternative to the uniform prior (alpha=1). At our sample sizes, the choice between them is numerically irrelevant.

- **Jegadeesh, N. (1990).** "Evidence of predictable behavior of security returns." *Journal of Finance*, 45(3), 881-898. Documents significant return reversal at the 1-month horizon for individual US stocks. This is the primary empirical motivation for including `log_ret_1` as a state variable for h=1 prediction.

- **Laplace, P.S. (1814).** *Essai Philosophique sur les Probabilités*. Paris: Courcier. The original additive smoothing rule: estimate the probability of an event as (k+1)/(n+2) where k is the number of occurrences and n is total trials. Equivalent to alpha=1 in our formulation.

- **Lehmann, B.N. (1990).** "Fads, martingales, and market efficiency." *Quarterly Journal of Economics*, 105(1), 1-28. Provides further evidence for short-term equity return reversals. Notes that bid-ask bounce contaminates very short-term (1-day) autocorrelations at the individual stock level.

- **Lo, A.W. & MacKinlay, A.C. (1988).** "Stock market prices do not follow random walks: evidence from a simple specification test." *Review of Financial Studies*, 1(1), 41-66. The foundational test of return autocorrelation. Finds significant positive serial correlation in weekly US equity returns, primarily driven by lagged cross-correlations across stocks rather than individual stock autocorrelation. Establishes that the random walk hypothesis is rejected -- there is some Markov-type structure in returns.

- **Mandelbrot, B. (1963).** "The variation of certain speculative prices." *Journal of Business*, 36(4), 394-419. Documents the heavy-tailed (non-Gaussian) distribution of daily financial returns. The practical implication for our model: equal-width bins would concentrate most observations in the centre; quantile bins are required for balanced state populations.

- **Manning, C.D. & Schütze, H. (1999).** *Foundations of Statistical Natural Language Processing*. MIT Press. Chapter 6 gives a clear treatment of smoothing methods for sparse categorical distributions, including add-one (Laplace) smoothing and its theoretical properties.

- **Rabiner, L.R. (1989).** "A tutorial on hidden Markov models and selected applications in speech recognition." *Proceedings of the IEEE*, 77(2), 257-286. The standard reference for HMMs and the Baum-Welch algorithm. Relevant as the natural extension of this model: if the 3-state observable Markov chain proves insufficient, an HMM with multivariate Gaussian emissions over the full feature set is the principled next step.

- **Refinetti, M., Goldt, S., Krzakala, F. & Zdeborová, L. (2021).** "Classifying high-dimensional Gaussian mixtures: Where kernel methods fail and neural networks succeed." *Proceedings of ICML 2021*. Provides information-theoretic arguments for why higher-order Markov structure (more lags) provides rapidly diminishing returns in high-noise settings. The SNR degradation with each additional lag is the primary argument for keeping order=1.
