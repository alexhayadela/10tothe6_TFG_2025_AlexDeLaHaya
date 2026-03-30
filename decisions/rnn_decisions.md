# RNN (LSTM/GRU) -- Design Decisions

**Date:** 2026-03-30
**Task:** Binary stock direction classification (h=1, next-day up/down)
**Data:** 30 IBEX35 stocks, ~5000 trading days each, 40 features (ft_type="macro", h=1)
**Training scheme:** 3-year sliding window (~750 days x 30 tickers)
**Evaluation metrics:** Shared with RF/XGBoost -- see `decisions/rf_decisions.md`, Part 2

---

## 1. Feature Set vs Tree Models

### The core question: pre-computed rolling features or raw inputs?

Tree models need manually engineered rolling features (SMA ratios, RSI, volatility ratios) because each decision tree sees only the current row -- it has no memory. LSTMs and GRUs have recurrent hidden states that can, in principle, learn temporal aggregations directly from raw sequences of returns and volumes.

However, "in principle" is doing heavy lifting. The practical question is whether an LSTM *actually* learns to compute rolling averages, RSI, or volatility ratios from raw price data given our constraints:

1. **Training data is small.** Each ticker contributes ~750 sequences per 3-year window. Even pooling all 30 tickers, we have ~22,000 training sequences. This is orders of magnitude below the data volumes where end-to-end feature learning succeeds in NLP or vision. The LSTM must simultaneously learn feature representations AND the predictive mapping from features to target. With weak signal (~52-55% accuracy ceiling), learning both is asking too much from limited data.

2. **The signal is weak and buried.** A 20-day rolling standard deviation is a simple linear computation, but the LSTM must discover that this specific computation is relevant from noisy gradient signals. With an accuracy ceiling of 55% at best, the reward signal for discovering "volatility clustering matters" is vanishingly small compared to the gradient noise.

3. **Empirical evidence favours engineered features.** Krauss et al. (2017) tested deep networks on S&P 500 daily direction prediction and found that hand-crafted technical features performed comparably to or better than raw price inputs. Sezer et al. (2020), in their survey of 140+ papers on financial deep learning, conclude that "the majority of studies that compare raw price inputs to technical indicators find that engineered features lead to faster convergence and slightly better or comparable performance." Fischer & Krauss (2018), who specifically applied LSTMs to daily S&P 500 direction, used lagged returns (a minimal form of pre-processing) rather than raw prices, achieving ~54% daily accuracy.

4. **Rolling features compress information efficiently.** An RSI(14) value packs 14 days of price history into a single number with a clear economic interpretation. Feeding the LSTM a sequence of RSI values gives it access to *the trajectory of RSI over time* -- a second-order signal (how overbought/oversold conditions evolve) that would require much longer sequences and more parameters to learn from raw returns alone.

### Decision: use the same pre-computed features as the tree models

**Keep all 40 features from the tree model h=1 feature set.** The LSTM receives a sequence of T timesteps, where each timestep is a 40-dimensional vector of pre-computed features. This is a "features-as-sequence" approach: the LSTM's job is to learn temporal *patterns* in the feature trajectories, not to learn the features themselves.

This is the standard approach in the financial deep learning literature. Bao et al. (2017) use stacked autoencoders on technical indicators fed into LSTMs. Tsantekidis et al. (2017) use pre-computed order book features as LSTM inputs. The consensus is clear: for low-SNR financial tasks with limited data, pre-computed features are the pragmatic choice.

### What about dropping redundant features the LSTM could learn?

One might argue: "The LSTM can compute vol_ratio_5_20 from the sequence of log_ret_1 values, so drop vol_ratio_5_20 and let the network learn it." This reasoning is flawed for two reasons:

1. **Pre-computed features do not hurt.** Unlike tree models where redundant features can cause split dilution, neural networks handle correlated features gracefully through weight sharing. If vol_ratio_5_20 is computable from log_ret_1, the network can simply learn a near-zero weight for one of them. The cost is negligible (a few extra parameters in the input projection).

2. **Pre-computed features help convergence.** Even if the LSTM *could* learn to compute vol_ratio_5_20, providing it directly reduces the optimisation burden. The network converges faster and more stably because it does not need to allocate hidden state capacity to replicating a simple rolling computation.

**No features are dropped from the tree set.** No raw features (OHLCV levels) are added either -- they are non-stationary and would require normalisation that the pre-computed ratios already handle.

### The full h=1 feature vector (40 features per timestep)

Micro (28):
`log_ret_1`, `log_ret_5`, `log_ret_10`, `log_ret_20`, `vol_5`, `vol_ratio_5_20`, `atr_pct`, `sma_ratio_5_20`, `sma_ratio_10_50`, `macd_hist`, `bb_pct`, `slope_10`, `dist_high_10`, `dist_low_10`, `dist_high_20`, `dist_low_20`, `rsi_14`, `volu_ratio_5`, `volu_ratio_20`, `volu_ret_1`, `obv_slope_10`, `amihud_10`, `ret_autocorr_10`, `intraday_ret`, `body`, `upper_wick`, `lower_wick`, `gap`

Cross (2):
`ibx_breadth`, `ibx_breadth_10d`

Macro (9):
`ibx_vol_10`, `ibx_vol_ratio_10_60`, `sp_vol_20`, `sp_vol_ratio_20_100`, `vix_chg_z_5`, `vix_pctile_250`, `rel_ret_5`, `rel_ret_20`, `rel_vol_20`

Horizon-specific (1):
`dow`

---

## 2. Macro Features

### Should macro features be included?

**Yes, include all 9 macro features plus the 2 cross-sectional breadth features.**

The reasoning:

1. **Macro features provide regime context that per-stock features cannot.** VIX percentile (where VIX sits in its 1-year distribution) and S&P volatility ratio (US vol regime) are impossible to derive from individual IBEX stock features. These are genuinely exogenous signals about the global risk environment. Removing them would blind the model to market regime, which is among the strongest predictors of short-term equity dynamics (Gu et al., 2020).

2. **The LSTM hidden state cannot learn macro regime from micro features alone.** One could argue that if all 30 stocks are simultaneously volatile, the LSTM could infer a stress regime from the per-stock vol features. But our model processes each ticker's sequence independently (see Section 5), so it never sees cross-sectional information within a single forward pass. The macro features are the only channel for market-level information.

3. **Date-level features repeated across tickers are not problematic for RNNs.** The concern with date-level features (same value for all 30 stocks on a given day) is that the model might overweight them. In tree models, this is a legitimate concern because a date-level feature can create a split that partitions the data by date rather than by stock characteristic. In RNNs, the feature enters through the input projection and is processed jointly with stock-specific features. The network learns to weight macro vs. micro signals through backpropagation, and the shared LSTM weights across tickers provide implicit regularisation against date-level overfitting.

4. **Relative-to-market features are essential.** `rel_ret_5`, `rel_ret_20`, and `rel_vol_20` measure alpha and beta relative to the IBEX index. These are among the strongest cross-sectional predictors in the asset pricing literature (Gu et al., 2020). For a shared model across 30 tickers, relative features help the model distinguish between a stock that dropped 2% on a flat market (idiosyncratic signal) and one that dropped 2% when the market dropped 3% (beta exposure). Without relative features, the model conflates these two very different scenarios.

### How to handle dow (day-of-week)

`dow` is an integer 0-4. For tree models, ordinal encoding works because trees can split at arbitrary thresholds. For the LSTM, `dow` should be either:

- **Cyclically encoded:** `sin(2*pi*dow/5)` and `cos(2*pi*dow/5)` -- this adds 1 extra feature (2 instead of 1) but preserves the circular nature (Friday is close to Monday).
- **Left as an integer and z-scored with the other features.** Since the LSTM sees sequences and the day-of-week follows a deterministic pattern (0,1,2,3,4,0,1,...), the network can learn the periodicity from the sequence itself.

**Decision:** Encode `dow` cyclically as `dow_sin` and `dow_cos`. This replaces the single `dow` column. The feature count becomes 41.

---

## 3. Sequence Length

### The trade-off

The LSTM receives a sequence of T consecutive trading days as input and produces a single prediction for day T+1. The choice of T involves several tensions:

- **More context (large T):** The LSTM can observe longer-term patterns (trend reversals, volatility regime shifts, seasonal effects). Features like `sma_ratio_10_50` already encode 50-day lookbacks in each timestep, so the LSTM seeing 60 timesteps of `sma_ratio_10_50` effectively has access to >100 days of price history.
- **Harder to train (large T):** Vanishing gradients degrade signal propagation over long sequences even in LSTMs (Pascanu et al., 2013). More timesteps mean more parameters in the unrolled computation graph and more sequential operations that cannot be parallelised.
- **Diminishing information (large T):** Lopez de Prado (2018, Ch. 5) argues that information in financial time series decays rapidly -- recent days carry far more predictive content than days 40+ in the past. There are diminishing returns to longer lookbacks for daily prediction.

### Evidence from the literature

- **Sezer et al. (2020)** survey 140+ papers and report that the majority of daily equity prediction studies use lookback windows of 10-30 trading days. T=20 (1 month) is the most common choice.
- **Fischer & Krauss (2018)** used T=240 for their LSTM on S&P 500 daily direction but achieved only ~54% accuracy -- and their dataset was much larger (500 stocks x 20 years). Long sequences did not translate to dramatically better performance.
- **Baek & Kim (2018)** tested T in {5, 10, 20, 40, 60} for LSTM-based stock prediction and found T=20 performed best, with T=60 showing signs of overfitting.
- **Practical consensus:** For daily equity direction, T=20 to T=30 is the sweet spot. Beyond T=40, returns diminish rapidly.

### Our specific constraints

With a 750-day training window per ticker:
- T=20: 730 usable sequences per ticker x 30 tickers = 21,900 training samples
- T=40: 710 x 30 = 21,300 samples
- T=60: 690 x 30 = 20,700 samples

The sample count difference is modest (<6%), so the choice is driven by signal quality, not data loss.

However, consider what the LSTM is seeing at each timestep. The features already encode substantial lookback:
- `log_ret_20` looks back 20 days
- `sma_ratio_10_50` encodes the 50-day SMA
- `vix_pctile_250` looks back 250 days
- `rsi_14` uses 14 days of history

A sequence of T=20 timesteps of these features gives the LSTM effective access to price history spanning 20+50 = 70 days (through `sma_ratio_10_50`) or even 20+250 = 270 days (through `vix_pctile_250`). The pre-computed features have already done the heavy lifting of long-range information aggregation. The LSTM's job is to detect *changes in these indicators over the recent past*, not to aggregate raw history.

For detecting changes and momentum in indicator trajectories, 20 trading days (1 calendar month) is sufficient. It covers:
- Multiple cycles of short-term reversal (log_ret_1 autocorrelation)
- One full cycle of the RSI oscillator
- Enough breadth trajectory to detect regime shifts
- The full day-of-week cycle 4 times

### Decision: T = 20

**Sequence length = 20 trading days.**

This is conservative and deliberate. A shorter sequence:
- Reduces the vanishing gradient problem
- Requires fewer sequential operations (faster training)
- Focuses the model on recent dynamics, which is where the predictive signal concentrates for h=1
- Aligns with the dominant approach in the literature

If ablation experiments show that T=20 underfits relative to T=30 or T=40, the sequence length can be increased. But T=20 is the safer starting point given our data constraints.

---

## 4. Architecture

### LSTM vs GRU

GRU (Gated Recurrent Unit) has 2 gates (reset, update) vs LSTM's 3 gates (forget, input, output) and no separate cell state. GRU has ~25% fewer parameters per layer for the same hidden size.

For our setting:
- **Data is limited:** Fewer parameters is an advantage. GRU's simpler architecture is less prone to overfitting.
- **Sequences are short (T=20):** LSTM's advantage over GRU is primarily in long-range dependencies (Chung et al., 2014). For T=20, the cell state's ability to carry information over very long distances is unnecessary.
- **Empirical parity:** Chung et al. (2014) and Greff et al. (2017) found no consistent winner between LSTM and GRU across tasks. For financial time series, the differences are negligible (Sezer et al., 2020).

**Decision: train both, select via validation.** The architecture is identical except for swapping `nn.LSTM` with `nn.GRU`. Both should be trained and compared on each rolling window. In practice, expect comparable results with GRU training ~20% faster.

**Default for initial experiments: GRU** (fewer parameters, faster iteration).

### Number of layers: 1

Stacking multiple recurrent layers (2+) allows the network to learn hierarchical temporal abstractions. This is valuable for complex sequential tasks (machine translation, speech recognition) where raw inputs (characters, audio frames) need multiple levels of abstraction.

For our setting, the inputs are already high-level abstractions (RSI, volatility ratios, MACD). There is no "low-level" representation to learn. A single recurrent layer processes the 40-dimensional feature trajectory and produces a hidden state that captures the relevant temporal patterns. A second layer would process the hidden state trajectory of the first layer -- a "pattern of patterns" -- which is unlikely to contain useful signal at our SNR level.

**Evidence:** Krauss et al. (2017) and Fischer & Krauss (2018) both used relatively simple architectures (1-2 recurrent layers) for daily equity prediction. Fischer & Krauss (2018) found that deeper architectures did not improve daily direction accuracy on S&P 500 data.

**Decision: 1 recurrent layer.** If the model underfits (training accuracy plateaus below 54%), a 2-layer architecture can be tried. But the risk of overfitting with 2 layers at our data scale significantly outweighs the potential benefit.

### Hidden size: 64

The hidden size controls the capacity of the recurrent layer. With input dimension d=41 (40 features + 1 from the cyclic dow encoding):

- **h=32:** 32-dimensional hidden state. Total recurrent parameters for GRU: 3 x (32 x 41 + 32 x 32 + 32) = 3 x (1,312 + 1,024 + 32) = 7,104. Very compact; may underfit if there are non-trivial temporal interactions.
- **h=64:** 64-dimensional hidden state. GRU parameters: 3 x (64 x 41 + 64 x 64 + 64) = 3 x (2,624 + 4,096 + 64) = 20,352. A reasonable capacity that can model several independent temporal patterns.
- **h=128:** 128-dimensional hidden state. GRU parameters: 3 x (128 x 41 + 128 x 128 + 128) = 3 x (5,248 + 16,384 + 128) = 65,280. Over 3x more parameters than h=64. With ~21,000 training sequences, this risks overfitting.

**Ratio of parameters to training samples:** With h=64 and a linear output layer (64 x 1 + 1 = 65), total parameters ~ 20,352 + 65 = 20,417. With 21,900 training sequences, the ratio is ~1.07 samples per parameter. This is tight but manageable with proper regularisation (dropout, early stopping, weight decay). At h=128, the ratio drops to ~0.34, which is in the overfitting danger zone for weak-signal problems.

**Decision: hidden_size = 64.**

### Dropout: 0.3

Apply dropout at two locations:

1. **Recurrent dropout (between timesteps):** Not recommended. Gal & Ghahramani (2016) showed that naive dropout within recurrent connections destroys the hidden state's ability to carry information. Their variational dropout is theoretically sound but adds implementation complexity and has marginal empirical benefit for short sequences (T=20).

2. **Dropout after the recurrent layer, before the linear output:** This is the standard and safe location. After the GRU processes all T timesteps and produces the final hidden state h_T, apply dropout before the classification head. This regularises the mapping from hidden state to output without disrupting the temporal dynamics.

**Rate = 0.3:** Moderate. Srivastava et al. (2014) recommend 0.2-0.5 for hidden layers. For our small dataset and weak signal, 0.3 provides meaningful regularisation without crippling the hidden representation. If overfitting persists, increase to 0.5.

### Output layer

```
Linear(hidden_size, 1) -> Sigmoid
```

A single sigmoid output producing P(up). Binary cross-entropy loss. This is consistent with the XGBoost configuration (`objective="binary:logistic"`) and produces calibrated probabilities for downstream use.

### Batch normalisation: not recommended

Batch normalisation normalises activations across the batch dimension. For recurrent networks, this is problematic:

1. **Batch statistics are unstable during inference.** At inference time, the model uses running statistics computed during training. For financial data where the distribution shifts across rolling windows, these running statistics may not reflect the test distribution.

2. **Layer normalisation is the standard alternative for RNNs** (Ba et al., 2016). It normalises across features within each sample, avoiding batch dependency. However, for our short sequences and pre-normalised features (all features are already ratios or z-scored values), the benefit is marginal.

3. **Input standardisation is more important.** Z-score all 41 features using training-set statistics (mean and std computed on the training window). This ensures all features enter the network on comparable scales. This is critical and must not be skipped.

**Decision: no batch norm, no layer norm. Apply z-score standardisation to all input features using training-set statistics only.** The scaler must be fit on the training window and applied (without refitting) to the validation and test sets. Per the existing `features_decisions.md` implementation note: "Never fit the scaler on the full dataset."

---

## 5. Cross-Sectional Training

### One model per ticker vs one shared model

**Option A -- One model per ticker:**
- 750 sequences per model (T=20 lookback, 730 usable per ticker)
- The model can learn ticker-specific dynamics (e.g., Santander may have different patterns than Inditex)
- 730 training sequences for a model with ~20,000 parameters: catastrophic overfitting is virtually guaranteed

**Option B -- One shared model across all 30 tickers:**
- ~21,900 sequences total
- The model learns patterns common across all IBEX35 stocks
- Cannot learn ticker-specific idiosyncrasies
- ~1 sample per parameter ratio: tight but feasible with regularisation

**Option C -- Shared model with ticker embedding:**
- One shared GRU + a learned ticker embedding vector concatenated to the input or hidden state
- Allows some ticker-specific adaptation while sharing temporal dynamics
- Adds 30 x embedding_dim parameters (e.g., 30 x 8 = 240 extra parameters -- negligible)
- Risk: with only ~730 sequences per ticker, the embedding may overfit to ticker identity rather than learning meaningful stock characteristics

### Decision: Option B -- one shared model, no ticker embedding

The primary argument is data sufficiency. 730 sequences per ticker is far too few for a model with 20,000 parameters. Cross-sectional pooling is essential.

Ticker-specific dynamics are already partially captured by the feature set. Features like `rel_ret_5` (return relative to IBEX), `amihud_10` (liquidity), and `vol_5` (volatility) differentiate stocks by their *current characteristics* rather than by their identity. A stock that is currently high-volatility and illiquid will produce similar GRU hidden states regardless of its ticker name. This is the right inductive bias: we want the model to generalise across stocks based on observable features, not memorise ticker-specific patterns.

**Ticker embeddings are not recommended for the initial implementation.** They can be revisited if the shared model shows clear evidence of underfitting (e.g., training accuracy significantly below what per-ticker models achieve).

### Sequence construction

Each training sample is a (sequence, label) pair:
- **Sequence:** A tensor of shape (T, 41) = (20, 41), representing 20 consecutive trading days of features for one ticker
- **Label:** The binary target (0/1) for day T+1

Sequences are constructed per-ticker with a sliding window of stride 1:

```
For ticker X with dates [d1, d2, ..., d750]:
  Sample 1: features[d1:d20]  -> target[d21]
  Sample 2: features[d2:d21]  -> target[d22]
  ...
  Sample 730: features[d730:d749] -> target[d750]
```

Sequences from different tickers are then pooled into a single dataset and shuffled for training. Each mini-batch contains sequences from multiple tickers and dates.

**Important:** Sequences must NOT span ticker boundaries. Each sequence belongs to exactly one ticker.

---

## 6. Training Configuration

### Loss function: Binary Cross-Entropy

```python
criterion = nn.BCEWithLogitsLoss()
```

Use `BCEWithLogitsLoss` (applies sigmoid internally) rather than `BCELoss` with an explicit sigmoid layer. This is numerically more stable due to the log-sum-exp trick.

### Optimiser: AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

Adam (Kingma & Ba, 2015) is the standard optimiser for recurrent networks. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the adaptive learning rate, providing more effective L2 regularisation. Weight decay of 1e-4 adds a mild penalty that complements dropout.

**Learning rate = 1e-3:** The Adam default. For small models on small datasets, this is a safe starting point. If training is unstable (loss oscillates), reduce to 3e-4.

### Learning rate scheduler: ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

Halve the learning rate when validation loss plateaus for 5 epochs. This is gentler than a step schedule and adapts to the specific convergence dynamics of each rolling window.

### Batch size: 128

With ~21,900 training samples, batch size 128 gives ~171 gradient updates per epoch. This is a reasonable balance between:
- Gradient noise (smaller batches add noise that can help escape local minima)
- Training speed (larger batches utilise GPU parallelism better)
- Memory constraints (128 x 20 x 41 x 4 bytes = ~0.4 MB per batch -- negligible)

### Epochs and early stopping

- **Max epochs: 100**
- **Early stopping patience: 10 epochs** (monitoring validation loss)
- **Validation split:** Same temporal 80/20 split as XGBoost (days 1-600 for training, 601-750 for validation within each rolling window)

Early stopping is critical for the same reasons as in XGBoost: the model will begin memorising noise after the genuine signal is captured. With weak financial signals, expect early stopping to trigger between epochs 15 and 40.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clip gradient norms to prevent exploding gradients, which are common in recurrent networks (Pascanu et al., 2013). Max norm of 1.0 is the standard default.

---

## 7. Final Configuration

### Model architecture

```python
import torch
import torch.nn as nn

class StockGRU(nn.Module):
    def __init__(self, input_size=41, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len=20, input_size=41)
        _, h_n = self.gru(x)          # h_n: (1, batch, 64)
        h_n = h_n.squeeze(0)          # (batch, 64)
        h_n = self.dropout(h_n)
        logits = self.fc(h_n)         # (batch, 1)
        return logits.squeeze(-1)     # (batch,)
```

### Hyperparameter summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cell type | GRU (default); LSTM as alternative | GRU has fewer parameters; no consistent advantage for LSTM at T=20 |
| Sequence length T | 20 | 1 month of trading days; aligns with literature consensus |
| Input features | 41 | All 40 tree features + cyclic dow encoding (sin/cos replaces integer) |
| Hidden size | 64 | ~20K params; ~1 sample/param ratio with 21,900 training sequences |
| Num layers | 1 | Single layer sufficient for pre-computed features; avoids overfitting |
| Dropout | 0.3 | After GRU, before linear output |
| Batch norm | None | Z-score input standardisation instead |
| Output | Linear(64,1) + sigmoid | BCEWithLogitsLoss for numerical stability |
| Optimiser | AdamW | lr=1e-3, weight_decay=1e-4 |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |
| Batch size | 128 | ~171 updates per epoch |
| Max epochs | 100 | Early stopping patience = 10 epochs |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |
| Training scope | Shared across 30 tickers | Cross-sectional pooling; no ticker embedding |
| Validation split | Temporal 80/20 | Days 1-600 train, 601-750 validation |
| Input scaling | Per-feature z-score | Fit on training window only |

### Changes from tree model configuration

| Aspect | Tree Models (RF/XGBoost) | RNN (GRU/LSTM) |
|--------|--------------------------|-----------------|
| Input shape | (n_samples, 40) flat | (n_samples, 20, 41) sequential |
| Feature set | 40 features, single row | Same 40 features + cyclic dow, over 20 timesteps |
| Feature scaling | Not needed | z-score per feature, fit on training set only |
| Macro features | Included | Included (identical) |
| dow encoding | Integer 0-4 | Cyclical: sin(2*pi*dow/5), cos(2*pi*dow/5) |
| Model capacity | RF: 500 trees x depth 5; XGB: ~150 rounds x depth 3 | ~20K parameters |
| Regularisation | Structural (depth, leaf size) | Dropout 0.3 + weight decay 1e-4 + early stopping |
| Training | Fast (<10s per window) | Slower (~30-60s per window on GPU) |
| Probability output | Tree vote fractions / logistic sigmoid | Sigmoid on final logit |

### Overfitting guardrails (ordered by importance)

1. **Early stopping** (patience=10 on validation loss)
2. **Dropout 0.3** (applied to hidden state before classification)
3. **Weight decay 1e-4** (L2 regularisation via AdamW)
4. **Small model** (hidden_size=64, 1 layer, ~20K parameters)
5. **Short sequences** (T=20 limits temporal overfitting)
6. **Learning rate reduction** (ReduceLROnPlateau halves LR on plateau)
7. **Gradient clipping** (prevents exploding gradients, not overfitting per se)

**Escalation order if overfitting persists** (validation >> test balanced accuracy by >3 points):
1. Increase dropout to 0.5
2. Reduce hidden_size to 32
3. Increase weight_decay to 1e-3
4. Reduce sequence length to 10

---

## References

- **Ba, J.L., Kiros, J.R. & Hinton, G.E. (2016).** "Layer Normalization." *arXiv:1607.06450*. Proposes layer normalisation as an alternative to batch normalisation for recurrent networks, normalising across features rather than across the batch.

- **Baek, Y. & Kim, H.Y. (2018).** "ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module." *Expert Systems with Applications*, 113, 457-480. Tests LSTM with various lookback windows for stock prediction; T=20 performs best among {5, 10, 20, 40, 60}.

- **Bao, W., Yue, J. & Rao, Y. (2017).** "A deep learning framework for financial time series using stacked autoencoders and long-short term memory." *PLOS ONE*, 12(7), e0180944. Uses stacked autoencoders to pre-process technical indicators before feeding into LSTM. Demonstrates that feature engineering before the recurrent layer improves performance.

- **Chung, J., Gulcehre, C., Cho, K. & Bengio, Y. (2014).** "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." *arXiv:1412.3555*. Compares LSTM and GRU across multiple tasks. Finds no consistent winner; GRU performs comparably with fewer parameters.

- **Fischer, T. & Krauss, C. (2018).** "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669. Applies LSTM to daily S&P 500 direction prediction with T=240 and lagged returns as features. Achieves ~54% accuracy. Key reference for LSTM on daily equity prediction.

- **Gal, Y. & Ghahramani, Z. (2016).** "A Theoretically Grounded Application of Dropout to Recurrent Neural Networks." *Advances in Neural Information Processing Systems*, 29. Shows that naive dropout within recurrent connections disrupts hidden state. Proposes variational dropout with the same mask across timesteps.

- **Greff, K., Srivastava, R.K., Koutnik, J., Steunebrink, B.R. & Schmidhuber, J. (2017).** "LSTM: A Search Space Odyssey." *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232. Systematic ablation of LSTM components. Finds that forget gate and output activation are most critical; GRU-style simplifications lose little.

- **Gu, S., Kelly, B. & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273. Large-scale ML comparison for financial prediction. Confirms that macro and relative-to-market features are among the strongest predictors cross-sectionally.

- **Kingma, D.P. & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." *Proceedings of ICLR 2015*. Introduces the Adam optimiser with adaptive per-parameter learning rates.

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. Finds that hand-crafted technical features perform comparably to raw price inputs for deep networks on daily equity direction.

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. Ch. 5: Information decay in financial time series; argues for shorter lookbacks. Ch. 7: Purged cross-validation.

- **Loshchilov, I. & Hutter, F. (2019).** "Decoupled Weight Decay Regularization." *Proceedings of ICLR 2019*. Introduces AdamW, which decouples weight decay from the adaptive learning rate for more effective regularisation.

- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013).** "On the difficulty of training recurrent neural networks." *Proceedings of ICML 2013*, 1310-1318. Analyses vanishing and exploding gradients in RNNs. Recommends gradient clipping as a practical mitigation.

- **Sezer, O.B., Gudelek, M.U. & Ozbayoglu, A.M. (2020).** "Financial time series forecasting with deep learning: a systematic literature review: 2005-2019." *Applied Soft Computing*, 90, 106181. Survey of 140+ papers. Reports that T=10-30 is most common for daily prediction; engineered features generally match or outperform raw inputs.

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014).** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research*, 15, 1929-1958. Foundational dropout paper. Recommends rates of 0.2-0.5 for hidden layers.

- **Tsantekidis, A., Passalis, N., Tefas, A., Kanniainen, J., Gabbouj, M. & Iosifidis, A. (2017).** "Forecasting Stock Prices from the Limit Order Book Using Convolutional Neural Networks." *IEEE Conference on Business Informatics*, 7-12. Uses pre-computed order book features as inputs to deep networks for financial prediction.
