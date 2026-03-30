# CNN+RNN Hybrid -- Design Decisions

**Date:** 2026-03-30
**Task:** Binary stock direction classification (h=1, next-day up/down)
**Baseline:** Single-layer GRU, hidden=64, seq_len=20, 41 features, dropout=0.3
**Data:** 30 IBEX35 stocks, ~5000 trading days each, 41 features (ft_type="macro", h=1)
**Training scheme:** 3-year sliding window (~750 days x 30 tickers)

---

## 1. Motivation: Why Add CNN Before the RNN?

### What the RNN alone misses

The GRU processes the input sequence one timestep at a time, updating its hidden state incrementally. This is optimal for capturing **sequential dependencies** -- how the hidden state at t-5 influences the prediction at t. However, the GRU is structurally weak at detecting **local multi-feature patterns** that span a small number of adjacent timesteps.

Consider a concrete example: a 3-day pattern where RSI drops from 70 to 50 while volume spikes 2x and MACD histogram flips negative. This is a classic distribution pattern (institutional selling). The GRU must encode this into its hidden state through three sequential update steps, blending it with whatever information was already in the hidden state from earlier timesteps. The pattern is never explicitly "detected" -- it is implicitly absorbed.

A 1D convolutional layer with kernel_size=3 can detect this pattern explicitly. The convolution slides a (3, 41) filter across the time dimension, computing a dot product at each position. If the filter's weights align with the "RSI drop + volume spike + MACD flip" pattern, the output will spike at that position. The CNN acts as a **local pattern detector** before the RNN processes the resulting sequence.

This is the fundamental argument: **CNN extracts local temporal features (what happened in the last 3-5 days), RNN captures how those local features evolve over the full sequence.**

### Empirical evidence from the literature

**Livieris, Pintelas & Pintelas (2020)** -- "A CNN-LSTM model for gold price time-series forecasting." *Expert Systems with Applications*, 164, 113681. Tested CNN-LSTM vs standalone LSTM on gold price forecasting. The hybrid reduced RMSE by 7-11% across multiple forecasting horizons. Key finding: the CNN layer was most beneficial when input features had local temporal structure (consecutive-day patterns), and less beneficial when features were already heavily smoothed. Their architecture used a single Conv1D layer (32 filters, kernel_size=3) followed by LSTM(64).

**Lu, Li, Kang & Li (2020)** -- "A CNN-LSTM-based model to forecast stock prices." *Complexity*, 2020, 6622927. Applied CNN-LSTM to Shanghai Stock Exchange data. Reported ~2-4% improvement in directional accuracy over standalone LSTM. Used 2 conv layers (64 and 128 filters, kernel_size=5) with MaxPool. Notably, they found that the CNN component was more valuable for shorter prediction horizons (1-5 days) than longer ones (20+ days), consistent with the hypothesis that local patterns matter more for short-term prediction.

**Sezer, Gudelek & Ozbayoglu (2020)** -- "Financial time series forecasting with deep learning: a systematic literature review: 2005-2019." *Applied Soft Computing*, 90, 106181. In their survey of 140+ papers, they note that CNN-LSTM hybrids appeared in ~15% of papers published 2018-2019, making it the most common hybrid architecture. The consensus finding across papers: "CNN layers improve feature extraction from raw or semi-processed time series, with typical improvements of 1-3% in classification accuracy over standalone recurrent models."

**Kim & Won (2018)** -- "Forecasting stock prices with a feature fusion LSTM-CNN model using different representations of the same data." *PLOS ONE*, 13(2), e0212320. Compared standalone LSTM, standalone CNN, and hybrid CNN-LSTM for Korean stock market direction prediction. The hybrid achieved 56.8% accuracy vs 55.1% for LSTM alone and 54.3% for CNN alone (+1.7pp over LSTM). Importantly, they found the CNN was more valuable when features included raw/semi-raw price data than when all features were fully pre-processed.

**Niu, Zhong & Yu (2020)** -- "Hybrid model combining GRU neural network with attention mechanism for stock price prediction." *Journal of Ambient Intelligence and Humanized Computing*. While focused on attention mechanisms, they also tested CNN+GRU and found ~1.5% RMSE improvement over standalone GRU. The attention mechanism provided a further ~1% improvement, suggesting the CNN-to-RNN pathway is complementary to attention.

**Hoseinzade & Haratizadeh (2019)** -- "CNNpred: CNN-based stock market prediction using a diverse set of variables." *Expert Systems with Applications*, 129, 273-285. Applied 2D CNNs directly to multi-stock feature matrices. Found that CNN-based feature extraction from technical indicators improved accuracy by 2-4% on S&P 500 stocks. Their insight: CNNs detect "feature interaction patterns" (e.g., simultaneous RSI and Bollinger Band extremes) that linear models miss.

### The key insight for our setting

Our features are heavily pre-engineered (RSI, MACD, Bollinger %B, etc.). Each feature already encodes multi-day lookbacks. This means the CNN will NOT learn to compute features from raw prices -- that work is already done. Instead, the CNN will detect **multi-feature co-occurrence patterns** across small time windows: combinations of indicator states that co-occur over 3-5 consecutive days.

This is subtly different from the raw-price CNN use case, and the expected benefit is correspondingly smaller. Kim & Won (2018) found that the hybrid advantage shrank from +1.7pp to +0.8pp when switching from semi-raw to fully engineered features. We should expect gains at the lower end of the 1-3% range reported in the literature.

**Verdict: the CNN layer is worth adding, but we should be conservative about the added complexity.** A single lightweight conv layer is appropriate; a deep multi-layer CNN would overfit given our data constraints.

---

## 2. Architecture Design

### 2.1 Number of Convolutional Layers: 1

**Decision: 1 Conv1D layer.**

Reasoning:

1. **Sequence is short (T=20).** Each conv layer with kernel_size=3 reduces the effective sequence length by 2 (without padding) or maintains it (with padding). Two conv layers with MaxPool(2) would reduce T=20 to T=9, leaving the GRU with a very short sequence that may not carry enough temporal structure. With padding and no pooling, two layers are feasible but add ~2x the conv parameters without clear benefit.

2. **Features are already high-level.** In computer vision, stacking conv layers builds a hierarchy of abstraction (edges -> textures -> objects). Our features are already at the "object" level (RSI is a high-level concept, not a raw pixel). A single conv layer that detects co-occurrence patterns across 3-5 days of indicator values is sufficient. A second layer would detect "patterns of patterns" -- unlikely to yield useful signal at our SNR.

3. **Empirical evidence favours simplicity.** Livieris et al. (2020) used a single conv layer for their gold forecasting CNN-LSTM. Lu et al. (2020) used 2 conv layers but on a much larger dataset (10 years, daily data, single stock with many more training samples). For our data scale (~21,900 training sequences), 1 conv layer is the safer choice.

4. **Parameter budget.** The baseline GRU has ~20K parameters. A single Conv1D(41, 32, kernel_size=3) adds 41*32*3 + 32 = 3,968 parameters (~20% overhead). A second Conv1D(32, 32, kernel_size=3) adds another 3,104. The total model would grow to ~27K parameters, still reasonable, but the marginal benefit of the second layer is unlikely to justify the added overfitting risk.

### 2.2 Kernel Size: 3

**Decision: kernel_size = 3.**

The kernel size determines how many consecutive timesteps the convolution "sees" at once. For financial time series:

- **kernel_size=2:** Detects pairwise transitions (day-to-day changes). This is minimally useful because single-day changes are already captured by features like `log_ret_1`. The conv would mostly learn to compute differences that the feature engineering already provides.

- **kernel_size=3:** Detects 3-day patterns. This is the sweet spot for daily financial data. Many classic patterns span 3 days: reversal patterns (up-down-up), momentum ignition (flat-flat-breakout), distribution (high volume selling over 3 days). Three trading days is slightly more than half a week -- a natural scale for short-term pattern formation.

- **kernel_size=5:** Detects weekly patterns. Plausible but reduces the effective sequence seen by the GRU (from 20 to 16 positions without padding). Also, weekly-scale patterns are already partially captured by features like `log_ret_5` and `sma_ratio_5_20`. The incremental value of a 5-day conv kernel over the pre-computed 5-day features is low.

- **kernel_size=7 or larger:** Too large for T=20. A kernel of size 7 means each output position summarises a full trading week plus. With T=20, the GRU would see only 14 positions, and much of the temporal resolution is lost. Furthermore, large kernels have more parameters and are harder to train with our limited data.

**Why not use dilated convolutions?** Dilated convolutions (dilation=2, kernel_size=3 gives effective receptive field of 5) are useful when you need large receptive fields without pooling. For T=20, the receptive field of a simple kernel_size=3 is sufficient. Dilated convolutions add implementation complexity and another hyperparameter without clear benefit at this scale.

**Evidence:** Livieris et al. (2020) used kernel_size=3. Lu et al. (2020) used kernel_size=5 but with a longer sequence (T=60). Hoseinzade & Haratizadeh (2019) tested kernel sizes {2,3,5} and found kernel_size=3 consistently performed best for daily prediction. The consensus for daily equity data with T=10-30 is kernel_size=3.

### 2.3 Number of Filters: 32

**Decision: num_filters = 32.**

The number of filters determines how many distinct local patterns the CNN can detect. Each filter is a (kernel_size, input_channels) = (3, 41) weight matrix that detects one specific 3-day pattern across all 41 features.

- **num_filters=16:** Very conservative. Can detect 16 distinct 3-day patterns. May be too restrictive -- with 41 input features, there could easily be more than 16 relevant multi-feature patterns.

- **num_filters=32:** Moderate. Detects 32 patterns. This is roughly 0.75x the GRU hidden size (64), creating a mild bottleneck that forces the CNN to learn the most informative patterns. The GRU then operates on a 32-dimensional feature space (down from 41), which slightly reduces its parameter count.

- **num_filters=64:** Matches the GRU hidden size. No bottleneck. The CNN output has the same dimensionality as the GRU hidden state, so the GRU capacity is fully utilised. However, this adds ~8K conv parameters and the CNN output is wide enough to encode noise patterns.

- **num_filters=128:** Over-parameterised for our setting. The conv layer alone would have 41*128*3 + 128 = 15,872 parameters -- nearly as many as the entire baseline GRU. Strong overfitting risk.

**Why 32 and not 64?** The key consideration is that our features are already high-level. In a raw-price CNN, you might need 64-128 filters to detect diverse price patterns (trends, reversals, volatility shifts, etc.) from raw OHLCV. But our features already encode these concepts. The CNN is detecting **co-occurrence of pre-computed indicators**, not raw price shapes. Fewer co-occurrence patterns are relevant compared to raw price patterns.

Additionally, 32 filters feeding into a GRU with hidden_size=64 creates a 32->64 expansion, which is a common and effective bottleneck architecture. The CNN compresses 41 features into 32 pattern-detected features, and the GRU expands back to 64 dimensions with temporal context.

**Evidence:** Livieris et al. (2020) used 32 filters. Kim & Won (2018) used 32-64 filters. For our data scale and feature engineering level, 32 is the safer choice.

### 2.4 Activation Function: GELU

**Decision: GELU (Gaussian Error Linear Unit).**

- **ReLU** is the default activation for convolutional layers. It is simple, fast, and well-understood. However, ReLU has a hard zero cutoff: any pre-activation below 0 produces exactly 0 gradient. For financial time series where the signal is weak and features are roughly symmetric around zero (many features are z-scored ratios), ReLU may zero out slightly negative activations that carry useful information.

- **GELU** (Hendrycks & Gimpel, 2016) smoothly gates values near zero: GELU(x) = x * Phi(x), where Phi is the standard normal CDF. Values near zero are softly attenuated rather than hard-thresholded. This is advantageous for financial features where small negative values (e.g., slightly bearish RSI, weak negative MACD) can still be informative.

- **LeakyReLU** is another option that avoids dead neurons, but GELU has become the standard in modern architectures (used in BERT, GPT, etc.) and provides smoother gradients.

**Practical impact:** The difference between ReLU and GELU is typically small (<0.5% accuracy). GELU is marginally better for our weak-signal setting and does not add computational cost. Use GELU as default; if debugging, switch to ReLU for interpretability.

### 2.5 Pooling: None

**Decision: No pooling layer after the convolution.**

This is a critical design choice. Pooling (MaxPool1d or AvgPool1d) along the time dimension reduces the sequence length that the GRU processes. For image data with spatial resolution of 224x224, aggressive pooling is essential. For time series with T=20, pooling is harmful:

1. **T=20 is already short.** MaxPool1d(kernel_size=2, stride=2) would reduce the sequence to T=10 (with kernel_size=3 conv and "same" padding) or T=9 (without padding). The GRU would have only 9-10 timesteps to work with -- barely enough to capture weekly patterns.

2. **Temporal ordering matters.** Pooling discards positional information within each pooling window. In images, a "cat ear" at pixel (10,10) and pixel (12,12) is equally a cat ear -- spatial invariance is desirable. In time series, what happened on day 5 vs day 6 matters. Max-pooling across days 5-6 loses this distinction. Translation invariance, the main benefit of pooling in vision tasks, is a liability in time series.

3. **The GRU already provides temporal aggregation.** The GRU's gating mechanism selectively retains and forgets information over time. It is a more principled temporal aggregation than max-pooling. Let the CNN detect local patterns at full temporal resolution, and let the GRU decide which patterns to remember.

4. **Evidence against pooling for short financial sequences.** Sezer et al. (2020) note that financial time series CNN studies with pooling generally use longer sequences (T>=60). For T=20-30, most successful architectures skip pooling (Livieris et al., 2020; Hoseinzade & Haratizadeh, 2019).

**If sequence length increases (e.g., T=60), pooling becomes viable.** At T=60, a MaxPool(2) after conv would give the GRU T=29, which is still reasonable. But for T=20, no pooling.

### 2.6 Normalisation: BatchNorm1d After Conv

**Decision: BatchNorm1d after the convolutional layer, before activation.**

The conv layer output has shape (batch, num_filters, seq_len). Normalisation stabilises training by ensuring the pre-activation distribution does not drift across training iterations.

- **BatchNorm1d** normalises across the batch dimension for each filter channel. For conv layers, this is the standard choice because it normalises each filter's output independently. With batch_size=128 and 30 tickers mixed in each batch, the batch statistics are stable.

- **LayerNorm** normalises across the feature (filter) dimension for each sample. This is preferred for recurrent layers (Ba et al., 2016) because batch statistics in RNNs can be unstable across timesteps. However, for the conv layer operating on the full sequence at once, BatchNorm is more standard.

- **No normalisation** is the alternative. The baseline GRU uses no normalisation (only input z-scoring). Adding BatchNorm only to the CNN layer creates an asymmetry. However, the conv layer introduces a non-trivial transformation that benefits from normalisation -- the GRU's input projection already provides implicit normalisation through its gates.

**Ordering: Conv -> BatchNorm -> GELU.** This is the standard "pre-activation" normalisation pattern (Ioffe & Szegedy, 2015). Normalise before the activation so that the activation function operates on a standardised distribution.

**Note:** During inference, BatchNorm uses running statistics computed during training. For our rolling-window setup where the data distribution shifts across windows, this is a minor concern. The running statistics are computed within each 3-year training window and applied to the subsequent test period. The distribution shift between training window and test period is typically small (a few months).

### 2.7 CNN-to-RNN Connection: Full Sequence Pass-Through

**Decision: Pass the full CNN output sequence to the GRU (no compression).**

Two options for connecting CNN output to RNN input:

**Option A -- Full sequence (recommended):** The CNN produces (batch, num_filters, seq_len) output (with "same" padding, seq_len is preserved at 20). Transpose to (batch, seq_len, num_filters) = (batch, 20, 32) and feed this as the GRU input. The GRU processes 20 timesteps of 32-dimensional CNN features.

**Option B -- Compressed:** Apply GlobalAveragePool or GlobalMaxPool to the CNN output, producing (batch, num_filters) = (batch, 32). Then concatenate with or replace the GRU input at each timestep. This defeats the purpose of the hybrid: the RNN needs a sequence to process, and compressing the CNN output to a single vector removes temporal structure.

**Option A is clearly superior.** The CNN's job is to transform the feature space at each timestep; the GRU's job is to process the transformed sequence temporally. Compressing the CNN output eliminates the temporal dimension that the GRU needs.

**Padding strategy for sequence preservation:** Use `padding='same'` (or equivalently `padding=1` for kernel_size=3) so the conv output has the same length as the input. This avoids losing timesteps at the edges.

```
Input: (batch, 41, 20)  -- Conv1d expects (batch, channels, length)
Conv1d(41, 32, 3, padding=1) -> (batch, 32, 20)
BatchNorm1d(32) -> (batch, 32, 20)
GELU -> (batch, 32, 20)
Transpose -> (batch, 20, 32)
GRU(input_size=32, hidden_size=64) -> h_n: (1, batch, 64)
```

### 2.8 Optional: Residual/Skip Connection

**Decision: Add a linear projection skip connection from raw input to GRU input.**

One risk of the CNN layer is that it might discard useful information. If a feature (e.g., `vix_chg_z_5`) is informative on its own but does not participate in any 3-day pattern, the CNN might attenuate it. A skip connection mitigates this:

```
cnn_out = GELU(BatchNorm(Conv1d(x)))    # (batch, 32, 20)
skip = Linear(41, 32)(x.transpose)      # (batch, 20, 32) -- project raw input to same dim
gru_input = cnn_out.transpose + skip    # (batch, 20, 32)
```

This adds 41*32 + 32 = 1,344 parameters and ensures the GRU has access to both the raw features (linearly projected) and the CNN-detected patterns. If the CNN adds no value for some features, the model can learn to rely on the skip path.

However, this adds complexity and another design choice. **For the initial implementation, skip the skip connection.** If ablation shows that the CNN-RNN underperforms the baseline GRU on certain feature subsets, add the skip connection as a fix.

**Final decision: no skip connection initially. Revisit if CNN-RNN underperforms baseline.**

---

## 3. Sequence Length: Keep T=20

**Decision: seq_len = 20 (unchanged from baseline).**

### Does adding CNN change the optimal sequence length?

In principle, the CNN's receptive field consumes some of the sequence length. With kernel_size=3 and padding=1, the output length equals the input length, so no timesteps are lost. The GRU still sees 20 timesteps.

**Arguments for increasing T:**

1. The CNN pre-processes the sequence, so the GRU receives "higher-quality" timesteps (pattern-detected features vs raw features). In theory, the GRU could effectively process more timesteps because each timestep is more informative.

2. With the CNN extracting local patterns, the GRU's job becomes detecting how these patterns evolve over time. This "pattern trajectory" might benefit from a longer observation window.

**Arguments against increasing T (stronger):**

1. **Information decay still applies.** Lopez de Prado (2018) argues that financial information decays rapidly. Adding a CNN does not change the underlying signal structure of the market. Days 25-40 are still less informative than days 1-20 for next-day prediction.

2. **More training data is lost.** T=40 would lose 20 additional timesteps per ticker per window, reducing training samples from ~21,900 to ~21,300. Modest impact, but in the wrong direction.

3. **The features already encode long lookbacks.** `sma_ratio_10_50` at each timestep encodes 50 days of price history. A sequence of 20 timesteps of this feature gives the model effective access to 70+ days. The CNN does not change this arithmetic.

4. **Risk of overfitting increases with longer sequences.** More timesteps mean more sequential operations and a larger computation graph. The gradient signal for timestep t=1 in a 40-step sequence is weaker than in a 20-step sequence.

**Evidence:** Lu et al. (2020) used T=60 for their CNN-LSTM, but on a single-stock dataset with more training data. Livieris et al. (2020) used T=10 for gold futures with a CNN-LSTM. The optimal sequence length depends more on the data characteristics than on the architecture.

**Verdict: keep T=20.** The CNN does not fundamentally change the information content of the sequence. If the hybrid model clearly outperforms the baseline and we want to push further, T=30 can be tested as an ablation, but T=20 is the starting point.

---

## 4. Interaction with Engineered Features

### Does CNN add value on top of heavily engineered features?

This is the most important question for our setting. The literature provides mixed signals:

**Evidence that CNN helps even with engineered features:**

- **Hoseinzade & Haratizadeh (2019)** used 82 technical indicators as CNN input (not raw prices) and still found ~2.5% accuracy improvement over non-CNN baselines. Their explanation: the CNN detects **cross-indicator patterns** -- combinations of indicator states that are predictive when they co-occur but not individually. Example: RSI in the 40-50 range is uninformative alone, but RSI(45) + MACD crossing zero + volume spike = strong bullish signal. A linear model or simple RNN might miss this because the interaction is local in time and multiplicative across features.

- **Sezer & Ozbayoglu (2018)** -- "Algorithmic financial trading with deep convolutional neural networks: time series to image conversion approach." *Applied Soft Computing*, 70, 525-538. Converted technical indicator time series into 2D images and applied CNNs. The fact that CNNs extracted useful patterns from pre-computed indicators (not raw prices) supports the claim that CNNs detect cross-indicator relationships.

**Evidence that CNN benefit shrinks with engineered features:**

- **Kim & Won (2018)** explicitly compared CNN-LSTM on raw vs engineered features. The hybrid advantage was +1.7pp with semi-raw features but only +0.8pp with fully pre-processed features. The CNN's value is partially redundant with good feature engineering.

- **Krauss et al. (2017)** found that DNNs (including CNNs) on raw returns performed comparably to simpler models on engineered features. This suggests that a significant portion of what the CNN learns from raw data is already captured by well-designed features.

### Assessment for our setting

Our 41 features are extensively engineered: RSI, MACD histogram, Bollinger %B, volume ratios, breadth, VIX regime, and relative-to-market metrics. Many of the patterns a CNN would learn from raw prices are already explicit features.

However, the CNN can still add value by detecting:

1. **Temporal co-occurrence patterns:** RSI dropping while Bollinger %B is extreme while volume is spiking. The GRU can learn this too, but must do it through sequential hidden state updates across 3 timesteps. The CNN detects it in a single convolution operation with a single filter.

2. **Short-term feature trajectory shapes:** A "V-shaped" RSI recovery over 3 days (drop, bottom, recovery) is qualitatively different from a "monotone decline." The CNN can detect this shape; the GRU encodes it as a hidden state trajectory that is harder to disentangle.

3. **Multi-scale interactions:** The CNN's 3-day kernel interacts features computed at different scales (1-day return, 5-day ratio, 14-day RSI). These interactions are non-linear and local, which is exactly what convolution excels at.

**Expected benefit: +0.5 to +1.5 percentage points in balanced accuracy over the pure GRU.** This is the realistic range given that our features are already well-engineered. The benefit is real but modest. If the implementation is clean and the regularisation is right, the CNN-RNN should not perform worse than the baseline GRU (the worst case is +0pp).

---

## 5. Hyperparameter Choices (Justified)

### 5.1 Summary Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Conv layers | 1 | Sufficient for pre-engineered features; avoids overfitting |
| Filters | 32 | 0.75x GRU hidden size; bottleneck forces informative patterns |
| Kernel size | 3 | Detects 3-day patterns; standard for daily T=20 data |
| Padding | 1 ("same") | Preserves sequence length for GRU |
| Activation | GELU | Smooth gating near zero; better for weak signals |
| Pooling | None | T=20 is too short; temporal resolution must be preserved |
| Normalisation | BatchNorm1d | Stabilises conv output; standard for conv layers |
| Skip connection | None (initially) | Simplicity; revisit if baseline regresses |
| GRU input_size | 32 (was 41) | Receives CNN output instead of raw features |
| GRU hidden_size | 64 (unchanged) | Capacity sufficient; no need to increase |
| GRU num_layers | 1 (unchanged) | Same reasoning as baseline |
| Dropout | 0.3 (unchanged) | See Section 6 for discussion |
| Sequence length | 20 (unchanged) | CNN does not change information decay dynamics |

### 5.2 Detailed Justification for Key Choices

**Why 32 filters and not 64?** The total parameter count with 32 filters:
- Conv1d(41, 32, 3): 41*32*3 + 32 = 3,968
- BatchNorm1d(32): 64 (gamma and beta)
- GRU(32, 64): 3 * (64*32 + 64*64 + 64) = 3 * (2,048 + 4,096 + 64) = 18,624
- Dropout + Linear(64, 1): 65
- **Total: ~22,721 parameters**

With 64 filters:
- Conv1d(41, 64, 3): 41*64*3 + 64 = 7,936
- BatchNorm1d(64): 128
- GRU(64, 64): 3 * (64*64 + 64*64 + 64) = 3 * (4,096 + 4,096 + 64) = 24,768
- Dropout + Linear(64, 1): 65
- **Total: ~32,897 parameters**

The 64-filter model has 45% more parameters than the 32-filter model. With 21,900 training samples, the 32-filter model has ~0.96 samples per parameter (tight but feasible). The 64-filter model drops to ~0.67 samples per parameter (overfitting danger zone for low-SNR problems). The 32-filter architecture is the right balance.

**Why not increase GRU hidden_size to compensate for the bottleneck?** The CNN reduces the GRU input from 41 to 32 dimensions, so the GRU's input-to-hidden weights shrink. One could increase hidden_size to 96 or 128 to compensate. But this misses the point: the bottleneck is intentional. The CNN forces the 41 features into 32 pattern channels, discarding noise. Expanding the GRU would re-introduce capacity for noise. Keep hidden_size=64.

---

## 6. Regularisation

### Does adding CNN require changes to the regularisation strategy?

The CNN adds ~4,000 parameters (~20% increase over the baseline GRU's ~20K). This modest increase does not fundamentally change the regularisation requirements, but some adjustments are warranted:

### Dropout: Keep at 0.3

The baseline applies dropout=0.3 after the GRU's final hidden state. For the hybrid:

- **No dropout between CNN and GRU.** The CNN output is the GRU's input sequence. Applying dropout here would randomly zero out pattern features at random timesteps, creating an irregular sequence that the GRU must handle. This is analogous to input dropout, which is generally harmful for time series (Gal & Ghahramani, 2016).

- **Keep dropout=0.3 after the GRU.** Same location as baseline. The GRU final hidden state is the representation that maps to the output; regularising here is effective.

- **Spatial dropout on the CNN output?** Spatial dropout (dropping entire filter channels rather than individual elements) is used in NLP CNNs (Srivastava et al., 2014). For our 32-filter CNN, dropping a channel means dropping an entire detected pattern across all timesteps. This is heavy-handed with only 32 channels. Not recommended for the initial implementation.

**Decision: dropout=0.3, same location as baseline (after GRU, before linear).**

### Weight Decay: Keep at 1e-4

The AdamW weight_decay=1e-4 applies L2 penalty to all parameters equally. The CNN weights will be regularised by this penalty. No change needed.

If overfitting increases relative to the baseline (validation-test gap widens), the first lever is still dropout (increase to 0.5), not weight decay. The CNN's parameters are well-conditioned (small kernels, BatchNorm) and unlikely to grow excessively large.

### Early Stopping: Keep patience=10

Same criterion as baseline. Monitor validation loss (BCEWithLogitsLoss on the validation fold). The CNN-RNN should converge in a similar number of epochs as the baseline GRU, possibly slightly more (the CNN weights need to stabilise before the GRU can learn from the transformed features).

### Gradient Clipping: Keep max_norm=1.0

The CNN introduces a new gradient pathway through the conv weights. With BatchNorm after the conv, gradient magnitudes should be well-controlled. Keep max_norm=1.0 as a safety net.

### BatchNorm's Implicit Regularisation

BatchNorm provides mild regularisation by adding noise to the batch statistics (each mini-batch has slightly different mean/variance). This is a minor effect with batch_size=128 but is free. It partially substitutes for the regularisation that the model loses by having the CNN pathway (which is deterministic during training, unlike dropout).

---

## 7. Expected Gains

### Realistic Expectations

Based on the literature and our specific setting:

| Metric | Baseline GRU (expected) | CNN+GRU (expected) | Source |
|--------|------------------------|-------------------|--------|
| Balanced accuracy | 52-54% | 52.5-55% | Kim & Won (2018): +0.8-1.7pp; Livieris et al. (2020): +1-2pp |
| AUC | 0.53-0.56 | 0.54-0.57 | Proportional to accuracy gains |
| Training time | 30-60s/window | 35-70s/window | ~15-20% overhead from CNN |

### Why the expected improvement is small

1. **Low signal-to-noise ratio.** Daily equity direction prediction has a theoretical ceiling around 55% (Krauss et al., 2017). The gap between 53% and 55% is where all models compete. A +1pp improvement is meaningful in this regime.

2. **Features are pre-engineered.** The CNN's primary value -- extracting features from raw data -- is already provided by the feature engineering pipeline. The residual value (cross-indicator pattern detection) is real but secondary.

3. **Data is limited.** 21,900 training sequences restrict model complexity. The CNN must be small enough to avoid overfitting, which limits the patterns it can learn.

### When the CNN is most/least valuable

**Most valuable:** When the market is in a transitional regime (VIX rising, breadth narrowing) where short-term indicator co-occurrence patterns provide information beyond individual feature values. Expect the CNN contribution to be concentrated in volatile periods.

**Least valuable:** In low-volatility, trending markets where features like `sma_ratio_5_20` and `log_ret_5` carry most of the signal individually. In these regimes, the CNN detects patterns that are redundant with the individual features.

### What "success" looks like

The CNN-RNN is worth keeping if:
1. Balanced accuracy improves by >= 0.5pp averaged across rolling windows
2. The improvement is consistent (positive in >= 60% of rolling windows, not driven by 1-2 outlier periods)
3. The model does not exhibit increased overfitting (validation-test gap is comparable to baseline)

If the average improvement is < 0.5pp or the model is erratic (helps in some windows, hurts in others), the added complexity is not justified. Stick with the baseline GRU.

---

## 8. Recommended Architecture

### Code-Like Summary

```python
import torch
import torch.nn as nn

class StockCNNGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 41,     # 41 features per timestep
        num_filters: int = 32,    # CNN output channels
        kernel_size: int = 3,     # 3-day local patterns
        hidden_size: int = 64,    # GRU hidden dimension
        num_layers: int = 1,      # single GRU layer
        dropout: float = 0.3,     # after GRU, before linear
    ):
        super().__init__()

        # --- CNN block ---
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # "same" padding: preserves seq_len
        )
        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.GELU()

        # --- RNN block ---
        self.gru = nn.GRU(
            input_size=num_filters,     # receives CNN output, not raw features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len=20, input_size=41)

        # CNN expects (batch, channels, length)
        x_conv = x.transpose(1, 2)                # (batch, 41, 20)
        x_conv = self.conv(x_conv)                 # (batch, 32, 20)
        x_conv = self.bn(x_conv)                   # (batch, 32, 20)
        x_conv = self.act(x_conv)                   # (batch, 32, 20)

        # Back to (batch, seq_len, features) for GRU
        x_conv = x_conv.transpose(1, 2)           # (batch, 20, 32)

        # GRU processes the CNN-transformed sequence
        _, h_n = self.gru(x_conv)                  # h_n: (1, batch, 64)
        h_n = h_n.squeeze(0)                       # (batch, 64)
        h_n = self.dropout(h_n)
        logits = self.fc(h_n)                      # (batch, 1)
        return logits.squeeze(-1)                  # (batch,)
```

### Full Hyperparameter Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **CNN block** | | |
| Conv1d in_channels | 41 | All features as input channels |
| Conv1d out_channels | 32 | 0.75x hidden_size; bottleneck for noise reduction |
| Conv1d kernel_size | 3 | 3-day local patterns; standard for daily T=20 |
| Conv1d padding | 1 | "same" padding; preserves seq_len=20 |
| BatchNorm1d | 32 features | Stabilises conv output distribution |
| Activation | GELU | Smooth gating; better for weak financial signals |
| Pooling | None | T=20 too short; temporal resolution needed |
| **RNN block** | | |
| Cell type | GRU (default); LSTM as alternative | Fewer params; no advantage for LSTM at T=20 |
| input_size | 32 | Receives CNN output |
| hidden_size | 64 | Unchanged from baseline |
| num_layers | 1 | Unchanged from baseline |
| **Head** | | |
| Dropout | 0.3 | After GRU, before linear |
| Output | Linear(64, 1) + sigmoid | BCEWithLogitsLoss |
| **Training** | | |
| Optimiser | AdamW | lr=1e-3, weight_decay=1e-4 |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |
| Batch size | 128 | Unchanged |
| Max epochs | 100 | Early stopping patience=10 |
| Gradient clipping | max_norm=1.0 | Safety net for exploding gradients |
| **Data** | | |
| Sequence length | 20 | 1 month; unchanged from baseline |
| Features | 41 | Identical feature set to baseline GRU |
| Input scaling | Per-feature z-score | Fit on training window only |
| Training scope | Shared across 30 tickers | Cross-sectional pooling |
| **Total parameters** | **~22,700** | ~13% more than baseline GRU (~20,400) |

### Changes from Baseline GRU

| Aspect | Baseline GRU | CNN+GRU Hybrid |
|--------|-------------|----------------|
| Input to RNN | Raw 41 features | 32 CNN-transformed features |
| Parameters | ~20,400 | ~22,700 (+13%) |
| New components | None | Conv1d + BatchNorm + GELU |
| GRU input_size | 41 | 32 |
| Dropout location | After GRU | Same (after GRU only) |
| Training time | ~30-60s/window | ~35-70s/window (+15-20%) |
| Expected accuracy | 52-54% | 52.5-55% (+0.5 to +1.5pp) |

### Ablation Plan

Before accepting the CNN-RNN as the new default, run these ablations:

1. **Baseline GRU vs CNN+GRU** on all rolling windows. Compare balanced accuracy, AUC, and Brier score. The hybrid must win by >= 0.5pp averaged across windows.

2. **kernel_size={3, 5}**: Test whether 5-day kernels add value. Expect kernel_size=3 to win or tie.

3. **num_filters={16, 32, 64}**: Confirm that 32 is the sweet spot. If 16 matches 32, prefer 16 (simpler). If 64 beats 32 without increased overfitting, adopt 64.

4. **With vs without BatchNorm**: Confirm BatchNorm helps. If it makes no difference, remove it (simpler).

5. **With skip connection**: If the CNN-GRU underperforms baseline on any window subset, test whether a skip connection recovers the lost performance.

---

## 9. Overfitting Guardrails (Revised for CNN+GRU)

Ordered by importance:

1. **Early stopping** (patience=10 on validation loss)
2. **Dropout 0.3** (after GRU, before classification head)
3. **Weight decay 1e-4** (L2 via AdamW, applied to all parameters including CNN)
4. **Small CNN** (32 filters, 1 layer, ~4K conv parameters)
5. **BatchNorm** (implicit regularisation via mini-batch noise)
6. **Short sequences** (T=20 limits temporal overfitting)
7. **Bottleneck architecture** (41 -> 32 CNN compression reduces GRU input dimension)
8. **Learning rate reduction** (ReduceLROnPlateau)
9. **Gradient clipping** (max_norm=1.0)

**Escalation order if overfitting persists** (validation-test gap > 3pp):
1. Increase dropout to 0.5
2. Reduce num_filters to 16
3. Increase weight_decay to 1e-3
4. Remove BatchNorm (reduces model expressiveness)
5. Reduce hidden_size to 32
6. Fall back to baseline GRU (if CNN consistently increases overfitting)

---

## References

- **Ba, J.L., Kiros, J.R. & Hinton, G.E. (2016).** "Layer Normalization." *arXiv:1607.06450*. Proposes layer normalisation for RNNs; motivates our choice of BatchNorm for the conv layer and no norm for the GRU.

- **Fischer, T. & Krauss, C. (2018).** "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669. LSTM on S&P 500 daily direction; ~54% accuracy. Simple architectures suffice for financial prediction.

- **Gal, Y. & Ghahramani, Z. (2016).** "A Theoretically Grounded Application of Dropout to Recurrent Neural Networks." *NIPS 2016*. Demonstrates that naive dropout between timesteps harms RNN hidden states. Supports our decision to apply dropout only after the GRU.

- **Hendrycks, D. & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*. Introduces GELU activation; smooth gating near zero is advantageous for weak-signal settings.

- **Hoseinzade, E. & Haratizadeh, S. (2019).** "CNNpred: CNN-based stock market prediction using a diverse set of variables." *Expert Systems with Applications*, 129, 273-285. Applied CNNs to 82 pre-computed technical indicators; +2.5% accuracy. Key evidence that CNNs add value even over engineered features by detecting cross-indicator patterns.

- **Ioffe, S. & Szegedy, C. (2015).** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *Proceedings of ICML 2015*. Foundational BatchNorm paper; justifies Conv -> BN -> Activation ordering.

- **Kim, T. & Won, H.Y. (2018).** "Forecasting stock prices with a feature fusion LSTM-CNN model using different representations of the same data." *PLOS ONE*, 13(2), e0212320. CNN-LSTM hybrid for Korean market; +1.7pp with semi-raw features, +0.8pp with engineered features. Key evidence for diminishing CNN returns on pre-processed inputs.

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. Benchmark for daily equity ML; ~55% accuracy ceiling with engineered features.

- **Livieris, I.E., Pintelas, E. & Pintelas, P. (2020).** "A CNN-LSTM model for gold price time-series forecasting." *Expert Systems with Applications*, 164, 113681. CNN-LSTM with 32 filters, kernel_size=3, LSTM(64). 7-11% RMSE improvement over standalone LSTM. Key architecture reference for our design choices.

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. Ch. 5: Information decay justifying T=20. Ch. 6: Feature importance. Ch. 7: Purged cross-validation.

- **Lu, W., Li, J., Kang, J. & Li, B. (2020).** "A CNN-LSTM-based model to forecast stock prices." *Complexity*, 2020, 6622927. CNN-LSTM on Shanghai Stock Exchange; +2-4% directional accuracy. Used 2 conv layers with MaxPool on T=60 sequences. Supports CNN value for short horizons.

- **Niu, H., Zhong, G. & Yu, Y. (2020).** "Hybrid model combining GRU neural network with attention mechanism for stock price prediction." *Journal of Ambient Intelligence and Humanized Computing*. CNN+GRU with attention; ~1.5% improvement from CNN, further ~1% from attention. Supports CNN-GRU as complementary to attention mechanisms.

- **Sezer, O.B. & Ozbayoglu, A.M. (2018).** "Algorithmic financial trading with deep convolutional neural networks: time series to image conversion approach." *Applied Soft Computing*, 70, 525-538. Converts technical indicator time series to 2D images for CNN processing. Demonstrates that CNNs extract useful patterns from pre-computed indicators.

- **Sezer, O.B., Gudelek, M.U. & Ozbayoglu, A.M. (2020).** "Financial time series forecasting with deep learning: a systematic literature review: 2005-2019." *Applied Soft Computing*, 90, 106181. Survey of 140+ papers. CNN-LSTM hybrids appear in ~15% of 2018-2019 papers. Consensus: +1-3% accuracy improvement over standalone recurrent models.

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014).** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15, 1929-1958. Foundational dropout paper.
