# Feature Engineering Decisions for IBEX35 Binary Direction Classification

**Date:** 2026-03-30
**Horizons:** h=1 (next-day), h=5 (next-week)
**Models:** Random Forest, XGBoost, GRU, LSTM

---

## Executive Summary

The current feature set is a solid foundation: ~39 features covering momentum, volatility, trend, volume, and macro regime. However, several improvements are warranted:

1. **Remove 5-7 redundant features** that are highly correlated and add estimation noise without information gain (e.g., `slope_10`/`slope_20` overlap with SMA ratios; `stoch_k` overlaps with `dist_low_14`/RSI).
2. **Add 8-12 well-motivated features** including MACD, Bollinger %B, overnight/intraday return decomposition, 12-1 month momentum, and calendar dummies.
3. **Differentiate h=1 vs h=5 feature sets:** h=1 should lean on short-window microstructure features; h=5 should lean on momentum and regime features.
4. **Fix a data leakage risk** in `cross_micro_features` (breadth uses same-day returns that include the target stock).

The guiding principle is: every feature must have either an economic rationale or robust empirical evidence. We prefer fewer, cleaner features over a kitchen-sink approach, especially for tree models which are vulnerable to overfitting on noise features (Lopez de Prado, 2018, Ch. 6).

---

## Feature-by-Feature Analysis

### A. Current Micro Features

| Feature | Decision | Rationale |
|---------|----------|-----------|
| `log_ret_1` | **KEEP (h=1, h=5)** | Core short-term reversal signal. Well-documented 1-day autocorrelation in European equities (Lo & MacKinlay, 1988). Essential input for both horizons. |
| `log_ret_3` | **DROP** | Redundant with `log_ret_5` and `log_ret_1`. The 3-day window has no specific economic justification and sits between two more standard lookbacks. Correlation with `log_ret_5` typically >0.85. |
| `log_ret_5` | **KEEP (h=1, h=5)** | Weekly momentum signal. Matches the h=5 target horizon, making it a natural mean-reversion/momentum indicator. |
| `log_ret_10` | **KEEP (h=5 only)** | Two-week return. More informative for the weekly horizon. For h=1, it is too slow-moving to be actionable. |
| `log_ret_20` | **KEEP (h=5 only)** | Monthly return. Core input for relative-to-market features. Keep for h=5; drop from h=1 feature set as it changes too slowly relative to 1-day target. |
| `ret_mean_5` | **DROP** | This is `log_ret_5 / 5` up to rounding. Perfectly collinear with `log_ret_5`. Adds no information. |
| `vol_5` | **KEEP (h=1)** | Short-term volatility captures regime shifts relevant for next-day prediction. Well-motivated by heteroskedasticity clustering (Bollerslev, 1986). |
| `vol_ratio_5_20` | **KEEP (h=1, h=5)** | Volatility regime indicator. Ratio >1 means short-term vol exceeds long-term (stress regime). This is more informative than raw vol levels because it is scale-free. |
| `atr_pct` | **KEEP (h=1, h=5)** | True range normalized by price. Captures intraday volatility distinct from close-to-close vol. Empirically useful for tree models (Krauss et al., 2017). |
| `sma_ratio_5_20` | **KEEP (h=1, h=5)** | Short-vs-medium trend ratio. Classic golden/death cross signal. Scale-free and well-behaved. |
| `sma_ratio_10_50` | **KEEP (h=5 only)** | Medium-vs-long trend. Too slow-moving for h=1 (the 50-day SMA barely moves day-to-day). Relevant for h=5 weekly regime. |
| `ema_ratio_5_20` | **DROP** | Highly correlated with `sma_ratio_5_20` (correlation typically >0.95). EMA and SMA ratios at the same windows carry near-identical information. Keep one, drop the other. SMA ratios are more interpretable. |
| `slope_10` | **KEEP (h=1)** | Linear regression slope captures trend direction over 2 weeks. Somewhat redundant with SMA ratios, but captures monotonicity that ratios miss. Keep for h=1 only. |
| `slope_20` | **DROP** | Highly correlated with `slope_10` and `sma_ratio_10_50`. The 20-day slope changes very slowly. Its information is largely captured by `sma_ratio_10_50`. |
| `dist_high_10` | **KEEP (h=1, h=5)** | Distance to 10-day high. Captures "how far from recent peak" -- a reversal/continuation signal. Empirically significant in Moskowitz et al. (2012) time-series momentum work. |
| `dist_low_10` | **KEEP (h=1, h=5)** | Distance to 10-day low. Complements `dist_high_10`. |
| `dist_high_20` | **KEEP (h=5 only)** | Monthly range context. Too slow for h=1, useful for h=5. |
| `dist_low_20` | **KEEP (h=5 only)** | Same rationale as `dist_high_20`. |
| `rsi_14` | **KEEP (h=1, h=5)** | Standard overbought/oversold oscillator. Well-studied empirically. Non-linearly related to returns -- tree models can exploit the 30/70 thresholds naturally. |
| `stoch_k` | **DROP** | Highly correlated with `rsi_14` and `dist_low_10` (all measure "where is price relative to recent range"). Empirical tests on European equities show negligible marginal information beyond RSI (Bajgrowicz & Scaillet, 2012). |
| `volu_ratio_5` | **KEEP (h=1, h=5)** | Short-term volume surge indicator. High volume confirms price moves (Karpoff, 1987). Scale-free. |
| `volu_ratio_20` | **KEEP (h=5 only)** | Monthly volume context. Too noisy for h=1. |
| `volu_ret_1` | **KEEP (h=1)** | Volume-weighted return. Positive `volu_ret_1` means high-volume up day (informed buying). Motivated by the volume-return relationship in Llorente et al. (2002). |
| `body` | **KEEP (h=1)** | Candlestick body size. Relevant for next-day reversal patterns. Less useful for h=5 (single-day candle patterns wash out over a week). |
| `upper_wick` | **KEEP (h=1)** | Upper rejection wick. h=1 microstructure signal. |
| `lower_wick` | **KEEP (h=1)** | Lower rejection wick. Same rationale. |
| `true_range_pct` | **DROP** | Nearly identical to `atr_pct` on any given day (ATR is the 14-day mean of true range; the daily value is very close). Keeping both adds collinearity. Keep `atr_pct` which is smoother. |
| `gap` | **KEEP (h=1)** | Overnight gap. Strong h=1 reversal signal (overnight returns tend to reverse intraday; Lou et al., 2019). Less relevant for h=5 as a single gap washes out. |

### B. Current Cross Features

| Feature | Decision | Rationale |
|---------|----------|-----------|
| `ibx_breadth` | **KEEP (h=1, h=5)** | Market breadth is a well-documented regime indicator. Broad participation signals sustainable moves vs narrow rallies. **Fix needed:** compute breadth excluding the target stock to avoid leakage (see Data Leakage section). |
| `ibx_breadth_10d` | **KEEP (h=5)** | Smoothed breadth trend. More relevant for weekly horizon. |

### C. Current Macro Features

| Feature | Decision | Rationale |
|---------|----------|-----------|
| `ibx_vol_10` | **KEEP (h=1, h=5)** | Market-level short-term volatility. Regime signal. |
| `ibx_vol_ratio_10_60` | **KEEP (h=1, h=5)** | Volatility regime ratio. Scale-free. Core feature. |
| `sp_vol_20` | **KEEP (h=5)** | US market volatility propagates to European markets with lag. More relevant at weekly horizon. |
| `sp_vol_ratio_20_100` | **KEEP (h=5)** | US vol regime. Same rationale. |
| `vix_chg_z_5` | **KEEP (h=1, h=5)** | Z-scored VIX change captures fear spikes relative to recent history. Well-motivated by the leverage effect and asymmetric volatility response (Black, 1976). |
| `vix_pctile_250` | **KEEP (h=5)** | Long-term VIX regime. Where current VIX sits in its 1-year distribution. Regime-level feature, more relevant for h=5. |
| `rel_ret_5` | **KEEP (h=1, h=5)** | Stock return relative to IBEX over 5 days. Alpha signal -- measures idiosyncratic momentum/reversal. |
| `rel_ret_20` | **KEEP (h=5)** | Monthly relative return. More appropriate for h=5 horizon. |
| `rel_vol_20` | **KEEP (h=5)** | Relative volatility to market. Beta/risk measure. Slow-moving, more useful for h=5. |

---

## Features to ADD

### 1. MACD Histogram (Normalized)

```
macd_line = ema_12 - ema_26
signal_line = ema_9(macd_line)
macd_hist = (macd_line - signal_line) / close
```

**Rationale:** MACD captures momentum regime changes at a different frequency than our existing SMA ratios. The histogram (difference between MACD and signal) is a second-derivative momentum indicator -- it measures whether momentum is accelerating or decelerating. Normalize by close to make it scale-free. Empirically, MACD adds marginal information beyond simple moving average crossovers in ML models (Krauss et al., 2017). **Keep for both h=1 and h=5.**

### 2. Bollinger Band %B

```
bb_mid = sma_20
bb_std = rolling_std(close, 20)
bb_pctb = (close - (bb_mid - 2*bb_std)) / (4*bb_std)
```

**Rationale:** %B measures where price is relative to its recent volatility-adjusted range. Values >1 mean price is above upper band (extreme), <0 means below lower band. This combines trend and volatility information in a single feature, distinct from our separate SMA ratios and vol features. Motivated by mean-reversion literature: extreme %B values predict reversals (Bollinger, 2002). **Keep for both h=1 and h=5.**

### 3. Overnight vs Intraday Return Decomposition

```
ret_overnight = log(open[t] / close[t-1])
ret_intraday  = log(close[t] / open[t])
```

**Rationale:** Overnight and intraday returns have markedly different statistical properties. Overnight returns capture informed trading and news absorption; intraday returns capture noise trading and liquidity. Lou et al. (2019) show that overnight returns are persistent while intraday returns mean-revert. Currently we have `gap` (which is `ret_overnight` in different form) but not the explicit intraday return. Adding `ret_intraday` gives the model both components. **Both for h=1; only `ret_overnight` for h=5.**

### 4. 12-1 Month Momentum (Jegadeesh-Titman Momentum)

```
mom_12_1 = log(close[t-21] / close[t-252])
```

**Rationale:** The single most documented cross-sectional return predictor in finance. Jegadeesh & Titman (1993) showed that stocks with high 12-month returns (excluding the most recent month) continue to outperform. The 1-month gap avoids short-term reversal contamination. This is a medium-frequency signal absent from the current feature set, which tops out at 20-day lookbacks. **Keep for h=5 only** (too slow for h=1).

### 5. Day-of-Week Dummy

```
dow = date.dayofweek  (0=Mon, 4=Fri)
```

**Rationale:** The day-of-week effect is one of the oldest calendar anomalies. Monday returns tend to be negative, Friday returns positive (French, 1980; Gibbons & Hess, 1981). While the effect has weakened in recent decades, tree models can exploit any residual non-linearity for free. Encode as a single integer 0-4 (trees handle ordinal encoding naturally; no need for one-hot). **h=1 only.** For h=5, the day-of-week of entry averages out.

### 6. Month-of-Year

```
month = date.month  (1-12)
```

**Rationale:** The January effect (Keim, 1983) and sell-in-May seasonality (Bouman & Jacobsen, 2002) are well-documented, though weaker in recent data. Again, cheap to include for tree models. Encode as integer 1-12. **Both h=1 and h=5,** but expect modest importance.

### 7. OBV Slope (On-Balance Volume Trend)

```
obv = cumsum(volume * sign(log_ret_1))
obv_slope_10 = rolling_slope(obv, 10) / volume_mean_20
```

**Rationale:** OBV tracks whether volume is flowing into or out of a stock. The raw OBV level is non-stationary, but its slope over 10 days captures volume-confirmed trends. Normalizing by average volume makes it scale-free. Motivated by Granville (1963) and empirically validated in Llorente et al. (2002). **Both h=1 and h=5.**

### 8. Amihud Illiquidity (5-day rolling)

```
illiq_5 = rolling_mean(|log_ret_1| / dollar_volume, 5)
```

**Rationale:** Amihud (2002) illiquidity ratio measures price impact per unit of trading. Illiquid stocks have higher expected returns (illiquidity premium) but also higher reversal tendency. Since we do not have dollar volume directly, use `close * volume` as proxy. Take 5-day rolling mean for stability. **Both h=1 and h=5.**

Note: if `close * volume` is too noisy because of corporate actions or stock splits, consider using the ratio form `|log_ret_1| / volu_ratio_5` which is already scale-adjusted.

### 9. Autocorrelation of Returns (5-day rolling)

```
ret_autocorr_5 = rolling_corr(log_ret_1, log_ret_1.shift(1), 5)
```

**Rationale:** Lo & MacKinlay (1988) showed that stock returns exhibit positive short-term autocorrelation, which weakens in liquid stocks. A rolling autocorrelation feature lets the model detect when a stock is in a trending vs mean-reverting regime. **h=1 only** (autocorrelation is a 1-step-ahead concept).

### 10. IBEX Intraday vs Overnight Return (Macro)

```
ibx_ret_overnight = log(ibx_open[t] / ibx_close[t-1])
ibx_ret_intraday  = log(ibx_close[t] / ibx_open[t])
```

**Rationale:** Macro-level overnight return captures global news absorbed between European sessions (Asian/US overnight developments). This is distinct from the stock-level overnight return. **h=1 only.**

---

## Features NOT Recommended (Considered but Rejected)

| Feature | Reason for Rejection |
|---------|---------------------|
| **VWAP** | Requires intraday data (tick/bar level). Cannot compute from daily OHLCV. |
| **Sector relative performance** | We lack sector classification data for IBEX35 stocks. Could be added if sector labels are obtained, but with only 30 stocks, sector groups would be very small (3-5 stocks), making sector averages noisy. |
| **Earnings season dummy** | Requires earnings calendar data we do not have. IBEX35 earnings dates are not standardized and would need an external data source. |
| **Fourier features** | Periodicity decomposition (e.g., FFT components) is theoretically appealing but empirically unstable in financial time series. Lopez de Prado (2018) warns against frequency-domain features that overfit to specific regimes. |
| **Hurst exponent** | Computationally expensive, requires long windows (~100+ days), and empirically adds little beyond the autocorrelation and vol_ratio features we already have. |
| **Sentiment / NLP features** | Out of scope given current data (OHLCV only). Would require news or social media data. |

---

## Data Leakage Risks

### 1. Cross-Sectional Leakage in Breadth (CRITICAL)

**Current issue:** `ibx_breadth` computes the fraction of ALL IBEX stocks with positive returns on day t. This includes the target stock itself. When predicting whether stock X goes up, you are partially telling the model whether stock X went up (through the breadth feature).

**Fix:** Compute leave-one-out breadth: for each stock, breadth = fraction of OTHER 29 stocks with positive returns. This eliminates self-inclusion leakage.

```python
# Instead of global breadth, compute per-stock leave-one-out breadth
for ticker in tickers:
    mask_others = df["ticker"] != ticker
    breadth_others = df[mask_others].groupby("date")["log_ret_1"].apply(lambda x: (x > 0).mean())
    # merge back to this ticker's rows
```

### 2. Target Leakage via Forward-Looking Features (LOW RISK, BUT VERIFY)

The `target_feature` function correctly uses `shift(-horizon)` which looks forward. No current features use `shift(-k)` for any k, so this is clean. However, verify that:
- `cross_micro_features` only uses day-t data (it does -- uses `log_ret_1` which is backward-looking).
- Macro features only use day-t or earlier data (they do -- all use backward rolling windows).

### 3. Macro Alignment / Holiday Calendar (MODERATE RISK)

**Current issue:** `align_macro` forward-fills macro data across holidays. When IBEX is open but US markets are closed, the S&P and VIX features will use the previous US trading day's values. This is correct and not leakage. However, when US markets are open but IBEX is closed, the forward-fill means the next IBEX trading day will see "stale" US data from the IBEX holiday. This is also correct (you would know that data in real-time).

**Risk:** If the S&P 500 features are computed using close-to-close returns and IBEX opens before S&P closes (time zone difference), day-t S&P features might not be available at IBEX close. **Recommendation:** Use S&P features lagged by 1 day (`sp_log_ret_1.shift(1)`) to be safe, or clearly document that features are computed at end-of-day after all markets close.

### 4. Purged Cross-Validation (IMPORTANT FOR TRAINING)

For h=5 predictions, the target `close[t+5]/close[t]` overlaps with `close[t+1+5]/close[t+1]`. This creates autocorrelation in targets across consecutive rows. Standard k-fold cross-validation will leak information between train and test sets.

**Fix:** Use purged k-fold cross-validation with an embargo period >= h (Lopez de Prado, 2018, Ch. 7). For h=5, purge 5 trading days between train and test folds. For h=1, purge 1 day minimum (but 5 is safer due to feature window overlap).

### 5. Survivorship Bias

IBEX35 composition changes over time. If the 30 current stocks are used for the entire 2006-2026 backtest, stocks that entered later or were removed create gaps and survivorship bias. This is a data issue, not a feature issue, but affects all features equally.

---

## Final Feature Sets

### Horizon 1 (h=1) Feature Set -- 30 features

**Micro (20):**
- `log_ret_1`, `log_ret_5`
- `vol_5`, `vol_ratio_5_20`
- `atr_pct`
- `sma_ratio_5_20`, `slope_10`
- `dist_high_10`, `dist_low_10`
- `rsi_14`
- `volu_ratio_5`, `volu_ret_1`
- `body`, `upper_wick`, `lower_wick`, `gap`
- `ret_intraday` (NEW)
- `macd_hist` (NEW)
- `bb_pctb` (NEW)
- `ret_autocorr_5` (NEW)

**Cross (2):**
- `ibx_breadth` (fixed: leave-one-out)
- `obv_slope_10` (NEW)

**Macro (6):**
- `ibx_vol_ratio_10_60`
- `vix_chg_z_5`
- `rel_ret_5`
- `ibx_ret_overnight` (NEW)
- `illiq_5` (NEW)
- `dow` (NEW)

### Horizon 5 (h=5) Feature Set -- 32 features

**Micro (18):**
- `log_ret_1`, `log_ret_5`, `log_ret_10`, `log_ret_20`
- `vol_5`, `vol_ratio_5_20`
- `atr_pct`
- `sma_ratio_5_20`, `sma_ratio_10_50`
- `dist_high_10`, `dist_low_10`, `dist_high_20`, `dist_low_20`
- `rsi_14`
- `volu_ratio_5`, `volu_ratio_20`
- `macd_hist` (NEW)
- `bb_pctb` (NEW)

**Cross (3):**
- `ibx_breadth` (fixed: leave-one-out)
- `ibx_breadth_10d`
- `obv_slope_10` (NEW)

**Macro (11):**
- `ibx_vol_10`, `ibx_vol_ratio_10_60`
- `sp_vol_20`, `sp_vol_ratio_20_100`
- `vix_chg_z_5`, `vix_pctile_250`
- `rel_ret_5`, `rel_ret_20`, `rel_vol_20`
- `mom_12_1` (NEW)
- `illiq_5` (NEW)
- `month` (NEW)

---

## Key Differences Between h=1 and h=5

| Aspect | h=1 | h=5 |
|--------|-----|-----|
| **Window focus** | Short (1-10 days) | Medium (5-50 days) |
| **Microstructure** | Candle patterns, gap, intraday return, autocorrelation | Dropped (single-day noise washes out) |
| **Momentum** | Short-term reversal dominant | 12-1 month momentum added |
| **Calendar** | Day-of-week matters | Month-of-year matters |
| **Macro detail** | Only fast-moving macro (VIX z-score, IBEX vol ratio) | Full macro suite including US vol, VIX percentile |
| **Volume** | `volu_ret_1` (volume-confirmed daily move) | `volu_ratio_20` (monthly volume trend) |
| **Relative-to-market** | Only `rel_ret_5` | Full suite: `rel_ret_5`, `rel_ret_20`, `rel_vol_20` |
| **Feature count** | ~30 | ~32 |
| **Purge/embargo** | 1-day minimum | 5-day minimum |

The core principle: h=1 features should capture **microstructure and short-term mean-reversion**; h=5 features should capture **momentum, regime, and cross-sectional factors**.

---

## Dropped Features Summary

| Feature | Reason |
|---------|--------|
| `log_ret_3` | Redundant with `log_ret_1` and `log_ret_5`; no economic justification for 3-day window specifically |
| `ret_mean_5` | Algebraically equivalent to `log_ret_5 / 5`; perfectly collinear |
| `ema_ratio_5_20` | Correlation >0.95 with `sma_ratio_5_20`; keep the simpler one |
| `slope_20` | Redundant with `sma_ratio_10_50`; both capture 1-month trend |
| `stoch_k` | Redundant with `rsi_14` and `dist_low_10`; all measure same concept |
| `true_range_pct` | Near-duplicate of `atr_pct` on daily basis |
| `volu_ret_1` (h=5) | Single-day volume-return interaction too noisy for weekly prediction |
| `body`, `upper_wick`, `lower_wick`, `gap` (h=5) | Single-day candle patterns irrelevant at weekly horizon |

---

## Implementation Notes

1. **Feature computation order matters.** New features (MACD, Bollinger, OBV) should be computed inside `micro_features()`. Calendar features should be computed in `build_features()` after the date column is available. Illiquidity should be computed in `micro_features()`.

2. **Normalization for GRU/LSTM.** Tree models do not need feature scaling, but RNN models do. Apply per-feature z-scoring using only training-set statistics (rolling or expanding window). Never fit the scaler on the full dataset.

3. **Feature importance monitoring.** After training, check permutation importance. Any new feature with importance < random noise should be dropped. Lopez de Prado (2018, Ch. 8) recommends mean decrease impurity (MDI) for forests and SHAP values for gradient boosting.

4. **`necessary_features()` must be updated** to reflect the new h=1 vs h=5 feature sets. Consider accepting the horizon parameter to return the appropriate feature list.

---

## References

- **Amihud, Y. (2002).** "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*, 5(1), 31-56. [Illiquidity ratio definition and premium]

- **Bajgrowicz, P. & Scaillet, O. (2012).** "Technical trading revisited: False discoveries, persistence tests, and transaction costs." *Journal of Financial Economics*, 106(3), 473-491. [Shows most technical indicators are redundant after multiple testing correction]

- **Black, F. (1976).** "Studies of stock price volatility changes." *Proceedings of the American Statistical Association*, 177-181. [Leverage effect: negative returns increase volatility]

- **Bollerslev, T. (1986).** "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327. [Volatility clustering justification for vol features]

- **Bollinger, J. (2002).** *Bollinger on Bollinger Bands*. McGraw-Hill. [%B and bandwidth features]

- **Bouman, S. & Jacobsen, B. (2002).** "The Halloween indicator, 'sell in May and go away': another puzzle." *American Economic Review*, 92(5), 1618-1635. [Monthly seasonality]

- **French, K. (1980).** "Stock returns and the weekend effect." *Journal of Financial Economics*, 8(1), 55-69. [Day-of-week effect]

- **Gibbons, M. & Hess, P. (1981).** "Day of the week effects and asset returns." *Journal of Business*, 54(4), 579-596. [Day-of-week effect confirmation]

- **Granville, J. (1963).** *Granville's New Key to Stock Market Profits*. Prentice-Hall. [On-Balance Volume]

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to buying winners and selling losers: implications for stock market efficiency." *Journal of Finance*, 48(1), 65-91. [12-1 month momentum: the foundational paper. 3-12 month past winners outperform past losers by ~1% per month. Replicated across markets and decades.]

- **Karpoff, J. (1987).** "The relation between price changes and trading volume: a survey." *Journal of Financial and Quantitative Analysis*, 22(1), 109-126. [Volume-return relationship]

- **Keim, D. (1983).** "Size-related anomalies and stock return seasonality." *Journal of Financial Economics*, 12(1), 13-32. [January effect]

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. [Empirical comparison of ML models for stock prediction using technical features. Shows tree models and DNNs achieve ~55% daily accuracy; MACD and ATR among most important features.]

- **Llorente, G., Michaely, R., Saar, G. & Wang, J. (2002).** "Dynamic volume-return relation of individual stocks." *Review of Financial Studies*, 15(4), 1005-1047. [Volume-return interaction: high-volume moves driven by information are persistent; high-volume moves driven by liquidity reverse.]

- **Lo, A. & MacKinlay, C. (1988).** "Stock market prices do not follow random walks: evidence from a simple specification test." *Review of Financial Studies*, 1(1), 41-66. [Short-term return autocorrelation. Weekly returns exhibit positive autocorrelation, stronger for small stocks. Justifies using lagged returns as features.]

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. [Ch. 5: Fractional differentiation for stationarity. Ch. 6: Feature importance and redundancy in financial ML. Ch. 7: Purged k-fold cross-validation to avoid leakage from overlapping targets. Ch. 8: MDI and MDA feature importance.]

- **Lou, D., Polk, C. & Skouras, S. (2019).** "A tug of war: overnight versus intraday expected returns." *Journal of Financial Economics*, 134(1), 192-213. [Overnight returns are persistent, intraday returns mean-revert. Decomposition adds information beyond total return.]

- **Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012).** "Time series momentum." *Journal of Financial Economics*, 104(2), 228-250. [Time-series momentum across asset classes. 12-month lookback with 1-month holding period. Justifies `dist_high` features as momentum indicators.]
