# Unit Test Suite Design

**Date:** 2026-04-24  
**Status:** approved

---

## Overview

Full unit test coverage for the IBEX35 ML pipeline. Tests run with `pytest`. No test requires a real DB, trained artifact, internet connection, or external API — all external deps are mocked. Tests do not execute (no model training, no data fetching) but all code paths are exercised with synthetic data.

**Key constraint:** Tests must be syntactically correct and logically sound but will not be run on this machine (missing data/models). Code correctness is the goal, not green CI.

---

## Test Runner & Dependencies

```
pytest
pytest-mock
```

All existing project deps (numpy, pandas, torch, sklearn, xgboost) are already installed.

---

## File Structure

```
tests/
  conftest.py                   # shared fixtures (synthetic OHLCV, dates, etc.)
  test_features.py              # models/trees/features.py — pure logic
  test_evaluate.py              # models/evaluate.py — pure logic
  test_windows.py               # models/base.py window generators — pure logic
  test_markov.py                # models/markov/markov.py MarkovChain — pure logic
  test_rnn_utils.py             # models/neural/lstm.py utilities — pure logic
  test_rate_limiter.py          # llm/rate_limit.py — pure logic
  test_news_classification.py   # news/classification.py — pure + mocked LLM
  test_news_rss.py              # news/news_rss.py — pure filter/sort functions
  test_newsletter.py            # newsletter/build.py — pure HTML builders
  test_predict_helpers.py       # models/predict.py helpers — synthetic artifacts
  test_db_sqlite.py             # db/sqlite/* — mocked sqlite3
  test_db_supabase.py           # db/supabase/* — mocked supabase Client
  test_llm_service.py           # llm/gpt_service.py — mocked Groq API
```

---

## Group 1 — Pure Logic

### `tests/test_features.py`

Covers `models/trees/features.py`. All tests use synthetic DataFrames.

**Fixtures (in conftest.py):**
- `ohlcv_df`: 60-row single-ticker OHLCV DataFrame (`ticker`, `date`, `open`, `high`, `low`, `close`, `volume`)
- `multi_ticker_ohlcv`: 60-row × 3 tickers OHLCV DataFrame
- `macro_df`: 60-row macro DataFrame (`ticker` = `^IBEX`, `^GSPC`, `^VIX`)

**Tests:**
- `test_rolling_slope_shape` — output length matches input
- `test_rolling_slope_constant_series` — slope of constant series is 0
- `test_rsi_range` — all RSI values in [0, 100]
- `test_rsi_all_up_days` — RSI near 100 when all returns positive
- `test_micro_features_output_columns` — known column names present in output
- `test_micro_features_no_infinite` — no ±inf in output after dropna
- `test_horizon_features_h1_adds_dow` — `dow` column present for h=1
- `test_horizon_features_h5_adds_momentum` — momentum column present for h=5
- `test_target_feature_binary` — `direction` column is 0 or 1 only
- `test_target_feature_continuous` — `future_log_ret` is float
- `test_cross_micro_features_adds_breadth` — breadth column present
- `test_macro_features_output_columns` — VIX, SP500 columns present
- `test_align_macro_fills_missing_dates` — no NaN in state vars after align
- `test_ml_ready_returns_five_tuple` — returns exactly 5 values
- `test_ml_ready_no_nan_in_X` — X matrix has no NaN
- `test_ml_ready_y_binary` — y is 0/1 for discrete
- `test_assert_columns_raises_on_missing` — raises KeyError/ValueError on missing col

### `tests/test_evaluate.py`

Covers `models/evaluate.py`. Fully pure.

**Tests:**
- `test_evaluate_model_keys` — returned dict has all 7 expected keys
- `test_evaluate_model_perfect_predictions` — accuracy=1.0, mcc=1.0 for perfect preds
- `test_evaluate_model_random_predictions` — no crash, all values in valid ranges
- `test_evaluate_regression_keys` — returned dict has all 5 expected keys
- `test_evaluate_regression_perfect` — mae=0, r2=1 for perfect preds
- `test_evaluate_regression_directional_accuracy` — known preds → known dir_acc

### `tests/test_windows.py`

Covers `sliding_windows` and `expanding_windows` from `models/base.py`.

**Tests:**
- `test_sliding_windows_count` — correct number of windows for known input
- `test_sliding_windows_train_size` — every train window is exactly `window` long
- `test_sliding_windows_embargo` — gap between last train date and first test date ≥ embargo
- `test_sliding_windows_no_overlap` — train and test sets are disjoint
- `test_expanding_windows_train_grows` — each fold's train set is strictly larger
- `test_expanding_windows_min_train` — first fold has at least `min_train` dates
- `test_windows_empty_on_insufficient_data` — returns [] when dates too short

### `tests/test_markov.py`

Covers `MarkovChain` from `models/markov/markov.py`.

**Tests:**
- `test_fit_builds_transition_dict` — after fit, `transition_` is non-empty
- `test_predict_proba_range` — all probas in [0, 1]
- `test_predict_binary` — all preds are 0 or 1
- `test_digitise_bins_in_range` — all bins in [0, n_states-1]
- `test_lag_features_order1` — returns single lag column
- `test_lag_features_order2` — returns two lag columns
- `test_unseen_state_returns_prior` — unseen state returns 0.5 (Laplace prior)

### `tests/test_rnn_utils.py`

Covers `add_cyclic_dow`, `build_sequences`, `SequenceDataset` from `models/neural/lstm.py`.

**Tests:**
- `test_add_cyclic_dow_removes_dow` — `dow` column gone, `dow_sin`/`dow_cos` added
- `test_add_cyclic_dow_sin_cos_range` — sin/cos values in [-1, 1]
- `test_build_sequences_shape` — output shape is `(n - seq_len + 1, seq_len, n_features)`
- `test_build_sequences_last_dates` — last_dates array length matches seqs
- `test_build_sequences_multi_ticker` — each ticker contributes independently
- `test_sequence_dataset_len` — `__len__` matches input
- `test_sequence_dataset_getitem` — returns correct tensors

### `tests/test_rate_limiter.py`

Covers `RateLimitState` from `llm/rate_limit.py`.

**Tests:**
- `test_record_and_prune` — old events pruned after 60s
- `test_no_wait_under_limits` — `wait_for_slot` returns immediately when under limits
- `test_estimate_tokens` — returns `len(text) // 4`, minimum 1

---

## Group 2 — News Layer

### `tests/test_news_classification.py`

Covers `news/classification.py`.

**Tests:**
- `test_extract_keywords_hit_match` — known keywords found in text
- `test_extract_keywords_hit_no_match` — empty list when no match
- `test_extract_keywords_case_insensitive` — uppercase text matches lowercase keyword
- `test_compute_relevance_company_specific` — category=company_specific scores higher
- `test_compute_relevance_generic_noise` — category=generic_noise scores low
- `test_compute_relevance_with_companies` — +0.2 for non-empty companies
- `test_compute_relevance_clamped` — result always in [0.0, 1.0]
- `test_split_into_batches_sizes` — batches have correct size
- `test_split_into_batches_last_partial` — last batch may be smaller
- `test_build_news_batch_prompt_contains_titles` — titles appear in output
- `test_news_classifier_prompt_returns_string` — returns non-empty string
- `test_classify_news_calls_llm_once_per_batch` — mock LLM called ceil(n/10) times
- `test_classify_news_output_keys` — each result has `category`, `companies`, `sentiment`, `relevance`, `keywords_hit`

### `tests/test_news_rss.py`

Covers pure filter/sort functions in `news/news_rss.py`.

**Tests:**
- `test_last_news_filters_yesterday` — items from yesterday kept, older dropped
- `test_last_news_empty_on_no_match` — returns [] if none from yesterday
- `test_top_news_returns_k_items` — returns exactly k items
- `test_top_news_sorted_by_relevance` — items in descending relevance order

### `tests/test_newsletter.py`

Covers `newsletter/build.py` HTML builders and `format_predictions`.

**Tests:**
- `test_add_header_contains_date` — date string appears in HTML output
- `test_add_footer_returns_html` — output contains `</` (valid HTML)
- `test_add_closing_returns_html` — output contains closing tags
- `test_add_news_contains_titles` — item titles appear in output
- `test_add_predictions_contains_tickers` — ticker names appear in output
- `test_format_predictions_adds_columns` — `name`, `action`, `proba.2f` columns added
- `test_format_predictions_action_values` — `action` is "Buy" or "Sell" only

---

## Group 3 — Predict Helpers

### `tests/test_predict_helpers.py`

Covers `models/predict.py` inference helpers. All artifacts built synthetically in-test — no real pkl files needed.

**Fixtures:**
- `tree_artifact`: dict with `model` (fitted RandomForestClassifier on 20 samples), `features` list, `target_type="discrete"`, `ft_type="micro"`
- `rnn_artifact`: dict with `model_state`, `model_config` (input_size=5, hidden=16, layers=1, dropout=0.0, cell="gru"), `scaler` (fitted StandardScaler), `features` list, `seq_len=5`, `target_type="discrete"`, `ft_type="micro"`, `model_key="gru"`
- `cnn_rnn_artifact`: same as rnn_artifact but `model_key="cnn_gru"` with `num_filters=8, kernel_size=3`

**Tests:**
- `test_load_artifact_raises_on_missing` — FileNotFoundError for non-existent path
- `test_predict_tree_returns_preds_and_probas` — shape matches input rows
- `test_predict_tree_preds_binary` — all preds are 0 or 1
- `test_predict_tree_continuous_skips_proba` — continuous artifact returns raw float preds
- `test_predict_tree_markov_1d_proba` — 1-D proba array handled correctly (ndim check)
- `test_reconstruct_rnn_gru` — returns StockRNN with gru cell
- `test_reconstruct_rnn_cnn_gru` — returns StockCNNRNN
- `test_reconstruct_rnn_in_eval_mode` — model is in eval mode after reconstruction
- `test_predict_rnn_output_shapes` — preds/probas/last_dates/last_tickers all same length
- `test_predict_rnn_preds_binary` — discrete preds are 0 or 1
- `test_predict_rnn_one_pred_per_ticker_sequence_end` — last_tickers matches ticker input
- `test_temporal_inner_split_no_leakage` — no test date appears in train dates

---

## Group 4 — DB Layer

### `tests/test_db_sqlite.py`

Mocks `sqlite3.connect` via `pytest-mock`. Tests all functions in `db/sqlite/`.

**Tests:**
- `test_get_last_date_returns_value` — mock cursor returns date string
- `test_get_last_date_returns_none` — mock cursor returns None
- `test_load_news_no_filter` — query has no WHERE clause
- `test_load_news_start_filter` — query includes `WHERE date >=`
- `test_load_news_end_filter` — query includes `WHERE date <=`
- `test_load_news_both_filters` — query includes both conditions
- `test_load_entities_no_filter` — query runs without crash
- `test_ingest_news_upserts_rows` — conn.execute called for each item
- `test_ingest_ohlcv_calls_execute` — executes insert for each row
- `test_sqlite_connection_commits_on_success` — conn.commit called
- `test_sqlite_connection_closes_on_exit` — conn.close called even on exception

### `tests/test_db_supabase.py`

Mocks `supabase.Client`. Tests all functions in `db/supabase/`.

**Tests:**
- `test_fetch_ohlcv_calls_rpc` — supabase.rpc called with correct args
- `test_fetch_ohlcv_returns_dataframe` — returns pd.DataFrame
- `test_ingest_ohlcv_calls_upsert` — supabase.table().upsert().execute() called
- `test_upload_preds_calls_upsert` — upsert called with on_conflict key
- `test_top_k_news_calls_select` — supabase select chain called
- `test_get_recipients_returns_list` — returns list of strings

---

## Group 5 — LLM Service

### `tests/test_llm_service.py`

Mocks Groq API (`openai.OpenAI`).

**Tests:**
- `test_estimate_tokens_basic` — `len(text) // 4` for typical string
- `test_estimate_tokens_minimum_one` — single char returns 1
- `test_query_returns_parsed_json` — mock API response → dict with `data` and `usage`
- `test_query_records_token_usage` — token count tracked after call
- `test_rate_limit_state_under_limits` — no sleep when under TPM/RPM
- `test_rate_limit_record_increments_usage` — token deque grows after record

---

## Bugs Fixed Before Tests

1. **`db/sqlite/queries_news.py`** — `params = None` → `params = []`, missing `WHERE` before first filter, `AND` before second filter. Fixed in commit `cf545bf`.

---

## What Is NOT Tested

- `BaseTrainer.run()` — full training pipeline (requires DB + real data)
- `MetaTrainer` — requires all 5 base models running
- `get_predictions()` — requires Supabase + artifact files
- `send_newsletter()` — requires SMTP + filesystem images
- All `ingest_*.py` DB write functions beyond smoke-test level
- `download_ticker()` — requires yfinance API
