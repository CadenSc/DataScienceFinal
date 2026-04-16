# Optional Notes For Stage 3

This file is for the teammate handling most of the modeling work. It follows the proposal's plan to use Logistic Regression as an interpretable baseline and Random Forest as a nonlinear comparison model.

## Dataset To Use

Use `engineered_blocks.csv` for modeling. Each row is one Ethereum block, sorted chronologically by `block_number`. The first columns are cleaned raw block metadata, and the later columns are engineered recent-activity and congestion features.

Useful target helpers are already included:

- `target_next_congestion_ratio`
- `target_next_congestion_flag_high`
- `target_next_gas_used`
- `target_next_base_fee_gwei`

For a classification task matching the proposal, `target_next_congestion_flag_high` is the cleanest target. It marks whether the next block's congestion ratio is at least 0.80.

## Modeling Cautions

- Use a chronological split rather than a fully random split because block rows are time ordered.
- Do not include `target_` columns as predictors.
- Avoid identifiers such as `block_number`, `slot`, `source_page`, `source_row`, and `source_url` as model features.
- Logistic Regression will likely need scaling for continuous features.
- Random Forest can handle nonlinear relationships, but it should still be evaluated conservatively.
- Accuracy is mentioned in the proposal, but a confusion matrix is also useful because high-congestion blocks are not the majority class.

## Feature Groups Worth Trying

Recent activity:

- `rolling_avg_txn_5`, `rolling_avg_txn_20`
- `rolling_avg_gas_used_5`, `rolling_avg_gas_used_20`
- `rolling_avg_base_fee_5`, `rolling_avg_base_fee_20`

Congestion and utilization:

- `congestion_ratio`
- `rolling_avg_congestion_5`, `rolling_avg_congestion_20`
- `congestion_proxy_1h`, `congestion_proxy_24h`
- `short_term_congestion_indicator`
- `congestion_flag_high`, `congestion_flag_low`

Momentum and spike behavior:

- `txn_change_1`, `gas_used_change_1`, `base_fee_change_1`
- `txn_momentum_5_20`, `gas_used_momentum_5_20`, `base_fee_momentum_5_20`
- `utilization_trend_20`
- `base_fee_spike_flag`, `gas_used_spike_flag`, `txn_spike_flag`

Fee-related context:

- `base_fee_gwei`
- `reward_eth`
- `burnt_fees_eth`
- `recent_reward_avg_20`
- `recent_burnt_fees_avg_20`

Time context:

- `hour_utc`
- `day_of_week_utc`
- `is_weekend_utc`
- `block_interval_seconds`

## Window Assumption

The 1-hour and 24-hour congestion proxies are block-window approximations. The code assumes about 12 seconds per Ethereum block, so 1 hour is 300 blocks and 24 hours is 7,200 blocks. Because this dataset has 4,997 cleaned rows, the 24-hour proxy uses available trailing history rather than a complete 24-hour window. The available row count is stored in `congestion_proxy_24h_window_blocks`.

## Feature Screening

`feature_selection_suggestions.csv` gives a simple correlation-based ranking against `target_next_congestion_ratio`. Treat it as a starting point, not proof that a feature will improve model performance.
