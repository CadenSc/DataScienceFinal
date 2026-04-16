# Stage 1 and Stage 2 Summary

This workflow follows `proposal.txt` as the source of truth. The project analyzes recent Ethereum block-level activity from Etherscan to understand how transaction count, gas usage, base fee, and recent congestion patterns change over time. The Stage 1 and Stage 2 work focuses on acquisition, cleaning, exploratory analysis, and recent-activity feature engineering. It does not claim final model performance.

## Problem Framing

Ethereum blocks vary in how many transactions they include and how much of the available gas limit they use. Those changes reflect network activity and congestion. The academic goal is to use blockchain data as a data science dataset for studying network behavior. The personal motivation is to better understand when Ethereum network demand may be lower, which can help users think about more cost-efficient transaction timing.

Later Stage 3 modeling can use this dataset to test whether recent block behavior helps explain upcoming block gas usage or congestion. The proposal mentions Logistic Regression as an interpretable baseline and Random Forest as a nonlinear comparison model.

## Data Source And Collection Method

- Source: public Etherscan Ethereum block pages, starting from `https://etherscan.io/blocks?ps=100&p=1`.
- Collection script: `scrape_blocks.py`.
- Scope requested: 50 pages at 100 rows per page, or 5,000 raw block rows.
- Collection approach: the scraper requests paginated Etherscan block pages with a short delay, parses table rows using BeautifulSoup, includes retry logic for pages that parse fewer rows than expected, and stores raw source text for traceability.
- Actual scrape window in this run: `2026-04-16T01:50:31+00:00` through `2026-04-16T01:51:08+00:00`.
- Block range after cleaning: block `24884214` through block `24889210`.
- Time range after cleaning: `2026-04-15T09:07:59+00:00` through `2026-04-16T01:50:23+00:00`.

## Dataset Size

| Dataset | File | Rows | Notes |
|---|---|---:|---|
| Raw | `raw_blocks.csv` | 5,000 | Direct scraped rows from Etherscan pages with source text retained. |
| Cleaned | `cleaned_blocks.csv` | 4,997 | Three duplicate block rows were removed after pagination shifted during scraping. |
| Engineered | `engineered_blocks.csv` | 4,997 | Cleaned rows plus rolling activity, congestion, spike, and next-block target columns. |

## Legal And Ethical Considerations

The data comes from public Ethereum block explorer pages and is used for academic analysis. The workflow collects block-level metadata only, not private or sensitive user data beyond public blockchain information already shown by Etherscan. Scraping should remain respectful and limited in scope, so the script uses a small page count for the class project, a request delay, and no attempts to bypass access controls.

## Raw Dataset Schema

| Column | Meaning |
|---|---|
| `source_url` | Etherscan page URL used for the row. |
| `source_page` | Pagination page number. |
| `source_row` | Row order on the scraped page. |
| `scraped_at_utc` | UTC scrape timestamp for the page request. |
| `block_number_raw` | Block number as scraped text. |
| `slot_raw` | Beacon slot as scraped text when available. |
| `age_raw` | Relative age text from Etherscan. |
| `block_datetime_raw` | Displayed block datetime text. |
| `block_timestamp_unix_raw` | Unix timestamp exposed in the block table. |
| `blobs_raw` | Blob count and percent text. |
| `txn_count_raw` | Transaction count text. |
| `fee_recipient_raw` | Fee recipient/builder text. |
| `gas_used_raw` | Gas used text, often including percent utilization. |
| `gas_limit_raw` | Gas limit text. |
| `base_fee_raw` | Base fee text in Gwei. |
| `reward_raw` | Reward text in ETH. |
| `burnt_fees_raw` | Burnt fee text and percent. |
| `parse_method` | Parser path used for the row. |
| `raw_row_text` | Full flattened row text for audit and repair. |

## Cleaned Dataset Schema

| Column | Type | Meaning |
|---|---|---|
| `block_number` | integer | Ethereum block number. |
| `slot` | integer | Beacon slot. |
| `block_datetime_utc` | datetime string | Standardized UTC block timestamp. |
| `block_timestamp_unix` | integer | Unix block timestamp. |
| `age_raw` | string | Original relative age text. |
| `age_seconds_at_scrape` | integer | Approximate age in seconds at scrape time. |
| `blobs_count` | integer | Number of blobs shown for the block. |
| `blobs_percent` | float | Blob percent from Etherscan when available. |
| `txn_count` | integer | Transaction count. |
| `fee_recipient` | string | Standardized fee recipient text. |
| `gas_used` | integer | Gas used by the block. |
| `gas_used_percent` | float | Gas used as percent of gas limit from Etherscan text. |
| `gas_limit` | integer | Block gas limit. |
| `base_fee_gwei` | float | Base fee converted to numeric Gwei. |
| `reward_eth` | float | Reward converted to numeric ETH. |
| `burnt_fees_eth` | float | Burnt fees converted to numeric ETH. |
| `burnt_fees_percent` | float | Burnt fee percent from Etherscan text. |
| `source_page`, `source_row`, `source_url`, `scraped_at_utc` | mixed | Source metadata retained for traceability. |

## Engineered Dataset Schema

`engineered_blocks.csv` contains every cleaned column plus recent-activity and congestion features. Required engineered fields include:

- `congestion_ratio = gas_used / gas_limit`
- `rolling_avg_txn_5`, `rolling_avg_txn_20`
- `rolling_avg_gas_used_5`, `rolling_avg_gas_used_20`
- `rolling_avg_base_fee_5`, `rolling_avg_base_fee_20`
- `rolling_avg_congestion_5`, `rolling_avg_congestion_20`
- `txn_change_1`, `gas_used_change_1`, `base_fee_change_1`
- `txn_volatility_20`, `gas_used_volatility_20`, `base_fee_volatility_20`

Additional useful fields include:

- Congestion state fields: `short_term_congestion_indicator`, `congestion_flag_high`, `congestion_flag_low`, `congestion_level`
- Rolling block-window proxies: `congestion_proxy_1h`, `congestion_proxy_24h`, `congestion_proxy_1h_window_blocks`, `congestion_proxy_24h_window_blocks`
- Momentum and trend fields: `txn_momentum_5_20`, `gas_used_momentum_5_20`, `base_fee_momentum_5_20`, `congestion_momentum_5_20`, `utilization_trend_20`
- Ratio-to-recent-average fields: `gas_used_to_recent_avg`, `txn_to_recent_avg`, `base_fee_to_recent_avg`
- Spike flags: `base_fee_spike_flag`, `gas_used_spike_flag`, `txn_spike_flag`
- Fee features: `recent_reward_avg_20`, `recent_burnt_fees_avg_20`
- Time fields: `block_interval_seconds`, `hour_utc`, `day_of_week_utc`, `is_weekend_utc`
- Stage 3 target helpers: `target_next_block_number`, `target_next_gas_used`, `target_next_base_fee_gwei`, `target_next_congestion_ratio`, `target_next_congestion_flag_high`

The 1-hour and 24-hour congestion features are approximations based on Ethereum's roughly 12-second block time. The script uses 300 blocks for a 1-hour proxy and 7,200 blocks for a 24-hour proxy. Because this dataset has 4,997 cleaned blocks, the 24-hour proxy uses all available trailing history rather than a full 24-hour window; the available window count is stored in `congestion_proxy_24h_window_blocks`.

## Cleaning Decisions And Justifications

- Duplicate handling: duplicate `block_number` rows were removed, keeping the first observed row by scrape page and row order. This removed 3 rows caused by pagination shifting while newer blocks were being added.
- Missing values: no cleaned rows were removed for missing required fields. If future scrapes produce missing required metrics, `clean_blocks.py` removes rows missing `block_number`, `txn_count`, `gas_used`, `gas_limit`, or `base_fee_gwei`.
- Text and unit stripping: commas, `%`, parentheses, `ETH`, `Gwei`, and label text are stripped only during cleaning. Raw text remains in `raw_blocks.csv` for auditability.
- Numeric conversion: block number, slot, blob count, transaction count, gas used, and gas limit are converted to integers. Base fee, reward, burnt fees, and percentage fields are converted to floats.
- Timestamp handling: Unix timestamps are used when present. Displayed datetime is treated as UTC. Relative age is kept but not treated as the primary timestamp because it changes while scraping.
- Outlier strategy: outliers are counted in `data_quality_report.md` but retained. High gas usage, fee spikes, and transaction bursts are meaningful Ethereum congestion behavior rather than automatic errors.

## EDA Summary

EDA output files are stored in `eda_outputs/`.

- `txn_count_distribution.svg`: transaction counts varied from 0 to 1,069, with median 244.
- `gas_used_distribution.svg`: gas used ranged from 0 to 59,999,764, showing why utilization features are useful.
- `base_fee_trend.svg`: base fee ranged from 0.031 to 2.363 Gwei during this scrape window.
- `congestion_ratio_trend.svg`: median congestion ratio was about 0.479, and 598 blocks met the high-congestion threshold of 0.80.
- `rolling_activity_trends.svg`: rolling averages make short-term transaction, gas, and base-fee movement easier to compare.
- `correlation_heatmap.svg`: correlations are used descriptively for feature screening, not as a claim of model performance.
- `before_after_missingness.svg`: shows before/after data quality checks.
- `spike_inspection.svg`: highlights high-congestion blocks while keeping them in the dataset.

## Feature Selection Suggestions

`feature_selection_suggestions.csv` ranks numeric fields by absolute Pearson correlation with `target_next_congestion_ratio`. This is only a Stage 2 screening aid. The Stage 3 teammate should still use a chronological train/test split and avoid identifier or target-leakage columns when modeling.
