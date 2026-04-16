# EDA Interpretations

Dataset analyzed: `engineered_blocks.csv` with 4997 engineered block rows.

## Visuals Created

- `txn_count_distribution.svg`: transaction counts vary by block, with median 244.00 and range 0.00 to 1069.00. This supports the proposal's focus on changing block activity over time.
- `gas_used_distribution.svg`: gas used ranges from 0.00 to 59999764.00; high values are retained because they are part of congestion behavior.
- `base_fee_trend.svg`: base fee is plotted in chronological block order, with observed range 0.031 to 2.363 Gwei.
- `congestion_ratio_trend.svg`: congestion ratio has median 0.4789; 598 rows meet the high-congestion threshold of 0.80.
- `rolling_activity_trends.svg`: rolling 20-block transaction count, gas used, and base fee are normalized together to compare recent activity momentum.
- `correlation_heatmap.svg`: strongest observed numeric relationships include `congestion_ratio` vs `gas_used` = 1.00; `rolling_avg_congestion_20` vs `rolling_avg_gas_used_20` = 1.00; `base_fee_gwei` vs `rolling_avg_base_fee_20` = 0.96; `base_fee_gwei` vs `burnt_fees_eth` = 0.88; `burnt_fees_eth` vs `rolling_avg_base_fee_20` = 0.79. Correlation is descriptive only and should not be interpreted as model performance.
- `before_after_missingness.svg`: block: raw 0.0% to cleaned 0.0%; txn: raw 0.0% to cleaned 0.0%; gas used: raw 0.0% to cleaned 0.0%; gas limit: raw 0.0% to cleaned 0.0%; base fee: raw 0.0% to cleaned 0.0%; reward: raw 0.0% to cleaned 0.0%; burnt fees: raw 0.0% to cleaned 0.0%
- `spike_inspection.svg`: high-congestion blocks are highlighted for outlier/spike inspection while preserving those rows for analysis.

## Stage 1/2 Takeaway

The visual checks support a conservative Stage 1 and Stage 2 workflow: the dataset is block-level, recent, public, and suited to analyzing short-term network utilization. Cleaning improves consistency without removing meaningful congestion spikes. Engineered rolling windows give the Stage 3 teammate interpretable predictors for Logistic Regression and Random Forest experiments without claiming predictive success yet.
