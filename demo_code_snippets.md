# Demo Code Snippets

Use this file during the demo as a quick reference for the main code methods behind Stage 1 and Stage 2. The goal is to show enough code to prove the pipeline exists without turning the presentation into a full code walkthrough.

## Slide 3: Data Acquisition

File: `scrape_blocks.py`

What this shows: the scraper targets 50 Etherscan pages with 100 block rows per page, giving 5,000 raw rows.

```python
BASE_URL = "https://etherscan.io/blocks"
DEFAULT_PAGES = 50
DEFAULT_PAGE_SIZE = 100
```

```python
def page_url(page: int, page_size: int) -> str:
    return f"{BASE_URL}?ps={page_size}&p={page}"
```

> We collected block-level Ethereum data from Etherscan by scraping 50 paginated block pages, with 100 rows per page. That gave us 5,000 raw block rows before cleaning.

## Slide 4: Data Cleaning

File: `clean_blocks.py`

What this shows: the required numeric fields that must exist for the row to be useful.

```python
REQUIRED_NUMERIC_FIELDS = [
    "block_number",
    "txn_count",
    "gas_used",
    "gas_limit",
    "base_fee_gwei",
]
```

What this shows: raw Etherscan strings are cleaned by removing commas and units before numeric conversion.

```python
text = text.replace(",", "")
text = re.sub(
    r"(?i)\b(gwei|eth|wei|gas|txns?|transactions?|fee recipient:)\b",
    " ",
    text,
)
```

What this shows: duplicate block numbers are removed, keeping the first observed row.

```python
seen_blocks: set[int] = set()
for row in sorted(required_rows, key=lambda item: (item.get("source_page"), item.get("source_row"))):
    block_number = row["block_number"]
    if block_number in seen_blocks:
        duplicate_removed += 1
        continue
    seen_blocks.add(block_number)
    deduped.append(row)
```

> The raw scrape included text formatting like commas, percentages, Gwei, and ETH. Cleaning stripped those units, converted the fields to numeric values, and removed 3 duplicate block rows caused by pagination movement.

## Slide 5: Congestion Ratio

File: `feature_engineering.py`

What this shows: congestion ratio measures how full a block is relative to the gas limit.

```python
congestion = [
    safe_divide(used, limit)
    for used, limit in zip(gas_used, gas_limit)
]
```



> Congestion ratio is gas used divided by gas limit. A value closer to 1 means the block was closer to full, so it is a direct measure of block utilization.

## Slide 5: Rolling Averages

File: `feature_engineering.py`

What this shows: recent activity is summarized with 5-block and 20-block rolling averages.

```python
rolling_avg_txn_5 = rolling_mean(txn, 5)
rolling_avg_txn_20 = rolling_mean(txn, 20)
rolling_avg_gas_used_5 = rolling_mean(gas_used, 5)
rolling_avg_gas_used_20 = rolling_mean(gas_used, 20)
rolling_avg_base_fee_5 = rolling_mean(base_fee, 5)
rolling_avg_base_fee_20 = rolling_mean(base_fee, 20)
rolling_avg_congestion_5 = rolling_mean(congestion, 5)
rolling_avg_congestion_20 = rolling_mean(congestion, 20)
```


> Rolling averages smooth out noisy block-by-block changes and describe the recent network environment before each block.

## Slide 5: One-Block Change Features

File: `feature_engineering.py`

What this shows: current block activity is compared with the previous block.

```python
row["txn_change_1"] = 0.0 if index == 0 else (txn[index] or 0.0) - (txn[index - 1] or 0.0)
row["gas_used_change_1"] = 0.0 if index == 0 else (gas_used[index] or 0.0) - (gas_used[index - 1] or 0.0)
row["base_fee_change_1"] = 0.0 if index == 0 else (base_fee[index] or 0.0) - (base_fee[index - 1] or 0.0)
```


> These features show whether transaction count, gas used, or base fee increased or decreased from the previous block.

## Slide 5: Volatility Features

File: `feature_engineering.py`

What this shows: volatility is measured as standard deviation over the last 20 blocks.

```python
txn_volatility_20 = rolling_stdev(txn, 20)
gas_used_volatility_20 = rolling_stdev(gas_used, 20)
base_fee_volatility_20 = rolling_stdev(base_fee, 20)
```



> Volatility captures instability. High volatility means recent blocks have been changing more sharply, which can indicate less predictable network conditions.

## Slide 5: Spike Flags

File: `feature_engineering.py`

What this shows: spike thresholds compare the current block to its recent average.

```python
BASE_FEE_SPIKE_RATIO = 1.50
GAS_USED_SPIKE_RATIO = 1.25
TXN_SPIKE_RATIO = 1.50
```

```python
row["base_fee_spike_flag"] = int((row["base_fee_to_recent_avg"] or 0.0) >= BASE_FEE_SPIKE_RATIO)
row["gas_used_spike_flag"] = int((row["gas_used_to_recent_avg"] or 0.0) >= GAS_USED_SPIKE_RATIO)
row["txn_spike_flag"] = int((row["txn_to_recent_avg"] or 0.0) >= TXN_SPIKE_RATIO)
```

What to say:

> Spike flags are yes/no indicators for unusual jumps. Transaction count and base fee use a 1.5x threshold, while gas used uses a 1.25x threshold because gas usage is more directly bounded by block capacity.

## Slide 5: Congestion Windows

File: `feature_engineering.py`

What this shows: time windows are approximated using Ethereum's average block time.

```python
AVERAGE_BLOCK_SECONDS = 12
BLOCKS_PER_HOUR = round(60 * 60 / AVERAGE_BLOCK_SECONDS)
BLOCKS_PER_24_HOURS = round(24 * 60 * 60 / AVERAGE_BLOCK_SECONDS)
```

```python
rolling_avg_congestion_1h = rolling_mean(congestion, BLOCKS_PER_HOUR)
rolling_avg_congestion_24h = rolling_mean(congestion, BLOCKS_PER_24_HOURS)
```



> Because the data is block-based instead of hourly, the 1-hour and 24-hour features are rolling block-window approximations. One hour is about 300 blocks. A full 24-hour window would require about 7,200 blocks, so our 5,000-block dataset only gives a partial 24-hour proxy. That feature becomes more useful with a larger scrape.

## Slide 6: EDA Charts

File: `eda.py`

What this shows: the script generates chart files used in the slides.

```python
draw_histogram(
    numeric_values(rows, "txn_count"),
    "Transaction Count Distribution",
    "Transactions per block",
    output_dir / "txn_count_distribution.svg",
)
```

```python
draw_line_chart(
    [
        ("congestion_ratio", [to_float(row.get("congestion_ratio")) for row in rows], "#2f80ed"),
        ("rolling_avg_congestion_20", [to_float(row.get("rolling_avg_congestion_20")) for row in rows], "#d62728"),
    ],
    "Congestion Ratio Over Recent Blocks",
    "Gas used / gas limit",
    output_dir / "congestion_ratio_trend.svg",
)
```


> EDA helps verify that the data is meaningful before modeling. We looked at distributions, trends across block number, congestion behavior, and correlations between numeric features.

## Slide 7: Modeling Handoff

File: `feature_engineering.py`

What this shows: the engineered dataset includes next-block targets for Stage 3.

```python
row["target_next_block_number"] = rows[index + 1].get("block_number")
row["target_next_gas_used"] = gas_used[index + 1]
row["target_next_base_fee_gwei"] = base_fee[index + 1]
row["target_next_congestion_ratio"] = congestion[index + 1]
row["target_next_congestion_flag_high"] = int(
    (congestion[index + 1] or 0.0) >= HIGH_CONGESTION_THRESHOLD
)
```



> The Stage 1 and 2 pipeline creates the modeling-ready dataset. The next-block target columns let Stage 3 test whether recent block behavior helps explain upcoming gas usage or congestion.


- `stage1_stage2_summary.md`
- `raw_blocks.csv`
- `data_quality_report.md`
- `feature_engineering.py`
- `engineered_blocks.csv`
- `eda_outputs/txn_count_distribution.svg`
- `eda_outputs/gas_used_distribution.svg`
- `eda_outputs/congestion_ratio_trend.svg`
- `eda_outputs/correlation_heatmap.svg`
- `model_outputs/model_comparison.md`

## Best Three Snippets 

1. Data acquisition: `DEFAULT_PAGES = 50` and `DEFAULT_PAGE_SIZE = 100`
2. Data cleaning: required numeric fields and unit stripping
3. Feature engineering: congestion ratio, rolling averages, and spike thresholds
