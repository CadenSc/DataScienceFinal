# DataScienceFinal

Stage 1 and Stage 2 workflow for the Ethereum block activity project described in `proposal.txt`.

## Files

- `scrape_blocks.py`: scrapes Etherscan block pages and saves `raw_blocks.csv`.
- `clean_blocks.py`: cleans and standardizes raw fields, saves `cleaned_blocks.csv`, and writes `data_quality_report.md`.
- `feature_engineering.py`: adds rolling recent-activity, congestion, spike, and next-block target features, saving `engineered_blocks.csv`.
- `eda.py`: creates Stage 1 EDA visuals and interpretations under `eda_outputs/`.
- `stage1_stage2_summary.md`: short report covering source, schema, ethics, cleaning, features, and EDA.
- `optional_notes_for_stage3.md`: modeling handoff notes for the teammate handling Stage 3.

## Run Order

```bash
python3 scrape_blocks.py --pages 50 --page-size 100 --delay 0.45 --parse-retries 5 --output raw_blocks.csv
python3 clean_blocks.py
python3 feature_engineering.py
python3 eda.py
```

If a scraped raw file has complete `raw_row_text` values but shifted columns because Etherscan exposed hidden table fields, repair it locally without scraping again:

```bash
python3 scrape_blocks.py --repair-existing raw_blocks.csv --output raw_blocks.csv
```

The current run contains 5,000 raw rows and 4,997 cleaned/engineered rows after removing duplicate block numbers caused by pagination movement.
