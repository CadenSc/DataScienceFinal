#!/usr/bin/env python3
"""Create recent-activity and congestion features for Ethereum blocks.

The engineered features align with proposal.txt: each row remains a block, and
the added variables describe recent transaction activity, gas utilization,
base-fee behavior, and simple congestion states for later Stage 3 modeling.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


CLEANED_PATH = Path("cleaned_blocks.csv")
ENGINEERED_PATH = Path("engineered_blocks.csv")
FEATURE_SELECTION_PATH = Path("feature_selection_suggestions.csv")

# Ethereum targets roughly 12-second blocks. The 1-hour and 24-hour congestion
# proxies therefore use rolling block windows instead of explicit hourly bins.
# With 5,000 rows, the 24-hour proxy uses all available history because a full
# 24 hours would require about 7,200 blocks.
AVERAGE_BLOCK_SECONDS = 12
BLOCKS_PER_HOUR = round(60 * 60 / AVERAGE_BLOCK_SECONDS)
BLOCKS_PER_24_HOURS = round(24 * 60 * 60 / AVERAGE_BLOCK_SECONDS)

HIGH_CONGESTION_THRESHOLD = 0.80
LOW_CONGESTION_THRESHOLD = 0.40
SHORT_TERM_CONGESTION_THRESHOLD = 0.75
BASE_FEE_SPIKE_RATIO = 1.50
GAS_USED_SPIKE_RATIO = 1.25
TXN_SPIKE_RATIO = 1.50


BASE_COLUMNS = [
    "block_number",
    "slot",
    "block_datetime_utc",
    "block_timestamp_unix",
    "age_raw",
    "age_seconds_at_scrape",
    "blobs_count",
    "blobs_percent",
    "txn_count",
    "fee_recipient",
    "gas_used",
    "gas_used_percent",
    "gas_limit",
    "base_fee_gwei",
    "reward_eth",
    "burnt_fees_eth",
    "burnt_fees_percent",
    "source_page",
    "source_row",
    "source_url",
    "scraped_at_utc",
]

ENGINEERED_COLUMNS = [
    "congestion_ratio",
    "rolling_avg_txn_5",
    "rolling_avg_txn_20",
    "rolling_avg_gas_used_5",
    "rolling_avg_gas_used_20",
    "rolling_avg_base_fee_5",
    "rolling_avg_base_fee_20",
    "rolling_avg_congestion_5",
    "rolling_avg_congestion_20",
    "txn_change_1",
    "gas_used_change_1",
    "base_fee_change_1",
    "txn_volatility_20",
    "gas_used_volatility_20",
    "base_fee_volatility_20",
    "short_term_congestion_indicator",
    "congestion_proxy_1h",
    "congestion_proxy_24h",
    "congestion_proxy_1h_window_blocks",
    "congestion_proxy_24h_window_blocks",
    "congestion_flag_high",
    "congestion_flag_low",
    "congestion_level",
    "one_hour_high_congestion_flag",
    "twenty_four_hour_high_congestion_flag",
    "txn_momentum_5_20",
    "gas_used_momentum_5_20",
    "base_fee_momentum_5_20",
    "congestion_momentum_5_20",
    "utilization_trend_20",
    "gas_used_to_recent_avg",
    "txn_to_recent_avg",
    "base_fee_to_recent_avg",
    "base_fee_spike_flag",
    "gas_used_spike_flag",
    "txn_spike_flag",
    "recent_reward_avg_20",
    "recent_burnt_fees_avg_20",
    "block_interval_seconds",
    "hour_utc",
    "day_of_week_utc",
    "is_weekend_utc",
    "target_next_block_number",
    "target_next_gas_used",
    "target_next_base_fee_gwei",
    "target_next_congestion_ratio",
    "target_next_congestion_flag_high",
]

OUTPUT_COLUMNS = BASE_COLUMNS + ENGINEERED_COLUMNS


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na", "-"}


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None or math.isnan(number):
        return None
    return int(number)


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    output: list[float | None] = []
    active_values: deque[float] = deque()
    active_flags: deque[bool] = deque()
    running_sum = 0.0

    for value in values:
        if value is None:
            active_flags.append(False)
        else:
            active_flags.append(True)
            active_values.append(value)
            running_sum += value

        if len(active_flags) > window:
            had_value = active_flags.popleft()
            if had_value:
                removed = active_values.popleft()
                running_sum -= removed

        count = len(active_values)
        output.append(running_sum / count if count else None)
    return output


def rolling_count_available(values: list[float | None], window: int) -> list[int]:
    counts: list[int] = []
    active_flags: deque[bool] = deque()
    active_count = 0
    for value in values:
        has_value = value is not None
        active_flags.append(has_value)
        active_count += 1 if has_value else 0
        if len(active_flags) > window:
            active_count -= 1 if active_flags.popleft() else 0
        counts.append(active_count)
    return counts


def rolling_stdev(values: list[float | None], window: int) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [value for value in values[start : index + 1] if value is not None]
        if len(window_values) <= 1:
            output.append(0.0)
        else:
            output.append(statistics.pstdev(window_values))
    return output


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def parse_datetime(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    denominator = x_den * y_den
    if denominator == 0:
        return None
    return numerator / denominator


def add_engineered_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: to_int(row.get("block_number")) or 0)

    txn = [to_float(row.get("txn_count")) for row in rows]
    gas_used = [to_float(row.get("gas_used")) for row in rows]
    gas_limit = [to_float(row.get("gas_limit")) for row in rows]
    base_fee = [to_float(row.get("base_fee_gwei")) for row in rows]
    reward = [to_float(row.get("reward_eth")) for row in rows]
    burnt_fees = [to_float(row.get("burnt_fees_eth")) for row in rows]
    timestamps = [to_float(row.get("block_timestamp_unix")) for row in rows]

    congestion = [safe_divide(used, limit) for used, limit in zip(gas_used, gas_limit)]

    rolling_avg_txn_5 = rolling_mean(txn, 5)
    rolling_avg_txn_20 = rolling_mean(txn, 20)
    rolling_avg_gas_used_5 = rolling_mean(gas_used, 5)
    rolling_avg_gas_used_20 = rolling_mean(gas_used, 20)
    rolling_avg_base_fee_5 = rolling_mean(base_fee, 5)
    rolling_avg_base_fee_20 = rolling_mean(base_fee, 20)
    rolling_avg_congestion_5 = rolling_mean(congestion, 5)
    rolling_avg_congestion_20 = rolling_mean(congestion, 20)
    rolling_avg_congestion_1h = rolling_mean(congestion, BLOCKS_PER_HOUR)
    rolling_avg_congestion_24h = rolling_mean(congestion, BLOCKS_PER_24_HOURS)
    congestion_1h_counts = rolling_count_available(congestion, BLOCKS_PER_HOUR)
    congestion_24h_counts = rolling_count_available(congestion, BLOCKS_PER_24_HOURS)
    recent_reward_avg_20 = rolling_mean(reward, 20)
    recent_burnt_fees_avg_20 = rolling_mean(burnt_fees, 20)

    txn_volatility_20 = rolling_stdev(txn, 20)
    gas_used_volatility_20 = rolling_stdev(gas_used, 20)
    base_fee_volatility_20 = rolling_stdev(base_fee, 20)

    for index, row in enumerate(rows):
        row["congestion_ratio"] = congestion[index]
        row["rolling_avg_txn_5"] = rolling_avg_txn_5[index]
        row["rolling_avg_txn_20"] = rolling_avg_txn_20[index]
        row["rolling_avg_gas_used_5"] = rolling_avg_gas_used_5[index]
        row["rolling_avg_gas_used_20"] = rolling_avg_gas_used_20[index]
        row["rolling_avg_base_fee_5"] = rolling_avg_base_fee_5[index]
        row["rolling_avg_base_fee_20"] = rolling_avg_base_fee_20[index]
        row["rolling_avg_congestion_5"] = rolling_avg_congestion_5[index]
        row["rolling_avg_congestion_20"] = rolling_avg_congestion_20[index]

        row["txn_change_1"] = 0.0 if index == 0 else (txn[index] or 0.0) - (txn[index - 1] or 0.0)
        row["gas_used_change_1"] = 0.0 if index == 0 else (gas_used[index] or 0.0) - (gas_used[index - 1] or 0.0)
        row["base_fee_change_1"] = 0.0 if index == 0 else (base_fee[index] or 0.0) - (base_fee[index - 1] or 0.0)

        row["txn_volatility_20"] = txn_volatility_20[index]
        row["gas_used_volatility_20"] = gas_used_volatility_20[index]
        row["base_fee_volatility_20"] = base_fee_volatility_20[index]

        current_congestion = congestion[index]
        row["short_term_congestion_indicator"] = int(
            (rolling_avg_congestion_5[index] or 0.0) >= SHORT_TERM_CONGESTION_THRESHOLD
        )
        row["congestion_proxy_1h"] = rolling_avg_congestion_1h[index]
        row["congestion_proxy_24h"] = rolling_avg_congestion_24h[index]
        row["congestion_proxy_1h_window_blocks"] = congestion_1h_counts[index]
        row["congestion_proxy_24h_window_blocks"] = congestion_24h_counts[index]
        row["congestion_flag_high"] = int((current_congestion or 0.0) >= HIGH_CONGESTION_THRESHOLD)
        row["congestion_flag_low"] = int((current_congestion or 0.0) <= LOW_CONGESTION_THRESHOLD)
        if current_congestion is None:
            row["congestion_level"] = ""
        elif current_congestion >= HIGH_CONGESTION_THRESHOLD:
            row["congestion_level"] = "high"
        elif current_congestion <= LOW_CONGESTION_THRESHOLD:
            row["congestion_level"] = "low"
        else:
            row["congestion_level"] = "medium"
        row["one_hour_high_congestion_flag"] = int(
            (rolling_avg_congestion_1h[index] or 0.0) >= SHORT_TERM_CONGESTION_THRESHOLD
        )
        row["twenty_four_hour_high_congestion_flag"] = int(
            (rolling_avg_congestion_24h[index] or 0.0) >= SHORT_TERM_CONGESTION_THRESHOLD
        )

        row["txn_momentum_5_20"] = (
            None
            if rolling_avg_txn_5[index] is None or rolling_avg_txn_20[index] is None
            else rolling_avg_txn_5[index] - rolling_avg_txn_20[index]
        )
        row["gas_used_momentum_5_20"] = (
            None
            if rolling_avg_gas_used_5[index] is None or rolling_avg_gas_used_20[index] is None
            else rolling_avg_gas_used_5[index] - rolling_avg_gas_used_20[index]
        )
        row["base_fee_momentum_5_20"] = (
            None
            if rolling_avg_base_fee_5[index] is None or rolling_avg_base_fee_20[index] is None
            else rolling_avg_base_fee_5[index] - rolling_avg_base_fee_20[index]
        )
        row["congestion_momentum_5_20"] = (
            None
            if rolling_avg_congestion_5[index] is None or rolling_avg_congestion_20[index] is None
            else rolling_avg_congestion_5[index] - rolling_avg_congestion_20[index]
        )
        row["utilization_trend_20"] = (
            None if current_congestion is None or rolling_avg_congestion_20[index] is None else current_congestion - rolling_avg_congestion_20[index]
        )
        row["gas_used_to_recent_avg"] = safe_divide(gas_used[index], rolling_avg_gas_used_20[index])
        row["txn_to_recent_avg"] = safe_divide(txn[index], rolling_avg_txn_20[index])
        row["base_fee_to_recent_avg"] = safe_divide(base_fee[index], rolling_avg_base_fee_20[index])
        row["base_fee_spike_flag"] = int((row["base_fee_to_recent_avg"] or 0.0) >= BASE_FEE_SPIKE_RATIO)
        row["gas_used_spike_flag"] = int((row["gas_used_to_recent_avg"] or 0.0) >= GAS_USED_SPIKE_RATIO)
        row["txn_spike_flag"] = int((row["txn_to_recent_avg"] or 0.0) >= TXN_SPIKE_RATIO)
        row["recent_reward_avg_20"] = recent_reward_avg_20[index]
        row["recent_burnt_fees_avg_20"] = recent_burnt_fees_avg_20[index]

        if index == 0 or timestamps[index] is None or timestamps[index - 1] is None:
            row["block_interval_seconds"] = ""
        else:
            row["block_interval_seconds"] = timestamps[index] - timestamps[index - 1]

        dt = parse_datetime(row.get("block_datetime_utc"))
        if dt is None:
            row["hour_utc"] = ""
            row["day_of_week_utc"] = ""
            row["is_weekend_utc"] = ""
        else:
            row["hour_utc"] = dt.hour
            row["day_of_week_utc"] = dt.weekday()
            row["is_weekend_utc"] = int(dt.weekday() >= 5)

        if index + 1 < len(rows):
            row["target_next_block_number"] = rows[index + 1].get("block_number")
            row["target_next_gas_used"] = gas_used[index + 1]
            row["target_next_base_fee_gwei"] = base_fee[index + 1]
            row["target_next_congestion_ratio"] = congestion[index + 1]
            row["target_next_congestion_flag_high"] = int((congestion[index + 1] or 0.0) >= HIGH_CONGESTION_THRESHOLD)
        else:
            row["target_next_block_number"] = ""
            row["target_next_gas_used"] = ""
            row["target_next_base_fee_gwei"] = ""
            row["target_next_congestion_ratio"] = ""
            row["target_next_congestion_flag_high"] = ""

    return rows


def write_feature_selection_suggestions(rows: list[dict[str, Any]], path: Path) -> None:
    target_column = "target_next_congestion_ratio"
    excluded_fragments = ("target_", "source_", "fee_recipient", "age_raw", "block_datetime")
    candidate_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column != target_column and not any(fragment in column for fragment in excluded_fragments)
    ]
    excluded_columns = {
        "block_number",
        "slot",
        "block_timestamp_unix",
        "source_page",
        "source_row",
        "congestion_level",
    }
    candidate_columns = [column for column in candidate_columns if column not in excluded_columns]

    suggestions: list[dict[str, Any]] = []
    for column in candidate_columns:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x = to_float(row.get(column))
            y = to_float(row.get(target_column))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        corr = pearson_correlation(xs, ys)
        if corr is None:
            continue
        suggestions.append(
            {
                "feature": column,
                "correlation_with_next_congestion_ratio": corr,
                "absolute_correlation": abs(corr),
                "rows_used": len(xs),
            }
        )

    suggestions.sort(key=lambda row: row["absolute_correlation"], reverse=True)
    write_csv(
        suggestions,
        path,
        [
            "feature",
            "correlation_with_next_congestion_ratio",
            "absolute_correlation",
            "rows_used",
        ],
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create engineered Ethereum block features.")
    parser.add_argument("--input", default=str(CLEANED_PATH), help="Cleaned CSV input path.")
    parser.add_argument("--output", default=str(ENGINEERED_PATH), help="Engineered CSV output path.")
    parser.add_argument(
        "--feature-selection-output",
        default=str(FEATURE_SELECTION_PATH),
        help="CSV of correlation-based feature suggestions for Stage 3.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_path = Path(args.output)
    feature_selection_path = Path(args.feature_selection_output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run clean_blocks.py first.", file=sys.stderr)
        return 1

    rows = read_csv(input_path)
    engineered_rows = add_engineered_features(rows)
    write_csv(engineered_rows, output_path, OUTPUT_COLUMNS)
    write_feature_selection_suggestions(engineered_rows, feature_selection_path)

    print(f"Loaded {len(rows)} cleaned rows.")
    print(f"Saved {len(engineered_rows)} engineered rows to {output_path}.")
    print(
        "Rolling window assumptions: "
        f"1 hour is {BLOCKS_PER_HOUR} blocks and 24 hours is {BLOCKS_PER_24_HOURS} blocks "
        f"at {AVERAGE_BLOCK_SECONDS} seconds per block."
    )
    print(f"Wrote feature selection suggestions to {feature_selection_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
