#!/usr/bin/env python3
"""Clean raw Etherscan block data for Stage 2.

The raw CSV intentionally preserves Etherscan text values. This script converts
those text fields into typed block-level metrics while documenting duplicate,
missing-value, unit-stripping, timestamp, and outlier decisions.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_PATH = Path("raw_blocks.csv")
CLEANED_PATH = Path("cleaned_blocks.csv")
REPORT_PATH = Path("data_quality_report.md")

CLEANED_FIELDNAMES = [
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

REQUIRED_NUMERIC_FIELDS = ["block_number", "txn_count", "gas_used", "gas_limit", "base_fee_gwei"]
KEY_RAW_FIELDS = [
    "block_number_raw",
    "txn_count_raw",
    "gas_used_raw",
    "gas_limit_raw",
    "base_fee_raw",
    "reward_raw",
    "burnt_fees_raw",
]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na", "-"}


def clean_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def first_number(value: Any) -> float | None:
    """Extract the first numeric value from a text field."""
    if is_missing(value):
        return None
    text = clean_whitespace(value)
    text = re.sub(r"(\d+)\s+\.\s+(\d+)", r"\1.\2", text)
    text = text.replace(",", "")
    text = re.sub(r"(?i)\b(gwei|eth|wei|gas|txns?|transactions?|fee recipient:)\b", " ", text)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def first_int(value: Any) -> int | None:
    number = first_number(value)
    if number is None or math.isnan(number):
        return None
    return int(number)


def extract_percent(value: Any) -> float | None:
    if is_missing(value):
        return None
    text = clean_whitespace(value).replace(",", "")
    match = re.search(r"\(([-+]?\d*\.?\d+)\s*%\)", text)
    if not match:
        match = re.search(r"([-+]?\d*\.?\d+)\s*%", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_age_seconds(age: Any) -> int | None:
    if is_missing(age):
        return None
    text = clean_whitespace(age).lower()
    match = re.search(r"(\d+)\s+(sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|day|days)", text)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("sec"):
        return value
    if unit.startswith("min"):
        return value * 60
    if unit in {"hr", "hrs", "hour", "hours"}:
        return value * 60 * 60
    if unit.startswith("day"):
        return value * 24 * 60 * 60
    return None


def parse_datetime_utc(raw_datetime: Any, raw_timestamp: Any, scraped_at_utc: Any, age_raw: Any) -> tuple[str, int | None]:
    """Return an ISO UTC timestamp and Unix timestamp when possible.

    Etherscan usually provides a Unix timestamp in the block table. When it is
    available, we treat it as the authoritative timestamp. If not, we parse the
    displayed datetime as UTC. As a final fallback, a relative age is converted
    using the scrape time, which is approximate because age changes while pages
    are loaded.
    """
    timestamp = first_int(raw_timestamp)
    if timestamp:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat(timespec="seconds"), timestamp

    text = clean_whitespace(raw_datetime)
    if text:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%b-%d-%Y %I:%M:%S %p %Z", "%b-%d-%Y %I:%M:%S %p"):
            try:
                parsed = datetime.strptime(text.replace("UTC", "").strip(), fmt.replace(" %Z", ""))
                parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.isoformat(timespec="seconds"), int(parsed.timestamp())
            except ValueError:
                continue

    scraped_text = clean_whitespace(scraped_at_utc)
    age_seconds = parse_age_seconds(age_raw)
    if scraped_text and age_seconds is not None:
        try:
            scraped = datetime.fromisoformat(scraped_text.replace("Z", "+00:00"))
            approximate = scraped.timestamp() - age_seconds
            dt = datetime.fromtimestamp(approximate, tz=timezone.utc)
            return dt.isoformat(timespec="seconds"), int(approximate)
        except ValueError:
            pass

    return "", None


def standardize_fee_recipient(value: Any) -> str:
    text = clean_whitespace(value)
    text = re.sub(r"(?i)^fee recipient:\s*", "", text).strip()
    return text or "Unknown"


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def count_missing(rows: list[dict[str, str]], columns: list[str]) -> dict[str, int]:
    return {column: sum(1 for row in rows if is_missing(row.get(column))) for column in columns}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def iqr_outlier_count(values: list[float]) -> tuple[int, float, float]:
    values = [value for value in values if value is not None and not math.isnan(value)]
    if len(values) < 4:
        return 0, float("nan"), float("nan")
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = sum(1 for value in values if value < lower or value > upper)
    return count, lower, upper


def clean_row(raw: dict[str, str], invalid_counts: dict[str, int]) -> dict[str, Any]:
    numeric_map = {
        "block_number": ("block_number_raw", first_int),
        "slot": ("slot_raw", first_int),
        "blobs_count": ("blobs_raw", first_int),
        "txn_count": ("txn_count_raw", first_int),
        "gas_used": ("gas_used_raw", first_int),
        "gas_limit": ("gas_limit_raw", first_int),
        "base_fee_gwei": ("base_fee_raw", first_number),
        "reward_eth": ("reward_raw", first_number),
        "burnt_fees_eth": ("burnt_fees_raw", first_number),
    }

    cleaned: dict[str, Any] = {}
    for output_column, (raw_column, parser) in numeric_map.items():
        value = parser(raw.get(raw_column))
        cleaned[output_column] = value
        if value is None and not is_missing(raw.get(raw_column)):
            invalid_counts[output_column] = invalid_counts.get(output_column, 0) + 1

    cleaned["blobs_percent"] = extract_percent(raw.get("blobs_raw"))
    cleaned["gas_used_percent"] = extract_percent(raw.get("gas_used_raw"))
    cleaned["burnt_fees_percent"] = extract_percent(raw.get("burnt_fees_raw"))
    cleaned["fee_recipient"] = standardize_fee_recipient(raw.get("fee_recipient_raw"))
    cleaned["age_raw"] = clean_whitespace(raw.get("age_raw"))
    cleaned["age_seconds_at_scrape"] = parse_age_seconds(raw.get("age_raw"))

    block_datetime_utc, block_timestamp_unix = parse_datetime_utc(
        raw.get("block_datetime_raw"),
        raw.get("block_timestamp_unix_raw"),
        raw.get("scraped_at_utc"),
        raw.get("age_raw"),
    )
    cleaned["block_datetime_utc"] = block_datetime_utc
    cleaned["block_timestamp_unix"] = block_timestamp_unix

    cleaned["source_page"] = first_int(raw.get("source_page"))
    cleaned["source_row"] = first_int(raw.get("source_row"))
    cleaned["source_url"] = clean_whitespace(raw.get("source_url"))
    cleaned["scraped_at_utc"] = clean_whitespace(raw.get("scraped_at_utc"))
    return cleaned


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def clean_blocks(raw_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    invalid_counts: dict[str, int] = {}
    cleaned_candidates = [clean_row(row, invalid_counts) for row in raw_rows]

    missing_required_removed = 0
    required_rows: list[dict[str, Any]] = []
    for row in cleaned_candidates:
        if any(row.get(field) is None for field in REQUIRED_NUMERIC_FIELDS):
            missing_required_removed += 1
            continue
        required_rows.append(row)

    deduped: list[dict[str, Any]] = []
    seen_blocks: set[int] = set()
    duplicate_removed = 0
    for row in sorted(
        required_rows,
        key=lambda item: (
            item.get("source_page") if item.get("source_page") is not None else 999999,
            item.get("source_row") if item.get("source_row") is not None else 999999,
        ),
    ):
        block_number = row["block_number"]
        if block_number in seen_blocks:
            duplicate_removed += 1
            continue
        seen_blocks.add(block_number)
        deduped.append(row)

    # Chronological ordering makes rolling windows in feature_engineering.py
    # represent recent past block activity.
    deduped.sort(key=lambda item: item["block_number"])

    summary = {
        "raw_rows": len(raw_rows),
        "cleaned_candidate_rows": len(cleaned_candidates),
        "rows_removed_missing_required": missing_required_removed,
        "duplicates_removed": duplicate_removed,
        "cleaned_rows": len(deduped),
        "raw_missing_counts": count_missing(raw_rows, KEY_RAW_FIELDS),
        "cleaned_missing_counts": count_missing(
            [{key: format_value(value) for key, value in row.items()} for row in deduped],
            CLEANED_FIELDNAMES,
        ),
        "invalid_conversion_counts": invalid_counts,
        "outlier_counts": {},
    }

    for column in ["txn_count", "gas_used", "gas_limit", "base_fee_gwei", "reward_eth", "burnt_fees_eth"]:
        values = [float(row[column]) for row in deduped if row.get(column) is not None]
        count, lower, upper = iqr_outlier_count(values)
        summary["outlier_counts"][column] = {"count": count, "lower_bound": lower, "upper_bound": upper}

    return deduped, summary


def markdown_table(mapping: dict[str, Any]) -> str:
    lines = ["| Field | Value |", "|---|---:|"]
    for key, value in mapping.items():
        if isinstance(value, float):
            rendered = f"{value:.4g}" if not math.isnan(value) else ""
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    return "\n".join(lines)


def write_quality_report(summary: dict[str, Any], report_path: Path) -> None:
    outlier_lines = ["| Field | IQR outlier count | Lower bound | Upper bound |", "|---|---:|---:|---:|"]
    for field, values in summary["outlier_counts"].items():
        lower = values["lower_bound"]
        upper = values["upper_bound"]
        outlier_lines.append(
            f"| `{field}` | {values['count']} | {lower:.4g} | {upper:.4g} |"
        )

    report = f"""# Data Quality Report

This report is generated by `clean_blocks.py` for the Ethereum block-level dataset described in `proposal.txt`.

## Before And After Counts

| Metric | Count |
|---|---:|
| Raw rows loaded | {summary["raw_rows"]} |
| Rows removed for missing required numeric fields | {summary["rows_removed_missing_required"]} |
| Duplicate block rows removed | {summary["duplicates_removed"]} |
| Cleaned rows saved | {summary["cleaned_rows"]} |

## Raw Missing Counts

{markdown_table(summary["raw_missing_counts"])}

## Cleaned Missing Counts

{markdown_table(summary["cleaned_missing_counts"])}

## Invalid Numeric Conversion Counts

{markdown_table(summary["invalid_conversion_counts"] or {"none": 0})}

## Outlier Inspection

{chr(10).join(outlier_lines)}

Outliers are counted using the 1.5 IQR rule for inspection only. They are kept in the cleaned dataset because Ethereum block activity naturally has bursts, spikes, and uneven network demand. Removing those rows would remove exactly the congestion behavior this project is trying to study.

## Cleaning Decisions

- Duplicate handling: duplicate `block_number` rows are removed after sorting by scrape page and row, keeping the first observed record.
- Missing values: rows missing required numeric fields (`block_number`, `txn_count`, `gas_used`, `gas_limit`, or `base_fee_gwei`) are removed because they cannot support the Stage 1/2 activity and congestion analysis. Optional fields such as reward or burnt fees are retained as blank when unavailable.
- Text and unit stripping: commas, percentages, parentheses, and units such as `Gwei` and `ETH` are stripped during numeric conversion while the raw CSV remains unchanged for traceability.
- Timestamp and age handling: Unix block timestamps are used when present. If unavailable, the displayed datetime is treated as UTC. Relative age is preserved and converted to approximate seconds only as a fallback because it changes during scraping.
- Outlier strategy: outliers are documented but retained because high gas use, fee spikes, and bursts in transactions are meaningful blockchain behavior rather than obvious data entry errors.
"""
    report_path.write_text(report, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Etherscan block rows.")
    parser.add_argument("--input", default=str(RAW_PATH), help="Input raw CSV path.")
    parser.add_argument("--output", default=str(CLEANED_PATH), help="Cleaned CSV output path.")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Markdown quality report output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run scrape_blocks.py first.", file=sys.stderr)
        return 1

    raw_rows = read_csv(input_path)
    cleaned_rows, summary = clean_blocks(raw_rows)
    write_csv(cleaned_rows, output_path, CLEANED_FIELDNAMES)
    write_quality_report(summary, report_path)

    print(f"Loaded {summary['raw_rows']} raw rows.")
    print(f"Removed {summary['rows_removed_missing_required']} rows with missing required metrics.")
    print(f"Removed {summary['duplicates_removed']} duplicate block rows.")
    print(f"Saved {summary['cleaned_rows']} cleaned rows to {output_path}.")
    print(f"Wrote quality report to {report_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
