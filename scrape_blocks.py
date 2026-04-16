#!/usr/bin/env python3
"""Scrape recent Ethereum block-level metadata from Etherscan.

This script supports the Stage 1 data acquisition work described in
proposal.txt. It collects recent block rows from Etherscan's public blocks
pages and saves the raw text values to raw_blocks.csv. The cleaning script
handles numeric conversion and unit stripping later, so this file intentionally
preserves the source strings as they appeared on the page.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://etherscan.io/blocks"
DEFAULT_PAGES = 50
DEFAULT_PAGE_SIZE = 100
DEFAULT_DELAY_SECONDS = 0.35
REQUEST_TIMEOUT_SECONDS = 30

RAW_FIELDNAMES = [
    "source_url",
    "source_page",
    "source_row",
    "scraped_at_utc",
    "block_number_raw",
    "slot_raw",
    "age_raw",
    "block_datetime_raw",
    "block_timestamp_unix_raw",
    "blobs_raw",
    "txn_count_raw",
    "fee_recipient_raw",
    "gas_used_raw",
    "gas_limit_raw",
    "base_fee_raw",
    "reward_raw",
    "burnt_fees_raw",
    "parse_method",
    "raw_row_text",
]


@dataclass
class ScrapeConfig:
    pages: int
    page_size: int
    delay_seconds: float
    parse_retries: int
    output_path: Path
    repair_existing_path: Path | None = None


def clean_whitespace(value: str | None) -> str:
    """Normalize spacing while preserving the original text content."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def normalize_header(value: str) -> str:
    value = clean_whitespace(value).lower()
    value = value.replace("(eth)", "")
    value = value.replace("/", " ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def page_url(page: int, page_size: int) -> str:
    return f"{BASE_URL}?ps={page_size}&p={page}"


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://etherscan.io/",
        }
    )
    return session


def fetch_page(session: requests.Session, url: str, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            html = response.text
            if "Access Denied" in html or "Just a moment" in html:
                raise RuntimeError("Etherscan returned an anti-bot or access-check page.")
            return html
        except Exception as exc:  # noqa: BLE001 - retry network/parser failures.
            last_error = exc
            if attempt < retries:
                time.sleep(1.25 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def extract_table_headers(table) -> list[str]:
    header_cells = table.select("thead th")
    if not header_cells:
        first_row = table.find("tr")
        if first_row:
            header_cells = first_row.find_all(["th", "td"])
    return [normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells]


def header_lookup(headers: list[str], cells: list[str], candidates: Iterable[str]) -> str:
    """Return a cell value using fuzzy header names."""
    wanted = tuple(candidates)
    for idx, header in enumerate(headers):
        if idx >= len(cells):
            continue
        if any(candidate in header for candidate in wanted):
            return cells[idx]
    return ""


def extract_block_number_from_links(row) -> str:
    for anchor in row.find_all("a"):
        text = clean_whitespace(anchor.get_text(" ", strip=True))
        href = anchor.get("href") or ""
        if re.search(r"/block/\d+", href) and re.fullmatch(r"\d+", text):
            return text
    return ""


def extract_slot_from_links(row) -> str:
    for anchor in row.find_all("a"):
        text = clean_whitespace(anchor.get_text(" ", strip=True))
        href = anchor.get("href") or ""
        if "slot" in href.lower() and re.fullmatch(r"\d+", text):
            return text
    return ""


def extract_datetime(text: str) -> str:
    match = re.search(r"\b20\d{2}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2}\b", text)
    return match.group(0) if match else ""


def extract_unix_timestamp(text: str) -> str:
    matches = re.findall(r"\b1[5-9]\d{8,9}\b", text)
    return matches[-1] if matches else ""


def extract_age(text: str) -> str:
    match = re.search(
        r"\b\d+\s+(?:sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|day|days)\s+ago\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else ""


def extract_metrics_from_flat_text(text: str) -> dict[str, str]:
    """Extract trailing metric fields from a flattened Etherscan row string."""
    text = clean_whitespace(text)
    metric_pattern = re.compile(
        r"(?P<blobs>\d+\s*\(\s*[-+]?\d*\.?\d+\s*%\s*\))\s+"
        r"(?P<txn>\d+)\s+"
        r"(?P<fee>.+?)\s+"
        r"(?P<gas_used>\d[\d,]*\s*\(\s*[-+]?\d*\.?\d+\s*%\s*\))\s+"
        r"(?P<gas_limit>\d[\d,]*)\s+"
        r"(?P<base_fee>\d+\s*(?:\.\s*)?\d*\s*Gwei)\s+"
        r"(?P<reward>\d+\s*(?:\.\s*)?\d*\s*ETH)\s+"
        r"(?P<burnt>\d+(?:\.\d+)?\s*(?:\(\s*[-+]?\d*\.?\d+\s*%\s*\))?)\s*$",
        flags=re.IGNORECASE,
    )
    match = metric_pattern.search(text)
    if not match:
        return {}
    return {
        "blobs_raw": clean_whitespace(match.group("blobs")),
        "txn_count_raw": clean_whitespace(match.group("txn")),
        "fee_recipient_raw": clean_whitespace(match.group("fee")),
        "gas_used_raw": clean_whitespace(match.group("gas_used")),
        "gas_limit_raw": clean_whitespace(match.group("gas_limit")),
        "base_fee_raw": clean_whitespace(match.group("base_fee")),
        "reward_raw": clean_whitespace(match.group("reward")),
        "burnt_fees_raw": clean_whitespace(match.group("burnt")),
    }


def find_metric_start(cells: list[str]) -> tuple[int, int] | None:
    """Find datetime, age, timestamp, then return datetime index and metric start."""
    for index in range(1, max(1, len(cells) - 8)):
        if (
            extract_datetime(cells[index])
            and index + 2 < len(cells)
            and extract_age(cells[index + 1])
            and extract_unix_timestamp(cells[index + 2])
        ):
            return index, index + 3
    return None


def map_cells(headers: list[str], cells: list[str], row) -> dict[str, str]:
    """Map table cells into the raw schema with header and position fallbacks."""
    row_text = clean_whitespace(row.get_text(" ", strip=True))
    block_info = header_lookup(headers, cells, ["block"]) or (cells[0] if cells else "")
    block_info = clean_whitespace(block_info)

    block_number = extract_block_number_from_links(row)
    if not block_number:
        match = re.search(r"\b\d{7,}\b", block_info or row_text)
        block_number = match.group(0) if match else ""

    slot = header_lookup(headers, cells, ["slot"])
    if not slot:
        slot = extract_slot_from_links(row)

    block_datetime = extract_datetime(block_info) or extract_datetime(row_text)
    age = header_lookup(headers, cells, ["age"])
    if not age:
        age = extract_age(block_info) or extract_age(row_text)
    unix_timestamp = extract_unix_timestamp(block_info) or extract_unix_timestamp(row_text)

    values = {
        "block_number_raw": block_number,
        "slot_raw": slot,
        "age_raw": age,
        "block_datetime_raw": block_datetime,
        "block_timestamp_unix_raw": unix_timestamp,
        "blobs_raw": header_lookup(headers, cells, ["blob"]),
        "txn_count_raw": header_lookup(headers, cells, ["txn", "txns", "transaction"]),
        "fee_recipient_raw": header_lookup(headers, cells, ["fee_recipient", "recipient"]),
        "gas_used_raw": header_lookup(headers, cells, ["gas_used"]),
        "gas_limit_raw": header_lookup(headers, cells, ["gas_limit"]),
        "base_fee_raw": header_lookup(headers, cells, ["base_fee"]),
        "reward_raw": header_lookup(headers, cells, ["reward"]),
        "burnt_fees_raw": header_lookup(headers, cells, ["burnt"]),
        "raw_row_text": row_text,
    }

    # Current Etherscan layout includes sortable block/epoch/slot/date fields
    # before the visible metrics. Locate the datetime, age, unix timestamp trio
    # and read metrics immediately after it.
    metric_location = find_metric_start(cells)
    if metric_location and len(cells) >= metric_location[1] + 8:
        datetime_index, metric_start = metric_location
        values.update(
            {
                "block_datetime_raw": cells[datetime_index],
                "age_raw": cells[datetime_index + 1],
                "block_timestamp_unix_raw": cells[datetime_index + 2],
                "blobs_raw": cells[metric_start],
                "txn_count_raw": cells[metric_start + 1],
                "fee_recipient_raw": cells[metric_start + 2],
                "gas_used_raw": cells[metric_start + 3],
                "gas_limit_raw": cells[metric_start + 4],
                "base_fee_raw": cells[metric_start + 5],
                "reward_raw": cells[metric_start + 6],
                "burnt_fees_raw": cells[metric_start + 7],
            }
        )

    # Etherscan's table has changed over time. If headers fail or are hidden,
    # map by the common row layout:
    # block info, blobs, txn, fee recipient, gas used, gas limit, base fee,
    # reward, burnt fees.
    elif len(cells) >= 9 and not values["txn_count_raw"]:
        values.update(
            {
                "blobs_raw": cells[1],
                "txn_count_raw": cells[2],
                "fee_recipient_raw": cells[3],
                "gas_used_raw": cells[4],
                "gas_limit_raw": cells[5],
                "base_fee_raw": cells[6],
                "reward_raw": cells[7],
                "burnt_fees_raw": cells[8],
            }
        )
    elif len(cells) >= 8 and not values["txn_count_raw"]:
        values.update(
            {
                "txn_count_raw": cells[1],
                "fee_recipient_raw": cells[2],
                "gas_used_raw": cells[3],
                "gas_limit_raw": cells[4],
                "base_fee_raw": cells[5],
                "reward_raw": cells[6],
                "burnt_fees_raw": cells[7],
            }
        )

    flat_metrics = extract_metrics_from_flat_text(row_text)
    if flat_metrics and not re.fullmatch(r"\d+", values.get("txn_count_raw", "")):
        values.update(flat_metrics)

    return {key: clean_whitespace(value) for key, value in values.items()}


def parse_blocks_from_html(html: str, url: str, page: int, scraped_at_utc: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_rows: list[dict[str, str]] = []

    for table in soup.find_all("table"):
        headers = extract_table_headers(table)
        header_blob = " ".join(headers)
        if "block" not in header_blob or "gas" not in header_blob:
            continue

        body_rows = table.select("tbody tr") or table.find_all("tr")
        for row_index, row in enumerate(body_rows, start=1):
            cells = [clean_whitespace(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
            if not cells:
                continue
            mapped = map_cells(headers, cells, row)
            if not mapped["block_number_raw"]:
                continue
            mapped.update(
                {
                    "source_url": url,
                    "source_page": str(page),
                    "source_row": str(row_index),
                    "scraped_at_utc": scraped_at_utc,
                    "parse_method": "html_table",
                }
            )
            parsed_rows.append({field: mapped.get(field, "") for field in RAW_FIELDNAMES})

    if parsed_rows:
        return parsed_rows

    # Last-resort fallback: parse block links from the page text. This usually
    # captures fewer fields, but keeps acquisition from failing silently if
    # Etherscan makes a small HTML layout change.
    fallback_rows: list[dict[str, str]] = []
    for row_index, anchor in enumerate(soup.find_all("a", href=re.compile(r"/block/\d+")), start=1):
        block_number = clean_whitespace(anchor.get_text(" ", strip=True))
        if not re.fullmatch(r"\d{7,}", block_number):
            continue
        fallback_rows.append(
            {
                "source_url": url,
                "source_page": str(page),
                "source_row": str(row_index),
                "scraped_at_utc": scraped_at_utc,
                "block_number_raw": block_number,
                "slot_raw": "",
                "age_raw": "",
                "block_datetime_raw": "",
                "block_timestamp_unix_raw": "",
                "blobs_raw": "",
                "txn_count_raw": "",
                "fee_recipient_raw": "",
                "gas_used_raw": "",
                "gas_limit_raw": "",
                "base_fee_raw": "",
                "reward_raw": "",
                "burnt_fees_raw": "",
                "parse_method": "block_link_fallback",
                "raw_row_text": clean_whitespace(anchor.parent.get_text(" ", strip=True) if anchor.parent else block_number),
            }
        )
    return fallback_rows


def scrape_blocks(config: ScrapeConfig) -> list[dict[str, str]]:
    session = build_session()
    all_rows: list[dict[str, str]] = []

    for page in range(1, config.pages + 1):
        url = page_url(page, config.page_size)
        rows: list[dict[str, str]] = []
        for parse_attempt in range(1, config.parse_retries + 1):
            scraped_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            attempt_note = "" if parse_attempt == 1 else f" (retry {parse_attempt}/{config.parse_retries})"
            print(f"Fetching page {page}/{config.pages}{attempt_note}: {url}", flush=True)
            html = fetch_page(session, url)
            rows = parse_blocks_from_html(html, url, page, scraped_at_utc)
            print(f"  parsed {len(rows)} rows", flush=True)
            if len(rows) >= config.page_size:
                break
            if parse_attempt < config.parse_retries:
                time.sleep(max(config.delay_seconds, 1.0))
        if len(rows) < config.page_size:
            print(
                f"Warning: page {page} parsed {len(rows)} rows after {config.parse_retries} attempts.",
                file=sys.stderr,
            )
        all_rows.extend(rows)
        if page < config.pages:
            time.sleep(config.delay_seconds)

    return all_rows


def write_raw_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def repair_existing_raw_csv(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    repaired: list[dict[str, str]] = []
    for row in rows:
        normalized = {field: clean_whitespace(row.get(field, "")) for field in RAW_FIELDNAMES}
        row_text = normalized.get("raw_row_text", "")
        metrics = extract_metrics_from_flat_text(row_text)
        if metrics:
            normalized.update(metrics)
        if not normalized["block_datetime_raw"]:
            normalized["block_datetime_raw"] = extract_datetime(row_text)
        if not normalized["age_raw"]:
            normalized["age_raw"] = extract_age(row_text)
        if not normalized["block_timestamp_unix_raw"]:
            normalized["block_timestamp_unix_raw"] = extract_unix_timestamp(row_text)
        if normalized["parse_method"] == "html_table":
            normalized["parse_method"] = "html_table_repaired_from_row_text"
        repaired.append({field: normalized.get(field, "") for field in RAW_FIELDNAMES})
    return repaired


def parse_args(argv: list[str]) -> ScrapeConfig:
    parser = argparse.ArgumentParser(description="Scrape recent Ethereum block metadata from Etherscan.")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="Number of paginated block pages to scrape.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Rows per Etherscan page.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between page requests.")
    parser.add_argument("--parse-retries", type=int, default=4, help="Retry a page if fewer rows than expected parse.")
    parser.add_argument(
        "--repair-existing",
        help="Repair an existing raw CSV by extracting metric fields from raw_row_text instead of scraping.",
    )
    parser.add_argument("--output", default="raw_blocks.csv", help="Raw CSV output path.")
    args = parser.parse_args(argv)

    if args.pages <= 0:
        parser.error("--pages must be positive")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.parse_retries <= 0:
        parser.error("--parse-retries must be positive")

    return ScrapeConfig(
        pages=args.pages,
        page_size=args.page_size,
        delay_seconds=args.delay,
        parse_retries=args.parse_retries,
        output_path=Path(args.output),
        repair_existing_path=Path(args.repair_existing) if args.repair_existing else None,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv or sys.argv[1:])
    if config.repair_existing_path:
        rows = repair_existing_raw_csv(config.repair_existing_path)
        write_raw_csv(rows, config.output_path)
        print(f"Repaired {len(rows)} raw rows from {config.repair_existing_path} into {config.output_path}.")
        return 0

    rows = scrape_blocks(config)
    write_raw_csv(rows, config.output_path)
    unique_blocks = {row["block_number_raw"] for row in rows if row.get("block_number_raw")}
    print(
        f"Saved {len(rows)} raw rows and {len(unique_blocks)} unique block numbers to {config.output_path}",
        flush=True,
    )
    if len(rows) < config.pages * config.page_size:
        print(
            "Warning: fewer rows were parsed than requested. Review parse_method and raw_row_text fields.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
