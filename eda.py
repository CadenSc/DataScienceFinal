#!/usr/bin/env python3
"""Stage 1 exploratory data analysis for Ethereum block data.

The script creates lightweight SVG visualizations and a short interpretation
file under eda_outputs/. It does not require pandas, matplotlib, or seaborn.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ENGINEERED_PATH = Path("engineered_blocks.csv")
CLEANED_PATH = Path("cleaned_blocks.csv")
RAW_PATH = Path("raw_blocks.csv")
OUTPUT_DIR = Path("eda_outputs")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a", "na", "-"}


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [value for row in rows if (value := to_float(row.get(column))) is not None]


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def safe_label(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:20px;font-weight:700}",
        ".axis{stroke:#475569;stroke-width:1}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        ".small{font-size:12px}",
        ".label{font-size:13px;font-weight:600}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="32">{safe_label(title)}</text>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def scale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    if old_max == old_min:
        return (new_min + new_max) / 2
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


def draw_histogram(values: list[float], title: str, xlabel: str, output_path: Path, bins: int = 30) -> None:
    width, height = 920, 560
    margin_left, margin_right, margin_top, margin_bottom = 72, 32, 68, 72
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    lines = svg_start(width, height, title)

    if not values:
        lines.append('<text x="72" y="120">No numeric values available.</text>')
        write_svg(output_path, lines)
        return

    low, high = min(values), max(values)
    if low == high:
        low -= 0.5
        high += 0.5
    bin_width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, int((value - low) / bin_width))
        counts[idx] += 1

    max_count = max(counts) or 1
    x0, y0 = margin_left, height - margin_bottom
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0 + chart_width}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{margin_top}"/>')

    for i, count in enumerate(counts):
        bar_x = x0 + i * chart_width / bins
        bar_w = chart_width / bins - 2
        bar_h = count / max_count * chart_height
        bar_y = y0 - bar_h
        lines.append(f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2f80ed" opacity="0.82"/>')

    for tick in range(5):
        pct = tick / 4
        y = y0 - pct * chart_height
        count_label = round(pct * max_count)
        lines.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + chart_width}" y2="{y:.2f}"/>')
        lines.append(f'<text class="small" x="30" y="{y + 4:.2f}">{count_label}</text>')

    lines.append(f'<text class="label" x="{x0 + chart_width / 2 - 70:.2f}" y="{height - 24}">{safe_label(xlabel)}</text>')
    lines.append(f'<text class="small" x="{x0}" y="{height - 48}">{low:.4g}</text>')
    lines.append(f'<text class="small" x="{x0 + chart_width - 70}" y="{height - 48}">{high:.4g}</text>')
    lines.append(f'<text class="label" x="24" y="{margin_top - 18}">Row count</text>')
    write_svg(output_path, lines)


def downsample(points: list[tuple[float, float]], max_points: int = 700) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    return points[::stride]


def draw_line_chart(
    series: list[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
    output_path: Path,
    normalize: bool = False,
) -> None:
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 78, 36, 76, 78
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    lines = svg_start(width, height, title)

    all_points: list[tuple[float, float]] = []
    processed: list[tuple[str, list[tuple[float, float]], str]] = []
    for label, values, color in series:
        points = [(float(index), value) for index, value in enumerate(values) if value is not None]
        if normalize and points:
            ys = [value for _, value in points]
            low, high = min(ys), max(ys)
            points = [(x, scale(y, low, high, 0, 1)) if high != low else (x, 0.5) for x, y in points]
        processed.append((label, points, color))
        all_points.extend(points)

    if not all_points:
        lines.append('<text x="78" y="130">No numeric values available.</text>')
        write_svg(output_path, lines)
        return

    x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_min, y_max = min(y for _, y in all_points), max(y for _, y in all_points)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    x0, y0 = margin_left, height - margin_bottom
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0 + chart_width}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{margin_top}"/>')

    for tick in range(5):
        pct = tick / 4
        y = y0 - pct * chart_height
        label = y_min + pct * (y_max - y_min)
        lines.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + chart_width}" y2="{y:.2f}"/>')
        lines.append(f'<text class="small" x="18" y="{y + 4:.2f}">{label:.4g}</text>')

    for label, points, color in processed:
        if len(points) < 2:
            continue
        sampled = downsample(points)
        svg_points = []
        for x, y in sampled:
            sx = scale(x, x_min, x_max, x0, x0 + chart_width)
            sy = scale(y, y_min, y_max, y0, margin_top)
            svg_points.append(f"{sx:.2f},{sy:.2f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(svg_points)}"/>')

    legend_x = x0 + 12
    legend_y = margin_top - 22
    for idx, (label, _, color) in enumerate(processed):
        offset = idx * 170
        lines.append(f'<rect x="{legend_x + offset}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text class="small" x="{legend_x + offset + 20}" y="{legend_y + 2}">{safe_label(label)}</text>')

    lines.append(f'<text class="label" x="{x0 + chart_width / 2 - 88:.2f}" y="{height - 28}">Chronological block order</text>')
    lines.append(f'<text class="label" x="20" y="{margin_top - 18}">{safe_label(ylabel)}</text>')
    write_svg(output_path, lines)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return None if denominator == 0 else numerator / denominator


def color_for_correlation(value: float | None) -> str:
    if value is None:
        return "#f1f5f9"
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        intensity = int(255 - value * 145)
        return f"rgb(255,{intensity},{intensity})"
    intensity = int(255 + value * 145)
    return f"rgb({intensity},{intensity},255)"


def draw_correlation_heatmap(rows: list[dict[str, Any]], columns: list[str], output_path: Path) -> list[tuple[str, str, float]]:
    width, height = 980, 760
    margin_left, margin_top = 220, 180
    cell = 42
    lines = svg_start(width, height, "Correlation Heatmap For Block Activity Features")
    correlations: dict[tuple[str, str], float | None] = {}
    pairs_ranked: list[tuple[str, str, float]] = []

    for col_a in columns:
        for col_b in columns:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = to_float(row.get(col_a))
                y = to_float(row.get(col_b))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            corr = pearson(xs, ys)
            correlations[(col_a, col_b)] = corr
            if col_a < col_b and corr is not None:
                pairs_ranked.append((col_a, col_b, corr))

    for i, col in enumerate(columns):
        x = margin_left + i * cell + cell / 2
        lines.append(f'<text class="small" transform="translate({x:.2f},{margin_top - 12}) rotate(-55)" text-anchor="start">{safe_label(col)}</text>')
        y = margin_top + i * cell + cell * 0.65
        lines.append(f'<text class="small" x="{margin_left - 10}" y="{y:.2f}" text-anchor="end">{safe_label(col)}</text>')

    for row_idx, col_a in enumerate(columns):
        for col_idx, col_b in enumerate(columns):
            corr = correlations[(col_a, col_b)]
            x = margin_left + col_idx * cell
            y = margin_top + row_idx * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color_for_correlation(corr)}" stroke="#ffffff"/>')
            label = "" if corr is None else f"{corr:.2f}"
            lines.append(f'<text class="small" x="{x + cell / 2:.2f}" y="{y + cell * 0.62:.2f}" text-anchor="middle">{label}</text>')

    lines.append('<text class="small" x="690" y="690">Blue = negative, red = positive, white = weak/zero</text>')
    write_svg(output_path, lines)
    pairs_ranked.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs_ranked


def draw_missingness_comparison(raw_rows: list[dict[str, str]], cleaned_rows: list[dict[str, str]], output_path: Path) -> dict[str, tuple[float, float]]:
    width, height = 980, 560
    margin_left, margin_top, margin_bottom = 82, 78, 120
    chart_width = width - margin_left - 40
    chart_height = height - margin_top - margin_bottom
    mappings = [
        ("block", "block_number_raw", "block_number"),
        ("txn", "txn_count_raw", "txn_count"),
        ("gas used", "gas_used_raw", "gas_used"),
        ("gas limit", "gas_limit_raw", "gas_limit"),
        ("base fee", "base_fee_raw", "base_fee_gwei"),
        ("reward", "reward_raw", "reward_eth"),
        ("burnt fees", "burnt_fees_raw", "burnt_fees_eth"),
    ]
    comparison: dict[str, tuple[float, float]] = {}
    for label, raw_col, cleaned_col in mappings:
        raw_pct = 0.0 if not raw_rows else 100 * sum(1 for row in raw_rows if is_missing(row.get(raw_col))) / len(raw_rows)
        clean_pct = 0.0 if not cleaned_rows else 100 * sum(1 for row in cleaned_rows if is_missing(row.get(cleaned_col))) / len(cleaned_rows)
        comparison[label] = (raw_pct, clean_pct)

    lines = svg_start(width, height, "Before/After Missingness Comparison")
    x0, y0 = margin_left, height - margin_bottom
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0 + chart_width}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{margin_top}"/>')
    group_width = chart_width / len(mappings)
    for idx, (label, _, _) in enumerate(mappings):
        raw_pct, clean_pct = comparison[label]
        x = x0 + idx * group_width + 16
        raw_h = raw_pct / 100 * chart_height
        clean_h = clean_pct / 100 * chart_height
        lines.append(f'<rect x="{x:.2f}" y="{y0 - raw_h:.2f}" width="22" height="{raw_h:.2f}" fill="#f2994a"/>')
        lines.append(f'<rect x="{x + 28:.2f}" y="{y0 - clean_h:.2f}" width="22" height="{clean_h:.2f}" fill="#27ae60"/>')
        lines.append(f'<text class="small" transform="translate({x + 28:.2f},{y0 + 18}) rotate(35)" text-anchor="start">{safe_label(label)}</text>')
    for tick in range(5):
        pct = tick / 4
        y = y0 - pct * chart_height
        lines.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + chart_width}" y2="{y:.2f}"/>')
        lines.append(f'<text class="small" x="30" y="{y + 4:.2f}">{pct * 100:.0f}%</text>')
    lines.append('<rect x="710" y="56" width="14" height="14" fill="#f2994a"/><text class="small" x="732" y="68">Raw missing %</text>')
    lines.append('<rect x="830" y="56" width="14" height="14" fill="#27ae60"/><text class="small" x="852" y="68">Cleaned missing %</text>')
    write_svg(output_path, lines)
    return comparison


def draw_spike_inspection(rows: list[dict[str, Any]], output_path: Path) -> None:
    congestion = [to_float(row.get("congestion_ratio")) for row in rows]
    high_flags = [to_float(row.get("congestion_flag_high")) for row in rows]
    spike_indices = [index for index, flag in enumerate(high_flags) if flag == 1]
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 78, 36, 76, 78
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    lines = svg_start(width, height, "High Congestion Blocks Over Time")
    points = [(float(index), value) for index, value in enumerate(congestion) if value is not None]
    if not points:
        lines.append('<text x="78" y="130">No congestion ratio values available.</text>')
        write_svg(output_path, lines)
        return

    x_min, x_max = points[0][0], points[-1][0]
    y_min, y_max = 0.0, max(1.0, max(value for _, value in points))
    x0, y0 = margin_left, height - margin_bottom
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0 + chart_width}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{margin_top}"/>')
    threshold_y = scale(0.80, y_min, y_max, y0, margin_top)
    lines.append(f'<line x1="{x0}" y1="{threshold_y:.2f}" x2="{x0 + chart_width}" y2="{threshold_y:.2f}" stroke="#d62728" stroke-dasharray="6 6"/>')
    lines.append(f'<text class="small" x="{x0 + 8}" y="{threshold_y - 8:.2f}">0.80 high congestion threshold</text>')

    svg_points = []
    for x, y in downsample(points):
        sx = scale(x, x_min, x_max, x0, x0 + chart_width)
        sy = scale(y, y_min, y_max, y0, margin_top)
        svg_points.append(f"{sx:.2f},{sy:.2f}")
    lines.append(f'<polyline fill="none" stroke="#2f80ed" stroke-width="2" points="{" ".join(svg_points)}"/>')

    for index in spike_indices[:: max(1, math.ceil(len(spike_indices) / 120))]:
        value = congestion[index]
        if value is None:
            continue
        sx = scale(index, x_min, x_max, x0, x0 + chart_width)
        sy = scale(value, y_min, y_max, y0, margin_top)
        lines.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="#d62728" opacity="0.85"/>')

    lines.append(f'<text class="label" x="{x0 + chart_width / 2 - 88:.2f}" y="{height - 28}">Chronological block order</text>')
    lines.append('<text class="label" x="20" y="58">Congestion ratio</text>')
    write_svg(output_path, lines)


def write_interpretations(
    rows: list[dict[str, Any]],
    output_dir: Path,
    strongest_correlations: list[tuple[str, str, float]],
    missingness: dict[str, tuple[float, float]] | None,
) -> None:
    txn_stats = describe(numeric_values(rows, "txn_count"))
    gas_stats = describe(numeric_values(rows, "gas_used"))
    base_fee_stats = describe(numeric_values(rows, "base_fee_gwei"))
    congestion_stats = describe(numeric_values(rows, "congestion_ratio"))
    high_count = sum(1 for row in rows if to_float(row.get("congestion_flag_high")) == 1)
    spike_count = sum(1 for row in rows if to_float(row.get("base_fee_spike_flag")) == 1)
    total = len(rows)

    correlation_text = "No correlation pairs were available."
    if strongest_correlations:
        top = strongest_correlations[:5]
        correlation_text = "; ".join(f"`{a}` vs `{b}` = {corr:.2f}" for a, b, corr in top)

    missing_text = "Before/after missingness visual was not generated because raw or cleaned CSV was unavailable."
    if missingness:
        improved = [
            f"{label}: raw {raw_pct:.1f}% to cleaned {clean_pct:.1f}%"
            for label, (raw_pct, clean_pct) in missingness.items()
        ]
        missing_text = "; ".join(improved)

    markdown = f"""# EDA Interpretations

Dataset analyzed: `{ENGINEERED_PATH}` with {total} engineered block rows.

## Visuals Created

- `txn_count_distribution.svg`: transaction counts vary by block, with median {txn_stats['median']:.2f} and range {txn_stats['min']:.2f} to {txn_stats['max']:.2f}. This supports the proposal's focus on changing block activity over time.
- `gas_used_distribution.svg`: gas used ranges from {gas_stats['min']:.2f} to {gas_stats['max']:.2f}; high values are retained because they are part of congestion behavior.
- `base_fee_trend.svg`: base fee is plotted in chronological block order, with observed range {base_fee_stats['min']:.4g} to {base_fee_stats['max']:.4g} Gwei.
- `congestion_ratio_trend.svg`: congestion ratio has median {congestion_stats['median']:.4f}; {high_count} rows meet the high-congestion threshold of 0.80.
- `rolling_activity_trends.svg`: rolling 20-block transaction count, gas used, and base fee are normalized together to compare recent activity momentum.
- `correlation_heatmap.svg`: strongest observed numeric relationships include {correlation_text}. Correlation is descriptive only and should not be interpreted as model performance.
- `before_after_missingness.svg`: {missing_text}
- `spike_inspection.svg`: high-congestion blocks are highlighted for outlier/spike inspection while preserving those rows for analysis.

## Stage 1/2 Takeaway

The visual checks support a conservative Stage 1 and Stage 2 workflow: the dataset is block-level, recent, public, and suited to analyzing short-term network utilization. Cleaning improves consistency without removing meaningful congestion spikes. Engineered rolling windows give the Stage 3 teammate interpretable predictors for Logistic Regression and Random Forest experiments without claiming predictive success yet.
"""
    (output_dir / "eda_interpretations.md").write_text(markdown, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 1 EDA visualizations for Ethereum block data.")
    parser.add_argument("--input", default=str(ENGINEERED_PATH), help="Engineered CSV input path.")
    parser.add_argument("--cleaned", default=str(CLEANED_PATH), help="Cleaned CSV path for before/after comparison.")
    parser.add_argument("--raw", default=str(RAW_PATH), help="Raw CSV path for before/after comparison.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for EDA SVGs and interpretations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    cleaned_path = Path(args.cleaned)
    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run feature_engineering.py first.", file=sys.stderr)
        return 1

    rows = read_csv(input_path)
    draw_histogram(numeric_values(rows, "txn_count"), "Transaction Count Distribution", "Transactions per block", output_dir / "txn_count_distribution.svg")
    draw_histogram(numeric_values(rows, "gas_used"), "Gas Used Distribution", "Gas used per block", output_dir / "gas_used_distribution.svg")
    draw_line_chart(
        [("base_fee_gwei", [to_float(row.get("base_fee_gwei")) for row in rows], "#7b2cbf")],
        "Base Fee Trend Over Recent Blocks",
        "Base fee (Gwei)",
        output_dir / "base_fee_trend.svg",
    )
    draw_line_chart(
        [
            ("congestion_ratio", [to_float(row.get("congestion_ratio")) for row in rows], "#2f80ed"),
            ("rolling_avg_congestion_20", [to_float(row.get("rolling_avg_congestion_20")) for row in rows], "#d62728"),
        ],
        "Congestion Ratio Over Recent Blocks",
        "Gas used / gas limit",
        output_dir / "congestion_ratio_trend.svg",
    )
    draw_line_chart(
        [
            ("rolling_avg_txn_20", [to_float(row.get("rolling_avg_txn_20")) for row in rows], "#2f80ed"),
            ("rolling_avg_gas_used_20", [to_float(row.get("rolling_avg_gas_used_20")) for row in rows], "#27ae60"),
            ("rolling_avg_base_fee_20", [to_float(row.get("rolling_avg_base_fee_20")) for row in rows], "#f2994a"),
        ],
        "Rolling Recent Activity Trends",
        "Normalized rolling values",
        output_dir / "rolling_activity_trends.svg",
        normalize=True,
    )
    heatmap_columns = [
        "txn_count",
        "gas_used",
        "base_fee_gwei",
        "reward_eth",
        "burnt_fees_eth",
        "congestion_ratio",
        "rolling_avg_txn_20",
        "rolling_avg_gas_used_20",
        "rolling_avg_base_fee_20",
        "rolling_avg_congestion_20",
    ]
    strongest_correlations = draw_correlation_heatmap(rows, heatmap_columns, output_dir / "correlation_heatmap.svg")
    draw_spike_inspection(rows, output_dir / "spike_inspection.svg")

    missingness = None
    if raw_path.exists() and cleaned_path.exists():
        missingness = draw_missingness_comparison(read_csv(raw_path), read_csv(cleaned_path), output_dir / "before_after_missingness.svg")

    write_interpretations(rows, output_dir, strongest_correlations, missingness)
    print(f"Created EDA outputs in {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
