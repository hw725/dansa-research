#!/usr/bin/env python3
"""
Sentence-level marker pattern report.

This script summarizes comma-separated marker sequences from
`data/sentence_normalized.csv` and writes a report to
`reports/sentence_patterns.md`.
"""

from collections import Counter
from pathlib import Path
import json
import pandas as pd

DATA_PATH = Path("data/sentence_normalized.csv")
REPORT_PATH = Path("reports/sentence_patterns.md")
JSON_PATH = Path("results/sentence_patterns.json")
TOP_N = 100


def parse_marker_list(value):
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    return parts


def main():
    if not DATA_PATH.exists():
        print(f"Missing file: {DATA_PATH}")
        return 1

    df = pd.read_csv(DATA_PATH)
    if 'marker_normalized' in df.columns:
        marker_col = 'marker_normalized'
    elif 'marker' in df.columns:
        marker_col = 'marker'
    else:
        print("No marker column found.")
        return 1

    patterns = Counter()
    markers = Counter()
    total_rows = len(df)
    with_markers = 0
    single_marker = 0
    multi_marker = 0

    for value in df[marker_col]:
        marker_list = parse_marker_list(value)
        if not marker_list:
            continue
        with_markers += 1
        if len(marker_list) == 1:
            single_marker += 1
        else:
            multi_marker += 1
        pattern_key = ",".join(marker_list)
        patterns[pattern_key] += 1
        markers.update(marker_list)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Sentence-level marker patterns\n")
    lines.append("\n")
    lines.append(f"Source: {DATA_PATH.as_posix()}\n")
    lines.append("\n")
    lines.append("## Summary\n")
    lines.append("\n")
    lines.append(f"- Total sentences: {total_rows:,}\n")
    lines.append(f"- Sentences with markers: {with_markers:,}\n")
    lines.append(f"- Single-marker sentences: {single_marker:,}\n")
    lines.append(f"- Multi-marker sentences: {multi_marker:,}\n")
    lines.append(f"- Unique patterns: {len(patterns):,}\n")
    lines.append("\n")

    lines.append(f"## Top {TOP_N} sentence patterns\n")
    lines.append("\n")
    lines.append("| Rank | Pattern | Count |\n")
    lines.append("|------|---------|-------|\n")
    for idx, (pattern, count) in enumerate(patterns.most_common(TOP_N), start=1):
        lines.append(f"| {idx} | `{pattern}` | {count:,} |\n")
    lines.append("\n")

    lines.append(f"## Top {TOP_N} markers\n")
    lines.append("\n")
    lines.append("| Rank | Marker | Count |\n")
    lines.append("|------|--------|-------|\n")
    for idx, (marker, count) in enumerate(markers.most_common(TOP_N), start=1):
        lines.append(f"| {idx} | `{marker}` | {count:,} |\n")
    lines.append("\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    json_payload = {
        "source": DATA_PATH.as_posix(),
        "total_sentences": total_rows,
        "sentences_with_markers": with_markers,
        "single_marker_sentences": single_marker,
        "multi_marker_sentences": multi_marker,
        "unique_patterns": len(patterns),
        "top_patterns": patterns.most_common(TOP_N),
        "top_markers": markers.most_common(TOP_N),
    }
    JSON_PATH.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written: {REPORT_PATH}")
    print(f"JSON written: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
