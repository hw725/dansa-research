#!/usr/bin/env python3
"""Verify active cleaned section2 result files and canonical stats."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MODELS = ("gpt5mini", "gemini", "claude_sonnet")
KEY_COLS = ("book", "문단식별자", "문장식별자", "marker_type")


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(col, "") for col in KEY_COLS)


def verify_model(model: str) -> None:
    base = read_rows(RESULTS / model / "section2_decision_judgments.csv")
    supp_path = RESULTS / model / "supplement_section2_judgments.csv"
    supplement = read_rows(supp_path) if supp_path.exists() else []
    rows = {key(row): row for row in [*base, *supplement]}.values()
    counts = Counter(row["marker_type"] for row in rows)
    ok = counts["니라"] == 11135 and counts["라"] == 11135
    status = "PASS" if ok else "FAIL"
    print(f"{model}: 니라={counts['니라']:,}, 라={counts['라']:,} [{status}]")


def main() -> int:
    print("=== section2 CSV verification ===")
    for model in MODELS:
        verify_model(model)

    stats_path = RESULTS / "final_stats_v3.1_cleaned_balanced.json"
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    l2 = stats["sections"]["section2"]
    print("\n=== Canonical section2 consensus ===")
    print(json.dumps(l2["consensus"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
