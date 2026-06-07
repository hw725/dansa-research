#!/usr/bin/env python3
"""번역문 컬럼 익명화 - SHA-256 16자 해시.

비공개 raw CSV에 포함된 번역문(한국고전번역원 DB 미공개 자료)을 해시로 치환해
공개 가능한 _anon.csv 짝꿍을 생성한다. 원문(한문)은 공개 도메인이므로 보존.

대상:
- data/sentence_normalized.csv (공개용 전체 입력 데이터셋)
- results/beomnon_no_heosa.csv
- results/{gpt5mini,gemini,claude_sonnet}/{section*,supplement_section*}_judgments.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
TARGET_COLUMN = "번역문"
ANON_SUFFIX = "_anon"

# data/ 입력 중 공개용 anon 생성 대상 (원문 보존, 번역문 해시)
DATA_INPUTS = ("sentence_normalized.csv",)


def anon_hash(value: str | None) -> str | None:
    if value is None or value == "":
        return value
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def anonymize_csv(in_path: Path, out_path: Path) -> tuple[int, int]:
    with open(in_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if TARGET_COLUMN not in fieldnames:
            rows = sum(1 for _ in reader)
            print(f"  [skip] '{TARGET_COLUMN}' 컬럼 없음: {in_path.name}")
            return 0, rows
        rows = list(reader)

    for row in rows:
        row[TARGET_COLUMN] = anon_hash(row.get(TARGET_COLUMN))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    return n_total, n_total


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    for name in DATA_INPUTS:
        p = REPO / "data" / name
        if p.exists():
            targets.append(p)
    one = REPO / "results" / "beomnon_no_heosa.csv"
    if one.exists():
        targets.append(one)
    for model_dir in ("gpt5mini", "gemini", "claude_sonnet"):
        d = REPO / "results" / model_dir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            if p.name.endswith(f"{ANON_SUFFIX}.csv"):
                continue
            targets.append(p)
    return targets


def main() -> int:
    targets = collect_targets()
    if not targets:
        print("대상 없음")
        return 0

    print(f"익명화 대상: {len(targets)}개")
    total_rows = 0
    for src in targets:
        out = src.with_name(src.stem + ANON_SUFFIX + src.suffix)
        rows, _ = anonymize_csv(src, out)
        total_rows += rows
        rel = src.relative_to(REPO)
        rel_out = out.relative_to(REPO)
        print(f"  {rel} -> {rel_out} ({rows:,}행)")
    print(f"합계: {total_rows:,}행 익명화")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
