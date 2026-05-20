#!/usr/bin/env python3
"""Sentence-unit metadata analysis for the gisa-jidan marker hada."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "sentence_normalized.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "hada_metadata_stats.json"
DEFAULT_LOG = REPO_ROOT / "logs" / "hada_metadata_analysis.jsonl"

GENRE_ORDER = ["歷史書", "文集", "經傳", "詩", "기타"]
ZZTJGM_PREFIX = "자치통감강목"
MAX_CURRENT_ZZTJGM_VOLUME = 7


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_genre(book_name: Any) -> str:
    """Classify books with the same buckets used in the prior hada paragraph."""
    if pd.isna(book_name):
        return "기타"

    book = str(book_name)
    if "자치통감강목" in book or "춘추좌씨전" in book:
        return "歷史書"
    if "당송팔대가문초" in book:
        return "文集"
    if any(key in book for key in ["예기", "논어", "맹자", "대학장구", "중용"]):
        return "經傳"
    if "시경집전" in book or "당시삼백수" in book:
        return "詩"
    return "기타"


def percent(numerator: int, denominator: int, digits: int = 4) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, digits)


def p_value_label(p_value: float) -> str:
    if p_value == 0.0 or p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.3g}"


def cramers_v(chi_square: float, n: int, rows: int = 2, columns: int = 2) -> float:
    denominator = n * min(rows - 1, columns - 1)
    if denominator <= 0:
        return 0.0
    return math.sqrt(chi_square / denominator)


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_out_of_scope_book(book: str) -> bool:
    if not book.startswith(ZZTJGM_PREFIX):
        return False
    suffix = book.removeprefix(ZZTJGM_PREFIX)
    return suffix.isdigit() and int(suffix) > MAX_CURRENT_ZZTJGM_VOLUME


def find_out_of_scope_books(df: pd.DataFrame) -> dict[str, int]:
    books = df["book"].dropna().astype(str)
    out_of_scope = books[books.map(is_out_of_scope_book)]
    return {book: int(count) for book, count in sorted(out_of_scope.value_counts().items())}


def compute_hada_metadata(input_path: Path, allow_out_of_scope: bool = False) -> dict[str, Any]:
    df = pd.read_csv(input_path, encoding="utf-8")
    required = {"book", "marker_normalized"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out_of_scope_books = find_out_of_scope_books(df)
    if out_of_scope_books and not allow_out_of_scope:
        details = ", ".join(f"{book}={count}" for book, count in out_of_scope_books.items())
        raise ValueError(
            "Out-of-scope legacy volumes are present in the input. "
            f"Run on the current sentence_normalized.csv or pass --allow-out-of-scope. {details}"
        )

    work = df.copy()
    work["genre"] = work["book"].map(classify_genre)
    work["is_hada"] = work["marker_normalized"].astype(str).str.endswith("하다")

    by_genre: list[dict[str, Any]] = []
    for genre in GENRE_ORDER:
        genre_df = work[work["genre"] == genre]
        records = int(len(genre_df))
        hada_count = int(genre_df["is_hada"].sum())
        by_genre.append(
            {
                "genre": genre,
                "records": records,
                "hada_count": hada_count,
                "non_hada_count": records - hada_count,
                "hada_percent": percent(hada_count, records),
            }
        )

    history = work[work["genre"] == "歷史書"]
    non_history = work[work["genre"] != "歷史書"]
    table = np.array(
        [
            [
                int(history["is_hada"].sum()),
                int(len(history) - history["is_hada"].sum()),
            ],
            [
                int(non_history["is_hada"].sum()),
                int(len(non_history) - non_history["is_hada"].sum()),
            ],
        ]
    )

    chi2_yates, p_yates, dof, expected = stats.chi2_contingency(table, correction=True)
    chi2_uncorrected, p_uncorrected, _, _ = stats.chi2_contingency(
        table, correction=False
    )

    by_genre_map = {row["genre"]: row for row in by_genre}
    history_rate = by_genre_map["歷史書"]["hada_percent"]
    classics_rate = by_genre_map["經傳"]["hada_percent"]
    history_to_classics = (
        round(history_rate / classics_rate, 4) if classics_rate else None
    )

    hada_variants = (
        work.loc[work["is_hada"], "marker_normalized"]
        .astype(str)
        .value_counts()
        .head(25)
        .to_dict()
    )

    total_records = int(len(work))
    total_hada = int(work["is_hada"].sum())
    result = {
        "analysis": "hada_metadata_by_genre_sentence",
        "script_version": "2026-05-20-sentence-only",
        "source": {
            "path": repo_relative(input_path),
            "sha256": sha256_file(input_path),
            "records": total_records,
        },
        "method": {
            "input_unit": "sentence",
            "hada_rule": "marker_normalized.endswith('하다')",
            "input_scope_rule": "use the current sentence corpus scope",
            "out_of_scope_books_present": out_of_scope_books,
            "genre_rule": {
                "歷史書": ["자치통감강목", "춘추좌씨전"],
                "文集": ["당송팔대가문초"],
                "經傳": ["예기", "논어", "맹자", "대학장구", "중용"],
                "詩": ["시경집전", "당시삼백수"],
                "기타": "주역전의, 서경집전, and unclassified books",
            },
        },
        "totals": {
            "records": total_records,
            "hada_count": total_hada,
            "non_hada_count": total_records - total_hada,
        },
        "by_genre": by_genre,
        "history_vs_non_history": {
            "table": {
                "rows": ["歷史書", "非歷史書"],
                "columns": ["hada", "non_hada"],
                "values": table.astype(int).tolist(),
            },
            "chi_square_yates": round(float(chi2_yates), 6),
            "p_value_yates": float(p_yates),
            "p_value_yates_label": p_value_label(float(p_yates)),
            "degrees_of_freedom": int(dof),
            "expected": [[round(float(value), 6) for value in row] for row in expected],
            "cramers_v_yates": round(cramers_v(float(chi2_yates), total_records), 6),
            "chi_square_uncorrected": round(float(chi2_uncorrected), 6),
            "p_value_uncorrected": float(p_uncorrected),
            "p_value_uncorrected_label": p_value_label(float(p_uncorrected)),
        },
        "rate_ratios": {
            "history_to_classics": history_to_classics,
        },
        "hada_marker_variants_top25": {
            str(marker): int(count) for marker, count in hada_variants.items()
        },
        "report_text_ko": (
            "記史之斷(‘하다’)은 메타데이터 분석만으로 文體 편중이 확인된다. "
            f"현재 sentence 데이터셋에서 ‘하다’의 출현 "
            f"{total_hada:,}건의 서종별 출현율은 "
            f"歷史書 {by_genre_map['歷史書']['hada_percent']:.2f}%, "
            f"文集 {by_genre_map['文集']['hada_percent']:.2f}%, "
            f"經傳 {by_genre_map['經傳']['hada_percent']:.2f}%, "
            f"詩 {by_genre_map['詩']['hada_percent']:.2f}%, "
            f"기타 {by_genre_map['기타']['hada_percent']:.2f}%이며, "
            f"歷史書 대 經傳의 격차는 약 {history_to_classics:.0f}배이다. "
            f"歷史書 대 非歷史書 검정 결과는 χ² = {chi2_yates:.2f}, "
            f"p {p_value_label(float(p_yates))}이다."
        ),
    }
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sentence-unit hada metadata by genre."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--allow-out-of-scope",
        "--allow-contaminated",
        action="store_true",
        help="Allow legacy out-of-scope volumes in the input for historical comparison.",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Dry-run: compare computed stats with the existing output JSON.",
    )
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    log_path = args.log.resolve()

    result = compute_hada_metadata(
        input_path=input_path,
        allow_out_of_scope=args.allow_out_of_scope,
    )
    result_hash = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()

    event: dict[str, Any] = {
        "event": "hada_metadata_analysis",
        "input": repo_relative(input_path),
        "output": repo_relative(output_path),
        "result_hash": result_hash,
        "records": result["totals"]["records"],
        "hada_count": result["totals"]["hada_count"],
        "mode": "check-existing" if args.check_existing else "write",
    }

    if args.check_existing:
        if not output_path.exists():
            raise FileNotFoundError(f"Existing output not found: {output_path}")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_hash = hashlib.sha256(
            canonical_json(existing).encode("utf-8")
        ).hexdigest()
        event["existing_hash"] = existing_hash
        event["matched_existing"] = existing == result
        if not args.no_log:
            append_log(log_path, event)
        if existing != result:
            print("FAIL hada metadata stats differ from existing output")
            print(f"computed_hash={result_hash}")
            print(f"existing_hash={existing_hash}")
            return 1
        print("PASS hada metadata stats match existing output")
    else:
        write_json(output_path, result)
        if not args.no_log:
            append_log(log_path, event)
        print(f"Wrote {repo_relative(output_path)}")

    print(result["report_text_ko"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
