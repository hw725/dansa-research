#!/usr/bin/env python3
"""hyeonto XML(현토) → xlsx(구/문장/문단 병렬) 생성

요구사항:
- 입력: hyeonto/*.xml (원문/번역문 쌍)
- 대상: 기본 4종(논어/맹자/대학/중용)만 처리 (옵션으로 확장 가능)
- 출력: hyeonto/xlsx/{book_name}/ 아래에 *_구병렬.xlsx, *_문장병렬.xlsx, *_문단병렬.xlsx 생성
- 기존 데이터(xlsx/, datasets/, reports/)와 완전히 분리

실행 예:
  python scripts/hyeonto_build_xlsx.py
  python scripts/hyeonto_build_xlsx.py --include-all
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# local imports (require workspace root in sys.path)
from xlsx_scripts.xml_to_tsv_converter import process_xml_pairs as build_gu
from xlsx_scripts.xml_to_sentence_parallel import process_xml_pairs as build_sentence
from xlsx_scripts.renumber_excel_indices import renumber_excel_files
from xlsx_scripts.create_paragraph_parallel import create_paragraph_parallel

HYEONTO_DIR = WORKSPACE_ROOT / "hyeonto"
OUT_XLSX_DIR = HYEONTO_DIR / "xlsx"

DEFAULT_BOOKS = ["논어집주", "맹자집주", "대학장구", "중용장구"]


def _iter_hyeonto_xml_pairs(hyeonto_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for src in sorted(hyeonto_dir.glob("*_원문_*.xml")):
        tgt = src.parent / src.name.replace("원문", "번역문")
        if tgt.exists():
            pairs.append((src, tgt))
    return pairs


def _extract_book_name_from_filename(name: str) -> str | None:
    # xlsx_scripts와 동일한 규칙(패치됨): [현토]책이름_원문/번역문
    import re

    m = re.search(r"\[(?:역주|현토)\](.+?)_(?:원문|번역문)", name)
    return m.group(1) if m else None


def _copy_selected_pairs(src_dir: Path, tmp_dir: Path, books: set[str]) -> int:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src, tgt in _iter_hyeonto_xml_pairs(src_dir):
        book = _extract_book_name_from_filename(src.name)
        if not book or book not in books:
            continue
        shutil.copy2(src, tmp_dir / src.name)
        shutil.copy2(tgt, tmp_dir / tgt.name)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Build hyeonto xlsx outputs under hyeonto/xlsx")
    ap.add_argument("--include-all", action="store_true", help="hyeonto 폴더의 모든 [현토] 서종 처리")
    ap.add_argument(
        "--books",
        type=str,
        default=",".join(DEFAULT_BOOKS),
        help="처리할 책 이름 목록(콤마). 기본: 논어/맹자/대학/중용",
    )
    ap.add_argument("--clean", action="store_true", help="출력(hyeonto/xlsx)을 비우고 다시 생성")
    args = ap.parse_args()

    if not HYEONTO_DIR.exists():
        raise SystemExit(f"hyeonto 폴더가 없습니다: {HYEONTO_DIR}")

    if bool(args.clean) and OUT_XLSX_DIR.exists():
        shutil.rmtree(OUT_XLSX_DIR)

    OUT_XLSX_DIR.mkdir(parents=True, exist_ok=True)

    if bool(args.include_all):
        # 직접 hyeonto를 source_dir로 사용
        source_dir = HYEONTO_DIR
    else:
        books = {b.strip() for b in str(args.books).split(",") if b.strip()}
        tmp_dir = HYEONTO_DIR / "_tmp_selected_sources"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        copied = _copy_selected_pairs(HYEONTO_DIR, tmp_dir, books=books)
        if copied <= 0:
            raise SystemExit(f"선택된 책의 XML 쌍을 찾지 못했습니다. books={sorted(books)}")
        source_dir = tmp_dir

    # 1) 구병렬
    build_gu(str(source_dir), str(OUT_XLSX_DIR))

    # 2) 문장병렬
    build_sentence(str(source_dir), str(OUT_XLSX_DIR))

    # 3) 문단식별자 정규화 (문장병렬 파일 대상)
    renumber_excel_files(str(OUT_XLSX_DIR))

    # 4) 문단병렬 생성
    create_paragraph_parallel(str(OUT_XLSX_DIR))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
