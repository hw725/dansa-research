#!/usr/bin/env python3
"""hyeonto XML 서종/구조 점검

목적:
- 문단/문장 수가 이상할 때, XML 구조가 '특수 사례'인지 빠르게 확인
- hyeonto 데이터는 대체로 <단락 id="..."> 기반이지만, 책마다 혼합되어 있을 수 있어 체크

출력:
- book별로 단락/문장/식별자 사용 여부 요약

사용 예:
  C:/Users/junto/Downloads/head-repo/hw725/CSP/.venv/Scripts/python.exe scripts/hyeonto_check_book_structure.py
  C:/Users/junto/Downloads/head-repo/hw725/CSP/.venv/Scripts/python.exe scripts/hyeonto_check_book_structure.py --include-all
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
HYEONTO_DIR = WORKSPACE_ROOT / "hyeonto"

DEFAULT_BOOKS = ["논어집주", "맹자집주", "대학장구", "중용장구"]


def _extract_book_name(filename: str) -> str | None:
    m = re.search(r"\[(?:역주|현토)\](.+?)_(?:원문|번역문)", filename)
    return m.group(1) if m else None


def _iter_pairs(hyeonto_dir: Path) -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    for src in sorted(hyeonto_dir.glob("*_원문_*.xml")):
        tgt = src.parent / src.name.replace("원문", "번역문")
        if tgt.exists():
            out.append((src, tgt))
    return out


def _analyze_one(xml_path: Path) -> dict[str, object]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    danrak = root.findall(".//단락")
    danrak_ids = [d.get("id", "") for d in danrak if d.get("id")]

    # '식별자'가 붙은 상위 노드(예: 원문 식별자=ID:W1 등)
    ident_nodes = [e for e in root.iter() if isinstance(e.attrib, dict) and ("식별자" in e.attrib)]
    ident_vals = [e.get("식별자", "") for e in ident_nodes if e.get("식별자")]

    s_nodes = root.findall(".//s")
    s_ids = [s.get("id", "") for s in s_nodes if s.get("id")]

    # 간단 이상 징후
    suspicious: list[str] = []
    if len(danrak) == 0:
        suspicious.append("단락(<단락>) 없음")
    if len(set(danrak_ids)) != len(danrak_ids):
        suspicious.append("단락 id 중복")
    if len(s_nodes) == 0:
        suspicious.append("문장(<s>) 없음")
    if len(set(s_ids)) != len(s_ids):
        suspicious.append("문장 id 중복")

    return {
        "path": str(xml_path),
        "root_lang": root.get("lang", ""),
        "danrak_count": len(danrak),
        "danrak_unique": len(set(danrak_ids)),
        "ident_count": len(ident_nodes),
        "ident_unique": len(set(ident_vals)),
        "s_count": len(s_nodes),
        "s_unique": len(set(s_ids)),
        "suspicious": suspicious,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check hyeonto XML structure per book")
    ap.add_argument("--include-all", action="store_true")
    ap.add_argument("--books", type=str, default=",".join(DEFAULT_BOOKS))
    args = ap.parse_args()

    if not HYEONTO_DIR.exists():
        raise SystemExit(f"hyeonto 폴더가 없습니다: {HYEONTO_DIR}")

    if bool(args.include_all):
        allow = None
    else:
        allow = {b.strip() for b in str(args.books).split(",") if b.strip()}

    pairs = _iter_pairs(HYEONTO_DIR)
    if not pairs:
        raise SystemExit("원문/번역문 XML 쌍을 찾지 못했습니다.")

    rows = []
    for src, tgt in pairs:
        book = _extract_book_name(src.name)
        if not book:
            continue
        if allow is not None and book not in allow:
            continue

        a_src = _analyze_one(src)
        a_tgt = _analyze_one(tgt)
        rows.append((book, a_src, a_tgt))

    if not rows:
        raise SystemExit("선택된 책이 없습니다. --include-all 또는 --books 확인")

    print("=" * 80)
    print("hyeonto XML 구조 점검")
    print("=" * 80)

    suspicious_books = []
    for book, a_src, a_tgt in rows:
        print(f"\n[{book}]")
        print(f"- 원문 lang={a_src['root_lang']} 단락={a_src['danrak_count']} (uniq {a_src['danrak_unique']}) 문장={a_src['s_count']} (uniq {a_src['s_unique']}) 식별자노드={a_src['ident_count']} (uniq {a_src['ident_unique']})")
        print(f"- 번역 lang={a_tgt['root_lang']} 단락={a_tgt['danrak_count']} (uniq {a_tgt['danrak_unique']}) 문장={a_tgt['s_count']} (uniq {a_tgt['s_unique']}) 식별자노드={a_tgt['ident_count']} (uniq {a_tgt['ident_unique']})")

        sus = list(a_src.get("suspicious", [])) + list(a_tgt.get("suspicious", []))
        sus = sorted(set([s for s in sus if s]))
        if sus:
            suspicious_books.append(book)
            print(f"- ⚠ suspicious: {', '.join(sus)}")

    if suspicious_books:
        print("\n=" * 40)
        print("⚠ 이상 징후가 있는 책:")
        for b in suspicious_books:
            print(f"- {b}")
    else:
        print("\n✅ 이상 징후 없음")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
