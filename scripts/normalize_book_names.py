#!/usr/bin/env python3
"""
서적명 정규화

book_name → book 변환 (표준화된 서적명)

사용법:
    python scripts/normalize_book_names.py \
        --csv hyeonto/datasets/reclustered_with_syntax.csv \
        --out-csv hyeonto/datasets/reclustered_final.csv
"""

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd


# 서적명 정규화 매핑 테이블
BOOK_NAME_MAPPING: Dict[str, str] = {
    # 사서 (Four Books)
    '논어집주': '논어집주',
    '논어': '논어집주',
    '맹자집주': '맹자집주',
    '맹자': '맹자집주',
    '대학장구': '대학장구',
    '대학': '대학장구',
    '중용장구': '중용장구',
    '중용': '중용장구',

    # 삼경 (Three Classics)
    '시경집전(상)': '시경',
    '시경집전(하)': '시경',
    '시경집전': '시경',
    '시경': '시경',

    '서경집전(상)': '서경',
    '서경집전(하)': '서경',
    '서경집전': '서경',
    '서경': '서경',

    '주역전의(상)': '역경',
    '주역전의(하)': '역경',
    '주역전의': '역경',
    '역경': '역경',
    '주역': '역경',

    # 오경 추가 (나머지)
    '춘추좌씨전1': '춘추좌전',
    '춘추좌씨전2': '춘추좌전',
    '춘추좌씨전3': '춘추좌전',
    '춘추좌씨전4': '춘추좌전',
    '춘추좌씨전': '춘추좌전',
    '춘추': '춘추좌전',

    '예기': '예기',

    # 사서삼경 외 중요 텍스트
    '소학': '소학',
    '근사록': '근사록',
    '가례': '가례',
    '심경': '심경',
    '성리대전': '성리대전',
    '주자대전': '주자대전',

    # 역사서
    '자치통감강목1': '자치통감강목',
    '자치통감강목2': '자치통감강목',
    '자치통감강목3': '자치통감강목',
    '자치통감강목4': '자치통감강목',
    '자치통감강목5': '자치통감강목',
    '자치통감강목': '자치통감강목',

    # 당송팔대가문초
    '당송팔대가문초유종원1': '당송팔대가문초',
    '당송팔대가문초유종원2': '당송팔대가문초',
    '당송팔대가문초유종원3': '당송팔대가문초',
    '당송팔대가문초증공1': '당송팔대가문초',
    '당송팔대가문초증공2': '당송팔대가문초',
    '당송팔대가문초왕안석1': '당송팔대가문초',
    '당송팔대가문초왕안석2': '당송팔대가문초',
    '당송팔대가문초구양수1': '당송팔대가문초',
    '당송팔대가문초구양수2': '당송팔대가문초',
    '당송팔대가문초구양수3': '당송팔대가문초',
    '당송팔대가문초구양수4': '당송팔대가문초',
    '당송팔대가문초구양수5': '당송팔대가문초',
    '당송팔대가문초소순1': '당송팔대가문초',
    '당송팔대가문초소순2': '당송팔대가문초',
    '당송팔대가문초소식1': '당송팔대가문초',
    '당송팔대가문초소식2': '당송팔대가문초',
    '당송팔대가문초소식3': '당송팔대가문초',
    '당송팔대가문초소철1': '당송팔대가문초',
    '당송팔대가문초한유1': '당송팔대가문초',
    '당송팔대가문초한유2': '당송팔대가문초',
    '당송팔대가문초한유3': '당송팔대가문초',

    # 문집/기타
    '열녀전': '열녀전',
    '동문선': '동문선',
    '동몽선습': '동몽선습',
    '명심보감': '명심보감',
}


def normalize_book_name(name: str, mapping: Dict[str, str]) -> str:
    """서적명 정규화

    Args:
        name: 원본 book_name
        mapping: 정규화 매핑 테이블

    Returns:
        정규화된 book 이름
    """
    # 정확히 매칭되는 경우
    if name in mapping:
        return mapping[name]

    # 패턴 매칭: '주역전의(상)' → '역경'
    # 괄호 제거
    name_without_parens = re.sub(r'\([^)]*\)', '', name).strip()
    if name_without_parens in mapping:
        return mapping[name_without_parens]

    # 숫자 제거: '자치통감강목3' → '자치통감강목'
    name_without_nums = re.sub(r'\d+$', '', name).strip()
    if name_without_nums in mapping:
        return mapping[name_without_nums]

    # 매칭 실패 시 원본 그대로 반환
    return name


def main() -> int:
    p = argparse.ArgumentParser(description="서적명 정규화")
    p.add_argument("--csv", type=Path, required=True, help="입력 CSV 파일")
    p.add_argument("--out-csv", type=Path, required=True, help="출력 CSV 파일")
    p.add_argument("--mapping", type=Path, help="커스텀 매핑 JSON (선택)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    # 출력 디렉토리 생성
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 커스텀 매핑 로드 (있으면)
    mapping = BOOK_NAME_MAPPING.copy()
    if args.mapping and args.mapping.exists():
        import json
        with open(args.mapping, 'r', encoding='utf-8') as f:
            custom_mapping = json.load(f)
        mapping.update(custom_mapping)
        print(f"✅ 커스텀 매핑 로드: {len(custom_mapping)}개")

    # CSV 로드
    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✅ {len(df):,}행 로드")

    # book_name 컬럼 확인
    if 'book_name' not in df.columns:
        print("❌ book_name 컬럼이 없습니다")
        return 1

    print(f"\n📊 정규화 전 통계:")
    print(f"   고유 book_name: {df['book_name'].nunique()}개")

    # book 컬럼 생성
    print("\n🔧 서적명 정규화 중...")
    df['book'] = df['book_name'].apply(lambda x: normalize_book_name(x, mapping))

    print(f"\n📊 정규화 후 통계:")
    print(f"   고유 book: {df['book'].nunique()}개")

    # 매핑 성공률
    unchanged_count = (df['book'] == df['book_name']).sum()
    changed_count = len(df) - unchanged_count
    print(f"   변경됨: {changed_count:,}개 ({changed_count / len(df) * 100:.1f}%)")
    print(f"   변경 안됨: {unchanged_count:,}개 ({unchanged_count / len(df) * 100:.1f}%)")

    # Top 10 book
    print("\n🔝 Top 10 Book (정규화 후):")
    for book, count in df['book'].value_counts().head(10).items():
        print(f"   {book}: {count:,}개 ({count / len(df) * 100:.1f}%)")

    # 사서 카테고리 통계
    saseo_books = ['논어집주', '맹자집주', '대학장구', '중용장구']
    saseo_count = df['book'].isin(saseo_books).sum()
    print(f"\n📚 사서 (Four Books): {saseo_count:,}개 ({saseo_count / len(df) * 100:.1f}%)")

    # 삼경 카테고리 통계
    samgyeong_books = ['시경', '서경', '역경']
    samgyeong_count = df['book'].isin(samgyeong_books).sum()
    print(f"📖 삼경 (Three Classics): {samgyeong_count:,}개 ({samgyeong_count / len(df) * 100:.1f}%)")

    # CSV 저장
    print(f"\n💾 CSV 저장: {args.out_csv}")
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {len(df):,}행")

    # 통계 파일 생성
    stats_path = args.out_csv.parent / f"{args.out_csv.stem}_book_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# 서적명 정규화 통계\n\n")
        f.write(f"입력 파일: {args.csv}\n")
        f.write(f"출력 파일: {args.out_csv}\n\n")
        f.write(f"총 행 수: {len(df):,}\n")
        f.write(f"고유 book_name (정규화 전): {df['book_name'].nunique()}개\n")
        f.write(f"고유 book (정규화 후): {df['book'].nunique()}개\n\n")
        f.write(f"변경됨: {changed_count:,}개 ({changed_count / len(df) * 100:.1f}%)\n")
        f.write(f"변경 안됨: {unchanged_count:,}개 ({unchanged_count / len(df) * 100:.1f}%)\n\n")
        f.write(f"사서 (Four Books): {saseo_count:,}개 ({saseo_count / len(df) * 100:.1f}%)\n")
        f.write(f"삼경 (Three Classics): {samgyeong_count:,}개 ({samgyeong_count / len(df) * 100:.1f}%)\n\n")
        f.write("## Top 20 Book (정규화 후):\n\n")
        for book, count in df['book'].value_counts().head(20).items():
            f.write(f"{book}: {count:,}개 ({count / len(df) * 100:.1f}%)\n")

    print(f"📄 통계 파일 저장: {stats_path}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
