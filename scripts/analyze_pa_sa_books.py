#!/usr/bin/env python3
"""PA/SA 데이터셋의 서종 분석 - 사서/삼경/문집/역사서 포함 여부 확인"""
import pandas as pd

# PA 데이터 로드
pa = pd.read_csv('hyeonto/datasets/sentence_train_merged.csv')
print(f"PA 총 행수: {len(pa):,}")
print(f"PA 총 서종: {pa['book_name'].nunique()}개\n")

# 서종 목록
books = sorted(pa['book_name'].unique())

# 사서/삼경/문집 분류 (기존 normalize_book_names.py 참고)
사서 = []
삼경 = []
문집 = []
역사서 = []
기타 = []

for book in books:
    if '논어' in book or '맹자' in book or '대학' in book or '중용' in book:
        사서.append(book)
    elif '시경' in book or '서경' in book or '역경' in book:
        삼경.append(book)
    elif '동문선' in book or '팔대가문초' in book or '열녀전' in book:
        문집.append(book)
    elif '자치통감' in book or '십팔사략' in book or '사략' in book:
        역사서.append(book)
    else:
        기타.append(book)

print("=== 서종 분류 ===\n")
print(f"사서 ({len(사서)}개):")
for book in 사서:
    count = len(pa[pa['book_name'] == book])
    print(f"  - {book}: {count:,}행")

print(f"\n삼경 ({len(삼경)}개):")
for book in 삼경:
    count = len(pa[pa['book_name'] == book])
    print(f"  - {book}: {count:,}행")

print(f"\n문집 ({len(문집)}개):")
for book in 문집:
    count = len(pa[pa['book_name'] == book])
    print(f"  - {book}: {count:,}행")

print(f"\n역사서 ({len(역사서)}개):")
for book in 역사서:
    count = len(pa[pa['book_name'] == book])
    print(f"  - {book}: {count:,}행")

print(f"\n기타 ({len(기타)}개):")
for book in 기타:
    count = len(pa[pa['book_name'] == book])
    print(f"  - {book}: {count:,}행")

# 통계
print("\n=== 요약 ===")
사서_count = pa[pa['book_name'].isin(사서)].shape[0]
삼경_count = pa[pa['book_name'].isin(삼경)].shape[0]
문집_count = pa[pa['book_name'].isin(문집)].shape[0]
역사서_count = pa[pa['book_name'].isin(역사서)].shape[0]
기타_count = pa[pa['book_name'].isin(기타)].shape[0]

total = len(pa)
print(f"사서: {사서_count:,}행 ({사서_count/total*100:.1f}%)")
print(f"삼경: {삼경_count:,}행 ({삼경_count/total*100:.1f}%)")
print(f"문집: {문집_count:,}행 ({문집_count/total*100:.1f}%)")
print(f"역사서: {역사서_count:,}행 ({역사서_count/total*100:.1f}%)")
print(f"기타: {기타_count:,}행 ({기타_count/total*100:.1f}%)")
