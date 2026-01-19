#!/usr/bin/env python3
"""
SA의 구들을 합쳤을 때 PA의 문장과 텍스트가 일치하는지 검증
공백 정규화 포함
"""
import pandas as pd
import re

def normalize_text(text):
    """텍스트 정규화: 공백 제거"""
    if pd.isna(text):
        return ''
    text = str(text)
    # 모든 공백 제거
    text = re.sub(r'\s+', '', text)
    return text

print("PA와 SA 데이터 로딩...")
pa = pd.read_csv('hyeonto/datasets/sentence_train_merged.csv')
sa = pd.read_csv('hyeonto/datasets/phrase_train_merged.csv')

print(f"PA: {len(pa):,}행")
print(f"SA: {len(sa):,}행")

# 공통 문장만 추출
pa_sentences = pa[['book_name', '문장식별자']].drop_duplicates()
sa_sentences = sa[['book_name', '문장식별자']].drop_duplicates()

pa_set = set(zip(pa_sentences['book_name'], pa_sentences['문장식별자']))
sa_set = set(zip(sa_sentences['book_name'], sa_sentences['문장식별자']))

common_set = pa_set & sa_set

print(f"\n공통 문장: {len(common_set):,}개")

# 샘플 검증
print("\n=== 샘플 검증 (처음 10개 문장) ===\n")

mismatch_count = 0
match_count = 0
sample_count = 0

for book, sent_id in sorted(list(common_set))[:100]:  # 처음 100개만 체크
    # PA에서 해당 문장
    pa_row = pa[(pa['book_name'] == book) & (pa['문장식별자'] == sent_id)]
    if len(pa_row) == 0:
        continue

    pa_text = pa_row.iloc[0]['원문']
    pa_normalized = normalize_text(pa_text)

    # SA에서 해당 문장의 모든 구
    sa_rows = sa[(sa['book_name'] == book) & (sa['문장식별자'] == sent_id)]
    if len(sa_rows) == 0:
        continue

    # SA 구들을 합치기
    sa_text = ''.join(sa_rows['원문'].astype(str).tolist())
    sa_normalized = normalize_text(sa_text)

    sample_count += 1

    if pa_normalized == sa_normalized:
        match_count += 1
        if sample_count <= 5:
            print(f"[OK] Match #{sample_count}")
            print(f"   Book: {book}, Sentence ID: {sent_id}")
            print(f"   PA: {pa_text[:60]}...")
            print(f"   SA: {sa_text[:60]}...")
            print()
    else:
        mismatch_count += 1
        if mismatch_count <= 5:
            print(f"[MISMATCH] #{mismatch_count}")
            print(f"   책: {book}, 문장ID: {sent_id}")
            print(f"   PA: {pa_text}")
            print(f"   SA: {sa_text}")
            print(f"   PA정규화: {pa_normalized}")
            print(f"   SA정규화: {sa_normalized}")
            print()

print(f"\n=== 검증 결과 (샘플 {sample_count}개) ===")
print(f"일치: {match_count}개 ({match_count/sample_count*100:.1f}%)")
print(f"불일치: {mismatch_count}개 ({mismatch_count/sample_count*100:.1f}%)")

# 전체 검증
if mismatch_count == 0:
    print("\n[OK] 샘플 검증 통과! 전체 검증 시작...")

    total_match = 0
    total_mismatch = 0

    for idx, (book, sent_id) in enumerate(sorted(list(common_set))):
        if idx % 10000 == 0:
            print(f"  진행: {idx:,} / {len(common_set):,}")

        pa_row = pa[(pa['book_name'] == book) & (pa['문장식별자'] == sent_id)]
        sa_rows = sa[(sa['book_name'] == book) & (sa['문장식별자'] == sent_id)]

        if len(pa_row) == 0 or len(sa_rows) == 0:
            continue

        pa_normalized = normalize_text(pa_row.iloc[0]['원문'])
        sa_text = ''.join(sa_rows['원문'].astype(str).tolist())
        sa_normalized = normalize_text(sa_text)

        if pa_normalized == sa_normalized:
            total_match += 1
        else:
            total_mismatch += 1

    print(f"\n=== 전체 검증 결과 ===")
    print(f"일치: {total_match:,}개 ({total_match/(total_match+total_mismatch)*100:.2f}%)")
    print(f"불일치: {total_mismatch:,}개 ({total_mismatch/(total_match+total_mismatch)*100:.2f}%)")

    if total_mismatch == 0:
        print("\n[OK][OK][OK] 완벽하게 일치합니다! PA와 SA는 정확히 같은 텍스트입니다.")
    else:
        print(f"\n[WARNING] {total_mismatch:,}개 문장이 불일치합니다. 원인 분석이 필요합니다.")
else:
    print(f"\n[WARNING] 샘플에서 {mismatch_count}개 불일치 발견. 위 예시를 확인하세요.")
