#!/usr/bin/env python3
"""
원본 4개 파일을 올바르게 병합:
1. hyeonto/datasets/pa/train.csv
2. hyeonto/datasets/sa/train.csv
3. datasets/pa/train.csv
4. datasets/sa/train.csv
"""
import pandas as pd
import re

print("=== Step 1: 원본 파일 구조 확인 ===\n")

files = {
    'hyeonto_pa': 'hyeonto/datasets/pa/train.csv',
    'hyeonto_sa': 'hyeonto/datasets/sa/train.csv',
    'root_pa': 'datasets/pa/train.csv',
    'root_sa': 'datasets/sa/train.csv',
}

dfs = {}
for name, path in files.items():
    df = pd.read_csv(path)
    dfs[name] = df
    print(f"{name}:")
    print(f"  경로: {path}")
    print(f"  행 수: {len(df):,}")
    print(f"  컬럼: {df.columns.tolist()}")
    print(f"  book_name 샘플: {df['book_name'].unique()[:3].tolist()}")
    print()

print("\n=== Step 2: 4개 파일 병합 ===\n")

# PA 병합 (hyeonto + root)
pa_merged = pd.concat([dfs['hyeonto_pa'], dfs['root_pa']], ignore_index=True)
print(f"PA 병합: {len(dfs['hyeonto_pa']):,} + {len(dfs['root_pa']):,} = {len(pa_merged):,}행")
print(f"PA 고유 book_name: {pa_merged['book_name'].nunique()}개")

# SA 병합 (hyeonto + root)
sa_merged = pd.concat([dfs['hyeonto_sa'], dfs['root_sa']], ignore_index=True)
print(f"SA 병합: {len(dfs['hyeonto_sa']):,} + {len(dfs['root_sa']):,} = {len(sa_merged):,}행")
print(f"SA 고유 book_name: {sa_merged['book_name'].nunique()}개")

print("\n=== Step 3: 공통 문장 확인 ===\n")

# 컬럼 통일 확인
print("PA 컬럼:", pa_merged.columns.tolist())
print("SA 컬럼:", sa_merged.columns.tolist())

# 공통 문장 찾기
if '문장식별자' in pa_merged.columns and '문장식별자' in sa_merged.columns:
    pa_sentences = set(zip(pa_merged['book_name'], pa_merged['문장식별자']))
    sa_sentences = set(zip(sa_merged['book_name'], sa_merged['문장식별자']))

    common = pa_sentences & sa_sentences
    only_pa = pa_sentences - sa_sentences
    only_sa = sa_sentences - pa_sentences

    print(f"\n공통 문장: {len(common):,}개")
    print(f"PA에만 있는 문장: {len(only_pa):,}개")
    print(f"SA에만 있는 문장: {len(only_sa):,}개")
else:
    print("\n[WARNING] 문장식별자 컬럼이 없습니다!")
    common = None

print("\n=== Step 4: 텍스트 일치 검증 (샘플 100개) ===\n")

if common:
    import random
    sample = random.sample(list(common), min(100, len(common)))

    match = 0
    mismatch = 0

    def normalize_text(text):
        if pd.isna(text):
            return ''
        return re.sub(r'\s+', '', str(text))

    for book, sent_id in sample:
        pa_row = pa_merged[(pa_merged['book_name'] == book) & (pa_merged['문장식별자'] == sent_id)]
        sa_rows = sa_merged[(sa_merged['book_name'] == book) & (sa_merged['문장식별자'] == sent_id)]

        if len(pa_row) == 0 or len(sa_rows) == 0:
            continue

        pa_norm = normalize_text(pa_row.iloc[0]['원문'])
        sa_norm = normalize_text(''.join(sa_rows['원문'].astype(str).tolist()))

        if pa_norm == sa_norm:
            match += 1
        else:
            mismatch += 1
            if mismatch <= 3:
                print(f"[MISMATCH {mismatch}] Book: {book}, Sent: {sent_id}")
                print(f"  PA len: {len(pa_norm)}, SA len: {len(sa_norm)}")

    print(f"\n검증 결과:")
    print(f"  일치: {match}/{match+mismatch} ({match/(match+mismatch)*100:.1f}%)")
    print(f"  불일치: {mismatch}/{match+mismatch} ({mismatch/(match+mismatch)*100:.1f}%)")

print("\n=== Step 5: 저장 ===\n")

# 공통 문장만 필터링
if common:
    pa_filtered = pa_merged[pa_merged.apply(lambda x: (x['book_name'], x['문장식별자']) in common, axis=1)]
    sa_filtered = sa_merged[sa_merged.apply(lambda x: (x['book_name'], x['문장식별자']) in common, axis=1)]

    print(f"PA 필터링: {len(pa_merged):,} -> {len(pa_filtered):,}행")
    print(f"SA 필터링: {len(sa_merged):,} -> {len(sa_filtered):,}행")

    # 저장
    pa_filtered.to_csv('hyeonto/datasets/pa_merged_v2.csv', index=False, encoding='utf-8-sig')
    sa_filtered.to_csv('hyeonto/datasets/sa_merged_v2.csv', index=False, encoding='utf-8-sig')

    print("\n저장 완료:")
    print("  hyeonto/datasets/pa_merged_v2.csv")
    print("  hyeonto/datasets/sa_merged_v2.csv")
else:
    print("[ERROR] 공통 문장을 찾을 수 없어서 저장하지 못했습니다.")

print("\n=== 완료 ===")
