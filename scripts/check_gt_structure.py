#!/usr/bin/env python3
"""GT와 Candidates CSV 구조 비교"""

import pandas as pd

# GT 파일 (문단 단위)
df_gt = pd.read_csv('datasets/pd/test_100.csv', encoding='utf-8-sig')
print('=== GT (datasets/pd/test_100.csv) ===')
print(f'Rows: {len(df_gt)}')
print(f'Columns: {df_gt.columns.tolist()}')
print(f'\nFirst row:')
print(df_gt.iloc[0])
print(f'\nPID 10 count: {len(df_gt[df_gt["문단식별자"]==10])}')

# Candidates CSV (PA 생성, 문장 단위)
df_candidates = pd.read_csv('test_results/pa_strict_pd_test100_candidates.csv', encoding='utf-8-sig')
print('\n=== Candidates CSV (PA 생성, 문장 단위) ===')
print(f'Rows: {len(df_candidates)}')
print(f'Columns: {df_candidates.columns.tolist()[:8]}...')
print(f'\nPID 10 count: {len(df_candidates[df_candidates["문단식별자"]==10])}')

# Gold standard 확인 (datasets/pa 폴더)
try:
    df_gold = pd.read_csv('datasets/pa/test_100_from_pd.csv', encoding='utf-8-sig')
    print('\n=== Gold Standard (datasets/pa/test_100_from_pd.csv) ===')
    print(f'Rows: {len(df_gold)}')
    print(f'Columns: {df_gold.columns.tolist()[:8]}...')
    print(f'\nPID 10 count: {len(df_gold[df_gold["문단식별자"]==10])}')
except FileNotFoundError:
    print('\n⚠️ datasets/pa/test_100_from_pd.csv not found')
