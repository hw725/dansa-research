#!/usr/bin/env python3
"""새 Gold v2와 sentence test 비교 검증"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s]+', '', str(t).strip())

# 새 Gold v2 로드
gold = pd.read_csv('datasets/phrase/test_gold_v2.csv')
sent = pd.read_csv('datasets/sentence/test.csv')

print('=== 데이터 크기 ===')
print(f'sentence test: {len(sent)} 행, {sent["문장식별자"].nunique()} 문장')
print(f'Gold v2: {len(gold)} 행, {gold["문장식별자"].nunique()} 문장')

# 문장 76 비교
sid = 76
g_rows = gold[gold['문장식별자'] == sid]
s_rows = sent[sent['문장식별자'] == sid]

print(f'\n=== 문장 {sid} 비교 ===')
if len(s_rows) > 0:
    s_src = s_rows.iloc[0]['원문']
    print(f'sentence test: [{s_src[:60]}...]')
    
if len(g_rows) > 0:
    print(f'Gold v2 ({len(g_rows)} segments):')
    g_full = ''.join([r['원문'] for _, r in g_rows.iterrows()])
    print(f'  합계: [{g_full[:60]}...]')

# 정규화 후 비교
if len(s_rows) > 0 and len(g_rows) > 0:
    s_norm = norm(s_rows.iloc[0]['원문'])
    g_norm = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
    print(f'\n정규화 비교:')
    print(f'  sent ({len(s_norm)} chars): [{s_norm[:60]}...]')
    print(f'  gold ({len(g_norm)} chars): [{g_norm[:60]}...]')
    print(f'  일치: {s_norm == g_norm}')

# 전체 검증 (상위 100개)
print('\n=== 전체 검증 (상위 100개) ===')
matches = mismatches = 0
for sid in sorted(sent['문장식별자'].unique())[:100]:
    g_rows = gold[gold['문장식별자'] == sid]
    s_rows = sent[sent['문장식별자'] == sid]
    if len(s_rows) == 0 or len(g_rows) == 0:
        continue
    s_norm = norm(s_rows.iloc[0]['원문'])
    g_norm = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
    if s_norm == g_norm:
        matches += 1
    else:
        mismatches += 1
        
print(f'일치: {matches}, 불일치: {mismatches}')
print(f'일치율: {100*matches/(matches+mismatches):.1f}%')
