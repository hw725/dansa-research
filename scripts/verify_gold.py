#!/usr/bin/env python3
"""새 Gold와 sentence test 비교 검증"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s]+', '', str(t).strip())

# 데이터 로드
gold = pd.read_csv('datasets/phrase/test_reconstructed.csv')
sent = pd.read_csv('datasets/sentence/test.csv')

# 문장 1 비교
sid = 1
g_rows = gold[gold['문장식별자'] == sid]
s_row = sent[sent['문장식별자'] == sid].iloc[0]

print('=== 문장 1 비교 ===')
print(f'sentence/test.csv 원문: [{s_row["원문"]}]')
print(f'Gold reconstructed ({len(g_rows)} segments):')
for _, r in g_rows.iterrows():
    print(f'  [{r["원문"]}]')

g_full = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
s_norm = norm(s_row['원문'])
print(f'\nsentence 정규화 ({len(s_norm)} chars): [{s_norm[:80]}...]')
print(f'Gold 합계 정규화 ({len(g_full)} chars): [{g_full[:80]}...]')
print(f'일치: {s_norm == g_full}')

# 전체 검증
print('\n=== 전체 검증 ===')
matches = 0
mismatches = []
for sid in sent['문장식별자'].unique()[:100]:
    g_rows = gold[gold['문장식별자'] == sid]
    s_rows = sent[sent['문장식별자'] == sid]
    if len(s_rows) == 0 or len(g_rows) == 0:
        continue
    s_norm = norm(s_rows.iloc[0]['원문'])
    g_full = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
    if s_norm == g_full:
        matches += 1
    else:
        mismatches.append({'sid': sid, 's_len': len(s_norm), 'g_len': len(g_full)})

print(f'상위 100개 문장 중 일치: {matches}')
print(f'불일치: {len(mismatches)}')
if mismatches:
    print('불일치 샘플:')
    for m in mismatches[:3]:
        print(f'  문장 {m["sid"]}: sent {m["s_len"]} chars, gold {m["g_len"]} chars')
