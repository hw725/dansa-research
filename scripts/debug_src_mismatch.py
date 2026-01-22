#!/usr/bin/env python3
"""원문 비교 분석"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

gold = pd.read_csv('datasets/phrase/test.csv')
pred = pd.read_csv('test_results/s2p_test_mapped.csv')

# 문장 1의 원문 비교
sid = 1
g_rows = gold[gold['문장식별자'] == sid]
p_rows = pred[pred['원본문장식별자'] == sid]

print(f'=== 문장 {sid} 원문 비교 ===')
print(f'Gold ({len(g_rows)} segments):')
for _, r in g_rows.head(4).iterrows():
    print(f'  [{r["원문"]}]')

print(f'\nPred ({len(p_rows)} segments):')
for _, r in p_rows.head(4).iterrows():
    print(f'  [{r["원문"]}]')

g_full = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
p_full = ''.join([norm(r['원문']) for _, r in p_rows.iterrows()])
print(f'\nGold full src ({len(g_full)} chars): {g_full[:100]}...')
print(f'Pred full src ({len(p_full)} chars): {p_full[:100]}...')
print(f'\nExact match: {g_full == p_full}')

# 더 많은 샘플 확인
common_sids = set(gold['문장식별자'].unique()) & set(pred['원본문장식별자'].unique())
mismatches = []
for sid in list(common_sids)[:50]:
    g_rows = gold[gold['문장식별자'] == sid]
    p_rows = pred[pred['원본문장식별자'] == sid]
    g_full = ''.join([norm(r['원문']) for _, r in g_rows.iterrows()])
    p_full = ''.join([norm(r['원문']) for _, r in p_rows.iterrows()])
    if g_full != p_full:
        mismatches.append({
            'sid': sid,
            'g_len': len(g_full),
            'p_len': len(p_full),
            'diff': len(g_full) - len(p_full)
        })

print(f'\n=== 원문 불일치 통계 (상위 50개 문장 중) ===')
print(f'불일치 수: {len(mismatches)} / 50')
if mismatches:
    print('샘플:')
    for m in mismatches[:5]:
        print(f'  문장 {m["sid"]}: Gold {m["g_len"]} chars, Pred {m["p_len"]} chars, diff={m["diff"]}')
