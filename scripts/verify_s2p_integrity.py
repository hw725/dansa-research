#!/usr/bin/env python3
"""S2P 입력 vs 출력 원문 무결성 검증"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

# S2P 입력
sent_test = pd.read_csv('datasets/sentence/test.csv')
# S2P 출력 (매핑된 원본 문장식별자 포함)
s2p_out = pd.read_csv('test_results/s2p_test_mapped.csv')

print('=== S2P 입출력 원문 무결성 검증 ===')

# 문장별 비교
matches = 0
mismatches = []

common_sids = set(sent_test['문장식별자'].unique()) & set(s2p_out['원본문장식별자'].unique())
print(f'공통 문장: {len(common_sids)}')

for sid in list(common_sids)[:500]:
    # 입력 원문 (sentence test에서)
    in_rows = sent_test[sent_test['문장식별자'] == sid]
    in_src = ''.join([norm(r['원문']) for _, r in in_rows.iterrows()])
    
    # 출력 원문 (S2P 출력에서)
    out_rows = s2p_out[s2p_out['원본문장식별자'] == sid]
    out_src = ''.join([norm(r['원문']) for _, r in out_rows.iterrows()])
    
    if in_src == out_src:
        matches += 1
    else:
        mismatches.append({
            'sid': sid,
            'in_len': len(in_src),
            'out_len': len(out_src),
            'in_src': in_src[:50],
            'out_src': out_src[:50]
        })

print(f'일치: {matches} / 500 ({100*matches/500:.1f}%)')
print(f'불일치: {len(mismatches)}')

if mismatches:
    print('\n샘플 불일치:')
    for m in mismatches[:5]:
        print(f'  문장 {m["sid"]}: 입력 {m["in_len"]} → 출력 {m["out_len"]} chars')
        print(f'    입력: [{m["in_src"]}...]')
        print(f'    출력: [{m["out_src"]}...]')
