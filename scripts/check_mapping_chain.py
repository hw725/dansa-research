#!/usr/bin/env python3
"""매핑 체인 확인"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s]+', '', str(t).strip())

# 데이터 확인
s2p_mapped = pd.read_csv('test_results/s2p_test_mapped.csv')
p2s_input = pd.read_excel('test_results/p2s_full_fixed.xlsx')
sent_test = pd.read_csv('datasets/sentence/test.csv')

print('=== 매핑 체인 확인 ===')

# s2p_mapped의 문장식별자 1 (P2S 체계)
s2p_sid1 = s2p_mapped[s2p_mapped['문장식별자'] == 1]
print(f'S2P 출력 문장식별자=1:')
print(f'  rows: {len(s2p_sid1)}')
if len(s2p_sid1) > 0:
    s = ''.join([norm(r['원문']) for _, r in s2p_sid1.iterrows()])
    print(f'  원문: [{s[:60]}...]')
    orig_sid = s2p_sid1.iloc[0]["원본문장식별자"]
    print(f'  원본문장식별자: {orig_sid}')

# p2s_input의 문장식별자 1
p2s_sid1 = p2s_input[p2s_input['문장식별자'] == 1]
print(f'\nP2S 입력 문장식별자=1:')
print(f'  rows: {len(p2s_sid1)}')
if len(p2s_sid1) > 0:
    print(f'  원문: [{norm(p2s_sid1.iloc[0]["원문"])[:60]}...]')

# sent_test에서 원본문장식별자로 찾기
if len(s2p_sid1) > 0:
    orig_sid = int(s2p_sid1.iloc[0]["원본문장식별자"])
    sent_orig = sent_test[sent_test['문장식별자'] == orig_sid]
    print(f'\nsent_test 문장식별자={orig_sid}:')
    if len(sent_orig) > 0:
        s = ''.join([norm(r['원문']) for _, r in sent_orig.iterrows()])
        print(f'  원문: [{s[:60]}...]')
