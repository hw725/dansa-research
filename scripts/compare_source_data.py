#!/usr/bin/env python3
"""원본 데이터 구조 비교"""
import pandas as pd

sent = pd.read_csv('datasets/sentence/test.csv')
phrase = pd.read_csv('datasets/phrase/test.csv')

# 문장 1 비교
sid = 1
sent_rows = sent[sent['문장식별자'] == sid]
phrase_rows = phrase[phrase['문장식별자'] == sid]

print('=== 원본 데이터 비교 (문장 1) ===')
print(f'sentence/test.csv에서 문장 1:')
for _, r in sent_rows.iterrows():
    print(f'  원문: [{r["원문"][:80]}...]')
    print(f'  번역문: [{r["번역문"][:80]}...]')

print(f'\nphrase/test.csv에서 문장 1 ({len(phrase_rows)} segments):')
for i, (_, r) in enumerate(phrase_rows.iterrows()):
    if i >= 5:
        print(f'  ... 외 {len(phrase_rows) - 5}개 더')
        break
    print(f'  {i+1}. 원문: [{r["원문"]}] / 번역문: [{r["번역문"][:30]}...]')

# 원문 합계 비교
sent_src = sent_rows['원문'].iloc[0] if len(sent_rows) > 0 else ''
phrase_src = ''.join(phrase_rows['원문'].values)
print(f'\nsentence 원문 길이: {len(sent_src)}')
print(f'phrase 원문 합계 길이: {len(phrase_src)}')

# 공백 제거 후 비교
import re
def norm(t): return re.sub(r'\s+', '', str(t))
print(f'\n정규화 후 일치: {norm(sent_src) == norm(phrase_src)}')
print(f'정규화 sentence 길이: {len(norm(sent_src))}')
print(f'정규화 phrase 길이: {len(norm(phrase_src))}')
