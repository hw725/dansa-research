#!/usr/bin/env python3
"""문장 76 원문 직접 비교"""
import pandas as pd

sent_test = pd.read_csv('datasets/sentence/test.csv')
sent_xlsx = pd.read_excel('xlsx/당송팔대가문초구양수1/당송팔대가문초구양수1_문장병렬.xlsx')

# 문장 76 비교
print('=== 문장 76 비교 ===')
s76_test = sent_test[sent_test['문장식별자'] == 76]
s76_xlsx = sent_xlsx[sent_xlsx['문장식별자'] == 76]

print(f'sent_test 문장 76 ({len(s76_test)} rows):')
for _, r in s76_test.iterrows():
    book = r["book_name"]
    src = r["원문"][:80]
    print(f'  book: {book}')
    print(f'  원문: [{src}...]')
    
print(f'\n문장병렬 xlsx 문장 76 ({len(s76_xlsx)} rows):')
for _, r in s76_xlsx.iterrows():
    src = r["원문"][:80]
    print(f'  원문: [{src}...]')

# 일치하는지 확인
if len(s76_test) > 0 and len(s76_xlsx) > 0:
    import re
    def norm(t): return re.sub(r'\s+', '', str(t))
    t1 = norm(s76_test.iloc[0]["원문"])
    t2 = norm(s76_xlsx.iloc[0]["원문"])
    print(f'\n정규화 후:')
    print(f'  sent_test: [{t1}]')
    print(f'  xlsx:      [{t2}]')
    print(f'  일치: {t1 == t2}')
