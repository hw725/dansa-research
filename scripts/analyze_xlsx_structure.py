#!/usr/bin/env python3
"""xlsx 원본 데이터 구조 분석"""
import pandas as pd

# 데이터 로드
sent_test = pd.read_csv('datasets/sentence/test.csv')
phrase_xlsx = pd.read_excel('xlsx/당송팔대가문초구양수1/당송팔대가문초구양수1_구병렬.xlsx')
sent_xlsx = pd.read_excel('xlsx/당송팔대가문초구양수1/당송팔대가문초구양수1_문장병렬.xlsx')

# 문장식별자 범위 확인
print('=== 데이터 구조 분석 ===')
book1 = sent_test[sent_test['book_name'] == '당송팔대가문초구양수1']
print(f'sent_test (당송팔1):')
print(f'  문장식별자 범위: {book1["문장식별자"].min()} - {book1["문장식별자"].max()}')
print(f'  문장 개수: {len(book1)}')

print(f'\n구병렬 xlsx:')
print(f'  문장식별자 범위: {phrase_xlsx["문장식별자"].min()} - {phrase_xlsx["문장식별자"].max()}')
print(f'  고유 문장 개수: {phrase_xlsx["문장식별자"].nunique()}')
print(f'  총 구 행: {len(phrase_xlsx)}')

print(f'\n문장병렬 xlsx:')
print(f'  문장식별자 범위: {sent_xlsx["문장식별자"].min()} - {sent_xlsx["문장식별자"].max()}')
print(f'  문장 개수: {len(sent_xlsx)}')

# 문장 1 비교
print('\n=== 문장 1 원문 비교 ===')
s1_test = sent_test[sent_test['문장식별자'] == 1]
s1_xlsx = sent_xlsx[sent_xlsx['문장식별자'] == 1]
p1_xlsx = phrase_xlsx[phrase_xlsx['문장식별자'] == 1]

if len(s1_test) > 0:
    print(f'sent_test 문장 1:\n  원문: {s1_test.iloc[0]["원문"]}')
if len(s1_xlsx) > 0:
    print(f'문장병렬 xlsx 문장 1:\n  원문: {s1_xlsx.iloc[0]["원문"]}')
if len(p1_xlsx) > 0:
    p1_full = ''.join(p1_xlsx['원문'].values)
    print(f'구병렬 xlsx 문장 1 ({len(p1_xlsx)} segments):\n  원문 합: {p1_full[:100]}...')
    
# 문장 76 비교 (불일치가 발생한 것)
print('\n=== 문장 76 원문 비교 ===')
s76_test = sent_test[sent_test['문장식별자'] == 76]
p76_xlsx = phrase_xlsx[phrase_xlsx['문장식별자'] == 76]

if len(s76_test) > 0:
    print(f'sent_test 문장 76:\n  원문: {s76_test.iloc[0]["원문"]}')
if len(p76_xlsx) > 0:
    p76_full = ''.join(p76_xlsx['원문'].values)
    print(f'구병렬 xlsx 문장 76 ({len(p76_xlsx)} segments):\n  원문 합: {p76_full[:100]}...')
