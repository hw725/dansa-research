#!/usr/bin/env python3
"""S2P 입력 데이터 확인"""
import pandas as pd

# 입력으로 사용된 p2s_full_fixed.xlsx
p2s = pd.read_excel('test_results/p2s_full_fixed.xlsx')
print('p2s_full_fixed.xlsx (S2P 입력):')
print(f'  행 수: {len(p2s)}')
print(f'  문장ID 범위: {p2s["문장식별자"].min()} - {p2s["문장식별자"].max()}')
print(f'  고유 문장: {p2s["문장식별자"].nunique()}')

# datasets/sentence/test.csv
sent = pd.read_csv('datasets/sentence/test.csv')
print(f'\ndatasets/sentence/test.csv:')
print(f'  행 수: {len(sent)}')
print(f'  문장ID 범위: {sent["문장식별자"].min()} - {sent["문장식별자"].max()}')
print(f'  고유 문장: {sent["문장식별자"].nunique()}')

# 첫 번째 문장 비교
print(f'\n첫 번째 문장 원문 비교:')
print(f'  p2s 문장ID 1: [{p2s[p2s["문장식별자"]==1].iloc[0]["원문"][:60]}...]')

sent1 = sent[sent["문장식별자"] == sent["문장식별자"].min()]
print(f'  sent 첫째 문장ID {sent["문장식별자"].min()}: [{sent1.iloc[0]["원문"][:60]}...]')

# 문장ID 체계가 같은지?
print(f'\n문장ID 체계 비교:')
p2s_ids = set(p2s["문장식별자"].unique())
sent_ids = set(sent["문장식별자"].unique())
overlap = p2s_ids & sent_ids
print(f'  p2s IDs: {min(p2s_ids)} - {max(p2s_ids)}')
print(f'  sent IDs: {min(sent_ids)} - {max(sent_ids)}')
print(f'  겹치는 ID 수: {len(overlap)}')
