#!/usr/bin/env python3
"""PA/SA 데이터셋 구조 확인"""
import pandas as pd
import re

# PA 데이터 확인
print("=== PA (문장병렬) 데이터 ===")
pa = pd.read_csv('hyeonto/datasets/pa_train_merged.csv', nrows=10)
print(f"행 수: {len(pa)}")
print(f"컬럼: {pa.columns.tolist()}")
print("\n샘플:")
for idx, row in pa.iterrows():
    src = str(row['원문'])
    hangul = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', src)
    print(f"{idx}: 원문=\"{src[:60]}\"")
    print(f"    한글 추출: {hangul}")
    if idx >= 2:
        break

print("\n" + "="*60)
print("=== SA (구병렬) 데이터 ===")
sa = pd.read_csv('hyeonto/datasets/sa_train_merged.csv', nrows=10)
print(f"행 수: {len(sa)}")
print(f"컬럼: {sa.columns.tolist()}")
print("\n샘플:")
for idx, row in sa.iterrows():
    src = str(row['원문'])
    hangul = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', src)
    print(f"{idx}: 원문=\"{src[:60]}\"")
    print(f"    한글 추출: {hangul}")
    if idx >= 2:
        break
