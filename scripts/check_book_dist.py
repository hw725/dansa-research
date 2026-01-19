#!/usr/bin/env python3
"""Book별 train/val/test 분포 확인"""

import pandas as pd

train = pd.read_csv('datasets/s2p/train.csv')
val = pd.read_csv('datasets/s2p/val.csv')
test = pd.read_csv('datasets/s2p/test.csv')

# Book별 문장 수 및 평균 구 개수
def analyze_by_book(df, name):
    book_stats = []
    for book, group in df.groupby('book_name'):
        n_sents = group['문장식별자'].nunique()
        avg_segs = group.groupby('문장식별자').size().mean()
        book_stats.append((book, n_sents, avg_segs))
    return pd.DataFrame(book_stats, columns=['book', f'{name}_sents', f'{name}_avg'])

train_stats = analyze_by_book(train, 'train')
val_stats = analyze_by_book(val, 'val')
test_stats = analyze_by_book(test, 'test')

# 병합
merged = train_stats.merge(val_stats, on='book', how='outer')
merged = merged.merge(test_stats, on='book', how='outer')
merged = merged.fillna(0)

print("=== Book별 분포 ===")
print(merged.sort_values('train_sents', ascending=False).head(10).to_string())

print("\n=== 평균 구 개수가 크게 다른 책들 ===")
for _, row in merged.iterrows():
    if row['train_sents'] > 0 and row['test_sents'] > 0:
        diff = abs(row['train_avg'] - row['test_avg'])
        if diff > 10:
            print(f"{row['book']}: train_avg={row['train_avg']:.1f}, test_avg={row['test_avg']:.1f}")
