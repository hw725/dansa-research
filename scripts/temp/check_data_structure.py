# -*- coding: utf-8 -*-
"""데이터 구조 검토 스크립트"""
import pandas as pd

print("=== PA Full Dataset ===")
df = pd.read_csv("hyeonto/datasets/pa_train_full.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n=== Reclustered Dataset ===")
df2 = pd.read_csv("hyeonto/reports/recluster_k16_child/reclustered.csv")
print(f"Total rows: {len(df2)}")
print(f"Columns: {list(df2.columns)}")
print("\nSample row:")
for col in df2.columns:
    val = df2[col].iloc[0]
    val_str = str(val)[:100] if pd.notna(val) else "NaN"
    print(f"  {col}: {val_str}")
