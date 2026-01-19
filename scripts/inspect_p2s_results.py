import pandas as pd
df = pd.read_excel('test_results/p2s_test_10.xlsx')
print(f'총 행 수: {len(df)}')
print(f'컬럼: {list(df.columns)}')
src_lens = df['원문'].astype(str).str.len()
tgt_lens = df['번역문'].astype(str).str.len()
print(f'원문 길이: min={src_lens.min()}, max={src_lens.max()}, mean={src_lens.mean():.1f}')
print(f'번역문 길이: min={tgt_lens.min()}, max={tgt_lens.max()}, mean={tgt_lens.mean():.1f}')
print()
print('=== 첫 5개 ===')
for i, row in df.head(5).iterrows():
    src = str(row['원문'])[:60]
    tgt = str(row['번역문'])[:60]
    print(f'{i}: [{src}] -> [{tgt}]')
print()
print('=== 마지막 5개 ===')
for i, row in df.tail(5).iterrows():
    src = str(row['원문'])[:60]
    tgt = str(row['번역문'])[:60]
    print(f'{i}: [{src}] -> [{tgt}]')
