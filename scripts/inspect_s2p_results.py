import pandas as pd
df = pd.read_excel('test_results/s2p_test_10.xlsx')
print(f'총 행 수: {len(df)}')

# 원문 길이 분포
src_lens = df['원문'].astype(str).str.len()
print(f'원문 길이: min={src_lens.min()}, max={src_lens.max()}, mean={src_lens.mean():.1f}')

# 번역문 길이 분포
tgt_lens = df['번역문'].astype(str).str.len()
print(f'번역문 길이: min={tgt_lens.min()}, max={tgt_lens.max()}, mean={tgt_lens.mean():.1f}')

# 첫 5개와 마지막 5개 출력
print()
print('=== 첫 5개 ===')
for i, row in df.head(5).iterrows():
    print(f'{i}: [{row["원문"]}] -> [{row["번역문"]}]')
print()
print('=== 마지막 5개 ===')
for i, row in df.tail(5).iterrows():
    print(f'{i}: [{row["원문"]}] -> [{row["번역문"]}]')
