#!/usr/bin/env python3
"""
P2S와 S2P의 공통 문장만 추출하여 병합

P2S: 문장병렬 (문장 단위, 여러 현토 포함)
S2P: 구병렬 (구 단위, 현토 1개씩)
"""
import pandas as pd
import re

print("P2S와 S2P 데이터 로딩...")
p2s = pd.read_csv('hyeonto/datasets/sentence_train_merged.csv')
s2p = pd.read_csv('hyeonto/datasets/phrase_train_merged.csv')

print(f"P2S 원본: {len(p2s):,}행")
print(f"S2P 원본: {len(s2p):,}행")

# 공통 문장 추출
print("\n공통 문장 추출 중...")
p2s_sentences = p2s[['book_name', '문장식별자']].drop_duplicates()
s2p_sentences = s2p[['book_name', '문장식별자']].drop_duplicates()

p2s_set = set(zip(p2s_sentences['book_name'], p2s_sentences['문장식별자']))
s2p_set = set(zip(s2p_sentences['book_name'], s2p_sentences['문장식별자']))

common_set = p2s_set & s2p_set
print(f"공통 문장: {len(common_set):,}개")

# 공통 문장만 필터링
p2s_filtered = p2s[p2s.apply(lambda x: (x['book_name'], x['문장식별자']) in common_set, axis=1)]
s2p_filtered = s2p[s2p.apply(lambda x: (x['book_name'], x['문장식별자']) in common_set, axis=1)]

print(f"\nP2S 필터링 후: {len(p2s_filtered):,}행")
print(f"S2P 필터링 후: {len(s2p_filtered):,}행")

# 현토 마커 추출 함수
def extract_hyeonto_markers(text):
    """src 텍스트에서 한글 현토 마커 추출"""
    if pd.isna(text):
        return ''
    # 한글만 추출
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ''.join(matches) if matches else ''

print("\n현토 마커 추출 중...")

# P2S: 문장병렬 (여러 현토를 join)
p2s_filtered = p2s_filtered.copy()
p2s_filtered['marker'] = p2s_filtered['원문'].apply(extract_hyeonto_markers)
p2s_filtered['source_type'] = 'P2S'  # 문장병렬 표시

# S2P: 구병렬 (구마다 현토 1개씩)
s2p_filtered = s2p_filtered.copy()
s2p_filtered['marker'] = s2p_filtered['원문'].apply(extract_hyeonto_markers)
s2p_filtered['source_type'] = 'S2P'  # 구병렬 표시

# 컬럼 통일
p2s_filtered = p2s_filtered.rename(columns={'문단식별자': 'paragraph_id', '문장식별자': 'sentence_id'})
s2p_filtered = s2p_filtered.rename(columns={'문장식별자': 'sentence_id', '구식별자': 'phrase_id'})

# P2S에 phrase_id 추가 (없으므로 NaN)
p2s_filtered['phrase_id'] = pd.NA

# 컬럼 순서 통일
common_cols = ['book_name', 'paragraph_id', 'sentence_id', 'phrase_id',
               '원문', '번역문', 'marker', 'source_type']

# paragraph_id가 없는 S2P에 추가
if 'paragraph_id' not in s2p_filtered.columns:
    s2p_filtered['paragraph_id'] = pd.NA

p2s_selected = p2s_filtered[common_cols]
s2p_selected = s2p_filtered[common_cols]

# 병합
print("\nP2S와 S2P 병합 중...")
merged = pd.concat([p2s_selected, s2p_selected], ignore_index=True)

print(f"\n병합 완료: {len(merged):,}행")
print(f"  - P2S (문장병렬): {len(p2s_selected):,}행")
print(f"  - S2P (구병렬): {len(s2p_selected):,}행")

# 마커 통계
print("\n현토 마커 통계:")
print(f"고유 마커 수: {merged['marker'].nunique():,}개")
print(f"빈 마커: {(merged['marker'] == '').sum():,}개")

# Top 10 마커
print("\nTop 10 마커:")
print(merged['marker'].value_counts().head(10))

# 저장
output_path = 'hyeonto/datasets/sentence_s2p_merged.csv'
merged.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")

# 통계 저장
stats = {
    'p2s_original': len(p2s),
    's2p_original': len(s2p),
    'common_sentences': len(common_set),
    'p2s_filtered': len(p2s_filtered),
    's2p_filtered': len(s2p_filtered),
    'merged_total': len(merged),
    'unique_markers': merged['marker'].nunique(),
    'empty_markers': (merged['marker'] == '').sum(),
}

stats_path = 'hyeonto/datasets/sentence_s2p_merged_stats.txt'
with open(stats_path, 'w', encoding='utf-8') as f:
    f.write("P2S + S2P 병합 통계\n")
    f.write("="*50 + "\n\n")
    for key, value in stats.items():
        f.write(f"{key}: {value:,}\n")

print(f"통계 저장 완료: {stats_path}")

