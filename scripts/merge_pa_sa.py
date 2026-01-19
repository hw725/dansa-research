#!/usr/bin/env python3
"""
PA와 SA의 공통 문장만 추출하여 병합

PA: 문장병렬 (문장 단위, 여러 현토 포함)
SA: 구병렬 (구 단위, 현토 1개씩)
"""
import pandas as pd
import re

print("PA와 SA 데이터 로딩...")
pa = pd.read_csv('hyeonto/datasets/sentence_train_merged.csv')
sa = pd.read_csv('hyeonto/datasets/phrase_train_merged.csv')

print(f"PA 원본: {len(pa):,}행")
print(f"SA 원본: {len(sa):,}행")

# 공통 문장 추출
print("\n공통 문장 추출 중...")
pa_sentences = pa[['book_name', '문장식별자']].drop_duplicates()
sa_sentences = sa[['book_name', '문장식별자']].drop_duplicates()

pa_set = set(zip(pa_sentences['book_name'], pa_sentences['문장식별자']))
sa_set = set(zip(sa_sentences['book_name'], sa_sentences['문장식별자']))

common_set = pa_set & sa_set
print(f"공통 문장: {len(common_set):,}개")

# 공통 문장만 필터링
pa_filtered = pa[pa.apply(lambda x: (x['book_name'], x['문장식별자']) in common_set, axis=1)]
sa_filtered = sa[sa.apply(lambda x: (x['book_name'], x['문장식별자']) in common_set, axis=1)]

print(f"\nPA 필터링 후: {len(pa_filtered):,}행")
print(f"SA 필터링 후: {len(sa_filtered):,}행")

# 현토 마커 추출 함수
def extract_hyeonto_markers(text):
    """src 텍스트에서 한글 현토 마커 추출"""
    if pd.isna(text):
        return ''
    # 한글만 추출
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ''.join(matches) if matches else ''

print("\n현토 마커 추출 중...")

# PA: 문장병렬 (여러 현토를 join)
pa_filtered = pa_filtered.copy()
pa_filtered['marker'] = pa_filtered['원문'].apply(extract_hyeonto_markers)
pa_filtered['source_type'] = 'PA'  # 문장병렬 표시

# SA: 구병렬 (구마다 현토 1개씩)
sa_filtered = sa_filtered.copy()
sa_filtered['marker'] = sa_filtered['원문'].apply(extract_hyeonto_markers)
sa_filtered['source_type'] = 'SA'  # 구병렬 표시

# 컬럼 통일
pa_filtered = pa_filtered.rename(columns={'문단식별자': 'paragraph_id', '문장식별자': 'sentence_id'})
sa_filtered = sa_filtered.rename(columns={'문장식별자': 'sentence_id', '구식별자': 'phrase_id'})

# PA에 phrase_id 추가 (없으므로 NaN)
pa_filtered['phrase_id'] = pd.NA

# 컬럼 순서 통일
common_cols = ['book_name', 'paragraph_id', 'sentence_id', 'phrase_id',
               '원문', '번역문', 'marker', 'source_type']

# paragraph_id가 없는 SA에 추가
if 'paragraph_id' not in sa_filtered.columns:
    sa_filtered['paragraph_id'] = pd.NA

pa_selected = pa_filtered[common_cols]
sa_selected = sa_filtered[common_cols]

# 병합
print("\nPA와 SA 병합 중...")
merged = pd.concat([pa_selected, sa_selected], ignore_index=True)

print(f"\n병합 완료: {len(merged):,}행")
print(f"  - PA (문장병렬): {len(pa_selected):,}행")
print(f"  - SA (구병렬): {len(sa_selected):,}행")

# 마커 통계
print("\n현토 마커 통계:")
print(f"고유 마커 수: {merged['marker'].nunique():,}개")
print(f"빈 마커: {(merged['marker'] == '').sum():,}개")

# Top 10 마커
print("\nTop 10 마커:")
print(merged['marker'].value_counts().head(10))

# 저장
output_path = 'hyeonto/datasets/sentence_sa_merged.csv'
merged.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")

# 통계 저장
stats = {
    'pa_original': len(pa),
    'sa_original': len(sa),
    'common_sentences': len(common_set),
    'pa_filtered': len(pa_filtered),
    'sa_filtered': len(sa_filtered),
    'merged_total': len(merged),
    'unique_markers': merged['marker'].nunique(),
    'empty_markers': (merged['marker'] == '').sum(),
}

stats_path = 'hyeonto/datasets/sentence_sa_merged_stats.txt'
with open(stats_path, 'w', encoding='utf-8') as f:
    f.write("PA + SA 병합 통계\n")
    f.write("="*50 + "\n\n")
    for key, value in stats.items():
        f.write(f"{key}: {value:,}\n")

print(f"통계 저장 완료: {stats_path}")
