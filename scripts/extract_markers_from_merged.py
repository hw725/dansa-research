#!/usr/bin/env python3
"""
PA와 SA merged 데이터에서 현토 마커 추출

PA: 문장 패턴 (여러 현토 join)
SA: 개별 현토 (각 구마다 1개)
"""
import pandas as pd
import re

def extract_hyeonto_markers(text):
    """src 텍스트에서 한글 현토 마커 추출"""
    if pd.isna(text):
        return ''
    # 한글만 추출
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ''.join(matches) if matches else ''

print("=== PA 데이터 처리 ===")
pa = pd.read_csv('hyeonto/datasets/pa_merged_v2.csv')
print(f"PA 행수: {len(pa):,}")

# 현토 마커 추출
pa['marker'] = pa['원문'].apply(extract_hyeonto_markers)
pa['source_type'] = 'PA'

print(f"PA 고유 마커: {pa['marker'].nunique():,}개")
print(f"PA 빈 마커: {(pa['marker'] == '').sum():,}개")

print("\n=== SA 데이터 처리 ===")
sa = pd.read_csv('hyeonto/datasets/sa_merged_v2.csv')
print(f"SA 행수: {len(sa):,}")

# 현토 마커 추출
sa['marker'] = sa['원문'].apply(extract_hyeonto_markers)
sa['source_type'] = 'SA'

print(f"SA 고유 마커: {sa['marker'].nunique():,}개")
print(f"SA 빈 마커: {(sa['marker'] == '').sum():,}개")

print("\n=== 병합 ===")
# PA는 paragraph_id 있음, SA는 없음 (phrase_id 있음)
pa_cols = ['book_name', '문단식별자', '문장식별자', '원문', '번역문', 'marker', 'source_type']
sa_cols = ['book_name', '문장식별자', '구식별자', '원문', '번역문', 'marker', 'source_type']

# SA에 paragraph_id 추가
if '문단식별자' not in sa.columns:
    sa['문단식별자'] = pd.NA

# PA에 phrase_id 추가
if '구식별자' not in pa.columns:
    pa['구식별자'] = pd.NA

# 컬럼 통일
common_cols = ['book_name', '문단식별자', '문장식별자', '구식별자', '원문', '번역문', 'marker', 'source_type']

pa_selected = pa[common_cols]
sa_selected = sa[common_cols]

merged = pd.concat([pa_selected, sa_selected], ignore_index=True)

print(f"\n병합 완료: {len(merged):,}행")
print(f"  PA (문장병렬): {len(pa_selected):,}행")
print(f"  SA (구병렬): {len(sa_selected):,}행")
print(f"\n전체 고유 마커: {merged['marker'].nunique():,}개")

# 마커 통계
print("\n=== 마커 통계 ===")
print("\nTop 20 마커 (전체):")
print(merged['marker'].value_counts().head(20))

print("\nTop 20 마커 (PA만):")
print(pa['marker'].value_counts().head(20))

print("\nTop 20 마커 (SA만):")
print(sa['marker'].value_counts().head(20))

# 저장
output_path = 'hyeonto/datasets/pa_sa_merged_with_markers.csv'
merged.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")
