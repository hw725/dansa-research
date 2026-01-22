#!/usr/bin/env python3
"""
PA 클러스터 결과를 대립 가설 검증용으로 준비

1. 현토 마커 추출 (src_left, src_right에서)
2. syntactic_function 분류
3. book 정규화 (사서/삼경/문집/역사서)
"""
import pandas as pd
import re
import json

# 1. PA 클러스터 결과 로드
print("=== PA 클러스터 결과 로드 ===")
df = pd.read_csv('hyeonto/reports/boundary_function_clusters/boundary_clusters.csv')
print(f"행수: {len(df):,}")
print(f"컬럼: {df.columns.tolist()}")

# 2. 현토 마커 추출
print("\n=== 현토 마커 추출 ===")

def extract_hyeonto_markers(text):
    """src 텍스트에서 한글 현토 마커 추출"""
    if pd.isna(text):
        return ''
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return '~'.join(matches) if matches else ''  # ~로 연속 표시

# src_left와 src_right에서 마커 추출
df['marker_left'] = df['src_left'].apply(extract_hyeonto_markers)
df['marker_right'] = df['src_right'].apply(extract_hyeonto_markers)

# 경계 패턴: left/right (슬래시로 구분)
df['marker_pattern'] = df['marker_left'] + '/' + df['marker_right']

print(f"고유 marker_pattern: {df['marker_pattern'].nunique():,}개")
print("\nTop 10 패턴:")
print(df['marker_pattern'].value_counts().head(10))

# 3. Syntactic function 분류 (간단 버전)
print("\n=== Syntactic Function 분류 ===")

# configs/syntactic_function_mapping.json 로드
try:
    with open('configs/syntactic_function_mapping.json', 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        syntactic_mapping = mapping_data.get('mappings', {})
    print(f"매핑 테이블 로드: {len(syntactic_mapping)}개 항목")
except FileNotFoundError:
    print("매핑 테이블 없음. 기본값 '기타' 사용")
    syntactic_mapping = {}

def classify_syntactic_function(marker):
    """마커의 syntactic function 분류"""
    if pd.isna(marker) or marker == '':
        return '기타'

    # 슬래시로 분리된 경우 마지막 현토를 기준으로 판단
    main_marker = marker.split('/')[-1] if '/' in marker else marker

    # 매핑 테이블 조회
    if main_marker in syntactic_mapping:
        return syntactic_mapping[main_marker]

    # 규칙 기반
    if '?' in main_marker or main_marker in ['니', '가', 'ㄴ가', '뇨', 'ㄹ까', 'ㄴ저']:
        return '의문종결'
    elif main_marker in ['라', '니라', '도다', '리라', '러라']:
        return '평서종결'
    else:
        return '기타'

df['syntactic_function_left'] = df['marker_left'].apply(classify_syntactic_function)
df['syntactic_function_right'] = df['marker_right'].apply(classify_syntactic_function)

print("\nLeft marker function 분포:")
print(df['syntactic_function_left'].value_counts())

print("\nRight marker function 분포:")
print(df['syntactic_function_right'].value_counts())

# 4. Book 정규화
print("\n=== Book 정규화 ===")

BOOK_NORMALIZATION = {
    # 사서
    '논어집주': '사서',
    '맹자집주': '사서',
    '대학': '사서',
    '중용': '사서',

    # 삼경
    '시경': '삼경',
    '시경집전(상)': '삼경',
    '시경집전(하)': '삼경',
    '서경': '삼경',
    '서경집전(상)': '삼경',
    '서경집전(하)': '삼경',
    '역경': '삼경',

    # 역사서
    '자치통감': '역사서',
    '십팔사략': '역사서',

    # 문집 (당송팔대가문초)
    '당송팔대가문초': '문집',
}

def normalize_book(book_name):
    """Book 이름 정규화"""
    if pd.isna(book_name):
        return '기타'

    book_str = str(book_name)

    # 정확한 매칭
    if book_str in BOOK_NORMALIZATION:
        return BOOK_NORMALIZATION[book_str]

    # 부분 매칭
    for key, value in BOOK_NORMALIZATION.items():
        if key in book_str:
            return value

    # 패턴 매칭
    if '논어' in book_str or '맹자' in book_str:
        return '사서'
    elif '시경' in book_str or '서경' in book_str or '역경' in book_str:
        return '삼경'
    elif '팔대가' in book_str or '문초' in book_str:
        return '문집'
    elif '사략' in book_str or '통감' in book_str:
        return '역사서'
    else:
        return '기타'

df['book_category'] = df['book_name'].apply(normalize_book)

print("\nBook 카테고리 분포:")
print(df['book_category'].value_counts())

# 5. 저장
output_path = 'hyeonto/reports/boundary_function_clusters/pa_clusters_with_features.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")

# 통계
print("\n=== 최종 통계 ===")
print(f"전체 행수: {len(df):,}")
print(f"클러스터 수: {df['cluster_id'].nunique()}")
print(f"고유 마커 패턴: {df['marker_pattern'].nunique():,}")
print(f"서종 카테고리: {df['book_category'].nunique()}")

# 클러스터별 서종 분포
print("\n클러스터별 사서 비율:")
for cluster_id in sorted(df['cluster_id'].unique()):
    cluster_df = df[df['cluster_id'] == cluster_id]
    saseo_count = len(cluster_df[cluster_df['book_category'] == '사서'])
    saseo_ratio = saseo_count / len(cluster_df) * 100 if len(cluster_df) > 0 else 0
    print(f"  Cluster {cluster_id}: {saseo_ratio:.1f}% ({saseo_count}/{len(cluster_df)})")
