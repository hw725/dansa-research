#!/usr/bin/env python3
"""
Phrase 데이터 정규화 및 준비

phrase_full.csv → data/phrase_normalized.csv
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import hashlib
import regex

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
HYEONTO_DIR = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "data"

# hyeonto_normalizer import
sys.path.insert(0, str(HYEONTO_DIR))
from hyeonto_normalizer import normalize_hyeonto_marker

DATA_DIR.mkdir(exist_ok=True)

def extract_markers_from_text(text: str) -> str:
    """
    원문에서 현토 마커 추출
    - 문장/구 내부가 아니라 "끝부분" 한글만 추출
    - 가운뎃점(ㆍ)과 소괄호 내용 제외
    """
    if pd.isna(text):
        return ''

    text = str(text).strip()

    # 소괄호 제거: (음가) 형태 제외 (경계 보존을 위해 공백으로 치환)
    text = regex.sub(r'\([^)]*\)', ' ', text)

    # 가운뎃점 제거
    text = text.replace('ㆍ', '')

    # 말미의 비한글 제거 후, 끝부분 한글만 추출
    text = regex.sub(r'[^\p{Hangul}]+$', '', text)
    match = regex.search(r'[\p{Hangul}]+$', text)
    if not match:
        return ''

    return match.group(0)

def prepare_phrase_data():
    """phrase_full.csv 또는 기존 phrase_normalized.csv 로드"""
    phrase_full = DATASETS_DIR / "phrase_full.csv"
    phrase_existing = DATA_DIR / "phrase_normalized.csv"

    if phrase_full.exists():
        source_path = phrase_full
    elif phrase_existing.exists():
        source_path = phrase_existing
    else:
        print(f"❌ {phrase_full} 없음")
        print(f"❌ {phrase_existing} 없음")
        return False

    print(f"📖 로드: {source_path}")
    df = pd.read_csv(source_path)
    print(f"   행 수: {len(df):,}")
    
    # 컬럼 확인
    print(f"   컬럼: {list(df.columns)}")
    
    # 원문 열 확인
    if '원문' not in df.columns:
        print(f"❌ 원문 열 없음. 사용 가능: {list(df.columns)}")
        return False
    
    # 마커 추출 (원문에서)
    print(f"   원문에서 마커 추출 중...")
    df['marker'] = df['원문'].apply(extract_markers_from_text)
    if 'marker_final' in df.columns:
        df = df.drop(columns=['marker_final'])
    if 'marker_normalized' in df.columns:
        df = df.drop(columns=['marker_normalized'])
    
    print(f"   마커 추출 완료")
    
    return df

def normalize_markers(df):
    """마커 정규화"""
    print("\n📋 마커 정규화 중...")
    
    # marker_final 열 생성 또는 이미 있으면 유지
    if 'marker_final' not in df.columns:
        df['marker_final'] = df['marker'].fillna('').apply(
            lambda x: normalize_hyeonto_marker(x) if x else ''
        )
    
    # 정규화 통계
    normalized_count = (df['marker_final'] != '').sum()
    print(f"   정규화됨: {normalized_count:,}건")
    
    return df

def anonymize_translations(df):
    """
    번역문 익명화 (SHA-256 해시)
    ⚠️  분석이 모두 완료된 후에만 호출!
    """
    print("\n🔐 번역문 익명화 중...")
    
    # 번역문 열 확인
    if '번역문' not in df.columns:
        print(f"   ⚠️  번역문 열 없음")
        return df
    
    print(f"   익명화 열: 번역문")
    
    def anonymize(text):
        if pd.isna(text) or text == "":
            return ""
        return hashlib.sha256(str(text).encode('utf-8')).hexdigest()[:16]
    
    original_count = df['번역문'].notna().sum()
    df['번역문'] = df['번역문'].apply(anonymize)
    hashed_count = (df['번역문'] != '').sum()
    print(f"   익명화됨: {original_count:,} → {hashed_count:,}건")
    
    return df

def add_metadata(df):
    """메타데이터 추가"""
    print("\n📌 메타데이터 추가 중...")
    
    # book 정보 추가 (있으면 유지)
    if 'book' not in df.columns:
        df['book'] = 'hyeonto'
    
    # marker_normalized = marker_final (호환성)
    if 'marker_normalized' not in df.columns and 'marker_final' in df.columns:
        df['marker_normalized'] = df['marker_final']
    
    return df

def save_data(df, output_path):
    """저장"""
    print(f"\n💾 저장 중: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ {len(df):,}행 저장 완료")
    return True

def main():
    print("="*70)
    print("📖 Phrase 데이터 정규화")
    print("="*70)
    
    # Step 1: phrase_full.csv 로드
    df = prepare_phrase_data()
    if df is None or df is False:
        return False
    
    # Step 2: 마커 정규화
    df = normalize_markers(df)
    
    # Step 3: 메타데이터 추가
    df = add_metadata(df)
    
    # Step 4: 저장 (번역문 익명화 전!)
    output_path = DATA_DIR / "phrase_normalized.csv"
    save_data(df, output_path)
    
    print("\n" + "="*70)
    print("✅ 정규화 완료!")
    print("="*70)
    print("⚠️  번역문 익명화는 분석 완료 후 별도로 실행하세요:")
    print("   python anonymize_dataset.py")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
