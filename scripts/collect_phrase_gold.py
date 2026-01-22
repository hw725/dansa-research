#!/usr/bin/env python3
"""
datasets/sentence/test.csv에 해당하는 구병렬 Gold 데이터를 xlsx에서 수집

1. datasets/sentence/test.csv 로드
2. 각 문장의 book_name과 문장식별자로 해당 xlsx 구병렬 파일 찾기
3. 해당 문장에 속하는 구병렬 행들 추출
4. 새로운 Gold 데이터로 저장
"""
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import re

def normalize_src(t):
    """공백 제거 정규화"""
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

def main():
    # 1. sentence test 데이터 로드
    print("📂 datasets/sentence/test.csv 로딩...")
    sent_test = pd.read_csv('datasets/sentence/test.csv')
    print(f"  행 수: {len(sent_test)}")
    print(f"  고유 문장: {sent_test['문장식별자'].nunique()}")
    print(f"  컬럼: {sent_test.columns.tolist()}")
    
    # book_name별 그룹화
    books = sent_test['book_name'].unique()
    print(f"  책 수: {len(books)}")
    print(f"  책 목록: {list(books)[:5]}...")
    
    # 2. 각 책의 구병렬 xlsx 파일에서 데이터 수집
    all_phrase_rows = []
    xlsx_base = Path('xlsx')
    
    for book in books:
        book_dir = xlsx_base / book
        phrase_file = book_dir / f"{book}_구병렬.xlsx"
        
        if not phrase_file.exists():
            print(f"  ⚠️ {phrase_file} 없음")
            continue
        
        print(f"  📖 {book} 처리 중...")
        
        # 이 책에서 test에 포함된 문장식별자들
        book_sents = sent_test[sent_test['book_name'] == book]
        target_sent_ids = set(book_sents['문장식별자'].values)
        
        # 구병렬 xlsx 로드
        try:
            phrase_df = pd.read_excel(phrase_file, engine='openpyxl')
        except Exception as e:
            print(f"    ❌ 읽기 실패: {e}")
            continue
        
        # 컬럼 확인
        if '문장식별자' not in phrase_df.columns:
            print(f"    ❌ '문장식별자' 컬럼 없음. 컬럼: {phrase_df.columns.tolist()}")
            continue
        
        # 타겟 문장에 해당하는 행 추출
        matched = phrase_df[phrase_df['문장식별자'].isin(target_sent_ids)].copy()
        matched['book_name'] = book
        
        print(f"    ✅ {len(matched)} 구 행 추출 (from {len(target_sent_ids)} 문장)")
        all_phrase_rows.append(matched)
    
    # 3. 결합 및 저장
    if not all_phrase_rows:
        print("❌ 수집된 데이터 없음!")
        return
    
    result = pd.concat(all_phrase_rows, ignore_index=True)
    print(f"\n📊 최종 수집 결과:")
    print(f"  총 구 행: {len(result)}")
    print(f"  고유 문장: {result['문장식별자'].nunique()}")
    print(f"  컬럼: {result.columns.tolist()}")
    
    # 저장
    output_path = 'datasets/phrase/test_reconstructed.csv'
    result.to_csv(output_path, index=False)
    print(f"\n💾 저장됨: {output_path}")
    
    # Excel도 저장
    output_xlsx = 'datasets/phrase/test_reconstructed.xlsx'
    result.to_excel(output_xlsx, index=False)
    print(f"💾 저장됨: {output_xlsx}")
    
    # 4. 검증: sentence test와 매칭 확인
    print("\n🔍 검증...")
    sent_ids_in_test = set(sent_test['문장식별자'].values)
    sent_ids_in_result = set(result['문장식별자'].values)
    
    coverage = len(sent_ids_in_result) / len(sent_ids_in_test) * 100
    print(f"  sentence test 문장 수: {len(sent_ids_in_test)}")
    print(f"  수집된 문장 수: {len(sent_ids_in_result)}")
    print(f"  커버리지: {coverage:.1f}%")
    
    missing = sent_ids_in_test - sent_ids_in_result
    if missing:
        print(f"  ⚠️ 누락 문장: {len(missing)}개")
        print(f"    샘플: {list(missing)[:10]}")

if __name__ == '__main__':
    main()
