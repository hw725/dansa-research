#!/usr/bin/env python3
"""
S2P Gold 재구축 v2
- datasets/sentence/test.csv의 각 문장에 대해
- 해당 문장의 원문과 일치하는 구병렬 행들을 xlsx에서 찾음
- 원문 텍스트 매칭 기반 (문장식별자가 아닌)
"""
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

def normalize(t):
    """공백/개행 제거 정규화"""
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

def main():
    print("📂 데이터 로딩...")
    sent_test = pd.read_csv('datasets/sentence/test.csv')
    print(f"  sentence test: {len(sent_test)} 행, {sent_test['문장식별자'].nunique()} 문장")
    
    books = sent_test['book_name'].unique()
    print(f"  책 수: {len(books)}")
    
    all_results = []
    match_stats = {'total': 0, 'matched': 0, 'partial': 0, 'missed': 0}
    
    for book in books:
        print(f"\n📖 {book} 처리 중...")
        
        # 이 책의 test 문장들
        book_sents = sent_test[sent_test['book_name'] == book]
        
        # 구병렬 xlsx 로드
        phrase_file = Path(f'xlsx/{book}/{book}_구병렬.xlsx')
        if not phrase_file.exists():
            print(f"  ⚠️ 파일 없음: {phrase_file}")
            continue
            
        phrase_df = pd.read_excel(phrase_file)
        
        # 구병렬에서 문장식별자별로 원문 합산 인덱스 생성
        # (문장식별자 → 전체 원문, 구 리스트)
        phrase_by_sent_id = defaultdict(lambda: {'full_src': '', 'rows': []})
        for idx, row in phrase_df.iterrows():
            sid = row['문장식별자']
            phrase_by_sent_id[sid]['full_src'] += normalize(row['원문'])
            phrase_by_sent_id[sid]['rows'].append(row)
        
        # 원문 → 문장식별자 역매핑
        src_to_phrase_sid = {}
        for sid, data in phrase_by_sent_id.items():
            src_to_phrase_sid[data['full_src']] = sid
        
        # 각 test 문장에 대해 구병렬 매칭
        for _, sent_row in book_sents.iterrows():
            match_stats['total'] += 1
            
            sent_id = sent_row['문장식별자']
            sent_src = normalize(sent_row['원문'])
            
            # 완전 일치 찾기
            if sent_src in src_to_phrase_sid:
                phrase_sid = src_to_phrase_sid[sent_src]
                phrase_rows = phrase_by_sent_id[phrase_sid]['rows']
                
                for prow in phrase_rows:
                    all_results.append({
                        '문장식별자': sent_id,  # test의 문장식별자 사용
                        '구식별자': prow['구식별자'],
                        '원문': prow['원문'],
                        '번역문': prow['번역문'],
                        'book_name': book
                    })
                match_stats['matched'] += 1
            else:
                # 부분 일치 시도 (sent_src가 phrase 원문의 일부인 경우)
                found = False
                for phrase_src, phrase_sid in src_to_phrase_sid.items():
                    if sent_src in phrase_src or phrase_src.startswith(sent_src):
                        # 부분 매칭 - 전체 phrase 사용은 위험, 스킵
                        match_stats['partial'] += 1
                        found = True
                        break
                
                if not found:
                    match_stats['missed'] += 1
    
    # 결과 저장
    if all_results:
        result_df = pd.DataFrame(all_results)
        output_path = 'datasets/phrase/test_gold_v2.csv'
        result_df.to_csv(output_path, index=False)
        print(f"\n💾 저장됨: {output_path}")
        print(f"  총 구 행: {len(result_df)}")
        print(f"  고유 문장: {result_df['문장식별자'].nunique()}")
    
    print(f"\n📊 매칭 통계:")
    print(f"  총 문장: {match_stats['total']}")
    print(f"  완전 매칭: {match_stats['matched']} ({100*match_stats['matched']/match_stats['total']:.1f}%)")
    print(f"  부분 매칭: {match_stats['partial']}")
    print(f"  실패: {match_stats['missed']}")

if __name__ == '__main__':
    main()
