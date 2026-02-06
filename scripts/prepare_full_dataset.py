#!/usr/bin/env python3
"""
현토 분석용 전체 데이터셋 준비
- datasets/sentence (train+val+test) 43권
- datasets/phrase (train+val+test) 43권
- hyeonto/datasets/*.xml (사서삼경 10권)

총 53권 전체 데이터로 sentence_full.csv, phrase_full.csv 생성
"""
import pandas as pd
import re
from pathlib import Path
from lxml import etree

# 경로 설정
CSP_ROOT = Path(__file__).parent.parent.parent
HYEONTO_DIR = CSP_ROOT / "hyeonto"
DATASETS_DIR = CSP_ROOT / "datasets"
OUTPUT_DIR = HYEONTO_DIR / "datasets"

def extract_hangul_markers(text):
    """원문에서 한글 현토 마커 추출"""
    if pd.isna(text):
        return ''
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ','.join(matches) if matches else ''

def load_43books_data():
    """43권 데이터 로드 (train+val+test)"""
    print("=== 43권 데이터 로드 ===")
    
    # Sentence 데이터
    sentence_files = [
        DATASETS_DIR / "sentence" / "train.csv",
        DATASETS_DIR / "sentence" / "val.csv",
        DATASETS_DIR / "sentence" / "test.csv",
    ]
    sentence_dfs = []
    for f in sentence_files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,}행")
        sentence_dfs.append(df)
    sentence_df = pd.concat(sentence_dfs, ignore_index=True)
    print(f"  Sentence 합계: {len(sentence_df):,}행")
    
    # Phrase 데이터
    phrase_files = [
        DATASETS_DIR / "phrase" / "train.csv",
        DATASETS_DIR / "phrase" / "val.csv",
        DATASETS_DIR / "phrase" / "test.csv",
    ]
    phrase_dfs = []
    for f in phrase_files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,}행")
        phrase_dfs.append(df)
    phrase_df = pd.concat(phrase_dfs, ignore_index=True)
    print(f"  Phrase 합계: {len(phrase_df):,}행")
    
    return sentence_df, phrase_df

def parse_xml_file(xml_path):
    """XML 파일에서 문장 단위 데이터 추출"""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    sentences = []
    for s_elem in root.iter('s'):
        s_id = s_elem.get('id', '')
        text = ''.join(s_elem.itertext()).strip()
        sentences.append({
            'sentence_id': s_id,
            'text': text
        })
    return sentences

def load_saseo_samgyeong():
    """사서삼경 10권 XML에서 데이터 추출"""
    print("\n=== 사서삼경 XML 로드 ===")
    
    xml_dir = OUTPUT_DIR
    xml_files = list(xml_dir.glob("*.xml"))
    
    # 원문/번역문 매칭
    book_data = {}
    for xml_file in xml_files:
        name = xml_file.stem
        if "_원문_" in name:
            book_name = name.split("-")[1].replace("[현토]", "").split("_원문")[0]
            if book_name not in book_data:
                book_data[book_name] = {}
            book_data[book_name]['원문'] = xml_file
        elif "_번역문_" in name:
            book_name = name.split("-")[1].replace("[현토]", "").split("_번역문")[0]
            if book_name not in book_data:
                book_data[book_name] = {}
            book_data[book_name]['번역문'] = xml_file
    
    sentence_rows = []
    for book_name, files in book_data.items():
        if '원문' not in files or '번역문' not in files:
            print(f"  ⚠️ {book_name}: 원문/번역문 쌍 불완전")
            continue
            
        src_sentences = parse_xml_file(files['원문'])
        tgt_sentences = parse_xml_file(files['번역문'])
        
        # sentence_id로 매칭
        tgt_dict = {s['sentence_id']: s['text'] for s in tgt_sentences}
        
        for src in src_sentences:
            sid = src['sentence_id']
            tgt_text = tgt_dict.get(sid, '')
            sentence_rows.append({
                '문단식별자': '',  # XML에서는 문단 구분 없음
                '문장식별자': sid,
                '원문': src['text'],
                '번역문': tgt_text,
                'book': book_name
            })
        
        print(f"  {book_name}: {len(src_sentences):,}문장")
    
    df = pd.DataFrame(sentence_rows)
    print(f"  사서삼경 합계: {len(df):,}행")
    return df

def main():
    print("=" * 60)
    print("현토 분석용 전체 데이터셋 준비")
    print("=" * 60)
    
    # 1. 43권 데이터 로드
    sentence_43, phrase_43 = load_43books_data()
    
    # 2. 사서삼경 데이터 로드
    saseo_df = load_saseo_samgyeong()
    
    # 3. Sentence 데이터 병합 (43권 + 사서삼경)
    print("\n=== Sentence 데이터 병합 ===")
    # 컬럼 통일
    sentence_43_renamed = sentence_43.rename(columns={
        '문단식별자': '문단식별자',
        '문장식별자': '문장식별자', 
        '원문': '원문',
        '번역문': '번역문',
        'book': 'book'
    })
    
    sentence_full = pd.concat([sentence_43_renamed, saseo_df], ignore_index=True)
    print(f"  43권: {len(sentence_43):,}행")
    print(f"  사서삼경: {len(saseo_df):,}행")
    print(f"  합계: {len(sentence_full):,}행")
    
    # 마커 추출
    sentence_full['marker'] = sentence_full['원문'].apply(extract_hangul_markers)
    
    # 저장
    sentence_output = OUTPUT_DIR / "sentence_full.csv"
    sentence_full.to_csv(sentence_output, index=False, encoding='utf-8-sig')
    print(f"  저장: {sentence_output}")
    
    # 4. Phrase 데이터 (43권만, 사서삼경은 구 단위 없음)
    print("\n=== Phrase 데이터 ===")
    phrase_43['marker'] = phrase_43['원문'].apply(extract_hangul_markers)
    phrase_output = OUTPUT_DIR / "phrase_full.csv"
    phrase_43.to_csv(phrase_output, index=False, encoding='utf-8-sig')
    print(f"  43권: {len(phrase_43):,}행")
    print(f"  저장: {phrase_output}")
    
    # 검증
    print("\n=== 검증: 잇고/잇가 보존 ===")
    print(f"  Sentence 잇고: {sentence_full['원문'].astype(str).str.contains('잇고').sum()}")
    print(f"  Sentence 잇가: {sentence_full['원문'].astype(str).str.contains('잇가').sum()}")
    print(f"  Phrase 잇고: {phrase_43['원문'].astype(str).str.contains('잇고').sum()}")
    print(f"  Phrase 잇가: {phrase_43['원문'].astype(str).str.contains('잇가').sum()}")
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()
