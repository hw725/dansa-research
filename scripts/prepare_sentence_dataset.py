#!/usr/bin/env python3
"""
Sentence 데이터 정규화

datasets/sentence_full.csv → data/sentence_normalized.csv

1. 원문 말미에서 한글 현토 마커 추출
2. normalize_hyeonto로 정규화
3. 斷辭 카테고리 분류 (정확 매칭, 구두해법 원전 기준)
4. 복합형 태그 (인용·존경 중첩)

분류 근거:
  - 任圭直 『句讀解法』 斷辭 6종 (최식 2011, 尹容善 2009)
  - 句讀指南·句讀解法·俚讀解 3문헌 비교 (wiki/종합분석/현토3문헌비교.md)
"""

import sys
from pathlib import Path
import pandas as pd
import regex
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(SCRIPTS_DIR))
from normalize_hyeonto import normalize_hyeonto_marker

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# 斷辭 6종 — 구두해법 원전 기준 정확 매칭
# ============================================================
# 각 斷辭의 대표 마커 (정규화 후 형태). 정확 일치만 해당.
# 복합형(인용·존경 중첩)은 별도 compound_tags로 처리.
DANSA_EXACT = {
    # 游辭以斷: '이로다' → 로다
    '로다': '游辭以斷',
    # 夬絶之斷: '이니라'/'하니라' → 니라/하니라
    '니라': '夬絶之斷',
    '하니라': '夬絶之斷',
    # 微絶之斷: '이라' → 라
    '라': '微絶之斷',
    # 記史之斷: '하다'
    '하다': '記史之斷',
    # 敍述之斷: '이러라'/'더라' → 러라/하더라
    '러라': '敍述之斷',
    '하더라': '敍述之斷',
    # 汎論以斷: '하나니라'/'하나니'
    '하나니라': '汎論以斷',
    '하나니': '汎論以斷',
}

# 인용 프레임 탐지용: 斷辭 base + 하/호 로 시작하는 패턴
# 긴 base부터 매칭하여 '하나니라하' > '나니라하' > '니라하' > '라하' 순서 보장
QUOTE_BASES = sorted(DANSA_EXACT.keys(), key=len, reverse=True)


def extract_marker_from_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = str(text).strip()
    text = regex.sub(r'\([^)]*\)', ' ', text)
    text = text.replace('ㆍ', '')
    text = regex.sub(r'[^\p{Hangul}]+$', '', text)
    match = regex.search(r'[\p{Hangul}]+$', text)
    return match.group(0) if match else ""


def classify_dansa(marker_normalized: str) -> str:
    """斷辭 6종 분류 — 정확 매칭만 사용"""
    if not marker_normalized:
        return ""
    return DANSA_EXACT.get(marker_normalized, "")


def classify_compound_tags(marker_normalized: str) -> str:
    """복합형 태그 — 3문헌 근거, 플랫 태그 나열

    오버레이(인용/존경/自稱/겸양) + 종결(X)/연결을 동등한 태그로 쉼표 구분.
    오버레이가 1개 이상일 때만 태그 생성 (단순형은 빈 문자열).

    근거:
      인용  — 구두해법 #14 引古語及他人之辭
      존경  — 구두해법 존칭 '시'
      自稱  — 구두해법 #8 自稱之辭, 俚讀解 #26 自言己事
      겸양  — 구두해법 #8 범위(願·請·乞→하소서)
    """
    if not marker_normalized:
        return ""

    tags = []

    # 1. 인용 (구두해법 #14)
    has_quote = False
    for base in QUOTE_BASES:
        for bridge in ('하', '호'):
            prefix = base + bridge
            if marker_normalized.startswith(prefix) and len(marker_normalized) > len(prefix):
                tags.append(f"인용내({DANSA_EXACT[base]})")
                has_quote = True
                break
        if has_quote:
            break

    # 2. 존경 (구두해법 존칭)
    if '시' in marker_normalized:
        tags.append("존경")

    # 3. 自稱 (구두해법 #8, 俚讀解 #26)
    if '노' in marker_normalized:
        tags.append("自稱")
    elif not has_quote and marker_normalized.startswith('호니'):
        tags.append("自稱")

    # 4. 겸양 (구두해법 #8 범위)
    if '소이' in marker_normalized or '소니' in marker_normalized:
        tags.append("겸양")
    elif marker_normalized.endswith('이다') and marker_normalized != '하다':
        tags.append("겸양")

    if not tags:
        return ""

    # 종결/연결 (마커 전체 기준, 1회)
    for base in QUOTE_BASES:
        if marker_normalized.endswith(base):
            tags.append(f"종결({DANSA_EXACT[base]})")
            break
    else:
        if marker_normalized.endswith('이다'):
            tags.append("종결")
        else:
            tags.append("연결")

    return ','.join(tags)


def main():
    print("=" * 60)
    print("Sentence 데이터 정규화")
    print("=" * 60)

    src = DATASETS_DIR / "sentence_full.csv"
    if not src.exists():
        print(f"ERROR: {src} not found")
        return False

    print(f"로드: {src}")
    df = pd.read_csv(src, encoding='utf-8')
    print(f"  행 수: {len(df):,}")
    print(f"  컬럼: {list(df.columns)}")

    # 마커 추출
    print("\n마커 추출 중...")
    tqdm.pandas(desc="extract")
    df['marker_raw'] = df['원문'].progress_apply(extract_marker_from_text)

    raw_nonempty = (df['marker_raw'] != '').sum()
    print(f"  마커 있는 행: {raw_nonempty:,} / {len(df):,}")

    # 정규화
    print("\n마커 정규화 중...")
    tqdm.pandas(desc="normalize")
    df['marker_normalized'] = df['marker_raw'].progress_apply(
        lambda x: normalize_hyeonto_marker(x) if x else ''
    )

    # 斷辭 카테고리 (정확 매칭)
    print("\n斷辭 카테고리 분류 (정확 매칭)...")
    df['dansa_category'] = df['marker_normalized'].apply(classify_dansa)

    # 복합형 태그 (인용·존경)
    print("복합형 태그 분류...")
    df['compound_tags'] = df['marker_normalized'].apply(classify_compound_tags)

    # 통계 출력
    cats = df['dansa_category'].value_counts()
    print("\n斷辭 카테고리 분포 (정확 매칭):")
    for cat, cnt in cats.items():
        if cat:
            print(f"  {cat}: {cnt:,}")
    empty_cat = (df['dansa_category'] == '').sum()
    print(f"  (미분류): {empty_cat:,}")

    compound_nonempty = (df['compound_tags'] != '').sum()
    print(f"\n복합형 태그 있음: {compound_nonempty:,}건")
    if compound_nonempty > 0:
        tag_counts = df[df['compound_tags'] != '']['compound_tags'].value_counts()
        for tag, cnt in tag_counts.head(10).items():
            print(f"  {tag}: {cnt:,}")
        if len(tag_counts) > 10:
            print(f"  ... 외 {len(tag_counts) - 10}종")

    # 카테고리별 복합형 비율
    print("\n카테고리별 복합형 비율:")
    for cat in ['夬絶之斷', '記史之斷', '敍述之斷', '游辭以斷', '汎論以斷', '微絶之斷']:
        sub = df[df['dansa_category'] == cat]
        comp = (sub['compound_tags'] != '').sum()
        print(f"  {cat}: {comp:,}/{len(sub):,} ({comp/len(sub)*100:.1f}%)" if len(sub) else f"  {cat}: 0")

    # 汎論以斷 세부
    beomnon = df[df['dansa_category'] == '汎論以斷']
    print(f"\n汎論以斷 상세 ({len(beomnon):,}건):")
    print(beomnon['marker_normalized'].value_counts().head(10).to_string())

    # 저장
    out = DATA_DIR / "sentence_normalized.csv"
    df.to_csv(out, index=False, encoding='utf-8')
    print(f"\n저장: {out} ({len(df):,}행)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
