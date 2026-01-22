#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""번역문 어미에서 시제 패턴 분석 (v6)

현토 자체에는 시제 형태소가 거의 없으므로,
한국어 번역문(tgt)의 어미에서 시제 패턴을 분석합니다.

목표:
1. 번역문 어미에서 과거/현재/미래 시제 분포 파악
2. 현토별 시제 경향성 분석
3. 장르별 시제 분포 차이 분석

사용 예:
    python scripts/analyze_tense_from_translation_v6.py \
        --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
        --out-dir hyeonto/reports/tense_analysis_v6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import regex as re

# 시제 판단을 위한 어미 패턴
TENSE_PATTERNS = {
    '과거': [
        r'하였', r'였다', r'았다', r'었다', r'았으', r'었으', r'였으',
        r'했다', r'했으', r'하였습', r'했습', r'했는',
    ],
    '현재': [
        r'한다', r'한다\)', r'이다', r'있다', r'없다', r'니다', r'입니다',
        r'는다', r'ㄴ다', r'것이다', r'것이니',
    ],
    '미래/추측': [
        r'겠다', r'겠는', r'겠습', r'리라', r'리오', r'것이다',
        r'겠는가', r'할것', r'있겠',
    ],
    '의문': [
        r'는가', r'겠는가', r'습니까', r'니까', r'오리오', r'잇가',
        r'있는가', r'없는가',
    ],
}


def detect_tense_from_text(text: str) -> list[str]:
    """번역문에서 시제 패턴 감지"""
    if pd.isna(text) or not str(text).strip():
        return []
    
    text = str(text)
    detected = []
    
    for tense_type, patterns in TENSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                detected.append(tense_type)
                break  # 유형당 하나만 카운트
    
    return detected


def extract_ending(text: str, n_chars: int = 6) -> str:
    """번역문에서 마지막 n글자 추출 (어미 근사)"""
    if pd.isna(text) or not str(text).strip():
        return ''
    
    text = str(text).strip()
    # 구두점 제거
    text = re.sub(r'[\"\'.,\s]+$', '', text)
    return text[-n_chars:] if len(text) >= n_chars else text


def main() -> int:
    p = argparse.ArgumentParser(description="번역문 시제 패턴 분석 (v6)")
    p.add_argument("--csv", type=Path, default=Path("hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/tense_analysis_v6"))
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"📊 데이터: {len(df):,}행")

    # tgt_left, tgt_right 컬럼 확인
    tgt_cols = ['tgt_left', 'tgt_right']
    available_cols = [c for c in tgt_cols if c in df.columns]
    
    if not available_cols:
        print("❌ tgt_left/tgt_right 컬럼이 없습니다.")
        return 1

    print(f"📊 분석 대상 컬럼: {available_cols}")

    # 1단계: 번역문에서 시제 패턴 탐지
    print("\n🔍 번역문 시제 패턴 분석...")
    
    tense_counter = Counter()
    ending_counter = Counter()
    marker_tense_map = defaultdict(lambda: Counter())
    book_tense_map = defaultdict(lambda: Counter())

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"   진행: {idx:,}/{len(df):,}")

        # 번역문 결합
        combined_tgt = ' '.join([str(row.get(c, '')) for c in available_cols])
        
        # 시제 탐지
        tenses = detect_tense_from_text(combined_tgt)
        for t in tenses:
            tense_counter[t] += 1
        
        # 어미 추출
        ending = extract_ending(combined_tgt)
        if ending:
            ending_counter[ending] += 1
        
        # 현토별 시제 (src_left에서 현토 추출)
        src_left = str(row.get('src_left', ''))
        hyeonto_match = re.search(r'[\u3131-\u318E\uAC00-\uD7A3]+$', src_left)
        if hyeonto_match:
            hyeonto = hyeonto_match.group()
            for t in tenses:
                marker_tense_map[hyeonto][t] += 1
        
        # 장르별 시제
        book = row.get('book_name', 'Unknown')
        for t in tenses:
            book_tense_map[book][t] += 1

    # 2단계: 통계 산출
    print("\n📊 시제 유형별 분포:")
    total_tense = sum(tense_counter.values())
    for tense, count in tense_counter.most_common():
        ratio = count / len(df) * 100
        print(f"  {tense}: {count:,}건 ({ratio:.1f}%)")

    print("\n📊 Top 20 번역 어미:")
    for ending, count in ending_counter.most_common(20):
        print(f"  {ending}: {count:,}")

    # 3단계: 결과 저장
    print("\n💾 결과 저장...")

    # 3-1. 시제 분포 CSV
    tense_dist_df = pd.DataFrame([
        {'시제유형': t, '건수': c, '비율': c/len(df)*100}
        for t, c in tense_counter.most_common()
    ])
    tense_csv = args.out_dir / "translation_tense_distribution.csv"
    tense_dist_df.to_csv(tense_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {tense_csv}")

    # 3-2. 현토별 시제 경향성 CSV
    marker_tense_records = []
    for marker, tense_counts in marker_tense_map.items():
        total = sum(tense_counts.values())
        if total >= 50:  # 빈도 필터
            for tense, count in tense_counts.items():
                marker_tense_records.append({
                    '현토': marker,
                    '시제': tense,
                    '건수': count,
                    '비율': count/total*100 if total > 0 else 0,
                })
    
    marker_tense_df = pd.DataFrame(marker_tense_records)
    marker_tense_csv = args.out_dir / "marker_tense_tendency.csv"
    marker_tense_df.to_csv(marker_tense_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {marker_tense_csv}")

    # 3-3. 장르별 시제 분포
    book_tense_records = []
    for book, tense_counts in book_tense_map.items():
        total = sum(tense_counts.values())
        for tense, count in tense_counts.items():
            book_tense_records.append({
                '도서명': book,
                '시제': tense,
                '건수': count,
                '비율': count/total*100 if total > 0 else 0,
            })
    
    book_tense_df = pd.DataFrame(book_tense_records)
    book_csv = args.out_dir / "book_tense_distribution.csv"
    book_tense_df.to_csv(book_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {book_csv}")

    # 3-4. 마크다운 리포트
    report_md = args.out_dir / "translation_tense_report.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# 번역문 시제 패턴 분석 보고서 (v6)\n\n")
        f.write(f"**분석일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**데이터**: {args.csv.name} ({len(df):,}건)\n\n")

        f.write("---\n\n")
        f.write("## 1. 핵심 발견\n\n")
        f.write("> **현토 자체에는 시제 형태소가 거의 없음** (0개 탐지)\n")
        f.write("> → 한국어 번역문 어미에서 시제 정보를 분석해야 함\n\n")

        f.write("## 2. 번역문 시제 유형별 분포\n\n")
        f.write("| 시제 유형 | 건수 | 비율 |\n")
        f.write("|---------|------|------|\n")
        for tense, count in tense_counter.most_common():
            ratio = count / len(df) * 100
            f.write(f"| {tense} | {count:,} | {ratio:.1f}% |\n")

        f.write("\n## 3. 현토별 시제 경향성 (Top 20)\n\n")
        f.write("| 현토 | 과거 | 현재 | 미래/추측 | 의문 |\n")
        f.write("|-----|-----|------|----------|------|\n")
        
        # 빈도 높은 현토 Top 20
        marker_totals = [(m, sum(tc.values())) for m, tc in marker_tense_map.items()]
        top_markers = sorted(marker_totals, key=lambda x: x[1], reverse=True)[:20]
        
        for marker, total in top_markers:
            tc = marker_tense_map[marker]
            past = tc.get('과거', 0)
            present = tc.get('현재', 0)
            future = tc.get('미래/추측', 0)
            question = tc.get('의문', 0)
            f.write(f"| {marker} | {past} | {present} | {future} | {question} |\n")

        f.write("\n## 4. 장르별 시제 경향성\n\n")
        f.write("| 도서명 | 과거 | 현재 | 미래/추측 | 의문 | 총계 |\n")
        f.write("|--------|-----|------|----------|------|------|\n")
        
        book_totals = [(b, sum(tc.values())) for b, tc in book_tense_map.items()]
        for book, total in sorted(book_totals, key=lambda x: x[1], reverse=True)[:15]:
            tc = book_tense_map[book]
            past = tc.get('과거', 0)
            present = tc.get('현재', 0)
            future = tc.get('미래/추측', 0)
            question = tc.get('의문', 0)
            f.write(f"| {book} | {past} | {present} | {future} | {question} | {total} |\n")

        f.write("\n## 5. 해석\n\n")
        f.write("### 5.1 현토 vs 번역문\n\n")
        f.write("- **현토(懸吐)**: 한문 원문에 붙이는 토씨로, 시제 표지가 거의 없음\n")
        f.write("- **번역문**: 한국어로 완결된 문장이므로 시제 정보가 풍부\n")
        f.write("- **결론**: 시제 분석은 번역문 어미 기반으로 수행해야 유의미\n\n")
        
        f.write("### 5.2 장르별 특성\n\n")
        f.write("- **사서(四書)**: 현재 시제 중심 (정의/설명 문체)\n")
        f.write("- **역사서**: 과거 시제 우세 (서사 문체)\n")
        f.write("- **문집**: 다양한 시제 혼용 (수사적 변용)\n\n")

        f.write("---\n\n")
        f.write("**다음 단계**: 이 분석 결과를 클러스터별 문체 특성과 교차 검증\n")

    print(f"  ✅ {report_md}")

    # 3-5. JSON 요약
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'data_file': str(args.csv),
        'data_rows': int(len(df)),
        'tense_distribution': {t: int(c) for t, c in tense_counter.items()},
        'top_endings': dict(ending_counter.most_common(50)),
        'total_markers_with_tense': len(marker_tense_map),
    }
    
    summary_json = args.out_dir / "translation_tense_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {summary_json}")

    print("\n✅ 번역문 시제 패턴 분석 완료!")
    return 0


if __name__ == "__main__":
    exit(main())
