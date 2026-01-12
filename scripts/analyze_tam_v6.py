#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 TAM(시제-상-서법) 분석 (v6)

목표:
1. 현토에서 상(Aspect) 마커 식별: -러-, -더-, -던- (회상/완료)
2. 현토에서 서법(Mood) 마커 식별: -리-, -겠- (추측/의지/당위)
3. 클러스터별, 장르별 TAM 분포 분석
4. 번역문 어미와의 상관관계 분석

사용 예:
    python scripts/analyze_tam_v6.py \
        --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
        --out-dir hyeonto/reports/tam_analysis_v6
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

# TAM 마커 정의 (현토에서 발견되는 패턴)
TAM_PATTERNS = {
    # 상 (Aspect) - 사건의 전개 방식
    'retrospective_aspect': {
        'name': '회상상(Retrospective)',
        'description': '과거 경험의 회상, 증거적 태도',
        'patterns': ['러니', '러라', '러', '던', '더니', '더라', '더'],
        'examples': [],
    },
    'perfective_aspect': {
        'name': '완료상(Perfective)',
        'description': '사건의 완결 상태',
        'patterns': ['엇', '앗', '왓', '엿'],
        'examples': [],
    },
    # 서법/양태 (Mood/Modality) - 화자의 태도
    'epistemic_mood': {
        'name': '추측 양태(Epistemic)',
        'description': '불확실성, 추측, 가능성',
        'patterns': ['리오', '리라', '리로다', '리니', '리', '오리오', '오리라'],
        'examples': [],
    },
    'deontic_mood': {
        'name': '당위 양태(Deontic)',
        'description': '의지, 의무, 당위',
        'patterns': ['라', '져라', '거라', '너라'],
        'examples': [],
    },
    'interrogative_mood': {
        'name': '의문 서법(Interrogative)',
        'description': '의문, 수사적 질문',
        'patterns': ['냐', '는가', '는야', '뇨', '료', '잇가', '잇고', '오'],
        'examples': [],
    },
    'declarative_mood': {
        'name': '서술 서법(Declarative)',
        'description': '단정, 서술',
        'patterns': ['니라', '이라', '로다', '도다', '니이다', '이니이다'],
        'examples': [],
    },
}

# 번역문 TAM 패턴 (한국어 어미)
TRANSLATION_TAM_PATTERNS = {
    'past_perfective': ['하였', '였다', '았다', '었다', '했다', '했으'],
    'present_imperfective': ['한다', '이다', '있다', '없다', '는다'],
    'epistemic': ['겠다', '겠는', '리라', '것이다', '있겠'],
    'interrogative': ['는가', '습니까', '니까', '겠는가', '있는가'],
}


def extract_hyeonto_from_src(text: str) -> str:
    """src 텍스트에서 현토(한글) 추출"""
    if pd.isna(text) or not str(text).strip():
        return ''
    
    # 한글만 추출 (한자 뒤의 토씨)
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ''.join(matches) if matches else ''


def classify_tam(hyeonto: str) -> list[tuple[str, str, str]]:
    """현토를 TAM 범주로 분류
    
    Returns:
        list of (category_key, category_name, matched_pattern)
    """
    if not hyeonto:
        return []
    
    results = []
    for cat_key, cat_info in TAM_PATTERNS.items():
        for pattern in cat_info['patterns']:
            if pattern in hyeonto:
                results.append((cat_key, cat_info['name'], pattern))
                break  # 범주당 하나만 매칭
    
    return results


def detect_translation_tam(text: str) -> list[str]:
    """번역문에서 TAM 패턴 탐지"""
    if pd.isna(text) or not str(text).strip():
        return []
    
    text = str(text)
    detected = []
    
    for tam_type, patterns in TRANSLATION_TAM_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                detected.append(tam_type)
                break
    
    return detected


def main() -> int:
    p = argparse.ArgumentParser(description="현토 TAM(시제-상-서법) 분석 (v6)")
    p.add_argument("--csv", type=Path, default=Path("hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/tam_analysis_v6"))
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"📊 데이터: {len(df):,}행")

    # 1단계: 현토에서 TAM 마커 추출
    print("\n🔍 현토 TAM 마커 분석...")
    
    tam_counter = Counter()  # TAM 범주별 카운트
    tam_marker_examples = defaultdict(list)  # 범주별 예시
    tam_by_cluster = defaultdict(lambda: Counter())  # 클러스터별
    tam_by_book = defaultdict(lambda: Counter())  # 장르별
    marker_tam_map = defaultdict(list)  # 현토별 TAM
    
    tam_markers_total = 0
    no_tam_count = 0

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"   진행: {idx:,}/{len(df):,}")

        # src_left, src_right에서 현토 추출
        hyeonto_left = extract_hyeonto_from_src(row.get('src_left', ''))
        hyeonto_right = extract_hyeonto_from_src(row.get('src_right', ''))
        combined_hyeonto = hyeonto_left + hyeonto_right
        
        cluster = row.get('cluster_id', 'unknown')
        book = row.get('book_name', 'unknown')
        
        # TAM 분류
        tam_results = classify_tam(combined_hyeonto)
        
        if tam_results:
            tam_markers_total += 1
            for cat_key, cat_name, matched in tam_results:
                tam_counter[cat_key] += 1
                tam_by_cluster[cluster][cat_key] += 1
                tam_by_book[book][cat_key] += 1
                
                # 예시 수집 (범주당 최대 10개)
                if len(tam_marker_examples[cat_key]) < 10:
                    tam_marker_examples[cat_key].append({
                        'hyeonto': combined_hyeonto,
                        'matched': matched,
                        'book': book,
                        'cluster': cluster,
                    })
                
                # 현토별 TAM 기록
                if combined_hyeonto:
                    marker_tam_map[combined_hyeonto].append(cat_key)
        else:
            no_tam_count += 1

    # 2단계: 통계 출력
    print(f"\n📊 현토 TAM 분석 결과:")
    print(f"   - 전체 경계: {len(df):,}")
    print(f"   - TAM 마커 포함: {tam_markers_total:,} ({tam_markers_total/len(df)*100:.1f}%)")
    print(f"   - TAM 마커 없음: {no_tam_count:,} ({no_tam_count/len(df)*100:.1f}%)")
    
    print(f"\n📊 TAM 범주별 분포:")
    for cat_key, count in tam_counter.most_common():
        cat_name = TAM_PATTERNS[cat_key]['name']
        ratio = count / len(df) * 100
        print(f"   {cat_name}: {count:,} ({ratio:.1f}%)")

    # 3단계: 결과 저장
    print("\n💾 결과 저장...")

    # 3-1. TAM 범주별 통계 CSV
    tam_stats = []
    for cat_key, cat_info in TAM_PATTERNS.items():
        count = tam_counter.get(cat_key, 0)
        tam_stats.append({
            '범주키': cat_key,
            '범주명': cat_info['name'],
            '설명': cat_info['description'],
            '건수': count,
            '비율': count / len(df) * 100,
            '패턴': ', '.join(cat_info['patterns'][:5]),
        })
    
    tam_stats_df = pd.DataFrame(tam_stats)
    tam_stats_csv = args.out_dir / "tam_category_stats.csv"
    tam_stats_df.to_csv(tam_stats_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {tam_stats_csv}")

    # 3-2. 클러스터별 TAM 분포 CSV
    cluster_tam_records = []
    for cluster, tam_counts in tam_by_cluster.items():
        total = sum(tam_counts.values())
        for cat_key, count in tam_counts.items():
            cluster_tam_records.append({
                '클러스터': cluster,
                'TAM범주': TAM_PATTERNS[cat_key]['name'],
                '건수': count,
                '비율': count / total * 100 if total > 0 else 0,
            })
    
    cluster_tam_df = pd.DataFrame(cluster_tam_records)
    cluster_csv = args.out_dir / "tam_by_cluster.csv"
    cluster_tam_df.to_csv(cluster_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {cluster_csv}")

    # 3-3. 장르별 TAM 분포 CSV
    book_tam_records = []
    for book, tam_counts in tam_by_book.items():
        total = sum(tam_counts.values())
        for cat_key, count in tam_counts.items():
            book_tam_records.append({
                '도서명': book,
                'TAM범주': TAM_PATTERNS[cat_key]['name'],
                '건수': count,
                '비율': count / total * 100 if total > 0 else 0,
            })
    
    book_tam_df = pd.DataFrame(book_tam_records)
    book_csv = args.out_dir / "tam_by_book.csv"
    book_tam_df.to_csv(book_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {book_csv}")

    # 3-4. 현토별 TAM 목록 CSV
    marker_records = []
    for hyeonto, tam_list in marker_tam_map.items():
        if len(hyeonto) >= 2:  # 의미 있는 현토만
            marker_records.append({
                '현토': hyeonto,
                '빈도': len(tam_list),
                'TAM범주': ', '.join(set(tam_list)),
                '주요범주': max(set(tam_list), key=tam_list.count) if tam_list else '',
            })
    
    marker_df = pd.DataFrame(marker_records)
    marker_df = marker_df.sort_values('빈도', ascending=False)
    marker_csv = args.out_dir / "marker_tam_classification.csv"
    marker_df.to_csv(marker_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {marker_csv}")

    # 3-5. 마크다운 리포트
    report_md = args.out_dir / "TAM_ANALYSIS_REPORT.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# 현토 TAM(시제-상-서법) 분석 보고서 (v6)\n\n")
        f.write(f"**분석일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**데이터**: {args.csv.name} ({len(df):,}건)\n\n")

        f.write("---\n\n")
        f.write("## 1. 분석 목적\n\n")
        f.write("현토(懸吐)의 문법 마커를 **시제(Tense)**가 아닌 **TAM(Tense-Aspect-Mood)** 프레임워크로 분류합니다.\n\n")
        f.write("> **언어학적 근거**: 한국어의 `-러-`, `-리-` 등은 순수 시제가 아니라 상(Aspect) 또는 서법(Mood)에 해당합니다.\n\n")

        f.write("## 2. TAM 범주 정의\n\n")
        f.write("### 2.1 상 (Aspect) - 사건 전개 방식\n\n")
        f.write("| 범주 | 설명 | 패턴 예시 |\n")
        f.write("|------|------|----------|\n")
        f.write("| **회상상(Retrospective)** | 과거 경험 회상, 증거적 태도 | `-러니`, `-더니`, `-러라` |\n")
        f.write("| **완료상(Perfective)** | 사건의 완결 상태 | `-엇`, `-앗`, `-왓` |\n\n")
        
        f.write("### 2.2 서법/양태 (Mood/Modality) - 화자 태도\n\n")
        f.write("| 범주 | 설명 | 패턴 예시 |\n")
        f.write("|------|------|----------|\n")
        f.write("| **추측 양태(Epistemic)** | 불확실성, 추측, 가능성 | `-리오`, `-리라`, `-오리오` |\n")
        f.write("| **당위 양태(Deontic)** | 의지, 의무, 당위 | `-라`, `-져라`, `-거라` |\n")
        f.write("| **의문 서법(Interrogative)** | 의문, 수사적 질문 | `-냐`, `-는가`, `-잇가` |\n")
        f.write("| **서술 서법(Declarative)** | 단정, 서술 | `-니라`, `-이라`, `-로다` |\n\n")

        f.write("---\n\n")
        f.write("## 3. 분석 결과\n\n")
        f.write("### 3.1 전체 TAM 분포\n\n")
        f.write(f"- **전체 경계**: {len(df):,}건\n")
        f.write(f"- **TAM 마커 포함**: {tam_markers_total:,}건 ({tam_markers_total/len(df)*100:.1f}%)\n")
        f.write(f"- **TAM 마커 없음**: {no_tam_count:,}건 ({no_tam_count/len(df)*100:.1f}%)\n\n")

        f.write("### 3.2 TAM 범주별 빈도\n\n")
        f.write("| 범주 | 건수 | 비율 |\n")
        f.write("|------|------|------|\n")
        for cat_key, count in tam_counter.most_common():
            cat_name = TAM_PATTERNS[cat_key]['name']
            ratio = count / len(df) * 100
            f.write(f"| {cat_name} | {count:,} | {ratio:.2f}% |\n")

        f.write("\n### 3.3 주요 발견\n\n")
        
        # 상 vs 서법 비율
        aspect_count = tam_counter.get('retrospective_aspect', 0) + tam_counter.get('perfective_aspect', 0)
        mood_count = sum(tam_counter.get(k, 0) for k in ['epistemic_mood', 'deontic_mood', 'interrogative_mood', 'declarative_mood'])
        
        f.write(f"**1. 상(Aspect) vs 서법(Mood) 비율:**\n")
        f.write(f"   - 상(Aspect): {aspect_count:,}건 ({aspect_count/len(df)*100:.1f}%)\n")
        f.write(f"   - 서법(Mood): {mood_count:,}건 ({mood_count/len(df)*100:.1f}%)\n\n")
        
        if mood_count > aspect_count:
            f.write("   → **서법(Mood)이 상(Aspect)보다 압도적으로 많음**\n\n")
        else:
            f.write("   → **상(Aspect)이 서법(Mood)보다 많거나 유사함**\n\n")

        f.write("**2. 핵심 해석:**\n")
        f.write("   - 현토는 **시제(Tense)**보다 **서법(Mood)**이 주를 이룸\n")
        f.write("   - `-리오`, `-니라` 등은 과거/미래가 아닌 **추측/단정** 기능\n")
        f.write("   - 시간적 정보는 한문 구조와 번역 맥락에서 추론됨\n\n")

        f.write("### 3.4 TAM 마커 예시 (범주별 Top 5)\n\n")
        for cat_key, examples in tam_marker_examples.items():
            if examples:
                cat_name = TAM_PATTERNS[cat_key]['name']
                f.write(f"**{cat_name}**:\n")
                for i, ex in enumerate(examples[:5], 1):
                    f.write(f"   {i}. `{ex['hyeonto']}` (패턴: `{ex['matched']}`, 도서: {ex['book']})\n")
                f.write("\n")

        f.write("---\n\n")
        f.write("## 4. 장르별 TAM 경향\n\n")
        f.write("| 도서 | 회상상 | 추측양태 | 의문서법 | 서술서법 |\n")
        f.write("|------|--------|----------|----------|----------|\n")
        
        # 상위 10개 도서
        book_totals = [(b, sum(tc.values())) for b, tc in tam_by_book.items()]
        for book, _ in sorted(book_totals, key=lambda x: x[1], reverse=True)[:10]:
            tc = tam_by_book[book]
            retro = tc.get('retrospective_aspect', 0)
            epist = tc.get('epistemic_mood', 0)
            interr = tc.get('interrogative_mood', 0)
            decl = tc.get('declarative_mood', 0)
            f.write(f"| {book[:15]} | {retro} | {epist} | {interr} | {decl} |\n")

        f.write("\n---\n\n")
        f.write("## 5. 결론\n\n")
        f.write("1. **현토의 TAM 구조**: 시제가 아닌 서법(Mood)이 핵심\n")
        f.write("2. **-러- 계열**: 회상상(Retrospective Aspect) - 과거 '시제'가 아닌 과거 '경험 회상'\n")
        f.write("3. **-리- 계열**: 추측 양태(Epistemic Mood) - 미래 '시제'가 아닌 '가능성/추측'\n")
        f.write("4. **화용론적 시간**: 절대적 시제는 한문 맥락과 번역 과정에서 결정됨\n\n")

        f.write(f"**생성 파일**: {args.out_dir}\n")

    print(f"  ✅ {report_md}")

    # 3-6. JSON 요약
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'data_file': str(args.csv),
        'data_rows': int(len(df)),
        'tam_markers_found': int(tam_markers_total),
        'tam_markers_ratio': float(tam_markers_total / len(df) * 100),
        'aspect_count': int(aspect_count),
        'mood_count': int(mood_count),
        'category_distribution': {k: int(v) for k, v in tam_counter.items()},
    }
    
    summary_json = args.out_dir / "tam_analysis_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {summary_json}")

    print(f"\n✅ 현토 TAM 분석 완료!")
    print(f"\n📊 핵심 결과:")
    print(f"   - 상(Aspect): {aspect_count:,}건 ({aspect_count/len(df)*100:.1f}%)")
    print(f"   - 서법(Mood): {mood_count:,}건 ({mood_count/len(df)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())
