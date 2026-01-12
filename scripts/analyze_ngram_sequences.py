#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 n-gram 시퀀스 분석 스크립트

문장/구 내에서 현토 마커들이 어떤 순서로 연쇄되는지 분석합니다.
장르별로 특징적인 시퀀스 패턴을 식별합니다.

출력:
- ngram_frequency.csv: n-gram 빈도표
- ngram_by_genre.csv: 장르별 n-gram 분포
- ngram_analysis.md: 분석 리포트
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 현토 마커 추출 정규식 (한글 어미 패턴)
MARKER_PATTERN = re.compile(r"[가-힣]+(?:라|니|하니|하여|하고|하야|하다|는|은|를|을|에|로|면|요|니라|이라|잇고|잇가|시니|러니|되|대)")


def extract_markers_from_text(text: str) -> List[str]:
    """텍스트에서 현토 마커를 순서대로 추출"""
    if not text or pd.isna(text):
        return []
    # 간단한 접근: 알려진 주요 마커들을 순서대로 추출
    # 실제 구현에서는 원문-번역 정렬 정보를 활용할 수 있음
    markers = MARKER_PATTERN.findall(str(text))
    return markers


def extract_markers_from_boundary(src_left: str, src_right: str) -> List[str]:
    """경계 양쪽 문장에서 마커 추출"""
    left_markers = extract_markers_from_text(src_left)
    right_markers = extract_markers_from_text(src_right)
    return left_markers + right_markers


def generate_ngrams(markers: List[str], n: int) -> List[Tuple[str, ...]]:
    """마커 리스트에서 n-gram 생성"""
    if len(markers) < n:
        return []
    return [tuple(markers[i:i+n]) for i in range(len(markers) - n + 1)]


def load_cluster_data(path: Path) -> pd.DataFrame:
    """클러스터 데이터 로드"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def analyze_ngrams(df: pd.DataFrame, n_values: List[int] = [2, 3]) -> Dict:
    """n-gram 분석 수행"""
    results = {}
    
    for n in n_values:
        print(f"  {n}-gram 분석 중...")
        
        # 전체 n-gram 빈도
        all_ngrams = Counter()
        # 장르별 n-gram 빈도
        genre_ngrams = defaultdict(Counter)
        # 클러스터별 n-gram 빈도
        cluster_ngrams = defaultdict(Counter)
        
        for _, row in df.iterrows():
            src_left = str(row.get("src_left", ""))
            src_right = str(row.get("src_right", ""))
            book = row.get("book_name", "unknown")
            cluster = row.get("cluster_id", -1)
            
            markers = extract_markers_from_boundary(src_left, src_right)
            ngrams = generate_ngrams(markers, n)
            
            for ng in ngrams:
                ng_str = "→".join(ng)
                all_ngrams[ng_str] += 1
                genre_ngrams[book][ng_str] += 1
                cluster_ngrams[cluster][ng_str] += 1
        
        results[n] = {
            "all": all_ngrams,
            "by_genre": dict(genre_ngrams),
            "by_cluster": dict(cluster_ngrams),
        }
    
    return results


def compute_genre_specificity(ngram_results: Dict, genre: str, n: int) -> List[Tuple[str, float, int]]:
    """특정 장르에서 과대표된 n-gram 계산"""
    if n not in ngram_results:
        return []
    
    all_ngrams = ngram_results[n]["all"]
    genre_ngrams = ngram_results[n]["by_genre"].get(genre, Counter())
    
    if not genre_ngrams:
        return []
    
    total_all = sum(all_ngrams.values())
    total_genre = sum(genre_ngrams.values())
    
    if total_all == 0 or total_genre == 0:
        return []
    
    specificity = []
    for ng, cnt in genre_ngrams.items():
        expected = (all_ngrams[ng] / total_all) * total_genre
        if expected > 0:
            ratio = cnt / expected
            specificity.append((ng, ratio, cnt))
    
    specificity.sort(key=lambda x: x[1], reverse=True)
    return specificity[:20]


def write_ngram_report(
    out_dir: Path,
    ngram_results: Dict,
    df: pd.DataFrame,
    analysis_type: str
) -> None:
    """n-gram 분석 리포트 작성"""
    
    # 1. 전체 빈도 CSV
    for n, data in ngram_results.items():
        freq_data = [{"ngram": k, "count": v} for k, v in data["all"].most_common(500)]
        pd.DataFrame(freq_data).to_csv(
            out_dir / f"{n}gram_frequency_{analysis_type.lower()}.csv",
            index=False, encoding="utf-8-sig"
        )
    
    # 2. 마크다운 리포트
    lines = [
        f"# {analysis_type} 현토 n-gram 시퀀스 분석 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**분석 데이터**: {len(df)}건",
        "",
        "---",
        "",
    ]
    
    for n in sorted(ngram_results.keys()):
        data = ngram_results[n]
        lines.append(f"## {n}-gram 분석")
        lines.append("")
        
        # 상위 30개 빈출 패턴
        lines.append(f"### 상위 30개 빈출 {n}-gram")
        lines.append("")
        lines.append("| 순위 | 패턴 | 빈도 |")
        lines.append("|:---:|:---|---:|")
        for i, (ng, cnt) in enumerate(data["all"].most_common(30), 1):
            lines.append(f"| {i} | `{ng}` | {cnt:,} |")
        lines.append("")
        
        # 사서 특화 패턴
        canon_books = ["논어집주", "맹자집주", "대학장구", "중용장구"]
        canon_ngrams = Counter()
        for book in canon_books:
            if book in data["by_genre"]:
                canon_ngrams.update(data["by_genre"][book])
        
        if canon_ngrams:
            lines.append(f"### 사서 특화 {n}-gram (상위 20개)")
            lines.append("")
            lines.append("| 순위 | 패턴 | 사서 빈도 | 전체 빈도 | 집중도 |")
            lines.append("|:---:|:---|---:|---:|---:|")
            
            total_all = sum(data["all"].values())
            total_canon = sum(canon_ngrams.values())
            
            specificity = []
            for ng, cnt in canon_ngrams.items():
                if data["all"][ng] > 0:
                    expected = (data["all"][ng] / total_all) * total_canon if total_all > 0 else 0
                    ratio = cnt / expected if expected > 0 else 0
                    specificity.append((ng, cnt, data["all"][ng], ratio))
            
            specificity.sort(key=lambda x: x[3], reverse=True)
            for i, (ng, canon_cnt, all_cnt, ratio) in enumerate(specificity[:20], 1):
                lines.append(f"| {i} | `{ng}` | {canon_cnt:,} | {all_cnt:,} | {ratio:.2f}x |")
            lines.append("")
        
        # 클러스터별 대표 패턴
        lines.append(f"### 클러스터별 대표 {n}-gram")
        lines.append("")
        lines.append("| 클러스터 | 대표 패턴 (상위 3개) |")
        lines.append("|:---:|:---|")
        
        for cluster in sorted(data["by_cluster"].keys()):
            cluster_data = data["by_cluster"][cluster]
            top3 = [f"`{ng}`({cnt})" for ng, cnt in Counter(cluster_data).most_common(3)]
            lines.append(f"| p{cluster} | {', '.join(top3)} |")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 핵심 발견 섹션
    lines.extend([
        "## 핵심 발견",
        "",
        "### 1. 사서 시그니처 시퀀스",
        "",
        "사서(논어, 맹자, 대학, 중용)에서 과대표된 n-gram 패턴들:",
        "",
    ])
    
    if 2 in ngram_results:
        lines.append("**2-gram 시그니처:**")
        for ng, ratio, cnt in compute_genre_specificity(ngram_results, "논어집주", 2)[:5]:
            lines.append(f"- `{ng}` (집중도: {ratio:.2f}x, n={cnt})")
        lines.append("")
    
    lines.extend([
        "### 2. 장르 분화 패턴",
        "",
        "역사서와 문집에서 특징적으로 나타나는 시퀀스가 사서와 어떻게 다른지 확인 필요.",
        "",
        "---",
        "",
        "**분석 완료**",
    ])
    
    (out_dir / f"ngram_analysis_{analysis_type.lower()}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="현토 n-gram 시퀀스 분석")
    p.add_argument("--input", type=str, required=True, help="입력 CSV (boundary_clusters.csv)")
    p.add_argument("--out-dir", type=str, required=True, help="출력 디렉토리")
    p.add_argument("--analysis-type", type=str, default="SA", help="분석 타입 (PA/SA)")
    p.add_argument("--n-values", type=str, default="2,3", help="분석할 n 값들 (콤마 구분)")
    
    args = p.parse_args()
    
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    
    print(f"[1/3] 데이터 로딩: {in_path}")
    df = load_cluster_data(in_path)
    print(f"  -> {len(df)}건 로드")
    
    print(f"[2/3] n-gram 분석 (n={n_values})...")
    results = analyze_ngrams(df, n_values)
    
    print(f"[3/3] 리포트 작성...")
    write_ngram_report(out_dir, results, df, args.analysis_type)
    
    print(f"완료: {out_dir}")
    
    # 요약 출력
    for n in n_values:
        print(f"\n=== {n}-gram 요약 ===")
        top5 = results[n]["all"].most_common(5)
        for ng, cnt in top5:
            print(f"  {ng}: {cnt:,}")


if __name__ == "__main__":
    main()
