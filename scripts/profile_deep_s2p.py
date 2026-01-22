#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SA 심층 프로파일링 (Deep Profiler)

K=24 등 세분화된 클러스터에 대해 심층적인 통계와 특성을 분석합니다.
- 사서 비율(Canonicity) 상세 분포
- 도서별 특이도 (Lift Analysis)
- 마커 다양성 (Entropy)
- 구문 기능 추정 (Syntactic Guessing)
- 대표 예문 추출
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

# --- 상수 정의 ---
CANON_BOOKS = ["논어", "맹자", "대학", "중용"]

# 구문 기능 매핑 (간소화)
SYNTACTIC_MAP = {
    '면': 'Conditional (조건)', '거든': 'Conditional (조건)', '니': 'Causal/Cond (하니)',
    '라': 'Declarative (평서)', '다': 'Declarative (평서)', '니라': 'Declarative (평서)', '이라': 'Declarative (이다)',
    '는': 'Topic (주제/대조)', '은': 'Topic (주제/대조)',
    '가': 'Subject (주어)', '이': 'Subject (주어)',
    '고': 'Connective (나열)', '며': 'Connective (나열)',
    '을': 'Object (목적)', '를': 'Object (목적)',
    '야': 'Vocative (호격)', '여': 'Vocative (호격)',
    '요': 'Polite/Conn (연결/종결)',
    '나': 'Adversative (역접)',
    '한대': 'Narrative (설명)', '어늘': 'Narrative (설명)', '거늘': 'Narrative (설명)',
    '러니': 'Retrospective (회상)',
    '하야': 'Connective (하여)', '하여': 'Connective (하여)',
    '하고': 'Connective (하고)',
}

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def extract_marker(text: str) -> str:
    """한글 접미사(현토) 추출"""
    text = str(text)
    # 한글만 추출 (마지막 연속된 한글)
    match = re.search(r'([가-힣]+)$', text)
    if match:
        return match.group(1)
    return ""

def calculate_lift(df: pd.DataFrame, cluster_df: pd.DataFrame, min_count: int = 10) -> list:
    """도서별 Lift 계산 (Top 5)"""
    global_counts = df['book_name'].value_counts()
    global_probs = global_counts / len(df)
    
    local_counts = cluster_df['book_name'].value_counts()
    local_probs = local_counts / len(cluster_df)
    
    lifts = []
    for book, l_prob in local_probs.items():
        if local_counts[book] < min_count:
            continue
        g_prob = global_probs.get(book, 0)
        if g_prob > 0:
            lift = l_prob / g_prob
            lifts.append((book, lift, local_counts[book]))
            
    return sorted(lifts, key=lambda x: x[1], reverse=True)[:5]

def get_syntactic_guess(markers: dict) -> str:
    """주요 마커를 기반으로 구문 기능 추정"""
    func_counts = defaultdict(float)
    total_w = sum(markers.values())
    if total_w == 0: return "Unknown"
    
    for m, count in markers.items():
        suffix = ""
        # 긴 마커에서 뒷부분 매칭 시도
        for k, v in SYNTACTIC_MAP.items():
            if m.endswith(k):
                suffix = k
                func_counts[v] += count
                break
        if not suffix:
            func_counts["Other"] += count
            
    # 지배적인 기능 찾기
    sorted_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_funcs:
        return "Unknown"
    
    top_func, score = sorted_funcs[0]
    ratio = score / total_w
    if ratio > 0.4:
        return f"{top_func} ({ratio*100:.0f}%)"
    else:
        return "Mixed"

def analyze_cluster(cid: int, cluster_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    """개별 클러스터 심층 분석"""
    size = len(cluster_df)
    
    # 1. 사서 비율
    is_canon = cluster_df['book_name'].apply(lambda x: any(b in str(x) for b in CANON_BOOKS))
    canon_ratio = is_canon.mean()
    
    # 2. Lift (특이 도서)
    top_lift_books = calculate_lift(full_df, cluster_df)
    
    # 3. Marker Stats
    if 'marker' not in cluster_df.columns:
        cluster_df['marker'] = cluster_df['src_left'].apply(extract_marker)
        
    marker_counts = cluster_df['marker'].value_counts()
    # Entropy (bits)
    marker_probs = marker_counts / size
    marker_ent = entropy(marker_probs, base=2)
    
    top_markers = marker_counts.head(10).to_dict()
    
    # 4. Syntactic Guess
    syn_guess = get_syntactic_guess(top_markers)
    
    # 5. Examples (Random short ones)
    # 길이가 적당히 짧은 것들 중에서 랜덤 샘플링 (10~50자)
    mask = (cluster_df['src_left'].str.len() + cluster_df['src_right'].str.len()).between(2, 50)
    candidates = cluster_df[mask]
    if len(candidates) < 5:
        candidates = cluster_df
    
    examples = []
    if not candidates.empty:
        samples = candidates.sample(n=min(5, len(candidates)), random_state=42)
        for _, row in samples.iterrows():
            examples.append({
                'book': row['book_name'],
                'text': f"{row.get('src_left','')} | {row.get('src_right','')}"
            })
            
    return {
        'id': cid,
        'size': size,
        'canon_ratio': canon_ratio,
        'lift_books': top_lift_books,
        'entropy': marker_ent,
        'top_markers': top_markers,
        'syntactic_guess': syn_guess,
        'examples': examples
    }

def generate_report(stats: list, total_rows: int, out_path: Path):
    """Markdown 리포트 생성"""
    lines = [
        f"# SA 심층 프로파일 (Deep Profile) - K={len(stats)}",
        "",
        f"**분석 대상**: {total_rows:,}개 구(Phrase) 경계",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 클러스터 요약 (Summary)",
        "",
        "| ID | Size | Canonicity | Entropy | Syntactic Function | Top Markers |",
        "|:---:|---:|---:|---:|:---|:---|",
    ]
    
    for s in stats:
        top_m = ", ".join(list(s['top_markers'].keys())[:3])
        lines.append(f"| p{s['id']} | {s['size']:,} | {s['canon_ratio']*100:.1f}% | {s['entropy']:.2f} | **{s['syntactic_guess']}** | {top_m} |")
        
    lines.extend([
        "",
        "---",
        "",
        "## 2. 클러스터별 상세 분석",
        ""
    ])
    
    for s in stats:
        lines.append(f"### Cluster p{s['id']} ({s['syntactic_guess']})")
        lines.append(f"- **규모**: {s['size']:,} ({s['size']/total_rows*100:.1f}%)")
        lines.append(f"- **사서 비중**: {s['canon_ratio']*100:.2f}%")
        lines.append(f"- **마커 다양성(Entropy)**: {s['entropy']:.2f} bits (Low=집중, High=다양)")
        
        # Lift Books
        lift_str = ", ".join([f"{b}(x{l:.1f})" for b, l, c in s['lift_books']])
        lines.append(f"- **특이 도서 (Lift)**: {lift_str}")
        
        # Top Markers
        m_str = ", ".join([f"{k}({v})" for k, v in s['top_markers'].items()])
        lines.append(f"- **주요 마커**: {m_str}")
        
        # Examples
        lines.append("- **대표 예문**:")
        for ex in s['examples']:
            lines.append(f"  - [{ex['book']}] {ex['text']}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="SA Deep Profiling")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data: {args.csv}")
    df = load_data(args.csv)
    
    if 'marker' not in df.columns:
        print("Extracting markers globally...")
        df['marker'] = df['src_left'].astype(str).apply(extract_marker)
    
    stats = []
    cluster_ids = sorted(df['cluster_id'].unique())
    
    print(f"Profiling {len(cluster_ids)} clusters...")
    for cid in cluster_ids:
        c_df = df[df['cluster_id'] == cid]
        s = analyze_cluster(cid, c_df, df)
        stats.append(s)
        if cid % 5 == 0:
            print(f"  Processed p{cid}")
            
    # Report
    generate_report(stats, len(df), args.out_dir / "sa_deep_profile.md")
    
    # JSON Dump
    with open(args.out_dir / "sa_deep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
