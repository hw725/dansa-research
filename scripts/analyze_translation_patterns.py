#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA/SA 클러스터별 번역 패턴 분석

각 클러스터에서 번역문 어미 패턴을 추출하여 의미역 특성을 보강합니다.

출력:
- pa_translation_patterns.csv
- sa_translation_patterns.csv
- translation_pattern_summary.md
"""

import pandas as pd
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
import json

def extract_translation_ending(tgt_text: str, max_chars: int = 8) -> str:
    """번역문에서 어미 추출 (마지막 N글자)"""
    if not tgt_text or str(tgt_text) == "nan":
        return ""
    tgt = str(tgt_text).strip()
    if not tgt:
        return ""
    # 마지막 어미 부분 추출
    ending = tgt[-max_chars:] if len(tgt) > max_chars else tgt
    # 한글만 추출
    hangul_only = re.sub(r'[^\p{Hangul}]', '', ending)
    return hangul_only[-6:] if len(hangul_only) > 6 else hangul_only


def analyze_translation_patterns(csv_path: Path, cluster_col: str, output_prefix: str, out_dir: Path):
    """클러스터별 번역 패턴 분석"""
    
    print(f"📄 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df):,}")
    
    # 클러스터 컬럼 확인
    if cluster_col not in df.columns:
        if "parent_cluster_id" in df.columns:
            cluster_col = "parent_cluster_id"
        elif "cluster_id" in df.columns:
            cluster_col = "cluster_id"
        else:
            print(f"   ⚠️ No cluster column found. Skipping.")
            return None
    
    print(f"   Cluster column: {cluster_col}")
    
    # 번역문 컬럼 확인
    tgt_col = None
    for col in ["번역문", "tgt", "translation"]:
        if col in df.columns:
            tgt_col = col
            break
    
    if not tgt_col:
        print(f"   ⚠️ No translation column found. Skipping.")
        return None
    
    print(f"   Translation column: {tgt_col}")
    
    # 클러스터별 번역 패턴 수집
    cluster_patterns = defaultdict(Counter)
    cluster_counts = Counter()
    
    for _, row in df.iterrows():
        cluster = row[cluster_col]
        tgt = str(row.get(tgt_col, ""))
        ending = extract_translation_ending(tgt)
        
        if ending:
            cluster_patterns[cluster][ending] += 1
            cluster_counts[cluster] += 1
    
    # 결과 정리
    results = []
    for cluster in sorted(cluster_patterns.keys()):
        patterns = cluster_patterns[cluster]
        total = cluster_counts[cluster]
        top5 = patterns.most_common(5)
        
        results.append({
            "cluster": cluster,
            "total": total,
            "top1_pattern": top5[0][0] if top5 else "",
            "top1_count": top5[0][1] if top5 else 0,
            "top1_ratio": f"{top5[0][1]/total*100:.1f}%" if top5 and total > 0 else "",
            "top2_pattern": top5[1][0] if len(top5) > 1 else "",
            "top3_pattern": top5[2][0] if len(top5) > 2 else "",
            "top4_pattern": top5[3][0] if len(top5) > 3 else "",
            "top5_pattern": top5[4][0] if len(top5) > 4 else "",
            "pattern_diversity": len(patterns),
        })
    
    # CSV 저장
    out_csv = out_dir / f"{output_prefix}_translation_patterns.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"   ✅ Saved: {out_csv}")
    
    return results


def main():
    out_dir = Path("hyeonto/reports/translation_patterns")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # PA 분석
    pa_csv = Path("hyeonto/reports/clusters_raw_data/pa_clusters_with_features.csv")
    if pa_csv.exists():
        pa_results = analyze_translation_patterns(pa_csv, "parent_cluster_id", "pa", out_dir)
    else:
        print(f"⚠️ PA file not found: {pa_csv}")
        pa_results = None
    
    # SA 분석
    sa_csv = Path("hyeonto/reports/clusters_raw_data/sa_clusters_with_features.csv")
    if sa_csv.exists():
        sa_results = analyze_translation_patterns(sa_csv, "parent_cluster_id", "sa", out_dir)
    else:
        print(f"⚠️ SA file not found: {sa_csv}")
        sa_results = None
    
    # 요약 리포트
    summary_path = out_dir / "translation_pattern_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 클러스터별 번역 패턴 분석\n\n")
        f.write("**분석 목적**: 각 클러스터의 번역문 어미 패턴을 추출하여 의미역 특성 보강\n\n")
        
        if pa_results:
            f.write("## PA (문장 단위) 번역 패턴\n\n")
            f.write("| 클러스터 | 샘플 수 | Top 1 패턴 | 비율 | Top 2 | Top 3 | 다양성 |\n")
            f.write("|:---:|:---:|:---|:---:|:---|:---|:---:|\n")
            for r in pa_results:
                f.write(f"| p{r['cluster']} | {r['total']:,} | {r['top1_pattern']} | {r['top1_ratio']} | {r['top2_pattern']} | {r['top3_pattern']} | {r['pattern_diversity']} |\n")
            f.write("\n")
        
        if sa_results:
            f.write("## SA (구 단위) 번역 패턴\n\n")
            f.write("| 클러스터 | 샘플 수 | Top 1 패턴 | 비율 | Top 2 | Top 3 | 다양성 |\n")
            f.write("|:---:|:---:|:---|:---:|:---|:---|:---:|\n")
            for r in sa_results:
                f.write(f"| p{r['cluster']} | {r['total']:,} | {r['top1_pattern']} | {r['top1_ratio']} | {r['top2_pattern']} | {r['top3_pattern']} | {r['pattern_diversity']} |\n")
            f.write("\n")
    
    print(f"\n📊 Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
