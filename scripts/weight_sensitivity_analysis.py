#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""가중치 민감도 분석

여러 가중치 조합으로 잔차화 분석 수행 후 결과 비교.
최적 가중치를 도출하기 위한 체계적 탐색.

Usage:
    python scripts/weight_sensitivity_analysis.py \
        --input hyeonto/datasets/sentence_train_full.csv \
        --clusters hyeonto/reports/recluster_k16_child/reclustered.csv \
        --out-dir hyeonto/reports/weight_sensitivity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


# 테스트할 가중치 시나리오
WEIGHT_SCENARIOS = [
    {"name": "uniform", "saseo": 1.0, "samgyeong": 1.0, "other_gyeong": 1.0},
    {"name": "weak", "saseo": 2.0, "samgyeong": 1.5, "other_gyeong": 1.2},
    {"name": "moderate", "saseo": 3.0, "samgyeong": 2.0, "other_gyeong": 1.5},
    {"name": "strong", "saseo": 5.0, "samgyeong": 3.0, "other_gyeong": 2.0},
]


def run_residualized_analysis(
    input_path: Path,
    clusters_path: Path,
    out_dir: Path,
    weight_saseo: float,
    weight_samgyeong: float,
    weight_other_gyeong: float,
) -> dict:
    """
    잔차화 분석 실행 후 결과 반환
    """
    script_path = Path(__file__).parent / "analyze_residualized_markers.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--input", str(input_path),
        "--clusters", str(clusters_path),
        "--genre-level", "detail",
        "--weight-saseo", str(weight_saseo),
        "--weight-samgyeong", str(weight_samgyeong),
        "--weight-other-gyeong", str(weight_other_gyeong),
        "--out-dir", str(out_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ 분석 실패: {out_dir}")
        print(result.stderr)
        return {}
    
    # 결과 읽기
    config_path = out_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_cluster_stability(results: list[dict]) -> pd.DataFrame:
    """
    여러 시나리오 결과 간의 안정성 지표 계산
    
    ARI (Adjusted Rand Index)를 계산하려면 클러스터 할당이 필요하지만,
    현재는 장르 엔트로피를 주요 지표로 사용
    """
    records = []
    for r in results:
        if r:
            records.append({
                "scenario": r.get("scenario_name", "unknown"),
                "weight_saseo": r.get("weight_saseo", 0),
                "weight_samgyeong": r.get("weight_samgyeong", 0),
                "weight_other_gyeong": r.get("weight_other_gyeong", 0),
                "avg_genre_entropy": r.get("avg_genre_entropy", 0),
                "n_markers": r.get("n_markers", 0),
            })
    return pd.DataFrame(records)


def generate_comparison_report(
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """비교 보고서 생성"""
    
    report = []
    report.append("# 가중치 민감도 분석 보고서\n")
    report.append(f"**분석일**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append("\n---\n")
    
    report.append("\n## 1. 시나리오별 결과 요약\n")
    report.append("\n| 시나리오 | 사서 | 삼경 | 기타경전 | 평균 장르 엔트로피 | 마커 수 |\n")
    report.append("|----------|------|------|----------|-------------------|--------|\n")
    
    for _, row in summary_df.iterrows():
        report.append(
            f"| {row['scenario']} | {row['weight_saseo']:.1f}x | "
            f"{row['weight_samgyeong']:.1f}x | {row['weight_other_gyeong']:.1f}x | "
            f"{row['avg_genre_entropy']:.4f} | {row['n_markers']} |\n"
        )
    
    report.append("\n## 2. 해석\n")
    report.append("\n### 장르 엔트로피\n")
    report.append("- **높은 엔트로피**: 클러스터 내 장르 혼재 ↑ → 장르 효과 제거 성공\n")
    report.append("- **낮은 엔트로피**: 클러스터가 여전히 장르에 의해 분리됨\n")
    
    # 최고 엔트로피 시나리오
    if len(summary_df) > 0:
        best_idx = summary_df["avg_genre_entropy"].idxmax()
        best = summary_df.loc[best_idx]
        report.append(f"\n### 권장 시나리오\n")
        report.append(f"**{best['scenario']}** (엔트로피: {best['avg_genre_entropy']:.4f})\n")
        report.append(f"- 사서: {best['weight_saseo']:.1f}x\n")
        report.append(f"- 삼경: {best['weight_samgyeong']:.1f}x\n")
        report.append(f"- 기타 경전: {best['weight_other_gyeong']:.1f}x\n")
    
    report.append("\n---\n")
    report.append("\n## 3. 추가 권장 사항\n")
    report.append("- 엔트로피가 가장 높은 시나리오가 항상 최선은 아닐 수 있음\n")
    report.append("- 클러스터별 대표 마커의 해석 일관성도 고려해야 함\n")
    report.append("- 전문가 검토를 통해 최종 결정 권장\n")
    
    report_path = out_dir / "sensitivity_report.md"
    report_path.write_text("".join(report), encoding="utf-8")
    print(f"✅ Saved: {report_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="가중치 민감도 분석")
    p.add_argument("--input", type=Path, default=Path("hyeonto/datasets/sentence_train_full.csv"))
    p.add_argument("--clusters", type=Path, default=Path("hyeonto/reports/recluster_k16_child/reclustered.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/weight_sensitivity"))
    args = p.parse_args()
    
    if not args.input.exists():
        print(f"❌ 입력 파일 없음: {args.input}")
        return 1
    
    if not args.clusters.exists():
        print(f"❌ 클러스터 파일 없음: {args.clusters}")
        return 1
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for scenario in WEIGHT_SCENARIOS:
        name = scenario["name"]
        print(f"\n{'='*60}")
        print(f"🔄 시나리오: {name}")
        print(f"   사서={scenario['saseo']}x, 삼경={scenario['samgyeong']}x, 기타={scenario['other_gyeong']}x")
        print(f"{'='*60}")
        
        scenario_dir = args.out_dir / name
        
        result = run_residualized_analysis(
            args.input,
            args.clusters,
            scenario_dir,
            scenario["saseo"],
            scenario["samgyeong"],
            scenario["other_gyeong"],
        )
        
        if result:
            result["scenario_name"] = name
            results.append(result)
            print(f"   ✅ 평균 장르 엔트로피: {result.get('avg_genre_entropy', 'N/A'):.4f}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 결과 요약")
    print(f"{'='*60}")
    
    summary_df = compute_cluster_stability(results)
    print(summary_df.to_string(index=False))
    
    # 결과 저장
    summary_df.to_csv(args.out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved: {args.out_dir / 'sensitivity_summary.csv'}")
    
    # 보고서 생성
    generate_comparison_report(summary_df, args.out_dir)
    
    # 시각화 (matplotlib 사용)
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(summary_df))
        bars = ax.bar(x, summary_df["avg_genre_entropy"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['scenario']}\n(사서:{row['weight_saseo']}x)" for _, row in summary_df.iterrows()])
        ax.set_ylabel("평균 장르 엔트로피")
        ax.set_title("가중치 시나리오별 장르 엔트로피 비교\n(높을수록 장르 효과 제거 성공)")
        
        # 값 표시
        for bar, val in zip(bars, summary_df["avg_genre_entropy"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(args.out_dir / "entropy_comparison.png", dpi=150)
        print(f"✅ Saved: {args.out_dir / 'entropy_comparison.png'}")
        plt.close()
        
    except ImportError:
        print("⚠️ matplotlib 미설치, 시각화 생략")
    
    print(f"\n✅ 모든 분석 완료: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
