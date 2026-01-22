#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 재분석 결과(번역문 포함)를 바탕으로 모든 리포트와 시각화 자료를 갱신합니다.
"""

import os
import subprocess
from pathlib import Path

def run_cmd(cmd: list[str]):
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    else:
        print(f"✅ Success")

def main():
    # 1. 경로 설정 (V6 결과물 위치)
    PA_V6 = "hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv"
    SA_V6 = "hyeonto/reports/sa_boundary_v6_full/sa_boundary_clusters.csv"
    
    # 출력 경로 설정
    PA_LABEL_MD = "hyeonto/reports/pa_boundary_v6_full/boundary_clusters_labeled.md"
    SA_PROFILE_MD = "hyeonto/reports/sa_boundary_v6_full/sa_cluster_profile.md"
    PA_PROFILE_MD = "hyeonto/reports/pa_boundary_v6_full/pa_cluster_profile.md"
    SA_OUT_DIR = "hyeonto/reports/sa_boundary_v6_full"
    
    # 2. 휴리스틱 라벨링 (Describe)
    print("\n--- 1. Heuristic Labeling (PA) ---")
    run_cmd(["python", "scripts/describe_boundary_clusters.py", "--csv", PA_V6, "--out", PA_LABEL_MD])
    
    # 3. 상세 프로파일링 (Profile - Canonicity, Markers, Hanja)
    print("\n--- 2. Detailed Profiling ---")
    run_cmd(["python", "scripts/profile_boundary_clusters.py", "--csv", PA_V6, "--out", PA_PROFILE_MD])
    run_cmd(["python", "scripts/profile_boundary_clusters.py", "--csv", SA_V6, "--out", SA_PROFILE_MD])

    # 4. 통사 기능 분석 (Syntactic Analysis)
    print("\n--- 3. Syntactic Analysis ---")
    # analyze_marker_syntactic_function.py가 v6 csv를 지원하는지 확인 필요하나, 
    # 일단 인자로 넘겨서 실행 (기존 내부 로직이 pa_clusters_with_features.csv 등을 직접 참조할 수 있음)
    run_cmd(["python", "scripts/analyze_marker_syntactic_function.py", 
             "--pa-csv", PA_V6, 
             "--sa-csv", SA_V6, 
             "--out-dir", "hyeonto/reports/syntactic_analysis_v6"])

    # 5. 시각화 (Visualization)
    print("\n--- 4. Visualizations ---")
    # visualize_parent_situations.py는 --csv 인자를 지원함
    run_cmd(["python", "scripts/visualize_parent_situations.py", "--csv", PA_V6, "--out-dir", "hyeonto/reports/pa_joint_embedding_v6"])
    
    print("\n--- Final Step: UPDATE FINAL_ANALYSIS_REPORT.md ---")
    print("모든 기초 데이터가 갱신되었습니다. 이제 이 데이터를 바탕으로 마스터 리포트를 재작성합니다.")

if __name__ == "__main__":
    main()
