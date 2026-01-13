
import os
import sys
import json
import random
import subprocess
from pathlib import Path
import pandas as pd

# 설정 (검증용으로 5개만)
SEED = 5
SAMPLE_SIZE = 5 
EXCLUDE_FILE = "/workspace/datasets/pa_exclude_250.json"
OUTPUT_FILE = "/workspace/test_results/sanity_check_output.csv"
INPUT_DATASET = "/workspace/datasets/pd/test.csv"

def main():
    print(f"[{os.getcwd()}] Sanity Check 시작")
    
    with open(EXCLUDE_FILE, 'r', encoding='utf-8') as f:
        exclude_list = json.load(f)
    exclude_set = set(tuple(k) for k in exclude_list)
    print(f"제외 목록 로드 완료: {len(exclude_set)}개")

    df = pd.read_csv(INPUT_DATASET)
    df['paragraph_id'] = pd.to_numeric(df['문단식별자'], errors='coerce')
    
    valid_indices = []
    for idx, row in df.iterrows():
        key = (row['book_name'], int(row['paragraph_id']))
        if key not in exclude_set:
            valid_indices.append(idx)
            
    # 샘플링 (5개)
    random.seed(SEED)
    sample_indices = sorted(random.sample(valid_indices, min(len(valid_indices), SAMPLE_SIZE)))
    
    sample_df = df.iloc[sample_indices].copy()
    sample_input_path = "/workspace/test_results/sanity_check_input.csv"
    sample_df.to_csv(sample_input_path, index=False, encoding='utf-8-sig')
    print(f"샘플링 완료(5개): {sample_input_path}")
    
    cmd = [
        "python", "pa/main.py",
        sample_input_path,
        OUTPUT_FILE,
        "--use-boundary-model",
        "--threshold", "0.7",
        "--boundary-threshold", "0.72", 
        "--max-workers", "2", 
        "--batch-size", "50",
        "--enable-refine",
        #" --enable-adjacent-boundary-refine",
        "--seed", str(SEED),
    ]
    
    alignment_params = {
        "boundary_bonus": 0.4,
        "supar_bonus": 0.05,
        "boundary_weight_terminal": 0.006,
        "boundary_weight_continuation": -0.01,
        "similarity_threshold": 0.7,
        "length_penalty": 0.0001,
        "distance_decay": 0.05,
        "particle_bonus": 0.0,
        "dp_window": 30,
        "sim_gamma": 3.0,
        "hanja_bonus": 0.1
    }
    
    env = os.environ.copy()
    env["CSP_ALIGNMENT_PARAMS"] = json.dumps(alignment_params)
    
    print("PA 실행 시작 (Sanity Check)...")
    subprocess.run(cmd, env=env, check=True)
    print("PA 실행 완료.")

if __name__ == "__main__":
    main()
