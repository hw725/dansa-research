
import os
import sys
import json
import random
import subprocess
from pathlib import Path
import pandas as pd

# 설정
SEED = 5
SAMPLE_SIZE = 1000
EXCLUDE_FILE = "/workspace/datasets/pa_exclude_250.json"
OUTPUT_FILE = "/workspace/test_results/nightly_test_1000_sb0.05.csv"
INPUT_DATASET = "/workspace/datasets/pd/test.csv"

# 인자 설정
# prior_bonus=0.4, supar_bonus=0.05, bsp_wt=0.006, bsp_wc=-0.01
# use_boundary_model=True
# enable_refine=True

def main():
    print(f"[{os.getcwd()}] PA Nightly Test 시작")
    
    # 1. 제외 목록 로드
    with open(EXCLUDE_FILE, 'r', encoding='utf-8') as f:
        exclude_list = json.load(f)  # [[book, pid], ...]
    exclude_set = set(tuple(k) for k in exclude_list)
    print(f"제외 목록 로드 완료: {len(exclude_set)}개")

    # 2. 전체 데이터셋 로드 및 필터링
    df = pd.read_csv(INPUT_DATASET)
    df['paragraph_id'] = pd.to_numeric(df['문단식별자'], errors='coerce')
    
    valid_indices = []
    for idx, row in df.iterrows():
        key = (row['book_name'], int(row['paragraph_id']))
        if key not in exclude_set:
            valid_indices.append(idx)
            
    print(f"제외 후 유효 문단: {len(valid_indices)}개")
    
    # 3. 샘플링 (1000개)
    random.seed(SEED)
    sample_indices = sorted(random.sample(valid_indices, min(len(valid_indices), SAMPLE_SIZE)))
    
    sample_df = df.iloc[sample_indices].copy()
    sample_input_path = "/workspace/test_results/nightly_test_1000_input.csv"
    sample_df.to_csv(sample_input_path, index=False, encoding='utf-8-sig')
    print(f"샘플링 완료: {len(sample_df)}개 -> {sample_input_path}")
    
    # 4. PA 실행
    # python pa/main.py input output --args...
    cmd = [
        "python", "pa/main.py",
        sample_input_path,
        OUTPUT_FILE,
        "--use-boundary-model",
        "--threshold", "0.7",
        "--boundary-threshold", "0.72", # Default
        "--max-workers", "6", # Docker 환경에 맞게 조정
        "--batch-size", "50",
        "--enable-refine",
        # "--enable-adjacent-boundary-refine",  # Default enabled, option does not exist (only disable exists)
        "--seed", str(SEED),
    ]
    
    # 환경변수로 bonus 파라미터 주입 (pa/main.py가 argparse로 안 받을 경우 대비)
    # 하지만 pa/main.py가 bonus 인자를 직접 안 받는다면 코드 수정 없이 config/env로 해야 함.
    # CSP_ALIGNMENT_PARAMS 환경변수를 통해 JSON 형태로 주입하는 것이 가장 확실함.
    
    alignment_params = {
        "boundary_bonus": 0.4,           # prior_bonus
        "supar_bonus": 0.05,             # supar_bonus (NEW!)
        "boundary_weight_terminal": 0.006,
        "boundary_weight_continuation": -0.01,
        # 기본값 유지
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
    
    print("PA 실행 시작...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Params: {env['CSP_ALIGNMENT_PARAMS']}")
    
    with open("/workspace/test_results/nightly_test_run.log", "w") as log_f:
        subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        
    print("PA 실행 완료.")

if __name__ == "__main__":
    main()
