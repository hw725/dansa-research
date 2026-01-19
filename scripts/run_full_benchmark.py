import subprocess
import os
import sys
import pandas as pd
import time
from datetime import datetime

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🚀 [Step: {desc}] 시작 - {datetime.now()}")
    print(f"명령어: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # 실시간 출력을 위해 p.stdout을 읽거나, 간단히 check_call 사용
        subprocess.check_call(cmd, shell=True)
        elapsed = time.time() - start_time
        print(f"\n✅ [Step: {desc}] 완료 ({elapsed:.1f}초)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Step: {desc}] 실패 (Exit Code: {e.returncode})")
        return False

def main():
    # 0. 디렉토리 설정
    os.makedirs("test_results", exist_ok=True)
    
    # === 1. S2P 파이프라인 ===
    s2p_input = "datasets/sentence/test.csv"
    s2p_output = "test_results/s2p_full_final.xlsx"
    s2p_gold = "datasets/phrase/test.csv"
    # CSV 변환이 필요할 수 있으므로 평가용 CSV 경로 미리 정의
    s2p_output_csv = s2p_output.replace(".xlsx", ".csv")

    cmd_s2p_run = f"python s2p/main.py {s2p_input} {s2p_output} --batch-size 32"
    
    if run_command(cmd_s2p_run, "S2P 파이프라인 가동"):
        # 평가를 위해 XLSX -> CSV 변환 (Evaluator 호환성 보장)
        try:
            print("📊 평가를 위해 S2P 결과를 CSV로 변환 중...")
            df = pd.read_excel(s2p_output)
            df.to_csv(s2p_output_csv, index=False)
            print("✅ CSV 변환 완료")
            
            # S2P 평가 (인자 순서: Gold Pred) - s2p_evaluator.py는 인자 순서가 중요할 수 있음. 확인 필요.
            # 기존 사용 패턴: python accuracy/s2p_evaluator.py datasets/phrase/test_10_gold.csv test_results/...
            cmd_s2p_eval = f"python accuracy/s2p_evaluator.py {s2p_gold} {s2p_output_csv} --project sa"
            run_command(cmd_s2p_eval, "S2P F1 평가")
        except Exception as e:
            print(f"❌ S2P 결과 변환/평가 중 오류: {e}")

    # === 2. P2S 파이프라인 ===
    p2s_input = "datasets/paragraph/test.csv"
    p2s_output = "test_results/p2s_full_final.xlsx"
    p2s_gold = "datasets/sentence/test.csv"
    
    cmd_p2s_run = f"python p2s/main.py {p2s_input} {p2s_output}"
    # P2S Eval (p2s_evaluator.py는 --input --gold 인자 사용)
    cmd_p2s_eval = f"python accuracy/p2s_evaluator.py --input {p2s_output} --gold {p2s_gold}"

    if run_command(cmd_p2s_run, "P2S 파이프라인 가동 (재실행)"):
        run_command(cmd_p2s_eval, "P2S F1 평가")

if __name__ == "__main__":
    main()
