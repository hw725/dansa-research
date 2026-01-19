import pandas as pd
import subprocess
import os

# 1. Excel -> CSV 변환 (S2P 예측)
pred_xlsx = 'test_results/s2p_test_10_v3.xlsx'
pred_csv = 'test_results/s2p_test_10_v3.csv'
if os.path.exists(pred_xlsx):
    df = pd.read_excel(pred_xlsx, engine='openpyxl')
    df.to_csv(pred_csv, index=False)
    print("✅ S2P 예측 CSV 변환 완료")

# 2. Excel -> CSV 변환 (P2S 예측)
p2s_xlsx = 'test_results/p2s_test_10.xlsx'
p2s_csv = 'test_results/p2s_test_10.csv'
if os.path.exists(p2s_xlsx):
    df = pd.read_excel(p2s_xlsx, engine='openpyxl')
    df.to_csv(p2s_csv, index=False)
    print("✅ P2S 예측 CSV 변환 완료")

# 3. Evaluator 실행 (S2P)
# 순서: Gold(정답) Pred(예측)
gold_s2p = 'datasets/phrase/test_10_gold.csv'
cmd_s2p = f"python accuracy/s2p_evaluator.py {gold_s2p} {pred_csv} --project sa"
print(f"🚀 실행: {cmd_s2p}")
with open('test_results/s2p_eval_final.txt', 'w', encoding='utf-8') as f:
    subprocess.run(cmd_s2p, shell=True, stdout=f, stderr=subprocess.STDOUT)

# 4. Evaluator 실행 (P2S)
gold_p2s = 'datasets/sentence/test_10_gold.csv'
cmd_p2s = f"python accuracy/p2s_evaluator.py --input {p2s_csv} --gold {gold_p2s}"
print(f"🚀 실행: {cmd_p2s}")
with open('test_results/p2s_eval_final.txt', 'w', encoding='utf-8') as f:
    subprocess.run(cmd_p2s, shell=True, stdout=f, stderr=subprocess.STDOUT)
