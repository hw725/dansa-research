# results 기준

활성 정량 결과는 다음 세 파일이다.

- `final_stats_v3.1_cleaned_balanced.json`
- `cleaned_balanced_stats.json`
- `truth_tables_v3.1_cleaned_balanced.json`

모델별 raw 판정 CSV는 `results/{gpt5mini,gemini,claude_sonnet}/` 아래에 남아 있으나 원문 번역문을 포함하므로 git 추적 대상이 아니다. 공유용 파일은 `*_anon.csv`만 사용한다.

모델별 판정 anon(`results/{model}/*_anon.csv`)만으로 위 세 통계를 재계산할 수 있다 — 통계는 marker·판정값만 쓰기 때문이다. 검증: `python scripts/compute_final_stats.py --check --source anon`.
