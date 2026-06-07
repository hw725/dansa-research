# results 기준

활성 정량 결과는 다음 세 파일이다.

- `final_stats_v3.1_cleaned_balanced.json`
- `cleaned_balanced_stats.json`
- `truth_tables_v3.1_cleaned_balanced.json`

모델별 raw 판정 CSV는 `results/{gpt5mini,gemini,claude_sonnet}/` 아래에 남아 있으나 원문 번역문을 포함하므로 git 추적 대상이 아니다. 공유용 파일은 `*_anon.csv`만 사용한다.

공개용 병합 분석 테이블 `consensus_analysis_table_anon.csv`(28,074행, 번역문 해시)는 모델별 O/X와 consensus를 담아, 익명 파일만으로 위 통계를 재계산할 수 있게 한다. `python scripts/compute_final_stats.py --check --source anon`으로 검증한다.
