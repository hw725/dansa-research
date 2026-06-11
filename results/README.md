# results 기준

활성 정량 결과는 다음 세 파일이다.

- `final_stats_v3.1_cleaned_balanced.json`
- `cleaned_balanced_stats.json`
- `truth_tables_v3.1_cleaned_balanced.json`

모델별 raw 판정 CSV는 `results/{gpt5mini,gemini,claude_sonnet}/` 아래에 남아 있으나 원문 번역문을 포함하므로 git 추적 대상이 아니다. 공유용 파일은 `*_anon.csv`만 사용한다.

모델별 판정 anon(`results/{model}/*_anon.csv`)만으로 위 세 통계를 재계산할 수 있다 — 통계는 marker·판정값만 쓰기 때문이다. 검증: `python scripts/compute_final_stats.py --check --source anon`.

강건성·일치도 보조 통계는 `scripts/compute_robustness_stats.py`가 위 판정 CSV에서 산출한다. LLM 호출이 없고 book·marker·판정값만 쓰므로 anon 소스만으로 동일 수치가 재현된다(raw 대조 일치 확인 완료).

- `robustness_stats.json` — 모델 간 일치도(Fleiss·Cohen κ), 효과크기 95% CI(비율차 Newcombe·OR Woolf·Cramér’s V 부트스트랩), 합의 정의 민감도, 서종(book·部) 층화 Mantel-Haenszel OR
- `ROBUSTNESS_REPORT.md` — 위 JSON의 표 형식 보고서
