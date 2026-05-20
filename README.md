# dansa-research

任圭直 《句讀解法》의 斷辭 분류를 한국 한문-한국어 병렬 코퍼스와 3모델 LLM 판정으로 검증하는 연구 저장소.

## 현재 기준

- 최신 정량 기준: `results/final_stats_v3.1_cleaned_balanced.json`
- 요약 통계: `results/cleaned_balanced_stats.json`
- 합의 판정 truth table: `results/truth_tables_v3.1_cleaned_balanced.json`
- ‘하다’ 메타데이터 통계: `results/hada_metadata_stats.json`
- 표준 sentence 입력: `data/sentence_normalized.csv` 150,545행
- 질적 분석 입력: `analysis/parallel_data_v2_cleaned.tsv` 11,327행
- 구버전 보존 위치: `archive/2026-05-20_csp_sync/`

## 핵심 결과

| 섹션 | Target | Control | Consensus O | χ² | V |
|---|---:|---:|---:|---:|---:|
| 섹션 1 游辭以斷 | 2,606 | 2,606 | 41.0% vs 6.2% | 1401.76 | 0.519 |
| 섹션 2 夬絶之斷 vs 微絶之斷 | 11,135 | 11,135 | 45.7% vs 27.7% | 986.45 | 0.210 |
| 섹션 3 汎論以斷 | 296 | 296 | 80.7% vs 46.3% | 88.68 | 0.387 |

현재 sentence 기준 ‘하다’ 메타데이터 통계는 출현 12,988건, 歷史書 24.56%, 文集 5.93%, 經傳 0.23%, 詩 0.03%, 기타 0.02%, 歷史書 대 非歷史書 χ² = 17574.78, p < 0.001이다.

## 재현 순서

```bash
python scripts/preflight_llm_pipeline.py
python scripts/analyze_hada_metadata.py
python scripts/analyze_hada_metadata.py --check-existing
python scripts/run_multimodel_judgments.py
python scripts/run_supplement_judgments.py
python scripts/compute_final_stats.py
python scripts/export_anonymized_results.py
```

`run_multimodel_judgments.py`는 보충 대조군 반영 전 중간 통계를 `results/intermediate_multimodel_stats.json`에 쓴다. 논문과 문서에서 인용할 기준은 `results/final_stats_v3.1_cleaned_balanced.json`이다.

보고된 통계 수치의 exact 재현에는 `data/llm_manifests/`의 LLM 입력 표본 manifest를 사용한다. 새 랜덤 표본에서 효과 크기가 유지되는지는 별도 robustness 검증으로 다룬다.

## 공개용 데이터

원문 번역문은 미공개 자료이므로 raw CSV는 git 추적 대상이 아니다. 공개가 필요한 경우 `scripts/export_anonymized_results.py`로 생성한 `*_anon.csv`와 `*_anon.tsv`만 사용한다.

## 아카이브 원칙

2026-05-20 정리에서 `CSP-dansa` 스냅샷과 `dansa-research` 구버전 파일을 모두 `archive/2026-05-20_csp_sync/`에 보존했다. 이 폴더는 삭제가 아니라 보존용 격리이며, 기본적으로 git 추적에서 제외한다.

## 스크립트 파일명
현재 스크립트 목록과 명명 규칙은 docs/SCRIPTS.md를 기준으로 한다.
