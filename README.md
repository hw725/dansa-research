# dansa-research

任圭直 《句讀解法》의 斷辭 분류를 한국 한문-한국어 병렬 코퍼스와 3모델 LLM 판정으로 검증하는 연구 저장소.

## 저장소 구성

이 저장소는 두 단계로 구성된다.

- **A. 핵심 정량 연구** — 3개 LLM(gpt-5-mini·gemini·claude-sonnet) 합의 판정으로 斷辭 분류를 검증한다. 파이프라인은 `scripts/`, 입력은 `data/`, 산출물은 `results/`, 재현 절차는 루트 `REPRODUCE.md`에 있다.
- **B. 夬絶 가설 분석** — 종결어미 ‘니라’ 대 ‘라’의 효과를 임베딩·클러스터링·LightRAG·DCI·장르 통제로 검토한다. 전부 `analysis/` 아래에 있으며 입구는 [analysis/README.md](analysis/README.md)이다.

```
├─ REPRODUCE.md 재현 가이드 (표본 추출 기준 §8 포함)
├─ scripts/    A단계 재현 스크립트 (목록·명명 규칙은 scripts/README.md)
├─ data/       입력 데이터 (대부분 비공개·로컬 전용 — data/README.md)
├─ results/    A단계 정본 통계와 공개용 익명 CSV
├─ analysis/   B단계 질적·보조 분석 (자체 README)
├─ logs/       실행 로그 (비추적)
├─ sandbox/    재현 샌드박스 + 웹 진입 앱 (sandbox/README.md)
└─ archive/    보관용 스냅샷 (비추적)
```

## 처음 보는 분께

1. 무엇을 했는가 → 아래 [핵심 결과](#핵심-결과)
2. 어떻게 재현하는가 → [REPRODUCE.md](REPRODUCE.md)
3. 데이터를 어떻게 얻는가 → [data/README.md](data/README.md). raw 입력은 비공개이며 공개본은 익명화 CSV뿐이다.

## 현재 기준

- 최신 정량 기준: `results/final_stats_v3.1_cleaned_balanced.json`
- 요약 통계: `results/cleaned_balanced_stats.json`
- 합의 판정 truth table: `results/truth_tables_v3.1_cleaned_balanced.json`
- 강건성·일치도 통계: `results/robustness_stats.json` (보고서: `results/ROBUSTNESS_REPORT.md`)
- ‘하다’ 메타데이터 통계: `results/hada_metadata_stats.json`
- 표준 sentence 입력: `data/sentence_normalized.csv` 150,545행
- 질적 분석 입력: `analysis/parallel_data_v2_cleaned.tsv` 11,327행

## 핵심 결과

| 섹션 | Target | Control | Consensus O | χ² | V |
|---|---:|---:|---:|---:|---:|
| 섹션 1 游辭以斷 | 2,606 | 2,606 | 41.0% vs 6.2% | 1401.76 | 0.519 |
| 섹션 2 夬絶之斷 vs 微絶之斷 | 11,135 | 11,135 | 45.7% vs 27.7% | 986.45 | 0.210 |
| 섹션 3 汎論以斷 | 296 | 296 | 80.7% vs 46.3% | 88.68 | 0.387 |

현재 sentence 기준 ‘하다’ 메타데이터 통계는 출현 12,988건, 歷史書 24.56%, 文集 5.93%, 經傳 0.23%, 詩 0.03%, 기타 0.02%, 歷史書 대 非歷史書 χ² = 17574.78, p < 0.001이다.

세 섹션 효과는 서종(book) 층화 Mantel-Haenszel OR 7.67·2.27·2.04로 층화 후에도 유지된다. 모델 간 일치도(Fleiss κ), 효과크기 95% CI, 합의 정의 민감도는 [results/ROBUSTNESS_REPORT.md](results/ROBUSTNESS_REPORT.md)를 본다.

## 재현

전체 재현 절차(preflight → 판정 → 보충 → 통계 → 익명화)는 [REPRODUCE.md](REPRODUCE.md)를 단일 기준으로 한다. 스크립트 목록과 명명 규칙은 [scripts/README.md](scripts/README.md), 표본 추출과 seed robustness는 REPRODUCE.md §8에 있다.

논문과 문서에서 인용할 정량 기준은 `results/final_stats_v3.1_cleaned_balanced.json`이다. `run_multimodel_judgments.py`가 보충 대조군 반영 전 중간 통계를 `results/intermediate_multimodel_stats.json`에 쓰지만, 인용 기준은 final 파일이다.

## 재현 샌드박스 (라이브)

斷辭 분류의 LLM 판정은 입력으로 번역문(한국고전번역원 국역, 미공개 저작물)을 사용한다. 번역문을 공개하지 않으면서 외부에서 재현할 수 있도록 두 층으로 제공한다.

- **통계 재현 — 영구·완전 공개**: 공개 익명 데이터만으로 보고된 합의 통계를 그대로 재계산한다. 번역문 없이 marker와 판정값만 쓰므로 누구나 영구히 재현할 수 있다.
  ```bash
  python scripts/compute_final_stats.py --check --source anon   # CHECK PASS
  ```
- **LLM 판정 재현 — 한시 공개 샌드박스**: 공개 웹 재현 환경에서 동일한 판정 절차를 다시 실행하고 동결 기준 결과와 비교한다. 미공개 입력 자료는 공개 저장소에 포함하지 않는다.
  - 현재 라이브 주소(임시 — 변경 시 이 줄을 갱신): https://synthetic-purpose-mother-november.trycloudflare.com
  - 구축·운영·보안 설계: [sandbox/README.md](sandbox/README.md)

라이브 샌드박스는 한시 공개이며(공개 종료일: 미정) 종료 후에는 요청 시 제공한다. 영구 재현 기준은 공개 익명화 데이터셋과 코드(위 통계 재현)다.
