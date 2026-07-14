# scripts — 스크립트 목록과 명명 규칙

2026-05-20 정리 이후 스크립트 파일명은 역할 접두사를 먼저 둔다. 파일명만 보고 입력 준비, 모델 실행, 통계 산출, 익명화, 검증, 표본 추출, 감사 작업을 구분할 수 있어야 한다.

## 접두사 규칙

| 접두사 | 용도 |
|---|---|
| `prepare_` | 데이터 입력 파일 준비 |
| `run_` | LLM 판정 또는 질의 실행 |
| `compute_` | 통계와 truth table 산출 |
| `export_` | 공개용 파생 파일 생성 |
| `verify_` | 현재 결과물 검증 |
| `analyze_` | 연구 질문별 분석 |
| `audit_` | 데이터 품질과 정규화 점검 |
| `sample_` | 예시와 후보 표본 추출 |
| `build_` | 임베딩, 클러스터, 보고서 구성 |
| `summarize_` | 패턴 요약 산출 |
| `classify_` | 분류 작업 |
| `normalize_` | 정규화 유틸리티 |
| `freeze_` | 기준 실행 산출물·코퍼스 동결(매니페스트·백업) |

## 주요 재현 스크립트

```bash
python scripts/preflight_llm_pipeline.py
python scripts/build_llm_input_manifests.py
python scripts/run_multimodel_judgments.py
python scripts/run_supplement_judgments.py
python scripts/compute_final_stats.py
python scripts/compute_robustness_stats.py
python scripts/export_anonymized_results.py
python scripts/verify_section2_results.py
```

`preflight_llm_pipeline.py`는 외부 API를 호출하지 않고 입력 파일, manifest, prompt 생성, O/X 파서, mock 판정 배치, 기존 결과 resume 상태를 점검한다.

`build_llm_input_manifests.py`는 현재 기준 LLM 입력 표본을 `data/llm_manifests/` 아래에 고정한다. 이 manifest는 보고된 수치의 exact 재현을 위한 로컬 전용 자료이며, 원문과 번역문을 포함하므로 git 추적 대상이 아니다.

표본 추출 기준과 여러 seed robustness 검증 설계는 `REPRODUCE.md` §8에 정리한다.

`compute_final_stats.py`는 `--source {auto,raw,anon}`로 입력을 고르고 `--check`로 기준 JSON과 대조한다. 익명 판정 파일(`results/{model}/*_anon.csv`)만으로도 동일 통계가 재현된다. 상세는 `REPRODUCE.md`. `compute_truth_tables.py`는 이 스크립트의 호환 래퍼로, 같은 main을 호출해 truth table을 포함한 동일 산출을 낸다.

`compute_robustness_stats.py`는 같은 판정 CSV에서 final 통계를 보완하는 강건성 지표를 산출한다 — 모델 간 일치도(Fleiss·Cohen κ), 효과크기 95% CI(비율차 Newcombe·OR Woolf·Cramér’s V 부트스트랩), 합의 정의 민감도(만장일치/과반/1표 이상), 서종(book·部) 층화 Mantel-Haenszel OR과 Woolf 동질성·sign test. LLM 호출이 없고 표준 라이브러리만 쓰며 `--source anon`으로 동일 수치가 재현된다. 출력은 `results/robustness_stats.json`·`results/ROBUSTNESS_REPORT.md`, 실행 기록은 `logs/robustness_stats.jsonl`.

`freeze_run.py`는 기준 산출물과 로컬 입력 파일의 SHA-256 매니페스트(`RUN_MANIFEST.json`)·물리 백업을 만들어 동결 기준을 남긴다. 재현 샌드박스(`sandbox/`)는 이 기준과 재실행 결과를 비교하며, 라이브 재현 환경 구성은 `sandbox/README.md`를 따른다.

## 데이터 준비

```bash
python scripts/prepare_sentence_dataset.py
python scripts/prepare_phrase_data.py
python scripts/prepare_full_corpus.py
```

`classify_premodern_markers.py`(전근대 원전 기준 현토 재분류 — phrase 단위 보조 분석용)와 `normalize_hyeonto.py`(현토 정규화 유틸리티)는 현행 재현 절차(§주요 재현 스크립트)에 포함되지 않는 보조 유틸이다.

## 보조 분석

```bash
python scripts/analyze_beomnon_heosa.py
python scripts/analyze_hada_metadata.py
python scripts/analyze_hada_metadata.py --check-existing
python scripts/audit_normalization_gaps.py
python scripts/sample_consensus_examples.py
python scripts/sample_dansa_examples.py
python scripts/summarize_sentence_patterns.py
```

`analyze_beomnon_heosa.py`는 기본적으로 `汎論以斷/하나니라` 296건에서 `夫`·`凡`·`蓋`·`大抵`의 원문 문자열 공기율을 계산하고 `results/beomnon_heosa_stats.json`을 쓴다. `--category-all`을 붙이면 `하나니`와 `하나니라`를 합친 `汎論以斷` 전체 641건 기준으로 확인한다.

`analyze_hada_metadata.py`는 `data/sentence_normalized.csv`만 입력으로 사용한다.

## 질적 분석과 LightRAG 입력 구성

```bash
python analysis/scripts/build_embeddings_from_tsv.py
python analysis/scripts/build_section2_parallel_tsv.py
python analysis/scripts/build_section2_embeddings.py
python analysis/scripts/build_section2_clusters.py
python analysis/scripts/run_dci_queries.py
python analysis/scripts/run_dci_cross_comparison.py
python analysis/scripts/analyze_genre_by_category.py
python analysis/scripts/analyze_genre_controlled.py
python analysis/scripts/extract_translator_metadata.py
python analysis/scripts/audit_cells.py
python analysis/lightrag_out/run_all.py
python analysis/lightrag_out/build_percat_report.py
python analysis/lightrag_out/run_unified.py
python analysis/dci_out/build_dci_report.py
python analysis/build_comparison_report.py
```

LightRAG 폴더 내부에서는 디렉터리명이 이미 분석 방식을 표시하므로 파일명에서 `lightrag` 반복을 제거했다. 기본 실행은 `run_all.py`이고, 단일 범주 작업은 `run_category.py`, 질의 재실행은 `run_queries.py`, 통합 KG 실험은 `run_unified.py`를 사용한다.
