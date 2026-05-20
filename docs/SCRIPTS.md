# 스크립트 파일명 기준

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

## 주요 재현 스크립트

```bash
python scripts/preflight_llm_pipeline.py
python scripts/build_llm_input_manifests.py
python scripts/run_multimodel_judgments.py
python scripts/run_supplement_judgments.py
python scripts/compute_final_stats.py
python scripts/export_anonymized_results.py
python scripts/verify_section2_results.py
```

`preflight_llm_pipeline.py`는 외부 API를 호출하지 않고 입력 파일, manifest, prompt 생성, O/X 파서, mock 판정 배치, 기존 결과 resume 상태를 점검한다.

`build_llm_input_manifests.py`는 현재 기준 LLM 입력 표본을 `data/llm_manifests/` 아래에 고정한다. 이 manifest는 보고된 수치의 exact 재현을 위한 로컬 전용 자료이며, 원문과 번역문을 포함하므로 git 추적 대상이 아니다.

표본 추출 기준과 여러 seed robustness 검증 설계는 `docs/SAMPLING.md`에 정리한다.

## 데이터 준비

```bash
python scripts/prepare_sentence_dataset.py
python scripts/prepare_phrase_data.py
python scripts/prepare_full_corpus.py
```

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
python analysis/dci_out/build_dci_report.py
```
