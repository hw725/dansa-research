# 연구 재현 가이드

본 문서는 2026-05-20 정리 이후의 `dansa-research` 기준 재현 절차만 남긴다. 이전 `CSP-dansa` v3 산출물과 구버전 백업은 `archive/2026-05-20_csp_sync/`에 보존되어 있다.

## 1. 기준 산출물

| 파일 | 성격 |
|---|---|
| `data/sentence_normalized.csv` | 정제 완료 sentence 기준 입력 150,545행 |
| `analysis/parallel_data_v2_cleaned.tsv` | 질적 분석용 정제 TSV 11,327행 |
| `results/hada_metadata_stats.json` | sentence 기준 ‘하다’ 메타데이터 통계 |
| `results/final_stats_v3.1_cleaned_balanced.json` | 최신 메타데이터 포함 최종 통계 |
| `results/cleaned_balanced_stats.json` | 논문 표 작성용 요약 통계 |
| `results/truth_tables_v3.1_cleaned_balanced.json` | 3모델 합의 truth table |
| `results/*/*_anon.csv` | 공개용 익명화 판정 CSV |

## 2. 데이터 상태

원본 병렬 코퍼스와 번역문 포함 raw CSV는 미공개 자료를 포함하므로 git 추적 대상이 아니다. 공개 또는 공유에는 `scripts/export_anonymized_results.py`로 생성한 `*_anon.csv`와 `*_anon.tsv`만 사용한다.

정제 기준은 다음과 같다.

- 섹션 1 대조군 12건, 섹션 2 대조군 465건, 섹션 3 대조군 30건 보충
- 섹션 1, 섹션 2, 섹션 3은 target/control 균형 표본으로 최종 통계 산출

## 3. 환경

```bash
pip install pandas numpy tqdm openai python-dotenv
```

현재 저장소의 검증용 통계 재계산과 익명화는 표준 라이브러리만으로 동작한다.

```bash
python scripts/compute_final_stats.py
python scripts/export_anonymized_results.py
```

LLM 재실행에는 `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`가 필요하다. 이 저장소에서는 로컬 `env` 파일을 읽을 수 있지만, 해당 파일은 git 추적 대상이 아니다.

## 4. 재현 순서

### 4.0 LLM 판정 파이프라인 preflight

실제 API를 호출하기 전에 입력, manifest, prompt, parser, mock batch, resume 상태를 확인한다.

```bash
python scripts/preflight_llm_pipeline.py
```

보고된 수치를 exact 재현하려면 현재 LLM 입력 표본 manifest를 사용한다.

```bash
python scripts/build_llm_input_manifests.py
```

manifest는 `data/llm_manifests/`에 저장되며 원문과 번역문을 포함하므로 로컬 전용이다. 새 랜덤 표본에서도 효과 크기가 유지되는지는 exact 재현이 아니라 별도 robustness 검증 문제다.

### 4.1 sentence 입력 준비

`data/sentence_normalized.csv`가 없거나 재생성해야 할 때만 실행한다.

```bash
python scripts/prepare_sentence_dataset.py
```

최신 sentence 입력은 이미 `data/sentence_normalized.csv`에 반영되어 있다. 원본 보존본과 이전 결과는 `archive/2026-05-20_csp_sync/`에서 확인한다.

### 4.1.1 ‘하다’ 메타데이터 분석

記史之斷 계열 ‘하다’ 통계는 phrase 파일을 사용하지 않는다. `data/sentence_normalized.csv`에서 `marker_normalized`가 ‘하다’로 끝나는 행만 집계한다.

```bash
python scripts/analyze_hada_metadata.py
python scripts/analyze_hada_metadata.py --check-existing
```

출력은 `results/hada_metadata_stats.json`이고 실행 로그는 `logs/hada_metadata_analysis.jsonl`에 JSONL로 남는다. `--check-existing`은 새로 계산한 값이 기존 JSON과 같은지 확인하는 dry-run 검증이다.

### 4.2 3모델 판정

```bash
python scripts/run_multimodel_judgments.py
```

이 단계는 `results/{gpt5mini,gemini,claude_sonnet}/` 아래에 모델별 판정 CSV를 생성한다. 보충 대조군 반영 전 중간 통계는 `results/intermediate_multimodel_stats.json`에 저장된다.

### 4.3 보충 대조군 판정

```bash
python scripts/run_supplement_judgments.py
```

입력은 다음 세 파일이다.

| 파일 | 건수 |
|---|---:|
| `data/supplement_section1_control_12.csv` | 12 |
| `data/supplement_section2_control_465.csv` | 465 |
| `data/supplement_section3_control_30.csv` | 30 |

### 4.4 최종 통계 산출

```bash
python scripts/compute_final_stats.py
```

이 스크립트는 base 판정과 supplement 판정을 병합한 뒤 다음 파일을 갱신한다.

| 출력 | 설명 |
|---|---|
| `results/cleaned_balanced_stats.json` | 섹션별 consensus와 per-model 통계 |
| `results/final_stats_v3.1_cleaned_balanced.json` | 메타데이터 포함 최종 통계 |
| `results/truth_tables_v3.1_cleaned_balanced.json` | consensus truth table |

### 4.5 익명화

```bash
python scripts/export_anonymized_results.py
```

`번역문` 컬럼을 SHA-256 16자 해시로 바꾼 `*_anon.csv`를 생성한다.

## 5. 최신 통계 요약

| 섹션 | Target n | Control n | Consensus O | χ² | V |
|---|---:|---:|---:|---:|---:|
| 섹션 1 游辭以斷 | 2,606 | 2,606 | 41.0% vs 6.2% | 1401.76 | 0.519 |
| 섹션 2 夬絶之斷 vs 微絶之斷 | 11,135 | 11,135 | 45.7% vs 27.7% | 986.45 | 0.210 |
| 섹션 3 汎論以斷 | 296 | 296 | 80.7% vs 46.3% | 88.68 | 0.387 |

‘하다’ 메타데이터 통계는 sentence 150,545행 기준 출현 12,988건이다. 서종별 출현율은 歷史書 24.56%, 文集 5.93%, 經傳 0.23%, 詩 0.03%, 기타 0.02%이며, 歷史書 대 非歷史書 검정은 χ² = 17574.78, p < 0.001이다.

## 6. 질적 분석

LightRAG와 DCI 구버전 산출물은 아카이브에 보존되어 있으며, 활성 보고서는 11,327행 정제 TSV 기준으로 다시 생성해야 한다. 정량 표에는 반드시 `results/final_stats_v3.1_cleaned_balanced.json`을 사용하고, 질적 서술의 출처 빈도는 TSV 실측값으로 다시 확인한다.

## 7. 폐기된 기준

다음 항목은 활성 기준에서 제외하고 아카이브에 보존했다.

- `results/final_stats_v3.json`
- `results/final_stats_v3.1_cleaned.json`
- `results/truth_tables_v3.json`
- `results/archive_old_prompt/`
- 기존 `data/dansa_section*_judgments_anon.csv`
- 모든 `*.bak` 파일

## 8. 스크립트 파일명 기준
현재 재현 명령과 보조 스크립트 목록은 docs/SCRIPTS.md를 기준으로 한다.
