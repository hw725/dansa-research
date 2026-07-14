# 연구 재현 가이드

본 문서는 현재 `dansa-research` 기준 재현 절차만 다룬다.

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
| `data/sentence_normalized_anon.csv` | 공개용 전체 sentence 입력 150,545행 (번역문 해시) |

## 2. 데이터 상태

원본 병렬 코퍼스와 번역문 포함 raw CSV는 미공개 자료를 포함하므로 git 추적 대상이 아니다. 공개 또는 공유에는 `scripts/export_anonymized_results.py`로 생성한 `*_anon.csv`만 사용한다.

정제 기준은 다음과 같다.

- 섹션 1 대조군 12건, 섹션 2 대조군 465건, 섹션 3 대조군 30건 보충
- 섹션 1, 섹션 2, 섹션 3은 target/control 균형 표본으로 최종 통계 산출

## 3. 환경

```bash
pip install -r requirements.txt
```

LLM 판정·보조 분석 스크립트는 `scipy`를 임포트하므로 개별 설치 대신 `requirements.txt` 전체 설치를 권장한다.

현재 저장소의 검증용 통계 재계산과 익명화는 표준 라이브러리만으로 동작한다.

```bash
python scripts/compute_final_stats.py
python scripts/export_anonymized_results.py
```

LLM 재실행에는 `OPENAI_API_KEY`(gpt-5-mini)와 `OPENROUTER_API_KEY`(gemini·claude sonnet — OpenRouter 경유)가 필요하다. 이 저장소에서는 로컬 `env` 파일을 읽을 수 있지만, 해당 파일은 git 추적 대상이 아니다.

## 4. 재현 순서

### 4.0 LLM 판정 파이프라인 preflight

실제 API를 호출하기 전에 입력, manifest, prompt, parser, mock batch, resume 상태를 확인한다.

```bash
python scripts/preflight_llm_pipeline.py
```

보고된 수치를 exact 재현하려면 현재 LLM 입력 표본 manifest를 사용한다. manifest가 있으면 사용자가 추출한 동일 표본을 그대로 다시 쓰며, 새 표본을 뽑지 않는다.

```bash
python scripts/build_llm_input_manifests.py
```

manifest는 `data/llm_manifests/`에 저장되며 원문과 번역문을 포함하므로 로컬 전용이다. manifest가 없을 때의 고정 seed fallback과 여러 seed robustness 검증은 아래 §8 표본 추출 기준을 따른다.

### 4.1 sentence 입력 준비

`data/sentence_normalized.csv`가 없거나 재생성해야 할 때만 실행한다.

```bash
python scripts/prepare_sentence_dataset.py
```

최신 sentence 입력은 이미 `data/sentence_normalized.csv`에 반영되어 있다.

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

원본 raw CSV가 없는 클론에서도 익명 판정 파일(`results/{model}/*_anon.csv`)만으로 동일 통계를 재계산할 수 있다. 통계는 marker_type과 판정값만 사용하므로 raw와 anon 결과가 같다.

```bash
python scripts/compute_final_stats.py --check --source anon   # 익명 파일로 재현 검증 (CHECK PASS)
```

`--source`는 입력을 고른다(`auto`: raw 있으면 raw, 없으면 anon / `raw` / `anon`). `--check`는 기준 JSON과 대조만 하고 파일을 쓰지 않는다.

### 4.5 익명화

```bash
python scripts/export_anonymized_results.py
```

`번역문` 컬럼을 SHA-256 16자 해시로 바꾼 `*_anon.csv`를 생성한다.

### 4.6 강건성·일치도 통계

기존 판정 CSV에서 final 통계를 보완하는 지표를 산출한다. LLM 호출은 없다.

```bash
python scripts/compute_robustness_stats.py                 # raw 우선 (auto)
python scripts/compute_robustness_stats.py --source anon   # 공개 익명본 — 동일 수치
```

| 산출 항목 | 내용 |
|---|---|
| 모델 간 일치도 | Fleiss κ(전체·군별), pairwise Cohen κ, 일치율 |
| 효과크기 95% CI | O율차 Newcombe CI, OR Woolf CI, Cramér’s V 부트스트랩 CI |
| 합의 정의 민감도 | O 정의를 만장일치/과반/1표 이상으로 바꿔 효과 방향 점검 |
| 서종 층화 | book·部 층화 Mantel-Haenszel OR, Woolf 동질성, book sign test, 모델별 MH |

출력은 `results/robustness_stats.json`과 `results/ROBUSTNESS_REPORT.md`, 실행 기록은 `logs/robustness_stats.jsonl`이다. 부트스트랩은 고정 seed(기본 20260611)라 재실행 수치가 같고, raw와 anon 소스의 결과 일치를 확인했다.

## 5. 최신 통계 요약

| 섹션 | Target n | Control n | Consensus O | χ² | V |
|---|---:|---:|---:|---:|---:|
| 섹션 1 游辭以斷 | 2,606 | 2,606 | 41.0% vs 6.2% | 1401.76 | 0.519 |
| 섹션 2 夬絶之斷 vs 微絶之斷 | 11,135 | 11,135 | 45.7% vs 27.7% | 986.45 | 0.210 |
| 섹션 3 汎論以斷 | 296 | 296 | 80.7% vs 46.3% | 88.68 | 0.387 |

‘하다’ 메타데이터 통계는 sentence 150,545행 기준 출현 12,988건이다. 서종별 출현율은 歷史書 24.56%, 文集 5.93%, 經傳 0.23%, 詩 0.03%, 기타 0.02%이며, 歷史書 대 非歷史書 검정은 χ² = 17574.78, p < 0.001이다.

## 6. 질적 분석

LightRAG와 DCI 보고서는 11,327행 정제 TSV 기준으로 생성한다. 정량 표에는 반드시 `results/final_stats_v3.1_cleaned_balanced.json`을 사용하고, 질적 서술의 출처 빈도는 TSV 실측값으로 다시 확인한다.

## 7. 스크립트 파일명 기준
현재 재현 명령과 보조 스크립트 목록은 scripts/README.md를 기준으로 한다.

## 8. 표본 추출 기준

표본 추출은 exact 재현과 robustness 검증을 분리한다.

### 8.1 Exact 재현

보고된 통계 수치를 다시 만들 때는 `data/llm_manifests/`의 manifest를 사용한다. manifest는 이미 LLM 판정에 사용한 행 목록이므로, 재현 실행에서는 새 표본을 뽑지 않고 같은 문장들을 그대로 다시 사용한다.

### 8.2 Manifest가 없을 때의 fallback

manifest가 없으면 `scripts/run_multimodel_judgments.py`가 고정 seed로 대조군을 추출한다. 실행마다 달라지는 랜덤이 아니라, 같은 입력 CSV와 같은 코드에서는 같은 표본이 나오는 재현 가능한 fallback이다.

| 섹션 | Target | Control |
|---|---|---|
| 섹션 1 | `游辭以斷/로다` 전체 | `微絶之斷/라`에서 로다 개수만큼 추출, seed 42 |
| 섹션 2 | `夬絶之斷/니라` 전체 | `微絶之斷/라`에서 니라 개수만큼 추출, seed 42 |
| 섹션 3 | `汎論以斷/하나니라` 전체 | `微絶之斷/라`에서 하나니라 개수만큼 추출, seed 99 |

이 fallback은 서종별 층화추출이 아니다. 서종별 같은 개수나 비율을 맞추는 검증은 별도 robustness 설계가 필요하다.

### 8.3 Robustness 검증

통계적 안정성 확인은 exact 재현과 다른 작업이다. 여러 seed로 대조군을 다시 뽑아도 효과 방향과 크기가 안정적인지 확인한다. 예: seed 1~100으로 대조군을 반복 추출하고 seed별 Consensus O, χ², Cramér’s V, target/control 비율 분포를 비교한다. 단, 현재 결과 CSV는 모든 `라` 후보의 LLM 판정을 포함하지 않으므로, 새 seed robustness에는 더 넓은 control 후보 풀의 판정을 먼저 만들어야 한다.

현재 표본 안에서의 안정성 점검 — 부트스트랩 CI, 합의 정의 민감도, 서종 층화 MH OR, 모델 간 일치도 — 은 §4.6의 `compute_robustness_stats.py`로 산출되며 익명 데이터만으로 재현된다. 새 seed 대조군 robustness는 위 제약대로 추가 LLM 판정이 필요한 별도 작업으로 남는다.
