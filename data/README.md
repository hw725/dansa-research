# data — 입력 데이터

분석 파이프라인의 입력 데이터를 담는 폴더다. 원문(한문)은 공개 사료이지만 번역문은 한국고전번역원 국역으로 미공개·저작권 자료이므로, 번역문을 포함한 raw CSV는 git 추적 대상이 아니다. 따라서 **클론에는 `book_names.txt` 하나만 포함**되고, 나머지는 로컬에서 생성하거나 별도로 입수해야 한다.

## 파일 목록

| 파일 | 추적 | 행수 | 내용 | 생성·입수 |
|---|:---:|---:|---|---|
| `book_names.txt` | ✅ | 52 | 코퍼스 서명 목록 | 저장소에 포함 |
| `sentence_normalized.csv` | ❌ | 150,545 | sentence 기준 정제 입력 (원문·번역문·marker·dansa_category) | `scripts/prepare_sentence_dataset.py` |
| `phrase_normalized.csv` | ❌ | 643,357 | phrase(구) 기준 정제 입력 | `scripts/prepare_phrase_data.py` |
| `supplement_section1_control_12.csv` | ❌ | 12건 | 섹션 1 대조군 보충 | `scripts/run_supplement_judgments.py` 입력 |
| `supplement_section2_control_465.csv` | ❌ | 465건 | 섹션 2 대조군 보충 | 〃 |
| `supplement_section3_control_30.csv` | ❌ | 30건 | 섹션 3 대조군 보충 | 〃 |
| `dansa_section1_judgments.csv` | ❌ | 5,169 | 섹션 1 단사 판정 데이터 (로컬 전용) | 파생 산출 |
| `dansa_section2_judgments.csv` | ❌ | 25,216 | 섹션 2 단사 판정 데이터 (로컬 전용) | 파생 산출 |
| `llm_manifests/` | ❌ | — | LLM 입력 표본 manifest (`section{1,2,3}_base.csv`) | `scripts/build_llm_input_manifests.py` |

대조군 보충 건수(12·465·30)는 [docs/REPRODUCE.md](../docs/REPRODUCE.md)의 정제 기준과 같다.

## 주요 컬럼

- `sentence_normalized.csv`: book, 문단식별자, 문장식별자, 원문, 번역문, marker_raw, marker_normalized, dansa_category, compound_tags
- `phrase_normalized.csv`: book, 문장식별자, 구식별자, 원문, 번역문, marker, marker_final, marker_normalized

## 데이터 입수와 공개

raw 입력은 미공개 병렬 코퍼스에서 생성된다. 코퍼스를 확보하면 위 `생성` 스크립트로 재생성하며, 원본 병렬 자료는 추후 한국고전번역원 경유로 공개 예정이다. 공개·공유에는 번역문을 SHA-256 해시로 치환한 `results/**/*_anon.csv`만 사용한다 (`scripts/export_anonymized_results.py`).

`llm_manifests/`는 보고된 수치의 exact 재현용 입력 표본이며 원문·번역문을 포함하므로 로컬 전용이다. 상세는 [docs/SAMPLING.md](../docs/SAMPLING.md)를 따른다.
