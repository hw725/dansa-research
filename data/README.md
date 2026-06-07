# data — 입력 데이터

분석 파이프라인의 입력 데이터를 담는 폴더다. 원문(한문)은 공개 사료이지만 번역문은 한국고전번역원 국역으로 미공개·저작권 자료이므로, 번역문이 평문으로 든 raw CSV는 git 추적 대상이 아니다.

대신 **전체 sentence 입력의 익명화본**(`sentence_normalized_anon.csv` — 번역문을 SHA-256 16자 해시로 치환, 원문·marker·메타데이터는 보존)을 추적해 공개한다. 클론만으로 입력 분포·marker 정규화·표본 추출 기반을 확인할 수 있다.

## 파일 목록

| 파일 | 추적 | 행수 | 내용 |
|---|:---:|---:|---|
| `sentence_normalized_anon.csv` | ✅ | 150,545 | **공개용 전체 sentence 입력** — 번역문 해시, 원문·marker_normalized·dansa_category·compound_tags 보존 |
| `book_names.txt` | ✅ | 52 | 코퍼스 서명 목록 |
| `sentence_normalized.csv` | ❌ | 150,545 | 번역문 평문 포함 raw (로컬 전용) |
| `phrase_normalized.csv` | ❌ | 643,357 | phrase(구) 기준 입력 (로컬 전용) |
| `supplement_section{1,2,3}_control_*.csv` | ❌ | 12·465·30건 | 섹션별 대조군 보충 |
| `dansa_section{1,2}_judgments.csv` | ❌ | 5,169·25,216 | 섹션별 단사 판정 (로컬 전용) |
| `llm_manifests/` | ❌ | — | LLM 입력 표본 manifest (`section{1,2,3}_base.csv`) |

대조군 보충 건수(12·465·30)는 [docs/REPRODUCE.md](../docs/REPRODUCE.md)의 정제 기준과 같다.

## 주요 컬럼

`book, 문단식별자, 문장식별자, 원문, 번역문, marker_raw, marker_normalized, dansa_category, compound_tags` — 익명화본은 `번역문`만 해시값이고 나머지는 동일하다.

## 생성·재현

```bash
python scripts/export_anonymized_results.py    # sentence_normalized_anon.csv + results/*_anon.csv 재생성
python scripts/prepare_sentence_dataset.py     # raw sentence_normalized.csv 재생성 (코퍼스 필요)
```

번역문을 해시로 치환하므로 익명화본으로는 임베딩·LightRAG·LLM 재판정을 할 수 없다(번역문 실텍스트 필요). 정량 통계는 marker·판정값만 쓰므로 익명화본과 익명 판정(`results/{model}/*_anon.csv`)만으로 재현된다 — 검증은 `python scripts/compute_final_stats.py --check --source anon`.

raw 입력은 미공개 병렬 코퍼스에서 생성되며, 원본 병렬 자료는 추후 한국고전번역원 경유로 공개 예정이다. 상세 절차는 [docs/REPRODUCE.md](../docs/REPRODUCE.md), 표본 기준은 [docs/SAMPLING.md](../docs/SAMPLING.md).
