# LightRAG 산출물 상태

LightRAG 실행 기준 입력은 `analysis/parallel_data_v2_cleaned.tsv` 11,327행이다.

재생성 순서:

```bash
python analysis/scripts/build_embeddings_from_tsv.py
python analysis/scripts/build_section2_clusters.py
python analysis/lightrag_out/run_all.py
python analysis/lightrag_out/build_percat_report.py
```

## 스크립트

| 파일 | 역할 |
|---|---|
| `run_all.py` | 4범주를 각각 독립 subprocess로 실행하는 기본 진입점 |
| `run_category.py` | 단일 범주 KG 구축과 Q1~Q6 질의 실행 |
| `run_queries.py` | 구축된 범주별 KG에서 질의만 재실행 |
| `run_unified.py` | 4범주 통합 KG와 CQ1~CQ7 질의 실행 |
| `build_percat_report.py` | 범주별 질의 결과(24파일)를 `REPORT_v4.md`로 통합 — 정본 per-category 빌더 |
| `repair_kv.py` | 범주별 KV 저장소 교차오염 정리용 보수 스크립트 |

per-category 빌더는 `build_percat_report.py` 하나다. 구 `build_report.py`(→`REPORT.md`)와 로컬에서 폐기된 unified(v5) 빌더는 제거했고, v5 코드는 `archive/`에, 결과물 `REPORT_v5.md`는 로컬에 보존했다.
