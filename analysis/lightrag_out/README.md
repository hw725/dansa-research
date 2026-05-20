# LightRAG 산출물 상태

LightRAG 실행 스크립트는 유지하지만, 이전 `results/` 출력과 로그는 `analysis/parallel_data_v2_cleaned.tsv` 11,327행 기준과 맞지 않아 아카이브로 이동했다.

보존 위치:

`archive/2026-05-20_csp_sync/dansa-research_legacy/analysis/lightrag_out/`

재생성 순서:

```bash
python analysis/scripts/build_embeddings_from_tsv.py
python analysis/scripts/build_section2_clusters.py
python analysis/lightrag_out/run_lightrag_safe.py
python analysis/lightrag_out/build_report.py
```
