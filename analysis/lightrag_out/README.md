# LightRAG 산출물 상태

LightRAG 실행 기준 입력은 `analysis/parallel_data_v2_cleaned.tsv` 11,327행이다.

재생성 순서:

```bash
python analysis/scripts/build_embeddings_from_tsv.py
python analysis/scripts/build_section2_clusters.py
python analysis/lightrag_out/run_lightrag_safe.py
python analysis/lightrag_out/build_report.py
```
