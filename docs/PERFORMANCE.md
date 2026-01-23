## 📊 성능 최적화 가이드 (2026-01-23)

### 1. P2S (Paragraph-to-Sentence) 성능

#### 권장 설정
```bash
python -m p2s.main datasets/paragraph/test.csv output.xlsx \
    --use-boundary-model --max-workers 4 --batch-size 64
```

#### 벤치마크 (2,211 문단 처리 기준)
| 항목 | 사양 | 결과 |
|------|------|------|
| **처리 시간** | RTX 4090 (24GB) / Docker | **~25분** |
| **GPU 메모리** | Max Peak | ~6GB |
| **평가 지표** | F1 / 유사도 | **0.8413 / 0.9174** |

---

### 2. S2P (Sentence-to-Phrase) 성능

#### 권장 설정 (안정성 중심)
```bash
python -u -m s2p.main datasets/sentence/test.csv output.csv \
    --embedder bge --use-boundary-model \
    --chunk-size 300 --batch-size 32
```
> **Batch Size 팁**: Embeddings 연산 시 128까지 가능하나, Boundary Model 추론(Inference) 시 GPU OOM 방지를 위해 **32** 권장.

#### 벤치마크 (10,175 문장 처리 기준)
| 항목 | 사양 | 결과 |
|------|------|------|
| **처리 시간** | RTX 4090 (24GB) / Docker | **~4시간** (캐시 미사용 시 ~10-12시간) |
| **속도** | Throughput | ~40-50 rows/min (캐시 Hit 시) |
| **평가 지표** | F1 / 유사도 | **0.8091 / 0.8323** |

---

### 3. 디스크 및 캐시 최적화

#### 임베딩 캐시 전략 (BGE)
- **위치**: `~/.cache/huggingface` 및 로컬 `.cache`
- **전략**: **Disk Cache**는 청크 단위로 저장되어 I/O 병목을 줄임.
- **팁**: 첫 실행 이후 재실행 시 **약 3-4배 속도 향상** (임베딩 캐시 Hit).

#### 로그 및 체크포인트
- **로그**: `python -u` (Unbuffered) 옵션을 사용하여 실시간 로그 확인 권장 (Docker logs 지연 방지).
- **체크포인트**: `{output_filename}_checkpoint.csv`가 청크 단위로 자동 생성됨. 중단 시 마지막 체크포인트부터 이어서 실행 가능.

---

**최근 업데이트**: 2026년 01월 23일 - S2P/P2S 풀 코퍼스 벤치마크 반영
