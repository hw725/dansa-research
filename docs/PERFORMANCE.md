# ⚡ 성능 최적화 가이드

## 시스템 요구사항

### 최소 사양 (단일 책)

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | CUDA 11.0+ | RTX 3060+ (12GB) |
| RAM | 8GB | 32GB |
| SSD | 50GB | 200GB |
| 처리 시간 | ~10분 | ~3분 |

### 권장 사양 (배치 처리 43권)

| 항목 | 최소 | 최적 |
|------|------|------|
| GPU | RTX 2080 (8GB) | RTX 3090 (24GB) |
| RAM | 32GB | 64GB+ |
| SSD | 500GB | 1TB |
| 소요 시간 | ~10시간 | ~3시간 |

---

## 🚀 빠른 성능 체크

```bash
# 1. GPU 확인
nvidia-smi

# 2. 메모리 확인
free -h

# 3. SSD 용량 확인
df -h

# 4. 단일 책 테스트 (시간 측정)
time python p2s/main.py \
  xlsx/당송팔대가문초한유3/당송팔대가문초한유3_문단병렬.xlsx \
  output.xlsx \
  --embedder bge
```

---

## 📊 PA 성능 최적화

### 1. 배치 크기 조절

```bash
# 작은 배치 (메모리 부족 시)
python p2s/main.py input.xlsx output.xlsx --batch-size 16

# 중간 배치 (균형)
python p2s/main.py input.xlsx output.xlsx --batch-size 64

# 큰 배치 (메모리 충분 시)
python p2s/main.py input.xlsx output.xlsx --batch-size 256
```

**영향**:
- 배치 ↑ → 속도 ↑, 메모리 ↑
- 배치 ↓ → 속도 ↓, 메모리 ↓

**추천**:
- RTX 3060 (12GB): 64-128
- RTX 3090 (24GB): 256-512

---

### 2. 워커 수 조절

```bash
# 최소 워커 (메모리 절약)
python p2s/main.py input.xlsx output.xlsx --max-workers 1

# 최적 워커 (일반적)
python p2s/main.py input.xlsx output.xlsx --max-workers 4

# 최대 워커 (고성능)
python p2s/main.py input.xlsx output.xlsx --max-workers 8
```

**영향**:
- 워커 ↑ → 병렬 처리 ↑, 메모리 ↑
- 워커 ↓ → 메모리 절약, 속도 ↓

**추천**: CPU 코어 수의 절반

---

### 3. 임베딩 모델 선택

```bash
# BGE (권장, 정확도 높음)
python p2s/main.py input.xlsx output.xlsx --embedder bge

# m3e (빠름)
python p2s/main.py input.xlsx output.xlsx --embedder m3e

# 캐시 활용 (반복 처리 시 빠름)
python p2s/main.py input.xlsx output.xlsx --embedder bge --cache-embeddings
```

**모델 비교**:

| 모델 | 속도 | 정확도 | 메모리 |
|------|------|--------|--------|
| BGE | 1.0x | 95% | 2GB |
| M3E | 1.5x | 92% | 1.5GB |

---

### 4. GPU 메모리 최대화

```bash
# 1. CUDA 캐시 비우기
python -c "import torch; torch.cuda.empty_cache()"

# 2. GPU 메모리 추적
nvidia-smi --query-gpu=memory.free --format=csv --loop=1

# 3. 혼합 정밀도 (fp16, 더 빠름)
export TORCH_DTYPE=fp16
python p2s/main.py input.xlsx output.xlsx

# 4. 단일 GPU 사용
export CUDA_VISIBLE_DEVICES=0
python p2s/main.py input.xlsx output.xlsx
```

---

## 📊 SA 성능 최적화

### 1. 임베딩 캐시

```bash
# 첫 실행 (캐시 생성)
python s2p/main.py input.xlsx output.xlsx

# 두 번째 실행 (캐시 사용, 빠름)
python s2p/main.py input.xlsx output.xlsx  # 자동으로 캐시 사용
```

**캐시 위치**:
```
embeddings_cache_openai/
embeddings_cache_similarity/
```

---

### 2. 매칭 알고리즘 최적화

```python
# 코사인 유사도 (기본, 빠름)
similarity = cosine_similarity(a, b)

# 유클리드 거리 (느림)
similarity = euclidean_distance(a, b)
```

---

## 🔧 배치 처리 최적화

### 1. 병렬 처리 (여러 책 동시 처리)

```bash
# 순차 처리 (기본)
python batch_43books.py

# 병렬 처리 (GNU Parallel)
ls xlsx/*/`*_문단병렬.xlsx | parallel 'python p2s/main.py {}'
```

**주의**: GPU 메모리 공유로 인한 성능 저하 가능

---

### 2. 선택적 처리 (특정 책만)

```bash
# 1-10권만 처리
python batch_43books.py --start 1 --end 10

# 특정 책만 처리
python batch_43books.py --books 당송팔대가문초한유3 춘추좌씨전1
```

---

### 3. 평가 건너뛰기

```bash
# PA만 실행 (평가 제외)
python p2s/main.py input.xlsx output.xlsx --skip-eval

# 배치에서도 마찬가지
export SKIP_EVAL=1
python batch_43books.py
```

---

## 💾 디스크 최적화

### 1. 임시 파일 정리

```bash
# 캐시 정리
rm -rf embeddings_cache_*/

# 로그 정리
rm -rf logs/*.log

# 테스트 결과 정리
rm -rf test_results/
```

**용량 절감**:
- 임베딩 캐시: 1-2GB
- 로그: 100-500MB
- 테스트 결과: 100-500MB

### 2. 결과 압축

```bash
# 배치 결과 압축
tar -czf xlsx_pipeline_results.tar.gz xlsx_pipeline_results/

# 용량 확인
du -sh xlsx_pipeline_results*
```

---

## 🐳 Docker 성능 최적화

### 1. GPU 지원 활성화

```bash
# docker-compose.yml 확인
services:
  csp:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. 메모리 제한

```yaml
# docker-compose.yml
services:
  csp:
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 24G
```

### 3. 빌드 최적화

```bash
# 이미지 다시 빌드 (캐시 무효화)
docker-compose build --no-cache

# 이미지 크기 확인
docker images
```

---

## 📈 성능 모니터링

### 1. 실시간 모니터링

```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# 프로세스별 메모리 사용
ps aux --sort=-%mem | head -20

# 디스크 사용량
watch -n 1 df -h
```

### 2. 로그 분석

```bash
# PA 처리 시간 추출
grep "처리 시간" logs/*.log | awk '{print $NF}'

# 메모리 피크 확인
grep "메모리" logs/*.log
```

---

## 🎯 최적화 체크리스트

### PA 처리 (단일 책)

- [ ] GPU: `nvidia-smi` 확인 (메모리 충분?)
- [ ] 배치 크기: 현재 시스템의 50%부터 시작
- [ ] 워커 수: CPU 코어 수 확인 (권장: 절반)
- [ ] 모델: BGE vs M3E 선택 (정확도 우선 vs 속도 우선)
- [ ] 캐시: 첫 실행은 느림 (임베딩 다운로드)

### 배치 처리 (43권)

- [ ] 디스크 공간: 500GB 이상 확인
- [ ] 메모리: 32GB 이상 권장
- [ ] GPU: RTX 3060+ 권장
- [ ] 평가 스킵: 필요시 `--skip-eval`
- [ ] 모니터링: `watch -n 1 nvidia-smi` 실행

### 최적화 전후 비교

```bash
# 최적화 전
time python p2s/main.py input.xlsx output.xlsx --batch-size 32
# real    10m42s

# 최적화 후
time python p2s/main.py input.xlsx output.xlsx --batch-size 256 --max-workers 8
# real     3m15s
```

---

## 🔍 성능 벤치마크

### 단일 책 처리 시간

| 책 | 크기 | PA | SA | 합계 |
|----|------|----|----|------|
| 당송팔대가문초한유3 | 500 문단 | 3분 | 2분 | 5분 |
| 춘추좌씨전1 | 1000+ 문단 | 8분 | 5분 | 13분 |
| 자치통감강목1 | 1500+ 문단 | 12분 | 8분 | 20분 |

### 배치 처리 시간 (43권)

| 설정 | GPU | 배치크기 | 워커 | 예상 시간 |
|------|-----|---------|------|----------|
| 기본 | RTX 3060 | 64 | 4 | ~8시간 |
| 최적화 | RTX 3090 | 256 | 8 | ~3시간 |

---

## 💡 팁과 트릭

### 1. 첫 실행 준비

```bash
# 임베딩 모델 사전 다운로드 (네트워크 시간 절약)
python -c "from transformers import AutoTokenizer, AutoModel; \
           AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5')"
```

### 2. 중단 후 재시작

```bash
# 이미 처리된 책 확인
ls xlsx_pipeline_results/*/

# 추가 책만 처리하도록 스크립트 수정
# batch_43books.py의 books 리스트 수정
```

### 3. 메모리 누수 확인

```bash
# 메모리 사용 추이 확인
ps -p <PID> -o %mem,rss | watch 'cat'

# 누수 발견 시 재시작
docker-compose restart csp
```

---

**최근 업데이트**: 2025년 12월 19일 - XLSX 기반 완전 재정리
