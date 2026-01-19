# 🐛 문제 해결 가이드

## PA 관련 문제

### 1. PA 무결성 경고: "원문 손실 2.84%"

```
❌ 무결성 경고: 원문 손실 2.84%
```

**원인**: 이는 **정상**입니다. 공백 문자만 손실되었습니다.

**확인 방법**:
1. PA 출력 파일에서 `integrity_losses` 시트 확인
2. 모든 손실된 문자가 공백(space)인지 확인
3. 실제 한자/한글 손실이 있는지 확인

**기술 배경**:
- PA의 Word Span Slicing은 단어 경계(`\S+`)를 기준으로 작동
- 단어 경계 사이의 공백은 손실될 수 있음
- 원문과 번역문의 공백 배치가 다를 수 있음

**해결 방법**: 
- **아무것도 할 필요 없음** (정상 동작)
- 실제 문자 손실이 있으면 아래 "PA 문자 손실" 참고

---

### 2. PA 문자 손실: "한자/한글 손실됨"

```
원문 길이: 12,345
출력 길이: 12,000
손실: 345 문자 (공백 아님) ❌
```

**원인**: 일반적으로 발생하지 않음. 아래를 확인하세요:

1. **DP 알고리즘 오류**
   - 단어 할당 실패
   - 스팬 계산 오류

2. **입력 데이터 문제**
   - NaN 값 포함
   - 특수 문자 손상

3. **인코딩 문제**
   - UTF-8 변환 오류

**해결 방법**:
```bash
# 입력 파일 검증
python -c "
import pandas as pd
df = pd.read_excel('xlsx/책이름/책이름_문단병렬.xlsx')
print('NaN 개수:', df.isna().sum().sum())
print('원문 첫 100자:', df['원문'].iloc[0][:100])
"
```

---

### 3. PA 평가 F1 점수 낮음: "34.6% F1"

```
평가 결과:
- 매칭: 1845/1908 (96.7%)
- Source Mismatch: 98.7% ❌
- F1: 34.6%
```

**원인** (이미 수정됨): 
- 행 매칭 전략이 잘못되었음 (i=i matching)
- PA는 문장을 다르게 분할하므로 행 인덱스가 맞지 않음

**현재 상태** ✅:
- 소스 기반 스마트 매칭으로 개선
- F1 점수: 79.0% (개선됨)

**평가 결과 확인**:
```bash
python accuracy/accuracy_evaluator.py \
  xlsx/책이름/책이름_문장병렬.xlsx \
  xlsx_pipeline_results/책이름/책이름_PA_문장병렬.xlsx \
  --project pa \
  -o eval_result.xlsx
```

---

### 4. PA 타임아웃: "시간 초과"

```
timeout: Command exceeded 600 seconds
```

**원인**: 
- 큰 책 (1000+ 문단)
- GPU 부족
- 메모리 부족

**해결 방법**:

```bash
# 1. 타임아웃 증가 (docker-compose에서)
# timeout 600 → 1200

# 2. 배치 크기 감소
python p2s/main.py input.xlsx output.xlsx --batch-size 32

# 3. 워커 수 감소
python p2s/main.py input.xlsx output.xlsx --max-workers 2

# 4. GPU 메모리 확인
nvidia-smi
```

---

### 5. PA "문장 수 차이": GT 1908 vs Pred 1845

```
GT 행 수: 1908
예측 행 수: 1845
차이: -63 (63개 문장 감소)
```

**원인** (정상):
- PA는 **덜 엄격하게** 문장을 분할
- 마지막 문장 기준만 사용
- GT(문장병렬)는 더 세밀한 분할 기준 사용

**해석**:
```
PA: "其執以來。見此君子。" → 2개 문장
GT: "其執以來。見此君子。" → 3개 문장 (더 세밀)
```

**해결 방법**: 
- 분할 로직 개선 (필요시)
- 또는 평가 시 n:m 매칭으로 대응 (이미 구현됨)

---

## SA 관련 문제

### 1. SA "입력 파일 없음"

```
⚠️ SA 입력 파일 없음: xlsx/책이름/책이름_문장병렬.xlsx
```

**원인**:
1. PA가 아직 완료되지 않음
2. GT 문장병렬 파일이 없음
3. 파일명 오타

**해결 방법**:
```bash
# 1. 파일 확인
ls -la xlsx/책이름/

# 2. PA 실행 (필요시)
python p2s/main.py \
  xlsx/책이름/책이름_문단병렬.xlsx \
  xlsx/책이름/책이름_문장병렬.xlsx

# 3. GT 파일 복사 (평가용)
cp xlsx/책이름/책이름_문장병렬.xlsx \
   accuracy/sa_gt/책이름.xlsx
```

---

### 2. SA 평가 NaN 값

```
⚠️ NaN 발견 (행 46)
```

**원인**:
- 입력 데이터에 NaN 포함
- 임베딩 계산 실패

**확인 방법**:
```bash
python -c "
import pandas as pd
df = pd.read_excel('xyz.xlsx')
print(df[df.isna().any(axis=1)])  # NaN 포함 행 출력
"
```

**해결 방법**:
```bash
# NaN 행 제거
python -c "
import pandas as pd
df = pd.read_excel('input.xlsx')
df = df.dropna(subset=['원문', '번역문'])
df.to_excel('input_cleaned.xlsx', index=False)
"
```

---

## 배치 처리 문제

### 1. 배치 중단: 특정 책에서 멈춤

```
[5/43] 춘추좌씨전3
  PA 실행 중...
  (영구 멈춤)
```

**원인**:
1. 메모리 부족
2. GPU 메모리 부족
3. 잠금 파일 문제

**해결 방법**:

```bash
# 1. 메모리 확인
free -h  # Linux
wsl -e free -h  # Windows WSL

# 2. GPU 메모리 확인
nvidia-smi

# 3. 프로세스 종료 후 재시작
docker-compose down
docker-compose run csp python batch_43books.py
```

---

### 2. 배치 "PA 실패"

```
❌ PA 실패
```

**진단**:
```bash
# 같은 책으로 단일 테스트
python p2s/main.py \
  xlsx/책이름/책이름_문단병렬.xlsx \
  test_output.xlsx
```

**일반적인 원인**:
1. 입력 파일 손상
2. 메모리 부족
3. 의존성 부족

---

### 3. 배치 "SA 실패"

```
❌ SA 실패
```

**진단**:
```bash
# SA 단일 테스트
python s2p/main.py \
  xlsx/책이름/책이름_문장병렬.xlsx \
  test_output.xlsx
```

---

## 데이터 문제

### 1. NaN 값 많음

**확인**:
```bash
python -c "
import pandas as pd
for book in books:
    df = pd.read_excel(f'xlsx/{book}/{book}_문단병렬.xlsx')
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f'{book}: {nan_count}개 NaN')
"
```

**해결**:
```bash
# NaN 행 제거
python -c "
import pandas as pd
import glob
for file in glob.glob('xlsx/*/*.xlsx'):
    df = pd.read_excel(file)
    df_clean = df.dropna(subset=['원문', '번역문'])
    if len(df_clean) < len(df):
        print(f'Removed {len(df) - len(df_clean)} NaN rows from {file}')
        df_clean.to_excel(file, index=False)
"
```

---

### 2. 특수 문자 문제

```
❌ UnicodeEncodeError: 'utf-8' codec can't encode character
```

**원인**: 파일 인코딩 문제

**해결 방법**:
```bash
# 파일을 UTF-8로 재인코딩
python -c "
import pandas as pd
df = pd.read_excel('input.xlsx', dtype=str)
df.to_excel('output.xlsx', index=False, encoding='utf-8')
"
```

---

## 평가 문제

### 1. 정답 파일 경로 오류

```
⚠️ 정답 파일 없음 (건너뜀)
```

**원인**: 정답 파일을 찾을 수 없음

**경로 우선순위**:
1. `accuracy/pa_gt/{책이름}.xlsx` (첫 번째 선택)
2. `xlsx/{책이름}/{책이름}_문장병렬.xlsx` (GT용)

**해결**:
```bash
# 방법 1: GT 파일을 정답 디렉토리로 복사
mkdir -p accuracy/pa_gt
cp xlsx/책이름/책이름_문장병렬.xlsx accuracy/pa_gt/책이름.xlsx

# 방법 2: 환경변수로 경로 지정
export PA_GT_DIR=/path/to/gt
python batch_43books.py
```

---

### 2. 평가 점수 불일치

```
첫 평가: F1 34.6%
두 번째 평가: F1 79.0% (다른 코드)
```

**원인**: 매칭 전략 변경

**확인 방법**:
```bash
# 사용된 매칭 전략 확인
grep -n "smart_match" accuracy/accuracy_evaluator.py
```

---

## 성능 문제

### 1. 처리 속도 느림

```
예상: 3분/책
실제: 15분/책
```

**원인**:
1. GPU 부족
2. 임베딩 모델 로딩 시간
3. I/O 병목

**최적화**:
```bash
# 1. 배치 크기 증가
python p2s/main.py input.xlsx output.xlsx --batch-size 128

# 2. 워커 수 증가
python p2s/main.py input.xlsx output.xlsx --max-workers 8

# 3. GPU 메모리 최대화
export CUDA_VISIBLE_DEVICES=0
python p2s/main.py input.xlsx output.xlsx
```

---

### 2. 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결**:
```bash
# 배치 크기 감소
python p2s/main.py input.xlsx output.xlsx --batch-size 32

# GPU 메모리 정리
nvidia-smi --query-gpu=memory.free --format=csv
```

---

## 로그 확인

### Docker 로그

```bash
# 실시간 로그 확인
docker-compose logs -f csp

# 마지막 100줄
docker-compose logs --tail=100 csp

# 특정 패턴 필터
docker-compose logs csp | grep "ERROR"
```

### 로컬 로그

```bash
# 로그 파일 위치
logs/

# 최근 로그 확인
tail -100 logs/*.log
```

---

## 자주 묻는 질문

**Q: 배치 처리 중 오류 발생 시 어디서부터 다시 시작?**

A: 오류난 책부터 수동으로 재시작 가능합니다.
```bash
python p2s/main.py xlsx/책이름/책이름_문단병렬.xlsx output.xlsx
```

**Q: 평가 없이 PA/SA만 실행?**

A: GT 파일이 없으면 자동으로 건너뜀.
```bash
# 또는 직접 실행
python p2s/main.py input.xlsx output.xlsx
```

**Q: 결과 파일 구조 변경?**

A: `p2s/main.py`의 `save_results()` 함수 수정.

---

**최근 업데이트**: 2025년 12월 19일 - XLSX 기반 완전 재정리
