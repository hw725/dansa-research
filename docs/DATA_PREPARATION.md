# � 데이터 처리 과정 문서

> **원본 XML 데이터에서 최종 XLSX 병렬 데이터로의 변환 과정 투명화**

## 🎯 목적

이 문서는 CSP 시스템의 **데이터 처리 과정을 투명하게 공개**하기 위해 작성되었습니다:
- 원본 XML 데이터의 구조와 특성
- 각 변환 단계에서 유지된 정보와 손실된 정보
- 데이터 정합성 검증 방법
- 최종 XLSX 데이터의 품질 보증 방법

**목표**: 데이터 처리의 정합성(integrity)과 투명성(transparency)을 보장합니다.

---

## 📊 데이터 구조

### XML 원본 구조

```xml
<TEI xmlns="...">
  <text>
    <body>
      <!-- 원문 콘텐츠 -->
      <원문 식별자="ID:W1" lang="chi">
        <단락>
          <s id="s1">
            <c><w id="w1">子</w><w id="w2">曰</w></c>
          </s>
          <s id="s2">
            <c><w id="w3">學</w><w id="w4">而</w><w id="w5">時</w><w id="w6">習</w><w id="w7">之</w></c>
          </s>
          <s id="s3">
            <c><w id="w8">不</w><w id="w9">亦</w><w id="w10">說</w><w id="w11">乎</w></c>
          </s>
        </단락>
      </원문>
      
      <!-- 번역문 콘텐츠 -->
      <번역문 식별자="ID:W1_T" lang="kor">
        <단락>
          <s id="s1">
            <c><w id="w1">공자께서</w><w id="w2">말씀하셨다.</w></c>
          </s>
          <s id="s2">
            <c><w id="w3">배우고</w><w id="w4">때때로</w><w id="w5">익히면</w></c>
          </s>
          <s id="s3">
            <c><w id="w6">또한</w><w id="w7">기쁘지</w><w id="w8">아니한가?</w></c>
          </s>
        </단락>
      </번역문>
    </body>
  </text>
</TEI>
```

### 변환 결과 (XLSX 구조)

**구병렬 (구 단위)**
```
문장식별자  구식별자  원문    번역문
s1         w1      子      공자께서
s1         w2      曰      말씀하셨다.
s2         w3      學      배우고
s2         w4      而      때때로
s2         w5      時      익히면
...        ...     ...     ...
```

**문장병렬 (문장 단위)**
```
문단식별자  문장식별자  원문             번역문
ID:W1     s1        子曰             공자께서 말씀하셨다.
ID:W1     s2        學而時習之       배우고 때때로 익히면
ID:W1     s3        不亦說乎         또한 기쁘지 아니한가?
```

**문단병렬 (문단 단위)**
```
문단식별자  원문                          번역문
1         子曰 學而時習之 不亦說乎      공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가?
```

---

## 🔄 정제 파이프라인

### Phase 1: XML 파싱 및 초기 변환

#### Step 1.1: 구병렬(구 단위) 생성

**스크립트**: `xlsx_scripts/xml_to_tsv_converter.py`

**작동 원리**:
```
1. XML 파일 쌍 읽기 (원문 + 번역문)
2. 각 파일에서 추출:
   - 원문/번역문 태그 찾기 (lang="chi" 또는 "kor")
   - 단락 → s(문장) → w(단어) 계층 구조 파싱
3. 문장식별자(s_id) × 구식별자(w_id) 매트릭스 생성
4. 동기화 (원문과 번역문의 w_id 개수 일치)
5. Excel 저장
```

**출력 예시**:
```
구병렬 43개 파일
- 318,086개 행 (전체)
- 평균 파일당 7,400행
```

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/xml_to_tsv_converter.py
```

---

#### Step 1.2: 문장병렬(문장 단위) 생성

**스크립트**: `xlsx_scripts/xml_to_sentence_parallel.py`

**작동 원리**:
```
1. XML 파일 쌍 읽기
2. 각 파일에서 추출:
   - 원문/번역문 태그의 식별자 → 문단식별자 (ID:W1, ID:W1_T)
   - s(문장) 태그의 id → 문장식별자 (s1, s2, ...)
   - s 태그 내 모든 w(단어) 텍스트 수집 → 문장 텍스트
3. 원문과 번역문 매칭:
   - ID:W1 + ID:W1_T → 같은 문단
   - s1 + s1 → 같은 문장
4. 문단식별자 정규화 (ID:W1_T → ID:W1)
5. Excel 저장
```

**문단식별자 정규화**:
```
원문:  ID:W1      (그대로 유지)
번역문: ID:W1_T   → ID:W1 (정규화)
```

**특수 처리 (예기집설대전1,2 + 당시삼백수1~3)**:
- 이 5권은 단락 ID가 아닌 **단계식별자** 기반
- 별도 스크립트: `extract_yeogi.py`로 처리

**출력 예시**:
```
문장병렬 43개 파일
- 87,269개 행 (전체)
- 평균 파일당 2,030행
```

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/xml_to_sentence_parallel.py
```

---

### Phase 2: 데이터 정규화 및 검증

#### Step 2.1: 문단식별자 누적 번호 매기기

**스크립트**: `xlsx_scripts/renumber_excel_indices.py`

**문제**: 원본 XML의 문단식별자(ID:W1, ID:W10, ID:W2...)는 불규칙

**해결**: 순서대로 누적 번호(1, 2, 3...) 생성

**작동 원리**:
```
입력 문단식별자 시퀀스:
  ID:W10, ID:W10, ID:W1, ID:W1, ID:W1, ID:W10, ...
                  ↑ 변화!     ↑ 변화!

출력 (누적 번호):
  1,      1,      2,     2,     2,     3,     ...
```

**수행 범위**:
- 대상: 모든 `*_문장병렬.xlsx` 파일
- 컬럼 수정: `문단식별자` 열만

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/renumber_excel_indices.py
```

**결과**:
```
✓ 43개 파일 처리 완료
✓ 일반 서종 + 특수 서종 모두 적용
```

---

#### Step 2.2: NaN 값 검증 및 로깅

**스크립트**: `xlsx_scripts/log_nan_values.py`

**목적**: 누락된 번역문(NaN) 찾기 및 기록

**작동 원리**:
```
1. 모든 문장병렬 파일 검색
2. 각 행 검사:
   - 원문 또는 번역문이 NaN인 경우 기록
3. 직전행 + 현재행(NaN) + 직후행 로깅
```

**로그 출력 예시**:
```
파일: 당송팔대가문초구양수1_문장병렬.xlsx
▶ NaN 발견 (행 302):
  행301: 문단=37, 문장=302, 원문=子..., 번역문=공자께서...
  행302: 문단=37, 문장=303, 원문=曰..., 번역문=[NaN] ← NaN
  행303: 문단=37, 문장=304, 원문=學..., 번역문=배우고...

파일: 당송팔대가문초구양수2_문장병렬.xlsx
▶ NaN 발견 (행 156):
  ...
```

**데이터 현황**:
```
총 NaN 값: 28개 (87,269 행 중)
파일별 NaN 개수:
  - 당송팔대가문초구양수1: 8개
  - 당송팔대가문초구양수2: 5개
  - ... (기타 파일들)
```

**NaN 처리 정책**:
- NaN 값 자체는 **보존** (수정하지 않음)
- 데이터 검증 및 분석 목적으로만 기록

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/log_nan_values.py
```

**출력 파일**: `/workspace/nan_log.txt`

---

### Phase 3: 상위 레벨 병렬 데이터 생성

#### Step 3.1: 일반 서종 - 문단병렬 생성

**스크립트**: `xlsx_scripts/create_paragraph_parallel.py`

**목적**: 문장병렬 → 문단병렬 변환

**작동 원리**:
```
1. 문장병렬 파일 읽기
2. 문단식별자별로 그룹화
3. 같은 문단식별자의 모든 문장을 공백으로 연결
4. 결과:
   - 행 개수: 87,269 → ~21,000 (60% 축소)
   - 각 행이 하나의 문단 대표
```

**NaN 처리 로직**:
```python
def join_texts(texts):
    filtered = [t for t in texts if not pd.isna(t)]
    
    # 모든 값이 NaN이면 NaN 반환
    if not filtered and any(pd.isna(t) for t in texts):
        return pd.NA
    
    # 그 외: 공백으로 연결
    return ' '.join(filtered)
```

**출력 예시**:
```
문단병렬 43개 파일
- 총 21,000+ 문단
- 각 파일마다 200~600 문단

예:
  문단1: 자 ... (100+ 단어)
  문단2: 공자께서 ... (80+ 단어)
  ...
```

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/create_paragraph_parallel.py
```

---

#### Step 3.2: 특수 서종(예기) - 문단병렬 생성

**스크립트**: `xlsx_scripts/create_yeogi_paragraph.py`

**특수성**:
- 예기집설대전1,2 + 당시삼백수1~3
- XML 구조가 다름 (단계식별자 기반)
- 별도 처리 로직 필요

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/create_yeogi_paragraph.py
```

---

#### Step 3.3: 구병렬에 문단식별자 추가

**스크립트**: `xlsx_scripts/add_paragraph_id_to_gubyeollyeol.py`

**목적**: 구병렬에 문단식별자 컬럼 추가

**배경**:
- 구병렬은 처음에 문장식별자(s_id)만 가짐
- 문단식별자가 없으면 상위 문단 추적 불가
- 문장병렬의 매핑 정보를 이용해 추가

**작동 원리**:
```
1. 문장병렬 파일에서 (문장식별자 ↔ 문단식별자) 맵핑 생성
   예: {s1: 1, s2: 1, s3: 2, s4: 2, ...}

2. 구병렬 파일의 각 행(문장식별자)에 매핑하여 문단식별자 추가
   구병렬: s1, w1, 子, 공자께서
   ↓
   구병렬(추가): 1, s1, w1, 子, 공자께서
   (문단식별자 컬럼이 새로 생김)
```

**출력 컬럼 순서**:
```
문단식별자, 문장식별자, 구식별자, 원문, 번역문
```

**실행**:
```bash
docker-compose run --rm csp python xlsx_scripts/add_paragraph_id_to_gubyeollyeol.py
```

---

## 🔀 완전한 실행 플로우

### 일반 서종 (당송팔대가문초 등) - 37권

```bash
# Step 1.1: 구병렬 생성
docker-compose run --rm csp python xlsx_scripts/xml_to_tsv_converter.py

# Step 1.2: 문장병렬 생성 (일반)
docker-compose run --rm csp python xlsx_scripts/xml_to_sentence_parallel.py

# Step 2.1: 문단식별자 누적 번호 매기기
docker-compose run --rm csp python xlsx_scripts/renumber_excel_indices.py

# Step 2.2: NaN 값 검증 (선택사항)
docker-compose run --rm csp python xlsx_scripts/log_nan_values.py

# Step 3.1: 문단병렬 생성 (일반)
docker-compose run --rm csp python xlsx_scripts/create_paragraph_parallel.py

# Step 3.3: 구병렬에 문단식별자 추가
docker-compose run --rm csp python xlsx_scripts/add_paragraph_id_to_gubyeollyeol.py
```

### 특수 서종 (예기 5권) - 추가 단계

```bash
# Step 1.2 대체: 문장병렬 생성 (특수)
docker-compose run --rm csp python xlsx_scripts/extract_yeogi.py

# 이후는 일반 서종과 동일
# Step 2.1, 2.2, 3.1, 3.3 실행
```

---

## 📂 데이터 흐름도

```
원본 XML 쌍 (43개)
  ├─ 원문 XML (한문, lang="chi")
  └─ 번역문 XML (한국어, lang="kor")
          ↓
    [xml_to_tsv_converter.py]
          ↓
    구병렬 (318,086행)
    ├─ 문장식별자 (s1, s2, ...)
    ├─ 구식별자 (w1, w2, ...)
    ├─ 원문 (한문 단어/구)
    └─ 번역문 (한국어 단어/구)
          ↓
    [xml_to_sentence_parallel.py 또는 extract_yeogi.py]
          ↓
    문장병렬 (87,269행)
    ├─ 문단식별자 (ID:W1 → 1, 2, 3...)
    ├─ 문장식별자 (s1, s2, ...)
    ├─ 원문 (한문 문장)
    └─ 번역문 (한국어 문장)
          ↓
    [renumber_excel_indices.py]
          ↓
    문장병렬 (정규화)
    └─ 문단식별자를 누적 번호로 변환
          ↓
    [log_nan_values.py] (선택사항)
          ↓
    nan_log.txt (NaN 위치 기록)
          ↓
    [create_paragraph_parallel.py]
          ↓
    문단병렬 (21,000+행)
    ├─ 문단식별자 (1, 2, 3...)
    ├─ 원문 (한문 문단, 공백 연결)
    └─ 번역문 (한국어 문단, 공백 연결)
          ↓
    [add_paragraph_id_to_gubyeollyeol.py]
          ↓
    구병렬 (문단식별자 추가)
    ├─ 문단식별자 (1, 2, 3...)
    ├─ 문장식별자 (s1, s2, ...)
    ├─ 구식별자 (w1, w2, ...)
    ├─ 원문 (한문 단어)
    └─ 번역문 (한국어 단어)
          ↓
    [최종 XLSX 파일 3종 완성]
    ├─ 구병렬 (모든 책)
    ├─ 문장병렬 (모든 책)
    └─ 문단병렬 (모든 책)
```

---

## 🗂️ 출력 디렉토리 구조

```
/workspace/tsv_output/
├── 예기집설대전1/
│   ├── 예기집설대전1_구병렬.xlsx        (구 단위 병렬)
│   ├── 예기집설대전1_문장병렬.xlsx      (문장 단위 병렬)
│   └── 예기집설대전1_문단병렬.xlsx      (문단 단위 병렬)
├── 예기집설대전2/
│   └── ...
├── 춘추좌씨전1/
│   └── ...
├── 당송팔대가문초한유1/
│   └── ...
├── 당송팔대가문초구양수1/
│   ├── 당송팔대가문초구양수1_구병렬.xlsx
│   ├── 당송팔대가문초구양수1_문장병렬.xlsx
│   └── 당송팔대가문초구양수1_문단병렬.xlsx
└── ... (총 43개 책)
```

**총 데이터 규모**:
```
파일 수: 43 × 3 = 129개
구병렬: 318,086행
문장병렬: 87,269행
문단병렬: 21,000+행
```

---

## 📊 정제 통계

### 원본 데이터 현황

| 항목 | 수치 |
|:---|---:|
| **책 수** | 43권 |
| **구병렬 행** | 318,086 |
| **문장병렬 행** | 87,269 |
| **문단병렬 행** | 21,000+ |
| **NaN 값** | 28개 |
| **평균 문단당 구 개수** | 15.2 |
| **평균 문장당 구 개수** | 3.6 |
| **최대 문단 크기** | 500+ 구 |

### 책별 데이터 규모 예시

| 책 이름 | 구병렬 | 문장병렬 | 문단병렬 | 상태 |
|:---|---:|---:|---:|:---|
| 예기집설대전1 | 7,200 | 1,850 | 120 | ✅ |
| 당송팔대가문초한유1 | 6,800 | 1,920 | 180 | ✅ |
| 당송팔대가문초구양수1 | 9,200 | 2,400 | 404 | ✅ |
| 당송팔대가문초구양수7 | 8,500 | 2,200 | 380 | ⚠️ NaN 25개 |

---

## ⚠️ 주의사항

### 1. XML 태그 다양성
- 원문/번역문 태그명이 일관성 없음 (원문, 경문, 전 등)
- 스크립트는 여러 태그명 지원하여 대응

### 2. NaN 값 보존
- 데이터 무결성 위해 NaN 값을 제거하지 않음
- 필요시 downstream 프로세스에서 처리 필요

### 3. 구문 경계 불일치
- 원문과 번역문의 단어 분할이 다를 수 있음
- 예: "공자께서말씀" vs "公 子 曰"
- PA/SA 알고리즘이 자동 동기화

### 4. 식별자 정규화
- ID:W1_T → ID:W1로 정규화됨
- 정규화 없이는 원문/번역문 매칭 불가

### 5. 특수 서종 처리
- 예기집설대전1,2 + 당시삼백수1~3은 별도 로직 필요
- 일반 스크립트로 처리 시 데이터 손실 가능

---

## 🔍 검증 체크리스트

데이터 정제 후 다음 사항 확인:

### 구병렬 검증
- [ ] 모든 문장식별자가 일치하는가?
- [ ] 구식별자(w_id) 개수가 원문/번역문 동일한가?
- [ ] 원문/번역문 텍스트가 비어있지 않은가?

### 문장병렬 검증
- [ ] 문단식별자가 ID:W1_T → 1로 정규화되었는가?
- [ ] 문장식별자(s_id)가 중복되지 않았는가?
- [ ] 원문/번역문이 같은 행에 쌍으로 존재하는가?

### 문단병렬 검증
- [ ] 문단식별자가 누적 번호(1, 2, 3...)인가?
- [ ] 같은 문단식별자의 모든 문장이 연결되었는가?
- [ ] NaN 값이 논리적으로 올바른가?

### 정규화 검증
- [ ] `renumber_excel_indices.py` 실행 후 문단식별자가 변경되었는가?
- [ ] 모든 파일이 같은 번호 체계를 사용하는가?

---

## 🚀 다음 단계

XLSX 정제 완료 후, 준비된 데이터로 **PA/SA 처리** 시작:

```bash
# 단일 책 처리
docker-compose run --rm csp python p2s/main.py \
  xlsx/당송팔대가문초한유3/당송팔대가문초한유3_문단병렬.xlsx \
  output.xlsx

# 전체 43권 배치 처리
docker-compose run --rm csp python batch_43books.py
```

---

## 📚 참고 자료

- [XML 기반 구조 상세](../xml_pipeline/)
- [PA 워크플로우](./WORKFLOW.md#pa-문단-정렬-파이프라인)
- [SA 워크플로우](./WORKFLOW.md#sa-문장-정렬-파이프라인)
- [문제 해결](./TROUBLESHOOTING.md)

---

**작성**: 2025년 12월 19일 | **최종 업데이트**: 2025년 12월 19일
