# 📚 CSP 문서 가이드

> **XLSX 기반 문단 정렬(PA) 및 문장 정렬(SA) 시스템 완전 문서**

## 📖 문서 로드맵

### 🚀 처음 시작하는 사용자

1. **[README.md](README.md)** - 5분
   - 프로젝트 개요
   - 빠른 시작 가이드
   - 기본 명령어

2. **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - 10분 (선택)
   - XML 원본 데이터 구조
   - XLSX 정제 파이프라인
   - 데이터 변환 과정

3. **[WORKFLOW.md](WORKFLOW.md)** - 15분
   - PA/SA 파이프라인 상세 설명
   - 알고리즘 원리
   - 데이터 흐름

4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 필요시
   - 자주 발생하는 문제
   - 진단 방법
   - 해결책

### 👨‍💻 개발자/시스템 관리자

1. **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - 데이터 흐름 이해
   - XML to XLSX 변환 상세
   - 스크립트별 역할
   - 검증 체크리스트

2. **[WORKFLOW.md](WORKFLOW.md)** - 시스템 이해
   - 아키텍처
   - 모듈별 역할
   - 무결성 검증 시스템

3. **[PERFORMANCE.md](PERFORMANCE.md)** - 최적화
   - 시스템 요구사항
   - 성능 튜닝
   - 벤치마크

4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 디버깅
   - 로그 분석
   - 데이터 검증

### 📊 결과 분석가

1. **[README.md](README.md)** - 기본 이해
2. **[WORKFLOW.md](WORKFLOW.md)** - 평가 시스템 섹션
   - 정확도 지표
   - 결과 해석
3. **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - 데이터 출처 이해
4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 평가 문제

---

## 📄 문서 요약

### README.md

**용도**: 프로젝트 전체 개요

**주요 내용**:
- 프로젝트 목표 및 특징
- 디렉토리 구조
- 빠른 시작 (설치, 실행, 배치 처리)
- 주요 모듈 설명
- 트러블슈팅 링크

**분량**: ~520줄
**읽기 시간**: 5-10분

---

### DATA_PREPARATION.md

**용도**: XML to XLSX 데이터 정제 프로세스

**주요 내용**:
1. **개요** - 변환 목표 및 데이터 구조
2. **XML 원본 구조** - TEI XML 형식 상세
3. **XLSX 변환 결과** - 구병렬, 문장병렬, 문단병렬 구조
4. **정제 파이프라인** (3단계)
   - Phase 1: XML 파싱 및 초기 변환
   - Phase 2: 데이터 정규화 및 검증
   - Phase 3: 상위 레벨 병렬 데이터 생성
5. **완전한 실행 플로우** - 일반/특수 서종 구분
6. **데이터 흐름도** - 시각적 변환 과정
7. **출력 디렉토리 구조** - 최종 파일 위치
8. **정제 통계** - 데이터 규모 및 현황
9. **주의사항** - XML 다양성, NaN 처리, 특수 서종 등
10. **검증 체크리스트** - 정제 후 확인 사항

**대상 사용자**:
- 새로운 데이터 추가 시 필요
- 데이터 전처리 이해 필요한 경우
- 원본 XML 구조 파악 필요한 경우

**분량**: ~700줄
**읽기 시간**: 10-15분

---

### WORKFLOW.md

**용도**: 시스템 상세 이해

**주요 내용**:
1. **시스템 아키텍처** - 전체 플로우
2. **PA 파이프라인** - 4단계 알고리즘 상세 설명
   - Target Split (마지막 문장 추출)
   - DP Word-boundary (단어 경계 기준 배치)
   - Word Span Slicing (무결성 보존)
   - Integrity Verification (검증)
3. **SA 파이프라인** - 3단계 알고리즘
4. **평가 시스템** - 정확도 평가 방법론
5. **배치 처리** - 43권 자동 처리
6. **무결성 검증** - 질 보증 메커니즘

**분량**: ~400줄
**읽기 시간**: 15-20분

---

### TROUBLESHOOTING.md

**용도**: 문제 해결 참고서

**주요 내용**:
1. **PA 문제** (5가지)
   - 무결성 경고
   - 문자 손실
   - 평가 점수 낮음
   - 타임아웃
   - 문장 수 차이

2. **SA 문제** (2가지)
   - 입력 파일 없음
   - NaN 값

3. **배치 처리 문제** (3가지)
   - 중단
   - PA 실패
   - SA 실패

4. **데이터 문제** (2가지)
   - NaN 값 많음
   - 특수 문자 문제

5. **평가 문제** (2가지)
   - 경로 오류
   - 점수 불일치

6. **성능 문제** (2가지)
   - 처리 속도 느림
   - 메모리 부족

**분량**: ~500줄
**읽기 시간**: 필요시만 (검색 활용)

---

### PERFORMANCE.md

**용도**: 성능 최적화 가이드

**주요 내용**:
1. **시스템 요구사항**
   - 최소/권장 사양
   - 빠른 성능 체크

2. **PA 최적화** (4가지)
   - 배치 크기 조절
   - 워커 수 조절
   - 모델 선택
   - GPU 메모리 최대화

3. **SA 최적화** (2가지)
   - 임베딩 캐시
   - 매칭 알고리즘

4. **배치 처리 최적화** (3가지)
   - 병렬 처리
   - 선택적 처리
   - 평가 건너뛰기

5. **디스크 최적화**
   - 임시 파일 정리
   - 결과 압축

6. **Docker 최적화**
7. **모니터링**
8. **벤치마크**

**분량**: ~400줄
**읽기 시간**: 10-15분

---

### ROADMAP_TO_F1_0.9.md

**용도**: PA F1 0.90 달성을 위한 로드맵

**주요 내용**:
- 현황 분석 및 개선 전략
- Grid Search 실행 계획
- Supar Bonus, Ensemble Voting 등 기법
- 체크리스트 및 다음 단계

**분량**: ~326줄
**읽기 시간**: 15분

---



### OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md

**용도**: AI 에이전트와의 효율적 협업을 위한 설계 매뉴얼

**주요 내용**:
1. **3대 원칙**: 관측성 우선, 증거 기반 디버깅, 컴포넌트별 기여도 분리
2. **프롬프트 템플릿**: 초기 요구사항 작성법
3. **단계별 체크리스트**: 설계 → 인프라 → 구현 → 실험
4. **코드 스니펫**: TraceWriter, Stage Decorator, Ablation Runner
5. **실전 예시**: 이 프로젝트에서의 적용 사례

**대상**: AI 에이전트와 복잡한 시스템을 개발하는 개발자
**분량**: ~440줄
**읽기 시간**: 15분

---

### MULTIVECTOR_VS_DENSE.md

**용도**: BGE-M3 임베딩 방식 비교 설명

**주요 내용**:
1. **Dense Vector**: 의미적 유사도 (1024차원)
2. **Sparse Vector**: 키워드 기반 매칭 (100차원)
3. **ColBERT Vector**: 토큰별 상세 표현 (512차원)
4. **작업별 선택 가이드**: SA는 Dense, PA는 Multi-Vector
5. **성능 비교**: 속도, 메모리, 정확도

**대상**: 임베딩 선택에 혼란을 겪는 개발자
**분량**: ~430줄
**읽기 시간**: 10분

---

## 🗂️ 파일 구조

```
documents/
├── INDEX.md                          ← 📍 이 파일 (문서 가이드)
├── DATA_PREPARATION.md               ← 데이터 처리 과정
├── WORKFLOW.md                       ← PA/SA 워크플로우
├── ROADMAP_TO_F1_0.9.md              ← PA F1 0.90 로드맵
├── TROUBLESHOOTING.md                ← 문제 해결
├── PERFORMANCE.md                    ← 성능 최적화
├── OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md ← 관측성 우선 설계 매뉴얼
└── MULTIVECTOR_VS_DENSE.md           ← Multi-Vector vs Dense 임베딩 비교
```

---

## 🔍 빠른 검색

### 특정 주제별 찾기

#### PA 알고리즘
- 원리 이해: [WORKFLOW.md#pa-파이프라인](WORKFLOW.md#pa-문단-정렬-파이프라인)
- 문제 해결: [TROUBLESHOOTING.md#pa-관련-문제](TROUBLESHOOTING.md#pa-관련-문제)
- 최적화: [PERFORMANCE.md#pa-성능-최적화](PERFORMANCE.md#pa-성능-최적화)

#### SA 알고리즘
- 원리 이해: [WORKFLOW.md#sa-파이프라인](WORKFLOW.md#sa-문장-정렬-파이프라인)
- 문제 해결: [TROUBLESHOOTING.md#sa-관련-문제](TROUBLESHOOTING.md#sa-관련-문제)
- 최적화: [PERFORMANCE.md#sa-성능-최적화](PERFORMANCE.md#sa-성능-최적화)

#### 평가 시스템
- 원리: [WORKFLOW.md#평가-시스템](WORKFLOW.md#평가-시스템)
- 결과 해석: [WORKFLOW.md#평가-결과-해석](WORKFLOW.md#평가-결과-해석)
- 문제 해결: [TROUBLESHOOTING.md#평가-문제](TROUBLESHOOTING.md#평가-문제)

#### 배치 처리
- 실행: [README.md#배치-처리](README.md#배치-처리)
- 흐름: [WORKFLOW.md#배치-처리-흐름](WORKFLOW.md#배치-처리-흐름)
- 문제: [TROUBLESHOOTING.md#배치-처리-문제](TROUBLESHOOTING.md#배치-처리-문제)
- 최적화: [PERFORMANCE.md#배치-처리-최적화](PERFORMANCE.md#배치-처리-최적화)

#### 성능
- 요구사항: [PERFORMANCE.md#시스템-요구사항](PERFORMANCE.md#시스템-요구사항)
- 튜닝: [PERFORMANCE.md#pa-성능-최적화](PERFORMANCE.md#pa-성능-최적화)
- 모니터링: [PERFORMANCE.md#모니터링](PERFORMANCE.md#모니터링)

---

## 📝 문서 업데이트 이력

### 2025년 12월 19일
- **XLSX 기반 완전 재정리**
- 구 XML 파이프라인 문서 삭제
- 새로운 4개 문서 작성:
  - README.md (프로젝트 개요)
  - WORKFLOW.md (시스템 상세)
  - TROUBLESHOOTING.md (문제 해결)
  - PERFORMANCE.md (성능 최적화)
  - INDEX.md (이 파일)

---

## 💡 문서 사용 팁

### 1. 효율적인 읽기

```
첫 방문 → README.md (5분)
           ↓
문제 발생 → TROUBLESHOOTING.md (검색)
           ↓
깊이 있는 이해 → WORKFLOW.md (15분)
           ↓
성능 개선 → PERFORMANCE.md (10분)
```

### 2. 검색 방법

```bash
# 1. 로컬 검색 (현재 폴더)
grep -r "keywords" documents/

# 2. GitHub 웹 검색
# https://github.com/hw725/CSP/search?q=keywords

# 3. 파일 내 검색 (Editor)
Ctrl+F (문서 내 검색)
Ctrl+Shift+F (폴더 전체 검색)
```

### 3. 오프라인 액세스

```bash
# 모든 문서 다운로드
git clone https://github.com/hw725/CSP.git

# 또는 ZIP 다운로드
# https://github.com/hw725/CSP/archive/main.zip
```

---

## 🆘 지원

### 문서 관련 문제

**오타, 오류, 개선사항**:
- GitHub Issues: [new issue](https://github.com/hw725/CSP/issues)
- 제목: `[docs] 문제 설명`

**새로운 문서 요청**:
- 제목: `[docs] 추가 필요 항목`

### 기술 지원

**코드 문제**:
- GitHub Issues: [new issue](https://github.com/hw725/CSP/issues)

**성능 문제**:
- [PERFORMANCE.md](PERFORMANCE.md) 참고
- 자세한 로그와 함께 Issue 제출

---

## 📊 문서 통계

| 문서 | 라인 | 섹션 | 읽기시간 |
|------|------|------|---------|
| README.md | ~300 | 10 | 5-10분 |
| WORKFLOW.md | ~400 | 15 | 15-20분 |
| TROUBLESHOOTING.md | ~500 | 20 | 필요시 |
| PERFORMANCE.md | ~400 | 15 | 10-15분 |
| **합계** | **~1600** | **60** | **40-60분** |

---

## 🎯 다음 단계

1. **[README.md](README.md) 읽기** (5분)
   → 프로젝트 전체 이해

2. **단일 책 처리 실행** (5분)
   ```bash
   docker-compose run csp python p2s/main.py input.xlsx output.xlsx
   ```

3. **결과 확인** (5분)
   - 무결성 검증 시트 확인
   - F1 점수 확인

4. **배치 처리 실행** (필요시)
   ```bash
   docker-compose run csp python batch_43books.py
   ```

5. **결과 분석** (필요시)
   - [WORKFLOW.md#평가-결과-해석](WORKFLOW.md#평가-결과-해석) 참고

---

**마지막 업데이트**: 2026년 1월 13일
**문서 버전**: 2.1 (PA F1 0.87 달성 반영, 문서 정리)
