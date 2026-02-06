# 연구 재현 가이드 (Reproduction Guide)

본 문서는 데이터셋 생성부터 분석 결과까지 재현하는 방법을 안내합니다.

---

## 1. 환경 설정

### 1.1 필수 요구사항

- **Python**: 3.9 이상
- **CUDA**: 11.8 이상 (GPU 사용 시)

### 1.2 패키지 설치

```bash
pip install pandas numpy scipy tqdm openai regex
```

### 1.3 OpenAI API 설정 (LLM 분석용)

```bash
# Windows
set OPENAI_API_KEY=sk-...

# Linux/Mac
export OPENAI_API_KEY=sk-...
```

---

## 2. 데이터셋

### 2.1 원본 데이터

본 연구에 사용된 원본 코퍼스(한문-한국어 병렬 번역문)는 현재 **한국고전번역원 동양고전연구소 DB구축 지원사업**으로 구축 중에 있으며, 추후 한국고전번역원을 통해 공개될 예정입니다.

- 총 364,007건의 병렬 코퍼스 (원문-번역문 쌍)
- 현토 마커 664종 추출

### 2.2 익명화 데이터셋

본 저장소에서는 재현 가능성을 위해 익명화된 데이터를 제공합니다:
- `results/dansa_level1_judgments_anon.csv`: Level 1-2 판정 결과
- `results/dansa_level2_judgments_anon.csv`: Level 2 판정 결과
- 번역문: SHA-256 해시 (16자 truncate)로 대체

---

## 3. 분석 파이프라인

### Step 1: 마커 정규화
```bash
# 조사 변이형(을/를, 은/는)과 어미 변이형 통일
python scripts/hyeonto_normalizer.py
```

### Step 2: 전근대 기준 분류 (Phase 4)
```bash
# 664종 마커를 38개 범주로 분류
python scripts/phase4_premodern_classify.py
```

### Step 3: 단사 전수조사 (Phase 5)
```bash
# Level 1, 2 통계 검증
python scripts/dansa_full_survey.py
```

### Step 4: 장르별 분석
```bash
# Level 3 기사지단 가설 검증
python scripts/analyze_hada_by_genre.py
```

---

## 4. 결과 파일

| 파일 | 설명 |
|------|------|
| `interim_reports/CLASSIFIED_MARKERS.md` | 마커 분류 상세 보고서 |
| `results/dansa_full_survey.json` | 통계 검정 결과 |
| `results/dansa_reproduced_stats.json` | 재현 통계 |

---

## 5. 통계 검증 재현

### 5.1 Level 1: 유사이단 (`로다`)

```python
from scipy.stats import chi2_contingency

# 관측값 (Target: 로다, Control: 무작위)
observed = [[344, 608], [140, 812]]
chi2, p, dof, expected = chi2_contingency(observed)
print(f"chi-sq = {chi2:.2f}, p = {p:.2e}")
# 예상 출력: chi-sq = 114.16, p = 1.20e-26
```

### 5.2 Level 2: 쾌절 vs 미절 (`니라` vs `라`)

```python
observed = [[2815, 1417], [2160, 2073]]
chi2, p, dof, expected = chi2_contingency(observed)
print(f"chi-sq = {chi2:.2f}, p = {p:.2e}")
# 예상 출력: chi-sq = 208.90, p = 2.38e-47
```

---

## 6. 트러블슈팅

### Q: OpenAI API 오류
A: API 키 설정과 잔액을 확인하세요.

### Q: 인코딩 오류 (mojibake)
A: 모든 파일이 UTF-8로 저장되었는지 확인하세요.

---

## 7. 인용

```
Hyeonto Analysis Project (2026).
전근대 현토 분류 및 단사(斷辭) 통계 검증.
https://github.com/hw725/CSP
```
