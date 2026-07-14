# analysis: 任圭直 夬絶 가설 검증 파이프라인

## 가설
종결어미 ‘니라’가 ‘라’보다 행동·태도를 분명하게 결정하며 종결하는 형태를 더 자주 표지한다.

## 파이프라인

```
1. b_prompt 판정 (3-model consensus: gpt-5-mini, gemini, claude_sonnet)
2. 임베딩 (text-embedding-3-large, 3072d)
3. K-means 클러스터링 (silhouette score 기반 K 선택)
4. LightRAG per-category KG 구축 + 질의 (Q1~Q6)
5. 장르 통제 분석 (chi-square, odds ratio, sign test)
6. 보고서 생성
```

- unified KG 횡단 비교(CQ1~CQ7)는 폐기. LightRAG는 정량 횡단 비교에 부적합 (top_k 제한으로 범주 누락 반복).
- per-category Q7(가설 평가)도 제거. 단일 범주 KG로는 횡단 비교 불가.
- 정량 비교는 pandas/scipy로 직접 계산.

## 보고서

정본 보고서 (git 추적):
- `lightrag_out/REPORT_v4.md` — LightRAG per-category KG 분석 (범주별 × 6질의)
- `dci_out/REPORT.md`, `dci_out/cross/CQ1~6.md` — DCI 종합·횡단 비교
- `CROSS_CATEGORY_REPORT.md` — 니라(I+II) 대 라(III+IV) 종합
- `COMPARISON_REPORT.md` — DCI 대 LightRAG 방법 비교

`CROSS_CATEGORY_REPORT.md`와 `COMPARISON_REPORT.md`는 `analysis/build_comparison_report.py`로 재생성한다.

`REPORT_v5.md`(unified KG, 위에서 폐기한 방법)와 각 `results/`·`results_unified/`의 범주별·통합 질의 산출은 로컬 전용이다. `results_unified/`가 없는 클론에서는 `build_comparison_report.py`의 Part B(unified 대조)가 결과 없음으로 비게 된다.

A단계 정량 결과의 강건성 — 모델 간 일치도, 효과크기 CI, 합의 정의 민감도, 섹션 1·2·3 전체의 서종 층화 MH OR — 은 `../results/ROBUSTNESS_REPORT.md`가 다룬다. 여기 장르 통제 분석(섹션 2 만장일치분 한정)과 상보적이다.

## 모델 설정

| 용도 | 모델 |
|---|---|
| O/X 판정 | gpt-5-mini + gemini-2.5-flash + claude-sonnet (3모델 unanimous) |
| 임베딩 (클러스터링 + LightRAG) | text-embedding-3-large (3072d) — 반드시 통일 |
| KG 추출 | gpt-5-mini (reasoning_effort=minimal) |
| 질의 응답 | gpt-5-mini (reasoning_effort=medium) |
| 클러스터 요약 | gpt-5-mini |

## 실행 제약

- LightRAG JsonKVStorage 싱글턴 → 카테고리별 subprocess 격리 필수
- OpenAI 동시 요청: llm_model_max_async=4 권장
- `parallel_data_v2_cleaned.tsv`의 `cell` 컬럼이 4범주 키 (I_니라_O / II_니라_X / III_라_O / IV_라_X)

## 연구 결정: 단일 임베딩·LLM 유지 근거

핵심 정량 지표(OR, chi-square, sign test)는 종결어미 × O/X 교차표에서 산출되며 임베딩·KG를 경유하지 않는다. 임베딩은 질적 분석(클러스터링·LightRAG 검색)에만 사용되므로, 모델 교체는 서술 차이만 유발하고 결론에 영향 없음.

> 논문 기술: “본 연구의 가설 검증은 3개 LLM 합의 판정과 정량 통계에 기초하며, 임베딩 모델은 관여하지 않는다. 임베딩은 질적 분석에만 사용되었으며, 모델 교체는 질적 서술의 세부 표현에 영향을 줄 수 있으나 핵심 결론에는 영향을 미치지 않는다.”

## 향후 과제

- [ ] ~200종 전수 확보 후 임베딩 단계부터 재실행
- [ ] 사부분류별 통제 분석 재수행 (표본 증가로 집부 유의성 변화 확인)
- [ ] 역자 효과 분리 분석 (정태현 역전이 번역 관습인지 텍스트 특성인지)
- [ ] 논문 초고 작성
- [ ] 연결어미 기능 분화 분석 (하고/하며/(이)고/(이)며) — 동일 파이프라인 재활용, 판정 프롬프트 재설계 필요
