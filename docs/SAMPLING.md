# 표본 추출 기준

본 연구의 표본 추출은 exact 재현과 robustness 검증을 분리한다.

## 1. Exact 재현

보고된 통계 수치를 다시 만들 때는 `data/llm_manifests/`의 manifest를 사용한다.

manifest는 이미 LLM 판정에 사용한 행 목록이다. 따라서 재현 실행에서는 새 표본을 뽑지 않고, 같은 문장들을 그대로 다시 사용한다.

## 2. Manifest가 없을 때의 fallback

manifest가 없으면 `scripts/run_multimodel_judgments.py`가 고정 seed로 대조군을 추출한다. 이 방식은 실행할 때마다 달라지는 랜덤 추출이 아니라, 같은 입력 CSV와 같은 코드에서는 같은 표본이 나오도록 만든 재현 가능한 fallback이다.

현재 fallback 규칙은 다음과 같다.

| 섹션 | Target | Control |
|---|---|---|
| 섹션 1 | `游辭以斷/로다` 전체 | `微絶之斷/라`에서 로다 개수만큼 추출, seed 42 |
| 섹션 2 | `夬絶之斷/니라` 전체 | `微絶之斷/라`에서 니라 개수만큼 추출, seed 42 |
| 섹션 3 | `汎論以斷/하나니라` 전체 | `微絶之斷/라`에서 하나니라 개수만큼 추출, seed 99 |

이 fallback은 서종별 층화추출이 아니다. 서종별 같은 개수나 같은 비율을 맞추는 검증은 별도 robustness 설계가 필요하다.

## 3. Robustness 검증

통계적 안정성 확인은 exact 재현과 다른 작업이다. 여러 seed로 대조군을 다시 뽑아도 효과 방향과 효과 크기가 안정적인지 확인한다.

예시는 다음과 같다.

- seed 1, 2, 3, ..., 100으로 대조군을 반복 추출
- 각 seed별로 LLM 판정을 새로 수행하거나, 전체 후보 풀에 대한 판정값이 있을 때는 그 판정값을 재사용
- seed별 `Consensus O`, χ², Cramer’s V, target/control 비율 분포를 비교

주의할 점은, 현재 결과 CSV는 모든 가능한 `라` 후보의 LLM 판정을 포함하지 않는다는 것이다. 따라서 기존 결과만으로 임의 seed의 효과 크기를 바로 계산할 수 없다. 새 seed robustness를 하려면 새 표본에 대해 LLM 판정을 돌리거나, 먼저 더 넓은 control 후보 풀의 판정값을 만들어야 한다.
