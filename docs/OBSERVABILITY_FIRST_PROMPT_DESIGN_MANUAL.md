# 관측성 우선 프롬프트 설계 매뉴얼
## AI 에이전트와 복잡한 시스템 개발 시 효율적 디버깅을 위한 실전 가이드

> **작성 배경**: 멀티벡터/경계 학습/정렬 시스템 개발 중 "점수 폭락 + 속도 증가" 현상을 겪으며, **"단계가 실제로 실행됐는지", "각 컴포넌트가 얼마나 기여했는지"를 증거로 입증하는 데 수일이 소요되었습니다**. 이를 처음부터 방지하기 위한 프롬프트 설계 원칙과 체크리스트를 제공합니다.

---

## 0. 핵심 원칙: 효율적 개발을 위한 3대 원칙

### ✅ 관측성 우선 (Observability First)
- **"작동 여부를 가정하지 말고, 작동했다는 증거를 기록해야 합니다"**
- 모든 주요 단계(모델 로딩, 임베딩, 후보 생성, 선택, 폴백)는 **구조화된 로그(JSONL/structured log)**로 기록합니다
- 단순 print문 대신 **stage metadata + 입출력 샘플 + 타이밍 정보**를 포함해야 합니다

### ✅ 증거 기반 디버깅 (Evidence-Based Debugging)
- **"느낌상 느려졌다/나빠졌다" 대신 "trace 집계 결과 X%가 폴백으로 전환되었다"로 표현합니다**
- 성능 변화 발생 시: (1) 로그 집계 스크립트로 통계 추출 → (2) 전/후 비교표 생성 → (3) 원인 가설을 수치로 검증
- **"의심"을 "확정"으로 전환하는 데 필요한 데이터를 미리 설계해야 합니다**

### ✅ 컴포넌트별 기여도 분리 (Component Attribution)
- **"여러 요소 중 어느 것이 실제로 효과가 있었는가?"에 답할 수 있어야 합니다**
- Ablation 자동화: 각 컴포넌트를 on/off하며 멀티 seed 실험 수행 → 통계적 유의성 확인
- 최소 단위: seed 10회 이상의 통계적 반복 + 평균/표준편차/신뢰구간 산출

---

## 1. 프로젝트 시작 시 필수 프롬프트 구조

### 1.1 초기 요구사항 프롬프트 템플릿

```markdown
## 프로젝트: [시스템 이름]
## 목표: [핵심 목표 1-2문장]

### 핵심 요구사항
1. [기능 요구사항 - 예: "문단 경계를 학습 모델로 예측하고, 다국어 정렬"]
2. [성능 요구사항 - 예: "F1 > 0.85, 처리속도 < 2sec/doc"]

### ⚠️ 관측성 필수사항 (이 프로젝트에서 반드시 구현)
1. **Structured Logging**
   - 모든 주요 단계(전처리/임베딩/후보생성/선택/폴백)를 JSONL로 기록
   - 각 record는 최소한 포함: `{stage, timestamp, input_id, output_summary, metadata}`
   - 폴백/스킵 발생 시 **반드시 사유(reason) 필드** 기록

2. **Component Tracing**
   - 다음 컴포넌트들이 "실제로 호출됐는지/몇 번/평균 소요시간"을 집계 가능하도록:
     - [컴포넌트 목록 명시 - 예: tokenizer, boundary_model, supar, embedder, dp_solver]
   - 각 컴포넌트는 호출 시 `{component, invoked_at, duration_ms, result_summary}` trace 남김

3. **Selection Transparency**
   - 후보가 여러 개 생성되는 경우, **모든 후보의 점수/메타를 기록**
   - 최종 선택 시: `{candidates_total, candidates_considered, candidates_skipped, skip_reasons[], best_candidate_detail}`
   - 폴백 전환 시: 폴백 사유를 명시적으로 남김 (`insufficient_candidates`, `all_below_threshold` 등)

4. **Ablation Runner**
   - 다음 실험을 자동화할 수 있는 러너 스크립트:
     - Seed 1~10 반복 (통계적 유의성 확보)
     - 각 컴포넌트 on/off 조합 (최소 baseline + 각 요소별 단독 + full)
   - 결과를 CSV/리포트로 자동 집계 (평균/표준편차/신뢰구간)

5. **Diff-Friendly Output**
   - 동일 입력에 대해 seed만 바뀌면 deterministic하게 재현 가능
   - 출력 파일명에 실험 설정 포함 (예: `pa_trace_seed3_boundary_on_whitespace_off.jsonl`)

### ⚠️ 금지사항 (피해야 할 안티패턴)
- ❌ `print(...)` 위주의 디버깅 (구조화되지 않아 집계 불가능)
- ❌ "모델 로딩 완료" 메시지만 남기고 **실제 추론 결과는 기록하지 않음**
- ❌ 후보 선택 시 "best만 기록, 나머지는 버림" → "왜 이것이 선택되었는가?" 추적 불가
- ❌ 폴백 발생 시 사유를 기록하지 않음 → "단계가 누락되었는가?" 논쟁 반복
- ❌ seed 1회만 실행 후 성능 판단 (통계적으로 무의미한 결론)

### 구현 순서 (AI 에이전트에게 명시)
1. **먼저**: trace/logging 인프라 구축 (JSONL writer, stage decorator, 집계 스크립트)
2. **그다음**: 기능 구현 (각 단계마다 trace 호출 추가)
3. **마지막**: ablation runner + 리포트 자동화
4. **검증**: seed 3개 정도로 trace가 제대로 쌓이는지 확인 후 본 실험
```

---

## 2. 단계별 체크리스트

### Phase 1: 설계 단계 (코드 작성 전)
- [ ] 시스템의 주요 단계(5~10개)를 나열했는가?
- [ ] 각 단계마다 "성공/실패/폴백" 조건을 명시했는가?
- [ ] 후보 생성 → 선택 구조가 있다면, **모든 후보의 메타를 기록**하는 방식을 정했는가?
- [ ] Ablation 대상 컴포넌트 목록을 확정했는가? (최소 3~5개)
- [ ] Seed 반복 횟수를 결정했는가? (권장: 10회 이상)

### Phase 2: 인프라 구축 (기능 구현 전)
- [ ] JSONL trace writer 함수가 준비됐는가? (`write_trace(stage, data)`)
- [ ] Stage decorator 또는 context manager로 자동 기록 가능한가?
  ```python
  @trace_stage("embedding")
  def embed_texts(texts):
      ...
  ```
- [ ] 집계 스크립트 초안이 있는가? (예: `scripts/summarize_trace.py --stage X`)
- [ ] 멀티 seed 러너가 있는가? (`scripts/run_multitest.py --seeds 1-10`)

### Phase 3: 기능 구현 (각 단계마다 확인)
- [ ] 새 함수/모듈 추가 시, trace 호출을 빼먹지 않았는가?
- [ ] 폴백/스킵 로직에 **반드시** `reason` 필드를 남겼는가?
- [ ] 후보 선택 시 `candidates_total`, `candidates_considered`, `candidates_skipped` + 각 후보별 메타 기록했는가?
- [ ] 모델 로딩 시 **로딩 성공 여부 + 추론 1회 샘플** trace를 남겼는가?

### Phase 4: 첫 실험 (seed 3개 정도로 빠른 검증)
- [ ] Trace JSONL이 생성되는가?
- [ ] 집계 스크립트로 주요 단계별 실행 횟수/평균 시간을 뽑을 수 있는가?
- [ ] 폴백이 발생했다면, 그 사유가 trace에 명시됐는가?
- [ ] 후보 선택 단계에서 "considered vs skipped" 분포를 확인할 수 있는가?

### Phase 5: 본 실험 (seed 10회 × ablation 조합)
- [ ] Baseline(모든 컴포넌트 off 또는 가장 단순한 설정) 실험 완료?
- [ ] 각 컴포넌트 단독 활성화 실험 완료? (A only, B only, C only ...)
- [ ] Full 설정 실험 완료?
- [ ] 통계 리포트(평균/표준편차/신뢰구간)가 자동 생성되는가?

### Phase 6: 성능 변화 발생 시 (디버깅)
- [ ] **먼저** trace 집계로 "어느 단계에서 변화가 생겼는지" 확인 (폴백 증가? 스킵 증가?)
- [ ] **그다음** 해당 단계의 상세 trace 샘플 10~20개 직접 열람
- [ ] 가설 수립: "X 조건에서 Y가 스킵되면서 Z만 평가됨" → 통계로 입증
- [ ] 수정 후 **동일 seed로 재실험** → 전/후 비교표 생성

---

## 3. 실전 예시: 이번 프로젝트에서의 적용

### 문제 상황
- "점수 폭락 + 속도 증가" 발생
- "단계 누락/폴백 의심" → 하지만 **증거 없음**
- 초기 로그: `print("boundary model loaded")` 수준 → 실제 추론 여부 확인 불가

### 비효율적 작업 과정 (소요 시간: 5일)
1. "모델 로딩 완료" 메시지만 보고 "정상 작동"으로 가정
2. 점수 하락 후 "실제로 사용되지 않았을 가능성" 의심 시작
3. Trace 기능 추가 → 재실험 → "후보가 insufficient로 스킵됨" 발견
4. 집계 스크립트 작성 → "considered==1이 51%!" 수치 확정
5. 원인 규명: `desired_tgt_len`이 큰 경우 boundary/supar가 자동 제외됨

### 매뉴얼 적용 시 예상 프로세스 (소요 시간: 1일)
1. **초기 프롬프트에 다음 요구사항 포함**:
   ```
   - 후보 생성 단계(boundary/supar/whitespace)마다 trace 기록
   - 후보 선택 시 모든 후보의 메타데이터 + 스킵 사유 기록
   - 집계 스크립트: candidates_considered 분포 자동 출력
   ```
2. **첫 실험(seed 3) 직후** 집계 스크립트 실행 → `considered==1` 과다 현상 즉시 발견
3. **원인 추적**: trace 샘플 10개 검토 → "insufficient_for_desired" 사유 확인
4. **수정 + 재실험 + 전후 비교** → 1일 내 완료

---

## 4. 프롬프트 작성 실전 팁

### Tip 1: "측정 불가능한 요구사항은 무의미"
❌ 나쁜 예: "경계 예측 모델을 써서 정확도를 높여줘"
✅ 좋은 예: 
```
경계 예측 모델(boundary model)을 추가하되:
1. 모델 로딩 시 trace에 {model_path, load_success, params} 기록
2. 추론 시마다 {input_len, predicted_boundaries_count, confidence_avg} 기록
3. 후보 생성 시 이 모델의 결과를 'boundary(th=0.7, count=N)' 형태로 후보 목록에 추가
4. 최종 선택 시 이 후보가 선택됐는지 여부를 best_tag에 기록
5. Ablation: boundary model on/off를 자동 비교할 수 있도록 --use-boundary-model 플래그 지원
```

### Tip 2: "폴백은 반드시 사유와 함께"
❌ 나쁜 예: `if not candidates: fallback()`
✅ 좋은 예:
```python
if not candidates:
    trace(stage="fallback_triggered", reason="no_sufficient_candidates", 
          attempted_methods=["boundary", "supar"], threshold_used=0.5)
    fallback()
```

### Tip 3: "집계 스크립트를 먼저 설계하세요"
- 코드 작성 전에 "원하는 분석 결과 형식"을 먼저 정의합니다:
  ```
  | stage              | invoked | avg_duration | fallback_rate |
  |--------------------|---------|--------------|---------------|
  | boundary_predict   | 200     | 45ms         | 0%            |
  | candidate_select   | 200     | 12ms         | 5.5%          |
  ```
- 그 다음 "이 표를 생성하려면 trace에 어떤 필드가 필요한가?"를 역산합니다

### Tip 4: "AI 에이전트에게 '예시 trace 구조'를 명시하세요"
```markdown
프롬프트에 포함할 내용:

예시 trace 레코드 (다음 구조를 준수):
{
  "stage": "src_matched_selected",
  "timestamp": "2026-01-06T10:23:45",
  "src_id": "doc123_para5",
  "candidates_total": 3,
  "candidates_considered": 1,
  "candidates_skipped": 2,
  "top_candidates": [
    {"family": "boundary", "tag": "boundary(th=0.7,2)", "score": 0.85, "prior_bonus": 0.15, "considered": false, "skip_reason": "insufficient_for_desired"},
    {"family": "whitespace_dp", "tag": "whitespace_dp(20)", "score": 0.72, "prior_bonus": 0.0, "considered": true}
  ],
  "best_tag": "whitespace_dp(20)",
  "best_score": 0.72
}
```

### Tip 5: "Seed 1회 실행은 실험이 아닙니다"
- 프롬프트에 명시: **"모든 성능 비교는 seed 10회 이상 반복 후 평균±표준편차로 보고해야 합니다"**
- 단일 seed 실행은 "빠른 smoke test"용도로만 사용합니다

---

## 5. 구현 패턴 (복사해서 쓸 수 있는 코드 스니펫)

### 5.1 JSONL Trace Writer

```python
import json
from pathlib import Path
from datetime import datetime

class TraceWriter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, 'w', encoding='utf-8')
    
    def write(self, stage: str, data: dict):
        record = {
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()
```

### 5.2 Stage Decorator (자동 trace)

```python
from functools import wraps
import time

def trace_stage(stage_name: str, tracer: TraceWriter):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                tracer.write(stage_name, {
                    "status": "success",
                    "duration_ms": duration_ms,
                    "result_summary": str(result)[:100]  # 샘플만
                })
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                tracer.write(stage_name, {
                    "status": "error",
                    "duration_ms": duration_ms,
                    "error": str(e)
                })
                raise
        return wrapper
    return decorator
```

### 5.3 후보 선택 Trace (투명성 확보)

```python
def select_best_candidate(candidates, desired_len, tracer):
    considered = []
    skipped = []
    
    for cand in candidates:
        if abs(len(cand.result) - desired_len) > desired_len * 0.3:
            skipped.append({
                "family": cand.family,
                "tag": cand.tag,
                "skip_reason": "insufficient_for_desired",
                "actual_len": len(cand.result),
                "desired_len": desired_len
            })
        else:
            considered.append(cand)
    
    best = max(considered, key=lambda c: c.score) if considered else None
    
    tracer.write("candidate_selection", {
        "candidates_total": len(candidates),
        "candidates_considered": len(considered),
        "candidates_skipped": len(skipped),
        "skip_details": skipped,
        "best_tag": best.tag if best else "none",
        "best_score": best.score if best else 0.0
    })
    
    return best
```

### 5.4 Ablation Runner 스크립트

```python
# scripts/run_ablation.py
import subprocess
import itertools

components = ["boundary_model", "supar", "whitespace_dp", "marker_bonus"]
seeds = range(1, 11)

# Baseline
for seed in seeds:
    subprocess.run([
        "python", "main.py",
        "--seed", str(seed),
        "--output-dir", f"results/baseline_seed{seed}",
        # 모든 컴포넌트 off
    ])

# 각 컴포넌트 단독
for comp in components:
    for seed in seeds:
        subprocess.run([
            "python", "main.py",
            "--seed", str(seed),
            f"--use-{comp}",
            "--output-dir", f"results/{comp}_only_seed{seed}"
        ])

# Full
for seed in seeds:
    subprocess.run([
        "python", "main.py",
        "--seed", str(seed),
        *[f"--use-{c}" for c in components],
        "--output-dir", f"results/full_seed{seed}"
    ])
```

### 5.5 통계 집계 스크립트

```python
# scripts/summarize_ablation.py
import json
import pandas as pd
from pathlib import Path

def aggregate_results(base_dir):
    results = []
    for exp_dir in Path(base_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        # exp_dir 이름에서 설정 추출 (예: baseline_seed1)
        config = exp_dir.name.split('_seed')[0]
        seed = int(exp_dir.name.split('_seed')[1])
        
        # 결과 CSV 읽기
        result_csv = exp_dir / "pa_results.csv"
        if result_csv.exists():
            df = pd.read_csv(result_csv)
            accuracy = df['accuracy'].mean()
            results.append({
                "config": config,
                "seed": seed,
                "accuracy": accuracy
            })
    
    df_results = pd.DataFrame(results)
    summary = df_results.groupby('config')['accuracy'].agg(['mean', 'std', 'count'])
    summary['ci95'] = 1.96 * summary['std'] / (summary['count'] ** 0.5)
    
    print(summary)
    summary.to_csv(Path(base_dir) / "ablation_summary.csv")

if __name__ == "__main__":
    aggregate_results("results/")
```

---

## 6. 체크리스트: "이 프롬프트, 제대로 짰나?"

프로젝트 시작 전에 이 질문들에 모두 "예"라고 답할 수 있어야 함:

- [ ] "이 시스템의 5가지 주요 단계가 실제로 실행됐는지"를 나중에 증거로 보여줄 수 있나?
- [ ] "A 컴포넌트를 껐을 때 vs 켰을 때" 성능 차이를 통계적으로 비교할 수 있나?
- [ ] 후보 선택/폴백 로직이 있다면, "왜 이 후보가 선택됐나?"를 trace만 보고 설명 가능한가?
- [ ] 성능이 갑자기 떨어졌을 때, "어느 단계에서 문제 생겼나?"를 1시간 내 특정 가능한가?
- [ ] Seed 10회 실험을 1 커맨드로 돌리고, 통계 리포트를 자동 생성할 수 있나?

---

## 7. 결론: 처음부터 "증거 우선" 문화 구축

### 이 매뉴얼의 핵심 메시지
1. **"작동한다고 가정"은 위험한 접근** → 항상 "작동했다는 증거"를 기록해야 합니다
2. **Trace는 디버깅 도구가 아니라 필수 인프라** → 기능 구현과 동시에 trace를 설계합니다
3. **"느낌"이 아니라 "수치"로 소통** → seed 1회는 의미 없으며, 최소 10회 이상 반복해야 합니다
4. **폴백/스킵은 반드시 사유와 함께 기록** → "insufficient_for_desired" 같은 명시적 reason 필드 필수
5. **집계 스크립트를 먼저 설계** → "원하는 분석 결과" → "필요한 trace 필드" 순서로 역산합니다

### AI 에이전트에게 전달할 핵심 프롬프트
```
"이 시스템은 모든 주요 단계를 JSONL trace로 기록하고, 
후보 선택/폴백 시 사유를 명시하며, 
seed 10회 이상의 ablation을 자동화하고, 
통계 리포트를 1 커맨드로 생성할 수 있어야 합니다. 
'작동할 것으로 예상됩니다'가 아니라 '이 trace가 증거입니다'로 답변해주세요."
```

---

## 부록: 이번 프로젝트 교훈 요약

| 문제 | 원인 | 매뉴얼 적용 시 방지 방법 |
|------|------|----------------------|
| "점수 폭락" 원인 불명 | 후보 선택 trace 미구현 | 초기 설계 단계에서 `candidates_considered` 집계 포함 |
| "단계 누락 의심" 논쟁 | 모델 로딩만 로깅, 추론 미기록 | 추론 시마다 `{invoked, duration, result_sample}` 기록 |
| "마커 보너스 0" 오해 | 집계 필드 불일치 | 집계 스크립트를 trace 설계와 함께 구축 |
| considered==1 과다 발견 (5일 소요) | trace 후속 추가 + 재실험 반복 | 첫 실험부터 집계 자동화 포함 |
| Ablation 수동 반복 작업 | Runner 스크립트 부재 | Phase 2에서 ablation runner 필수 구축 |

**결론**: 이 매뉴얼의 원칙을 적용했다면, **5일이 소요된 원인 규명 작업을 1일 내에 완료**할 수 있었을 것으로 추정됩니다.

---

**이 매뉴얼을 다음 프로젝트에 적용하는 방법**:
1. 프로젝트 킥오프 시 AI 에이전트에게 이 문서를 첨부합니다
2. "Section 1.1 템플릿을 참고하여 초기 요구사항을 작성해주세요" 지시
3. Phase 1~6 체크리스트를 각 단계마다 확인합니다
4. 첫 실험(seed 3회) 후 trace 품질을 반드시 검증합니다
5. 본 실험은 항상 seed 10회 이상 + 자동 통계 리포트 생성을 포함합니다

**"비효율적 작업은 선택이 아니라, 관측성 부족에서 비롯된 필연적 결과입니다."**
