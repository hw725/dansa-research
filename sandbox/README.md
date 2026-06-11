# 재현 샌드박스

이 디렉터리는 LLM 판정 절차를 다시 실행하고 동결 기준 결과와 비교하기 위한
도구를 둔다. 공개 저장소에는 원문·번역문 포함 입력 파일이 포함되지 않으며,
공개 가능한 익명 데이터와 통계 산출물만 추적한다.

## 공개 기준

- 인용 기준 결과: `results/final_stats_v3.1_cleaned_balanced.json`
- 입력 무결성 기준: `RUN_MANIFEST.json`
- 공개 통계 재현:

```bash
python scripts/compute_final_stats.py --check --source anon
```

## 구성

| 파일 | 역할 |
|---|---|
| `entrypoint.py` | 입력 확인, 모델 호출, 비교 리포트 생성을 순서대로 실행 |
| `verify_corpus.py` | 로컬 입력 파일의 SHA-256 값을 `RUN_MANIFEST.json` 과 대조 |
| `vendor_clients.py` | 벤더별 모델 호출 래퍼 |
| `run_a_run.py` | 동결 기준과 같은 표본·프롬프트·파서로 판정을 재실행 |
| `compare_run_drift.py` | 동결 기준 대비 비율·효과방향·문장 단위 일치를 비교 |
| `Dockerfile` / `.dockerignore` | 공개 가능한 파일만 이미지 빌드 컨텍스트에 포함 |
| `docker-compose.yml` | 재현 실행 컨테이너 구성 |
| `squid.conf` / `egress_allowlist.md` | 벤더 API 연결에 필요한 네트워크 설정 |

## 실행 요약

로컬에 비공개 입력 파일과 LLM manifest가 있는 환경에서 실행한다.

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
  docker compose -f sandbox/docker-compose.yml run --rm sandbox
```

키를 제공한 모델만 실행된다. 실행 결과는 동결 기준 결과와 비교되어 섹션별
효과방향과 문장 단위 일치율로 요약된다.

## 기준 결과와 재실행 결과

LLM 모델은 시점과 설정에 따라 응답이 달라질 수 있으므로 재실행 결과가 동결
기준 수치와 완전히 같아야 한다고 보지 않는다. 공개 논문·문서에서 인용하는
기준 수치는 `results/final_stats_v3.1_cleaned_balanced.json` 이며, 재실행은
방법과 효과 방향을 확인하기 위한 보조 검증 절차다.
