# 웹 재현 앱

`sandbox/web/app.py` 는 재현 실행을 브라우저에서 시작하기 위한 FastAPI 앱이다.
실제 판정은 저장소 루트의 `sandbox/docker-compose.yml` 을 통해 실행된다.

## 실행

```bash
pip install -r sandbox/web/requirements.txt
docker build -f sandbox/Dockerfile -t dansa-sandbox .
uvicorn --app-dir sandbox/web app:app --host 127.0.0.1 --port 8000
```

`REVIEW_ACCESS_TOKEN` 이 비어 있으면 누구나 실행할 수 있는 OPEN 모드,
설정하면 토큰 검증을 거치는 GATED 모드로 동작한다. 공개 테스트용
가짜 실행은 `WEB_MOCK=1` 로 켤 수 있다.

## 동작 규칙 (app.py 기준)

- OPEN 모드에서는 Gemini 입력이 비활성화되어 OpenAI·Anthropic 2모델만 실행된다.
  3모델 전체 재현은 GATED 모드에서 제공한다.
- 재현 실행은 동시 1건만 허용한다. 진행 중이면 다음 요청은 거부된다.
- 실행 타임아웃은 `RUN_TIMEOUT_SEC`(기본 2400초)로 조정하며, 초과하거나
  연결이 끊기면 컨테이너를 종료한다.
- 웹 폼의 문장 표시 상한 기본값은 `DEFAULT_VIEW_CAP`(기본 10)으로 지정한다.

## 운영 메모

- 공개 배포 시 HTTPS 앞단을 둔다.
- 원문·번역문 포함 입력 파일은 저장소에 올리지 않는다.
- 배포 주소, 접속 방식, 임시 키 같은 운영 세부사항은 `local_private/` 에 둔다.
