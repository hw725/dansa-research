# 웹 재현 앱

`sandbox/web/app.py` 는 재현 실행을 브라우저에서 시작하기 위한 FastAPI 앱이다.
실제 판정은 저장소 루트의 `sandbox/docker-compose.yml` 을 통해 실행된다.

## 실행

```bash
pip install -r sandbox/web/requirements.txt
docker build -f sandbox/Dockerfile -t dansa-sandbox .
uvicorn --app-dir sandbox/web app:app --host 127.0.0.1 --port 8000
```

접근 토큰을 쓰려면 실행 환경에 `REVIEW_ACCESS_TOKEN` 을 설정한다. 공개 테스트용
가짜 실행은 `WEB_MOCK=1` 로 켤 수 있다.

## 운영 메모

- 공개 배포 시 HTTPS 앞단을 둔다.
- 원문·번역문 포함 입력 파일은 저장소에 올리지 않는다.
- 배포 주소, 접속 방식, 임시 키 같은 운영 세부사항은 `local_private/` 에 둔다.
