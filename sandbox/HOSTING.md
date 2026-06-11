# 재현 환경 운영 요약

이 문서는 공개 저장소용 운영 개요다. 원문·번역문 포함 입력 파일, 임시 키,
개별 접속 방식, 상세 운영 메모는 로컬 전용으로 관리한다.

## 준비

```bash
docker build -f sandbox/Dockerfile -t dansa-sandbox .
python sandbox/verify_corpus.py
```

`verify_corpus.py` 는 로컬 입력 파일의 SHA-256 값을 `RUN_MANIFEST.json` 과
대조한다. 공개 저장소에는 해당 입력 파일이 포함되지 않는다.

## 실행

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
  docker compose -f sandbox/docker-compose.yml run --rm sandbox
```

실행은 입력 확인, 모델 호출, 동결 기준 대비 비교 리포트 생성 순서로 진행된다.
키를 제공한 모델만 호출된다.

## 웹 앱

`sandbox/web/` 은 같은 재현 실행을 브라우저에서 시작하기 위한 FastAPI 래퍼다.
공개 배포 시에는 HTTPS 앞단을 두고, 운영 환경의 비공개 입력 파일과 키 관리는
저장소 밖에서 처리한다.

## 공개 저장소 원칙

- 원문·번역문 포함 입력 파일은 git 추적 대상이 아니다.
- 공개 가능한 익명 데이터와 통계 산출물만 원격 저장소에 올린다.
- 로컬 운영 메모는 `local_private/` 아래에 두며 `.gitignore` 로 차단한다.
