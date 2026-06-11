# 네트워크 설정 개요

`docker-compose.yml` 은 재현 실행 컨테이너와 프록시 컨테이너를 분리한다.
프록시는 벤더 API 호출에 필요한 도메인 연결만 담당한다.

## 현재 벤더 엔드포인트

```text
api.openai.com:443
api.anthropic.com:443
generativelanguage.googleapis.com:443
```

모델 제공자의 API 경로가 바뀌면 `squid.conf` 의 도메인 목록과 실행 전
preflight 결과를 함께 확인한다.

## 점검

```bash
docker compose -f sandbox/docker-compose.yml config
```

구성 변경 후에는 재현 실행 전에 벤더별 preflight가 통과하는지 확인한다.
