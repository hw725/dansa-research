# dansa-research 재현 샌드박스 이미지.
# 빌드 컨텍스트 = 저장소 루트 (Dockerfile 은 sandbox/Dockerfile).
#   docker build -f sandbox/Dockerfile -t dansa-sandbox .
#
# 이미지에 굽는 것: 코드(scripts, sandbox), 익명 판정·통계(results 의 *_anon.csv·*.json),
#   RUN_MANIFEST.json. (번역문 든 raw 판정·corpus 는 .dockerignore 로 제외)
# 런타임 마운트(read-only): corpus, llm_manifests. 키: 실행 환경변수.
FROM python:3.12-slim

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY sandbox/requirements.txt /app/sandbox/requirements.txt
RUN pip install -r /app/sandbox/requirements.txt

# .dockerignore 가 raw 판정·corpus·env·archive·analysis 를 컨텍스트에서 제외한다.
COPY scripts/ /app/scripts/
COPY sandbox/ /app/sandbox/
COPY results/ /app/results/
COPY RUN_MANIFEST.json /app/RUN_MANIFEST.json

# 비루트 사용자
RUN useradd -m -u 10001 sandbox && chown -R sandbox:sandbox /app
USER sandbox

# 재현 실행 진입점.
ENTRYPOINT ["python", "/app/sandbox/entrypoint.py"]
