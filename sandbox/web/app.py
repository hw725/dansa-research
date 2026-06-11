"""sandbox/web/app.py — 재현 샌드박스 웹앱.

브라우저에서 재현 실행을 시작하고 stdout 를 실시간 스트리밍한다.

개방/게이트 전환:
- 환경변수 REVIEW_ACCESS_TOKEN 이 비어 있으면 OPEN 모드.
- 토큰이 설정돼 있으면 GATED 모드.

운영:
- docker compose 를 고정 인자로 호출한다.
- 동시 1건 + 타임아웃 + 끊김 시 컨테이너 종료.
- 공개 배포 시 HTTPS 뒤에서 서빙한다.

실행: REVIEW_ACCESS_TOKEN= uvicorn --app-dir sandbox/web app:app --port 8000
      (토큰 비우면 개방). WEB_MOCK=1 추가 시 키 없이 가짜 판정.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "sandbox" / "docker-compose.yml"

ACCESS_TOKEN = os.environ.get("REVIEW_ACCESS_TOKEN", "").strip()
OPEN = ACCESS_TOKEN == ""              # 토큰 없으면 개방 모드
MOCK = os.environ.get("WEB_MOCK") == "1"
RUN_TIMEOUT = int(os.environ.get("RUN_TIMEOUT_SEC", "2400"))
DEFAULT_VIEW_CAP = os.environ.get("DEFAULT_VIEW_CAP", "10")

_slot = threading.BoundedSemaphore(1)  # 동시 1건

app = FastAPI(title="dansa-research 재현 샌드박스")


def page(open_mode: bool) -> str:
    token_field = "" if open_mode else (
        '<label>접근 토큰</label>'
        '<input name="token" type="password" required autocomplete="off">')
    if open_mode:
        gemini_field = (
            '<label>Google Gemini API Key</label>'
            '<input name="gemini_key" type="password" disabled '
            'placeholder="현재 비활성화">'
            '<small>현재 공개 데모에서는 Gemini가 비활성화되어 있습니다. '
            '3모델 전체 재현은 요청 시 제공합니다.</small>')
        intro = ("LLM 판정 절차를 다시 실행하고 동결 기준 결과와 비교합니다"
                 "(공개 데모: OpenAI·Anthropic 2모델).")
    else:
        gemini_field = ('<label>Google Gemini API Key</label>'
                        '<input name="gemini_key" type="password" autocomplete="off">')
        intro = "LLM 판정 절차를 다시 실행하고 동결 기준 결과와 비교합니다(3모델)."
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dansa-research 재현 샌드박스</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem}}
 label{{font-weight:600;font-size:.9rem;display:block;margin-top:.5rem}}
 input{{width:100%;padding:.5rem;margin:.25rem 0;box-sizing:border-box}}
 input:disabled{{background:#eee;color:#999}}
 button{{padding:.6rem 1.2rem;font-size:1rem;cursor:pointer;margin-top:.5rem}}
 pre{{background:#111;color:#eee;padding:1rem;white-space:pre-wrap;word-break:break-all;
     max-height:60vh;overflow:auto;border-radius:6px}}
 small{{color:#666;font-weight:400;display:block;margin-bottom:.5rem}}
</style></head><body>
<h1>재현 샌드박스</h1>
<p>{intro}</p>
<form id="f">
 {token_field}
 <label>OpenAI API Key <small style="display:inline">(미입력 시 해당 모델 제외)</small></label>
 <input name="openai_key" type="password" autocomplete="off">
 <label>Anthropic API Key</label><input name="anthropic_key" type="password" autocomplete="off">
 {gemini_field}
 <label>문장 표시 상한 VIEW_CAP <small style="display:inline">(0=번역문 숨김)</small></label>
 <input name="view_cap" value="{DEFAULT_VIEW_CAP}">
 <button type="submit">재현 실행</button>
</form>
<p><small>입력한 키는 현재 실행 요청에만 전달됩니다.</small></p>
<pre id="out">대기 중…</pre>
<script>
const f=document.getElementById('f'), out=document.getElementById('out');
f.addEventListener('submit', async e=>{{
  e.preventDefault(); out.textContent='실행 시작…\\n';
  let r;
  try{{ r=await fetch('/run',{{method:'POST',body:new FormData(f)}}); }}
  catch(err){{ out.textContent='연결 오류: '+err; return; }}
  if(!r.ok){{ out.textContent='오류 '+r.status+': '+await r.text(); return; }}
  const reader=r.body.getReader(), dec=new TextDecoder();
  for(;;){{ const {{value,done}}=await reader.read(); if(done) break;
    out.textContent+=dec.decode(value,{{stream:true}}); out.scrollTop=out.scrollHeight; }}
}});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return page(OPEN)


@app.post("/run")
def run(token: str = Form(""), openai_key: str = Form(""),
        anthropic_key: str = Form(""), gemini_key: str = Form(""),
        view_cap: str = Form("")) -> StreamingResponse:
    if not OPEN and not secrets.compare_digest(token, ACCESS_TOKEN):
        raise HTTPException(status_code=403, detail="접근 토큰 불일치")
    if OPEN:
        gemini_key = ""  # 개방 데모는 Gemini 비활성
    if not any(k.strip() for k in (openai_key, anthropic_key, gemini_key)):
        raise HTTPException(status_code=400, detail="API 키를 최소 1개 입력하세요")

    # 키는 서브프로세스 env 로 전달한다.
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = openai_key.strip()
    env["ANTHROPIC_API_KEY"] = anthropic_key.strip()
    env["GEMINI_API_KEY"] = gemini_key.strip()
    env["VIEW_CAP"] = str(int(view_cap)) if view_cap.strip().isdigit() else DEFAULT_VIEW_CAP
    env["SANDBOX_MOCK"] = "1" if MOCK else ""

    cmd = ["docker", "compose", "-f", str(COMPOSE), "run", "--rm", "-T", "sandbox"]

    def gen():
        if not _slot.acquire(blocking=False):
            yield "다른 재현이 진행 중입니다. 잠시 후 다시 시도하세요.\n"
            return
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
            )
            start = time.monotonic()
            for line in proc.stdout:
                yield line
                if time.monotonic() - start > RUN_TIMEOUT:
                    proc.kill()
                    yield "\n[시간 초과 — 중단]\n"
                    break
        except Exception as exc:  # noqa: BLE001
            yield f"\n[서버 오류: {type(exc).__name__}]\n"
        finally:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                if proc and proc.stdout:
                    proc.stdout.close()
            finally:
                _slot.release()

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
