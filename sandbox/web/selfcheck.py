"""selfcheck.py — 웹앱 로직 무오류 검증(docker 없이, 즉시). 개방·게이트 양쪽.

docker compose 호출은 가짜 Popen 으로 대체해 토큰·키·스트리밍 플러밍만 빠르게
검증한다. 실제 컨테이너 경로는 mock e2e 로 이미 검증됨.

실행:  python sandbox/web/selfcheck.py   (fastapi 설치 필요)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["REVIEW_ACCESS_TOKEN"] = "testtok"  # 우선 게이트 모드로 로드

import app as webapp  # noqa: E402


class _FakePopen:
    def __init__(self, *a, **k):
        self.args = a
        self.env = k.get("env", {})
        self.returncode = 0
        self._lines = iter(["[1/4] 검증\n", "[verify] PASS\n", "완료.\n"])
        self.stdout = self

    def __iter__(self):
        return self._lines

    def close(self):
        pass

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


captured: dict = {}


def _cap(*a, **k):
    captured["env"] = k.get("env")
    return _FakePopen(*a, **k)


webapp.subprocess.Popen = _cap  # type: ignore[assignment]

from fastapi.testclient import TestClient  # noqa: E402

c = TestClient(webapp.app)
ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


def stream_body(**data):
    with c.stream("POST", "/run", data=data) as r:
        body = "".join(line + "\n" for line in r.iter_lines())
    return r.status_code, body


print("[GATED 모드: 토큰 필요, Gemini 활성]")
webapp.ACCESS_TOKEN = "testtok"
webapp.OPEN = False
r = c.get("/")
check("GET / 폼 + 토큰칸", r.status_code == 200 and 'name="token"' in r.text)
check("Gemini 활성(비활성 안내 없음)", "현재 비활성화" not in r.text)
check("나쁜 토큰 → 403", c.post("/run", data={"token": "x", "openai_key": "k"}).status_code == 403)
check("키 없음 → 400", c.post("/run", data={"token": "testtok"}).status_code == 400)
captured.clear()
sc, body = stream_body(token="testtok", openai_key="ok", gemini_key="gk", view_cap="7")
check("정상 → 200 스트리밍", sc == 200 and "완료." in body)
env = captured.get("env", {})
check("OpenAI 키 전달", env.get("OPENAI_API_KEY") == "ok")
check("게이트: Gemini 키 전달", env.get("GEMINI_API_KEY") == "gk")
check("VIEW_CAP 전달", env.get("VIEW_CAP") == "7")

print("[OPEN 모드: 토큰 없음, Gemini 비활성]")
webapp.ACCESS_TOKEN = ""
webapp.OPEN = True
r = c.get("/")
check("GET / 토큰칸 없음", r.status_code == 200 and 'name="token"' not in r.text)
check("Gemini 비활성 안내 표시", "disabled" in r.text and "현재 비활성화" in r.text)
captured.clear()
sc, body = stream_body(openai_key="ok2", gemini_key="should_ignore")
check("개방: 토큰 없이 → 200", sc == 200 and "완료." in body)
env = captured.get("env", {})
check("개방: Gemini 강제 빈값(무시)", env.get("GEMINI_API_KEY") == "")
check("개방: VIEW_CAP 기본 10", env.get("VIEW_CAP") == "10")
check("개방: 키 전무 → 400", c.post("/run", data={}).status_code == 400)

print("SELFCHECK", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
