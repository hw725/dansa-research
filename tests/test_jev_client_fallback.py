"""직결 Jev가 HTTP로 거절되면 같은 모델을 OpenRouter 경유로 다시 부른다(2026-10-06 402 크레딧 소진).

네트워크 없이 가짜 opener로 확인한다. 실행기가 결과 파일 이름을 client.model에서 고르므로
넘어가도 model은 그대로여야 하고, 어느 길이었는지는 usage()의 via_openrouter로 남는다.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import jev_client  # noqa: E402


def _opener(statuses):
    seen: list[dict] = []
    queue = list(statuses)

    @contextmanager
    def opener(req, timeout=30):
        seen.append({"url": req.full_url, "body": json.loads(req.data.decode("utf-8")),
                     "auth": dict(req.header_items()).get("Authorization")})
        status = queue.pop(0)
        if status is not None:
            raise urllib.error.HTTPError(req.full_url, status, "x", {}, io.BytesIO(b"{}"))
        yield io.BytesIO(json.dumps({"answers": {"q": {"noul": 0.5}},
                                     "usage": {"cost": 0.00001}}).encode("utf-8"))

    return opener, seen


def _client(statuses, **kw):
    op, seen = _opener(statuses)
    kw.setdefault("fallback_key", "ork")
    c = jev_client.JevClient(api_key="tk", url=jev_client.TYPESAFE_URL, model="jev-latest",
                             retries=0, opener=op, **kw)
    return c, seen


def _ask(c):
    return c.ask("x", {"q": {"type": "noul", "instructions": "?"}})


def test_402_on_direct_moves_to_openrouter_and_keeps_model_name():
    c, seen = _client([402, None, None])
    assert _ask(c) == {"q": {"noul": 0.5}}
    _ask(c)
    assert [s["url"] for s in seen] == [jev_client.TYPESAFE_URL, jev_client.OPENROUTER_URL,
                                        jev_client.OPENROUTER_URL], "넘어간 뒤로는 직행"
    assert seen[1]["body"]["model"] == jev_client.OPENROUTER_MODEL
    assert seen[1]["auth"] == "Bearer ork" and seen[0]["auth"] == "Bearer tk"
    assert c.model == "jev-latest", "결과 파일 이름의 근거라 바꾸지 않는다"
    assert c.fell_back == "openrouter(typesafe 402)"
    assert c.usage()["via_openrouter"] == 1


def test_no_fallback_when_disabled_without_key_or_for_other_judges(monkeypatch):
    c, seen = _client([402, None], fallback=False)
    try:
        _ask(c)
        raise AssertionError("402가 올라와야 한다")
    except jev_client.JevCallFailed as exc:
        assert exc.status == 402 and len(seen) == 1
    assert c.usage()["via_openrouter"] == 0

    c, seen = _client([402, None], fallback_key=None)
    monkeypatch.setattr(jev_client, "resolve_key", lambda *a, **k: None)
    try:
        _ask(c)
        raise AssertionError("OpenRouter 키가 없으면 그대로 실패")
    except jev_client.JevCallFailed:
        assert len(seen) == 1

    op, seen = _opener([402, None])
    solar = jev_client.JevClient(api_key="u", url="https://solar-mini4-jev.vercel.app/v1/systemone",
                                 model="solar-mini4-jev", retries=0, opener=op, fallback_key="ork")
    try:
        solar.ask("x", {"q": {"type": "noul", "instructions": "?"}})
        raise AssertionError("다른 판정자는 넘기지 않는다")
    except jev_client.JevCallFailed:
        assert len(seen) == 1
