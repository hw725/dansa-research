"""vendor_clients.py — 벤더별 OpenAI 호환 클라이언트.

재현 실행은 동결 기준과 동일한 프롬프트·파서를 쓰고, 이 모듈에서 모델별
클라이언트와 preflight 호출만 담당한다. 모델 ID와 base_url 은 환경변수로
덮어쓸 수 있다.
"""
from __future__ import annotations

import os
import re

from openai import AsyncOpenAI

MOCK = os.environ.get("SANDBOX_MOCK") == "1"

# model_key -> 직결 설정. 모델 ID·base_url 은 env 로 덮어쓸 수 있다.
SANDBOX_MODELS: dict[str, dict] = {
    "gpt5mini": {
        "display": "GPT-5-mini",
        "vendor": "openai",
        "model_id": os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "base_url": os.environ.get("OPENAI_BASE_URL") or None,  # None = api.openai.com
        "key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "display": "Gemini 2.5 Flash",
        "vendor": "google",
        "model_id": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "base_url": os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "key_env": "GEMINI_API_KEY",
    },
    "claude_sonnet": {
        "display": "Claude Sonnet",
        "vendor": "anthropic",
        "model_id": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/"),
        "key_env": "ANTHROPIC_API_KEY",
    },
}


def available_models() -> list[str]:
    """키가 주입된 모델만 반환한다(MOCK 이면 전부)."""
    if MOCK:
        return list(SANDBOX_MODELS)
    return [mk for mk, cfg in SANDBOX_MODELS.items()
            if os.environ.get(cfg["key_env"], "").strip()]


def get_client(model_key: str) -> AsyncOpenAI:
    cfg = SANDBOX_MODELS[model_key]
    if MOCK:
        return AsyncOpenAI(api_key="mock")  # 구성만 — call_vendor 가 단락하므로 미사용
    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise SystemExit(f"[sandbox] 키 없음: {cfg['key_env']} (실행 환경변수 확인 필요)")
    kwargs: dict = {"api_key": key}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return AsyncOpenAI(**kwargs)


def _request_kwargs(cfg: dict, prompt: str) -> dict:
    kwargs: dict = {
        "model": cfg["model_id"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg["vendor"] == "openai":
        kwargs["store"] = False
        # gpt-5 계열은 추론에 토큰을 쓰므로 max_tokens 미지정(동결 기준과 동일)
    else:
        kwargs["max_tokens"] = 200
    return kwargs


def _mock_response(prompt: str) -> str:
    """프롬프트의 번호 항목 수만큼 결정적 O/X 를 만든다(네트워크 없음)."""
    n = len(re.findall(r"(?m)^\s*\d+\.\s", prompt)) or 1
    return "\n".join(f"{i}. {'O' if i % 2 else 'X'}" for i in range(1, n + 1))


async def call_vendor(client: AsyncOpenAI, model_key: str, prompt: str) -> str:
    """단일 호출. 동결 기준의 call_llm 규칙을 따른다."""
    if MOCK:
        return _mock_response(prompt)
    cfg = SANDBOX_MODELS[model_key]
    try:
        resp = await client.chat.completions.create(**_request_kwargs(cfg, prompt))
        return resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        # 예외 메시지 본문은 출력하지 않는다.
        print(f"  [{model_key}] 호출 오류: {type(exc).__name__}")
        return ""


async def preflight(model_keys: list[str]) -> list[tuple[str, bool, str]]:
    """실제 호출 전 각 모델을 번역문 없는 탐침 1건으로 확인.

    키·모델ID·엔드포인트가 틀리면 여기서 명확한 에러로 잡는다.
    탐침 프롬프트엔 번역문이 없으므로 에러 상세를 노출해도 안전하다.
    """
    out: list[tuple[str, bool, str]] = []
    probe = '번호와 O/X로만 답하세요.\n1. test\n예: "1. O"'
    for mk in model_keys:
        if MOCK:
            out.append((mk, True, "mock"))
            continue
        cfg = SANDBOX_MODELS[mk]
        try:
            client = get_client(mk)
            resp = await client.chat.completions.create(**_request_kwargs(cfg, probe))
            txt = (resp.choices[0].message.content or "").strip()
            out.append((mk, bool(txt), "ok" if txt else "빈 응답"))
        except Exception as exc:  # noqa: BLE001
            out.append((mk, False, f"{type(exc).__name__}: {str(exc)[:90]}"))
    return out
