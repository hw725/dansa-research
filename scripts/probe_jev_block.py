#!/usr/bin/env python3
"""TypeSafe Jev 차단(403, Cloudflare 1010)이 풀렸는지 주기적으로 본다.

2026-09-23 오전에는 통과하던 호출이 같은 날 08:48부터 403 «error code: 1010»으로 막혔다.
1010은 클라이언트 서명으로 막는 코드라 우회하지 않고 **풀리기를 기다린다.**
탐침은 한 번에 아주 짧은 noul 질문 1개(입력 수십 토큰, $0.00001 미만)다.
결과는 logs/jev_probe.jsonl에 한 줄씩 남기고, 통과하면 종료 코드 0으로 끝난다.

사용: py scripts/probe_jev_block.py --every 1200 --hours 24
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_client  # noqa: E402

LOG = Path(__file__).resolve().parent.parent / "logs" / "jev_probe.jsonl"


def probe() -> dict:
    c = jev_client.JevClient(max_calls=1, retries=0, timeout=30)
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds")}
    if not c.has_key:
        return {**rec, "ok": False, "status": None, "detail": "no_key"}
    try:
        c.ask("probe", {"q": {"type": "noul", "instructions": "Is this a probe?"}})
        return {**rec, "ok": True, "status": 200, "detail": ""}
    except Exception as e:  # noqa: BLE001
        return {**rec, "ok": False, "status": getattr(e, "status", None),
                "detail": " ".join((getattr(e, "detail", "") or str(e))[:80].split())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1200, help="탐침 간격(초)")
    ap.add_argument("--hours", type=float, default=24.0, help="최대 대기 시간")
    a = ap.parse_args(argv)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.hours * 3600
    while True:
        rec = probe()
        with LOG.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if rec["ok"]:
            return 0
        if time.time() + a.every > deadline:
            return 3
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
