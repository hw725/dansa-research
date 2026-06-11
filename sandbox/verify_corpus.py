"""verify_corpus.py — 로컬 입력이 동결 기준 입력과 동일한지 확인.

샌드박스 안에서 corpus 의 SHA-256 을 다시 계산해 RUN_MANIFEST.json 의 기록과
대조한다. 불일치면 즉시 중단한다.

RUN_MANIFEST.json 은 이미지에 구워져 있고(공개 가능, 해시·경로·행수만 — 번역문
평문 없음), corpus 는 런타임에 read-only 로 마운트된다.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
MANIFEST = APP / "RUN_MANIFEST.json"

# 검증 대상: corpus + (있으면) LLM 입력 manifest. 둘 다 번역문 평문을 담는다.
REQUIRED = ["data/sentence_normalized.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"[verify] RUN_MANIFEST.json 없음: {MANIFEST}")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_path = {e["path"]: e for e in manifest.get("files", [])}

    print(f"[verify] 기준 동결: commit {manifest.get('git', {}).get('commit_short')} "
          f"/ tag {manifest.get('git', {}).get('tag')}")

    ok = True
    for rel in REQUIRED:
        p = APP / rel
        ref = by_path.get(rel)
        if ref is None:
            print(f"[verify] 매니페스트에 항목 없음: {rel}")
            ok = False
            continue
        if not p.exists():
            print(f"[verify] 파일 없음(마운트 확인): {rel}")
            ok = False
            continue
        got = sha256(p)
        if got == ref["sha256"]:
            print(f"[verify] OK  {rel}  ({ref.get('rows')}행, sha256 {got[:12]}…)")
        else:
            print(f"[verify] 불일치 {rel}\n   기대 {ref['sha256'][:16]}…\n   실제 {got[:16]}…")
            ok = False

    print(f"[verify] {'PASS — 입력이 동결 기준과 동일' if ok else 'FAIL — 입력 무결성 깨짐'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
