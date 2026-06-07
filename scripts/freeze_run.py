#!/usr/bin/env python3
"""freeze_run.py — 현재 dansa-research 상태를 불변 스냅샷으로 동결한다.

동결은 세 층으로 구성된다.

1. git 주석 태그 ``frozen/run-<date>`` — 추적 콘텐츠 전체를 현재 commit 해시에
   고정한다. origin 에 push 하면 오프사이트 미러가 된다(코드·문서·추적 보고서·
   익명 CSV 전부 포함).
2. 미추적 핵심 입력/출력 물리 백업 — git 이 잡지 않는, 재생성 불가능한 파일을
   ``archive/<date>_frozen_run/`` 아래에 상대경로 그대로 복사한다(raw corpus,
   raw 판정 CSV, supplement 입력, llm_manifests, logs).
3. ``RUN_MANIFEST.json`` — 정본 산출물 + raw corpus + 복사한 핵심 파일의
   SHA-256 무결성 앵커. 번역문 평문은 들어가지 않는다(해시·경로·행수만).

제외:
- ``archive/`` (2GB, 기존 백업·구버전 보관소 — 재귀/중복).
- ``analysis/`` 대용량 미추적 임베딩·KG 바이너리(재생성 가능; 추적 보고서는
  git 태그에 이미 포함). ``--include-analysis`` 로 물리 복사에 포함할 수 있다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                   encoding="utf-8").strip()


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def count_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)  # 헤더 제외


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def gather(include_analysis: bool) -> tuple[list[Path], list[Path]]:
    """(해시 대상, 물리복사 대상) 경로 목록을 만든다."""
    results = REPO / "results"
    data = REPO / "data"
    logs = REPO / "logs"

    # 추적 정본 산출물 (git 태그로도 고정되지만 단건 검증용으로 해시).
    canonical = sorted(results.glob("*.json"))
    canonical += sorted(results.glob("**/*_anon.csv"))
    for name in ("sentence_normalized_anon.csv", "book_names.txt"):
        p = data / name
        if p.exists():
            canonical.append(p)

    # 미추적 핵심 — git 이 못 잡고 재생성 비용/불가능한 입력·출력.
    raw: list[Path] = []
    raw += [p for p in sorted(data.glob("*.csv")) if not p.stem.endswith("_anon")]
    if (data / "llm_manifests").exists():
        raw += sorted((data / "llm_manifests").rglob("*"))
    raw += [p for p in sorted(results.glob("**/*judgments*.csv"))
            if not p.stem.endswith("_anon")]
    if logs.exists():
        raw += sorted(logs.rglob("*"))
    if include_analysis:
        analysis = REPO / "analysis"
        raw += [p for p in sorted(analysis.rglob("*"))
                if p.is_file() and ".git" not in p.parts]

    raw = [p for p in raw if p.is_file()]
    # 해시 대상 = 정본 + raw(중복 제거, 순서 유지)
    seen: set[Path] = set()
    hash_targets: list[Path] = []
    for p in canonical + raw:
        if p not in seen:
            seen.add(p)
            hash_targets.append(p)
    return hash_targets, raw


def main() -> int:
    ap = argparse.ArgumentParser(description="현재 저장소 상태 동결")
    ap.add_argument("--include-analysis", action="store_true",
                    help="analysis/ 대용량 미추적 파일도 물리 복사에 포함")
    ap.add_argument("--no-tag", action="store_true", help="git 태그 생성 생략")
    args = ap.parse_args()

    date = dt.date.today().isoformat()
    head = git("rev-parse", "HEAD")
    head_short = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = git("status", "--porcelain")
    dirty_tracked = [ln for ln in porcelain.splitlines() if not ln.startswith("??")]
    untracked = [ln for ln in porcelain.splitlines() if ln.startswith("??")]

    freeze_dir = REPO / "archive" / f"{date}_frozen_run"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    hash_targets, raw = gather(args.include_analysis)
    tracked_set = set(git("ls-files").splitlines())

    print(f"동결 대상 해시 {len(hash_targets)}건 · 물리복사 {len(raw)}건")

    # 물리 백업: 상대경로 보존 복사
    copied_bytes = 0
    for src in raw:
        dest = freeze_dir / src.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied_bytes += dest.stat().st_size

    # 매니페스트
    files = []
    total = 0
    for p in hash_targets:
        digest, size = sha256_file(p)
        total += size
        files.append({
            "path": rel(p),
            "sha256": digest,
            "bytes": size,
            "rows": count_rows(p),
            "tracked": rel(p) in tracked_set,
            "backed_up": p in raw,
        })

    manifest = {
        "generated_at": date,
        "purpose": "frozen reference run — 재현 샌드박스 비교 기준",
        "git": {
            "commit": head,
            "commit_short": head_short,
            "branch": branch,
            "tracked_clean": not dirty_tracked,
            "untracked_wip": len(untracked),
            "tag": None if args.no_tag else f"frozen/run-{date}",
        },
        "integrity_anchor": {
            "note": "raw corpus 해시는 통계·판정이 유래한 입력의 신원 증명. "
                    "샌드박스 내부에서 이 해시를 재계산해 일치를 확인하면, "
                    "내용을 보지 않고도 '그 입력'임을 검증할 수 있다.",
            "raw_corpus": "data/sentence_normalized.csv",
        },
        "physical_backup_dir": rel(freeze_dir),
        "physical_backup_bytes": copied_bytes,
        "excluded": {
            "archive/": "기존 백업·구버전 보관소(재귀/중복)",
            "analysis/ (대용량 미추적)": "재생성 가능한 임베딩·KG; 추적 보고서는 git 태그에 포함"
                if not args.include_analysis else "이번 실행에 포함됨",
        },
        "file_count": len(files),
        "total_hashed_bytes": total,
        "files": files,
    }

    manifest_path = REPO / "RUN_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {rel(manifest_path)} ({len(files)} files, "
          f"{total/1e6:.1f} MB hashed)")
    print(f"물리 백업 → {rel(freeze_dir)} ({copied_bytes/1e6:.1f} MB)")

    # git 태그
    if not args.no_tag:
        tag = f"frozen/run-{date}"
        existing = git("tag", "-l", tag)
        if existing:
            print(f"태그 {tag} 이미 존재 — 생략 (재생성하려면 git tag -d {tag})")
        else:
            subprocess.run(["git", "tag", "-a", tag, "-m",
                            f"Frozen reference run {date} (commit {head_short})"],
                           cwd=REPO, check=True)
            print(f"git 태그 생성: {tag} → {head_short}")
            print(f"  (공개 앵커로 push: git push origin {tag})")

    if dirty_tracked:
        print(f"\n주의: 추적 파일 미커밋 변경 {len(dirty_tracked)}건 — "
              f"태그는 마지막 commit({head_short}) 기준이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
