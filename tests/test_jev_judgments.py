"""Jev 실행기·채점기를 가짜 클라이언트로 잰다(실호출 없음).

raw 판정 CSV(로컬 전용)가 있어야 돈다 — 없으면 건너뛴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compute_jev_agreement as cja  # noqa: E402
import jev_client  # noqa: E402
import run_jev_judgments as rj  # noqa: E402

RAW = rj.REPO / "results" / "gpt5mini" / "section3_judgments.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw 판정 CSV 없음(로컬 전용)")


class FakeClient:
    """문항 텍스트에 «일반»이 들어 있으면 높게 — 형식만 재는 가짜. 호출 수는 진짜처럼 센다."""

    model = jev_client.TYPESAFE_MODEL

    def __init__(self, max_calls=1000, fail_on=None):
        self.calls_made = 0
        self.max_calls = max_calls
        self.fail_on = fail_on
        self.seen = []

    def gate(self, planned):
        if planned > self.max_calls:
            raise jev_client.JevGateExceeded("over")

    def ask(self, state, questions):
        self.calls_made += 1
        self.seen.append((state, questions))
        if self.fail_on == self.calls_made:
            raise jev_client.JevCallFailed("boom")
        return {qid: {"noul": 0.9 if "군자" in q["instructions"] else 0.2}
                for qid, q in questions.items()}

    def usage(self):
        return {"calls": self.calls_made}


@pytest.fixture()
def tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "OUT_DIR", tmp_path)
    return tmp_path


def test_gate_refuses_before_any_call(tmp_out):
    c = FakeClient(max_calls=2)
    with pytest.raises(jev_client.JevGateExceeded):
        rj.run("section3", "pos", 100, 20, c)  # 5회 계획 > 상한 2
    assert c.calls_made == 0
    assert not rj.out_path("section3", "pos").exists()


def test_question_quotes_sentence_and_state_has_no_marker(tmp_out):
    items = rj.order(rj.load_items("section3"))[:3]
    state, qs = rj.build_call("section3", "pos", items)
    assert len(qs) == 3
    for i, it in enumerate(items):
        assert it["text"] in qs[f"q{i + 1}"]["instructions"]
        assert qs[f"q{i + 1}"]["type"] == "noul"
    # 현토 표지·범주는 모델에 가지 않는다
    for word in ("하나니라", "汎論", "marker"):
        assert word not in state


def test_resume_and_no_text_saved(tmp_out):
    c = FakeClient(fail_on=2)
    r = rj.run("section3", "pos", 60, 20, c)
    assert len(r["errors"]) == 1 and r["answered"] == 40
    lines = rj.out_path("section3", "pos").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert set(rec) == {"key", "marker_type", "noul", "model", "batch"}  # 번역문은 남기지 않는다
    c2 = FakeClient()
    r2 = rj.run("section3", "pos", 60, 20, c2)
    assert c2.calls_made == 1 and r2["asked"] == 20  # 실패한 묶음만 다시


def test_pos_and_flip_share_order():
    a = rj.order(rj.load_items("section3"))[:50]
    b = rj.order(rj.load_items("section3"))[:50]
    assert [x["key"] for x in a] == [x["key"] for x in b]


def test_score_runs_and_flip_pairs(tmp_out):
    rj.run("section3", "pos", 100, 20, FakeClient())
    rj.run("section3", "neg", 100, 20, FakeClient())
    res = cja.score_section("section3")
    assert res["n"] == 100
    assert res["flip"]["n"] == 100
    assert 0.0 <= res["auc_unanimous"] <= 1.0
    # 가짜는 틀과 무관하게 같은 값을 주므로 p+q가 1에서 벗어나야 한다 — 대조군이 그걸 잡는가
    assert res["flip"]["both_yes_pct"] > 0 or res["flip"]["both_no_pct"] > 0


def test_kappa_auc_basics():
    assert cja.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cja.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert cja.auc([0.5], [0.5]) == 0.5
