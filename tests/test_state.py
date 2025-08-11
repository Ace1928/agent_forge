from pathlib import Path
import json
from core import state as S

def test_migrate_and_snapshot(tmp_path: Path):
    base = tmp_path / "state"
    v = S.migrate(base)
    assert v >= 1
    snap = S.snapshot(base)
    assert snap["schema"] == v
    assert snap["totals"]["note"] == 0
    assert snap["files"]["events"] >= 1  # at least version/journal files

def test_journal_counts(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    S.append_journal(base, "g1", etype="goal.created")
    S.append_journal(base, "p1", etype="plan.created")
    S.append_journal(base, "s1", etype="step.completed")
    S.append_journal(base, "metric logged", etype="metric.logged")
    S.append_journal(base, "note only")  # default type note

    snap = S.snapshot(base)
    assert snap["totals"]["goal"] == 1
    assert snap["totals"]["plan"] == 1
    assert snap["totals"]["step"] == 1
    assert snap["totals"]["metric"] == 1
    assert snap["totals"]["note"] >= 1  # includes our note
    assert len(snap["last_events"]) <= 5


def test_snapshot_save(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    path = S.save_snapshot(base)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] >= 1
    assert "totals" in data


def test_journal_tags(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    S.append_journal(base, "hello", etype="note", tags=["x", "y"])
    snap = S.snapshot(base)
    assert snap["last_events"][-1]["tags"] == ["x", "y"]


def test_migrate_idempotent(tmp_path: Path):
    base = tmp_path / "state"
    v1 = S.migrate(base)
    v2 = S.migrate(base)
    assert v1 == v2


def test_snapshot_ignores_bad_journal_lines(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    jp = (base / "events" / "journal.jsonl")
    jp.write_text('{"type":"note","text":"ok"}\nTHIS IS NOT JSON\n', encoding="utf-8")
    snap = S.snapshot(base)
    assert snap["totals"]["note"] == 1


def test_snapshot_last_param(tmp_path: Path):
    base = tmp_path / "state"
    S.migrate(base)
    for i in range(7):
        S.append_journal(base, f"n{i}")
    snap = S.snapshot(base, last=3)
    assert len(snap["last_events"]) == 3
    snap2 = S.snapshot(base, last=0)
    assert snap2["last_events"] == []

