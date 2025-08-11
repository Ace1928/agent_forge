#!/usr/bin/env python3
"""
Minimal state layer for Eidos E3.

- state/ layout assumed by bootstrap:
    state/
      events/
      vector_store/
      weights/
      adapters/
      snaps/
      meta/ (created on demand)

- JSONL journal at: state/events/journal.jsonl
- Schema/version marker: state/meta/version.json

Public API (stdlib only):
    migrate(base="state") -> int
    append_journal(base, text, *, etype="note", tags=None, extra=None) -> dict
    snapshot(base="state", *, last=5) -> dict
    save_snapshot(base="state", snap=None) -> Path
"""

from __future__ import annotations
import dataclasses as dc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

__all__ = ["migrate", "append_journal", "snapshot", "save_snapshot"]

SCHEMA_VERSION = 1

# ---------- internal paths ----------

def _p(base: Path) -> Dict[str, Path]:
    return {
        "base": base,
        "events": base / "events",
        "meta": base / "meta",
        "snaps": base / "snaps",
        "journal": base / "events" / "journal.jsonl",
        "version": base / "meta" / "version.json",
    }

def _ensure_dirs(base: Path) -> None:
    p = _p(base)
    p["events"].mkdir(parents=True, exist_ok=True)
    p["meta"].mkdir(parents=True, exist_ok=True)

# ---------- migrations ----------

def migrate(base: str | Path = "state") -> int:
    """Ensure directory structure and bump/create version marker if absent."""
    b = Path(base)
    _ensure_dirs(b)
    paths = _p(b)
    vp = paths["version"]
    jp = paths["journal"]
    if not jp.exists():
        jp.touch()
    if not vp.exists():
        vp.write_text(json.dumps({"schema": SCHEMA_VERSION, "created_at": _now_iso()}), encoding="utf-8")
        return SCHEMA_VERSION
    try:
        meta = json.loads(vp.read_text(encoding="utf-8"))
        return int(meta.get("schema", SCHEMA_VERSION))
    except Exception:
        # if corrupted, do not overwrite silently; surface but return current
        raise RuntimeError(f"Corrupted version file: {vp}")

# ---------- journal ----------

def append_journal(
    base: str | Path,
    text: str,
    *,
    etype: str = "note",
    tags: List[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Append a single journal event as JSONL and return the event dict."""
    b = Path(base)
    _ensure_dirs(b)
    evt = {
        "ts": _now_iso(),
        "type": str(etype),
        "text": str(text),
        "tags": list(tags or []),
        "extra": dict(extra or {}),
    }
    jp = _p(b)["journal"]
    with jp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    return evt

# ---------- snapshot ----------

def snapshot(base: str | Path = "state", *, last: int = 5) -> Dict[str, Any]:
    """Compute a light snapshot: counts by entity family, last ``N`` journal entries, file counts."""
    b = Path(base)
    _ensure_dirs(b)
    version = migrate(b)  # idempotent

    # parse journal (if any)
    journal_path = _p(b)["journal"]
    last_events: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"goal": 0, "plan": 0, "step": 0, "run": 0, "metric": 0, "journal": 0, "note": 0}
    if journal_path.exists():
        buf: List[Dict[str, Any]] = []
        with journal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = str(evt.get("type", "note"))
                family = etype.split(".", 1)[0] if etype else "note"
                counts[family] = counts.get(family, 0) + 1
                buf.append(evt)
        last_events = buf[-last:] if last > 0 else []

    # files (quick health signal)
    files = {
        "events": _file_count(_p(b)["events"]),
        "vector_store": _file_count(b / "vector_store"),
        "weights": _file_count(b / "weights"),
        "adapters": _file_count(b / "adapters"),
        "snaps": _file_count(b / "snaps"),
    }

    return {
        "schema": version,
        "base": str(b.resolve()),
        "totals": counts,
        "last_events": last_events,
        "files": files,
        "generated_at": _now_iso(),
    }


def save_snapshot(base: str | Path = "state", snap: Dict[str, Any] | None = None) -> Path:
    """Persist a snapshot to ``state/snaps`` and return the file path."""
    b = Path(base)
    _ensure_dirs(b)
    p = _p(b)
    p["snaps"].mkdir(parents=True, exist_ok=True)
    snap = snap or snapshot(b)
    ts = snap.get("generated_at") or _now_iso()
    safe = ts.replace(":", "").replace("+00:00", "Z").replace("+", "Z")
    out = p["snaps"] / f"{safe}.json"
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out

# ---------- helpers ----------

def _file_count(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob("*") if _.is_file())

def _now_iso() -> str:
    """Return UTC time in ISO-8601 with trailing 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":  # pragma: no cover
    # tiny manual smoke test
    migrate("state")
    append_journal("state", "hello world", etype="note", tags=["smoke"])
    print(json.dumps(snapshot("state"), indent=2))

