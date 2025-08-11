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
    save_snapshot(base="state", snap=None, name=None) -> Path
    iter_journal(base, *, etype=None, tag=None, since=None, until=None, limit=10) -> list[dict]
"""

from __future__ import annotations
import dataclasses as dc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

__all__ = [
    "migrate",
    "append_journal",
    "snapshot",
    "save_snapshot",
    "iter_journal",
    "rotate_journal",
    "load_snapshot",
    "diff_snapshots",
]

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


def iter_journal(
    base: str | Path,
    *,
    etype: str | None = None,
    tag: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = 10,
) -> List[Dict[str, Any]]:
    """Return journal events matching filters (AND)."""
    b = Path(base)
    p = _p(b)["journal"]
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = str(evt.get("ts", ""))
            if since and t < since:
                continue
            if until and t > until:
                continue
            if etype and str(evt.get("type", "")) != etype:
                continue
            if tag and tag not in (evt.get("tags") or []):
                continue
            out.append(evt)
    if limit is not None and limit >= 0:
        out = out[-limit:]
    return out

def rotate_journal(
    base: str | Path,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    force: bool = False,
) -> Path | None:
    """Rotate events/journal.jsonl if size exceeds ``max_bytes`` (or always if ``force``)."""
    b = Path(base)
    _ensure_dirs(b)
    p = _p(b)
    jp = p["journal"]
    if not jp.exists():
        return None
    size = jp.stat().st_size
    if not force and size <= max_bytes:
        return None
    ts = _now_iso().replace(":", "").replace("+00:00", "Z").replace("+", "Z")
    rot = p["events"] / f"journal-{ts}.jsonl"
    jp.rename(rot)
    jp.touch()
    return rot


def load_snapshot(path: str | Path) -> Dict[str, Any]:
    """Read a snapshot JSON file into a dict."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def diff_snapshots(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a minimal diff focusing on totals; positive numbers mean increases."""
    at = dict(a.get("totals", {}))
    bt = dict(b.get("totals", {}))
    keys = set(at) | set(bt)
    delta = {k: int(bt.get(k, 0)) - int(at.get(k, 0)) for k in sorted(keys)}
    return {
        "delta_totals": delta,
        "from": {"generated_at": a.get("generated_at"), "schema": a.get("schema")},
        "to": {"generated_at": b.get("generated_at"), "schema": b.get("schema")},
    }

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
    from . import events as _events  # local import to avoid circular

    files = {
        "events": _file_count(_p(b)["events"]),
        "bus": _events.files_count(b),
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


def save_snapshot(
    base: str | Path = "state",
    snap: Dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """Persist a snapshot to ``state/snaps`` and return the file path."""
    b = Path(base)
    _ensure_dirs(b)
    p = _p(b)
    p["snaps"].mkdir(parents=True, exist_ok=True)
    snap = snap or snapshot(b)
    ts = snap.get("generated_at") or _now_iso()
    safe = ts.replace(":", "").replace("+00:00", "Z").replace("+", "Z")
    label = ""
    if name:
        label = "-" + "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    out = p["snaps"] / f"{safe}{label}.json"
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

