#!/usr/bin/env python3
"""SQLite helpers for Eidos E3.

This module keeps the database footprint tiny and stdlib-only.
Tables:
    metrics(ts TEXT, key TEXT, value REAL)
    journal(ts TEXT, type TEXT, text TEXT)
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timezone

__all__ = ["init_db", "insert_metric", "insert_journal"]


def init_db(base: str | Path = "state") -> Path:
    """Ensure ``e3.sqlite`` exists with required tables and return its path."""
    b = Path(base)
    b.mkdir(parents=True, exist_ok=True)
    db_path = b / "e3.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics(ts TEXT, key TEXT, value REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS journal(ts TEXT, type TEXT, text TEXT)"
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def insert_metric(base: str | Path, key: str, value: float, ts: str | None = None) -> None:
    ts = ts or _now_iso()
    db = init_db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute("INSERT INTO metrics(ts, key, value) VALUES (?, ?, ?)", (ts, key, float(value)))
        conn.commit()
    finally:
        conn.close()


def insert_journal(base: str | Path, etype: str, text: str, ts: str | None = None) -> None:
    ts = ts or _now_iso()
    db = init_db(base)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO journal(ts, type, text) VALUES (?, ?, ?)",
            (ts, etype, text),
        )
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":  # pragma: no cover
    insert_metric("state", "smoke", 1.0)
    insert_journal("state", "note", "hi")
