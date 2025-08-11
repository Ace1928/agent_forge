# Eidos E3 Agent Forge

Minimal scaffolding for the Eidos E3 system.

## CLI tools
- `bin/eidctl` – inspect state snapshots and manage the JSONL journal.
- `bin/eidosd` – daemon shim; `--once` writes a metric to SQLite, emits an event,
  and appends a journal note.

## Core modules
- `core/config.py` – strict YAML config loader.
- `core/state.py` – state migrations, journal, snapshots.
- `core/events.py` – append-only event bus with rotation.
- `core/db.py` – SQLite helpers for metrics and journal tables.

## Bootstrap
Run `scripts/bootstrap.sh` to seed configs, create a virtual environment and
install dependencies.

## Testing

Run the test suite with the repository's virtual environment to ensure
dependencies like PyYAML are present:

```
.venv/bin/python -m pytest -q
```
