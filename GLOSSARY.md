# Glossary

- **eidctl**: CLI tool for inspecting state and journal.
- **eidosd**: Daemon shim that records a metric, event, and journal note.
- **snapshot**: JSON summary of current state counts and recent events.
- **event bus**: Append-only JSONL stream under `state/events/YYYYMMDD/` used for structured events.
- **journal**: Human-oriented append-only log at `state/events/journal.jsonl`.
- **metric**: (SQLite) numerical time series stored in `state/e3.sqlite`.
