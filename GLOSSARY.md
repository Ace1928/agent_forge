# Glossary

- **eidctl**: CLI tool for inspecting state and journal.
- **eidosd**: Daemon shim with heartbeat loop collecting metrics.
- **snapshot**: JSON summary of current state counts and recent events.
- **event bus**: Append-only JSONL stream under `state/events/YYYYMMDD/` used for structured events.
- **journal**: Human-oriented append-only log at `state/events/journal.jsonl`.
- **metric**: (SQLite) numerical time series stored in `state/e3.sqlite`.
- **smoke test**: quick end-to-end run verifying bootstrap, state, journal, and tests.
- **beat**: one scheduler iteration producing metrics and events.
- **tick**: configured interval between beats.
- **eidtop**: curses TUI showing recent beats and metrics.
- **retention**: pruning strategy keeping metrics, events, and journal size bounded.
- **maintenance cycle**: periodic run deleting old metrics and events and rotating journal.
- **prune_metrics**: DB helper removing older metric rows beyond a per-key limit.
- **prune_old_days**: event bus helper deleting day directories older than ``keep_days``.
- **prune_journal**: wrapper rotating journal file when size exceeds threshold.
