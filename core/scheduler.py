#!/usr/bin/env python3
"""Simple scheduler loop utilities for eidosd."""

from __future__ import annotations

import random
import signal
import time
import json
from dataclasses import dataclass
from typing import Callable, Optional
from core.state import add_plan, add_step, list_steps, list_steps_for_goal
from core.state import add_goal, list_goals
from actuators.shell_exec import run_step
from planners.registry import choose
from planners.htn import materialize
from . import events as BUS

STATE_DIR = "state"


@dataclass
class BeatCfg:
    tick_secs: float
    jitter_ms: int = 0
    max_backoff_secs: float = 30.0
    max_beats: int = 0  # 0 = infinite


class StopToken:
    """Cooperative stop helper."""

    def __init__(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped


def run_loop(cfg: BeatCfg, beat_fn: Callable[[], None], *, stop: StopToken) -> None:
    """Run ``beat_fn`` in a loop with jitter and exponential backoff."""

    beats = 0
    backoff = 1.0
    while not stop.stopped():
        if cfg.max_beats and beats >= cfg.max_beats:
            break
        try:
            beat_fn()
            beats += 1
            backoff = 1.0
            if cfg.tick_secs > 0:
                jitter = 0.0
                if cfg.jitter_ms:
                    jitter = random.uniform(-cfg.jitter_ms / 1000.0, cfg.jitter_ms / 1000.0)
                time.sleep(max(cfg.tick_secs + jitter, 0.0))
            continue
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, cfg.max_backoff_secs)
            continue


def install_sigint(stop: StopToken) -> None:
    """Install SIGINT/SIGTERM handlers that set ``stop``."""

    def handler(signum, frame):  # pragma: no cover - simple forwarding
        stop.stop()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def sense(ctx):  # existing hook: leave as-is if already present
    BUS.append(STATE_DIR, "tick.sense")


def plan(ctx, goal):
    kind, meta = choose(goal)
    p = add_plan(STATE_DIR, goal.id, kind, meta)
    for s in materialize(meta["template"], goal.title):
        add_step(STATE_DIR, p.id, s["idx"], s["name"], json.dumps(s["cmd"]),
                 s["budget_s"], "todo")
    BUS.append(STATE_DIR, "plan.created", {"goal_id": goal.id, "plan_id": p.id})


def gate(ctx, step_row):
    # approvals handled inside shell_exec.run_step; we still log intent
    BUS.append(STATE_DIR, "gate.checked", {"step_id": step_row.id})
    return True


def act(ctx, step_row):
    cmd = json.loads(step_row.cmd)
    res = run_step(STATE_DIR, step_row.id, cmd, cwd=".", budget_s=step_row.budget_s)
    BUS.append(STATE_DIR, "act.exec", {"step_id": step_row.id, "res": res})
    return res


def verify(ctx, step_row, res):
    # naive verification: rc==0 -> ok else fail
    status = "ok" if res.get("rc", 1) == 0 and res.get("status") == "ok" else "fail"
    # update DB status quickly (lightweight inline update)
    import sqlite3, pathlib
    db = (pathlib.Path(STATE_DIR) / "e3.sqlite")
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE steps SET status=? WHERE id=?", (status, step_row.id))
        conn.commit()
    finally:
        conn.close()
    BUS.append(STATE_DIR, "verify.done", {"step_id": step_row.id, "status": status})
