#!/usr/bin/env python3
"""Simple scheduler loop utilities for eidosd."""

from __future__ import annotations

import random
import signal
import time
from dataclasses import dataclass
from typing import Callable, Optional


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
