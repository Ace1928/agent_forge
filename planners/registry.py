from __future__ import annotations
from core.contracts import Goal
def choose(goal: Goal) -> tuple[str, dict]:
    if "hygiene" in goal.title.lower():
        return "htn", {"template":"hygiene"}
    return "htn", {"template":"hygiene"}  # default for now
