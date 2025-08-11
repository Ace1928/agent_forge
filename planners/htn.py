from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Dict, Any

def materialize(template_name: str, goal_title: str) -> List[Dict[str, Any]]:
    p = Path(__file__).resolve().parent / "templates" / f"{template_name}.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    steps = []
    for i, s in enumerate(data.get("steps", [])):
        steps.append({
            "idx": i,
            "name": s["name"],
            "cmd": s["cmd"],
            "budget_s": float(s.get("budget_s", 60)),
        })
    return steps
