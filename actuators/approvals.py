from __future__ import annotations
from pathlib import Path
from typing import Sequence

_DEFAULT_ALLOW = ["uv","pytest","ruff","black","git","bash","sh","make","python","pip"]
_DEFAULT_DENY_TOKENS = ["curl","wget","ssh","nc","iptables","docker"]

def allowed_cmd(cmd: Sequence[str], cwd: str, *,
                allow_prefixes=None, deny_tokens=None) -> tuple[bool,str]:
    """Return (ok, reason). Enforces prefix allowlist and deny token scan.
       CWD must be inside repo (no path escapes)."""
    if not cmd: return False, "empty command"
    allow = list(allow_prefixes or _DEFAULT_ALLOW)
    deny = set(deny_tokens or _DEFAULT_DENY_TOKENS)
    flat = " ".join(cmd)
    if any(t in flat for t in deny):
        return False, "denied token present"
    exe = Path(cmd[0]).name
    if not any(exe.startswith(p) for p in allow):
        return False, f"disallowed exe: {exe}"
    # scope cwd: disallow running from filesystem root
    here = Path(cwd).resolve()
    if here == Path("/"):
        return False, "cwd escapes repo"
    return True, "ok"
