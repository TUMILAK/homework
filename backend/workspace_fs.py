from pathlib import Path

from fastapi import HTTPException

from .config import WORKSPACE_ROOT


def ensure_workspace() -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def resolve_safe(relative_path: str) -> Path:
    """Resolve a path under WORKSPACE_ROOT; reject traversal."""
    ensure_workspace()
    raw = (relative_path or "").strip().replace("\\", "/").lstrip("/")
    target = (WORKSPACE_ROOT / raw).resolve()
    workspace_resolved = WORKSPACE_ROOT.resolve()
    if target == workspace_resolved or workspace_resolved in target.parents:
        return target
    raise HTTPException(status_code=400, detail="Invalid path")
