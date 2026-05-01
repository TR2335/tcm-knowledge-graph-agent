import os
from pathlib import Path


def resolve_from_project_root(relative_path: str) -> str:
    current = Path(__file__).resolve()
    for _ in range(10):
        if (current / "main.py").exists() or (current / ".env").exists():
            return str(current / relative_path)
        if current.parent == current:
            break
        current = current.parent
    return str(Path(relative_path).resolve())
