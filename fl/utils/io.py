from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

