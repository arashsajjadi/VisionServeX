"""General notebook utilities."""

from __future__ import annotations

import json
from pathlib import Path


def write_status(task_dir: Path | str, **kwargs) -> None:
    p = Path(task_dir) / "reports/status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(kwargs, indent=2, default=str))


def assert_no_forbidden(text: str) -> None:
    from shared.display import scan_text

    hits = scan_text(text)
    if hits:
        raise AssertionError(f"Forbidden strings found: {hits}")
