"""Run VisionServeX CLI commands from notebooks."""

from __future__ import annotations

import json
import subprocess
import sys


def run(args: list[str], *, timeout: int = 300, cwd=None) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    payload = {}
    try:
        payload = json.loads(proc.stdout.strip())
    except Exception:
        for line in proc.stdout.splitlines():
            if line.strip().startswith("{"):
                try:
                    payload = json.loads(line.strip())
                    break
                except Exception:
                    pass
    payload["_returncode"] = proc.returncode
    payload["_stderr_tail"] = proc.stderr[-300:]
    return payload
