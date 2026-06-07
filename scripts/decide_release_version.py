#!/usr/bin/env python3
"""Decide the release version from the V3 gate matrix.

v3.0.0 only if every blocking gate is PASS*. Otherwise the next v2.x minor.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def next_v2(current: str) -> str:
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)", current)
    major, minor, _patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"v{major}.{minor + 1}.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-matrix", default="notebook/99_final_report/reports/v3_gate_matrix.json")
    ap.add_argument("--current-version", default="v2.58.0")
    ap.add_argument("--if-pass", default="v3.0.0")
    ap.add_argument("--write", default="VERSION_DECISION.txt")
    args = ap.parse_args()

    gm = json.loads(Path(args.gate_matrix).read_text())
    v3_ready = gm["v3_ready"]
    blocking = gm["blocking_failures"]
    version = args.if_pass if v3_ready else next_v2(args.current_version)
    verdict = "V3 RELEASED" if v3_ready else "V3 NOT RELEASED"
    line = (
        f"{version} {verdict}\n"
        f"v3_ready={v3_ready}\n"
        f"blocking_gate_failures={','.join(blocking) if blocking else 'none'}\n"
        f"reason={'all blocking V3 gates pass' if v3_ready else 'blocking V3 gates fail: ' + ', '.join(blocking)}\n"
    )
    Path(args.write).write_text(line)
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
