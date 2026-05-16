#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Real TotalSegmentator smoke. Requires a user-supplied NIfTI volume —
# VisionServeX never bundles medical imaging data.
#
# Usage:
#   bash scripts/run_totalsegmentator_smoke.sh /path/to/ct.nii.gz [out_dir]
#
# Structured exits:
#   2 PYTHON_REQUIRED
#   3 INPUT_NOT_FOUND
#   4 TOTALSEGMENTATOR_REQUIRED

set -euo pipefail

INPUT="${1:-}"
OUT_DIR="${2:-/tmp/vsx-totalsegmentator-out}"
VENV_DIR="${VENV_DIR:-/tmp/vsx-totalsegmentator-venv}"

if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
  cat <<'JSON'
{
  "code": "INPUT_NOT_FOUND",
  "fix": "Pass the path to a user-owned NIfTI volume (.nii or .nii.gz). VisionServeX does not bundle medical data."
}
JSON
  exit 3
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install -q -U pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -q TotalSegmentator nibabel

mkdir -p "$OUT_DIR"
"$VENV_DIR/bin/python" - <<PY
import sys, time
try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError as exc:
    print({"code": "TOTALSEGMENTATOR_REQUIRED", "fix": "pip install TotalSegmentator", "import_error": str(exc)})
    sys.exit(4)

t0 = time.time()
totalsegmentator("$INPUT", "$OUT_DIR")
dt = time.time() - t0
print({
    "code": "OK",
    "input": "$INPUT",
    "out_dir": "$OUT_DIR",
    "runtime_s": round(dt, 2),
})
PY
