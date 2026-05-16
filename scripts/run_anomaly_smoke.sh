#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Real Anomalib + PatchCore smoke runner. Creates an isolated venv,
# installs anomalib, then trains PatchCore on tests/fixtures/anomaly/normal
# for one epoch and runs predict on the bundled defect image.
#
# Usage:
#   bash scripts/run_anomaly_smoke.sh [venv_dir]
#
# Verified on host: anomalib 2.4.2, torch 2.12, Python 3.13 in venv (2026-05-16).

set -euo pipefail

VENV_DIR="${1:-/tmp/vsx-anomaly-venv}"
OUT_DIR="${2:-/tmp/vsx_patchcore_real}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install -q -U pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -q anomalib
"$VENV_DIR/bin/python" -m pip install -q -e .

rm -rf "$OUT_DIR"
"$VENV_DIR/bin/python" - <<PY
from visionservex.integrations.anomalib_adapter import PatchCoreAdapter
adapter = PatchCoreAdapter()
train = adapter.train(
    data_dir="tests/fixtures/anomaly/normal",
    out_dir="$OUT_DIR",
    max_epochs=1,
)
print("train:", train)
predict = adapter.predict(
    model_dir="$OUT_DIR",
    image_path="tests/fixtures/anomaly/test.jpg",
)
print("predict:", predict)
PY
