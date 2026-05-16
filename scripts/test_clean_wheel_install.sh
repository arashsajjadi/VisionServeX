#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# v3 release gate: clean wheel install validation.
#
# Creates a fresh venv, installs the supplied wheel, and validates:
# - visionservex version
# - visionservex --help
# - visionservex model-zoo matrix
# - visionservex models load-matrix (113 models)
# - visionservex models load-matrix-run --mode unavailable_blocker_validate
# - visionservex readiness verdict (RELEASE_OK)
# - no optional dependency in base import
#
# Usage:
#   bash scripts/test_clean_wheel_install.sh dist/visionservex-3.0.0-py3-none-any.whl
#   bash scripts/test_clean_wheel_install.sh dist/visionservex-*.whl --keep

set -euo pipefail

WHEEL="${1:-}"
KEEP=false
VENV_DIR="/tmp/vsx-clean-install-v3"

if [[ -z "$WHEEL" ]]; then
  echo "{\"code\": \"WHEEL_REQUIRED\", \"fix\": \"Pass dist/visionservex-*.whl as first argument.\"}"
  exit 2
fi
if [[ ! -f "$WHEEL" ]]; then
  echo "{\"code\": \"WHEEL_NOT_FOUND\", \"path\": \"$WHEEL\"}"
  exit 3
fi

for arg in "${@:2}"; do
  [[ "$arg" == "--keep" ]] && KEEP=true
done

echo "==> Creating clean venv at $VENV_DIR"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q -U pip

echo "==> Installing wheel: $WHEEL"
"$VENV_DIR/bin/pip" install -q "$WHEEL"

VSX="$VENV_DIR/bin/visionservex"

# ─── Gate 1: version ─────────────────────────────────────────────────────────
echo "==> Gate 1: version"
VERSION=$("$VSX" version | head -1)
echo "  $VERSION"
echo "$VERSION" | grep -q "VisionServeX" || { echo "{\"code\": \"VERSION_FAIL\"}"; exit 10; }

# ─── Gate 2: --help ──────────────────────────────────────────────────────────
echo "==> Gate 2: --help"
"$VSX" --help > /dev/null || { echo "{\"code\": \"HELP_CRASH\"}"; exit 11; }
echo "  OK"

# ─── Gate 3: model-zoo matrix ────────────────────────────────────────────────
echo "==> Gate 3: model-zoo matrix"
"$VSX" model-zoo matrix --format json --out /tmp/vsx_clean_matrix.json
N_MODELS=$(python3 -c "import json; d=json.load(open('/tmp/vsx_clean_matrix.json')); print(len(d))")
echo "  n_models: $N_MODELS"
[[ "$N_MODELS" -gt 10 ]] || { echo "{\"code\": \"MATRIX_TOO_SMALL\"}"; exit 12; }

# ─── Gate 4: load-matrix ─────────────────────────────────────────────────────
echo "==> Gate 4: models load-matrix"
"$VSX" models load-matrix --format json --out /tmp/vsx_clean_load_matrix.json
N_LM=$(python3 -c "import json; d=json.load(open('/tmp/vsx_clean_load_matrix.json')); print(d['n_models'])")
echo "  n_models: $N_LM"
[[ "$N_LM" -gt 50 ]] || { echo "{\"code\": \"LOAD_MATRIX_TOO_SMALL\"}"; exit 13; }

# ─── Gate 5: load-matrix-run blockers ────────────────────────────────────────
echo "==> Gate 5: load-matrix-run (unavailable blockers)"
"$VSX" models load-matrix-run \
  --mode unavailable_blocker_validate \
  --ci-safe \
  --format json \
  --out /tmp/vsx_clean_blockers.json > /dev/null
CORE_FAIL=$(python3 -c "import json; d=json.load(open('/tmp/vsx_clean_blockers.json')); print(d['core_failures'])")
echo "  core_failures: $CORE_FAIL"
[[ "$CORE_FAIL" == "0" ]] || { echo "{\"code\": \"CORE_FAILURES_IN_BLOCKERS\"}"; exit 14; }

# ─── Gate 6: readiness verdict ───────────────────────────────────────────────
echo "==> Gate 6: readiness verdict"
VERDICT=$("$VSX" readiness verdict --json | python3 -c "import json,sys; print(json.load(sys.stdin)['verdict'])")
echo "  verdict: $VERDICT"
[[ "$VERDICT" == "RELEASE_OK" ]] || { echo "{\"code\": \"VERDICT_BLOCKED\", \"verdict\": \"$VERDICT\"}"; exit 15; }

# ─── Gate 7: no optional heavy dep at base import ────────────────────────────
echo "==> Gate 7: base import clean"
"$VENV_DIR/bin/python" -c "
import sys
# Import the base package — should not pull in torch/transformers/cv2.
import visionservex
heavy = ['torch', 'transformers', 'cv2', 'anomalib', 'mmcv', 'mmpose', 'mmdet']
for pkg in heavy:
    if pkg in sys.modules:
        print('FAIL: heavy dep imported at base level:', pkg)
        sys.exit(1)
print('OK: no heavy optional dep imported at base level')
"

echo ""
echo "{\"code\": \"OK\", \"wheel\": \"$WHEEL\", \"version\": \"$VERSION\", \"n_models\": $N_LM}"

if ! $KEEP; then
  rm -rf "$VENV_DIR"
  echo "  (venv removed)"
fi
