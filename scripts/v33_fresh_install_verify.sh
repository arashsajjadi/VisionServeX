#!/usr/bin/env bash
# v3.3 Phase 5 — fresh-install verification.
# Usage:
#   scripts/v33_fresh_install_verify.sh wheel  dist/visionservex-3.3.0-py3-none-any.whl
#   scripts/v33_fresh_install_verify.sh pypi    3.3.0
set -u
MODE="${1:-pypi}"
ARG="${2:-}"
VENV=/tmp/vsx_v33_truth
REPORT=notebook/99_final_report/reports/v33_fresh_install_verify.json
rm -rf "$VENV"; python3.12 -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -U pip >/dev/null 2>&1

if [ "$MODE" = "wheel" ]; then
  echo "installing local wheel: $ARG"
  "$VENV/bin/pip" install -q "$ARG" 2>&1 | tail -1
  SRC="local-wheel"
else
  echo "installing from PyPI: visionservex==$ARG (with retries for propagation)"
  for i in 1 2 3 4 5 6 7 8; do
    if "$VENV/bin/pip" install -q --no-cache-dir "visionservex==$ARG" 2>/dev/null; then echo "installed attempt $i"; break; fi
    echo "  attempt $i: not on PyPI yet, waiting 20s"; sleep 20
  done
  SRC="pypi"
fi

"$VENV/bin/python" - "$SRC" <<'PY'
import sys, json, subprocess, importlib.util as u
src = sys.argv[1]
import visionservex
res = {"source": src, "version": visionservex.__version__,
       "from_site_packages": "site-packages" in visionservex.__file__,
       "file": visionservex.__file__, "modules": {}, "cli": {}, "api_smoke": {}}
for m in ("visionservex.onnx_export","visionservex.sam2_runtime","visionservex.vsx",
          "visionservex.cli.vsx_commands","visionservex.cli.libreyolo_commands"):
    res["modules"][m] = bool(u.find_spec(m))
def cli(args):
    try:
        r = subprocess.run([sys.executable,"-m","visionservex",*args,"--help"],
                           capture_output=True,text=True,timeout=90)
        return r.returncode==0 and ("Usage" in r.stdout or "Commands" in r.stdout)
    except Exception as e:
        return f"ERR:{e}"
for g in ([], ["sam"], ["dino"], ["pipeline"], ["cv2-pro"], ["run"], ["list-models"]):
    res["cli"]["root" if not g else g[0]] = cli(g)
try:
    from visionservex import VSX
    res["api_smoke"]["sam_mobilesam_status"] = VSX.sam("mobilesam").status()
    from visionservex.onnx_export import onnx_eligible
    res["api_smoke"]["onnx_eligible"] = list(onnx_eligible())
    res["api_smoke"]["ok"] = True
except Exception as e:
    res["api_smoke"]["ok"] = False
    res["api_smoke"]["error"] = str(e)
print(json.dumps(res, indent=2))
import pathlib
pathlib.Path("notebook/99_final_report/reports/v33_fresh_install_verify.json").write_text(json.dumps(res, indent=2))
PY
echo "wrote $REPORT"
