#!/usr/bin/env bash
# VisionServeX v2.39.0 — clean + init ledger + run all task notebooks + reconcile.
# Usage: cd notebook && bash run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

export VISIONSERVEX_NOTEBOOK_RUN_ID="${VISIONSERVEX_NOTEBOOK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export VISIONSERVEX_NOTEBOOK_CALL_LEDGER="${VISIONSERVEX_NOTEBOOK_CALL_LEDGER:-$SCRIPT_DIR/99_final_report/reports/notebook_model_call_ledger.json}"

echo "[run_all] VISIONSERVEX_NOTEBOOK_RUN_ID=$VISIONSERVEX_NOTEBOOK_RUN_ID"

echo "[run_all] cleaning previous-run generated outputs"
visionservex notebook clean-outputs \
  --root . \
  --preserve-model-cache \
  --preserve-datasets \
  --preserve-env \
  --preserve-checkpoints \
  --format json \
  --out 99_final_report/reports/cleanup_before_run.json

echo "[run_all] initialising notebook call ledger"
visionservex notebook-call-ledger init \
  --run-id "$VISIONSERVEX_NOTEBOOK_RUN_ID" \
  --out "$VISIONSERVEX_NOTEBOOK_CALL_LEDGER"

python -m ipykernel install --user \
  --name visionservex-notebook \
  --display-name "VisionServeX Notebook" 2>/dev/null || true

KERNEL=visionservex-notebook
OPTS="--ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$KERNEL"
NB_CONVERT="jupyter nbconvert --to notebook --execute"

run_nb() {
  local nb_path="$1"
  local out_name="${nb_path%.*}_EXECUTED.ipynb"
  echo "[run_all] Executing: $nb_path"
  $NB_CONVERT "$nb_path" --output "$out_name" $OPTS
  echo "[run_all] Done: $out_name"
}

run_nb "01_object_detection/Object_Detection_Benchmark.ipynb"
run_nb "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb"
run_nb "03_promptable_segmentation/Promptable_Segmentation_Benchmark.ipynb"
run_nb "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb"
run_nb "05_classification/Classification_Smoke.ipynb"
run_nb "06_embedding_similarity/Embedding_Similarity_Demo.ipynb"
run_nb "07_medical/Medical_Demo.ipynb"
run_nb "08_agriculture/Agriculture_Demo.ipynb"
run_nb "09_aerial_obb/Aerial_OBB_Status.ipynb"
run_nb "10_anomaly_industrial/Anomaly_Industrial_Status.ipynb"
run_nb "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb"
run_nb "12_libreyolo/LibreYOLO_Audit_and_Smoke.ipynb"

echo "[run_all] reconciling model states from current-run evidence"
visionservex reports reconcile-model-states \
  --task-reports . \
  --resolution ../reports/v238_49_blocked_resolution_matrix.json \
  --notebook-call-ledger "$VISIONSERVEX_NOTEBOOK_CALL_LEDGER" \
  --out-json 99_final_report/reports/model_coverage_ledger.json \
  --out-csv 99_final_report/reports/model_coverage_ledger.csv \
  --final-winners 99_final_report/reports/final_winners.json

echo "[run_all] auditing for stale tables"
visionservex reports audit-stale-final-tables \
  --notebook-root . \
  --reports-root ../reports \
  --out 99_final_report/reports/v239_stale_final_table_audit.json

run_nb "99_final_report/Final_Report.ipynb"

echo "[run_all] done"
