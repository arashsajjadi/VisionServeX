#!/usr/bin/env bash
# VisionServeX v2.32.0 — run all task notebooks in order.
# Usage: cd notebook && bash run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

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
run_nb "99_final_report/Final_Report.ipynb"

echo ""
echo "All notebooks executed successfully."
echo "Run: grep -r 'NOT_WIRED\\|v20:\\|UNAVAILABLE_OR_FAILED' . && exit 1 || echo 'Scan clean'"
