# VisionServeX Notebook Workspace (v2.32.0)

## Structure

```
notebook/
  .venv/                          # single active Python environment (v2.32.0)
  models/checkpoints/             # model weights organized by family
  datasets/                       # COCO 400, smoke assets, domain demos
  shared/                         # config, utilities, registries
  01_object_detection/            # D-FINE, RF-DETR, LibreYOLO, Ultralytics benchmark
  02_automatic_segmentation/      # RF-DETR-Seg, YOLO-seg benchmark
  03_promptable_segmentation/     # SAM, SAM2, SAM2.1 smoke/eval
  04_open_vocab_vlm/              # OWLv2, GroundingDINO, Florence-2 demo
  05_classification/              # SwinV2, ConvNeXtV2, MaxViT smoke
  06_embedding_similarity/        # DINOv2, CLIP, SigLIP2 demo
  07_medical/                     # MedSAM smoke/demo
  08_agriculture/                 # Agriculture prompt detect/segment demo
  09_aerial_obb/                  # Aerial OBB status
  10_anomaly_industrial/          # Anomalib doctor / PatchCore
  11_surveillance_video_live/     # Tracking, video, live demo
  12_libreyolo/                   # LibreYOLO license audit + smoke
  99_final_report/                # Consolidated final report
  RUN_ALL.ipynb                   # Run all via Jupyter
  run_all.sh                      # Run all via shell
  archive_legacy/                 # Archived old notebooks/runs/envs
```

## Quick start

```bash
cd notebook

# Install kernel
.venv/bin/python -m ipykernel install --user \
  --name visionservex-notebook \
  --display-name "VisionServeX Notebook"

# Run a single task
.venv/bin/jupyter nbconvert --execute \
  01_object_detection/Object_Detection_Benchmark.ipynb \
  --ExecutePreprocessor.kernel_name=visionservex-notebook

# Run all (will take ~30 min)
bash run_all.sh
```

## Environment

Single `.venv/` with VisionServeX 2.32.0 installed from local wheel.

```
.venv/bin/pip install visionservex-2.32.0-py3-none-any.whl jupyter nbconvert ...
```

## Old notebooks

All older notebooks and run directories are in `archive_legacy/`. See
`archive_legacy/MANIFEST.md` for a full listing.
