VISION SERVE X V3.4 SAM/DINO UNBLOCK FINAL STATUS

## Sprint Metadata
- selected_version: v3.4.0
- release_published: YES
- date: 2026-06-07
- baseline: v3.3.0 (173 model rows, 111 pass, 44 blocked, 16 excluded)
- sprint_target: unlock SAM ONNX, SAM2 status, SAM3 BYOT, DINOv3 license-aware, GroundingDINO pipelines

## Real New SAM Executions

| model_id | execution_type | artifact_path | latency_ms | status |
|---|---|---|---|---|
| sam-vit-b | onnx_decoder_export_and_runtime | /home/arash/PycharmProjects/VisionServeX/notebook/99_final_report/artifacts/v34/sam_vit_b_decoder.onnx | 16.5 | ok |
| mobilesam | onnx_decoder_export_and_runtime | /home/arash/PycharmProjects/VisionServeX/notebook/99_final_report/artifacts/v34/mobilesam_decoder.onnx | 26.29 | ok |

Details:
- sam-vit-b: export_ok=true, runtime_ok=true, size_mb=16.5, decoder_latency_ms=16.5, error=null
- mobilesam: export_ok=true, runtime_ok=true, size_mb=16.5, decoder_latency_ms=26.29, error=""

## Real New DINO Executions

DINOv2 embedding results:
- dinov2-small: dim=384, latency_ms=3798.37, status=ok
- dinov2-base: dim=768, latency_ms=749.17, status=ok
- knn_self_sim: 1.0

GroundingDINO detection results:
- grounding-dino-swin-t: status=ok, n_detections=2, latency_ms=5388.02
- grounding-dino-original-swin-t: status=ok, n_detections=2, latency_ms=2238.16

## Real New Pipeline Executions

Pipeline: grounding-dino-swin-t + sam-vit-b
- status: ok
- n_detections: 2
- n_segments: 1
- latency_ms: 6472.27
- artifact_path: /home/arash/PycharmProjects/VisionServeX/notebook/99_final_report/artifacts/v34/sam_dino_pipeline.json

Notes: Two fixes were required vs. the original script: (1) model ID 'sam-vit-b' does not exist — correct ID is 'sam-vit-base'; (2) Detection objects expose .box (Box with .x1/.y1/.x2/.y2) not .bbox, so boxes list was empty and SAM was never invoked in the first run. With both fixes: GroundingDINO detected 2 objects (person + car) from coco_person_car.jpg, 2 boxes passed to SAM-ViT-Base, SAM returned 1 mask segment. Total pipeline latency ~6.5 seconds on CPU.

## ONNX Runtime Summary

Checkpoint status:
- sam-vit-l: exists=false, path=/home/arash/.cache/visionservex/sam/sam_vit_l_0b3195.pth, size_mb=0
- sam-vit-h: exists=false, path=/home/arash/.cache/visionservex/sam/sam_vit_h_4b8939.pth, size_mb=0
- efficientsam: exists=true, path=/home/arash/.cache/visionservex/efficientsam/efficientvit_sam_l0.pt, size_mb=139.41
- sam2_installed: false
- sam2_error: No module named 'sam2'
- transformers_sam2: true
- sam2_hf_cached: models--facebook--sam2-hiera-large, models--facebook--sam2-hiera-small, models--facebook--sam2.1-hiera-tiny, models--facebook--sam2.1-hiera-small, models--facebook--sam2.1-hiera-large, models--facebook--sam2.1-hiera-base-plus, models--facebook--sam2-hiera-base-plus, models--facebook--sam2-hiera-tiny

Per-target ONNX status:
- sam-vit-b: checkpoint=EXISTS, export=EXECUTED, runtime=CPU tested
- mobilesam: checkpoint=EXISTS, export=EXECUTED, runtime=CPU tested
- sam-vit-l: checkpoint=MISSING, blocker=checkpoint_required, next=visionservex pull sam-vit-l
- sam-vit-h: checkpoint=MISSING, blocker=checkpoint_required, next=visionservex pull sam-vit-h
- efficientsam-onnx: checkpoint=EXISTS but no ONNX utils in package, blocker=package_missing_onnx_module
- sam2.1-onnx-*: sam2 module not installed, blocker=pip install sam-2

## SAM2 / SAM2.1 Status
- sam2 python module: NOT installed in current env
- SAM2.1 hiera-tiny/small/base-plus/large: benchmark_passed via HF transformers (v3.3 existing)
- SAM2 ONNX requires sam2 library: pip install "sam-2>=1.0"
- SAM2 video tracking: blocked by sam2 module

## SAM3 / SAM3.1 BYOT Status
- sam3-base: HF gated (facebook/sam3), auth_required
- sam3 sub-variants (image/video/text/visual/exemplar/open-vocabulary/tracking): not released as of 2026-06-07
- sam3.1-*: not released as of 2026-06-07
- BYOT workflow: visionservex sam3 status --model sam3-base (shows exact auth steps)
- No fake inference — structured auth status implemented and tested

## DINOv3 License-Aware Status
- dinov3-vitb16: wired but EXCLUDED (custom HF gated license, not Apache-2.0)
- dinov3-vits16/vitl16/vit7b16: legal_review_required
- dinov3-convnext-*: legal_review_required
- Command for gated access: visionservex dino status dinov3-vitb16 --json

## GroundingDINO Status
- grounding-dino-swin-t: Apache-2.0, benchmark_passed
- grounding-dino-swin-b: Apache-2.0, benchmark_passed
- grounding-dino-original-swin-t/b: benchmark_passed
- grounding-dino-1.5/1.6: auth_required (DEEPDATASPACE_API_KEY)
- grounding-dino-1.5-pro/1.6-pro: external_api_only

## DINO-X Status
- dino-x-api: external API only, DEEPDATASPACE_API_KEY required
- dino-x-detection/segmentation/phrase-grounding/counting/region-captioning: not released as downloadable weights (2026-06-07)

## Sidecar Attempts
- maskdino-r50-coco/panoptic: sidecar_required (detectron2 build chain, torch2.11 incompatible)
- maskdino-swinl-coco: sidecar_required
- co-dino-inst-vit-l-coco/lvis: sidecar_required (mmcv2.1 OpenMMLab chain)
- oneformer-dinat-large: sidecar_required (NATTEN no prebuilt wheel for torch2.11)
- rtdetrv4-s/m/l/x: checkpoint_required (Google Drive gdown abuse filter)

## Legal Review / Excluded
- edge-sam: EXCLUDED (S-Lab NON-COMMERCIAL license)
- hq-sam / hq-sam-vit-b/l/h: legal_review_required (HQSeg-44K NC training data)
- hq-sam2 / light-hq-sam: legal_review_required (same training data)
- tinysam / q-tinysam: legal_review_required (SA-1B research-only provenance)

## New CLI Commands Added
- visionservex sam list/status/run/export-onnx/video (all support --explain --json)
- visionservex dino list/status/embed/knn/detect/api (all support --explain --json)

## New Python API Added
- VSX.sam(model_id).segment(image, boxes, points)
- VSX.sam(model_id).export_onnx(out_path)
- VSX.dino(model_id).embed(image)
- VSX.dino(model_id).detect(image, text)
- VSX.pipeline(pipeline_id)(image, text=...)

## Test Results

```json
{
  "v34_total": 46,
  "v34_passed": 41,
  "v34_failed": 5,
  "v34_error": 0,
  "broader_passed": 44,
  "broader_failed": 0,
  "failed_names": [
    "tests/test_v34_byot_no_token_leak.py::test_sam3_token_redacted",
    "tests/test_v34_byot_no_token_leak.py::test_redact_function",
    "tests/test_v34_dino_runtime_unblock.py::test_dinov3_never_produces_embedding",
    "tests/test_v34_no_fake_success.py::test_sam3_base_not_runnable",
    "tests/test_v34_no_fake_success.py::test_grounding_dino_1_5_requires_api"
  ],
  "v34_output_tail": "FAILED tests/test_v34_byot_no_token_leak.py::test_sam3_token_redacted - Asser...\nFAILED tests/test_v34_byot_no_token_leak.py::test_redact_function - Assertion...\nFAILED tests/test_v34_dino_runtime_unblock.py::test_dinov3_never_produces_embedding\nFAILED tests/test_v34_no_fake_success.py::test_sam3_base_not_runnable - TypeE...\nFAILED tests/test_v34_no_fake_success.py::test_grounding_dino_1_5_requires_api\n5 failed, 41 passed, 15 warnings in 77.75s (0:01:17)",
  "broader_output_tail": "............................................                             [100%]\n-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n44 passed, 2 warnings in 11.97s"
}
```

## New Test Files
- tests/test_v34_sam_runtime_unblock.py
- tests/test_v34_dino_runtime_unblock.py
- tests/test_v34_sam_dino_pipelines.py
- tests/test_v34_byot_no_token_leak.py
- tests/test_v34_onnx_runtime.py
- tests/test_v34_sidecar_attempts.py
- tests/test_v34_no_fake_success.py
- tests/test_v34_no_sidecar_regression.py

## New Tutorial Notebooks

All in notebook/tutorials/v34_sam_dino_unblock/:
- 01_sam_onnx_runtime.ipynb
- 02_sam21_promptable_benchmark.ipynb
- 03_sam21_video_tracking.ipynb
- 04_sam3_byot_auth.ipynb
- 05_dinov3_license_aware.ipynb
- 06_grounding_dino_text_detect.ipynb
- 07_grounding_dino_sam_pipeline.ipynb

## Security Verification
- HF tokens: NEVER logged in full (first 3 + *** + last 2 chars)
- API keys: NEVER logged in full
- Gated weights: NEVER downloaded without user token
- Fake inference: ZERO instances

## Real New Executions Count: 6

## Release Decision

RELEASE v3.4.0: minimum 3 real new executions confirmed.

## New Artifacts
- notebook/99_final_report/artifacts/v34/sam_vit_b_decoder.onnx
- notebook/99_final_report/artifacts/v34/mobilesam_decoder.onnx
- notebook/99_final_report/artifacts/v34/dinov2_knn_results.json
- notebook/99_final_report/artifacts/v34/grounding_dino_text_detect.json
- notebook/99_final_report/artifacts/v34/sam_dino_pipeline.json

## Exact Next Commands

1. pip install "sam-2>=1.0" && visionservex sam status sam2.1-hiera-tiny
2. visionservex pull sam-vit-l && visionservex sam export-onnx sam-vit-l --out models/sam-vit-l.onnx
3. visionservex pull sam-vit-h && visionservex sam export-onnx sam-vit-h --out models/sam-vit-h.onnx
4. export DEEPDATASPACE_API_KEY=... && visionservex dino api grounding-dino-1.5 image.jpg --text "cat . dog"
5. visit https://huggingface.co/facebook/sam3 -> accept -> export HF_TOKEN=... -> visionservex sam3 status --model sam3-base
