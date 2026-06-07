# VisionServeX — V3 Blockers Report

**V3 NOT RELEASED.** Released: **v2.60.0** (on PyPI via Trusted Publishing). v3.0.0 is blocked by exactly **one** gate.

**16 / 17 V3 gates PASS.** Sole blocking failure: **V3-11** (durable current-run evidence attribution).

## V3 gate matrix
| gate | status | blocking | title |
|---|---|---|---|
| V3-01 | PASS | yes | PyPI Trusted Publishing works |
| V3-02 | PASS | yes | Fresh PyPI install from real PyPI |
| V3-03 | PASS | yes | RUN_ALL executes after fresh install |
| V3-04 | PASS | yes | smoke_passed == 0 |
| V3-05 | PASS | yes | benchmark_failed == 0 or justified |
| V3-06 | PASS | yes | blocker_category unclassified == 0 |
| V3-07 | PASS | yes | no AGPL/GPL/NC/restricted in commercial-safe core |
| V3-08 | PASS | yes | every core model has code_license and weights_license |
| V3-09 | PASS | yes | gated/auth models BYOT, no mirrored gated weights |
| V3-10 | PASS | yes | no token leak in reports/notebooks/git |
| V3-11 | PARTIAL | yes | every benchmark_passed row has valid current-run evidence |
| V3-12 | PASS | yes | every target classified |
| V3-13 | PASS | yes | classic smart tools separated from model leaderboard |
| V3-14 | PASS | yes | README/docs explain core vs external restricted + BYOT |
| V3-15 | PASS | yes | final_winners schema does not mix core/restricted |
| V3-16 | PASS_WITH_CAVEAT | yes | package tests pass |
| V3-17 | PASS | yes | final report lists remaining blockers + lawful next actions |

## The single v3.0.0 blocker — V3-11 (PARTIAL)

- **What:** True
- **Exact next action:** Enhance the reconciler to attribute each task's current-run leaderboard (e.g. 01_object_detection/reports/detection_leaderboard.csv under the active RUN_ID) to every model it benchmarks, so benchmark_passed rows carry a current-run evidence_artifact.
- **Why not fixed this session:** the reconciler's evidence-attribution is a core pipeline behavior; the ~13 historical_validated healthy rows (deimv2-*, rtdetrv4-*, rfdetr-seg-large, florence-2-*) need current-run re-benchmarks (GPU/gated checkpoints) or a reconciler enhancement to link each task's current-run leaderboard to its models. Doing the reconciler surgery blind risked regressions; deferred deliberately.
- **Knock-on:** the project's own test_v243 (4 tests) stays red until V3-11 closes.

## Commercial-safety — FIXED DURABLY this release (survives RUN_ALL)

- **EdgeSAM** (NTU S-Lab License 1.0, NON-COMMERCIAL) removed from commercial-safe core. Fixed at every source the pipeline reads: `manifest.py`, `extended_manifest_v240.py`, reconciler winner-exclusion, `reports_commands._RESTRICTED`, and RUN_ALL Step-6b split. After RUN_ALL: edgesam is in `external_restricted_baselines` (15 rows), NOT core.
- **final_winners commercial-safety bug FIXED:** EdgeSAM was the computed promptable CORE winner; `_compute_final_winners` is now default_safe-aware -> core promptable winner = efficientsam (Apache-2.0).
- **HQ-SAM** default_safe=False (HQSeg-44K non-commercial training data).
- **agriclip** license ambiguity resolved: check -> CC-BY-4.0 (commercial-safe).
- **V3-08 closed:** explicit code_license + weights_license for ALL 173 core models (v3_core_model_rights.csv).

## Non-benchmark core rows
| model_id | final_state | blocker_code |
|---|---|---|
| agriclip | wired |  |
| anomalib-patchcore | dataset_required | MVTEC_DATASET_MISSING |
| bytetrack | micro_benchmark_passed |  |
| co-dino-inst-vit-l-coco | sidecar_required | OPENMMLAB_REQUIRED |
| co-dino-inst-vit-l-lvis | sidecar_required | OPENMMLAB_REQUIRED |
| deim-m | wired |  |
| deim-s | wired |  |
| deimv2-n | wired |  |
| dino-x-api | external_api_only | EXTERNAL_API_REQUIRED |
| dinov3-vitb16 | wired |  |
| florence-2-base | demo_passed_sidecar |  |
| florence-2-large | demo_passed_sidecar |  |
| grounding-dino-1.5 | auth_required | DEEPDATASPACE_API_KEY_MISSING |
| grounding-dino-1.5-pro | external_api_only | EXTERNAL_API_REQUIRED |
| grounding-dino-1.6 | auth_required | DEEPDATASPACE_API_KEY_MISSING |
| grounding-dino-1.6-pro | external_api_only | EXTERNAL_API_REQUIRED |
| grounding-dino-2-audit | not_advertised | CITATION_NUMBER_HALLUCINATION |
| internimage-b | sidecar_required | OPENMMLAB_REQUIRED |
| internimage-h | sidecar_required | OPENMMLAB_REQUIRED |
| internimage-l | sidecar_required | OPENMMLAB_REQUIRED |
| internimage-s | sidecar_required | OPENMMLAB_REQUIRED |
| internimage-t | sidecar_required | OPENMMLAB_REQUIRED |
| libreyolo-dfine-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-dfine-m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-dfine-n-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-dfine-s-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-dfine-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-r101-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-r18-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-r34-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-r50-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-r50m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-rtdetr-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-n-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-s-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-t-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| libreyolo-yolox-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD |
| maskdino-r50-coco | sidecar_required | OPENMMLAB_REQUIRED |
| maskdino-r50-panoptic | sidecar_required | OPENMMLAB_REQUIRED |
| maskdino-swinl-coco | sidecar_required | SIDECAR_ENV_MISSING |
| medsam2 | sidecar_required | SIDECAR_ENV_MISSING |
| nnunet-v2 | dataset_required | MEDICAL_SEGMENTATION_DATASET_MISSING |
| oneformer-convnext-large | wired |  |
| oneformer-dinat-large | sidecar_required | NATTEN_REQUIRED |
| osnet-x1.0 | dataset_required | REID_DATASET_MISSING |
| prithvi-eo-2.0 | wired |  |
| rtmdet-r-l | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r-m | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r-s | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r-t | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r2-l | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r2-m | sidecar_required | OPENMMLAB_REQUIRED |
| rtmdet-r2-s | dataset_required | DOTA_DATASET_MISSING |
| rtmdet-r2-t | sidecar_required | OPENMMLAB_REQUIRED |
| sam3-base | auth_required | HF_SAM3_ACCESS_NOT_APPROVED |
| seem-davit-d3 | sidecar_required | SIDECAR_REQUIRED |
| seem-focal-t | sidecar_required | SIDECAR_REQUIRED |
