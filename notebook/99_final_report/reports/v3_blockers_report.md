# VisionServeX — V3 Blockers Report

**V3 NOT RELEASED.** Selected release: **v2.59.0** (next minor after v2.58.0).

Decision derived strictly from `v3_gate_matrix.csv`: v3.0.0 ships only when every blocking gate is PASS*. Blocking failures: **V3-01, V3-02, V3-03, V3-08**.

## V3 gate matrix

| gate | status | blocking | title |
|---|---|---|---|
| V3-01 | NOT_VERIFIED | yes | PyPI Trusted Publishing works |
| V3-02 | FAIL | yes | Fresh PyPI install from real PyPI |
| V3-03 | NOT_VERIFIED | yes | RUN_ALL executes after fresh install |
| V3-04 | PASS | yes | smoke_passed == 0 |
| V3-05 | PASS | yes | benchmark_failed == 0 or justified |
| V3-06 | PASS | yes | blocker_category unclassified == 0 |
| V3-07 | PASS | yes | no AGPL/GPL/NC/restricted in commercial-safe core |
| V3-08 | PARTIAL | yes | every core model has code_license and weights_license |
| V3-09 | PASS | yes | gated/auth models BYOT, no mirrored gated weights |
| V3-10 | PASS | yes | no token leak in reports/notebooks/git |
| V3-11 | PASS_WITH_CAVEAT | yes | every benchmark_passed row has valid evidence artifact |
| V3-12 | PASS | yes | every target classified |
| V3-13 | PASS | yes | classic smart tools separated from model leaderboard |
| V3-14 | PASS | yes | README/docs explain core vs external restricted + BYOT |
| V3-15 | PASS | yes | final_winners schema does not mix core/restricted |
| V3-16 | PASS_WITH_CAVEAT | yes | package tests pass |
| V3-17 | PASS | yes | final report lists remaining blockers + lawful next actions |

## Blocking V3 gate failures — why and lawful next action

### V3-01 — PyPI Trusted Publishing works (NOT_VERIFIED)
- **Evidence:** True
- **Next action:** Configure GitHub Actions OIDC Trusted Publishing; cut a tag and watch the publish workflow.

### V3-02 — Fresh PyPI install from real PyPI (FAIL)
- **Evidence:** True
- **Next action:** Publish v2.59.0/v3 then `pip install --no-cache-dir visionservex[all-benchmark,classic-ml]==<ver>` in a fresh venv.

### V3-03 — RUN_ALL executes after fresh install (NOT_VERIFIED)
- **Evidence:** True
- **Next action:** After publish+fresh-install, run `jupyter nbconvert --execute RUN_ALL.ipynb`.

### V3-08 — every core model has code_license and weights_license (PARTIAL)
- **Evidence:** True
- **Next action:** Add explicit code_license + weights_license columns to the ledger for all core families (extend v3_model_rights_audit to detection/embedding).

## Non-benchmark core rows (state + blocker + next action)

61 of 173 core rows are not benchmark_passed. Breakdown:

| model_id | final_state | blocker_code | next action |
|---|---|---|---|
| agriclip | wired |  | visionservex models contract-run agriclip |
| anomalib-patchcore | dataset_required | MVTEC_DATASET_MISSING | visionservex dataset prepare-mvtec-mini --source /home/arash/datasets/ |
| bytetrack | micro_benchmark_passed |  | visionservex run bytetrack tests/assets/smoke/coco_person_car.jpg --ta |
| co-dino-inst-vit-l-coco | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare co-dino-inst-vit-l-coco --execute --runti |
| co-dino-inst-vit-l-lvis | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare co-dino-inst-vit-l-lvis --execute --runti |
| deim-m | wired |  | visionservex models contract-run deim-m |
| deim-s | wired |  | visionservex models contract-run deim-s |
| deimv2-n | wired |  | visionservex models contract-run deimv2-n |
| dino-x-api | external_api_only | EXTERNAL_API_REQUIRED | visionservex run dino-x-api <input> --api-key $DEEPDATASPACE_API_KEY |
| dinov3-vitb16 | wired |  | visionservex models contract-run dinov3-vitb16 |
| florence-2-base | demo_passed_sidecar |  | visionservex run florence-2-base tests/assets/smoke/coco_person_car.jp |
| florence-2-large | demo_passed_sidecar |  | visionservex run florence-2-large tests/assets/smoke/coco_person_car.j |
| grounding-dino-1.5 | auth_required | DEEPDATASPACE_API_KEY_MISSING | visionservex run grounding-dino-1.5 <input> --use-auth-if-available |
| grounding-dino-1.5-pro | external_api_only | EXTERNAL_API_REQUIRED | visionservex run grounding-dino-1.5-pro <input> --api-key $DEEPDATASPA |
| grounding-dino-1.6 | auth_required | DEEPDATASPACE_API_KEY_MISSING | visionservex run grounding-dino-1.6 <input> --use-auth-if-available |
| grounding-dino-1.6-pro | external_api_only | EXTERNAL_API_REQUIRED | visionservex run grounding-dino-1.6-pro <input> --api-key $DEEPDATASPA |
| grounding-dino-2-audit | not_advertised | CITATION_NUMBER_HALLUCINATION | visionservex models status grounding-dino-2-audit |
| hq-sam | legal_review_required | TRAINING_DATA_NONCOMMERCIAL_REVIEW | visionservex legal review hq-sam --reason HQSeg-44K-noncommercial-thin |
| internimage-b | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare internimage-b --execute --runtime interni |
| internimage-h | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare internimage-h --execute --runtime interni |
| internimage-l | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare internimage-l --execute --runtime interni |
| internimage-s | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare internimage-s --execute --runtime interni |
| internimage-t | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare internimage-t --execute --runtime interni |
| libreyolo-dfine-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-dfine-l-seg |
| libreyolo-dfine-m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-dfine-m-seg |
| libreyolo-dfine-n-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-dfine-n-seg |
| libreyolo-dfine-s-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-dfine-s-seg |
| libreyolo-dfine-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-dfine-x-seg |
| libreyolo-rtdetr-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-l-seg |
| libreyolo-rtdetr-r101-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-r101-seg |
| libreyolo-rtdetr-r18-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-r18-seg |
| libreyolo-rtdetr-r34-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-r34-seg |
| libreyolo-rtdetr-r50-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-r50-seg |
| libreyolo-rtdetr-r50m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-r50m-seg |
| libreyolo-rtdetr-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-rtdetr-x-seg |
| libreyolo-yolox-l-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-l-seg |
| libreyolo-yolox-m-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-m-seg |
| libreyolo-yolox-n-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-n-seg |
| libreyolo-yolox-s-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-s-seg |
| libreyolo-yolox-t-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-t-seg |
| libreyolo-yolox-x-seg | expected_blocker | MODEL_NOT_RUNNABLE_IN_THIS_BUILD | visionservex models status libreyolo-yolox-x-seg |
| maskdino-r50-coco | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare maskdino-r50-coco --execute --runtime mas |
| maskdino-r50-panoptic | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare maskdino-r50-panoptic --execute --runtime |
| maskdino-swinl-coco | sidecar_required | SIDECAR_ENV_MISSING | visionservex runtime prepare maskdino-swinl-coco --execute --runtime m |
| medsam2 | sidecar_required | SIDECAR_ENV_MISSING | visionservex runtime prepare medsam2 --execute --runtime medical_medsa |
| nnunet-v2 | dataset_required | MEDICAL_SEGMENTATION_DATASET_MISSING | visionservex dataset prepare-medical-seg-mini --source /home/arash/dat |
| oneformer-convnext-large | wired |  | visionservex models contract-run oneformer-convnext-large |
| oneformer-dinat-large | sidecar_required | NATTEN_REQUIRED | visionservex runtime prepare oneformer-dinat-large --execute --runtime |
| osnet-x1.0 | dataset_required | REID_DATASET_MISSING | visionservex dataset prepare-market1501-mini --source /home/arash/data |
| prithvi-eo-2.0 | wired |  | visionservex models contract-run prithvi-eo-2.0 |
| rtmdet-r-l | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r-l --execute --runtime obb_rtmdet |
| rtmdet-r-m | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r-m --execute --runtime obb_rtmdet |
| rtmdet-r-s | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r-s --execute --runtime obb_rtmdet |
| rtmdet-r-t | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r-t --execute --runtime obb_rtmdet |
| rtmdet-r2-l | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r2-l --execute --runtime obb_rtmde |
| rtmdet-r2-m | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r2-m --execute --runtime obb_rtmde |
| rtmdet-r2-s | dataset_required | DOTA_DATASET_MISSING | visionservex dataset prepare-dota-mini --source /home/arash/datasets/D |
| rtmdet-r2-t | sidecar_required | OPENMMLAB_REQUIRED | visionservex runtime prepare rtmdet-r2-t --execute --runtime obb_rtmde |
| sam3-base | auth_required | HF_SAM3_ACCESS_NOT_APPROVED | visionservex run sam3-base <input> --use-auth-if-available |
| seem-davit-d3 | sidecar_required | SIDECAR_REQUIRED | visionservex runtime prepare seem-davit-d3 --execute --runtime seem_xd |
| seem-focal-t | sidecar_required | SIDECAR_REQUIRED | visionservex runtime prepare seem-focal-t --execute --runtime seem_xde |

## Sidecar backlog strategy (21 sidecar_required)

| family | models | effort | v3 relevance | next command |
|---|---|---|---|---|
| medsam2 | 1 | low-medium (single conda env: torc | medium | `visionservex sidecar create medsam2 --execute && visionserve` |
| internimage | 5 | medium (OpenMMLab mmcv/mmdet + DCN | medium | `visionservex expert openmmlab create-env --execute && vision` |
| oneformer | 1 | medium (NATTEN wheel/compat for in | low-medium | `pip install natten -f https://shi-labs.com/natten/wheels && ` |
| rtmdet | 7 | medium-high (OpenMMLab + MMRotate  | low | `visionservex openmmlab create-env --with-mmrotate --execute ` |
| maskdino | 3 | high (Detectron2 + MaskDINO build) | low | `visionservex maskdino create-env --execute && visionservex r` |
| co-dino/codetr | 2 | high (Co-DETR OpenMMLab projects) | low | `visionservex expert codetr create-env --execute && visionser` |
| seem | 2 | high (X-Decoder/SEEM env) | low | `visionservex sidecar create seem --execute && visionservex r` |

## Commercial-safety corrections applied this session

- **EdgeSAM** — was in commercial-safe core as `Apache-2.0`/`benchmark_passed`/`default_safe=True`. Real license is **NTU S-Lab License 1.0 (non-commercial)**. Moved to `external_restricted_baselines.csv`; manifest + `_RESTRICTED` split corrected. **GATE V3-07 violation fixed.**
- **HQ-SAM** — Apache-2.0 weights but HQSeg-44K training data partly non-commercial (ThinObject-5K CC-BY-NC, DIS5K NC ToU). Marked `legal_review_required`, `default_safe=False`.
- **Evidence restore** — restored deleted v248 (dfine/rfdetr 400-img) + v256 (libreyolo-yolov9) benchmark artifacts and re-pointed 19 `benchmark_passed` rows whose evidence_artifact was NaN. **0 NaN-evidence rows remain.**
- **License backfill** — filled 21 NaN-license core rows (grounding-dino-original-swin Apache-2.0; libreyolo-seg). **0 NaN-license core rows remain.**

## Residual truthfulness caveats (recommended for v2.60+/V3)

- **V3-11 residual:** `rtdetrv4-{l,m,s,x}` and `siglip-base-patch16-224` carry `benchmark_passed` but their evidence points to a checkpoint-`pull.json` / contract-stderr, not a 400-image benchmark. Re-run or downgrade.
- **V3-16 residual (pre-existing test-debt):** the v2.46 KNOWN_CORRECTIONS state changes (`oneformer-convnext-large`, `deim-m`, `deim-s` -> `wired`) were never propagated to ~5 test files that still assert `wrong_registry_entry`/`upstream_deprecated`; plus one clean-outputs artifact test. These predate V3-prep and fail on a clean HEAD checkout. Recommend a dedicated test-debt cleanup.
- **V3-08 residual:** add explicit `code_license` + `weights_license` columns to the ledger for ALL core families (detection/embedding), extending `v3_model_rights_audit` beyond the 56 segmentation/grounded targets.