# v2.46 Execution Sprint Plan

**Source of truth:** `reports/v246_execution_sprint_plan.csv` (56 rows).
**Baseline (v2.45.0 PyPI):** 91 healthy / 50 non-healthy.
**Target:** decrease non-healthy by at least 20 if technically/legal/auth possible.
**Strategy:** four execution lanes, executed serially under `resource_guard`.

## Status snapshot (as of v2.46.0.dev0 commit `287ba55`)

Lane A — registry/license/alias corrections that ship in this commit:
**+6 healthy / -6 non-healthy already landed in the v2.46 reconciler**, no env
builds required:

| model_id                  | v2.45 state             | v2.46 state | reason |
|---------------------------|-------------------------|-------------|--------|
| agriclip                  | not_advertised          | wired       | CC-BY 4.0 confirmed permissive |
| prithvi-eo-2.0            | opt_in_license_required | wired       | Apache-2.0 confirmed permissive |
| dinov3-vitb16             | not_advertised          | wired       | Meta license commercial-friendly < 700M MAU |
| deim-m                    | upstream_deprecated     | wired       | aliased to deimv2-m (benchmark_passed) |
| deim-s                    | upstream_deprecated     | wired       | aliased to deimv2-s (benchmark_passed) |
| oneformer-convnext-large  | wrong_registry_entry    | wired       | HF id remap to shi-labs/oneformer_ade20k_convnext_large |

Plus the static-ledger regression fix that the user separately raised: the v2.46
reconciler writes a detailed 50-column ledger and `Final_Report.ipynb` no longer
overwrites it with the legacy 11-column ledger from `archive_legacy/`.

## Lane A — quick wins (no env build required)

Already implemented. 6 of 9 rows in Lane A are now `wired`. The remaining 3:

* `deimv2-n` — checkpoint not published. Deep Research mentions the Nano
  variant should exist; the HF repo is currently 404. Stays `checkpoint_required`
  until upstream publishes weights or a mirror is provided.
* `grounding-dino-1.5` / `grounding-dino-1.6` — registry conflates Edge (local
  weights) with Pro (API-only). v2.46 plan: split the registry entry into
  `*-edge-local` (local weights via groundingdino package) and `*-pro-api`
  (DeepDataSpace REST). Implementation deferred to a follow-up registry
  refactor; current state stays `auth_required`.

## Lane B — standalone sidecars (8 candidates)

These require building a single conda env each plus a repo clone. The runtime
broker's `runtime prepare --execute` path orchestrates this. Each build is
serialized under `resource_guard`.

| model_id          | runtime_id                  | est. build | first viable test |
|-------------------|-----------------------------|------------|-------------------|
| bytetrack         | tracking_bytetrack_py310    | ~10 min    | sample_video.mp4  |
| edgesam           | promptable_edgesam_py310    | ~10 min    | coco_person_car   |
| medsam2           | medical_medsam2_py310       | ~15 min    | medical_slice     |
| osnet-x1.0        | reid_torchreid_py310        | ~10 min    | person_crops      |
| nnunet-v2         | medical_medsam2_py310       | ~10 min    | dummy NIfTI       |
| mobilesam         | promptable_edgesam_py310    | ~5 min     | coco_person_car   |
| efficientsam      | promptable_edgesam_py310    | ~5 min     | coco_person_car   |
| hq-sam            | promptable_edgesam_py310    | ~5 min     | coco_person_car   |

Realistically achievable in this session: 1–3 builds. Each can flip one model to
`smoke_passed` or `contract_passed`. Total potential: +8 healthy.

## Lane C — OpenMMLab / custom-op heavy (20 candidates)

These need genuinely heavy environment work — DCNv3 CUDA op compile, MMRotate
1.x, Detectron2 from source, NATTEN API shim, etc. The runtime broker carries
the exact commands. Expected build time: 30–60 min per island. Realistic in this
session: 0–2 islands.

| island | runtime_id                   | models                                          |
|--------|------------------------------|-------------------------------------------------|
| Co-DETR | codetr_openmmlab_py310      | co-dino-inst-vit-l-coco / -lvis                 |
| InternImage | internimage_dcnv3_py310 | internimage-t/s/b/l/h                           |
| OBB | obb_rtmdetr2_py310              | rtmdet-r-t/s/m/l, rtmdet-r2-t/m/l               |
| MaskDINO | maskdino_detectron2_py310  | maskdino-r50-coco / -panoptic, -swinl-coco      |
| SEEM | seem_xdecoder_container        | seem-focal-t, seem-davit-d3                     |
| NATTEN | oneformer_natten_py310       | oneformer-dinat-large                           |

## Lane D — terminal gated (18 rows)

These cannot become healthy without user opt-in / token / fee. v2.46's job is
to make them **user-friendly**, not falsely healthy.

| sub-lane | models | gate |
|----------|--------|------|
| API only | dino-x-api, grounding-dino-1.5-pro, grounding-dino-1.6-pro | DINO_X_API_KEY / GDINO_API_KEY |
| Auth gated | sam3-base | HF_TOKEN + Meta ToS |
| AGPL-3.0 | fastsam-s/x, yolo-world, yolo11{l,x}-seg.pt, yolo11x.pt, yolo26{,x}-seg.pt, yolov10b.pt, yolov8{,x}-seg.pt | --accept-agpl |
| PML 1.0 | rfdetr-seg-xlarge, rfdetr-seg-2xlarge | --accept-pml + ROBOFLOW_ACCOUNT_ID |
| NC weights | totalsegmentator | --accept-non-commercial |

Lane D is **already user-friendly** through the `visionservex run <model>
--accept-*` flag and broker license-gate runtime. These rows are correctly
restricted; flipping them silently would violate the user's "do not bypass
licenses" rule.

## Realistic v2.46 outcomes by lane

| lane | models | best-case improvement | this-session estimate |
|------|--------|----------------------|-----------------------|
| A    | 9      | -6 (achieved)        | -6 (locked) |
| B    | 8      | -8                   | -1 to -3 (one or two sidecars) |
| C    | 20     | -10 (env-build success) | 0 to -2 (high failure risk) |
| D    | 18     | 0 (correctly gated)  | 0 (terminal) |

Realistic this-session total: **-7 to -11 non-healthy**. Reaching the user's
`-20` target requires multiple subsequent sessions to actually build the
OpenMMLab islands; the work for those is fully specified per row in the CSV.
