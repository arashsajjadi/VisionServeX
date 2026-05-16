# License Risk Table (v2.6.0)

VisionServeX cares about license compatibility before runtime. Everything in
the default core is permissive (Apache-2.0 or MIT). GPL/AGPL-licensed code
is never routed through the permissive core. Proprietary or non-commercial
weights are only reachable through explicit opt-in commands.

This table is regenerated from the research notes that drove v2.6.0
(date checked: 2026-05-16). It is the authoritative source whenever the
README, model-zoo matrix, or domain docs differ.

## Core-safe (default extras)

| Library / model        | License     | Route                          | Command |
|------------------------|-------------|--------------------------------|---------|
| Anomalib code          | Apache-2.0  | `[anomaly]` optional extra     | `pip install 'visionservex[anomaly]'` |
| ByteTrack (bytetracker)| MIT         | optional extra                 | `pip install bytetracker` |
| OC-SORT (ocsort)       | MIT         | optional extra                 | `pip install ocsort` |
| Torchreid / OSNet      | MIT         | optional extra                 | `pip install torchreid` |
| MedSAM (HF weights)    | Apache-2.0  | HF auto-pull                   | `visionservex model pull medsam` |
| SAM 2.1 (hiera-*)      | Apache-2.0  | HF auto-pull                   | `visionservex model pull sam2.1-hiera-tiny` |
| DINOv2 / SigLIP / OWL  | Apache-2.0  | HF auto-pull                   | `visionservex model pull dinov2-base` |
| OpenMMLab RTMDet/Pose  | Apache-2.0  | OpenMMLab isolated env         | `visionservex openmmlab create-env` |

## Optional extras (verified license, requires opt-in)

| Library / model              | License   | Route               | Notes |
|------------------------------|-----------|---------------------|-------|
| FastReID                     | Apache-2.0| expert sidecar      | Old environment; cannot ride core. |
| TotalSegmentator core total  | Apache-2.0| `[medical]` extra   | Re-verify before public install. |
| nnU-Net v2                   | Apache-2.0| expert sidecar      | No guaranteed pretrained weights. |
| MONAI / Auto3DSeg            | Apache-2.0| `[medical]` extra   | Bundles have individual licenses. |
| MaskDINO + Detectron2        | Apache-2.0| sidecar             | Custom CUDA ops. |
| Co-DINO / Co-DETR            | Apache-2.0| OpenMMLab sidecar   | mmdet required. |

## Non-core / excluded by default

| Library / model              | License        | Reason |
|------------------------------|----------------|--------|
| DeepSORT                     | GPL-3.0        | Excluded from permissive core. |
| StrongSORT                   | GPL (signal)   | Out of core until license re-verified. |
| FastSAM (-s / -x)            | AGPL-3.0       | Excluded from permissive core. |
| TotalSegmentator tissue stats| Proprietary    | Requires commercial license key. |
| RF-DETR Plus / XL / 2XL      | PML 1.0        | `non_core_license_optional`, manual install. |
| MVTec AD dataset             | CC BY-NC-SA 4.0| Never bundled in benchmark data. |

## Gated / external API

| Library / model         | License        | Reason |
|-------------------------|----------------|--------|
| SAM 3 / SAM 3.1         | Apache-2.0 code, gated weights | `visionservex sam-family login-help sam3.1` |
| Grounding DINO 1.5/1.6  | API-only       | IDEA-Research token required.  |

## Unavailable (no official source)

| Library / model     | Why |
|---------------------|-----|
| MGN (official)      | No verified implementation or checkpoint found at research time. |
| MedSAM2 (verified)  | Checkpoint packaging unverified — expert sidecar only. |

## Tests covering this table

- `tests/test_v260.py::test_license_risk_table_present`
- `tests/test_v260.py::test_non_core_models_have_runtime_status`
- `tests/test_v260.py::test_deepsort_not_in_permissive_core`
- `tests/test_v260.py::test_fastsam_not_in_permissive_core`

If a future contributor changes runtime status, this table must be updated
in the same PR. Any addition with `license_risk in {"agpl","gpl"}` and
`recommended_action="add_now"` is a release blocker.
