# Restricted models

Models that are **not** in the commercial-safe core. They are never
`default_safe`, never `production_allowed`, and never bundled. The live list is in
`notebook/99_final_report/reports/v38_license_policy_matrix.csv`.

## Non-commercial / research-only (`noncommercial_restricted`)

> WARNING: This model is non-commercial/restricted. Do not use it for paid SaaS,
> client work, production annotation, or commercial products unless you have
> written permission from the model owner.

Examples: `edge-sam` (NTU S-Lab 1.0), `locate-anything-3b` and
`describe-anything-3b` (NVIDIA non-commercial), `medsam2` (medical dataset terms),
`depth-anything-v2-large` (CC-BY-NC-4.0 — only the *small* V2 is Apache),
`simpleclick` (MAE CC-BY-NC backbone), `focalclick` (NVIDIA SegFormer backbone).

Run for research only:

```bash
visionservex model pull edge-sam --research-only --accept-noncommercial
```

## Enterprise / AGPL (`enterprise_license_required`)

> WARNING: This model requires an enterprise/commercial license or has
> AGPL/copyleft obligations. It is disabled in VisionServeX commercial-safe core.

Examples: `yolov8-seg`, `yolo11-seg` (Ultralytics AGPL-3.0), `fastsam-s`,
`fastsam-x` (YOLOv8-seg + ultralytics AGPL coupling). Commercial use requires an
Ultralytics Enterprise License.

## External API only (`external_api_only_terms_required`)

> External API model. Your data may leave the local environment. You must provide
> your own provider API key and comply with provider terms.

Examples: `grounding-dino-1.5`, `grounding-dino-1.5-pro`, `grounding-dino-1.6-pro`
(DeepDataSpace), the `dino-x-*` suite. No local weights; not counted as local
models.

## Legal review required (`legal_review_required`)

> License/provenance is unclear. Legal review required before commercial use.

Examples: `hq-sam`, `light-hq-sam`, `sam-hq2` (HQSeg-44K partly NC), `tinysam`,
`q-tinysam` (SA-1B subset provenance), `clickseg` (mixed permissive/NC variants),
`oneformer`, `internimage`, `medsam`, and `rfdetr-seg-xl` / `rfdetr-seg-2xl`
(seg weights reported Apache-2.0 by v3.7 research but held pending current
Roboflow terms confirmation).

## Not released / unverifiable (`not_released_or_unverifiable`)

Example: `grounding-dino-2` (no official source found at audit time).

---

Each restricted model still has a status/license command:

```bash
visionservex model license <model_id>
visionservex model status <model_id> --explain
visionservex model doctor <model_id>
```
