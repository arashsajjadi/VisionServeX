# Commercial Safety — Core vs External Restricted Baselines

VisionServeX separates models into three legally-distinct buckets. The split is
enforced in code (`reports generate-external-baselines`) and audited every release.

## 1. Commercial-safe core

`notebook/99_final_report/reports/model_coverage_ledger.csv` (default-safe).
Every row has a permissive **code** license **and** permissive **weights**
license (Apache-2.0 / MIT / BSD). These may be downloaded, cached, and used
commercially. Classic weight-free smart-annotation tools live in a separate
`smart_tool_coverage_ledger.csv` (their only legal surface is permissive
dependencies).

## 2. External restricted baselines

`external_restricted_baselines.csv` — benchmarked for comparison **only**, never
default-safe, excluded from `core_healthy`. Commercial use requires the user's
explicit opt-in / separate license:

- **AGPL-3.0**: FastSAM (-s/-x), YOLO-World, all Ultralytics `.pt`
  (YOLOv8/YOLO11/YOLO26 detect+seg, YOLOv10). Commercial closed-source use needs
  an Ultralytics Enterprise License.
- **PML-1.0 (Roboflow Platform Model License)**: RF-DETR-Seg XLarge / 2XLarge
  (registered account required). The smaller RF-DETR-Seg nano/small/medium/large
  are Apache-2.0 and **are** in core.
- **Custom non-commercial**: TotalSegmentator high-res / tissue / face subtasks.
- **NTU S-Lab License 1.0 (non-commercial)**: **EdgeSAM**. *(See note below.)*

## 3. Excluded non-commercial

`excluded_noncommercial_models.csv` — not even baselines: YOLO-NAS
(Deci-AI non-commercial weights license).

## Gated / API models (BYOT)

`sam3-base`, `grounding-dino-1.5/1.6`, `grounding-dino-1.5/1.6-pro`,
`dino-x-api` are gated (HF access request or DeepDataSpace API key). VisionServeX
**never mirrors gated weights**; the user brings their own token at runtime
(BYOT). See [gated_models.md](gated_models.md).

## V3-prep license corrections (this release)

A V3 commercial-safety audit (adversarial license verification) found two issues
in the core ledger and corrected them:

- **EdgeSAM** was incorrectly recorded as `Apache-2.0` and sat in commercial-safe
  core (`default_safe=True`, `benchmark_passed`). Its real license is the **NTU
  S-Lab License 1.0**, which restricts use to **non-commercial** purposes (source
  *and* binary). It has been moved to `external_restricted_baselines.csv` and the
  manifest corrected. This was a GATE V3-07 violation; it is now fixed.
- **HQ-SAM** (sam-hq) code/weights are declared Apache-2.0, but the **HQSeg-44K**
  fine-tuning set bundles non-commercial data (ThinObject-5K CC-BY-NC; DIS5K
  non-commercial Terms-of-Use). Whether that taints the Apache-2.0 weights is
  legally unsettled, so HQ-SAM is conservatively marked `legal_review_required`
  (`default_safe=False`) pending review. The SAM/SAM2/MobileSAM/EfficientSAM
  weights remain commercial-safe: Meta and the upstreams released those
  checkpoints under Apache-2.0 even though SA-1B itself is research-only, the same
  posture Meta uses commercially (e.g. SAM 2 on AWS SageMaker JumpStart).

The full audit is in `v3_model_rights_audit.csv` (code-vs-weights split per
target) and `v3_bad_license_scan.json`.
