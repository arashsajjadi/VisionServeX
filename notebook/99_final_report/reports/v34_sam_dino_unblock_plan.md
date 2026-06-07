# VisionServeX v3.4 SAM / DINO Unblock Plan

Generated: 2026-06-07
Total models tracked: 85
Source ledgers: v34_sam_execution_ledger.csv, v34_dino_execution_ledger.csv, v34_onnx_runtime_ledger.csv, v34_sidecar_attempt_ledger.csv, v34_failed_target_blockers.csv

---

## Summary

| Priority | Count | Families | Primary Blocker |
|----------|-------|----------|----------------|
| P0 | 9 | SAM1-ONNX, SAM2.1-ONNX, MobileSAM-ONNX, EfficientSAM-ONNX | checkpoint_required / sam2_not_installed / no_onnx_utils |
| P1 | 10 | SAM2-hiera, SAM2.1-hiera, SAM2.1-video | sam2_not_installed (P1a); already benchmark_passed (P1b) |
| P2 | 16 | SAM3, SAM3.1 | not_released / auth_required (HF gated) |
| P3 | 12 | DINOv2, DINOv3 | already benchmark_passed (DINOv2); auth_required / legal_review (DINOv3) |
| P4 | 8 | GroundingDINO open + 1.5/1.6/pro | already benchmark_passed (open); auth_required / external_api_only (closed) |
| P5 | 6 | DINO-X | external_api_only / not_released |
| P6 | 15 | MaskDINO, Co-DINO, OneFormer, DEIMv2, RTDETRv4 | sidecar_required / checkpoint_required |
| P7 | 9 | HQ-SAM family, TinySAM, Q-TinySAM, EdgeSAM | legal_review_required / excluded_restricted |

---

## P0 — SAM ONNX (9 models)

Highest-urgency ONNX unblock work targeting direct CPU-inference deployments.

| model_id | result_state | blocker | exact_command |
|----------|-------------|---------|---------------|
| sam-vit-b-onnx | onnx_exported | none | `visionservex sam export-onnx sam-vit-b --out models/sam-vit-b.onnx` |
| sam-vit-l-onnx | checkpoint_required | sam_vit_l_0b3195.pth not cached | `visionservex pull sam-vit-l && visionservex sam export-onnx sam-vit-l --out models/sam-vit-l.onnx` |
| sam-vit-h-onnx | checkpoint_required | sam_vit_h_4b8939.pth not cached | `visionservex pull sam-vit-h && visionservex sam export-onnx sam-vit-h --out models/sam-vit-h.onnx` |
| sam2.1-onnx-tiny | sam2_not_installed | sam2 python module missing | `pip install "sam-2>=1.0" && sam2 export-onnx sam2.1-tiny` |
| sam2.1-onnx-small | sam2_not_installed | sam2 python module missing | `pip install "sam-2>=1.0" && sam2 export-onnx sam2.1-small` |
| sam2.1-onnx-base-plus | sam2_not_installed | sam2 python module missing | `pip install "sam-2>=1.0" && sam2 export-onnx sam2.1-base-plus` |
| sam2.1-onnx-large | sam2_not_installed | sam2 python module missing | `pip install "sam-2>=1.0" && sam2 export-onnx sam2.1-large` |
| mobilesam-onnx | onnx_exported | none | `visionservex sam export-onnx mobilesam --out models/mobilesam.onnx` |
| efficientsam-onnx | no_onnx_utils_in_efficientsam_package | package lacks SamOnnxModel | `pip install efficientsam[onnx] || check efficientvit/efficientsam for SamOnnxModel` |

**Evidence**: sam-vit-b decoder ONNX confirmed at `artifacts/v34/sam_vit_b_decoder.onnx` (latency 17.74 ms, IoU 0.898, mask shape [1,1,1500,2250]). MobileSAM decoder at `artifacts/v34/mobilesam_decoder.onnx` (latency 19.37 ms, IoU 0.542).

---

## P1 — SAM2 / SAM2.1 hiera + video (10 models)

| model_id | result_state | blocker |
|----------|-------------|---------|
| sam2-hiera-tiny | sam2_not_installed | `pip install "sam-2>=1.0"` |
| sam2-hiera-small | sam2_not_installed | `pip install "sam-2>=1.0"` |
| sam2-hiera-base-plus | sam2_not_installed | `pip install "sam-2>=1.0"` |
| sam2-hiera-large | sam2_not_installed | `pip install "sam-2>=1.0"` |
| sam2.1-hiera-tiny | benchmark_passed | none (v3.3 pass via HF transformers) |
| sam2.1-hiera-small | benchmark_passed | none (v3.3 pass via HF transformers) |
| sam2.1-hiera-base-plus | benchmark_passed | none (v3.3 pass via HF transformers) |
| sam2.1-hiera-large | benchmark_passed | none (v3.3 pass via HF transformers) |
| sam2.1-video-tiny | sam2_not_installed | `pip install "sam-2>=1.0"` |
| sam2.1-video-small | sam2_not_installed | `pip install "sam-2>=1.0"` |

**Root cause**: The `sam2` Python module (Meta's `sam-2` PyPI package) is not installed in the current environment. SAM2 hiera models and SAM2.1 ONNX/video variants all depend on it. SAM2.1 hiera models already pass via HF Transformers.

---

## P2 — SAM3 / SAM3.1 (16 models)

All SAM3 and SAM3.1 variants are either gated on HuggingFace (sam3-base) or not yet publicly released as of 2026-06-07.

| model_id | result_state | blocker |
|----------|-------------|---------|
| sam3-base | auth_required | HF gated — visit huggingface.co/facebook/sam3, accept terms, set HF_TOKEN |
| sam3-image | not_released | No separate checkpoint published |
| sam3-video | not_released | No separate checkpoint published |
| sam3-text-prompt | not_released | No separate checkpoint published |
| sam3-visual-prompt | not_released | No separate checkpoint published |
| sam3-exemplar-prompt | not_released | No separate checkpoint published |
| sam3-open-vocabulary | not_released | No separate checkpoint published |
| sam3-tracking | not_released | No separate checkpoint published |
| sam3.1-base | not_released | SAM 3.1 not publicly released |
| sam3.1-image | not_released | SAM 3.1 not publicly released |
| sam3.1-video | not_released | SAM 3.1 not publicly released |
| sam3.1-real-time-tracking | not_released | SAM 3.1 not publicly released |
| sam3.1-text-prompt | not_released | SAM 3.1 not publicly released |
| sam3.1-visual-prompt | not_released | SAM 3.1 not publicly released |
| sam3.1-exemplar-prompt | not_released | SAM 3.1 not publicly released |
| sam3.1-open-vocabulary | not_released | SAM 3.1 not publicly released |

**Action**: Monitor `github.com/facebookresearch/sam3` for releases. For sam3-base, once HF access is approved: `export HF_TOKEN=hf_... && visionservex sam3 status --model sam3-base`.

---

## P3 — DINOv2 / DINOv3 (12 models)

| model_id | result_state | blocker |
|----------|-------------|---------|
| dinov2-vits14 | benchmark_passed | none (v3.3) |
| dinov2-vitb14 | benchmark_passed | none — kNN self-similarity=1.0 confirmed |
| dinov2-vitl14 | benchmark_passed | none (v3.3) |
| dinov2-vitg14 | benchmark_passed | none (v3.3) |
| dinov3-vits16 | auth_required | HF gated custom license |
| dinov3-vitb16 | auth_required | HF gated custom license |
| dinov3-vitl16 | legal_review_required | Custom license provenance not Apache-2.0 |
| dinov3-vit7b16 | legal_review_required | Custom license provenance not Apache-2.0 |
| dinov3-convnext-tiny | legal_review_required | Custom license |
| dinov3-convnext-small | legal_review_required | Custom license |
| dinov3-convnext-base | legal_review_required | Custom license |
| dinov3-convnext-large | legal_review_required | Custom license |

**Evidence**: DINOv2-vitb14 embedding confirmed at `artifacts/v34/dinov2_dinov2_base_embed.npy` (latency 749 ms, embed_dim=768). DINOv3 requires legal sign-off before integration.

---

## P4 — GroundingDINO (8 models)

| model_id | result_state | blocker |
|----------|-------------|---------|
| grounding-dino-swin-t | benchmark_passed | none — 2 detections on test image, latency 5388 ms |
| grounding-dino-swin-b | benchmark_passed | none (v3.3) |
| grounding-dino-original-swin-t | benchmark_passed | none — 2 detections, latency 2238 ms |
| grounding-dino-original-swin-b | benchmark_passed | none (v3.3) |
| grounding-dino-1.5 | auth_required | DEEPDATASPACE_API_KEY not set |
| grounding-dino-1.6 | auth_required | DEEPDATASPACE_API_KEY not set |
| grounding-dino-1.5-pro | external_api_only | Proprietary weights / API-only |
| grounding-dino-1.6-pro | external_api_only | Proprietary weights / API-only |

**Evidence**: Detection results at `artifacts/v34/grounding_dino_text_detect.json`. Open variants fully unblocked. 1.5/1.6 require DeepDataSpace API key; 1.5-pro/1.6-pro are proprietary external API only.

---

## P5 — DINO-X (6 models)

All DINO-X sub-task weights are not released as standalone downloadable checkpoints as of 2026-06-07. The API variant requires a DeepDataSpace key.

| model_id | result_state |
|----------|-------------|
| dino-x-api | external_api_only |
| dino-x-detection | not_released |
| dino-x-segmentation | not_released |
| dino-x-phrase-grounding | not_released |
| dino-x-counting | not_released |
| dino-x-region-captioning | not_released |

**Action**: Monitor `github.com/IDEACVR/DINO-X` for weight releases. API access: `export DEEPDATASPACE_API_KEY=... && visionservex dino api dino-x-api image.jpg --text cat`.

---

## P6 — MaskDINO / Co-DINO / OneFormer / DEIMv2 / RTDETRv4 (15 models)

These models require either complex sidecar conda environments or checkpoint downloads that are blocked by external infrastructure issues.

| model_id | result_state | blocker | isolation_command |
|----------|-------------|---------|-------------------|
| maskdino-r50-coco | sidecar_required | detectron2 build chain incompatible with torch2.11+ | `conda create -n maskdino python=3.10` + detectron2 torch2.1 wheel |
| maskdino-r50-panoptic | sidecar_required | same as above | same sidecar |
| maskdino-swinl-coco | sidecar_required | same + SwinL checkpoint | same sidecar |
| co-dino-inst-vit-l-coco | sidecar_required | OpenMMLab Co-DETR mmcv2.1 chain | `conda create -n codino python=3.10` + mmcv2.1 |
| co-dino-inst-vit-l-lvis | sidecar_required | same + LVIS checkpoint | same sidecar |
| oneformer-dinat-large | sidecar_required | NATTEN no prebuilt wheel for torch2.11+ | `conda create -n oneformer python=3.10` + natten==0.17.1 |
| deimv2-atto | checkpoint_required | checkpoint not cached | `visionservex pull deimv2-atto` |
| deimv2-n | checkpoint_required | checkpoint not cached | `visionservex pull deimv2-n` |
| deimv2-s | checkpoint_required | checkpoint not cached | `visionservex pull deimv2-s` |
| deimv2-m | checkpoint_required | checkpoint not cached | `visionservex pull deimv2-m` |
| deimv2-l | checkpoint_required | checkpoint not cached | `visionservex pull deimv2-l` |
| rtdetrv4-s | checkpoint_required | Google Drive gdown abuse filter | `gdown <drive-id> -O rtdetrv4_s.pth` |
| rtdetrv4-m | checkpoint_required | Google Drive gdown abuse filter | `gdown <drive-id> -O rtdetrv4_m.pth` |
| rtdetrv4-l | checkpoint_required | Google Drive gdown abuse filter | `gdown <drive-id> -O rtdetrv4_l.pth` |
| rtdetrv4-x | checkpoint_required | Google Drive gdown abuse filter | `gdown <drive-id> -O rtdetrv4_x.pth` |

---

## P7 — HQ-SAM family / TinySAM / EdgeSAM (9 models)

| model_id | result_state | blocker |
|----------|-------------|---------|
| hq-sam | legal_review_required | HQSeg-44K training data has non-commercial clause |
| hq-sam-vit-b | legal_review_required | HQSeg-44K non-commercial training data |
| hq-sam-vit-l | legal_review_required | HQSeg-44K non-commercial training data |
| hq-sam-vit-h | legal_review_required | HQSeg-44K non-commercial training data |
| hq-sam2 | legal_review_required | HQSeg-44K non-commercial training data |
| light-hq-sam | legal_review_required | HQSeg-44K non-commercial training data |
| tinysam | legal_review_required | SA-1B research-only distillation provenance |
| q-tinysam | legal_review_required | SA-1B research-only distillation provenance |
| edge-sam | excluded_restricted | S-Lab License 1.0 NON-COMMERCIAL — permanently excluded from distribution |

**Action for HQ-SAM family**: Legal review of HQSeg-44K license terms required before any distribution. Run `visionservex legal review <model_id>` to initiate review workflow. EdgeSAM is permanently excluded.

---

## Blocker Category Summary

| Blocker | Count | Notes |
|---------|-------|-------|
| not_released | 15 | SAM3 (7 sub-tasks) + SAM3.1 (8 sub-tasks) |
| sam2_not_installed | 10 | sam2-hiera x4 + sam2.1-onnx x4 + sam2.1-video x2 |
| legal_review_required | 9 | HQ-SAM x6 + TinySAM + Q-TinySAM + DINOv3 large variants |
| sidecar_required | 6 | MaskDINO x3 + Co-DINO x2 + OneFormer x1 |
| checkpoint_required | 9 | sam-vit-l/h + DEIMv2 x5 + RTDETRv4 x4 (minus 2 counted above) |
| auth_required | 5 | SAM3-base + DINOv3-vits16/vitb16 + GroundingDINO-1.5/1.6 |
| external_api_only | 4 | DINO-X-api + GroundingDINO-1.5-pro/1.6-pro + dino-x-api |
| no_onnx_utils | 1 | efficientsam-onnx |
| excluded_restricted | 1 | edge-sam |
| benchmark_passed | 12 | sam2.1-hiera x4 + dinov2 x4 + grounding-dino open x4 |

---

## Next Steps

1. **Immediate (P0)**: Pull sam-vit-l and sam-vit-h checkpoints. Install `sam-2>=1.0` to unblock all sam2/sam2.1 ONNX and video targets. File upstream issue for efficientsam ONNX utilities.
2. **Short-term (P1)**: After installing sam-2, run full benchmark pass for sam2-hiera-{tiny,small,base-plus,large} and sam2.1-video-{tiny,small}.
3. **Medium-term (P3/P4)**: Obtain DEEPDATASPACE_API_KEY and HF token with DINOv3 gated access for auth-blocked models.
4. **Legal track (P3/P7)**: Submit DINOv3 and HQ-SAM family for legal review of license compatibility before any integration or distribution.
5. **Sidecar track (P6)**: Build isolated conda environments for MaskDINO, Co-DINO, OneFormer as documented. For RTDETRv4, obtain checkpoints directly from authors if gdown remains blocked. For DEIMv2, run `visionservex pull deimv2-{atto,n,s,m,l}`.
6. **Monitor (P2/P5)**: Watch SAM3/SAM3.1 and DINO-X release channels; integrate immediately upon public release.
