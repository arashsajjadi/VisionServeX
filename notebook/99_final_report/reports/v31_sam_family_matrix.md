# v31_sam_family_matrix

Total rows: 68

State distribution: benchmark_passed=25, checkpoint_required=5, legal_review_required=20, auth_required=1, not_released=15, sidecar_required=1, excluded_restricted=1

| model_id | generation | before | after | weights_license | next_command |
|---|---|---|---|---|---|
| sam-vit-b | SAM1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam-vit-b image.jpg --box 10, |
| sam-vit-l | SAM1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam-vit-l image.jpg --box 10, |
| sam-vit-h | SAM1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam-vit-h image.jpg --box 10, |
| sam-vit-b-onnx | SAM1 | absent_from_ledger | checkpoint_required | Apache-2.0 (local ONNX export) | visionservex sam export-onnx sam-vit-b --out model |
| sam-vit-l-onnx | SAM1 | absent_from_ledger | checkpoint_required | Apache-2.0 (local ONNX export) | visionservex sam export-onnx sam-vit-l --out model |
| sam-vit-h-onnx | SAM1 | absent_from_ledger | checkpoint_required | Apache-2.0 (local ONNX export) | visionservex sam export-onnx sam-vit-h --out model |
| hq-sam | SAM1-HQ/medical | benchmark_passed | legal_review_required | Apache-2.0 (declared); HQSeg-4 | visionservex legal review hq-sam  # weights/traini |
| hq-sam-vit-b | SAM1-HQ/medical | benchmark_passed | legal_review_required | HQSeg-44K NC training data | visionservex legal review hq-sam-vit-b  # weights/ |
| hq-sam-vit-l | SAM1-HQ/medical | benchmark_passed | legal_review_required | HQSeg-44K NC training data | visionservex legal review hq-sam-vit-l  # weights/ |
| hq-sam-vit-h | SAM1-HQ/medical | benchmark_passed | legal_review_required | HQSeg-44K NC training data | visionservex legal review hq-sam-vit-h  # weights/ |
| medsam | SAM1-HQ/medical | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run medsam image.jpg --box 10,20, |
| medsam-vit-b | SAM1-HQ/medical | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run medsam-vit-b image.jpg --box  |
| sam2-hiera-tiny | SAM2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2-hiera-tiny image.jpg --b |
| sam2-hiera-small | SAM2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2-hiera-small image.jpg -- |
| sam2-hiera-base-plus | SAM2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2-hiera-base-plus image.jp |
| sam2-hiera-large | SAM2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2-hiera-large image.jpg -- |
| sam2-image-tiny | SAM2 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2-image-tiny image.jpg --b |
| sam2-image-small | SAM2 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2-image-small image.jpg -- |
| sam2-image-base-plus | SAM2 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2-image-base-plus image.jp |
| sam2-image-large | SAM2 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2-image-large image.jpg -- |
| sam2-video-tiny | SAM2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2-video-tiny  # weigh |
| sam2-video-small | SAM2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2-video-small  # weig |
| sam2-video-base-plus | SAM2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2-video-base-plus  #  |
| sam2-video-large | SAM2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2-video-large  # weig |
| sam2.1-hiera-tiny | SAM2.1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2.1-hiera-tiny image.jpg - |
| sam2.1-hiera-small | SAM2.1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2.1-hiera-small image.jpg  |
| sam2.1-hiera-base-plus | SAM2.1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2.1-hiera-base-plus image. |
| sam2.1-hiera-large | SAM2.1 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run sam2.1-hiera-large image.jpg  |
| sam2.1-image-tiny | SAM2.1 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2.1-image-tiny image.jpg - |
| sam2.1-image-small | SAM2.1 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2.1-image-small image.jpg  |
| sam2.1-image-base-plus | SAM2.1 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2.1-image-base-plus image. |
| sam2.1-image-large | SAM2.1 | benchmark_passed | benchmark_passed | unknown | visionservex sam run sam2.1-image-large image.jpg  |
| sam2.1-video-tiny | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-video-tiny  # wei |
| sam2.1-video-small | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-video-small  # we |
| sam2.1-video-base-plus | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-video-base-plus   |
| sam2.1-video-large | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-video-large  # we |
| sam2.1-onnx-tiny | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-onnx-tiny  # weig |
| sam2.1-onnx-small | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-onnx-small  # wei |
| sam2.1-onnx-base-plus | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-onnx-base-plus  # |
| sam2.1-onnx-large | SAM2.1 | absent_from_ledger | legal_review_required | unknown | visionservex legal review sam2.1-onnx-large  # wei |
| sam3-base | SAM3 | auth_required | auth_required | SAM License (Meta custom, gate | export HF_TOKEN=... && visionservex sam status sam |
| sam3-image | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-image: no separately published checkpoint a |
| sam3-video | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-video: no separately published checkpoint a |
| sam3-text-prompt | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-text-prompt: no separately published checkp |
| sam3-visual-prompt | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-visual-prompt: no separately published chec |
| sam3-exemplar-prompt | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-exemplar-prompt: no separately published ch |
| sam3-open-vocabulary | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-open-vocabulary: no separately published ch |
| sam3-tracking | SAM3 | absent_from_ledger | not_released | n/a (unreleased) | # sam3-tracking: no separately published checkpoin |
| sam3.1-base | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-base: no separately published checkpoint  |
| sam3.1-image | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-image: no separately published checkpoint |
| sam3.1-video | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-video: no separately published checkpoint |
| sam3.1-real-time-tracking | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-real-time-tracking: no separately publish |
| sam3.1-text-prompt | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-text-prompt: no separately published chec |
| sam3.1-visual-prompt | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-visual-prompt: no separately published ch |
| sam3.1-open-vocabulary | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-open-vocabulary: no separately published  |
| sam3.1-api-or-byot | SAM3.1 | absent_from_ledger | not_released | n/a (unreleased) | # sam3.1-api-or-byot: no separately published chec |
| mobilesam | lightweight | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run mobilesam image.jpg --box 10, |
| mobilesam-onnx | lightweight | absent_from_ledger | checkpoint_required | Apache-2.0 (local ONNX export) | visionservex sam export-onnx mobilesam --out model |
| efficientsam | lightweight | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run efficientsam image.jpg --box  |
| efficientsam-tiny | lightweight | benchmark_passed | benchmark_passed | unknown | visionservex sam run efficientsam-tiny image.jpg - |
| efficientsam-small | lightweight | benchmark_passed | benchmark_passed | unknown | visionservex sam run efficientsam-small image.jpg  |
| efficientsam-onnx | lightweight | absent_from_ledger | checkpoint_required | Apache-2.0 (local ONNX export) | visionservex sam export-onnx efficientsam --out mo |
| tinysam | lightweight | absent_from_ledger | legal_review_required | Apache-2.0 tag; SA-1B research | visionservex legal review tinysam  # weights/train |
| q-tinysam | lightweight | absent_from_ledger | legal_review_required | Apache-2.0 tag; SA-1B research | visionservex legal review q-tinysam  # weights/tra |
| light-hq-sam | lightweight | absent_from_ledger | legal_review_required | HQSeg-44K NC training data | visionservex legal review light-hq-sam  # weights/ |
| hq-sam2 | lightweight | absent_from_ledger | legal_review_required | HQSeg-44K NC training data | visionservex legal review hq-sam2  # weights/train |
| medsam2 | lightweight | sidecar_required | sidecar_required | Apache-2.0 | visionservex sam run medsam2 image.jpg --box 10,20 |
| edge-sam | lightweight | absent_from_ledger | excluded_restricted | S-Lab License 1.0 (NON-COMMERC | # edge-sam is non-commercial — external baseline o |