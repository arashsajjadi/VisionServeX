# v31_dino_family_matrix

Total rows: 44

State distribution: legal_review_required=11, benchmark_passed=17, auth_required=1, external_api_only=5, not_released=5, wired=1, checkpoint_required=4

| model_id | generation | before | after | weights_license | next_command |
|---|---|---|---|---|---|
| dino-vits8 | DINO/DINOv2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dino-vits8  # weights/tr |
| dino-vits16 | DINO/DINOv2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dino-vits16  # weights/t |
| dino-vitb8 | DINO/DINOv2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dino-vitb8  # weights/tr |
| dino-vitb16 | DINO/DINOv2 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dino-vitb16  # weights/t |
| dinov2-small | DINO/DINOv2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run dinov2-small image.jpg --box  |
| dinov2-base | DINO/DINOv2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run dinov2-base image.jpg --box 1 |
| dinov2-large | DINO/DINOv2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run dinov2-large image.jpg --box  |
| dinov2-giant | DINO/DINOv2 | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run dinov2-giant image.jpg --box  |
| dinov3-vits16 | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-vits16  # weights |
| dinov3-vitb16 | DINOv3 | wired | auth_required | DINOv3 License (Meta custom, g | export HF_TOKEN=... && visionservex sam status din |
| dinov3-vitl16 | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-vitl16  # weights |
| dinov3-vit7b16 | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-vit7b16  # weight |
| dinov3-convnext-tiny | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-convnext-tiny  #  |
| dinov3-convnext-small | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-convnext-small  # |
| dinov3-convnext-base | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-convnext-base  #  |
| dinov3-convnext-large | DINOv3 | absent_from_ledger | legal_review_required | unknown | visionservex legal review dinov3-convnext-large  # |
| grounding-dino-swin-t | GroundingDINO | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run grounding-dino-swin-t image.j |
| grounding-dino-swin-b | GroundingDINO | benchmark_passed | benchmark_passed | Apache-2.0 | visionservex sam run grounding-dino-swin-b image.j |
| grounding-dino-original-swin-t | GroundingDINO | benchmark_passed | benchmark_passed | unknown | visionservex sam run grounding-dino-original-swin- |
| grounding-dino-original-swin-b | GroundingDINO | benchmark_passed | benchmark_passed | unknown | visionservex sam run grounding-dino-original-swin- |
| grounding-dino-1.5 | GroundingDINO | auth_required | external_api_only | API/gated | export DEEPDATASPACE_API_KEY=... && visionservex d |
| grounding-dino-1.6 | GroundingDINO | auth_required | external_api_only | API/gated | export DEEPDATASPACE_API_KEY=... && visionservex d |
| grounding-dino-1.5-pro | GroundingDINO | external_api_only | external_api_only | proprietary/closed | export DEEPDATASPACE_API_KEY=... && visionservex d |
| grounding-dino-1.6-pro | GroundingDINO | external_api_only | external_api_only | proprietary/closed | export DEEPDATASPACE_API_KEY=... && visionservex d |
| dino-x-api | DINO-X | external_api_only | external_api_only | proprietary/closed | export DEEPDATASPACE_API_KEY=... && visionservex d |
| dino-x-detection | DINO-X | absent_from_ledger | not_released | n/a (unreleased) | # dino-x-detection: no separately published checkp |
| dino-x-segmentation | DINO-X | absent_from_ledger | not_released | n/a (unreleased) | # dino-x-segmentation: no separately published che |
| dino-x-phrase-grounding | DINO-X | absent_from_ledger | not_released | n/a (unreleased) | # dino-x-phrase-grounding: no separately published |
| dino-x-counting | DINO-X | absent_from_ledger | not_released | n/a (unreleased) | # dino-x-counting: no separately published checkpo |
| dino-x-region-captioning | DINO-X | absent_from_ledger | not_released | n/a (unreleased) | # dino-x-region-captioning: no separately publishe |
| deimv2-atto | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run deimv2-atto image.jpg --box 1 |
| deimv2-n | DETR/DINO-detect | wired | wired | unknown | visionservex sam run deimv2-n image.jpg --box 10,2 |
| deimv2-s | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run deimv2-s image.jpg --box 10,2 |
| deimv2-m | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run deimv2-m image.jpg --box 10,2 |
| deimv2-l | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run deimv2-l image.jpg --box 10,2 |
| dfine-n | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run dfine-n image.jpg --box 10,20 |
| dfine-s | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run dfine-s image.jpg --box 10,20 |
| dfine-m | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run dfine-m image.jpg --box 10,20 |
| dfine-l | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run dfine-l image.jpg --box 10,20 |
| dfine-x | DETR/DINO-detect | benchmark_passed | benchmark_passed | unknown | visionservex sam run dfine-x image.jpg --box 10,20 |
| rtdetrv4-s | DETR/DINO-detect | checkpoint_required | checkpoint_required | unknown | visionservex sam run rtdetrv4-s image.jpg --box 10 |
| rtdetrv4-m | DETR/DINO-detect | checkpoint_required | checkpoint_required | unknown | visionservex sam run rtdetrv4-m image.jpg --box 10 |
| rtdetrv4-l | DETR/DINO-detect | checkpoint_required | checkpoint_required | unknown | visionservex sam run rtdetrv4-l image.jpg --box 10 |
| rtdetrv4-x | DETR/DINO-detect | checkpoint_required | checkpoint_required | unknown | visionservex sam run rtdetrv4-x image.jpg --box 10 |