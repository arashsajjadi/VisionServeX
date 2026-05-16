# VisionServeX Model Matrix

| Model ID | Family | Task | Status | License | Install | Source | Blockers |
|----------|--------|------|--------|---------|---------|--------|---------|
| `dfine-s-o365-coco` | dfine | detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/Peterande/D-FINE | - |
| `dfine-x-o365-coco` | dfine | detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/Peterande/D-FINE | - |
| `dfine-l-o365-coco` | dfine | detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/Peterande/D-FINE | - |
| `rfdetr-small` | rfdetr | detect | runnable | Apache-2.0 | `pip install 'visionservex[rfdetr]'` | https://github.com/roboflow/rf-detr | - |
| `rfdetr-large` | rfdetr | detect | runnable | Apache-2.0 | `pip install 'visionservex[rfdetr]'` | https://github.com/roboflow/rf-detr | - |
| `rfdetr-seg-medium` | rfdetr | segment | runnable | Apache-2.0 | `pip install 'visionservex[rfdetr]'` | https://github.com/roboflow/rf-detr | - |
| `rfdetr-seg-large` | rfdetr | segment | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/roboflow/rf-detr | HF checkpoint roboflow/rf-detr-seg-large not yet published at last audit. |
| `deimv2-s` | deimv2 | detect | audit_only | Apache-2.0 | `git clone https://github.com/Intellindust-AI-Lab/DEIMv2 (manual)` | https://github.com/Intellindust-AI-Lab/DEIMv2 | Not yet supported by HF Transformers (issue #41211). |
| `deimv2-m` | deimv2 | detect | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/Intellindust-AI-Lab/DEIMv2 | Same as deimv2-s — repo native loader required. |
| `deimv2-l` | deimv2 | detect | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/Intellindust-AI-Lab/DEIMv2 | Heavy model. Native loader required. |
| `deimv2-x` | deimv2 | detect | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/Intellindust-AI-Lab/DEIMv2 | Heaviest DEIMv2. Native loader required. |
| `rtdetrv4-s` | rtdetrv4 | detect | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/RT-DETRs/RT-DETRv4 | RT-DETRv4 release/checkpoint URLs not officially numbered. |
| `co-dino-inst-vit-l-coco` | co-dino | segment | expert_sidecar | Apache-2.0 | `pip install visionservex` | https://github.com/Sense-X/Co-DETR | Requires OpenMMLab MMDetection sidecar. |
| `maskdino-swinl-coco` | maskdino | segment | expert_sidecar | Apache-2.0 | `pip install visionservex` | https://github.com/IDEA-Research/MaskDINO | Requires Detectron2 environment. |
| `sam-vit-base` | sam | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/segment-anything | - |
| `sam2-hiera-tiny` | sam2 | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/sam2 | - |
| `sam3-base` | sam3 | foundation_segment | external_api | Apache-2.0 | `pip install visionservex` | https://github.com/facebookresearch/sam3 | Access may be gated at the HF namespace. |
| `dinov2-small` | dinov2 | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/dinov2 | - |
| `dinov2-base` | dinov2 | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/dinov2 | - |
| `dinov2-large` | dinov2 | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/dinov2 | - |
| `dinov2-giant` | dinov2 | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/dinov2 | - |
| `dinov3-vitb16` | dinov3 | embed | audit_only | Apache-2.0 | `pip install visionservex` | https://github.com/facebookresearch/dinov3 | DINOv3 HF model card name not verified live. |
| `florence-2-base` | florence-2 | vlm | runnable | MIT | `pip install 'visionservex[hf]'` | https://github.com/microsoft/Florence-2 | - |
| `florence-2-large` | florence-2 | vlm | runnable | MIT | `pip install 'visionservex[hf]'` | https://github.com/microsoft/Florence-2 | - |
| `owlv2-base-patch16` | owlv2 | open_vocab_detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/owlv2-base-patch16-ensemble | - |
| `owlv2-large-patch14` | owlv2 | open_vocab_detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/owlv2-large-patch14-ensemble | - |
| `siglip2-base-patch16-224` | siglip2 | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/siglip2-base-patch16-224 | - |
| `grounding-dino-1.5-pro` | grounding-dino | open_vocab_detect | external_api | Custom | `pip install visionservex` | https://github.com/IDEA-Research/Grounding-DINO-1.5-API | API-gated. Requires IDEA-Research token. |
| `grounding-dino-1.6-pro` | grounding-dino | open_vocab_detect | external_api | Custom | `pip install visionservex` | https://github.com/IDEA-Research/Grounding-DINO-1.5-API | API-gated. Requires IDEA-Research token. |
| `dino-x-api` | dino-x | open_vocab_detect | external_api | Custom | `pip install visionservex` | https://github.com/IDEA-Research/Grounding-DINO-1.5-API | - |
| `anomalib-patchcore` | anomalib | anomaly | expert_sidecar | Apache-2.0 | `pip install anomalib  # not in visionservex core` | https://github.com/open-edge-platform/anomalib | anomalib is heavy; kept outside core dependencies. |
| `osnet-x1.0` | osnet | reid | expert_sidecar | MIT | `pip install torchreid  # not in core` | https://github.com/KaiyangZhou/deep-person-reid | torchreid not in core dependencies. |
| `bytetrack` | bytetrack | track | expert_sidecar | MIT | `pip install visionservex` | https://github.com/ifzhang/ByteTrack | ByteTrack not packaged; tracker pipeline pending. |
| `totalsegmentator` | totalsegmentator | segment | non_core_license_optional | Apache-2.0 | `pip install totalsegmentator  # not in core` | https://github.com/wasserth/TotalSegmentator | Heavy dependency on nibabel, SimpleITK, etc. |
| `medsam` | medsam | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/bowang-lab/MedSAM | - |
| `medsam2` | medsam2 | foundation_segment | expert_sidecar | Apache-2.0 | `pip install visionservex` | https://github.com/bowang-lab/MedSAM2 | - |
| `nnunet-v2` | nnunet | segment | expert_sidecar | Apache-2.0 | `pip install visionservex` | https://github.com/MIC-DKFZ/nnUNet | Training/inference framework, not a single model. |
| `rtmdet-r2-s` | rtmdet | obb | expert_sidecar | Apache-2.0 | `pip install visionservex` | https://github.com/open-mmlab/mmrotate | Requires OpenMMLab MMRotate sidecar. |
| `prithvi-eo-2.0` | prithvi | embed | non_core_license_optional | Apache-2.0 | `pip install visionservex` | https://github.com/NASA-IMPACT/Prithvi-EO-2.0 | Multispectral input — not standard RGB. |
| `agriclip` | agriclip | embed | audit_only | check | `pip install visionservex` | https://github.com/umair1221/AgriCLIP | License and HF availability not verified live. |
| `yolo-world` | yolo-world | open_vocab_detect | do_not_add | check | `pip install visionservex` | https://github.com/AILab-CVC/YOLO-World | License likely GPL/AGPL — excluded from permissive Apache/MIT core. |
| `owlvit-base-patch32` | owlvit | open_vocab_detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/owlvit-base-patch32 | - |
| `owlvit-large-patch14` | owlvit | open_vocab_detect | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/owlvit-large-patch14 | - |
| `clip-vit-base-patch32` | clip | embed | runnable | MIT | `pip install 'visionservex[hf]'` | https://github.com/openai/CLIP | - |
| `clip-vit-large-patch14` | clip | embed | runnable | MIT | `pip install 'visionservex[hf]'` | https://github.com/openai/CLIP | - |
| `siglip-base-patch16-224` | siglip | embed | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | google/siglip-base-patch16-224 | - |
| `convnextv2-tiny` | convnextv2 | classify | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/ConvNeXt-V2 | - |
| `convnextv2-base` | convnextv2 | classify | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/ConvNeXt-V2 | - |
| `convnextv2-large` | convnextv2 | classify | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/ConvNeXt-V2 | - |
| `sam2.1-hiera-tiny` | sam2.1 | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/sam2 | - |
| `sam2.1-hiera-small` | sam2.1 | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/sam2 | - |
| `sam2.1-hiera-base-plus` | sam2.1 | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/sam2 | - |
| `sam2.1-hiera-large` | sam2.1 | foundation_segment | runnable | Apache-2.0 | `pip install 'visionservex[hf]'` | https://github.com/facebookresearch/sam2 | - |
| `fastsam-s` | fastsam | foundation_segment | do_not_add | AGPL-3.0 | `git clone https://github.com/CASIA-IVA-Lab/FastSAM.git` | https://github.com/CASIA-IVA-Lab/FastSAM | License is AGPL-3.0. VisionServeX core requires Apache-2.0/MIT-compatible licenses. |
| `fastsam-x` | fastsam | foundation_segment | do_not_add | AGPL-3.0 | `git clone https://github.com/CASIA-IVA-Lab/FastSAM.git` | https://github.com/CASIA-IVA-Lab/FastSAM | License AGPL-3.0 — excluded from core. |
| `mobilesam` | mobilesam | foundation_segment | expert_sidecar | Apache-2.0 | `pip install mobile-sam  # or: git clone https://github.com/ChaoningZhang/MobileSAM` | https://github.com/ChaoningZhang/MobileSAM | No HF Hub model card for pip-installable weights. Checkpoint distributed via GitHub release. |
| `efficientsam` | efficientsam | foundation_segment | expert_sidecar | Apache-2.0 | `git clone https://github.com/yformer/EfficientSAM.git && pip install -e .` | https://github.com/yformer/EfficientSAM | No HF Hub model; GitHub-only install. |
| `hq-sam` | hq-sam | foundation_segment | expert_sidecar | Apache-2.0 | `pip install segment-anything-hq  # check latest release` | https://github.com/SysCV/sam-hq | No HF Hub model with standard SAM API compatibility confirmed. |
| `edgesam` | edgesam | foundation_segment | expert_sidecar | Apache-2.0 | `git clone https://github.com/chongzhou96/EdgeSAM.git && pip install -e .` | https://github.com/chongzhou96/EdgeSAM | GitHub-only install; no HF Hub model. |
