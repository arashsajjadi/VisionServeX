# VisionServeX Aerial / Remote Sensing Domain

Aerial and satellite imagery workflows using VisionServeX. Oriented bounding box (OBB) detection and geospatial embedding are the primary use cases.

---

## Supported Models

| Model | Status | Install | Notes |
|-------|--------|---------|-------|
| `rtmdet-r2-s` (RTMDet-R2) | expert_sidecar | OpenMMLab MMRotate | OBB detection for aerial images |
| Oriented R-CNN | expert_sidecar | OpenMMLab MMRotate | Classic OBB detection |
| `prithvi-eo-2.0` | optional_extra | `ibm-nasa-geospatial` HF | Multispectral; not standard RGB |
| RemoteCLIP | unavailable | TBD | Status/license not verified |
| DOTA / VisDrone support | via MMRotate | OpenMMLab | Standard aerial benchmarks |

---

## RTMDet-R / RTMDet-R2 (Oriented Bounding Box)

RTMDet-R is MMRotate's real-time OBB detector. It is the recommended model for aerial object detection where objects have arbitrary orientations (ships, aircraft, vehicles in satellite imagery).

### Status

`expert_sidecar` — requires OpenMMLab MMRotate.

### Install (OpenMMLab sidecar)

```bash
# Create isolated conda env
conda create -n vsx-openmmlab python=3.10 -y
conda activate vsx-openmmlab

# Install MMRotate and its deps
pip install -U openmim
mim install mmcv
mim install mmdet
mim install mmrotate

# Validate
visionservex openmmlab validate
```

### Inference

```bash
# VisionServeX aerial command (returns delegation if MMRotate not set up):
visionservex aerial detect aerial_image.jpg \
  --model rtmdet-r2-s \
  --out /tmp/aerial_out

# Returns structured error if sidecar not installed:
# {"code": "OPENMMLAB_REQUIRED", ...}

# Actual MMRotate inference (in vsx-openmmlab env):
python mmrotate/demo/image_demo.py \
  aerial_image.jpg \
  configs/rotated_rtmdet/rotated_rtmdet-r_s_9x_dota.py \
  --checkpoint /path/to/rtmdet_r2_s.pth \
  --out-file /tmp/aerial_result.jpg
```

### DOTA benchmark

DOTA (Dataset for Object deTection in Aerial images) is the standard aerial benchmark. RTMDet-R2 achieves competitive mAP on DOTA v1.0 and v2.0.

```bash
# Evaluate on DOTA:
visionservex aerial benchmark --dataset dota:/path/to/dota --model rtmdet-r2-s
```

### VisDrone support

VisDrone is a drone-view pedestrian/vehicle detection benchmark. Use `rfdetr-large` or `dfine-x-o365-coco` for standard (axis-aligned) detection.

---

## Oriented R-CNN

Classic oriented bounding box detector, also via OpenMMLab MMRotate.

### Install

Same as RTMDet-R2 (OpenMMLab MMRotate sidecar).

### Inference

```bash
# In vsx-openmmlab env:
python mmrotate/demo/image_demo.py \
  aerial.jpg \
  configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py \
  --checkpoint /path/to/oriented_rcnn.pth
```

---

## Prithvi-EO 2.0

Prithvi is a geospatial foundation model from IBM NASA trained on multispectral satellite imagery.

### Status

`optional_extra` — not standard RGB input; requires geospatial processing.

**Blockers:**
- Input is multispectral (6 bands: RGB + NIR + SWIR1 + SWIR2), not standard 3-channel RGB
- Requires geospatial raster libraries (`rasterio`, `earthpy`)
- Not in VisionServeX core; geospatial use case

### Install

```bash
pip install rasterio earthpy
# HF model:
# huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0
```

### Use case

Prithvi-EO is designed for:
- Crop mapping from satellite
- Flood extent mapping
- Wildfire burn scar detection
- Cloud detection

When standard RGB is sufficient, use `dinov2-large` or `clip-vit-large-patch14` for geospatial embedding.

---

## RemoteCLIP

CLIP fine-tuned on remote sensing imagery.

### Status

`unavailable` — license and HF availability not verified.

**Next action:** Re-audit https://github.com/ChenDelong1999/RemoteCLIP.

**Interim:** Use `siglip2-base-patch16-224` or `clip-vit-large-patch14` for satellite image retrieval.

---

## Recommended Workflow for Aerial Detection

### Zero-shot (no fine-tuning)

```bash
pip install 'visionservex[hf]'

# Open-vocab detection for objects in aerial images
visionservex open-vocab grounding-dino-swin-b aerial.jpg \
  --prompt "ship,airplane,vehicle,storage tank,bridge"

# Or OWLv2
visionservex open-vocab owlv2-large-patch14 aerial.jpg \
  --prompt "car,truck,airplane,ship"
```

### Fine-tuned OBB (expert sidecar path)

1. Set up OpenMMLab MMRotate (`visionservex openmmlab create-env`)
2. Download RTMDet-R2 checkpoint from MMRotate model zoo
3. Fine-tune on your DOTA-format annotated aerial dataset
4. Run inference via MMRotate CLI; VisionServeX wraps delegation

---

## Related Commands

```bash
visionservex aerial --help
visionservex model-zoo sources --domain aerial
visionservex openmmlab validate
visionservex model-zoo blockers --family rtmdet
```
