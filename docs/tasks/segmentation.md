# Instance segmentation

## Models

- `rfdetr-seg` (Apache-2.0): same engine family as RF-DETR detectors.
- `co-dino-inst` (Apache-2.0, license uncertain on some checkpoints):
  optional, accuracy-first; requires OpenMMLab toolchain.
- `oneformer-coco` (MIT): panoptic / instance / semantic via OneFormer.

`mock-segment` is always available for plumbing.

## Python

```python
from visionservex import VisionModel

m = VisionModel("rfdetr-seg")
r = m.predict("image.jpg")
for seg in r.segments:
    print(seg.label, seg.score, seg.box.to_xyxy(), seg.mask.shape)
r.save("annotated.jpg")
```

## HTTP

`POST /segment` (multipart upload + `model_id` form field).

Server JSON output omits the raw mask array (it is large and not
JSON-friendly) and reports `mask_shape` and `mask_pixels_on` instead. To
return the actual mask, use the Python API or request
`POST /predict/annotated` for an annotated JPEG.

## Notes

For mask formats: VisionServeX stores masks as `numpy.uint8` H,W arrays with
values 0/1. Convert to RLE/COCO as needed via `result.to_coco()` (returns
shape metadata; you are responsible for RLE encoding if you need it).
