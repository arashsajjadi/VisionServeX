# Object detection

VisionServeX supports two permissive-license detector families:

- **D-FINE** (`dfine-*`) — Apache-2.0
- **RF-DETR** (`rfdetr-*`) — Apache-2.0

The mock detection model (`mock-detect`) is always available.

## Python

```python
from visionservex import VisionModel

m = VisionModel("dfine-small")             # or "mock-detect"
result = m.predict("image.jpg")
for det in result.detections:
    print(det.label, det.score, det.box.to_xyxy())
result.save("out.jpg")
```

## CLI

```bash
visionservex predict dfine-small image.jpg --save out.jpg
visionservex predict mock-detect image.jpg --json
```

## HTTP

```bash
curl -F "image=@image.jpg" -F "model_id=dfine-small" \
     http://127.0.0.1:8080/detect
```

`POST /detect` requires `image` (multipart) and `model_id` (form). The
response uses the stable [PredictionResponse](../api_reference.md#prediction-response)
shape with one entry per detection.

## Choosing

- **Laptop / CPU**: `dfine-small`, `rfdetr-base` on small images.
- **GPU**: any. `rfdetr-large` benefits most.
- **Strict license**: both families are Apache-2.0; verify the specific
  checkpoint you choose against upstream.

## Limitations

Real backends are partial in 0.1.x. The schema, CLI, and HTTP contracts are
stable; raw outputs may come from the `MockEngine` until you install the
backend extras and upstream weights.
