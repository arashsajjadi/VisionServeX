# Quickstart

## 30-second Python

```python
from visionservex import VisionModel

model = VisionModel("mock-detect")
result = model.predict("image.jpg")
print(result.summary())
print(result.to_json(indent=2))
result.save("annotated.jpg")
```

The `mock-*` models work without any optional backend installed.

## 30-second CLI

```bash
pip install 'visionservex[server]'

visionservex doctor
visionservex list-models
visionservex predict mock-detect image.jpg --save out.jpg
```

JSON mode for automation:

```bash
visionservex predict mock-detect image.jpg --json
```

## 30-second server

```bash
visionservex serve
```

In another shell:

```bash
curl -F "image=@image.jpg" -F "model_id=mock-detect" \
     http://127.0.0.1:8080/detect | jq
```

## Using a real backend

Object detection with D-FINE (requires `pip install 'visionservex[torch]'`):

```python
from visionservex import VisionModel

m = VisionModel("dfine-small")
result = m.predict("image.jpg")
result.save("annotated.jpg")
```

> 0.1.x note: the D-FINE engine wiring is a stub that returns deterministic
> mock detections when the real backend cannot be loaded. The stable contract
> (model ids, response shapes, CLI commands) is final; real engine wiring
> ships incrementally.

Open-vocabulary detection with Grounding DINO (requires
`pip install 'visionservex[grounding]'`):

```python
m = VisionModel("grounding-dino-tiny")
result = m.predict("image.jpg", prompts=["cat", "person riding a bicycle"])
```

Foundation segmentation with SAM 2.1 (requires `visionservex[sam2]` and the
upstream `sam2` package):

```python
m = VisionModel("sam2-hiera-base")
result = m.predict("image.jpg", prompts=["foreground"])
```

## Going public, safely

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")

visionservex serve &
visionservex tunnel doctor
visionservex tunnel create visionservex
visionservex tunnel route visionservex api.example.com
visionservex tunnel config api.example.com --out tunnel.yaml
# Review tunnel.yaml. The catch-all 404 is mandatory; do not remove it.
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

See [`cloudflare_tunnel.md`](cloudflare_tunnel.md) and
[`security.md`](security.md).
