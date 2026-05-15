<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

# VisionServeX

> A permissive-license-aware Python framework for serving modern computer
> vision models locally and over Cloudflare Tunnel.
>
> By **Arash Sajjadi** · PhD Candidate, Department of Computer Science,
> University of Saskatchewan · supervised by **Prof. Mark Eramian** at the
> Computer Vision Lab.

VisionServeX is designed to be **easy enough for a complete beginner**:
install → `doctor` → `recommend` → `pull` → `predict` → `serve`. No need to
understand CUDA, PyTorch internals, HuggingFace Hub semantics, or
Cloudflare DNS to get started.

The framework is intentionally:

- **Beginner-friendly.** `visionservex doctor` tells you what your machine
  can run, and `visionservex recommend` picks a model for you.
- **Permissive-license aware.** Default models are Apache-2.0, MIT, or BSD.
  Anything restricted, gated, or commercially uncertain is clearly labelled.
- **Honest about implementation status.** Each registry entry says whether
  this build is `wired`, `partial`, or `stub`. Stubs do **not** silently
  fake real predictions.
- **Predictable.** Stable JSON schemas across the Python API, CLI
  (`--json` everywhere), and HTTP. Designed for LLM agents.
- **Secure by default.** Bound to `127.0.0.1`, public mode requires auth
  and explicit opt-in, SSRF and decompression-bomb guards, log redaction,
  catch-all 404 in Cloudflare ingress.

## Beginner quickstart

```bash
# 1. install
pip install 'visionservex[server]'

# 2. see what your machine can run
visionservex doctor

# 3. ask the system to recommend a model
visionservex recommend --task detect --simple

# 4. download a model (the recommended one — or any other)
visionservex pull mock-detect          # always works — no extras needed
# visionservex pull rfdetr-small       # once the rfdetr backend is wired

# 5. run a prediction
visionservex predict mock-detect examples/images/street.jpg --save outputs/out.jpg

# 6. start the API
visionservex serve

# 7. call it
curl -F "image=@examples/images/street.jpg" -F "model_id=mock-detect" \
     http://127.0.0.1:8080/detect
```

## Installation

Basic install (CLI, registry, Python API, mock backend, system diagnostics):

```bash
pip install visionservex
```

Optional extras:

```bash
pip install 'visionservex[server]'      # FastAPI HTTP server
pip install 'visionservex[hf]'          # Hugging Face Transformers/Hub
pip install 'visionservex[grounding]'   # Grounding DINO (real backend)
pip install 'visionservex[rfdetr]'      # RF-DETR
pip install 'visionservex[dfine]'       # D-FINE
pip install 'visionservex[sam2]'        # SAM 2 / SAM 2.1
pip install 'visionservex[onnx]'        # ONNX Runtime
pip install 'visionservex[all]'         # server + onnx + hf + cloudflare
```

OpenMMLab integrations (`rtmpose`, `rtmdet-r`, `co-dino-inst`, `internimage`)
are installed via `openmim`; see [`docs/installation.md`](docs/installation.md).

## Python API

```python
from visionservex import VisionModel

model = VisionModel("mock-detect")              # works without optional deps
result = model.predict("image.jpg")
print(result.summary())
print(result.to_json())
result.save("annotated.jpg")
```

Open-vocabulary detection (Grounding DINO; requires `visionservex[grounding]`):

```python
m = VisionModel("grounding-dino-tiny", auto_pull=True)
result = m.predict("image.jpg", prompts=["red car", "person", "leash"])
```

## What's actually wired, honestly

| Family                         | Models                                                        | This build                        |
| ------------------------------ | ------------------------------------------------------------- | --------------------------------- |
| Mock (deterministic)           | `mock-*`                                                      | **wired** (no install required)   |
| Open-vocabulary detection      | `grounding-dino-tiny`, `grounding-dino-swin-t`, `…-swin-b`    | **wired** (HF Transformers)       |
| Object detection (D-FINE)      | `dfine-n / s / m / l / x`                                     | stub — registry only              |
| Object detection (RF-DETR)     | `rfdetr-nano / small / medium / large`                        | stub — registry only              |
| Instance segmentation (RF-DETR-Seg) | `rfdetr-seg-nano / small / medium / large / xlarge / 2xlarge` | stub — registry only         |
| Pose (RTMPose)                 | `rtmpose-t / s / m / l / m-384 / l-384`                       | stub — manual install             |
| OBB (RTMDet-R, RTMDet-R2)      | `rtmdet-r-t/s/m/l`, `rtmdet-r2-t/s/m/l`                       | stub — manual install             |
| Classification (Swin V2)       | `swinv2-tiny / small / base / large`                          | stub — registry only              |
| Classification (InternImage)   | `internimage-t / s / b / l / h`                               | stub — manual install (CUDA ops)  |
| Foundation segmentation (SAM 2)| `sam2-hiera-tiny / small / base-plus / large`                 | stub — optional install           |
| Grounded universal seg         | `grounded-sam2`, `seem-*`, `oneformer-*`                      | stub — composed pipeline pending  |
| External / API-only            | `grounding-dino-1.5`, `grounding-dino-1.6`                    | external — no self-hosting        |

This table is generated from the bundled registry (`visionservex list-models`).
A stub does **not** silently return mock predictions; it raises a clear
"missing dependency" error unless you explicitly set
`VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK=true`.

## Which model should I start with?

- **I just want detection that works right now**: `mock-detect`.
- **I want text-prompt detection that really runs**: `grounding-dino-tiny`
  (after `pip install 'visionservex[grounding]'`).
- **I have an NVIDIA GPU**: run `visionservex doctor`; it picks for you.
- **I have a small laptop**: stick with `mock-*` and `grounding-dino-tiny`
  at small image sizes.

## HTTP server

```bash
visionservex serve   # 127.0.0.1:8080 by default
```

Stable response shape:

```json
{
  "request_id": "...",
  "status": "completed",
  "model_id": "mock-detect",
  "task": "detect",
  "backend": "mock",
  "device": "cpu",
  "precision": "fp32",
  "latency_ms": 3.1,
  "model_loaded_from": null,
  "cache_path": null,
  "fallback_reason": null,
  "results": [...],
  "warnings": [],
  "metadata": {}
}
```

Auto-pull during requests is opt-in (`VISIONSERVEX_MODELS__AUTO_PULL=true`)
and policy-controlled (`auto_pull_policy=never|easy_only|registry_allowed|all_auto_downloadable`).

When a model is missing and the client passes `?wait_for_download=false`,
the server returns a **job id** and progress URL:

```json
{
  "request_id": "...",
  "status": "downloading",
  "job_id": "abc123",
  "model_id": "grounding-dino-tiny",
  "message": "Model weights are being downloaded.",
  "progress_url": "/jobs/abc123"
}
```

## Secure public exposure

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")

visionservex tunnel doctor                 # checks for cloudflared
visionservex tunnel create visionservex
visionservex tunnel route visionservex api.example.com
visionservex tunnel config api.example.com --out tunnel.yaml
visionservex serve &
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

See [`docs/cloudflare_tunnel.md`](docs/cloudflare_tunnel.md) and
[`docs/security.md`](docs/security.md).

## Docs

- [`docs/index.md`](docs/index.md)
- [`docs/installation.md`](docs/installation.md)
- [`docs/beginner_quickstart.md`](docs/beginner_quickstart.md)
- [`docs/device_check.md`](docs/device_check.md)
- [`docs/model_downloads.md`](docs/model_downloads.md)
- [`docs/model_zoo.md`](docs/model_zoo.md)
- [`docs/model_licenses.md`](docs/model_licenses.md)
- [`docs/cloudflare_tunnel.md`](docs/cloudflare_tunnel.md)
- [`docs/security.md`](docs/security.md)
- [`docs/cli.md`](docs/cli.md) · [`docs/api_reference.md`](docs/api_reference.md) · [`docs/python_api.md`](docs/python_api.md)
- [`docs/troubleshooting.md`](docs/troubleshooting.md) · [`docs/performance.md`](docs/performance.md)
- [`docs/llm_agent_guide.md`](docs/llm_agent_guide.md)
- [`docs/about.md`](docs/about.md)

## License

Apache-2.0 (`SPDX-License-Identifier: Apache-2.0`). See [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE). Each integrated upstream model retains its own
license; see [`docs/model_licenses.md`](docs/model_licenses.md).

## Citation

If you use VisionServeX in academic work, please cite it via
[`CITATION.cff`](CITATION.cff). Suggested BibTeX:

```bibtex
@software{sajjadi2026visionservex,
  author = {Arash Sajjadi},
  title  = {{VisionServeX: A permissive-license-aware framework for local computer vision model serving}},
  year   = {2026},
  url    = {https://github.com/example/visionservex},
  note   = {Developed under the supervision of Prof. Mark Eramian, University of Saskatchewan.}
}
```

## Security

Report vulnerabilities privately per [`SECURITY.md`](SECURITY.md). Review
the public-exposure checklist before running `visionservex tunnel run`.
