# Beginner quickstart

Five minutes from zero to a working API.

## Step 1 — Install

```bash
pip install 'visionservex[server]'
```

That is the only required install. Optional extras (Hugging Face,
Grounding DINO, RF-DETR, SAM 2, OpenMMLab, ONNX Runtime) are documented in
[`installation.md`](installation.md) and are not needed for the first
prediction.

## Step 2 — Diagnose your system

```bash
visionservex doctor
```

You get a single screen with:

- OS / Python / package paths / cache path.
- CPU, RAM, free disk space at the cache.
- Detected devices (CPU + GPU when present), with VRAM totals when
  available.
- Optional dependencies and how to install each one.
- A recommended first model.
- The exact next command to copy and paste.

## Step 3 — See your model options

```bash
visionservex list-models --easy        # beginner-friendly only
visionservex list-models --can-run     # filter to models compatible with your devices
visionservex list-models --task detect
```

Add `--json` to any of these for machine-readable output.

## Step 4 — Recommendation

```bash
visionservex recommend --task detect --simple
```

The `--simple` flag prefers models that are:

- wired in this build,
- easy to install,
- auto-downloadable,
- compatible with your device.

## Step 5 — Pull weights

```bash
visionservex pull mock-detect          # always works
# visionservex pull grounding-dino-tiny  # real HF backend, needs `visionservex[grounding]`
```

For models marked `manual` or `external`, `pull` prints exact instructions
and refuses to invent a download path.

## Step 6 — Run a prediction

```bash
visionservex predict mock-detect examples/images/street.jpg --save outputs/out.jpg
```

For automation:

```bash
visionservex predict mock-detect examples/images/street.jpg --json
```

## Step 7 — Start the API

```bash
visionservex serve              # 127.0.0.1:8080 by default
```

In another shell:

```bash
curl -F "image=@examples/images/street.jpg" -F "model_id=mock-detect" \
     http://127.0.0.1:8080/detect | jq
```

## Step 8 — (Optional) Auto-pull on first request

If you want the server to download a model the first time someone asks for
it:

```bash
export VISIONSERVEX_MODELS__AUTO_PULL=true
export VISIONSERVEX_MODELS__AUTO_PULL_POLICY=easy_only
visionservex serve
```

The first request returns either the prediction (if `wait_for_download=true`)
or a job id (if `wait_for_download=false`); poll `/jobs/{id}` to track
progress.

## Step 9 — (Optional) Expose publicly via Cloudflare Tunnel

Read [`cloudflare_tunnel.md`](cloudflare_tunnel.md). VisionServeX refuses
to run the tunnel without authentication enabled and an explicit
confirmation flag.

## What if something goes wrong?

Run `visionservex doctor`. It will diagnose 90% of issues and print the
exact next command. For everything else see
[`troubleshooting.md`](troubleshooting.md).
