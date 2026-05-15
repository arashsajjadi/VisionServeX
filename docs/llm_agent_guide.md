# LLM agent guide

This document is written for AI coding assistants and automation that drive
VisionServeX. Everything below is deterministic.

## Stable surface

- CLI command names and flags are stable within a 0.x minor version.
- HTTP endpoint paths and stable response shapes are stable within 0.x.
- Result fields under `results: [...]` are stable per task; new fields may be
  added (backwards compatible) but existing fields will not be renamed.
- Error envelope (`{"request_id", "error": {"code", "message", "hint", "details"}}`)
  is stable. New error codes may be added.

## Probing

Always probe before acting:

```bash
visionservex doctor --json         # devices, deps, config, warnings
visionservex list-models --json    # registry contents
visionservex info <model_id> --json
```

## Predictable inference

CLI:

```bash
visionservex predict <model_id> <path> --json
visionservex predict <model_id> <path> --prompt "cat,dog" --json
```

HTTP:

```http
POST /detect
Content-Type: multipart/form-data
fields: image=@file, model_id=<id>
```

JSON open-vocab:

```http
POST /open-vocab/detect
Content-Type: application/json
{
  "model_id": "grounding-dino-tiny",
  "image_b64": "<base64>",
  "prompts": ["..."]
}
```

## Health checks

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/metrics
```

## Parsing errors

Always inspect `error.code` first. Map known codes:

- `MODEL_NOT_FOUND` → re-fetch `/models`.
- `TASK_MISMATCH` → pick a different model id.
- `ENGINE_UNAVAILABLE` → install the extra named in `message`/`hint`.
- `BUSY` → retry with exponential backoff (start at 200ms, cap at 5s).
- `RATE_LIMITED` → wait at least 1 minute or back off.
- `BAD_IMAGE`, `BAD_BASE64`, `BAD_MIME_TYPE` → fix input.
- `BAD_URL` → fall back to `image_b64`.
- `REQUEST_TOO_LARGE` → resize the image.
- `UNAUTHENTICATED` → check `Authorization: Bearer ...` header.

## Destructive commands

VisionServeX itself has very few destructive commands. The ones that exist:

- `visionservex cache clean` — irreversible, requires `--yes` for unattended
  runs.
- `visionservex config set` — writes to `.env` in the current directory.

The CLI never auto-downloads code or auto-runs external binaries unless you
explicitly invoke them.

## Avoid

- Spawning the server with `--public` while auth is disabled. The CLI warns
  but does not refuse.
- Running `visionservex tunnel run` without `--i-understand-this-is-public`
  (the CLI refuses anyway).
- Setting `VISIONSERVEX_INPUTS__ALLOW_URL_INPUTS=true` without an allowlist
  when the deployment is not on a trusted network.

## Example end-to-end automation

```bash
# 1. Diagnose
visionservex doctor --json > /tmp/doctor.json

# 2. Pick a stable mock for CI
MODEL=mock-detect

# 3. Start server
visionservex serve --host 127.0.0.1 --port 8080 &
sleep 1

# 4. Wait until ready
until curl -fsS http://127.0.0.1:8080/health >/dev/null; do sleep 0.2; done

# 5. Run prediction
curl -fsS -F "image=@/path/to/image.jpg" -F "model_id=$MODEL" \
  http://127.0.0.1:8080/detect | jq '.results | length'
```
