# Model downloads

VisionServeX has a first-class downloader. It supports four backends,
dispatched on each registry entry's `download_type`:

| `download_type`   | Behavior                                                       |
| ----------------- | -------------------------------------------------------------- |
| `synthetic`       | Built-in mocks — no download required.                         |
| `huggingface`     | `huggingface_hub.snapshot_download` (with `[hf]` extra).       |
| `github_release`  | Streaming HTTPS GET with resume + SHA-256 verification.         |
| `direct_url`      | Same as `github_release`; any HTTPS URL declared in the entry. |
| `manual`          | Refuses to download; prints upstream install instructions.     |
| `external_api`    | Refuses to self-host; points to upstream API terms.            |
| `not_available`   | Friendly error with a hint.                                    |

## Commands

```bash
visionservex pull <model_id>                 # interactive progress bar
visionservex pull <model_id> --json          # machine-readable
visionservex pull <model_id> --force         # re-download even if cached
visionservex pull <model_id> --offline       # use cache only

visionservex pull-easy                       # all beginner-friendly auto-downloadables
visionservex pull-recommended                # top recommendation per task
visionservex pull-all --task detect          # all detection models with auto_download=true
visionservex pull-all --include-non-auto     # include manual/external (will mostly error out)
visionservex pull-all --yes-i-understand-large-downloads
```

## Cache

```bash
visionservex cache path                      # where weights live
visionservex cache list                      # what is cached
visionservex cache verify [<model_id>]       # check SHA-256 where declared
visionservex cache clean [<model_id>]        # delete (asks for confirmation unless --yes)
visionservex cache repair <model_id>         # rebuild a manifest after a manual file drop
```

`VISION_SERVEX_CACHE_DIR` or `VISIONSERVEX_CACHE__CACHE_DIR` change the
location.

## Robustness

- **Resume**: partial files end in `.partial`; if interrupted, the next
  pull resumes from the last byte.
- **Atomic rename**: the file becomes its final name only after verification.
- **SHA-256**: verified when the registry declares `checkpoint_sha256`.
- **Retry**: exponential backoff (up to 4 attempts) for transient HTTP errors.
- **Disk-space check**: a pre-flight verifies free bytes before starting.
- **Per-model lock**: two concurrent requests for the same model share one
  download.
- **HF tokens**: read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if set.
  Tokens are never written to logs.

## Auto-pull

Disabled by default. Enable via config:

```bash
export VISIONSERVEX_MODELS__AUTO_PULL=true
export VISIONSERVEX_MODELS__AUTO_PULL_POLICY=easy_only
export VISIONSERVEX_MODELS__AUTO_PULL_MAX_SIZE_GB=5
export VISIONSERVEX_MODELS__AUTO_PULL_REQUIRE_AUTH=true
```

Policies:

| Policy                        | Meaning                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `never`                       | Never auto-pull (same as `auto_pull=false`).                            |
| `easy_only` (default)         | Only `very_easy`/`easy` models with `auto_download=true`.               |
| `registry_allowed`            | Any model with `auto_download=true`.                                    |
| `all_auto_downloadable`       | Same as above; alias.                                                   |

Public-mode safety: `auto_pull_require_auth` (true by default) refuses
auto-pull on a public endpoint when authentication is disabled.

## Job mode for long downloads

Predict endpoints accept `?wait_for_download=false`. When the requested
model is missing and auto-pull is allowed, the server returns:

```json
{
  "request_id": "...",
  "status": "downloading",
  "job_id": "...",
  "model_id": "...",
  "message": "Model weights are being downloaded.",
  "progress_url": "/jobs/<id>"
}
```

Poll the job:

```http
GET /jobs/<id>
GET /jobs/<id>/events
DELETE /jobs/<id>
```

Job statuses: `queued`, `checking_dependencies`, `downloading`, `verifying`,
`loading_model`, `running_inference`, `completed`, `failed`, `cancelled`.

## Safety guarantees

- We never download arbitrary URLs supplied via API requests. Only URLs
  that already appear in a bundled registry entry are fetched.
- We never auto-install pip packages from a request.
- We never execute remote code as part of a download.
- We never log tokens.
- We never bypass user-configured offline mode.
