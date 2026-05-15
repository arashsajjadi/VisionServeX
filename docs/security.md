# Security

This document complements `SECURITY.md` at the repo root with practical
guidance.

## Defaults at a glance

| Setting                  | Default       |
| ------------------------ | ------------- |
| Server bind              | `127.0.0.1`   |
| Public mode              | disabled      |
| API key auth             | disabled      |
| Remote URL image input   | disabled      |
| Local file path input    | disabled      |
| CORS                     | disabled      |
| Max upload size          | 20 MiB        |
| Max image pixel area     | ~33 MP        |
| Max image dimension      | 8192 px       |
| Rate limit               | 120 / minute  |
| Auto-execute remote code | never         |

## Authentication

VisionServeX accepts these credentials when `auth.enabled=true`:

- `Authorization: Bearer <api-key>` (preferred).
- `X-API-Key: <api-key>` (alternative).
- Cloudflare Access service tokens: `CF-Access-Client-Id` +
  `CF-Access-Client-Secret`. We additionally require the secret to match the
  configured `api_key` (defense in depth). Adjust if you operate a more
  sophisticated header check via a reverse proxy.

Health (`/health`), version (`/version`), and model listing (`/models`,
`/models/{id}`) do not require auth. All inference endpoints do.

## Rate limiting

A simple in-memory token bucket keyed by client IP enforces
`limits.rate_limit_per_minute`. Behind a reverse proxy or Cloudflare,
configure trusted proxies so the bucket sees the real client.

## Image safety

Uploads are validated for:

- MIME type allowlist (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`, `image/tiff`).
- Body size (`limits.max_upload_bytes`).
- Image area in pixels (`limits.max_image_pixels`) — decompression-bomb guard.
- Image dimensions (`limits.max_image_dim`).
- Decode success with truncated images rejected.

## SSRF protection

When `inputs.allow_url_inputs=true`, remote image fetches go through
`validate_remote_url`:

- Require `https://` (unless `allow_http=True` in code).
- Reject userinfo (`user:pass@host`).
- Reject `localhost`, `metadata.google.internal`, and similar.
- Reject any hostname that resolves to a loopback, private, link-local,
  reserved, multicast, or unspecified address.
- Optional allowlist of hostnames.

## Path traversal

`safe_join(root, candidate)` rejects absolute paths and `..` components, and
re-checks the resolved path lives under `root`. Local-path image inputs are
disabled by default and must be explicitly enabled.

## Log redaction

Logs run through `RedactingFilter` which rewrites common secret patterns
(Bearer tokens, `api_key=…`, Cloudflare Access secrets) to `***REDACTED***`.
Use `visionservex.utils.logging.log_safe_dict` when serializing config.

## Audit logging

Every request gets an `X-Request-Id`. The middleware emits an access log
line per request including the request id, method, path, status, and
latency. In public mode, ship these logs to a secure store.

## Threat model

- **Local-only mode** (default): trusted host, no network exposure.
- **LAN mode**: trusted network. Enable auth; disable URL inputs unless
  needed.
- **Public mode via Cloudflare Tunnel**: auth mandatory; Cloudflare Access
  recommended; rate limit and body-size limits tuned to your tier.
- **Production reverse proxy mode**: nginx/Caddy/Traefik in front for TLS,
  WAF rules, and additional logging.

## Out of scope

- Adversarial robustness of underlying models.
- Privacy of any specific user input. You are responsible for processing
  personal data per applicable law.
- Vendor-specific guarantees of model checkpoints' licensing — verify
  upstream.
