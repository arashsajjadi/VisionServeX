# Security Policy

VisionServeX targets safe local use first, with explicit opt-in for public
exposure. This document describes the threat model, secure defaults, and the
checklist that must hold before exposing an instance to the internet.

## Reporting a vulnerability

Please email the maintainers (replace with your contact in a fork) or file a
private security advisory on the project repository. Do not file public GitHub
issues for security reports.

We aim to acknowledge new reports within 5 business days.

## Secure defaults

VisionServeX ships with the following defaults:

| Setting                  | Default     |
| ------------------------ | ----------- |
| Server bind address      | `127.0.0.1` |
| Public mode              | disabled    |
| API key authentication   | disabled    |
| Remote URL image input   | disabled    |
| Local file path input    | disabled    |
| CORS                     | disabled    |
| Max upload size          | 20 MiB      |
| Max image pixel area     | ~33 MP      |
| Rate limit               | 120 / min   |
| Audit logging            | enabled     |
| Cloudflare auto-expose   | disabled    |

## Threat model

We distinguish four operating modes:

1. **Local-only mode (default).** Bound to loopback. Trusted client is the
   developer or a process on the same machine.
2. **LAN mode.** Bound to a private interface behind a trusted network.
   Recommended: enable API key, enable rate limiting, never expose remote URL
   inputs.
3. **Public Cloudflare Tunnel mode.** API key is mandatory. We recommend
   Cloudflare Access policies, service tokens for programmatic clients, and
   optional mTLS for high-security clients. Origin must not also be exposed
   directly on the public internet.
4. **Production reverse-proxy mode.** Behind nginx/Caddy/Traefik with TLS
   termination, authentication, and request size limits applied at both the
   proxy and the application layer.

## Implemented protections

- API key (Bearer) authentication via `Authorization` header.
- Optional Cloudflare Access headers (`CF-Access-Client-Id`,
  `CF-Access-Client-Secret`) recognized as alternative auth.
- Request body size limits.
- Image pixel-area and dimension limits to mitigate decompression bombs.
- Safe `Pillow.Image.open` usage with `Image.MAX_IMAGE_PIXELS` enforced.
- MIME type allowlist for uploads.
- Path traversal rejection for any path-style input.
- SSRF protection for remote URL fetches: blocks `localhost`, private IPv4
  ranges, loopback IPv6, link-local, and requires `https://` by default.
- Token redaction in logs.
- Audit logging when public mode is enabled.
- No automatic execution of remote code at runtime.
- Optional SHA-256 verification of downloaded checkpoints when hashes are
  declared in the registry.

## What we do NOT do

- We do not run arbitrary user-supplied Python code.
- We do not fetch and execute model code from third-party servers without
  explicit user opt-in.
- We do not claim production-grade adversarial robustness for any model.
- We do not provide legal advice; refer to upstream licenses before commercial
  use of any model.

## Public exposure checklist

Before flipping `VISIONSERVEX_SERVER__PUBLIC_MODE=true` or running
`visionservex tunnel run`, confirm:

- [ ] `VISIONSERVEX_AUTH__ENABLED=true`.
- [ ] `VISIONSERVEX_AUTH__API_KEY` is a high-entropy secret (>=32 bytes).
- [ ] Rate limit is appropriate for your tier.
- [ ] Max upload size and pixel area are tuned to your hardware.
- [ ] Origin (the host running VisionServeX) is firewalled from the public
      internet on the server port (only cloudflared talks to it).
- [ ] You have configured a Cloudflare Access policy (identity-based for
      browsers, service token for automation, optional mTLS for high-value
      clients).
- [ ] You have rotated tokens and recorded their use in a secret store.
- [ ] Audit logs are being collected.
- [ ] You understand which upstream model licenses apply to your deployment
      (see `docs/model_licenses.md`).
