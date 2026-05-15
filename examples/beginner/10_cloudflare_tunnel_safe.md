# Beginner example 10 — expose VisionServeX safely via Cloudflare Tunnel

This is a step-by-step checklist for a beginner who wants to expose their
local VisionServeX API to the internet *safely*.

**Prerequisite reading:** [`docs/cloudflare_tunnel.md`](../../docs/cloudflare_tunnel.md)
and [`docs/security.md`](../../docs/security.md).

> Going public is opt-in. VisionServeX will refuse to run the tunnel
> without an API key configured *and* an explicit confirmation flag.

## 1. Install `cloudflared`

```bash
visionservex tunnel doctor
```

It prints the right install command for your OS.

## 2. Log in to Cloudflare

```bash
visionservex tunnel login
```

This opens a browser. You select the Cloudflare zone you control.

## 3. Create a named tunnel

```bash
visionservex tunnel create visionservex
```

Cloudflare writes a credentials file under `~/.cloudflared/<UUID>.json`.

## 4. Route DNS to the tunnel

```bash
visionservex tunnel route visionservex api.example.com
```

Replace `api.example.com` with a hostname inside a zone you control.

## 5. Generate a safe ingress config

```bash
visionservex tunnel config api.example.com --tunnel-name visionservex --out tunnel.yaml
```

The file contains a **catch-all `http_status:404`** at the end. Keep it.

## 6. Enable VisionServeX authentication

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

Save the API key in a secret manager. It will be required on every API call.

## 7. Start VisionServeX

```bash
visionservex serve
```

## 8. Run the tunnel

```bash
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

Without the confirmation flag, the CLI prints the public-exposure
checklist and refuses to start.

## 9. (Recommended) Configure Cloudflare Access

In the Cloudflare Zero Trust dashboard → Access → Applications:

- Attach an Access policy to `api.example.com`.
- Use an identity provider for browser users.
- Issue **service tokens** (`CF-Access-Client-Id` / `CF-Access-Client-Secret`)
  for automation.
- Optional: enable mTLS for high-value clients.

## 10. Test it

```bash
curl https://api.example.com/detect \
  -H "Authorization: Bearer $VISIONSERVEX_AUTH__API_KEY" \
  -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
  -F image=@examples/images/street.jpg \
  -F model_id=mock-detect
```

If you can hit `/detect` *and* `/no-such-path` returns 404 (handled by the
catch-all), you are correctly configured.
