# Cloudflare Tunnel

VisionServeX integrates with Cloudflare Tunnel via the external
`cloudflared` binary. We do not vendor `cloudflared`; install it with your
platform's package manager.

## Why a tunnel?

`cloudflared` opens an *outbound* connection from your origin to Cloudflare
and lets Cloudflare proxy inbound requests to your local server. The
origin's listening port does not have to be publicly reachable; in fact it
should not be.

## Step-by-step

### 0. Install `cloudflared`

```bash
visionservex tunnel doctor
```

This prints the right install command for your OS. Linux Debian/Ubuntu:

```bash
curl -L https://pkg.cloudflare.com/cloudflared-stable-linux-amd64.deb -o /tmp/cf.deb
sudo dpkg -i /tmp/cf.deb
```

### 1. Authenticate

```bash
visionservex tunnel login
```

Follow the browser flow; cloudflared writes a certificate file to
`~/.cloudflared/cert.pem`.

### 2. Create a named tunnel

```bash
visionservex tunnel create visionservex
```

This writes a `~/.cloudflared/<UUID>.json` credentials file. Treat it like a
secret.

### 3. Route a hostname

```bash
visionservex tunnel route visionservex api.example.com
```

### 4. Enable VisionServeX authentication

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

### 5. Generate a safe ingress config

```bash
visionservex tunnel config api.example.com --tunnel-name visionservex --out tunnel.yaml
```

The generated file:

- Points `api.example.com` at `http://127.0.0.1:<port>` (your local origin).
- Sets reasonable connect/TLS timeouts.
- Ends with a **catch-all `http_status:404`** rule that prevents other
  hostnames from reaching your API through this tunnel. Do not remove it.
- Writes a placeholder `credentials-file:` value; replace it with the path
  printed by `tunnel create`.

### 6. Start the origin

```bash
visionservex serve &
```

### 7. Run the tunnel

```bash
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

The `--i-understand-this-is-public` flag is required. Without it, the CLI
prints the public-exposure checklist and refuses to run. The CLI also
refuses to run if `VISIONSERVEX_AUTH__ENABLED` is false.

## Cloudflare Access (highly recommended)

We strongly recommend adding a Cloudflare Access policy on top of the
tunnel. Open the Cloudflare Zero Trust dashboard → Access → Applications →
Add an Application, and attach a policy to `api.example.com`:

- **Browser users**: require an identity provider (Google, GitHub, OIDC, etc.).
- **Programmatic clients**: issue service tokens; clients send
  `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers.
- **High-value clients**: optional mTLS — Cloudflare presents a client
  certificate challenge.

### Service token example

```bash
curl https://api.example.com/detect \
  -H "Authorization: Bearer $VISIONSERVEX_API_KEY" \
  -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
  -F image=@image.jpg -F model_id=mock-detect
```

The VisionServeX API key is *defense in depth* — Access has already proven
the caller's identity at the edge.

## Running as a service

### Linux (systemd)

```ini
# /etc/systemd/system/visionservex-tunnel.service
[Unit]
Description=VisionServeX Cloudflare Tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=visionservex
ExecStart=/usr/bin/cloudflared tunnel --config /etc/visionservex/tunnel.yaml run
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now visionservex-tunnel
```

### macOS (launchd)

Create `~/Library/LaunchAgents/com.example.visionservex.tunnel.plist`
pointing at `cloudflared tunnel --config /Users/<user>/.cloudflared/tunnel.yaml run`,
then `launchctl load -w …plist`.

### Windows (service)

```powershell
cloudflared.exe service install --config C:\Users\<you>\.cloudflared\tunnel.yaml
```

## Closing thoughts

- Audit your generated YAML before running.
- Keep the catch-all 404 rule.
- Keep the origin firewalled from the public internet on the listening port;
  only cloudflared should talk to it (and only locally).
- Rotate API keys and service tokens periodically.
- Ship audit logs to a secure store.
