# Deployment

This page covers four common deployment shapes.

## 1. Local-only (default)

`pip install 'visionservex[server]'` then `visionservex serve`. Listens on
127.0.0.1:8080. Used for development, scripts on the same host, and
prototypes.

## 2. LAN

Bind to a private interface (e.g. `10.0.0.5`) behind your trusted network.

```bash
export VISIONSERVEX_SERVER__HOST=10.0.0.5
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
visionservex serve
```

Open the port only on the trusted interface. Never bind 0.0.0.0 without auth.

## 3. Cloudflare Tunnel (recommended for public exposure)

See [`cloudflare_tunnel.md`](cloudflare_tunnel.md). Summary:

1. `cloudflared` runs alongside VisionServeX (or in a separate container).
2. The origin port is bound to 127.0.0.1; only cloudflared talks to it.
3. A Cloudflare Access policy gates traffic at the edge.
4. VisionServeX enforces API key as defense in depth.

## 4. Reverse proxy (nginx / Caddy / Traefik)

Run VisionServeX on 127.0.0.1 and put a hardened reverse proxy in front for
TLS termination, request limits, and observability. Example Caddyfile:

```caddy
api.example.com {
  reverse_proxy 127.0.0.1:8080 {
    transport http {
      read_timeout  120s
      write_timeout 120s
    }
  }
  request_body {
    max_size 25MB
  }
  encode zstd gzip
}
```

The proxy and VisionServeX should both enforce body-size limits and
authentication.

## Docker

```bash
docker build -f docker/Dockerfile -t visionservex:cpu .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e VISIONSERVEX_AUTH__ENABLED=true \
  -e VISIONSERVEX_AUTH__API_KEY=$(openssl rand -hex 32) \
  visionservex:cpu
```

`docker/docker-compose.yml` includes an optional cloudflared sidecar.

## Concurrency

VisionServeX schedules inference per model with a configurable concurrency
cap (`runtime.per_model_concurrency`) and a bounded queue
(`runtime.queue_size`). When the queue is full it responds 503 BUSY rather
than blocking the event loop. Tune these to your hardware and SLA.

## Observability

`GET /metrics` returns counters, latency percentiles, queue stats, and
loaded models. Shape is stable; you can scrape with simple polling.

## Model cache

Models live in `~/.cache/visionservex` (Linux/macOS) or the platform
equivalent. Override with `VISION_SERVEX_CACHE_DIR`. In Docker, mount this
as a volume to persist downloads across container restarts.
