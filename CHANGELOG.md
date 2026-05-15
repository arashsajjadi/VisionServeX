# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-05-15

### Added
- Initial public scaffold.
- Model registry with permissive-license-first defaults.
- `VisionModel` high-level Python API with stable result schemas.
- FastAPI server with `/health`, `/ready`, `/version`, `/models`,
  `/predict`, `/detect`, `/segment`, `/pose`, `/classify`,
  `/open-vocab/detect`, `/grounded-segment`, and `/metrics`.
- Typer CLI: `doctor`, `list-models`, `info`, `pull`, `cache`, `predict`,
  `serve`, `tunnel`, `benchmark`, `export`, `config`.
- Security middleware: API key auth, rate limit, body-size limit,
  image validators, SSRF guard, log redaction.
- Cloudflare Tunnel integration via external `cloudflared` binary, with a
  safe ingress config generator that includes a catch-all 404.
- `MockEngine` for tests and engine stubs for D-FINE, RF-DETR, SAM 2,
  Grounding DINO with actionable install hints.
- LRU model cache with VRAM-aware lazy loading.
- Docker (CPU) image and `docker-compose.yml` with optional cloudflared
  sidecar.
- Documentation covering installation, quickstart, model zoo, tasks,
  Cloudflare Tunnel, security, deployment, performance, troubleshooting,
  model licenses, and an LLM-agent guide.
