# CLI reference

Add `--debug` for verbose logs and full stack traces. Add `--json` to
machine-readable commands.

## Global

```bash
visionservex --help
visionservex --debug doctor
visionservex --config path/to/config.yaml ...
```

## Meta

```bash
visionservex version
visionservex doctor [--json]
visionservex devices [--json]
visionservex examples
```

`doctor` is the one-stop diagnostic: system, devices, dependencies, safety
warnings, beginner pick, exact next command.

## Models

```bash
visionservex list-models [--task TASK] [--status STATUS] [--family FAMILY] [--easy] [--can-run] [--json]
visionservex info MODEL_ID [--json]
visionservex recommend [--task TASK] [--device DEVICE] [--vram GB] [--simple] [--limit N] [--json]
```

## Downloads

```bash
visionservex pull MODEL_ID [--force] [--offline] [--json]
visionservex pull-easy [--yes] [--json]
visionservex pull-recommended [--yes] [--json]
visionservex pull-all [--task TASK] [--only-auto-downloadable/--include-non-auto] [--yes-i-understand-large-downloads] [--json]
```

## Cache

```bash
visionservex cache path [--json]
visionservex cache list [--json]
visionservex cache verify [MODEL_ID] [--json]
visionservex cache clean [MODEL_ID] [--yes] [--json]
visionservex cache repair MODEL_ID [--json]
```

## Inference and serving

```bash
visionservex predict MODEL_ID INPUT [--save OUT] [--prompt "a,b"] [--auto-pull] [--json]
visionservex benchmark MODEL_ID INPUT [--n N] [--json]
visionservex export MODEL_ID --format onnx --out path.onnx
visionservex serve [--host HOST] [--port PORT] [--public] [--reload]
```

## Run-example

```bash
visionservex run-example check-device
visionservex run-example detect
visionservex run-example segment
visionservex run-example classify
visionservex run-example open-vocab
visionservex run-example api
```

## Cloudflare Tunnel

```bash
visionservex tunnel doctor [--json]
visionservex tunnel login
visionservex tunnel create [NAME]
visionservex tunnel route TUNNEL HOSTNAME
visionservex tunnel config HOSTNAME --out tunnel.yaml
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

## Config

```bash
visionservex config show [--json]
visionservex config set KEY VALUE          # writes to .env
```
