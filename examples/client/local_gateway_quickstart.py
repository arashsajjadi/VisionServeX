#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""
VisionServeX Local Gateway Quickstart

Start the gateway first:
    visionservex gateway start

Or with a profile:
    visionservex gateway start --profile laptop
"""

from visionservex import Client

# Connect to local gateway
client = Client("http://127.0.0.1:8080")

# Check health
print("Health:", client.health())

# List available models
models = client.models()
print(f"Available models: {len(models)}")
for m in models[:5]:
    print(f"  {m['id']:30s} {m['task']:20s} {m['implementation_status']}")

# Object detection
result = client.detect("dfine-n", "examples/images/street.jpg")
print(f"\nDetection result: {result}")
print(f"  device={result.device}, latency={result.latency_ms:.1f}ms")

# Classification
result = client.classify("swinv2-tiny", "examples/images/dog.jpg")
print(f"\nClassification: {result}")
for item in result.results[:3]:
    print(f"  {item.get('label', '?')}: {item.get('score', 0):.3f}")

# Text-prompted segmentation
result = client.grounded_segment("grounded-sam2", "examples/images/street.jpg", "car, person")
print(f"\nGrounded segmentation: {result}")

# Gateway status
status = client.gateway_status()
print(
    f"\nGateway: {status.get('best_device', {}).get('name')} | {len(status.get('loaded_models', []))} models loaded"
)
