# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX v2.46.0 runtime broker.

Users should never need to know whether a model wants Python 3.9, Python 3.10,
OpenMMLab, NATTEN, Detectron2, CUDA 11.7, CUDA 12.1, conda, pip, git clone,
a checkpoint pull, an auth gate, or a license gate. The runtime broker hides
all of that behind one entry point:

    visionservex run <model_id> <input> [--task ...]

The broker resolves the model's runtime spec, prepares the environment if
missing (or returns the exact next command), runs license/auth gates, executes
the sidecar, normalizes the output, and records the call in the ledger.
"""

from __future__ import annotations

from visionservex.runtime_broker.broker import (
    BrokerError,
    BrokerResult,
    RuntimeBroker,
)
from visionservex.runtime_broker.model_router import (
    UnknownModelError,
    resolve_runtime,
    routing_table,
)
from visionservex.runtime_broker.result_schema import (
    BROKER_BLOCKER_CODES,
    BrokerBlocker,
    CanonicalDetection,
    CanonicalMask,
    CanonicalOBB,
    CanonicalResult,
)
from visionservex.runtime_broker.spec_loader import (
    RuntimeSpec,
    SpecLoadError,
    load_specs,
)

__all__ = [
    "BROKER_BLOCKER_CODES",
    "BrokerBlocker",
    "BrokerError",
    "BrokerResult",
    "CanonicalDetection",
    "CanonicalMask",
    "CanonicalOBB",
    "CanonicalResult",
    "RuntimeBroker",
    "RuntimeSpec",
    "SpecLoadError",
    "UnknownModelError",
    "load_specs",
    "resolve_runtime",
    "routing_table",
]
