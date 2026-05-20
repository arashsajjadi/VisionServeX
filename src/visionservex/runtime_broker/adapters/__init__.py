# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Output adapters that map sidecar-native results to :class:`CanonicalResult`.

Each runtime declares its adapter as ``module:function`` in
``runtime_specs.yaml``. Adapters are tiny on purpose — the broker resolves
them lazily and only loads what the current run actually needs.

The functions here are stubs. A real recovery session will fill in each
adapter once the corresponding sidecar build succeeds and we have an
honest output sample to map.
"""

from __future__ import annotations

from typing import Any

from visionservex.runtime_broker.result_schema import (
    BrokerBlocker,
    CanonicalResult,
)


def _stub(
    runtime_id: str,
    model_id: str,
    task: str,
    *,
    blocker_code: str,
    blocker_message: str,
    next_action: str,
) -> CanonicalResult:
    return CanonicalResult(
        model_id=model_id,
        runtime_id=runtime_id,
        task=task,
        blocker=BrokerBlocker(
            code=blocker_code,
            message=blocker_message,
            runtime_id=runtime_id,
            model_id=model_id,
            next_action=next_action,
        ),
    )


def to_canonical_unimplemented(
    payload: Any, *, model_id: str, runtime_id: str, task: str
) -> CanonicalResult:
    """Generic placeholder used by adapters that do not yet have wiring."""

    return _stub(
        runtime_id,
        model_id,
        task,
        blocker_code="OUTPUT_ADAPTER_NOT_FOUND",
        blocker_message=(
            "Output adapter not wired yet for this runtime. The runtime spec "
            "is correct, the sidecar may even run, but the canonical schema "
            "adapter has not been implemented."
        ),
        next_action=(
            "Implement to_canonical(...) in this adapter module against a real "
            "sample output from the sidecar."
        ),
    )
