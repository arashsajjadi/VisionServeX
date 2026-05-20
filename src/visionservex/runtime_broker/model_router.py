# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Map ``model_id`` → ``runtime_id``.

The router uses two sources, in order:

1. The ``supported_models`` lists declared in ``runtime_specs.yaml``.
2. The v2.46 recovery plan in ``reports/v246_exact_50_recovery_plan.csv``.

The router does not guess. If a model is not present in either source, the
caller receives :class:`UnknownModelError` with the exact remediation step.
"""

from __future__ import annotations

import csv
from pathlib import Path

from visionservex.runtime_broker.spec_loader import RuntimeSpec, load_specs

__all__ = [
    "UnknownModelError",
    "resolve_runtime",
    "routing_table",
]


class UnknownModelError(KeyError):
    """Raised when a model_id has no known runtime mapping."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model_id = model_id

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"model_id '{self.model_id}' has no runtime mapping. Add it to a "
            f"runtime's supported_models list in runtime_specs.yaml or to the "
            f"v246_exact_50_recovery_plan.csv as a row with runtime_id."
        )


def _recovery_plan_path() -> Path:
    """Return the path to the v246 recovery plan CSV.

    The plan ships in ``reports/v246_exact_50_recovery_plan.csv`` at the repo
    root. When the package is installed without the ``reports/`` tree (e.g.,
    via PyPI), the function returns the path even if the file is absent;
    callers must guard for missing-file before reading.
    """

    return Path(__file__).resolve().parents[3] / "reports" / "v246_exact_50_recovery_plan.csv"


def _read_plan_mapping(plan_path: Path) -> dict[str, str]:
    """Return ``{model_id: runtime_id}`` from the recovery plan, or empty dict if missing."""

    if not plan_path.exists():
        return {}
    out: dict[str, str] = {}
    with plan_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = (row.get("model_id") or "").strip()
            rid = (row.get("runtime_id") or "").strip()
            if mid and rid:
                out[mid] = rid
    return out


def routing_table(
    specs: dict[str, RuntimeSpec] | None = None,
    plan_path: Path | str | None = None,
) -> dict[str, str]:
    """Return the full ``model_id -> runtime_id`` mapping used by the broker.

    The spec ``supported_models`` lists take precedence over the recovery
    plan — the recovery plan is only the back-up when a model has not yet
    been added to a runtime's ``supported_models`` list.
    """

    specs = specs or load_specs()
    table: dict[str, str] = {}

    for spec in specs.values():
        for mid in spec.supported_models:
            table.setdefault(mid, spec.id)

    plan = _read_plan_mapping(Path(plan_path) if plan_path is not None else _recovery_plan_path())
    for mid, rid in plan.items():
        if rid not in specs:
            continue
        table.setdefault(mid, rid)

    return table


def resolve_runtime(
    model_id: str,
    specs: dict[str, RuntimeSpec] | None = None,
    plan_path: Path | str | None = None,
) -> str:
    """Return the runtime_id for ``model_id`` or raise :class:`UnknownModelError`."""

    table = routing_table(specs=specs, plan_path=plan_path)
    if model_id not in table:
        raise UnknownModelError(model_id)
    return table[model_id]
