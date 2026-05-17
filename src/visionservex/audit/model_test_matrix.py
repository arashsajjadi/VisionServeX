# SPDX-License-Identifier: Apache-2.0
"""Build model test matrix rows."""

from __future__ import annotations

from typing import Any


def build_model_rows() -> list[dict[str, Any]]:
    """Return a list of test matrix rows (one per registry model)."""
    from visionservex.audit.builder import _model_row
    from visionservex.cli.model_health_commands import _load_matrix_rows
    from visionservex.registry import default_registry

    matrix_rows = {r["model_id"]: r for r in _load_matrix_rows()}
    reg = default_registry()
    return [_model_row(entry, matrix_rows.get(entry.id)) for entry in reg.list()]
