# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX licensing policy layer (v3.8).

A single, code-level source of truth for model license policy. The CSV matrix
(``v38_license_policy_matrix.csv``) and the policy report are *generated* from
:mod:`visionservex.licensing.policy`, so the CLI, the Python API, the tests, the
notebooks and the docs can never drift from each other.

This is an **advisory** layer. It classifies every advertised model into one of
the nine final-policy buckets and exposes the lawful next step. It does NOT
mutate the reconciled model-coverage ledger or the benchmark states of models
that already ship — it is consulted by the Hugging Face BYOT / model-license
workflow only.
"""

from __future__ import annotations

from visionservex.licensing.policy import (
    FINAL_POLICIES,
    POLICY,
    WARNING_TEXTS,
    ModelPolicy,
    get_policy,
    iter_policies,
    matrix_rows,
    resolve_model_id,
)

__all__ = [
    "FINAL_POLICIES",
    "POLICY",
    "WARNING_TEXTS",
    "ModelPolicy",
    "get_policy",
    "iter_policies",
    "matrix_rows",
    "resolve_model_id",
]
