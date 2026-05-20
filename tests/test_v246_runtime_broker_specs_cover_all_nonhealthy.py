# SPDX-License-Identifier: Apache-2.0
"""Every non-healthy model in the v2.46 recovery plan must map to a runtime."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from visionservex.runtime_broker import RuntimeBroker, load_specs

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "reports" / "v246_exact_50_recovery_plan.csv"


def _plan_rows() -> list[dict[str, str]]:
    if not PLAN_PATH.exists():
        pytest.skip(
            "v246 recovery plan not present (reports/ is gitignored - "
            "generate locally before running these tests)"
        )
    with PLAN_PATH.open() as f:
        return list(csv.DictReader(f))


def test_recovery_plan_has_exactly_50_rows() -> None:
    rows = _plan_rows()
    assert len(rows) == 50, f"expected 50 rows, got {len(rows)}"


def test_every_plan_row_has_runtime_id() -> None:
    rows = _plan_rows()
    missing = [r["model_id"] for r in rows if not (r.get("runtime_id") or "").strip()]
    assert not missing, f"rows missing runtime_id: {missing}"


def test_every_runtime_id_in_plan_exists_in_specs() -> None:
    specs = load_specs()
    rows = _plan_rows()
    unknown = sorted({r["runtime_id"] for r in rows if r["runtime_id"] not in specs})
    assert not unknown, f"plan references unknown runtimes: {unknown}"


def test_required_runtimes_are_all_loaded() -> None:
    specs = load_specs()
    required = {
        "core_py311",
        "rtdetrv4_py311_torch",
        "codetr_openmmlab_py310",
        "internimage_dcnv3_py310",
        "obb_rtmdetr2_py310",
        "obb_mmrotate_legacy_py39",
        "pose_mmpose_py310",
        "oneformer_natten_py310",
        "maskdino_detectron2_py310",
        "tracking_bytetrack_py310",
        "promptable_edgesam_py310",
        "medical_medsam2_py310",
        "seem_xdecoder_container",
        "license_gate_runtime",
        "auth_gate_runtime",
        "registry_audit_runtime",
    }
    missing = required - set(specs)
    assert not missing, f"required runtimes missing from runtime_specs.yaml: {missing}"


def test_broker_routing_covers_every_plan_model() -> None:
    broker = RuntimeBroker()
    table = broker.routing()
    rows = _plan_rows()
    uncovered = [r["model_id"] for r in rows if r["model_id"] not in table]
    assert not uncovered, f"broker routing missing models: {uncovered}"


def test_broker_routing_runtime_matches_plan_runtime() -> None:
    broker = RuntimeBroker()
    table = broker.routing()
    rows = _plan_rows()
    mismatches = [
        (r["model_id"], table[r["model_id"]], r["runtime_id"])
        for r in rows
        if table[r["model_id"]] != r["runtime_id"]
    ]
    assert not mismatches, f"broker routing disagrees with plan: {mismatches[:5]}"


@pytest.mark.parametrize(
    "model_id",
    [
        "deimv2-n",
        "co-dino-inst-vit-l-coco",
        "internimage-t",
        "oneformer-dinat-large",
        "rtmdet-r2-l",
        "bytetrack",
        "edgesam",
        "medsam2",
        "seem-focal-t",
        "maskdino-r50-coco",
        "yolo11x.pt",
        "sam3-base",
        "dino-x-api",
    ],
)
def test_broker_explain_returns_full_info(model_id: str) -> None:
    result = RuntimeBroker().explain(model_id)
    assert result.runtime_id != "<unknown>", f"{model_id} has no runtime"
    assert result.commands, f"{model_id} produced no commands"
    assert result.blocker is None or result.blocker.code == "BROKER_DRY_RUN_NO_EXECUTE"
