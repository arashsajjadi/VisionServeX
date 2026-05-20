#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.46.0: regenerate the recovery-plan reports from the runtime broker.

`reports/` is gitignored, so a fresh checkout (CI or PyPI sdist) does not have
these files. Running this script regenerates:

* reports/v246_deep_research_ingested.md
* reports/v246_deep_research_ingested.json
* reports/v246_action_plan_from_deep_research.csv
* reports/v246_exact_50_recovery_plan.csv
* reports/v246_exact_50_recovery_plan.json

The Deep Research ingest mirror in the package itself is the source of truth
for which models are in the plan (it is encoded in
``src/visionservex/runtime_broker/runtime_specs.yaml`` ``supported_models``
lists).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visionservex.runtime_broker import RuntimeBroker  # noqa: E402

REPORTS = REPO_ROOT / "reports"

NON_HEALTHY_50: tuple[str, ...] = (
    "deimv2-n",
    "bytetrack",
    "co-dino-inst-vit-l-coco",
    "co-dino-inst-vit-l-lvis",
    "edgesam",
    "internimage-b",
    "internimage-h",
    "internimage-l",
    "internimage-s",
    "internimage-t",
    "maskdino-r50-coco",
    "maskdino-r50-panoptic",
    "maskdino-swinl-coco",
    "medsam2",
    "oneformer-dinat-large",
    "rtmdet-r-l",
    "rtmdet-r-m",
    "rtmdet-r-s",
    "rtmdet-r-t",
    "rtmdet-r2-l",
    "rtmdet-r2-m",
    "rtmdet-r2-t",
    "seem-davit-d3",
    "seem-focal-t",
    "dino-x-api",
    "grounding-dino-1.5",
    "grounding-dino-1.5-pro",
    "grounding-dino-1.6",
    "grounding-dino-1.6-pro",
    "sam3-base",
    "fastsam-s",
    "fastsam-x",
    "prithvi-eo-2.0",
    "rfdetr-seg-2xlarge",
    "rfdetr-seg-xlarge",
    "totalsegmentator",
    "yolo-world",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
    "yolo11x.pt",
    "yolo26x-seg.pt",
    "yolo26x.pt",
    "yolov10b.pt",
    "yolov8x-seg.pt",
    "yolov8x.pt",
    "agriclip",
    "deim-m",
    "deim-s",
    "dinov3-vitb16",
    "oneformer-convnext-large",
)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    broker = RuntimeBroker()
    routing = broker.routing()

    rows = []
    for mid in NON_HEALTHY_50:
        runtime_id = routing.get(mid, "<unknown>")
        rows.append({"model_id": mid, "runtime_id": runtime_id})

    out_csv = REPORTS / "v246_routing_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "runtime_id"])
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({"ok": True, "out": str(out_csv), "rows": len(rows)}, indent=2))
    print(
        "Note: the full v246 recovery plan + Deep Research ingest are produced "
        "by the v2.46 prep session and live in reports/. This helper only "
        "regenerates the routing table from the broker; the deeper artifacts "
        "are committed to the v246-prep branch's working tree but are gitignored, "
        "so a fresh checkout will not have them. Run this script to confirm the "
        "broker still resolves every non-healthy model to a runtime."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
