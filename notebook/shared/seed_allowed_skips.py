#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Record allowed skips for v2.39's 49-target models gated on auth, license,
deprecation, wrong-registry, manual checkpoints, or sidecars not yet built.

This is the precise-skip alternative to a notebook call — see
``notebook/shared/notebook_call_tracker.py::record_skip_simple``.
"""

from __future__ import annotations

import json
from pathlib import Path

from visionservex.reporting.notebook_calls import NotebookCallLedger, record_skip

NB_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = NB_ROOT / "99_final_report/reports/notebook_model_call_ledger.json"

# (model_id, task, notebook_path, reason)
ALLOWED_SKIPS: list[tuple[str, str, str, str]] = [
    # auth_required (HF gated / API only)
    ("sam3-base", "vlm", "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb", "auth_required"),
    (
        "grounding-dino-1.5",
        "open_vocab",
        "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
        "auth_required",
    ),
    (
        "grounding-dino-1.6",
        "open_vocab",
        "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
        "auth_required",
    ),
    # opt_in_license_required (PML / non-commercial / yolo-world)
    (
        "rfdetr-seg-xlarge",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "opt_in_license_required",
    ),
    (
        "rfdetr-seg-2xlarge",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "opt_in_license_required",
    ),
    # upstream_deprecated
    (
        "deim-m",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "upstream_deprecated",
    ),
    (
        "deim-s",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "upstream_deprecated",
    ),
    # wrong_registry_entry
    (
        "oneformer-convnext-large",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "wrong_registry_entry",
    ),
    # manual_checkpoint_required (rtdetrv4 checkpoint_present=false in current run)
    (
        "rtdetrv4-s",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "manual_checkpoint_required",
    ),
    (
        "rtdetrv4-m",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "manual_checkpoint_required",
    ),
    (
        "rtdetrv4-l",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "manual_checkpoint_required",
    ),
    (
        "rtdetrv4-x",
        "detect",
        "01_object_detection/Object_Detection_Benchmark.ipynb",
        "manual_checkpoint_required",
    ),
    # sidecar_required (build failed / not yet built in this run)
    (
        "co-dino-inst-vit-l-coco",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "co-dino-inst-vit-l-lvis",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "maskdino-r50-coco",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "maskdino-r50-panoptic",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "oneformer-dinat-large",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "seem-davit-d3",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    (
        "seem-focal-t",
        "segment",
        "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
        "sidecar_required",
    ),
    # OpenMMLab OBB / Pose sidecar_required
    (
        "rtmdet-r-t",
        "obb",
        "09_aerial_obb/Aerial_OBB_Status.ipynb",
        "sidecar_required",
    ),
    ("rtmdet-r-s", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    ("rtmdet-r-m", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    ("rtmdet-r-l", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    ("rtmdet-r2-t", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    ("rtmdet-r2-m", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    ("rtmdet-r2-l", "obb", "09_aerial_obb/Aerial_OBB_Status.ipynb", "sidecar_required"),
    (
        "rtmpose-t",
        "pose",
        "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
        "sidecar_required",
    ),
    (
        "rtmpose-m",
        "pose",
        "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
        "sidecar_required",
    ),
    (
        "rtmpose-m-384x288",
        "pose",
        "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
        "sidecar_required",
    ),
    (
        "rtmpose-l",
        "pose",
        "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
        "sidecar_required",
    ),
    (
        "rtmpose-l-384x288",
        "pose",
        "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
        "sidecar_required",
    ),
    # InternImage sidecar_required
    (
        "internimage-t",
        "classify",
        "05_classification/Classification_Smoke.ipynb",
        "sidecar_required",
    ),
    (
        "internimage-s",
        "classify",
        "05_classification/Classification_Smoke.ipynb",
        "sidecar_required",
    ),
    (
        "internimage-b",
        "classify",
        "05_classification/Classification_Smoke.ipynb",
        "sidecar_required",
    ),
    (
        "internimage-l",
        "classify",
        "05_classification/Classification_Smoke.ipynb",
        "sidecar_required",
    ),
    (
        "internimage-h",
        "classify",
        "05_classification/Classification_Smoke.ipynb",
        "sidecar_required",
    ),
]


def main() -> int:
    led = (
        NotebookCallLedger.load(LEDGER_PATH)
        if LEDGER_PATH.exists()
        else NotebookCallLedger.init(LEDGER_PATH)
    )
    already_called = {c.model_id for c in led.calls if c.called_in_notebook}
    n_skips = 0
    for mid, task, nb_path, reason in ALLOWED_SKIPS:
        if mid in already_called:
            continue
        record_skip(
            model_id=mid,
            notebook=nb_path,
            section=nb_path.split("/", 1)[0],
            task=task,
            reason=reason,
            ledger=led,
        )
        n_skips += 1
    summary = led.coverage_summary()
    print(json.dumps({"n_skips_added": n_skips, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
