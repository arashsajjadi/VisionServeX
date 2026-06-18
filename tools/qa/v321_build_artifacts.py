# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the v3.21 Anastig contract + final model matrix from capabilities.

Pure projection of ``model_capabilities()``. Extends the v3.20 contract with the
v3.21 **sidecar dimension**: models that are live only through an isolated Docker
sidecar (Florence-2, OpenMMLab RTMPose) get their own disjoint primary buckets
(``*_ready_live_sidecar``) plus an overlapping ``sidecar_required_live`` view, and
every row carries ``sidecar_*`` + ``fine_tune_kind`` fields.

Outputs:
  docs/anastig_model_allowlist_v321.json
  docs/anastig_model_contract_v321.md
  docs/qa/v321_sidecar_blocker_elimination/final_model_matrix.json
  docs/qa/v321_sidecar_blocker_elimination/final_model_matrix.md
"""

from __future__ import annotations

import json
from pathlib import Path

import visionservex as vsx
from visionservex.core.model import model_capabilities
from visionservex.readiness import taxonomy as tx

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
QA = REPO / "docs" / "qa" / "v321_sidecar_blocker_elimination"

PRIMARY_BUCKETS = [
    "train_ready_live",
    "inference_ready_live",
    "embedding_fine_tune_ready_live",
    "foundation_segment_inference_only_live",
    "open_vocab_ready_live",
    "vlm_ready_live",
    "pose_ready_live",
    "obb_ready_live",
    # v3.21 sidecar-live primary buckets (disjoint from the host-live ones).
    "vlm_ready_live_sidecar",
    "inference_ready_live_sidecar",
    "segmentation_ready_live_sidecar",
    "gated_token_required",
    "hidden_catalog_only",
    "hidden_custom_loader_required",
    "blocked_dependency",
    "blocked_weights",
    "blocked_license",
    "blocked_partial",
    "blocked_oom",
    "blocked_upstream",
    "unknown_review_required",
]
VIEW_BUCKETS = [
    "fine_tune_ready_live",
    "sidecar_required_live",
    "sam_decoder_fine_tune_ready",
    "classification_train_ready_live",
    "detection_train_ready_live",
    "segmentation_train_ready_live",
]

UI_COPY = {
    "train_ready_live": "Train ready",
    "inference_ready_live": "Inference only",
    "embedding_fine_tune_ready_live": "Fine-tune ready (embedding head)",
    "fine_tune_ready_live": "Fine-tune ready",
    "sidecar_required_live": "Live via Docker sidecar",
    "sam_decoder_fine_tune_ready": "Fine-tune ready (SAM mask decoder)",
    "foundation_segment_inference_only_live": "Inference only (promptable segmentation)",
    "open_vocab_ready_live": "Inference only (open-vocabulary)",
    "vlm_ready_live": "Inference only (VLM)",
    "pose_ready_live": "Inference only (pose)",
    "obb_ready_live": "Inference only (oriented boxes)",
    "vlm_ready_live_sidecar": "Inference only (VLM, Docker sidecar)",
    "inference_ready_live_sidecar": "Inference only (Docker sidecar)",
    "segmentation_ready_live_sidecar": "Inference only (segmentation, Docker sidecar)",
    "gated_token_required": "Needs token",
    "hidden_catalog_only": "Hidden: engine not wired",
    "hidden_custom_loader_required": "Hidden: custom loader required",
    "blocked_dependency": "Admin only: dependency missing",
    "blocked_weights": "Hidden: weights unavailable",
    "blocked_license": "Hidden: legal/license",
    "blocked_partial": "Hidden: partial implementation",
    "blocked_oom": "Admin only: out of memory",
    "blocked_upstream": "Admin only: upstream crash",
    "unknown_review_required": "Admin only: review required",
}

_FOUNDATION_SEG = {"sam", "sam2", "sam2.1", "mobilesam", "hq-sam", "efficientsam"}


def _primary(c: dict) -> str:
    st, task, fam = c["readiness_state"], c["task"], c["family"]
    if st == tx.TRAIN_READY_LIVE:
        return "train_ready_live"
    if st == tx.EMBEDDING_READY_LIVE:
        return (
            "embedding_fine_tune_ready_live"
            if c["fine_tune_live_verified"]
            else "inference_ready_live"
        )
    if st == tx.SEGMENTATION_READY_LIVE:
        return (
            "foundation_segment_inference_only_live"
            if fam in _FOUNDATION_SEG
            else "inference_ready_live"
        )
    if st == tx.OPEN_VOCAB_READY_LIVE:
        return "open_vocab_ready_live"
    if st == tx.VLM_READY_LIVE:
        return "vlm_ready_live"
    if st == tx.VLM_READY_LIVE_SIDECAR:
        return "vlm_ready_live_sidecar"
    if st == tx.INFERENCE_READY_LIVE_SIDECAR:
        return "inference_ready_live_sidecar"
    if st == tx.SEGMENTATION_READY_LIVE_SIDECAR:
        return "segmentation_ready_live_sidecar"
    if st == tx.INFERENCE_READY_LIVE:
        if task == "pose":
            return "pose_ready_live"
        if task == "obb":
            return "obb_ready_live"
        return "inference_ready_live"
    return {
        tx.GATED_TOKEN_REQUIRED: "gated_token_required",
        tx.CATALOG_ONLY_ENGINE_NOT_WIRED: "hidden_catalog_only",
        tx.CUSTOM_LOADER_REQUIRED: "hidden_custom_loader_required",
        tx.DEPENDENCY_MISSING: "blocked_dependency",
        tx.WEIGHTS_MISSING: "blocked_weights",
        tx.LICENSE_BLOCKED: "blocked_license",
        tx.NON_COMMERCIAL_BLOCKED: "blocked_license",
        tx.PARTIAL_IMPLEMENTATION_BLOCKED: "blocked_partial",
        tx.OOM_BLOCKED: "blocked_oom",
        tx.UPSTREAM_CRASH: "blocked_upstream",
    }.get(st, "unknown_review_required")


def main() -> int:
    caps = {m: model_capabilities(m) for m in vsx.list_models()}
    buckets: dict[str, list[str]] = {b: [] for b in (PRIMARY_BUCKETS + VIEW_BUCKETS)}

    for mid, c in caps.items():
        buckets[_primary(c)].append(mid)
        if c["fine_tune_live_verified"]:
            buckets["fine_tune_ready_live"].append(mid)
        if c.get("sidecar_required"):
            buckets["sidecar_required_live"].append(mid)
        if c.get("fine_tune_kind") == "frozen_encoder_decoder":
            buckets["sam_decoder_fine_tune_ready"].append(mid)
        if c["train_live_verified"]:
            if c["task"] in ("classify", "classification"):
                buckets["classification_train_ready_live"].append(mid)
            if c["task"] in ("detect", "obb", "open_vocab_detect"):
                buckets["detection_train_ready_live"].append(mid)
            if c["task"] in ("segment", "foundation_segment", "grounded_segment"):
                buckets["segmentation_train_ready_live"].append(mid)
    for b in buckets:
        buckets[b] = sorted(buckets[b])

    (DOCS / "anastig_model_allowlist_v321.json").write_text(
        json.dumps(
            {
                "version": vsx.__version__,
                "generated_from": "visionservex.model_capabilities (do not hand-edit)",
                "primary_partition_buckets": PRIMARY_BUCKETS,
                "view_buckets": VIEW_BUCKETS,
                "ui_copy": UI_COPY,
                **buckets,
            },
            indent=2,
        )
        + "\n"
    )

    counts = {b: len(v) for b, v in buckets.items()}
    md = [
        "# Anastig Model Contract (v3.21)",
        "",
        f"Source of truth: `visionservex.model_capabilities(model_id)` (v{vsx.__version__}).",
        "The **primary buckets** are a disjoint partition of all models; the **view buckets**",
        "are overlapping live sub-views. v3.21 adds the **sidecar dimension**: models live only",
        "through an isolated Docker sidecar get `*_ready_live_sidecar` primary buckets and the",
        "`sidecar_required_live` view. Anastig drives UI from `anastig_visibility` /",
        "`anastig_train_visibility` / `anastig_finetune_visibility` / `anastig_sidecar_visibility`.",
        "",
        "## v3.21 capability fields",
        "- `sidecar_supported` / `sidecar_required` / `sidecar_name` — isolated-sidecar routing.",
        "- `sidecar_live` / `sidecar_cpu_verified` / `sidecar_gpu_verified` — sidecar live-proof.",
        "- `anastig_sidecar_visibility` — `show_*_sidecar` / `host_preferred_sidecar_available` / `none`.",
        "- `fine_tune_kind` — `full_supervised` / `frozen_backbone_head` / `frozen_encoder_decoder` / `none`.",
        "",
        "## Buckets + UI copy",
        "",
        "| Bucket | Count | UI copy |",
        "|---|---:|---|",
    ]
    for b in PRIMARY_BUCKETS + VIEW_BUCKETS:
        md.append(f"| `{b}` | {counts[b]} | {UI_COPY.get(b, '')} |")
    md += ["", "## Members", ""]
    for b in PRIMARY_BUCKETS + VIEW_BUCKETS:
        md.append(f"### `{b}` ({counts[b]})\n")
        md.append(", ".join(f"`{i}`" for i in buckets[b]) if buckets[b] else "_(none)_")
        md.append("")
    (DOCS / "anastig_model_contract_v321.md").write_text("\n".join(md) + "\n")

    rows = []
    for mid, c in caps.items():
        rows.append(
            {
                "model_id": mid,
                "task": c["task"],
                "family": c["family"],
                "license": c["license"],
                "gated": c["gated"],
                "requires_token": c["requires_token"],
                "commercial_safe": c["commercial_safe"],
                "readiness_state": c["readiness_state"],
                "inference_live": c["inference_live_verified"],
                "train_live": c["train_live_verified"],
                "fine_tune_live": c["fine_tune_live_verified"],
                "fine_tune_kind": c["fine_tune_kind"],
                "sidecar_supported": c["sidecar_supported"],
                "sidecar_required": c["sidecar_required"],
                "sidecar_name": c["sidecar_name"],
                "sidecar_live": c["sidecar_live"],
                "anastig_visibility": c["anastig_visibility"],
                "anastig_sidecar_visibility": c["anastig_sidecar_visibility"],
                "anastig_train_visibility": c["anastig_train_visibility"],
                "anastig_finetune_visibility": c["anastig_finetune_visibility"],
                "blocker": c["blocker"],
            }
        )
    rows.sort(key=lambda r: (r["task"], r["model_id"]))
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "final_model_matrix.json").write_text(
        json.dumps({"version": vsx.__version__, "n": len(rows), "models": rows}, indent=2) + "\n"
    )

    primary_total = sum(counts[b] for b in PRIMARY_BUCKETS)
    sidecar_total = (
        counts["vlm_ready_live_sidecar"]
        + counts["inference_ready_live_sidecar"]
        + counts["segmentation_ready_live_sidecar"]
    )
    print(
        f"[v321] contract + matrix for {len(rows)} models | primary partition total={primary_total}"
    )
    print(
        "[v321] live: inference={} train={} finetune={} sidecar={} sam_decoder_ft={}".format(
            sum(r["inference_live"] for r in rows),
            sum(r["train_live"] for r in rows),
            sum(r["fine_tune_live"] for r in rows),
            sidecar_total,
            counts["sam_decoder_fine_tune_ready"],
        )
    )
    if primary_total != len(rows):
        raise SystemExit(f"PARTITION ERROR: primary={primary_total} != models={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
