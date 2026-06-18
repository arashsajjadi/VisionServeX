# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the v3.19 Anastig contract + final model matrix from capabilities.

Pure projection of ``model_capabilities()`` — re-run after any promotion. Outputs:
  docs/anastig_model_allowlist_v319.json   (the 14-bucket contract)
  docs/anastig_model_contract_v319.md
  docs/qa/v319_operationalize_all_models/final_model_matrix.json
  docs/qa/v319_operationalize_all_models/final_model_matrix.md
"""

from __future__ import annotations

import json
from pathlib import Path

import visionservex as vsx
from visionservex.core.model import model_capabilities
from visionservex.readiness import taxonomy as tx

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
QA = REPO / "docs" / "qa" / "v319_operationalize_all_models"

# readiness_state -> v3.19 allowlist bucket.
_BUCKET = {
    tx.TRAIN_READY_LIVE: "train_ready_live",
    tx.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION: "train_ready_derived_admin_only",
    tx.INFERENCE_READY_LIVE: "inference_ready_live",
    tx.VLM_READY_LIVE: "inference_ready_live",
    tx.CORRESPONDENCE_READY_LIVE: "inference_ready_live",
    tx.SEGMENTATION_READY_LIVE: "segmentation_ready_live",
    tx.OPEN_VOCAB_READY_LIVE: "open_vocab_ready_live",
    tx.EMBEDDING_READY_LIVE: "embedding_ready_live",
    tx.GATED_TOKEN_REQUIRED: "gated_token_required",
    tx.CATALOG_ONLY_ENGINE_NOT_WIRED: "hidden_catalog_only",
    tx.CUSTOM_LOADER_REQUIRED: "hidden_custom_loader_required",
    tx.DEPENDENCY_MISSING: "blocked_dependency",
    tx.WEIGHTS_MISSING: "blocked_weights",
    tx.PARTIAL_IMPLEMENTATION_BLOCKED: "blocked_partial",
    tx.LICENSE_BLOCKED: "blocked_license",
    tx.NON_COMMERCIAL_BLOCKED: "blocked_license",
    tx.UPSTREAM_CRASH: "blocked_dependency",
    tx.OOM_BLOCKED: "blocked_dependency",
    tx.TASK_NOT_SUPPORTED: "blocked_dependency",
    tx.UNKNOWN_REVIEW_REQUIRED: "unknown_review_required",
    tx.INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION: "unknown_review_required",
}
BUCKETS = [
    "train_ready_live",
    "train_ready_derived_admin_only",
    "inference_ready_live",
    "segmentation_ready_live",
    "open_vocab_ready_live",
    "embedding_ready_live",
    "gated_token_required",
    "hidden_catalog_only",
    "hidden_custom_loader_required",
    "blocked_dependency",
    "blocked_weights",
    "blocked_partial",
    "blocked_license",
    "unknown_review_required",
]


def main() -> int:
    caps = {m: model_capabilities(m) for m in vsx.list_models()}
    buckets: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for mid, c in caps.items():
        buckets[_BUCKET.get(c["readiness_state"], "unknown_review_required")].append(mid)
    for b in buckets:
        buckets[b] = sorted(buckets[b])

    (DOCS / "anastig_model_allowlist_v319.json").write_text(
        json.dumps(
            {
                "version": vsx.__version__,
                "generated_from": "visionservex.model_capabilities (do not hand-edit)",
                **buckets,
            },
            indent=2,
        )
    )

    counts = {b: len(v) for b, v in buckets.items()}
    md = [
        "# Anastig Model Contract (v3.19)",
        "",
        f"Source of truth: `visionservex.model_capabilities(model_id)` (v{vsx.__version__}).",
        "Generated from capabilities — no hand-editing. Machine-readable:",
        "`docs/anastig_model_allowlist_v319.json`.",
        "",
        "Anastig shows **only** live-ready models by default (`anastig_visibility`",
        "starting with `show_` except `show_token_required`). `train_ready_derived_admin_only`",
        "must be labelled *needs-validation / admin-only* and never offered as plain",
        "training-ready. Everything else is hidden or admin-only.",
        "",
        "## Bucket counts",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for b in BUCKETS:
        md.append(f"| `{b}` | {counts[b]} |")
    md += ["", "## Buckets", ""]
    for b in BUCKETS:
        md.append(f"### `{b}` ({counts[b]})\n")
        md.append(", ".join(f"`{i}`" for i in buckets[b]) if buckets[b] else "_(none)_")
        md.append("")
    (DOCS / "anastig_model_contract_v319.md").write_text("\n".join(md) + "\n")

    # ---- final model matrix (every row, every field) ----
    rows = []
    for mid, c in caps.items():
        syn = c["validated_syntax"]
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
                "inference_status": c["predict_supported"],
                "train_status": c["train_supported"],
                "reload_status": c["checkpoint_load_supported"],
                "export_status": bool(c["export_supported"]),
                "live_verified_inference": c["live_verified_inference"],
                "live_verified_train": c["live_verified_train"],
                "blocker": c["blocker"],
                "anastig_visibility": c["anastig_visibility"],
                "syntax": syn.get("predict") or next(iter(syn.values()), ""),
            }
        )
    rows.sort(key=lambda r: (r["task"], r["model_id"]))
    (QA / "final_model_matrix.json").write_text(
        json.dumps({"version": vsx.__version__, "n": len(rows), "models": rows}, indent=2)
    )
    fm = [
        f"# v3.19 Final Model Matrix — {len(rows)} models (v{vsx.__version__})",
        "",
        "| Model | Task | License | Gated | Tok | Commercial | readiness_state | inf | train | reload | export | live_inf | live_train | Visibility | Blocker |",
        "|---|---|---|:-:|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|---|---|",
    ]
    for r in rows:

        def y(b):
            return "Y" if b else "—"

        fm.append(
            f"| `{r['model_id']}` | {r['task']} | {r['license']} | {y(r['gated'])} | "
            f"{y(r['requires_token'])} | {y(r['commercial_safe'])} | `{r['readiness_state']}` | "
            f"{y(r['inference_status'])} | {y(r['train_status'])} | {y(r['reload_status'])} | "
            f"{y(r['export_status'])} | {y(r['live_verified_inference'])} | {y(r['live_verified_train'])} | "
            f"{r['anastig_visibility']} | {(r['blocker'] or '')[:42]} |"
        )
    (QA / "final_model_matrix.md").write_text("\n".join(fm) + "\n")
    print(f"[v319] anastig contract + final matrix for {len(rows)} models")
    print("[v319] bucket counts:", {b: counts[b] for b in BUCKETS if counts[b]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
