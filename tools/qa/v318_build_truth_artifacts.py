# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate every v3.18 truth artifact programmatically from ``model_capabilities``.

Nothing here is hand-maintained: the inventory, the legal matrix/audit, and the
Anastig allowlist/contract are all pure projections of the single source of
truth — ``visionservex.core.model.model_capabilities`` over ``list_models()``.
Re-run after any registry/policy/live-evidence change.

Outputs:
  docs/qa/v318_full_model_truth/discovered_models.json
  docs/qa/v318_full_model_truth/discovered_models.md
  docs/qa/v318_full_model_truth/model_inventory.csv
  docs/qa/v318_full_model_truth/legal_matrix.json
  docs/legal_model_audit.md
  docs/anastig_model_allowlist_v318.json
  docs/anastig_model_contract_v318.md
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import visionservex as vsx
from visionservex.core.model import model_capabilities
from visionservex.readiness import taxonomy as tx

REPO = Path(__file__).resolve().parents[2]
QA_DIR = REPO / "docs" / "qa" / "v318_full_model_truth"
DOCS = REPO / "docs"


def _variant(model_id: str, family: str) -> str:
    """Best-effort sub-variant token (model id minus a leading family prefix)."""
    fam_tokens = [family, family.split("-")[0]]
    for ft in fam_tokens:
        if ft and model_id.startswith(ft + "-"):
            return model_id[len(ft) + 1 :]
    return model_id


def collect() -> list[dict]:
    rows = []
    for mid in vsx.list_models():
        c = model_capabilities(mid)
        vis = c["anastig_visibility"]
        default_visible = vis in (
            "show_train",
            "show_inference",
            "show_embedding",
            "show_segmentation",
        )
        rows.append(
            {
                "model_id": mid,
                "family": c["family"],
                "variant": _variant(mid, c["family"]),
                "task": c["task"],
                "engine": c["engine"],
                "implementation_status": c["implementation_status"],
                "license": c["license"],
                "license_class": c["license_class"],
                "commercial_safe": c["commercial_safe"],
                "gated": c["gated"],
                "requires_token": c["requires_token"],
                "readiness": c["readiness"],
                "readiness_state": c["readiness_state"],
                "live_verified_inference": c["live_verified_inference"],
                "live_verified_train": c["live_verified_train"],
                "anastig_visibility": vis,
                "default_visible": default_visible,
                "reason_if_hidden": None
                if default_visible
                else (c["blocker"] or c["readiness_state"]),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Phase 1 — inventory
# --------------------------------------------------------------------------- #
def write_inventory(rows: list[dict]) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "discovered_models.json").write_text(
        json.dumps({"version": vsx.__version__, "n": len(rows), "models": rows}, indent=2)
    )

    # CSV
    fields = [
        "model_id",
        "family",
        "variant",
        "task",
        "engine",
        "implementation_status",
        "license",
        "commercial_safe",
        "gated",
        "requires_token",
        "readiness_state",
        "default_visible",
        "reason_if_hidden",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (QA_DIR / "model_inventory.csv").write_text(buf.getvalue())

    # Markdown
    import collections

    by_state = collections.Counter(r["readiness_state"] for r in rows)
    md = [
        f"# v3.18 Discovered Models — {len(rows)} total (v{vsx.__version__})",
        "",
        "Programmatically discovered from `list_models()` + the registry + the license",
        "policy. Every row is a pure projection of `model_capabilities(model_id)`.",
        "",
        "## Readiness state distribution",
        "",
        "| readiness_state | count |",
        "|---|---:|",
    ]
    for s, n in sorted(by_state.items(), key=lambda kv: -kv[1]):
        md.append(f"| `{s}` | {n} |")
    md += [
        "",
        "## Full inventory",
        "",
        "| Model | Family | Task | Engine | License | Commercial | Gated | readiness_state | Visible |",
        "|---|---|---|---|---|:-:|:-:|---|:-:|",
    ]
    for r in sorted(rows, key=lambda r: (r["task"], r["model_id"])):
        md.append(
            f"| `{r['model_id']}` | {r['family']} | {r['task']} | {r['engine']} | "
            f"{r['license']} | {'yes' if r['commercial_safe'] else '—'} | "
            f"{'yes' if r['gated'] else '—'} | `{r['readiness_state']}` | "
            f"{'yes' if r['default_visible'] else '—'} |"
        )
    (QA_DIR / "discovered_models.md").write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
# Phase 3 — legal audit
# --------------------------------------------------------------------------- #
def write_legal(rows: list[dict]) -> None:
    import collections

    legal_rows = []
    for r in rows:
        legal_rows.append(
            {
                "model_id": r["model_id"],
                "license": r["license"],
                "license_class": r["license_class"],
                "commercial_safe": r["commercial_safe"],
                "gated": r["gated"],
                "requires_token": r["requires_token"],
                "readiness_state": r["readiness_state"],
                "runtime_allowed": r["readiness_state"]
                not in (tx.LICENSE_BLOCKED, tx.NON_COMMERCIAL_BLOCKED),
                "default_commercial_use": r["commercial_safe"],
            }
        )
    by_class = collections.Counter(r["license_class"] for r in rows)
    copyleft = [r["model_id"] for r in rows if r["license_class"] == "copyleft"]
    noncomm = [r["model_id"] for r in rows if r["license_class"] == "noncommercial"]
    gated = [r["model_id"] for r in rows if r["gated"]]
    unknown = [r["model_id"] for r in rows if r["license_class"] in ("custom_unknown", "unknown")]
    (QA_DIR / "legal_matrix.json").write_text(
        json.dumps(
            {
                "version": vsx.__version__,
                "license_class_counts": dict(by_class),
                "copyleft_blocked": copyleft,
                "noncommercial_blocked": noncomm,
                "gated_byot": gated,
                "unknown_or_custom": unknown,
                "rows": legal_rows,
            },
            indent=2,
        )
    )

    md = [
        "# VisionServeX Legal Model Audit (v3.18)",
        "",
        "Every model's license is gated before any runtime/training is enabled. This",
        "document is generated from `model_capabilities()` — see",
        "`docs/qa/v318_full_model_truth/legal_matrix.json` for the machine-readable form.",
        "",
        "## Hard rules enforced",
        "",
        "- **AGPL / GPL / SSPL** on a runtime/training path → `LICENSE_BLOCKED` (never default-safe).",
        "- **Non-commercial / research-only** → `NON_COMMERCIAL_BLOCKED` (never commercial-safe default).",
        "- **Gated** models → BYOT only: the user supplies their own token and accepts the",
        "  upstream license; VisionServeX never ships weights or tokens.",
        "- **Unknown / custom** license with no curated policy row → hidden pending review.",
        "- **No Ultralytics / AGPL import** on any runtime or training path (benchmark-only",
        "  comparison code is optional and never imported by the package runtime).",
        "",
        "## License class distribution",
        "",
        "| class | count |",
        "|---|---:|",
    ]
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        md.append(f"| {cls} | {n} |")
    md += [
        "",
        f"- **Copyleft (AGPL/GPL/SSPL) models:** {len(copyleft)} — {copyleft or 'none'}",
        f"- **Non-commercial models:** {len(noncomm)} — {noncomm or 'none'}",
        f"- **Gated (BYOT) models:** {len(gated)} — {gated or 'none'}",
        f"- **Unknown / custom license (hidden pending review):** {len(unknown)} — {unknown or 'none'}",
        "",
        "VisionServeX deliberately ships a **permissive-only** catalog: the registry",
        "contains no AGPL/GPL/SSPL and no non-commercial weights (e.g. Ultralytics YOLO",
        "and Deci YOLO-NAS are intentionally absent — LibreYOLO is the permissive",
        "replacement). The copyleft/non-commercial gates therefore correctly bind on an",
        "empty set today; the tests keep them binding forever.",
        "",
        "## Per-model legal status",
        "",
        "| Model | License | Class | Commercial-safe | Gated | Runtime allowed |",
        "|---|---|---|:-:|:-:|:-:|",
    ]
    for r in sorted(legal_rows, key=lambda r: r["model_id"]):
        md.append(
            f"| `{r['model_id']}` | {r['license']} | {r['license_class']} | "
            f"{'yes' if r['commercial_safe'] else '—'} | {'yes' if r['gated'] else '—'} | "
            f"{'yes' if r['runtime_allowed'] else 'NO'} |"
        )
    (DOCS / "legal_model_audit.md").write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
# Phase 9 — Anastig contract
# --------------------------------------------------------------------------- #
def write_anastig(rows: list[dict]) -> None:
    buckets: dict[str, list[str]] = {
        "train_ready_live": [],
        "inference_ready_live": [],
        "embedding_ready_live": [],
        "segmentation_ready_live": [],
        "open_vocab_ready_live": [],
        "gated_token_required": [],
        "hidden_catalog_only": [],
        "blocked": [],
        "license_blocked": [],
    }
    state_to_bucket = {
        tx.TRAIN_READY_LIVE: "train_ready_live",
        tx.INFERENCE_READY_LIVE: "inference_ready_live",
        tx.VLM_READY_LIVE: "inference_ready_live",
        tx.CORRESPONDENCE_READY_LIVE: "inference_ready_live",
        tx.EMBEDDING_READY_LIVE: "embedding_ready_live",
        tx.SEGMENTATION_READY_LIVE: "segmentation_ready_live",
        tx.OPEN_VOCAB_READY_LIVE: "open_vocab_ready_live",
        tx.GATED_TOKEN_REQUIRED: "gated_token_required",
        tx.CATALOG_ONLY_ENGINE_NOT_WIRED: "hidden_catalog_only",
        tx.CUSTOM_LOADER_REQUIRED: "hidden_catalog_only",
        tx.WEIGHTS_MISSING: "hidden_catalog_only",
        tx.LICENSE_BLOCKED: "license_blocked",
        tx.NON_COMMERCIAL_BLOCKED: "license_blocked",
    }
    for r in rows:
        st = r["readiness_state"]
        bucket = state_to_bucket.get(st, "blocked")
        buckets[bucket].append(r["model_id"])
    for k in buckets:
        buckets[k] = sorted(buckets[k])

    (DOCS / "anastig_model_allowlist_v318.json").write_text(
        json.dumps(
            {
                "version": vsx.__version__,
                "generated_from": "visionservex.model_capabilities (do not hand-edit)",
                **buckets,
            },
            indent=2,
        )
    )

    counts = {k: len(v) for k, v in buckets.items()}
    md = [
        "# Anastig Model Contract (v3.18)",
        "",
        f"Source of truth: `visionservex.model_capabilities(model_id)` (v{vsx.__version__}).",
        "Anastig drives its entire model UI from this contract — there is **no hardcoded",
        "model allowlist** required beyond optional product ranking. The machine-readable",
        "buckets live in `docs/anastig_model_allowlist_v318.json`.",
        "",
        "## How Anastig must consume each field",
        "",
        "Every model exposes `readiness_state` and `anastig_visibility`. Anastig switches",
        "on `anastig_visibility`:",
        "",
        "| anastig_visibility | Anastig behaviour |",
        "|---|---|",
        "| `show_train` | Show train **and** inference UI (state `TRAIN_READY_LIVE`). |",
        "| `show_inference` | Show inference UI (`INFERENCE_READY_LIVE` / `OPEN_VOCAB_READY_LIVE` / `VLM_READY_LIVE`). |",
        "| `show_embedding` | Show embedding UI (`EMBEDDING_READY_LIVE`). |",
        "| `show_segmentation` | Show segmentation UI (`SEGMENTATION_READY_LIVE`). |",
        "| `show_token_required` | Show BYOT/token UI (`GATED_TOKEN_REQUIRED`) — never run without the user's token. |",
        "| `hide` | Hide entirely (catalog-only, custom-loader, derived-not-live, partial, dependency/weights/crash/oom). |",
        "| `blocked_admin_only` | Hide from end users; visible to admins (legal-review, license-blocked, unknown). |",
        "",
        "**Hard rule:** only `*_READY_LIVE` states are usable-by-default. A",
        "`*_DERIVED_NEEDS_LIVE_CONFIRMATION` model is capability-derived and **not yet",
        "live-verified** — Anastig must keep it hidden until it is promoted to `*_LIVE`.",
        "",
        "## Bucket counts",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for k, n in counts.items():
        md.append(f"| `{k}` | {n} |")
    md += ["", "## Buckets", ""]
    for k, ids in buckets.items():
        md.append(f"### `{k}` ({len(ids)})")
        md.append("")
        md.append(", ".join(f"`{i}`" for i in ids) if ids else "_(none)_")
        md.append("")
    (DOCS / "anastig_model_contract_v318.md").write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
# Phase 5 — curated training matrix (top-level doc derived from the live matrix)
# --------------------------------------------------------------------------- #
def write_training_matrix(rows: list[dict]) -> None:
    train_matrix = QA_DIR / "live_train_lifecycle_matrix.json"
    live = {}
    if train_matrix.exists():
        for r in json.loads(train_matrix.read_text()).get("results", []):
            live[r["model_id"]] = r

    train_rows = [r for r in rows if model_capabilities(r["model_id"])["train_supported"]]
    md = [
        "# VisionServeX Training Matrix (v3.18)",
        "",
        "Live train lifecycle: **train → checkpoint → reload → predict-after-reload →",
        "schema → export**. A model is `TRAIN_READY_LIVE` only when every stage passed",
        "this sprint; otherwise it is honestly `TRAIN_READY_DERIVED` (e.g. RF-DETR,",
        "whose native COCO trainer is too heavy for a CPU smoke and is not faked).",
        "",
        "| Model | Family | readiness_state | Train | Ckpt | Reload | Predict | Export | live_train |",
        "|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]

    def y(b):
        return "yes" if b else "—"

    for r in sorted(train_rows, key=lambda r: (r["family"], r["model_id"])):
        mid = r["model_id"]
        cap = model_capabilities(mid)
        lr = live.get(mid, {})
        md.append(
            f"| `{mid}` | {r['family']} | `{cap['readiness_state']}` | "
            f"{y(lr.get('train'))} | {y(lr.get('checkpoint_path_exists'))} | "
            f"{y(lr.get('reload'))} | {y(lr.get('predict_after_reload'))} | "
            f"{y(lr.get('export'))} | {y(cap['live_verified_train'])} |"
        )
    n_live = sum(1 for r in train_rows if model_capabilities(r["model_id"])["live_verified_train"])
    md += [
        "",
        f"**{n_live} `TRAIN_READY_LIVE`** of {len(train_rows)} train-supported models. "
        "Full evidence: `docs/qa/v318_full_model_truth/live_train_lifecycle_matrix.json`.",
    ]
    (DOCS / "training_matrix.md").write_text("\n".join(md) + "\n")


def main() -> int:
    rows = collect()
    write_inventory(rows)
    write_legal(rows)
    write_anastig(rows)
    write_training_matrix(rows)
    print(f"[v318] wrote inventory + legal + anastig + training artifacts for {len(rows)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
