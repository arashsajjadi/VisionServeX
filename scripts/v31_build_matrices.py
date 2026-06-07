#!/usr/bin/env python3
"""Build v3.1 baseline + SAM/DINO family matrices + SAM+DINO pipeline ledger.

Every prompt target gets a row with an HONEST state and an exact next command.
Rows present in the canonical ledger inherit that state; the rest are classified
from known licensing facts (refined by the parallel research workflow). No row
is omitted; nothing is marked benchmark_passed without ledger evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

R = Path("notebook/99_final_report/reports")
led = pd.read_csv(R / "model_coverage_ledger.csv", dtype=str, keep_default_na=False)
LED = {r["model_id"]: r for _, r in led.iterrows()}

# alias: prompt target -> ledger model_id
ALIAS = {
    "sam-vit-b": "sam-vit-base", "sam-vit-l": "sam-vit-large", "sam-vit-h": "sam-vit-huge",
    "hq-sam-vit-b": "hq-sam", "hq-sam-vit-l": "hq-sam", "hq-sam-vit-h": "hq-sam",
    "medsam-vit-b": "medsam", "edge-sam": "edgesam",
    "sam2-image-tiny": "sam2-hiera-tiny", "sam2-image-small": "sam2-hiera-small",
    "sam2-image-base-plus": "sam2-hiera-base-plus", "sam2-image-large": "sam2-hiera-large",
    "sam2.1-image-tiny": "sam2.1-hiera-tiny", "sam2.1-image-small": "sam2.1-hiera-small",
    "sam2.1-image-base-plus": "sam2.1-hiera-base-plus", "sam2.1-image-large": "sam2.1-hiera-large",
    "efficientsam-tiny": "efficientsam", "efficientsam-small": "efficientsam",
}


def ledger_state(mid: str) -> str:
    nid = ALIAS.get(mid, mid)
    r = LED.get(nid)
    if r is not None:
        return r["final_state"]
    # external_restricted has edgesam
    return ""


# ---------------------------------------------------------------------------
# Honest classification facts (code/weights license, availability, state)
# Refined from the V3 rights audit + adversarial research.
# ---------------------------------------------------------------------------
# state legend: benchmark_passed (ledger), demo_passed_sidecar, user_checkpoint_required,
# checkpoint_required, auth_required, external_api_only, legal_review_required,
# excluded_restricted, dataset_required, sidecar_required, not_released
FACT = {
    # --- SAM1 ---
    "sam-vit-b": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam-vit-l": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam-vit-h": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    # ONNX variants: LOCAL export from Apache-2.0 weights is license-clean
    "sam-vit-b-onnx": ("Apache-2.0", "Apache-2.0 (local ONNX export)", "local_export", "yes"),
    "sam-vit-l-onnx": ("Apache-2.0", "Apache-2.0 (local ONNX export)", "local_export", "yes"),
    "sam-vit-h-onnx": ("Apache-2.0", "Apache-2.0 (local ONNX export)", "local_export", "yes"),
    # --- HQ-SAM (HQSeg-44K NC training data) ---
    "hq-sam": ("Apache-2.0", "Apache-2.0 (declared); HQSeg-44K NC training data", "public_download", "conditional"),
    "hq-sam-vit-b": ("Apache-2.0", "HQSeg-44K NC training data", "public_download", "conditional"),
    "hq-sam-vit-l": ("Apache-2.0", "HQSeg-44K NC training data", "public_download", "conditional"),
    "hq-sam-vit-h": ("Apache-2.0", "HQSeg-44K NC training data", "public_download", "conditional"),
    "hq-sam2": ("Apache-2.0", "HQSeg-44K NC training data", "public_download", "conditional"),
    "light-hq-sam": ("Apache-2.0", "HQSeg-44K NC training data", "public_download", "conditional"),
    # --- MedSAM ---
    "medsam": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "medsam-vit-b": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "medsam2": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    # --- SAM2 / SAM2.1 (Apache-2.0 weights, public) ---
    "sam2-hiera-tiny": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2-hiera-small": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2-hiera-base-plus": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2-hiera-large": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2.1-hiera-tiny": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2.1-hiera-small": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2.1-hiera-base-plus": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "sam2.1-hiera-large": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    # --- Lightweight ---
    "mobilesam": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "mobilesam-onnx": ("Apache-2.0", "Apache-2.0 (local ONNX export)", "local_export", "yes"),
    "efficientsam": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "efficientsam-onnx": ("Apache-2.0", "Apache-2.0 (local ONNX export)", "local_export", "yes"),
    "tinysam": ("Apache-2.0", "Apache-2.0 tag; SA-1B research-only provenance", "public_download", "conditional"),
    "q-tinysam": ("Apache-2.0", "Apache-2.0 tag; SA-1B research-only provenance", "public_download", "conditional"),
    "edge-sam": ("S-Lab License 1.0", "S-Lab License 1.0 (NON-COMMERCIAL)", "public_download", "no"),
    # --- SAM3 (custom Meta SAM License, HF gated) ---
    "sam3-base": ("SAM License (Meta custom)", "SAM License (Meta custom, gated)", "hf_gated", "conditional"),
    # --- DINO embeddings ---
    "dinov2-small": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "dinov2-base": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "dinov2-large": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "dinov2-giant": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "dinov3-vitb16": ("DINOv3 License (Meta custom)", "DINOv3 License (Meta custom, gated)", "hf_gated", "conditional"),
    # --- GroundingDINO ---
    "grounding-dino-swin-t": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "grounding-dino-swin-b": ("Apache-2.0", "Apache-2.0", "public_download", "yes"),
    "grounding-dino-1.5": ("Apache-2.0 (client)", "API/gated", "api_only", "conditional"),
    "grounding-dino-1.6": ("Apache-2.0 (client)", "API/gated", "api_only", "conditional"),
    "grounding-dino-1.5-pro": ("Apache-2.0 (client)", "proprietary/closed", "api_only", "conditional"),
    "grounding-dino-1.6-pro": ("Apache-2.0 (client)", "proprietary/closed", "api_only", "conditional"),
    "dino-x-api": ("Apache-2.0 (client)", "proprietary/closed", "api_only", "conditional"),
}

# default state when not in ledger, derived from FACT
STATE_FROM_AVAIL = {
    "public_download": "checkpoint_candidate",
    "local_export": "checkpoint_required",  # needs local export step
    "hf_gated": "auth_required",
    "api_only": "external_api_only",
    "unavailable": "not_released",
    "unknown": "legal_review_required",
}


def classify(mid: str) -> dict:
    cur = ledger_state(mid)
    f = FACT.get(mid)
    code_l, weights_l, avail, comm = f if f else ("unknown", "unknown", "unknown", "unknown")
    # commercial-safety overrides
    if comm == "no" or "NON-COMMERCIAL" in weights_l or "S-Lab" in weights_l:
        target = "excluded_restricted"
    elif comm == "conditional" and ("SA-1B" in weights_l or "HQSeg-44K" in weights_l):
        target = "legal_review_required"
    elif avail == "hf_gated":
        target = "auth_required"
    elif avail == "api_only":
        target = "external_api_only"
    elif avail == "unavailable":
        target = "not_released"
    elif cur in ("benchmark_passed", "wired", "demo_passed_sidecar", "micro_benchmark_passed"):
        target = cur  # already activated
    elif avail == "local_export":
        target = "checkpoint_required"  # ONNX local export variant
    elif cur:
        target = cur
    else:
        target = STATE_FROM_AVAIL.get(avail, "legal_review_required")
    return {"current_state_before": cur or "absent_from_ledger", "target_state_after": target,
            "code_license": code_l, "weights_license": weights_l, "weights_available": avail,
            "commercial_use_allowed": comm}


# ---------------------------------------------------------------------------
# Target lists (from the sprint prompt) — every row classified, none omitted.
# ---------------------------------------------------------------------------
SAM_TARGETS = {
    "SAM1": ["sam-vit-b", "sam-vit-l", "sam-vit-h", "sam-vit-b-onnx", "sam-vit-l-onnx", "sam-vit-h-onnx"],
    "SAM1-HQ/medical": ["hq-sam", "hq-sam-vit-b", "hq-sam-vit-l", "hq-sam-vit-h", "medsam", "medsam-vit-b"],
    "SAM2": ["sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large",
             "sam2-image-tiny", "sam2-image-small", "sam2-image-base-plus", "sam2-image-large",
             "sam2-video-tiny", "sam2-video-small", "sam2-video-base-plus", "sam2-video-large"],
    "SAM2.1": ["sam2.1-hiera-tiny", "sam2.1-hiera-small", "sam2.1-hiera-base-plus", "sam2.1-hiera-large",
               "sam2.1-image-tiny", "sam2.1-image-small", "sam2.1-image-base-plus", "sam2.1-image-large",
               "sam2.1-video-tiny", "sam2.1-video-small", "sam2.1-video-base-plus", "sam2.1-video-large",
               "sam2.1-onnx-tiny", "sam2.1-onnx-small", "sam2.1-onnx-base-plus", "sam2.1-onnx-large"],
    "SAM3": ["sam3-base", "sam3-image", "sam3-video", "sam3-text-prompt", "sam3-visual-prompt",
             "sam3-exemplar-prompt", "sam3-open-vocabulary", "sam3-tracking"],
    "SAM3.1": ["sam3.1-base", "sam3.1-image", "sam3.1-video", "sam3.1-real-time-tracking",
               "sam3.1-text-prompt", "sam3.1-visual-prompt", "sam3.1-open-vocabulary", "sam3.1-api-or-byot"],
    "lightweight": ["mobilesam", "mobilesam-onnx", "efficientsam", "efficientsam-tiny", "efficientsam-small",
                    "efficientsam-onnx", "tinysam", "q-tinysam", "light-hq-sam", "hq-sam2", "medsam2", "edge-sam"],
}

DINO_TARGETS = {
    "DINO/DINOv2": ["dino-vits8", "dino-vits16", "dino-vitb8", "dino-vitb16",
                    "dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"],
    "DINOv3": ["dinov3-vits16", "dinov3-vitb16", "dinov3-vitl16", "dinov3-vit7b16",
               "dinov3-convnext-tiny", "dinov3-convnext-small", "dinov3-convnext-base", "dinov3-convnext-large"],
    "GroundingDINO": ["grounding-dino-swin-t", "grounding-dino-swin-b", "grounding-dino-original-swin-t",
                      "grounding-dino-original-swin-b", "grounding-dino-1.5", "grounding-dino-1.6",
                      "grounding-dino-1.5-pro", "grounding-dino-1.6-pro"],
    "DINO-X": ["dino-x-api", "dino-x-detection", "dino-x-segmentation", "dino-x-phrase-grounding",
               "dino-x-counting", "dino-x-region-captioning"],
    "DETR/DINO-detect": ["deimv2-atto", "deimv2-n", "deimv2-s", "deimv2-m", "deimv2-l",
                         "dfine-n", "dfine-s", "dfine-m", "dfine-l", "dfine-x",
                         "rtdetrv4-s", "rtdetrv4-m", "rtdetrv4-l", "rtdetrv4-x"],
}

# not-yet-released honesty: sam3.1 + sam3 capability sub-variants not separately published
NOT_RELEASED = {
    "sam3-image", "sam3-video", "sam3-text-prompt", "sam3-visual-prompt", "sam3-exemplar-prompt",
    "sam3-open-vocabulary", "sam3-tracking",
    "sam3.1-base", "sam3.1-image", "sam3.1-video", "sam3.1-real-time-tracking", "sam3.1-text-prompt",
    "sam3.1-visual-prompt", "sam3.1-open-vocabulary", "sam3.1-api-or-byot",
    "dino-x-detection", "dino-x-segmentation", "dino-x-phrase-grounding", "dino-x-counting",
    "dino-x-region-captioning",
}


def next_command(mid: str, target: str) -> str:
    if target == "auth_required":
        return f"export HF_TOKEN=... && visionservex sam status {mid}  # request HF gated access first"
    if target == "external_api_only":
        return f"export DEEPDATASPACE_API_KEY=... && visionservex dino api {mid} image.jpg --text '...'"
    if target == "excluded_restricted":
        return f"# {mid} is non-commercial — external baseline only; do not add to commercial-safe core"
    if target == "legal_review_required":
        return f"visionservex legal review {mid}  # weights/training-data provenance unresolved"
    if target == "not_released":
        return f"# {mid}: no separately published checkpoint as of 2026-06; track upstream release"
    if target == "checkpoint_required" and "onnx" in mid:
        return f"visionservex sam export-onnx {mid.replace('-onnx','')} --out models/{mid}.onnx"
    if "video" in mid:
        return f"visionservex sam video {mid.split('-video')[0]+'-hiera'+mid.split('-video')[1] if False else mid} video.mp4 --box ..."
    return f"visionservex sam run {mid} image.jpg --box 10,20,200,220 --out runs/{mid}"


def build_matrix(targets: dict, fam_label: str) -> list:
    rows = []
    for gen, mids in targets.items():
        for mid in mids:
            if mid in NOT_RELEASED:
                c = {"current_state_before": "absent_from_ledger", "target_state_after": "not_released",
                     "code_license": "n/a (unreleased)", "weights_license": "n/a (unreleased)",
                     "weights_available": "unavailable", "commercial_use_allowed": "unknown"}
            else:
                c = classify(mid)
            rows.append({
                "family": fam_label, "generation": gen, "model_id": mid,
                **c,
                "evidence_artifact": (LED.get(ALIAS.get(mid, mid), {}).get("evidence_artifact", "")
                                      if isinstance(LED.get(ALIAS.get(mid, mid)), dict) else ""),
                "tutorial_notebook": f"notebook/tutorials/{fam_label.lower()}_family/{mid}.ipynb",
                "exact_user_command": next_command(mid, c["target_state_after"]),
                "blocker_if_any": ("non-commercial license" if c["target_state_after"] == "excluded_restricted"
                                   else "HF gated access" if c["target_state_after"] == "auth_required"
                                   else "API-only" if c["target_state_after"] == "external_api_only"
                                   else "not yet released" if c["target_state_after"] == "not_released"
                                   else "training-data provenance" if c["target_state_after"] == "legal_review_required"
                                   else "ONNX local-export step pending" if c["target_state_after"] == "checkpoint_required"
                                   else ""),
            })
    return rows


sam_rows = build_matrix(SAM_TARGETS, "SAM")
dino_rows = build_matrix(DINO_TARGETS, "DINO")

for rows, name in [(sam_rows, "v31_sam_family_matrix"), (dino_rows, "v31_dino_family_matrix")]:
    cols = list(rows[0].keys())
    with open(R / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # md
    from collections import Counter
    dist = Counter(r["target_state_after"] for r in rows)
    L = [f"# {name}\n", f"Total rows: {len(rows)}\n", "State distribution: " + ", ".join(f"{k}={v}" for k, v in dist.items()) + "\n",
         "| model_id | generation | before | after | weights_license | next_command |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model_id']} | {r['generation']} | {r['current_state_before']} | {r['target_state_after']} | {r['weights_license'][:30]} | {r['exact_user_command'][:50]} |")
    (R / f"{name}.md").write_text("\n".join(L))

# --- SAM+DINO pipeline ledger ---
PIPELINES = [
    "grounding-dino-swin-t+sam-vit-h", "grounding-dino-swin-b+sam-vit-h",
    "grounding-dino-swin-t+sam2.1-hiera-small", "grounding-dino-swin-b+sam2.1-hiera-large",
    "grounding-dino-original-swin-t+sam2-hiera-small", "grounding-dino-original-swin-b+sam2-hiera-large",
    "grounding-dino-1.5+sam3-base", "grounding-dino-1.6+sam3-base", "dino-x-api+sam3-base",
    "dinov3-vitb16+sam2.1-hiera-small", "dinov3-vitb16+sam3-base",
]
pipe_rows = []
for pid in PIPELINES:
    det, seg = pid.split("+")
    ds, ss = ledger_state(det), ledger_state(seg)
    det_safe = ds == "benchmark_passed"
    seg_safe = ss == "benchmark_passed"
    if det in ("dino-x-api",) or "1.5" in det or "1.6" in det:
        state = "auth_required" if "sam3" in seg else "external_api_only"
    elif "sam3" in seg:
        state = "auth_required"  # sam3 gated
    elif "dinov3" in det:
        state = "legal_review_required"  # dinov3 custom license
    elif det_safe and seg_safe:
        state = "pipeline_demo_ready"  # both parts runnable + commercial-safe
    else:
        state = "blocked_on_part"
    pipe_rows.append({
        "pipeline_id": pid, "detector": det, "detector_state": ds or "absent",
        "segmenter": seg or ("auth_required" if seg == "sam3-base" else "absent"),
        "segmenter_state": ss or ("auth_required" if seg == "sam3-base" else "absent"),
        "both_parts_commercial_safe": det_safe and seg_safe,
        "pipeline_state": state, "is_single_model": False,
        "exact_user_command": f"visionservex pipeline run {pid} image.jpg --text 'defect' --out runs/{pid.replace('+','_')}",
        "tutorial_notebook": f"notebook/tutorials/pipelines/{pid.replace('+','_')}.ipynb",
    })
with open(R / "v31_sam_dino_pipeline_ledger.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pipe_rows[0].keys()))
    w.writeheader()
    w.writerows(pipe_rows)

# --- baseline ---
fs = led["final_state"].value_counts().to_dict()
baseline = {
    "version_before": "v3.0.0", "sprint": "v3.1 SAM/DINO model expansion",
    "core_rows": len(led), "benchmark_passed": int((led["final_state"] == "benchmark_passed").sum()),
    "final_state_counts": {k: int(v) for k, v in fs.items()},
    "sam_targets_total": len(sam_rows), "dino_targets_total": len(dino_rows),
    "pipelines_total": len(pipe_rows),
    "sam_state_dist": dict(__import__("collections").Counter(r["target_state_after"] for r in sam_rows)),
    "dino_state_dist": dict(__import__("collections").Counter(r["target_state_after"] for r in dino_rows)),
    "activation_kpi_target": 20,
}
(R / "v31_model_expansion_baseline.json").write_text(json.dumps(baseline, indent=2))
(R / "v31_model_expansion_baseline.md").write_text(
    "# v3.1 SAM/DINO Model Expansion — Baseline\n\n"
    f"- Version before: v3.0.0\n- Core rows: {len(led)} | benchmark_passed: {baseline['benchmark_passed']}\n"
    f"- SAM targets classified: {len(sam_rows)} | DINO targets: {len(dino_rows)} | pipelines: {len(pipe_rows)}\n"
    f"- SAM state dist: {baseline['sam_state_dist']}\n- DINO state dist: {baseline['dino_state_dist']}\n"
    f"- Activation KPI target: 20 new activations across models/tools/pipelines\n")

print("SAM rows:", len(sam_rows), "| DINO rows:", len(dino_rows), "| pipelines:", len(pipe_rows))
print("SAM state dist:", baseline["sam_state_dist"])
print("DINO state dist:", baseline["dino_state_dist"])
print("pipeline states:", __import__("collections").Counter(r["pipeline_state"] for r in pipe_rows))
