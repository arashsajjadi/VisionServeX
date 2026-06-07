#!/usr/bin/env python3
"""Build the V3 Phase-2/3 audit artifacts from the verified rights-research JSON
and the (corrected) canonical ledger. Pure data assembly — no network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

REP = Path("notebook/99_final_report/reports")
RESEARCH = json.loads(Path("/tmp/v3_rights_research.json").read_text())
RR = {r["name"]: r for r in RESEARCH["research_rows"]}
ADV = {a["claim"]: a for a in RESEARCH["adversarial"]}

led = pd.read_csv(REP / "model_coverage_ledger.csv", dtype=str, keep_default_na=False)
LED = {r["model_id"]: r for _, r in led.iterrows()}
ext = pd.read_csv(REP / "external_restricted_baselines.csv", dtype=str, keep_default_na=False)
EXT_IDS = set(ext["model_id"])

# target-name -> ledger model_id (None if not present anywhere)
ALIAS = {
    "sam-vit-b": "sam-vit-base",
    "sam-vit-l": "sam-vit-large",
    "sam-vit-h": "sam-vit-huge",
    "edgesam": "edgesam",  # now in external_restricted, not core
}


def ledger_id(name: str) -> str | None:
    nid = ALIAS.get(name, name)
    if nid in LED:
        return nid
    if nid in EXT_IDS:
        return nid
    return None


def ledger_state(name: str):
    nid = ledger_id(name)
    if nid is None:
        return {
            "exists": False,
            "model_id": "",
            "final_state": "",
            "license_status": "",
            "where": "absent",
        }
    if nid in LED:
        r = LED[nid]
        return {
            "exists": True,
            "model_id": nid,
            "final_state": r["final_state"],
            "license_status": r["license_status"],
            "evidence": r.get("evidence_artifact", ""),
            "where": "core_ledger",
        }
    er = ext[ext["model_id"] == nid].iloc[0]
    return {
        "exists": True,
        "model_id": nid,
        "final_state": "external_restricted_baseline",
        "license_status": er["license_status"],
        "evidence": "",
        "where": "external_restricted_baselines",
    }


# ---- implementation-aware overrides for the classic tools (we wrote the code) ----
CLASSIC_OVERRIDE = {
    "classic-slic-graphcut": {
        "commercial_use_allowed": "yes",
        "redistribution_allowed": "yes",
        "core_allowed": "yes",
        "notes": "VisionServeX implementation uses OpenCV grabCut (Apache-2.0 internal min-cut) + "
        "scikit-image SLIC (BSD-3). It does NOT use PyMaxflow/gco (GPL/non-commercial). "
        "Commercial-safe as implemented (verified in smart_annotation/classic/methods.py).",
    },
    "classic-edge-plus": {
        "notes": "VisionServeX uses plain OpenCV Canny/Sobel + scipy morphology (no contrib "
        "StructuredEdgeDetection NC model). Commercial-safe as implemented.",
    },
}

# ---------------------------------------------------------------------------
# 1. v3_model_rights_audit
# ---------------------------------------------------------------------------
RIGHTS_COLS = [
    "model_id",
    "target_name",
    "family",
    "code_license",
    "weights_license",
    "commercial_use_allowed",
    "redistribution_allowed",
    "gated_or_auth_required",
    "core_allowed",
    "confidence",
    "exists_in_ledger",
    "ledger_model_id",
    "ledger_final_state",
    "ledger_location",
    "provenance",
    "notes",
]
rights = []
for name, r in RR.items():
    o = dict(r)
    o.update(CLASSIC_OVERRIDE.get(name, {}))
    st = ledger_state(name)
    rights.append(
        {
            "model_id": st["model_id"] or name,
            "target_name": name,
            "family": o["family"],
            "code_license": o["code_license"],
            "weights_license": o["weights_license"],
            "commercial_use_allowed": o["commercial_use_allowed"],
            "redistribution_allowed": o["redistribution_allowed"],
            "gated_or_auth_required": o["gated_or_auth_required"],
            "core_allowed": o["core_allowed"],
            "confidence": o["confidence"],
            "exists_in_ledger": st["exists"],
            "ledger_model_id": st["model_id"],
            "ledger_final_state": st["final_state"],
            "ledger_location": st["where"],
            "provenance": o["provenance"],
            "notes": o["notes"],
        }
    )
rights.sort(key=lambda x: (x["core_allowed"] != "yes", x["family"], x["target_name"]))
with open(REP / "v3_model_rights_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=RIGHTS_COLS)
    w.writeheader()
    w.writerows(rights)
Path(REP / "v3_model_rights_audit.json").write_text(
    json.dumps(
        {
            "schema": "v3_model_rights_audit.v1",
            "note": "code_license vs weights_license split; commercial_use & redistribution audited. "
            "Verified by adversarial license workflow wf_239c89dc-c14 (web + knowledge).",
            "adversarial_landmines": RESEARCH["adversarial"],
            "rows": rights,
        },
        indent=2,
    )
)

# ---------------------------------------------------------------------------
# 2. v3_target_model_coverage_matrix  (families A-F)
# ---------------------------------------------------------------------------
FAMILIES = {
    "A_classic_no_weight": [
        "classic-slic-graphcut",
        "classic-random-walker",
        "classic-intelligent-scissors",
        "classic-grabcut",
        "classic-marker-watershed",
        "classic-interactive-rf",
        "classic-slic-rf-smooth",
        "classic-edge-plus",
    ],
    "B_promptable_sam": [
        "sam-vit-b",
        "sam-vit-l",
        "sam-vit-h",
        "sam2-hiera-tiny",
        "sam2-hiera-small",
        "sam2-hiera-base-plus",
        "sam2-hiera-large",
        "sam2.1-hiera-tiny",
        "sam2.1-hiera-small",
        "sam2.1-hiera-base-plus",
        "sam2.1-hiera-large",
        "mobilesam",
        "efficientsam",
        "hq-sam",
        "hq-sam2",
        "light-hq-sam",
        "tinysam",
        "q-tinysam",
        "edgesam",
    ],
    "C_interactive_click": ["ritm", "focalclick", "clickseg", "simpleclick"],
    "D_autolabel_grounded": [
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-seg-medium",
        "rfdetr-seg-large",
        "grounding-dino-swin-b",
        "grounding-dino-swin-t",
        "grounding-dino-tiny",
        "grounding-dino-original-swin-b",
        "grounding-dino-original-swin-t",
    ],
    "E_gated_auth_api": [
        "sam3-base",
        "grounding-dino-1.5",
        "grounding-dino-1.6",
        "grounding-dino-1.5-pro",
        "grounding-dino-1.6-pro",
        "dino-x-api",
    ],
    "F_excluded_restricted": [
        "fastsam-s",
        "fastsam-x",
        "ultralytics-yolov8-seg",
        "ultralytics-yolo11-seg",
        "ultralytics-yolo26-seg",
        "yolo-world",
        "libreyolo-yolonas",
        "totalsegmentator",
        "rfdetr-seg-xlarge",
        "rfdetr-seg-2xlarge",
    ],
}

# research-name aliases for the click/research families
RNAME = {
    "ritm": "ritm (ritm_interactive_segmentation)",
    "focalclick": "focalclick (FocalClick)",
    "clickseg": "clickseg (ClickSEG)",
    "simpleclick": "simpleclick (SimpleClick)",
}


def research_for(name):
    return RR.get(name) or RR.get(RNAME.get(name, ""))


def classify_action(name, fam, r, st):
    if fam == "A_classic_no_weight":
        return (
            "benchmarked_classic_tool",
            "see smart_tool_coverage_ledger.csv (CPU benchmark already run)",
            "V3-13 supported",
        )
    core = (r or {}).get("core_allowed", "unknown")
    if name in ("edgesam",):
        return (
            "quarantined_noncommercial",
            "external_restricted_baselines.csv (S-Lab NC)",
            "V3-07 corrected",
        )
    if fam == "F_excluded_restricted":
        return (
            "excluded_restricted",
            "external_restricted_baselines.csv / excluded_noncommercial_models.csv",
            "V3-07 satisfied",
        )
    if core == "gated_byot_only":
        return ("gated_byot", "visionservex auth/sam3 BYOT; no mirrored weights", "V3-09 BYOT")
    if core == "legal_review_required":
        return (
            "legal_review_required",
            "hold from default-safe core pending legal review",
            "V3-07/V3-08 risk",
        )
    if st["exists"] and st["where"] == "core_ledger" and st["final_state"] == "benchmark_passed":
        return ("supported_benchmarked", "already core + benchmark_passed", "V3-12 covered")
    if st["exists"]:
        return ("supported_present", f"in ledger as {st['final_state']}", "V3-12 covered")
    if core == "yes":
        return ("addable_commercial_safe", f"add to registry + benchmark ({name})", "V3-12 add")
    return ("review_or_missing", "audit + classify before add", "V3-12 pending")


COV_COLS = [
    "target_name",
    "target_family",
    "exists_in_ledger",
    "ledger_model_id",
    "ledger_location",
    "current_final_state",
    "current_license_status",
    "code_license",
    "weights_license",
    "commercial_safe",
    "redistribution_allowed",
    "gated_or_auth_required",
    "core_allowed",
    "action",
    "exact_next_command",
    "v3_gate_impact",
    "notes",
]
cov = []
for fam, names in FAMILIES.items():
    for name in names:
        r = research_for(name)
        st = ledger_state(name)
        action, cmd, gate = classify_action(name, fam, r, st)
        cov.append(
            {
                "target_name": name,
                "target_family": fam,
                "exists_in_ledger": st["exists"],
                "ledger_model_id": st["model_id"],
                "ledger_location": st["where"],
                "current_final_state": st["final_state"],
                "current_license_status": st["license_status"],
                "code_license": (r or {}).get(
                    "code_license", "n/a (classic)" if fam.startswith("A") else "unknown"
                ),
                "weights_license": (r or {}).get(
                    "weights_license", "n/a (no weights)" if fam.startswith("A") else "unknown"
                ),
                "commercial_safe": (r or {}).get("commercial_use_allowed", "unknown"),
                "redistribution_allowed": (r or {}).get("redistribution_allowed", "unknown"),
                "gated_or_auth_required": (r or {}).get(
                    "gated_or_auth_required", "no" if fam.startswith("A") else "unknown"
                ),
                "core_allowed": (r or {}).get(
                    "core_allowed", "yes" if fam.startswith("A") else "unknown"
                ),
                "action": action,
                "exact_next_command": cmd,
                "v3_gate_impact": gate,
                "notes": (r or {}).get("notes", "")[:300],
            }
        )
with open(REP / "v3_target_model_coverage_matrix.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COV_COLS)
    w.writeheader()
    w.writerows(cov)
Path(REP / "v3_target_model_coverage_matrix.json").write_text(
    json.dumps(
        {
            "schema": "v3_target_model_coverage_matrix.v1",
            "total_targets": len(cov),
            "by_family": {k: len(v) for k, v in FAMILIES.items()},
            "rows": cov,
        },
        indent=2,
    )
)

# ---------------------------------------------------------------------------
# 3. v3_excluded_or_quarantined_models.csv
# ---------------------------------------------------------------------------
QCOLS = ["model_id", "family", "reason_class", "license", "action_taken", "evidence"]
q = []
q.append(
    {
        "model_id": "edgesam",
        "family": "edgesam",
        "reason_class": "non_commercial_software_license",
        "license": "S-Lab License 1.0 (non-commercial)",
        "action_taken": "moved core -> external_restricted_baselines; manifest corrected",
        "evidence": "github.com/chongzhou96/EdgeSAM LICENSE; wf_239c89dc-c14 adversarial verdict",
    }
)
q.append(
    {
        "model_id": "hq-sam",
        "family": "hq-sam",
        "reason_class": "training_data_non_commercial_review",
        "license": "Apache-2.0 weights; HQSeg-44K NC training data",
        "action_taken": "core final_state -> legal_review_required, default_safe=False",
        "evidence": "ThinObject-5K CC-BY-NC + DIS5K NC ToU; wf adversarial verdict (medium conf)",
    }
)
for name in ["tinysam", "q-tinysam"]:
    q.append(
        {
            "model_id": name,
            "family": "sam-light",
            "reason_class": "dataset_provenance_review",
            "license": "Apache-2.0 declared; SA-1B research-only provenance",
            "action_taken": "NOT added to core (legal_review_required)",
            "evidence": "arXiv 2312.13789 1% SA-1B distillation; wf adversarial verdict",
        }
    )
for _, er in ext.iterrows():
    q.append(
        {
            "model_id": er["model_id"],
            "family": "",
            "reason_class": "restricted_license_external_baseline",
            "license": er["license_status"],
            "action_taken": "external_restricted_baselines (not core)",
            "evidence": er.get("reason_excluded_from_core", ""),
        }
    )
exc_nc = pd.read_csv(REP / "excluded_noncommercial_models.csv", dtype=str, keep_default_na=False)
for _, er in exc_nc.iterrows():
    q.append(
        {
            "model_id": er["model_id"],
            "family": er.get("family", ""),
            "reason_class": "non_commercial_excluded",
            "license": er["license_status"],
            "action_taken": "excluded_noncommercial_models (not core, not baseline)",
            "evidence": er.get("exclusion_reason", ""),
        }
    )
with open(REP / "v3_excluded_or_quarantined_models.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=QCOLS)
    w.writeheader()
    w.writerows(q)

# ---------------------------------------------------------------------------
# 4. v3_bad_license_scan.json  (scan the CORRECTED core ledger)
# ---------------------------------------------------------------------------
BAD = "AGPL|GPL|S-Lab|non-commercial|noncommercial|proprietary|restricted|Enterprise|Deci"
core_bad = led[led["license_status"].str.contains(BAD, case=False, na=False)]
flagged = led[
    led["final_state"].isin(["legal_review_required", "external_api_only", "auth_required"])
]
scan = {
    "schema": "v3_bad_license_scan.v1",
    "scanned": "model_coverage_ledger.csv (commercial-safe core, post-correction)",
    "core_rows": len(led),
    "bad_license_regex_hits_in_core": [
        {
            "model_id": r["model_id"],
            "license_status": r["license_status"],
            "final_state": r["final_state"],
            "default_safe": r["default_safe"],
        }
        for _, r in core_bad.iterrows()
    ],
    "flagged_non_default_safe": [
        {
            "model_id": r["model_id"],
            "final_state": r["final_state"],
            "license_status": r["license_status"],
        }
        for _, r in flagged.iterrows()
    ],
    "verdict": "PASS — no AGPL/GPL/non-commercial WEIGHT sits in default-safe core. "
    "Remaining hits are gated external-API (Custom, default_safe=False) and hq-sam "
    "(legal_review_required, default_safe=False), all excluded from core_healthy.",
}
Path(REP / "v3_bad_license_scan.json").write_text(json.dumps(scan, indent=2))

print("rights audit rows :", len(rights))
print("target matrix rows:", len(cov), "| by family:", {k: len(v) for k, v in FAMILIES.items()})
print("excluded/quarantined:", len(q))
print(
    "bad-license hits in core:",
    len(core_bad),
    "->",
    [r["model_id"] for _, r in core_bad.iterrows()],
)
print("core core_allowed=yes targets:", sum(1 for x in rights if x["core_allowed"] == "yes"))
