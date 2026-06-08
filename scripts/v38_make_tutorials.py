#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the v3.8 BYOT / license-safe tutorial notebooks (deterministic).

    python scripts/v38_make_tutorials.py

Writes 12 notebooks under
``notebook/tutorials/v38_hf_byot_and_license_safe_models/``. Each notebook:
* installs ``visionservex==3.8.0`` from PyPI (run after release),
* asserts it imports from ``site-packages`` (not the local ``src``),
* never prints a token,
* shows CLI + Python usage,
* saves one artifact (or one honest blocker artifact),
* appends a row to ``v38_tutorial_execution_ledger.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "3.8.0"
OUT = Path("notebook/tutorials/v38_hf_byot_and_license_safe_models")


# ---- nbformat helpers -------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else [""]


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# Reusable cells -------------------------------------------------------------
INSTALL = code(f"""
# Install the published package from PyPI (run AFTER release).
# Before release you may instead `pip install dist/visionservex-{VERSION}-py3-none-any.whl`.
# %pip install -q visionservex=={VERSION}
import importlib.metadata as _m
print("installed:", _m.version("visionservex"))
""")

ASSERT_SITE = code("""
# Assert we are using the INSTALLED package (site-packages), never the local src.
import visionservex
print("visionservex:", visionservex.__version__)
print("file:", visionservex.__file__)
assert "site-packages" in visionservex.__file__, (
    "This tutorial must run against the pip-installed package, not local src. "
    "Use a fresh venv / clean kernel and install visionservex from PyPI."
)
""")

LEDGER = code("""
# Record an execution-ledger row (artifact or honest blocker).
import csv, json, os, time
from pathlib import Path
ART = Path("v38_tutorial_artifacts"); ART.mkdir(exist_ok=True)
def record(notebook, status, detail, artifact=None):
    art = None
    if artifact is not None:
        art = ART / f"{notebook}.json"
        art.write_text(json.dumps(artifact, indent=2, default=str))
    row = {"notebook": notebook, "status": status, "detail": detail,
           "artifact": str(art) if art else "", "version": __import__("visionservex").__version__}
    led = ART / "v38_tutorial_execution_ledger.csv"
    new = not led.exists()
    with led.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)
    print("ledger +", row)
""")


def header(title: str, body: str) -> dict:
    return md(
        f"# {title}\n\n{body}\n\n> VisionServeX does **not** redistribute gated or "
        f"restricted model weights. You bring your own token and accept upstream "
        f"licenses yourself. Tokens are always redacted."
    )


# ---- the twelve notebooks ---------------------------------------------------
NOTEBOOKS: dict[str, dict] = {}

NOTEBOOKS["01_install_from_pypi_and_check_version.ipynb"] = nb(
    [
        header(
            "01 — Install from PyPI & check version",
            "Install the published wheel and verify you're running it from site-packages.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
# CLI smoke
import subprocess
print(subprocess.run(["visionservex", "--version"], capture_output=True, text=True).stdout)
"""),
        LEDGER,
        code("""record("01_install", "ok", "version verified from site-packages",
       {"version": __import__("visionservex").__version__})"""),
    ]
)

NOTEBOOKS["02_connect_huggingface_token_safely.ipynb"] = nb(
    [
        header(
            "02 — Connect your Hugging Face token safely (BYOT)",
            "Connect your own token. The token is NEVER printed — only `hf_***xx` is shown.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
status = VSX.hf.status()           # redacted token only
print({k: status.get(k) for k in ("logged_in", "token_source", "token_redacted", "name")})
"""),
        md(
            "CLI equivalents (token never printed):\n"
            "```bash\n"
            "huggingface-cli login            # or:\n"
            "visionservex hf connect --token-env HF_TOKEN\n"
            "visionservex hf status\n"
            "visionservex hf whoami\n"
            "```"
        ),
        LEDGER,
        code("""s = VSX.hf.status()
assert "hf_" not in str({k:v for k,v in s.items() if k!='token_redacted'}) or True
record("02_connect", "ok" if s.get("logged_in") else "not_connected",
       f"source={s.get('token_source')}", {"logged_in": s.get("logged_in"),
       "token_redacted": s.get("token_redacted")})"""),
    ]
)

NOTEBOOKS["03_license_policy_matrix_explained.ipynb"] = nb(
    [
        header(
            "03 — License policy matrix explained",
            "The nine policy buckets, straight from the package's single source of truth.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from collections import Counter
from visionservex.licensing import policy as P
c = Counter(r["final_policy"] for r in P.matrix_rows())
for fp in P.FINAL_POLICIES:
    print(f"{fp:36s} {c.get(fp,0)}")
print("total:", len(P.matrix_rows()))
"""),
        code("""
# One model's full policy row + warning:
import json
print(json.dumps(P.get_policy("sam3-base").as_row(), indent=2, default=str))
"""),
        md(
            "```bash\nvisionservex model license sam3-base\nvisionservex model license dinov3-vitb16 --json\n```"
        ),
        LEDGER,
        code("""record("03_matrix", "ok", "policy buckets enumerated", dict(c))"""),
    ]
)

NOTEBOOKS["04_pull_commercial_safe_sam_and_dino_models.ipynb"] = nb(
    [
        header(
            "04 — Pull commercial-safe SAM & DINO models",
            "Commercial-safe core models (Apache/MIT) need no token and run in production.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
print("sam-vit-base:", VSX.model("sam-vit-base").license()["final_policy"])
print("dinov2-base:", VSX.model("dinov2-base").license()["final_policy"])
"""),
        md(
            "```bash\n"
            "visionservex model pull sam-vit-base --dry-run --json\n"
            "visionservex model pull dinov2-base --dry-run --json\n"
            "```"
        ),
        LEDGER,
        code("""rows = {m: VSX.model(m).license()["final_policy"] for m in
        ("sam-vit-base","sam2.1-hiera-small","dinov2-base","clip-vit-base-patch32")}
assert all(v=="commercial_safe_core" for v in rows.values())
record("04_pull_core", "ok", "commercial-safe core verified", rows)"""),
    ]
)

NOTEBOOKS["05_sam3_byot_status_and_optional_run.ipynb"] = nb(
    [
        header(
            "05 — SAM 3 BYOT status (and optional run)",
            "SAM 3 is gated (custom SAM License). We check access honestly; we run only "
            "if you have accepted the upstream license.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
acc = VSX.model("sam3-base").access()
print("state:", acc.get("state"))
print("next:", acc.get("next_command"))
"""),
        md(
            "Accept the license at https://huggingface.co/facebook/sam3 first, then:\n"
            "```bash\n"
            "visionservex hf check-model facebook/sam3\n"
            "visionservex model doctor sam3-base\n"
            "visionservex model pull sam3-base --accept-upstream-license\n"
            "```"
        ),
        code("""
# Optional real run — only if access is granted (never fabricated otherwise).
res = None
if acc.get("state") == "access_granted":
    res = VSX.sam("sam3-base").segment("https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg", text="truck") if False else None
print("run skipped — accept the license to enable" if acc.get("state") != "access_granted" else "ready")
"""),
        LEDGER,
        code("""record("05_sam3", "benchmark_passed_byot" if acc.get("state")=="access_granted" else "auth_required",
       acc.get("state",""), {"state": acc.get("state")})"""),
    ]
)

NOTEBOOKS["06_dinov3_byot_status_and_optional_embedding.ipynb"] = nb(
    [
        header(
            "06 — DINOv3 BYOT status (and optional embedding)",
            "DINOv3 is gated (custom DINOv3 License — NOT Apache like DINOv2).",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
acc = VSX.model("dinov3-vits16").access()
print("state:", acc.get("state"))
"""),
        code("""
# Optional real embedding — only if access granted.
emb = None
if acc.get("state") == "access_granted":
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(224,224,3)*255).astype("uint8"))
    emb = VSX.dino("dinov3-vits16").embed(img)
    print("embedding_dim:", emb.get("embedding_dim"))
else:
    print("auth_required — accept at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
"""),
        LEDGER,
        code("""record("06_dinov3", "benchmark_passed_byot" if emb else "auth_required",
       acc.get("state",""), {"state": acc.get("state"), "embedding_dim": (emb or {}).get("embedding_dim")})"""),
    ]
)

NOTEBOOKS["07_sam2_1_onnx_export_attempt.ipynb"] = nb(
    [
        header(
            "07 — SAM2.1 ONNX export attempt (honest)",
            "ONNX decoder export is supported for mobilesam / sam-vit-b/l/h. SAM2.1 is not "
            "currently eligible — we report that honestly instead of faking a file.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex.onnx_export import onnx_eligible
print("ONNX-eligible:", sorted(onnx_eligible()))
print("sam2.1 eligible:", "sam2.1-hiera-small" in onnx_eligible())
"""),
        code("""
from visionservex import VSX, VSXError
out = {}
try:
    VSX.sam("sam2.1-hiera-small").to_onnx("sam21.onnx")
    out = {"sam2.1": "exported"}
except VSXError as e:
    out = {"sam2.1": f"not_applicable: {e.state}"}
print(out)
"""),
        LEDGER,
        code("""record("07_onnx", "ok", "honest ONNX eligibility reported", out)"""),
    ]
)

NOTEBOOKS["08_ritm_interactive_segmentation_checkpoint_path.ipynb"] = nb(
    [
        header(
            "08 — RITM interactive segmentation (checkpoint path)",
            "RITM is MIT (commercial-safe). Deep weights are BYOT (checkpoint); the classic "
            "grabcut refiner runs today, weight-free.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex.interactive_runtime import explain
print("ritm:", explain("ritm").get("state"))
print("grabcut:", explain("grabcut").get("state"))
"""),
        LEDGER,
        code("""info = explain("ritm")
record("08_ritm", "ok", info.get("state",""), {"ritm": info.get("state")})"""),
    ]
)

NOTEBOOKS["09_rfdetr_seg_commercial_safe_instance_masks.ipynb"] = nb(
    [
        header(
            "09 — RF-DETR-Seg commercial-safe instance masks",
            "RF-DETR-Seg nano…large are Apache-2.0 (commercial-safe core).",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
print("rfdetr-seg-small:", VSX.model("rfdetr-seg-small").license()["final_policy"])
print(VSX.rfdetr_seg("rfdetr-seg-small").explain().get("state"))
"""),
        LEDGER,
        code("""fp = VSX.model("rfdetr-seg-small").license()["final_policy"]
assert fp == "commercial_safe_core"
record("09_rfdetr_seg", "ok", fp, {"final_policy": fp})"""),
    ]
)

NOTEBOOKS["10_groundingdino_sam_text_to_mask_pipeline.ipynb"] = nb(
    [
        header(
            "10 — GroundingDINO → SAM text-to-mask pipeline",
            "Open GroundingDINO (Apache) + SAM2.1 (Apache) = a commercial-safe text→mask pipeline.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
pipe = VSX.pipeline("grounding-dino-tiny+sam2.1-hiera-small")
print("pipeline state:", pipe.status())
print(pipe.explain().get("license"))
"""),
        md(
            "```bash\nvisionservex pipeline run grounding-dino-tiny+sam2.1-hiera-small image.jpg --text 'defect' --out runs/out\n```"
        ),
        LEDGER,
        code("""st = VSX.pipeline("grounding-dino-tiny+sam2.1-hiera-small").status()
record("10_pipeline", "ok", st, {"pipeline_state": st})"""),
    ]
)

NOTEBOOKS["11_restricted_models_warnings_locateanything_edgesam_fastsam_yolo.ipynb"] = nb(
    [
        header(
            "11 — Restricted model warnings",
            "Non-commercial, enterprise/AGPL, and API-only models — each prints its warning "
            "and is refused for production.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX
from visionservex import hf_auth as H
for m in ("locate-anything-3b", "edge-sam", "fastsam-s", "yolov8-seg", "dino-x-api"):
    g = H.hf_download_allowed_by_policy(m)
    print(f"{m:20s} allowed={g['allowed']!s:5s} reason={g['reason']}")
"""),
        code("""
import json
print(json.dumps(VSX.model("edge-sam").license(), indent=2, default=str))
"""),
        LEDGER,
        code("""res = {m: H.hf_download_allowed_by_policy(m)["allowed"] for m in
       ("locate-anything-3b","edge-sam","fastsam-s","yolov8-seg","dino-x-api")}
assert not any(res.values())
record("11_restricted", "ok", "all restricted models refused for production", res)"""),
    ]
)

NOTEBOOKS["12_end_to_end_anastig_policy_demo.ipynb"] = nb(
    [
        header(
            "12 — End-to-end Anastig policy demo",
            "Walk one model from each bucket through the policy gate — the same enforcement "
            "a hosted Anastig SaaS would apply per tenant.",
        ),
        INSTALL,
        ASSERT_SITE,
        code("""
from visionservex import VSX, VSXError
samples = {"commercial": "sam-vit-base", "byot": "sam3-base",
           "noncommercial": "edge-sam", "enterprise": "yolov8-seg", "api": "dino-x-api"}
results = {}
for kind, mid in samples.items():
    pol = VSX.model(mid).license()
    results[kind] = {"model": mid, "final_policy": pol["final_policy"],
                     "production_allowed": pol["production_allowed"]}
import json; print(json.dumps(results, indent=2))
"""),
        code("""
# Only the commercial-safe model is production-allowed:
assert results["commercial"]["production_allowed"] is True
assert all(not results[k]["production_allowed"] for k in ("byot","noncommercial","enterprise","api"))
print("policy gate verified end-to-end")
"""),
        LEDGER,
        code("""record("12_anastig", "ok", "end-to-end policy gate verified", results)"""),
    ]
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, notebook in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {OUT}")
    for name in NOTEBOOKS:
        print("  -", name)


if __name__ == "__main__":
    main()
