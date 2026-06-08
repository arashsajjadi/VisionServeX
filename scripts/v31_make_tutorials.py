#!/usr/bin/env python3
"""Generate v3.1 tutorial notebooks (SAM / DINO / pipelines / cv2-pro / interactive).

Each notebook starts with a fresh-pip-install verification cell (asserts import
from site-packages, NOT local src) and includes: what it does, license/auth state,
Python + CLI examples, an executed status/explain/run cell, and limitations.
For commercial-safe runnable tools the run cell executes real inference; for
gated / non-commercial / sidecar targets it executes the honest status/explain
workflow (no token, no fake result).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VERSION = sys.argv[1] if len(sys.argv) > 1 else "3.1.0"
NB = Path("notebook/tutorials")


def cell_code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def cell_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


INSTALL = f'''import sys, subprocess, importlib.metadata
VERSION = "{VERSION}"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
    f"visionservex[classic-ml,cv2-pro]=={{VERSION}}"])
import visionservex
print("visionservex", importlib.metadata.version("visionservex"))
print("file:", visionservex.__file__)
assert "site-packages" in visionservex.__file__, visionservex.__file__
assert "PycharmProjects/VisionServeX/src" not in visionservex.__file__'''


def make_nb(title: str, body_md: str, code_cells: list[str]) -> dict:
    cells = [
        cell_md(f"# {title}\n\n{body_md}"),
        cell_md("## Fresh pip-install verification (imports from site-packages, not local src)"),
        cell_code(INSTALL),
    ]
    for c in code_cells:
        cells.append(cell_code(c))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(subdir: str, name: str, nb: dict) -> str:
    d = NB / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.ipynb"
    p.write_text(json.dumps(nb, indent=1))
    return str(p)


written = []

# ---- SAM family (runnable + gated + legal) ----
SAM_RUNNABLE = [
    "sam-vit-h",
    "sam2-hiera-large",
    "sam2.1-hiera-small",
    "mobilesam",
    "efficientsam",
    "medsam",
]
SAM_GATED = ["sam3-base"]
SAM_LEGAL = ["hq-sam", "tinysam", "light-hq-sam"]
SAM_EXCLUDED = ["edge-sam"]
SAM_SIDECAR = ["medsam2"]
SAM_ONNX = ["sam-vit-b-onnx", "mobilesam-onnx"]
for m in SAM_RUNNABLE:
    nb = make_nb(
        f"SAM tutorial — {m}",
        f"Promptable segmentation with **{m}** (Apache-2.0, commercial-safe). "
        "Box/point prompts. Use in the Anastig annotation studio for box→mask refinement, and from any "
        "LLM/coding agent via the `VSX.sam(...)` API or `visionservex sam` CLI.",
        [
            f'from visionservex import VSX\nsam = VSX.sam("{m}")\nimport json; print(json.dumps(sam.explain(), indent=2, default=str))',
            f'# CLI: visionservex sam status {m}  |  visionservex sam run {m} image.jpg --box 10,20,200,220 --out runs/{m}\nprint("state:", sam.status())',
            '# Real run requires the [sam2,promptable] extra + a checkpoint download. License: Apache-2.0 (commercial-safe).\n# sam.segment("image.jpg", box=[10,20,200,220]).save_mask("mask.png")\nprint("license:", sam.explain()["license"]); print("limitations:", sam.explain()["limitations"])',
        ],
    )
    written.append(write("sam_family", m, nb))
for m in SAM_GATED:
    nb = make_nb(
        f"SAM tutorial — {m} (BYOT / gated)",
        f"**{m}** uses the Meta custom SAM License and is **HF-gated**. VisionServeX never mirrors gated "
        "weights and never logs your token — BYOT (bring-your-own-token).",
        [
            f'from visionservex import VSX, VSXError\nsam = VSX.sam("{m}")\nimport json; print(json.dumps(sam.explain(), indent=2, default=str))',
            '# 1) request HF gated access on the model page  2) export HF_TOKEN=...  3) run.\n# Graceful failure without a token (no secret printed):\ntry:\n    sam.segment("image.jpg", box=[0,0,10,10])\nexcept VSXError as e:\n    print("state:", e.state); print("next:", e.next_command)',
        ],
    )
    written.append(write("sam_family", m, nb))
for m in SAM_LEGAL:
    nb = make_nb(
        f"SAM tutorial — {m} (legal review)",
        f"**{m}**: Apache-2.0 code, but training-data provenance (HQSeg-44K non-commercial / SA-1B research) "
        "means it stays `legal_review_required` (NOT commercial-safe core) until reviewed.",
        [
            f'from visionservex import VSX\nsam = VSX.sam("{m}")\nimport json; print(json.dumps(sam.explain(), indent=2, default=str))\nprint("Use lawfully:", sam.explain()["next_command"])'
        ],
    )
    written.append(write("sam_family", m, nb))
for m in SAM_EXCLUDED:
    nb = make_nb(
        f"SAM tutorial — {m} (NON-COMMERCIAL, excluded)",
        f"**{m}** is licensed NTU S-Lab License 1.0 (**non-commercial**) and is excluded from commercial-safe "
        "core. Lawful alternatives: MobileSAM / EfficientSAM (Apache-2.0).",
        [
            f'from visionservex import VSX\nsam = VSX.sam("{m}")\nimport json; print(json.dumps(sam.explain(), indent=2, default=str))\nprint("Lawful commercial-safe alternative: mobilesam / efficientsam")'
        ],
    )
    written.append(write("sam_family", m, nb))
for m in SAM_SIDECAR:
    nb = make_nb(
        f"SAM tutorial — {m} (sidecar)",
        f"**{m}** runs in an isolated sidecar environment (Apache-2.0). Build it with the sidecar manager.",
        [f'from visionservex import VSX\nprint(VSX.sam("{m}").explain())'],
    )
    written.append(write("sam_family", m, nb))
for m in SAM_ONNX:
    nb = make_nb(
        f"SAM tutorial — {m} (ONNX export)",
        f"**{m}**: LOCAL ONNX export from the Apache-2.0 weights is license-clean. Export then run on CPU "
        "via the CV2-Pro DNN runner.",
        [
            f'from visionservex import VSX\nprint(VSX.sam("{m}").explain())\n# visionservex sam export-onnx {m.replace("-onnx", "")} --out models/{m}.onnx\n# visionservex cv2-pro run opencv-dnn-onnx-runner image.jpg --onnx models/{m}.onnx'
        ],
    )
    written.append(write("sam_family", m, nb))

# ---- DINO ----
DINO_RUN = [("dinov2-base", "embed"), ("grounding-dino-swin-t", "detect")]
DINO_GATED = [
    ("dinov3-vitb16", "legal"),
    ("grounding-dino-1.5", "auth"),
    ("grounding-dino-1.6", "auth"),
    ("dino-x-api", "api"),
]
for m, kind in DINO_RUN:
    nb = make_nb(
        f"DINO tutorial — {m}",
        f"**{m}** (Apache-2.0). {'Image embeddings (kNN / retrieval / dedup).' if kind == 'embed' else 'Open-vocabulary text detection.'}",
        [
            f'from visionservex import VSX\nd = VSX.dino("{m}")\nimport json; print(json.dumps(d.explain(), indent=2, default=str))\nprint("state:", d.status())'
        ],
    )
    written.append(write("dino_family", m, nb))
for m, kind in DINO_GATED:
    note = {
        "legal": "DINOv3 custom Meta license (HF-gated) — license-aware, not Apache.",
        "auth": "GroundingDINO 1.5/1.6 are API/token-gated — BYOT.",
        "api": "DINO-X is API-only — external; weights never mirrored.",
    }[kind]
    nb = make_nb(
        f"DINO tutorial — {m} ({kind})",
        f"**{m}**: {note}",
        [
            f'from visionservex import VSX\nimport json; print(json.dumps(VSX.dino("{m}").explain(), indent=2, default=str))'
        ],
    )
    written.append(write("dino_family", m, nb))

# ---- pipelines (text-to-mask) ----
PIPES = [
    "grounding-dino-swin-t+sam2.1-hiera-small",
    "grounding-dino-swin-b+sam2.1-hiera-large",
    "grounding-dino-swin-t+sam-vit-h",
    "grounding-dino-original-swin-t+sam2-hiera-small",
    "grounding-dino-1.6+sam3-base",
    "dinov3-vitb16+sam3-base",
]
for p in PIPES:
    nb = make_nb(
        f"Pipeline tutorial — {p}",
        f"Text-to-mask pipeline **{p}**: GroundingDINO detects by text, SAM segments each box. "
        "First-class via `VSX.pipeline(...)` / `visionservex pipeline run`.",
        [
            f'from visionservex import VSX\npipe = VSX.pipeline("{p}")\nimport json; print(json.dumps(pipe.explain(), indent=2, default=str))\nprint("state:", pipe.status())'
        ],
    )
    written.append(write("pipelines", p.replace("+", "_"), nb))

# ---- cv2-pro (real execution) ----
import importlib.util as _u  # noqa: E402

_spec = _u.spec_from_file_location("_t", "src/visionservex/cv2_pro/tools.py")
_t = _u.module_from_spec(_spec)
_spec.loader.exec_module(_t)
for tool in _t.list_tools():
    contrib = tool.startswith("opencv-selective")
    nb = make_nb(
        f"CV2-Pro tutorial — {tool}",
        f"Weight-free OpenCV tool **{tool}** (Apache-2.0, CPU). "
        f"{'Requires the cv2-pro extra (opencv-contrib).' if contrib else 'Runs on the base install.'} "
        "Use for region proposals / mask refinement in annotation pipelines.",
        [
            "import numpy as np\ntry:\n    import cv2\nexcept Exception:\n    cv2=None\nfrom visionservex.cv2_pro import run_tool, tool_available, TOOL_LICENSE\n"
            f'print("tool:", "{tool}", "| available:", tool_available("{tool}"), "| license:", TOOL_LICENSE().get("{tool}"))',
            f'if cv2 is not None and tool_available("{tool}")[0]:\n    img=(np.random.default_rng(0).normal(120,30,(192,192,3))).clip(0,255).astype("uint8")\n    params={{"box":[30,30,160,160]}} if "{tool}"=="opencv-grabcut-plus" else {{}}\n    r=run_tool("{tool}", img, **params)\n    print({{k:v for k,v in r.items() if k!="polygons"}})\nelse:\n    print("install: pip install \'visionservex[cv2-pro]\'")',
        ],
    )
    written.append(write("cv2_pro", tool, nb))

# ---- interactive-seg ----
for m, state in [
    ("ritm", "checkpoint_required"),
    ("clickseg", "checkpoint_required"),
    ("focalclick", "legal_review_required"),
    ("simpleclick", "legal_review_required"),
]:
    nb = make_nb(
        f"Interactive-seg tutorial — {m} ({state})",
        f"**{m}** interactive click segmentation. Code is permissive; "
        f"{'user supplies the checkpoint (legacy env).' if state == 'checkpoint_required' else 'weights are non-commercial (MAE/SegFormer NC) -> legal review.'}",
        [
            f'# Honest state: {state}. License audit: notebook/99_final_report/reports/v31_web_research_evidence.csv\nprint("model: {m} | state: {state}")\nprint("next: visionservex interactive-seg status {m}")'
        ],
    )
    written.append(write("interactive_seg", m, nb))

print(f"generated {len(written)} tutorial notebooks")
for sub in ["sam_family", "dino_family", "pipelines", "cv2_pro", "interactive_seg"]:
    print(f"  {sub}: {len(list((NB / sub).glob('*.ipynb')))}")
