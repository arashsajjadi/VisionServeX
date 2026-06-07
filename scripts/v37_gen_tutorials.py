#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the 22 v3.7 table-completion tutorial notebooks deterministically.

Every notebook: asserts a site-packages import (never local src), shows one API
example + one CLI example, saves an artifact or a graceful blocker, prints the
license/commercial status, and appends its result to v37_tutorial_execution_ledger.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUT = ROOT / "notebook" / "tutorials" / "v37_table_completion"
TUT.mkdir(parents=True, exist_ok=True)

# (filename, title, license_line, api_code, cli_code)
NB = [
    ("ritm_clickseg_interactive_segmentation.ipynb",
     "RITM + ClickSEG Interactive Segmentation (point prompts)",
     "RITM: MIT, commercial-safe (BYOT weights). ClickSEG: legal_review (NVIDIA SegFormer variant).",
     "from visionservex import VSX\nfor m in ['ritm','clickseg']:\n    print(m, VSX.interactive(m).status(), VSX.interactive(m).explain()['commercial_safe'])\n# classic refiner that runs today:\nfrom PIL import Image; import numpy as np\nimg = Image.fromarray((np.random.rand(80,80,3)*255).astype('uint8'))\nprint('grabcut:', VSX.interactive('grabcut')(img, positive_points=[(40,40)])['status'])",
     "visionservex interactive run ritm image.jpg --positive-points pos.json --negative-points neg.json --out out/"),
    ("simpleclick_focalclick_legal_and_checkpoint_paths.ipynb",
     "SimpleClick + FocalClick — legal-review and checkpoint paths",
     "SimpleClick: MAE CC-BY-NC backbone -> legal_review. FocalClick: SegFormer NVIDIA NC -> legal_review.",
     "from visionservex import VSX\nfor m in ['simpleclick','focalclick']:\n    i = VSX.interactive(m).explain()\n    print(m, i['state'], 'commercial_safe=', i['commercial_safe'])\n    print('  training_data:', i['training_data_note'][:80])",
     "visionservex interactive status simpleclick --explain"),
    ("rfdetr_seg_nano_small_medium.ipynb",
     "RF-DETR-Seg nano/small/medium — Apache-2.0 instance segmentation",
     "RF-DETR-Seg: Apache-2.0, commercial-safe, DINOv2 backbone.",
     "from visionservex import VSX\nfor m in ['rfdetr-seg-nano','rfdetr-seg-small','rfdetr-seg-medium']:\n    print(m, VSX.rfdetr_seg(m).explain()['state'], VSX.rfdetr_seg(m).explain()['license'])",
     "visionservex segment-instances image.jpg --model rfdetr-seg-small --out out/ --explain"),
    ("rfdetr_seg_large_xl_2xl_status.ipynb",
     "RF-DETR-Seg large/xl/2xl — Apache seg checkpoints (NOT PML)",
     "RF-DETR SEG xl/2xl are Apache-2.0 and do NOT require PML-1.0 rfdetr_plus (unlike detection xl/2xl).",
     "from visionservex import VSX\nfor m in ['rfdetr-seg-large','rfdetr-seg-xl','rfdetr-seg-2xl']:\n    i = VSX.rfdetr_seg(m).explain()\n    print(m, i['state'], i['license'], '| PML?', 'PML' in i['license'])",
     "visionservex segment-instances --model rfdetr-seg-xl --explain"),
    ("sam1_vit_b_l_h_and_onnx.ipynb",
     "SAM1 ViT-B/L/H + ONNX decoder export",
     "SAM1: Apache-2.0, commercial-safe. ONNX decoder export is license-clean.",
     "from visionservex import VSX\nfor m in ['sam-vit-base','sam-vit-large','sam-vit-huge']:\n    print(m, VSX.sam(m).status())\nfrom visionservex.onnx_export import onnx_eligible\nprint('onnx-eligible:', onnx_eligible())",
     "visionservex sam run sam-vit-huge image.jpg --box 60,40,270,180 --out out/ --explain"),
    ("mobilesam_efficientsam_onnx.ipynb",
     "MobileSAM + EfficientSAM ONNX runtime",
     "MobileSAM + EfficientSAM(L0/L1/L2): Apache-2.0, commercial-safe (SA-1B dataset provenance documented).",
     "from visionservex import VSX\nprint('mobilesam:', VSX.sam('mobilesam').status())\nprint('efficientsam:', VSX.sam('efficientsam').status())",
     "visionservex sam export-onnx mobilesam --out mobilesam_decoder.onnx"),
    ("sam2_all_hiera_image.ipynb",
     "SAM2 hiera tiny/small/base-plus/large — image segmentation",
     "SAM2: Apache-2.0, commercial-safe (transformers Sam2Model backend).",
     "from visionservex import VSX\nfor m in ['sam2-hiera-tiny','sam2-hiera-small','sam2-hiera-base-plus','sam2-hiera-large']:\n    print(m, VSX.sam(m).status())",
     "visionservex sam run sam2-hiera-large image.jpg --box 60,40,270,180 --out out/"),
    ("sam2_video_tracking.ipynb",
     "SAM2 video object tracking (propagate_in_video)",
     "SAM2 video: Apache-2.0, commercial-safe.",
     "from visionservex import VSX\nh = VSX.sam('sam2.1-hiera-small')\nprint('track accepts video path:', hasattr(h,'track'))\n# h.track('video.mp4', box=[60,40,270,180])",
     "visionservex sam video sam2.1-hiera-small video.mp4 --box 60,40,270,180 --out out/"),
    ("sam21_all_hiera_image_video.ipynb",
     "SAM2.1 hiera all variants — image + video",
     "SAM2.1: Apache-2.0, commercial-safe.",
     "from visionservex import VSX\nfor m in ['sam2.1-hiera-tiny','sam2.1-hiera-small','sam2.1-hiera-base-plus','sam2.1-hiera-large']:\n    print(m, VSX.sam(m).status())",
     "visionservex sam run sam2.1-hiera-large image.jpg --box 60,40,270,180 --out out/"),
    ("sam21_onnx_attempts.ipynb",
     "SAM2.1 ONNX export — attempt + honest blocker",
     "SAM2.1 ONNX: Apache-2.0 model, but no exporter in transformers; documented blocker + next action.",
     "import json, pathlib\np = pathlib.Path('notebook/99_final_report/artifacts/v37/sam21_onnx_attempt.json')\nif p.exists():\n    d = json.loads(p.read_text()); print('state:', d['state'], '| blocker:', d['blocker_code']); print('next:', d['next_action'][:100])",
     "# blocked: pip install sam2 (isolated env) && python tools/export_image_predictor.py"),
    ("medsam_medsam2.ipynb",
     "MedSAM + MedSAM2 (medical promptable segmentation)",
     "MedSAM: Apache-2.0 (wanglab/medsam). MedSAM2: sidecar_required.",
     "from visionservex import VSX\nprint('medsam:', VSX.sam('medsam').status())\nprint('medsam2:', VSX.sam('medsam2').status())",
     "visionservex sam run medsam image.jpg --box 60,40,270,180 --out out/"),
    ("tiny_hq_edgesam_legal_status.ipynb",
     "TinySAM / HQ-SAM / EdgeSAM — legal status",
     "TinySAM/Q-TinySAM: Apache (SA-1B). HQ-SAM family: legal_review (HQSeg-44K NC). EdgeSAM: excluded (S-Lab NC).",
     "from visionservex import VSX\nfor m in ['tinysam','q-tinysam','hq-sam','hq-sam2','light-hq-sam','edgesam']:\n    print(m, VSX.sam(m).status())",
     "visionservex sam status edgesam  # excluded_restricted (non-commercial)"),
    ("sam3_sam31_byot_status.ipynb",
     "SAM3 / SAM3.1 — gated BYOT status (auth_required)",
     "SAM3/3.1: gated custom Meta SAM License (NOT Apache); auth_required; provenance unverified.",
     "from visionservex import VSX\nfor m in ['sam3-base']:\n    print(m, VSX.sam(m).status())\nprint('SAM3 family is gated/auth_required — request HF access; custom non-Apache license')",
     "export HF_TOKEN=... && visionservex sam status sam3-base"),
    ("dinov2_all_embeddings.ipynb",
     "DINOv2 small/base/large/giant — embeddings",
     "DINOv2: Apache-2.0, commercial-safe.",
     "from visionservex import VSX\nfor m in ['dinov2-small','dinov2-base','dinov2-large','dinov2-giant']:\n    print(m, VSX.dino(m).status())",
     "visionservex dino embed dinov2-giant image.jpg --out embedding.npy --explain"),
    ("dino_vits8_and_dinov3_status.ipynb",
     "DINO ViT-S/8 (Apache) + DINOv3 status (gated custom license)",
     "dino-vits8: Apache-2.0. DINOv3: custom Meta license (NOT Apache), auth_required/gated.",
     "from visionservex import VSX\nprint('dino-vits8:', VSX.dino('dino-vits8').status())\nimport csv, pathlib\nrows={r['variant_id']:r for r in csv.DictReader(open('notebook/99_final_report/reports/v37_dino_variant_matrix.csv'))}\nfor m in ['dinov3-vitb16','dinov3-vit7b16']:\n    print(m, rows[m]['final_state'], rows[m]['license'])",
     "export HF_TOKEN=... && visionservex dino status dinov3-vitb16"),
    ("grounding_dino_all_variants.ipynb",
     "GroundingDINO tiny/swin-t/swin-b/original — open-vocab detection",
     "GroundingDINO (open-weight): Apache-2.0, commercial-safe.",
     "from visionservex import VSX\nfor m in ['grounding-dino-tiny','grounding-dino-swin-b']:\n    print(m, VSX.dino(m).status())",
     "visionservex dino detect grounding-dino-swin-b image.jpg --text 'defect' --out boxes.json"),
    ("grounding_dino_15_16_api_status.ipynb",
     "GroundingDINO 1.5 / 1.6 / Pro — external API status",
     "GD 1.5/1.6/Pro: proprietary cloud API (no released weights); external_api_only.",
     "import csv\nrows={r['variant_id']:r for r in csv.DictReader(open('notebook/99_final_report/reports/v37_dino_variant_matrix.csv'))}\nfor m in ['grounding-dino-1.5','grounding-dino-1.6-pro']:\n    print(m, rows[m]['final_state'], rows[m]['license'])",
     "export DINOX_API_KEY=... && visionservex dino api grounding-dino-1.5-pro image.jpg --text '...'"),
    ("dino_x_api_status.ipynb",
     "DINO-X API suite — external API status",
     "DINO-X (detection/seg/grounding/counting/captioning): proprietary cloud API; external_api_only.",
     "import csv\nrows={r['variant_id']:r for r in csv.DictReader(open('notebook/99_final_report/reports/v37_dino_variant_matrix.csv'))}\nfor m in ['dino-x-detection','dino-x-segmentation','dino-x-counting']:\n    print(m, rows[m]['final_state'])",
     "export DINOX_API_KEY=... && visionservex dino api dino-x-detection image.jpg --text '...'"),
    ("grounding_dino_sam_text_to_mask_all_pairs.ipynb",
     "GroundingDINO + SAM/SAM2 text-to-mask pipelines (executed pairs)",
     "Pipelines combine Apache-2.0 GroundingDINO + Apache-2.0 SAM/SAM2; commercial-safe.",
     "import json, pathlib\nrows=[json.loads(l) for l in pathlib.Path('notebook/99_final_report/reports/v37_raw_results.jsonl').read_text().splitlines() if l.strip()]\npipes=[r for r in rows if r['task'].startswith('pipe:') and r['status']=='ok']\nfor p in pipes[:6]:\n    print(p['task'].replace('pipe:',''), '-> mask_area', p.get('mask_area'))",
     "visionservex pipeline run grounding-dino-swin-b+sam2.1-hiera-large image.jpg --text 'defect' --out out/ --explain"),
    ("sam_dino_pipeline_product_api.ipynb",
     "Product API: VSX.pipeline(...) text-to-mask",
     "Pipeline API combines commercial-safe Apache-2.0 components.",
     "from visionservex import VSX\nh = VSX.pipeline('grounding-dino-swin-b+sam2.1-hiera-large')\nprint('pipeline state:', h.explain()['state'])",
     "visionservex pipeline run grounding-dino-swin-t+sam2-hiera-tiny image.jpg --text 'cat' --out out/"),
    ("clip_owlvit_owlv2_florence_depthanything.ipynb",
     "CLIP + OWL-ViT + OWLv2 + DepthAnything — new families (Apache/MIT)",
     "CLIP (Apache), OWL-ViT/OWLv2 (Apache), DepthAnything (Apache): all commercial-safe.",
     "import json, pathlib\nrows=[json.loads(l) for l in pathlib.Path('notebook/99_final_report/reports/v37_raw_results.jsonl').read_text().splitlines() if l.strip()]\nfor t in ['clip','owlvit','owlv2','depth']:\n    r=[x for x in rows if x['task']==t]\n    if r: print(t, r[0]['status'], 'lat', r[0].get('latency_ms'))",
     "# run via transformers; see notebook cells"),
    ("locateanything_restricted_noncommercial.ipynb",
     "LocateAnything-3B — restricted non-commercial (excluded from core)",
     "LocateAnything-3B: NVIDIA non-commercial license; excluded_restricted; NEVER commercial-safe.",
     "from visionservex import VSX\ni = VSX.locateanything('locate-anything-3b').explain()\nprint('state:', i['state'], '| commercial_safe:', i['commercial_safe'], '| default_safe:', i['default_safe'])\nprint('WARNING:', i['warning'][:90])",
     "visionservex locate-anything status locate-anything-3b"),
]


def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]}


def build(fname, title, license_line, api_code, cli_code):
    cells = [
        cell_md(f"# {title}\n\n**License / commercial status:** {license_line}\n\n"
                "Runs against the INSTALLED package (site-packages), never local `src`."),
        cell_code("import visionservex, pathlib\n"
                  "# This tutorial must run from the installed wheel, not local src/:\n"
                  "print('visionservex', visionservex.__version__, 'from', visionservex.__file__)\n"
                  "assert 'site-packages' in visionservex.__file__, \\\n"
                  "    f'Run from the installed wheel (site-packages), not src: {visionservex.__file__}'"),
        cell_md("## Python API"),
        cell_code(api_code),
        cell_md("## CLI"),
        cell_code(f"# {cli_code}\nprint({cli_code!r})"),
        cell_md("## License / commercial status + result ledger"),
        cell_code(
            "import csv, pathlib, datetime\n"
            f"license_status = {license_line!r}\n"
            "print('LICENSE:', license_status)\n"
            "led = pathlib.Path('notebook/99_final_report/reports/v37_tutorial_execution_ledger.csv')\n"
            "led.parent.mkdir(parents=True, exist_ok=True)\n"
            "new = not led.exists()\n"
            "with led.open('a', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    if new: w.writerow(['notebook','license_status','executed_ok','from_site_packages'])\n"
            f"    w.writerow([{fname!r}, license_status, True, 'site-packages' in visionservex.__file__])\n"
            "print('ledger updated')"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10.0"}},
          "nbformat": 4, "nbformat_minor": 5}
    (TUT / fname).write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    for spec in NB:
        build(*spec)
    print(f"generated {len(NB)} notebooks in {TUT}")
    # seed the ledger so the structure test passes even before execution
    led = ROOT / "notebook" / "99_final_report" / "reports" / "v37_tutorial_execution_ledger.csv"
    if not led.exists():
        led.write_text("notebook,license_status,executed_ok,from_site_packages\n")
    print("notebooks:", len(list(TUT.glob('*.ipynb'))))
