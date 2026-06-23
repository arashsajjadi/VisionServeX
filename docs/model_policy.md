<!-- SPDX-License-Identifier: Apache-2.0 -->
# VisionServeX Model License Policy

VisionServeX is a developer-friendly Python package for **commercial-safe computer
vision model serving and inference**. It is **license-aware by default**: the core
is commercial-safe, and restricted models are not enabled unless you explicitly
choose a research/BYO pathway and acknowledge the applicable restrictions.

> VisionServeX is **not a medical device** and **not a clinical diagnostic system**.

## Commercial-safe by default

> **VisionServeX is commercial-safe by default: restricted models are not enabled
> unless the user explicitly chooses a research/BYO pathway and acknowledges the
> applicable restrictions.**

A model is treated as commercial-safe only when **both its code and its pretrained
weights** are verified permissive. Code being permissive (e.g. Apache-2.0) is *not*
sufficient if the weights/dataset are restricted.

The single source of truth is `visionservex.policy`. Everything (Python API, CLI,
registry views, docs, tests) derives from it.

## Why VisionServeX is different from default-risk stacks

- **Ultralytics YOLO is AGPL-3.0 by default.** VisionServeX does **not** install
  Ultralytics by default and does **not** treat it as the commercial-safe baseline.
  Any AGPL/copyleft model is excluded from the commercial-safe set (enforced by a
  test). VisionServeX ships permissive detectors (D-FINE, RF-DETR, LibreYOLO
  YOLOX/YOLOv9, etc.) as commercial-safe alternatives.
- **Restricted weights cannot silently enter** the default registry, the
  commercial-safe list, quickstart examples, the default install path, or the
  auto-download path.

## Policy taxonomy

| Field | Values |
|---|---|
| `commercial_status` | `commercial_safe`, `noncommercial_restricted`, `research_only`, `agpl_restricted`, `legal_review_required`, `byo_license_only`, `framework_only`, `unknown` |
| `default_package_tier` | `core`, `optional_commercial_safe`, `research`, `byo`, `external`, `hidden` |
| `allowed_use_modes` | `commercial`, `research`, `education`, `internal_evaluation`, `byo_license` |
| `finetuning_status` | `supported`, `dry_run_only`, `external_only`, `not_trainable_by_design`, `unknown` |
| `cli_warning_level` | `none`, `info`, `warning`, `blocking` |

## Commercial-safe model families (examples)

**Curated** commercial-safe (code **and** weights verified against official
sources, 2026-06-22): SAM / SAM2 / SAM2.1 (Apache-2.0), CLIP (MIT), SigLIP / SigLIP2
(Apache-2.0), DINOv2 (Apache-2.0), OWLv2 / OWL-ViT (Apache-2.0), SwinV2 (MIT),
MaxViT (Apache-2.0), torchvision classifiers (BSD-3), **D-FINE** (Apache-2.0),
**RF-DETR / RF-DETR-Seg core** (Apache-2.0), **Grounding DINO (open)** (Apache-2.0),
**Florence-2** (MIT), RTMPose (Apache-2.0), Grounded-SAM (Apache-2.0).

> **Excluded after re-audit:** **ConvNeXtV2** — the upstream
> `facebookresearch/ConvNeXt-V2` LICENSE is **CC-BY-NC-4.0 (non-commercial)** while
> the HF model card tags `apache-2.0`. The conflict is resolved by the stricter
> interpretation → ConvNeXtV2 is **not commercial-safe** (`legal_review_required`).
> **RF-DETR-Seg XL/2XL** are held for enterprise-terms verification (not commercial-safe).

Coverage is auditable: `visionservex models coverage --json` reports how many
commercial-safe models are curated vs registry-derived (the latter are only
weight-less built-in mocks).

```bash
visionservex models list --commercial-safe         # the commercial-safe set
visionservex models list --research                 # research-only models
visionservex models list --byo                       # BYO-license/checkpoint models
visionservex models list --legal-review              # not commercial-safe (pending review)
visionservex models coverage --json                  # curated vs registry-derived
visionservex models policy sam2.1-hiera-tiny --json
```

## Research-only / BYO / restricted models

| Model | commercial_status | Why | How to use |
|---|---|---|---|
| **MedSAM2** | `research_only` | HF model card: *weights for research & education only* | research/BYO + acknowledgement (isolated env) |
| **MedSAM v1** | `legal_review_required` | code Apache-2.0; medical weight provenance | research pathway; not commercial-safe |
| **SAM3 / SAM3.1** | `byo_license_only` | HF-gated upstream license | BYO token + checkpoint |
| **HQ-SAM, OneFormer (swin/convnext/dinat-large)** | `legal_review_required` | provenance review pending | not commercial-safe |
| **ConvNeXtV2** | `legal_review_required` | upstream CC-BY-NC-4.0 vs HF apache-2.0 **conflict** → stricter wins | not commercial-safe |
| **RF-DETR-Seg XL / 2XL** | `legal_review_required` | enterprise-terms verification pending | not commercial-safe |
| **Ultralytics YOLO** (if ever added) | `agpl_restricted` | AGPL-3.0 | requires a separate commercial license; never in the commercial-safe set |

### MedSAM2 policy (canonical example)

MedSAM2 weights are **research/education only** (non-commercial). MedSAM2 is
therefore:
- **not commercial-safe**, **not in the commercial-safe list**, **not auto-downloaded**,
- **not** constructible through the default path,
- usable only via an explicit acknowledged research/BYO pathway.

```python
from visionservex import VisionModel

VisionModel("medsam2")  # raises MODEL_ACKNOWLEDGEMENT_REQUIRED

model = VisionModel(
    "medsam2",
    use_mode="research",
    acknowledge_license_restrictions=True,
)
```

The experimental real MedSAM2 2D runtime lives under
`visionservex medical medsam2 …` (isolated env). See `docs/medical_segmentation.md`.

### TotalSegmentator — task-level policy

TotalSegmentator is **task-specific**: the core `total` task may be open for
commercial use, but several sub-tasks are non-commercial / license-key gated and
must **never** be treated as commercial-safe, including (non-exhaustive):
**appendicular bones**, **tissue types** (body-composition), **heartchambers
highres**, and **face**. VisionServeX does **not** mark the whole package globally
commercial-safe — `totalsegmentator-tissue` is flagged `non_core_license_optional`
and requires a commercial license key. Verify each task/checkpoint against its
model card.

### NVIDIA / MONAI / VISTA / nnU-Net

- **nnU-Net v2, MONAI, Auto3DSeg, SwinUNETR, UNETR** are *frameworks*: the code may
  be commercially usable, but **pretrained weights are checkpoint-specific** and
  must be verified per model card. Treated as `framework_only` / external.
- **VISTA3D / NVIDIA VISTA / NV-Segment-CT / NV-Segment-CTMR** are **not present**
  in the VisionServeX registry and are therefore not in any commercial-safe list.
  If ever added, each must be verified per exact checkpoint/model card: NVIDIA NV /
  VISTA weights frequently carry **non-commercial / evaluation-only** terms (or a
  hosted NIM route under NVIDIA commercial terms) and would be gated
  `noncommercial_restricted` / `legal_review_required` / `external` — never
  commercial-safe unless the official license clearly permits commercial weight use.

## The acknowledgement gate

Restricted models require an explicit, ugly, non-accidental acknowledgement.

**Acknowledgement text:**

> I understand that this model is not commercial-safe by default. I confirm that I
> have the right to use the model weights for this use case. Outputs are AI-generated
> suggestions only and are not for diagnosis, treatment, or clinical decision-making.

**Python:**
```python
model = VisionModel("medsam2", use_mode="research", acknowledge_license_restrictions=True)
```

**CLI:**
```bash
visionservex predict medsam2 image.png --use-mode research --acknowledge-license-restrictions
# BYO checkpoint/license:
visionservex predict sam3-base image.png --use-mode byo_license \
    --checkpoint /path/to/your/checkpoint.pt --acknowledge-license-restrictions
```

**Structured error codes:** `MODEL_NOT_COMMERCIAL_SAFE`, `MODEL_LICENSE_RESTRICTED`,
`MODEL_ACKNOWLEDGEMENT_REQUIRED`, `MODEL_REQUIRES_BYO_CHECKPOINT`,
`MODEL_LICENSE_REVIEW_REQUIRED`, `MODEL_AGPL_RESTRICTED`, `MODEL_USE_MODE_NOT_ALLOWED`.

The same policy applies over the optional HTTP server: a restricted model returns a
clean **403** with the structured code (never a 500 or a fabricated result).

## Python helpers

```python
from visionservex.policy import (
    get_model_policy, list_commercial_safe_models, list_research_models,
    assert_commercial_safe, explain_model_license,
)

assert get_model_policy("medsam2").commercial_status == "research_only"
assert "medsam2" not in list_commercial_safe_models()
assert_commercial_safe("sam2.1-hiera-tiny")   # ok
print(explain_model_license("medsam2"))
```

## Not-for-diagnosis disclaimer

Medical models in VisionServeX are for **research and education only**. They make
**no diagnostic, treatment, or clinical claim**. Do not use any output for patient
care.

## What VisionServeX can claim

- "Commercial-safe by default; restricted models require an explicit, acknowledged
  research/BYO pathway."
- "License-aware: code AND weights are checked; AGPL/copyleft models are excluded
  from the commercial-safe set."
- "Permissive-license alternatives to AGPL-default model stacks."

## What VisionServeX must NOT claim

- ❌ "All VisionServeX models are commercial-safe."
- ❌ "MedSAM2 is commercial-safe."
- ❌ "Medical tumor segmentation is commercial-safe."
- ❌ "Diagnostic segmentation" / "Clinically validated."
- ❌ "Ultralytics-compatible without license obligations" (Ultralytics is AGPL-3.0).
