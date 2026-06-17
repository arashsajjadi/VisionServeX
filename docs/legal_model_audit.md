# VisionServeX Legal / Commercial-Safety Model Audit

_Per-model license posture from the curated policy table + registry._

## Summary

| Legal status | Models |
|---|--:|
| byot_license_required | 1 |
| commercial_safe_core | 55 |
| external_api_only_terms_required | 1 |
| legal_review_required | 4 |
| registry_license_only | 90 |

- **commercial-safe (curated policy default-safe):** 55
- **gated (HF token required):** 2
- **Ultralytics / AGPL / GPL / SSPL runtime:** none (test-enforced).
- **YOLO-NAS / Deci non-commercial training:** blocked.

## Gated models (BYOT — token server-side only, never logged/committed)

| Model ID | Family | License | Legal status |
|---|---|---|---|
| sam3-base | sam3 | SAM License (Meta custom, gated) | byot_license_required |
| grounding-dino-1.6 | grounding-dino | Custom | registry_license_only |

## Non-commercial / enterprise / review (NOT default-safe)

| Model ID | Family | License | Legal status |
|---|---|---|---|
| hq-sam | hq-sam | Apache-2.0 (code) | legal_review_required |
| medsam | sam | Apache-2.0 (code) | legal_review_required |
| oneformer-convnext-large | oneformer | MIT (code) | legal_review_required |
| oneformer-dinat-large | oneformer | MIT (code) | legal_review_required |
