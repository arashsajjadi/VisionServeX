# Model license policy

VisionServeX classifies every advertised model into one of **nine** policy
buckets. The classification is code (`visionservex.licensing.policy`) and is the
single source of truth for the CLI, the Python API, the tests, the notebooks, and
the generated matrix at
`notebook/99_final_report/reports/v38_license_policy_matrix.csv`.

## The nine buckets

| final_policy | meaning | default_safe | commercial_safe | production |
|---|---|:---:|:---:|:---:|
| `commercial_safe_core` | Permissive (Apache-2.0/MIT) weights pulled from official source | ✅ | ✅ | ✅ |
| `byot_license_required` | Gated / custom license — bring your own token, accept upstream license | ❌ | depends on accepted terms | ❌ by default |
| `auth_required_license_pending` | Gated, token present, upstream license not yet accepted (runtime state) | ❌ | ❌ | ❌ |
| `external_api_only_terms_required` | Hosted API, no local weights, data leaves the box | ❌ | ❌ | ❌ |
| `noncommercial_restricted` | Non-commercial / research-only | ❌ | ❌ | ❌ |
| `enterprise_license_required` | AGPL / copyleft / enterprise-only | ❌ | ❌ | ❌ |
| `legal_review_required` | License / provenance unclear | ❌ | ❌ | ❌ |
| `excluded_from_core` | Removed from core for policy reasons | ❌ | ❌ | ❌ |
| `not_released_or_unverifiable` | Not released / no official source | ❌ | ❌ | ❌ |

## Three licenses, tracked separately

For every model we record **code license**, **weights license**, and
**dataset / pretraining risk** independently. A permissive code license does not
make non-commercial weights commercial-safe, and a permissive weights license can
still carry dataset-provenance risk (tracked as `dataset_risk`).

## Hard rules (enforced in code + tests)

1. A Hugging Face token does **not** grant redistribution rights.
2. Gated models are **never** packaged into PyPI / GitHub / Docker — `can_ship_weights`
   is `False` for **every** row (VisionServeX never bundles any weights).
3. Non-commercial models are **never** `production_allowed` and **never** `default_safe`.
4. AGPL / enterprise models **never** enter the `default_safe` core.
5. API-only models are **never** counted as local models (`is_local=False`).
6. `legal_review` models are **never** `commercial_safe` until resolved.

## Inspect the policy

```bash
visionservex model license <model_id>          # full policy row for one model
visionservex model license <model_id> --json
python scripts/v38_generate_license_matrix.py  # regenerate the full matrix + report
```

See [commercial_safe_core.md](commercial_safe_core.md),
[byot_models.md](byot_models.md), and [restricted_models.md](restricted_models.md)
for the per-bucket details, and [huggingface_connection.md](huggingface_connection.md)
for the BYOT connection flow.
