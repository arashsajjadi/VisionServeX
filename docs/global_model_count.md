# Global Model Count (v3.10.0)

This document explains what the model counts in VisionServeX mean and how to
interpret them. Detailed counts are here so the README can avoid raw audit
numbers that mislead first-time readers.

---

## Summary

| Scope | Count | What it means |
|---|---|---|
| **Policy table rows** | 99 | License-classified model variants tracked in `licensing/policy.py` |
| **Manifest entries** | 170 | All known model IDs in the model zoo manifest |
| **Runnable in VisionServeX** | 95 | Can be loaded and run in the current local/CI environment |

---

## Why the counts differ

**Policy rows (99)** classify license-relevant model *variants* — each SAM2 size
is a separate row. They cover commercial-safe core, BYOT gated, external-API,
non-commercial, legal-review, enterprise, and not-yet-released models.

**Manifest entries (170)** include all known model IDs: runnable, blocked, sidecar,
external-API, aliases, and experimental stubs. Not all of these are usable locally.

**Runnable (95)** is the count that can actually be loaded in a standard local
environment with the listed extras. It excludes: sidecar-only models (OpenMMLab /
Detectron2), models with no public checkpoint URL, BYOT models until the user has
accepted the upstream license, and external-API models.

---

## Policy bucket breakdown (99 rows)

| Bucket | Count | Description |
|---|---|---|
| `commercial_safe_core` | 39 | Apache-2.0 / MIT; default-enabled; no token needed |
| `byot_license_required` | 28 | Gated on HF; custom upstream license; BYOT only |
| `legal_review_required` | 11 | License under review; disabled pending confirmation |
| `external_api_only` | 9 | Cannot self-host; API key required |
| `noncommercial_restricted` | 7 | Research-only license |
| `enterprise` | 4 | AGPL-3.0 or proprietary enterprise license |
| `not_released` | 1 | Registered but not yet publicly available |

---

## Versioning note

The v3.3.0 audit (January 2026) counted 173 model rows with 111 passing a broad
"functional/testable" criterion (64.16%). The v3.10.0 count of 99 policy rows and
170 manifest entries reflects a different methodology: deduplicated license-policy
classification + full manifest coverage including stubs and aliases. These are
**not directly comparable** — neither indicates project regression.

---

## How to regenerate

```bash
python -c "
from visionservex.licensing.policy import _ROWS
from collections import Counter
buckets = Counter(r.final_policy for r in _ROWS)
print('Total rows:', len(_ROWS))
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
"

python -c "
from visionservex.model_zoo.manifest import MODEL_ZOO
print('Manifest entries:', len(MODEL_ZOO))
"
```

The always-current CSV is at `notebook/99_final_report/reports/v38_license_policy_matrix.csv`.
