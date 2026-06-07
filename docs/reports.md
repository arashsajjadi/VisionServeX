# Reports & Truth Ledgers

VisionServeX ships its real, measured status as machine-readable ledgers under
`notebook/99_final_report/reports/`. Nothing here is hand-typed — every number is computed
from the canonical ledgers by `scripts/v33_truth_audit.py` and verified on disk.

## Canonical ledgers

| file | what it holds |
|---|---|
| `model_coverage_ledger.csv` | every core model (173 rows): `final_state`, `default_safe`, `license_status`, `evidence_artifact`, blocker codes |
| `v3_core_model_rights.csv` | per-model `code_license` + `weights_license` (commercial-safety audit) |
| `pipeline_coverage_ledger.csv` | composed text→mask pipelines |
| `smart_tool_coverage_ledger.csv` / `v31_cv2_pro_tool_ledger.csv` | classic CPU tools + CV2-Pro tools |
| `v31_sam_family_matrix.csv` / `v31_dino_family_matrix.csv` | full SAM / DINO target matrices (incl. gated/unreleased) |

## v3.3 truth-audit artifacts (regenerate any time)

```bash
python3 scripts/v33_truth_audit.py        # pass/fail/blocked/excluded matrices + percentages
python3 scripts/v33_evidence_integrity.py # every PASS row has a real, on-disk, task-appropriate metric
python3 scripts/v33_parse_pytest.py       # pytest junit XML -> summary + failed-tests CSV
```

Outputs: `v33_truth_audit_baseline.{md,json}`, `v33_model_pass_fail_matrix.csv`,
`v33_model_family_percentages.csv`, `v33_task_percentages.csv`,
`v33_pipeline_pass_fail_matrix.csv`, `v33_tool_pass_fail_matrix.csv`,
`v33_evidence_integrity_report.{csv,md}`, `v33_test_execution_matrix.csv`,
`v33_release_readiness_matrix.csv`.

## Truth taxonomy

- **PASS** — `benchmark_passed`, `micro_benchmark_passed`, `demo_passed_sidecar`,
  `tool_benchmark_passed`, or `wired` *with* an evidence artifact.
- **BLOCKED** — `auth_required`, `checkpoint_required`, `dataset_required`,
  `sidecar_required`, `external_api_only`, `expected_blocker`, `not_released`,
  `legal_review_required` (gated/heavy/legal — honest, not failures).
- **EXCLUDED** — `default_safe=False` or a restricted license (AGPL/GPL/NC/proprietary/S-Lab/PML).
  Quarantined to external baselines, never shipped as default-safe.
- **FAIL** — runtime/import/test/cli/api failure. (Core model rows: **0**.)

## v3.3 measured state (model_coverage_ledger, 173 rows)

| category | count | pct |
|---|---|---|
| PASS | 111 | 64.16% |
| BLOCKED | 44 | 25.43% |
| EXCLUDED | 16 | 9.25% |
| OTHER/UNVERIFIED (`wired` no-evidence + `not_advertised`) | 2 | 1.16% |
| FAIL | 0 | 0% |

- default-safe pass: **70.7%** (111/157) · evidence completeness: **100%** (every PASS row
  has a real on-disk, task-appropriate, non-NaN metric).
- Pipelines: **100%** (4/4) · Tools (CV2-Pro + classic): **100%** (21/21).
- SAM full matrix (68 targets): 39.71% pass / 58.82% blocked / 1.47% excluded.
- DINO full matrix (44 targets): 38.64% pass / 59.09% blocked.
- Sidecar models: 8.7% pass / 91.3% blocked (honest — OpenMMLab/Detectron2/NATTEN builds gated).
- Bad-license default-safe rows: **0** · token leaks: **0**.
