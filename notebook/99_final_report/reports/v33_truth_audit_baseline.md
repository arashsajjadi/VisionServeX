# VisionServeX v3.3 Truth Audit — Baseline (deterministic from canonical ledgers)

## Model zoo (173 rows in model_coverage_ledger.csv)

| category | count | pct |
|---|---|---|
| PASS | 111 | 64.16% |
| FAIL | 0 | 0.0% |
| BLOCKED | 44 | 25.43% |
| EXCLUDED | 16 | 9.25% |
| OTHER/UNVERIFIED | 2 | 1.16% |

- default-safe rows: 157 → default-safe pass = 70.7%
- evidence completeness (pass rows with on-disk artifact): 111/111 = 100.0%
- tutorial coverage: 10.4% · tutorial executed: 8.09%
- CLI coverage: 99.42% · Python API coverage: 100.0%
- bad-license default-safe rows: 0

## final_state distribution

- benchmark_passed: 109
- sidecar_required: 21
- expected_blocker: 18
- wired: 7
- dataset_required: 4
- checkpoint_required: 4
- external_api_only: 3
- auth_required: 3
- demo_passed_sidecar: 2
- micro_benchmark_passed: 1
- not_advertised: 1

## Pipelines: 4/4 pass = 100.0%
## Tools (cv2-pro + classic): 21/21 pass = 100.0%
## SAM comprehensive (v31, 68 targets): pass 39.71% / blocked 58.82% / excluded 1.47%
## DINO comprehensive (v31, 44 targets): pass 38.64% / blocked 59.09% / excluded 0.0%
## Sidecar: pass 2/23 = 8.7% / blocked 91.3%

## CLI help: root=True groups={'sam': True, 'dino': True, 'pipeline': True, 'cv2-pro': True, 'run': True, 'list-models': True}
## Python API import: True
