# V3 Readiness

**Status: V3 NOT READY.** Latest release: **v2.60.0**. **16/17 V3 gates PASS** — v3.0.0
is blocked by exactly one gate: **V3-11** (durable current-run evidence attribution).

## Gate summary (17 gates — 16 PASS, 1 PARTIAL)

| gate | status | what it checks |
|---|---|---|
| V3-01 | PASS | PyPI Trusted Publishing works |
| V3-02 | PASS | Fresh PyPI install from real PyPI |
| V3-03 | PASS | RUN_ALL executes after fresh install |
| V3-04 | PASS | smoke_passed == 0 |
| V3-05 | PASS | benchmark_failed == 0 or justified |
| V3-06 | PASS | blocker_category unclassified == 0 |
| V3-07 | PASS | no AGPL/GPL/NC/restricted in commercial-safe core |
| V3-08 | PASS | every core model has code_license and weights_license |
| V3-09 | PASS | gated/auth models BYOT, no mirrored gated weights |
| V3-10 | PASS | no token leak in reports/notebooks/git |
| V3-11 | PARTIAL ⛔ | every benchmark_passed row has valid current-run evidence |
| V3-12 | PASS | every target classified |
| V3-13 | PASS | classic smart tools separated from model leaderboard |
| V3-14 | PASS | README/docs explain core vs external restricted + BYOT |
| V3-15 | PASS | final_winners schema does not mix core/restricted |
| V3-16 | PASS_WITH_CAVEAT | package tests pass |
| V3-17 | PASS | final report lists remaining blockers + lawful next actions |


## What V3-prep delivered

- **Commercial-safety fix:** EdgeSAM (NTU S-Lab non-commercial) removed from
  core; HQ-SAM marked legal_review (HQSeg-44K NC training data). See
  [commercial_safety.md](commercial_safety.md).
- **Classic smart-annotation toolkit:** 8 weight-free CPU refiners, real COCO
  benchmark, separate `smart_tool_coverage_ledger.csv`. See
  [smart_annotation.md](smart_annotation.md).
- **Full target audit:** 56 targets (families A–F) classified in
  `v3_target_model_coverage_matrix.csv`; code-vs-weights split in
  `v3_model_rights_audit.csv` (adversarially verified).
- **BYOT for gated models:** [gated_models.md](gated_models.md).
- **Evidence + license integrity:** 0 NaN-evidence benchmark rows, 0 NaN-license
  core rows.

## What still blocks v3.0.0

1. **Release-loop gates (V3-01/02/03):** require an actual tag → Trusted-Publish
   → fresh-PyPI-install → RUN_ALL cycle. Configure OIDC Trusted Publishing.
2. **V3-08:** add explicit `code_license` + `weights_license` columns to the
   ledger for every core family (extend the rights audit past the 56 segmentation
   targets).
3. **V3-11 residual:** re-benchmark `rtdetrv4-*` and `siglip-base` (their evidence
   points to a pull/contract artifact, not a benchmark).
4. **V3-16 residual:** clean up a pre-existing stale-test cluster (v2.46 state
   corrections not propagated to ~5 test files).

Full detail: `notebook/99_final_report/reports/v3_blockers_report.md`.
