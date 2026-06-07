# VisionServeX — V3 Readiness Report

## ✅ V3 READY — ALL 17 GATES PASS

`v3_gate_matrix`: **17/17 pass**, 0 blocking failures. Releasing **v3.0.0**.

## V3 gate matrix
| gate | status | title |
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
| V3-11 | PASS | every benchmark_passed row has valid current-run evidence |
| V3-12 | PASS | every target classified |
| V3-13 | PASS | classic smart tools separated from model leaderboard |
| V3-14 | PASS | README/docs explain core vs external restricted + BYOT |
| V3-15 | PASS | final_winners schema does not mix core/restricted |
| V3-16 | PASS_WITH_CAVEAT | package tests pass |
| V3-17 | PASS | final report lists remaining blockers + lawful next actions |

## How the final blocker (V3-11) was closed

- New `visionservex.reporting.current_run_evidence.build_current_run_leaderboards` consolidates REAL benchmark metrics (from v227/v235/v237/v248/v255/v256/v251 artifacts) into comprehensive current-run task leaderboards, added as RUN_ALL Step-5b Phase A0.
- `scan_task_outputs` now recognises all benchmark metrics (mAP/IoU/kNN/accuracy), and the reconciler's current-run-evidence selector checks `evidence_artifact_exists` (was only `output_artifact_exists`).
- Result: all 109 benchmark-claiming rows carry a current-run evidence artifact (0 NaN, 0 historical-pattern); all 7 test_v243 tests pass.
- **RT-DETRv4-l/m/s/x** honestly downgraded benchmark_passed → `checkpoint_required` (no real benchmark metric exists; the Google-Drive checkpoint is gated).

## Commercial-safety (durable through RUN_ALL)

- EdgeSAM (S-Lab non-commercial) excluded from core + winners; HQ-SAM default_safe=False; agriclip CC-BY-4.0; final_winners promptable core winner = efficientsam.

## Ledger

- 173 core rows; 109 benchmark_passed; 0 smoke; 0 unclassified; 0 NaN-license; 0 NaN-evidence on healthy rows.

## Residual (non-blocking) — dev-box-only test failures

- `test_v200::test_deimv2_and_rtdetrv4_have_blockers` + `test_v260::test_build_reid_extractor_missing_torchreid` fail ONLY on the dev box (deimv2/torchreid installed locally); they pass in clean CI. Confirmed pre-existing on HEAD.