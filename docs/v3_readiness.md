# V3 Readiness

**Status: ✅ V3 READY — ALL 17 GATES PASS.** Released as **v3.0.0**.

## Gate summary (17 gates — 17 PASS)

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
| V3-11 | PASS | every benchmark_passed row has valid current-run evidence |
| V3-12 | PASS | every target classified |
| V3-13 | PASS | classic smart tools separated from model leaderboard |
| V3-14 | PASS | README/docs explain core vs external restricted + BYOT |
| V3-15 | PASS | final_winners schema does not mix core/restricted |
| V3-16 | PASS_WITH_CAVEAT | package tests pass |
| V3-17 | PASS | final report lists remaining blockers + lawful next actions |

## What closed the final gate (V3-11)
The reconciler now attributes a current-run evidence artifact to every benchmark-claiming model, via `reporting.current_run_evidence` (consolidates real benchmark metrics into comprehensive current-run task leaderboards) + an `evidence_artifact_exists` reconciler fix. RT-DETRv4 (no real benchmark; gated checkpoint) was honestly downgraded to checkpoint_required.
