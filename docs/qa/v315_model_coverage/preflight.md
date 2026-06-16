# v3.15.0 Preflight

Phase-0 freeze for the v3.15.0 model-coverage + capability-truth sprint.

| Field | Value |
|---|---|
| branch | `main` |
| HEAD (start) | `16a159c` |
| tag at HEAD | `v3.14.0` |
| origin/main | `16a159c` (in sync) |
| current version | `3.14.0` |
| torch / torchvision | 2.11.0 / 0.26.0 (both installed) |

## Dirty files (pre-existing, unrelated — NOT touched)

- ` D notebook/RUN_ALL_EXECUTED_v260.ipynb` (deleted in working tree)
- `?? notebook/99_final_report/artifacts/v3{10,4,5,7}/` (untracked notebook artifacts)
- `?? scripts/v310_sam3_debug.py` (untracked debug script)

These are outside `src/`, `tests/`, `docs/`, `tools/`, `pyproject.toml`, `README*`,
`CHANGELOG*` and are excluded from all v3.15 commits.

## Legal grep (pre-existing findings)

- Runtime engines/core/data: **CLEAN** — no `import ultralytics` / `from ultralytics`.
- Pre-existing optional ultralytics *benchmark-comparison* path in
  `src/visionservex/cli/benchmark_commands.py` (gated behind "not installed → skip";
  not a declared dependency; shipped since v3.12.0). Out of scope; not touched.
- `AGPL`/`GPL` string matches are doc comments stating the no-AGPL stance.

## Decision

**SAFE TO PROCEED.** Working tree is clean of relevant work; torch+torchvision are
present (BSD-3, commercial-safe) enabling the P1 torchvision classifier family with
no new heavy dependencies.
