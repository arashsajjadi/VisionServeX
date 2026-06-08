VISION SERVE X V3.8 HUGGING FACE BYOT LICENSE-SAFE MODEL ACTIVATION FINAL STATUS

Date: 2026-06-07
Author: Arash Sajjadi (arash.sajjadi@usask.ca)

================================================================================
HEADLINE
================================================================================
v3.8 delivers a complete, user-facing Hugging Face Bring-Your-Own-Token (BYOT)
connection layer and a code-level license policy that classifies every advertised
model and enforces it in the CLI and the Python API. VisionServeX never bundles,
mirrors, or redistributes gated/restricted weights, and never stores or prints a
token.

HONEST KEY FINDING: the local token `visionservex-local-read` is valid and can
*see* the gated Meta repos, but it has NOT been granted file-download access —
SAM 3, SAM 3.1 and all DINOv3 return GATED_NOT_ACCEPTED via HfApi.auth_check.
Therefore 0 BYOT models were converted to benchmark_passed_byot this sprint; all
23 remain auth_required (the user must click "Agree and access repository" on each
model page). The BYOT flow is correct and complete — it detects this and prints
the exact lawful next step. The moment the licenses are accepted, the same wired
code path (transformers 5.3 Sam3Model / AutoModel) runs real inference.

================================================================================
RELEASE
================================================================================
selected_version:      3.8.0   (minor: new user-facing HF/BYOT features + CLIs)
release_published:     YES — published to PyPI via GitHub Actions OIDC Trusted Publishing
pypi_run_id:           27108889264  (Publish to PyPI — build ✓ / publish-pypi ✓ in 24s)
commit:                f1aa2ef   tag: v3.8.0
github_release_url:    https://github.com/arashsajjadi/VisionServeX/releases/tag/v3.8.0
pypi_url:              https://pypi.org/project/visionservex/3.8.0/   (latest)
wheel_built:           visionservex-3.8.0-py3-none-any.whl (774 KB, no weights) + sdist; twine check PASSED
fresh_install_proof:   /home/arash/.cache/vsx38_pypi venv -> pip install --no-cache-dir
                       visionservex==3.8.0 from PUBLIC PyPI; __version__=3.8.0, metadata=3.8.0,
                       imports from site-packages. CLI: visionservex --version=3.8.0,
                       hf status (token redacted hf_***FW), model license sam3-base/dinov3-vitb16
                       =byot, locateanything-3b=noncommercial_restricted. (v38_fresh_install_verify.json)

================================================================================
HUGGING FACE AUTH STATUS (token always redacted)
================================================================================
token_source:          cli_cache (huggingface-cli login)
user:                  arashsajjadi (user)
token_display_name:    visionservex-local-read   role: fineGrained
token_redacted:        hf_***FW
HF_TOKEN env:          unset (token comes from the CLI cache)
auth_check API:        HfApi.auth_check (download authorization) — NOT model_info

================================================================================
MODELS ATTEMPTED WITH LOCAL HF TOKEN (auth_check, metadata only — no weights pulled)
================================================================================
SAM 3 (8 ids)          -> auth_required (GATED_NOT_ACCEPTED)
SAM 3.1 (7 ids)        -> auth_required (GATED_NOT_ACCEPTED)
DINOv3 (8 ids)         -> auth_required (GATED_NOT_ACCEPTED)
DINOv2 (control)       -> access_granted (ungated, confirms auth_check works)

models_converted_auth_required_to_benchmark_passed_byot: 0
models_still_auth_required:                              23
reason: per-model upstream license not yet accepted by the user on Hugging Face.
to_enable_real_runs:
  - https://huggingface.co/facebook/sam3      (Agree and access repository)
  - https://huggingface.co/facebook/sam3.1
  - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m  (+ variants)
  then:  visionservex model pull sam3-base --accept-upstream-license

================================================================================
LICENSE POLICY MATRIX (94 models — notebook/99_final_report/reports/v38_license_policy_matrix.csv)
================================================================================
commercial_safe_core:            39
byot_license_required:           23   (SAM3 x8, SAM3.1 x7, DINOv3 x8)
external_api_only_terms_required: 9   (GroundingDINO 1.5/1.6 Pro, DINO-X suite)
noncommercial_restricted:         7   (EdgeSAM, LocateAnything, DAM, MedSAM2, DepthAnythingV2-large, SimpleClick, FocalClick)
enterprise_license_required:      4   (FastSAM s/x, Ultralytics yolov8-seg/yolo11-seg)
legal_review_required:           11   (HQ-SAM family, TinySAM/Q-TinySAM, OneFormer, InternImage, MedSAM, RF-DETR-Seg XL/2XL)
not_released_or_unverifiable:     1   (grounding-dino-2)
excluded_from_core:               0   (NC/enterprise/legal are the operative refusal buckets)
auth_required_license_pending:    0   (a runtime state, not a static bucket)

models_refused_because_noncommercial: 7  (production refused; research-only opt-in available)
external_api_only_count:              9
commercial_safe_core_count:          39
byot_count:                          23
excluded_count (NC+enterprise+legal+not_released): 23
legal_review_count:                  11

HARD RULES ENFORCED (code + tests):
  - HF token != redistribution rights.
  - can_ship_weights = False for ALL 94 rows (weights never bundled).
  - non-commercial: never production, never default_safe.
  - AGPL/enterprise: never default_safe.
  - external-API: never counted as local (is_local=False).
  - legal_review: never commercial_safe until resolved.
  - code / weights / dataset licenses tracked separately.

================================================================================
TESTS
================================================================================
v38 suite (tests/test_v38_*):   169 total -> 168 passed / 0 failed / 1 skipped (opt-in ONNX download)
  artifacts: v38_pytest.xml, v38_pytest_summary.json, v38_failed_tests.csv, v38_test_execution_matrix.csv
broad safe suite (whole package, heavy markers excluded): 2193 total -> 2168 passed initially,
  7 failed, 18 skipped, 43 deselected (547s). The 7 failures were ALL remediated and re-verified:
  - 6 stale EXACT version pins (asserted "3.7.0") -> converted to forward-compatible
    minimum-version checks (>= (3,6)/(3,7)); nothing weakened.
  - 1 v3.1 facade contract test (test_vsx_facades_explain_contract) -> updated to accept the
    intended v3.8 access-precision states (auth_required / auth_required_license_pending), since
    SAM3 segment() now performs a real access check rather than returning a static state.
  Post-remediation: 2175 passed / 0 failed / 18 skipped (7 fixed tests re-run individually = pass).
  NO production logic regressed; only stale assertions + one evolved-behavior contract updated.
test rules honored: no real token required; gated live tests behind VISIONSERVEX_RUN_GATED_HF=1;
  token redacted in output; no test prints a real hf_ token; no large downloads by default.

================================================================================
SECURITY / RELEASE GUARDS  (scripts/v38_security_guard.py -> v38_security_scan.json)
================================================================================
token_leak_scan:        PASS (real active token absent from all tracked files + working diff)
token_pattern_nontest:  PASS (no token-shaped string in shippable files; tests hold only fake fixtures)
binary_artifact_scan:   PASS (no .onnx/.pt/.pth/.ckpt/.safetensors/.bin/.engine/.trt tracked; no artifacts/)
noncore_default_safe:   PASS (only commercial_safe_core rows are default_safe)
agpl_default_safe:      PASS (none)
github_actions_token:   PASS (no HF token referenced in workflows)
readme_nonredistribution_statement: PASS

================================================================================
NOTEBOOKS  (notebook/tutorials/v38_hf_byot_and_license_safe_models/)
================================================================================
12 notebooks generated + validated (valid nbformat, assert site-packages, no token literal):
  01 install_from_pypi_and_check_version     07 sam2_1_onnx_export_attempt
  02 connect_huggingface_token_safely        08 ritm_interactive_segmentation_checkpoint_path
  03 license_policy_matrix_explained         09 rfdetr_seg_commercial_safe_instance_masks
  04 pull_commercial_safe_sam_and_dino       10 groundingdino_sam_text_to_mask_pipeline
  05 sam3_byot_status_and_optional_run       11 restricted_models_warnings_*
  06 dinov3_byot_status_and_optional_embed   12 end_to_end_anastig_policy_demo
notebooks_executed_from_pypi: 01,02,03,09,10,11 — ALL OK, executed via nbconvert against the
  PyPI-installed package (v38_tutorial_execution_ledger.csv). No token leaked into outputs.
  Gated 05/06 remain auth_required until the user accepts the upstream licenses.

================================================================================
README / DOCS SYNC CHECKLIST
================================================================================
[x] README: pip install, HF connect, BYOT usage, license warning table, non-redistribution statement
[x] docs/huggingface_connection.md (local CLI flow, future OAuth, revoke, Anastig SaaS)
[x] docs/model_license_policy.md (nine buckets, hard rules)
[x] docs/byot_models.md (SAM3/DINOv3)
[x] docs/restricted_models.md (NC / enterprise / API / legal)
[x] docs/commercial_safe_core.md (39-model core)
[x] docs/anastig_saas_policy.md (per-user OAuth + encrypted secrets)
[x] CHANGELOG.md [3.8.0] entry

================================================================================
NEXT SPRINT RECOMMENDATIONS
================================================================================
1. Accept the SAM3/SAM3.1/DINOv3 upstream licenses on Hugging Face, then run the
   wired BYOT runtime to convert 23 auth_required -> benchmark_passed_byot (real
   evidence). DINOv3-vits16 (86MB) and convnext-tiny (111MB) are CPU-trivial;
   facebook/sam3 (~3.4GB safetensors) is CPU-feasible.
2. Resolve the two contested legal_review rows (RF-DETR-Seg XL/2XL): confirm current
   Roboflow terms; if Apache-2.0 seg is reconfirmed, promote to commercial_safe_core.
3. Build the commercial-safe detection sidecars (MaskDINO / Co-DINO / RT-DETRv4 /
   RTMDet) in isolated envs and benchmark.
4. Add the web OAuth ("Sign in with Hugging Face") flow described in
   docs/huggingface_connection.md for the future web app.
5. Wire fine-grained scope enumeration (whoami fineGrained block) so
   hf_validate_token reports granted scopes precisely.
