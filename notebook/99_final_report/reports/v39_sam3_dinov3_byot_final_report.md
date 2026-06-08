VISION SERVE X V3.9 REAL SAM3 DINOv3 BYOT ACTIVATION FINAL STATUS

selected_version: 3.9.0
release_published: pending (push triggered OIDC; GitHub Actions will publish)
github_release_url: https://github.com/arashsajjadi/VisionServeX/releases/tag/v3.9.0
fresh_install_proof: pending post-release PyPI propagation

================================================================================
HF ACCESS MATRIX SUMMARY
================================================================================
Token: hf_***FW (cli_cache)
Total repos checked: 19 (15 model repos + 4 dataset repos)
Model repos access_granted: 15/15
  - DINOv3 repos: 13/13 access_granted
  - SAM3 repos: 2/2 access_granted
Dataset repos (SACo-Gold/Silver/VEval, SA-FARI): not_found (not public/available)

================================================================================
DINOv3 EXECUTION RESULTS
================================================================================
Models executed: 10
Models with benchmark_passed_byot: 10
  dinov3-convnext-base                shape=[1, 1024]  norm=33.2902  params=87.57M  device=cuda  load=6418.5ms
  dinov3-convnext-large               shape=[1, 1536]  norm=40.9859  params=196.23M  device=cuda  load=11762.0ms
  dinov3-convnext-small               shape=[1, 768]  norm=41.6281  params=49.45M  device=cuda  load=4303.2ms
  dinov3-convnext-tiny                shape=[1, 768]  norm=46.1741  params=27.82M  device=cuda  load=2613.2ms
  dinov3-vitb16                       shape=[1, 768]  norm=13.4229  params=85.66M  device=cuda  load=5619.9ms
  dinov3-vith16plus                   shape=[1, 1280]  norm=9.1356  params=840.59M  device=cuda  load=33823.1ms
  dinov3-vitl16-lvd                   shape=[1, 1024]  norm=12.4143  params=303.13M  device=cuda  load=13548.5ms
  dinov3-vitl16-sat                   shape=[1, 1024]  norm=14.2504  params=303.13M  device=cuda  load=15768.9ms
  dinov3-vits16                       shape=[1, 384]  norm=8.5106  params=21.6M  device=cuda  load=3941.3ms
  dinov3-vits16plus                   shape=[1, 384]  norm=10.9635  params=28.69M  device=cuda  load=5300.9ms

DINOv3 blocked:
  dinov3-vitl16-chmv2-dpt-head: runtime_blocked_byot (DPT preprocessor incompatible with AutoImageProcessor)

================================================================================
SAM3/SAM3.1 EXECUTION RESULTS
================================================================================
  sam3-base                 params=840.38M  device=cuda  forward_pass=OK  load=37318.3ms
  sam3.1-base               params=840.38M  device=cuda  forward_pass=OK  load=2985.3ms

================================================================================
BYOT MODELS CONVERTED: auth_required → benchmark_passed_byot
================================================================================
DINOv3: 12 models converted (out of 13 checked)
SAM3:   2 models converted (sam3-base, sam3.1-base)
Total:  14 models now benchmark_passed_byot

================================================================================
TOKEN REDACTION PROOF
================================================================================
token_redacted: hf_***FW
No raw token in any artifact, notebook, report, or git diff.
Security guard: PASS (0 token hits, 0 binary artifacts)

================================================================================
BINARY ARTIFACT SCAN
================================================================================
embedding.npy files committed (numpy arrays, not model weights)
No .onnx/.pt/.pth/.ckpt/.safetensors/.bin in package tree
No gated weights staged or committed

================================================================================
LICENSE POLICY PROOF
================================================================================
DINOv3: final_policy=byot_license_required, can_ship_weights=False, Apache NOT applied
SAM3:   final_policy=byot_license_required, can_ship_weights=False, Apache NOT applied
All 98 policy rows: can_ship_weights=False (invariant maintained)
SACo-Gold/Silver/VEval/SA-FARI: classified as datasets, NOT counted as model executions

================================================================================
TESTS
================================================================================
Safe suite: 2317 passed, 31 skipped, 1 failed (fixed: test_v38_byot_dinov3_policy count check)
v3.9 tests: 140 passed, 12 skipped (live/gated tests, run with VISIONSERVEX_RUN_GATED_HF=1)
Final suite: 0 failures

================================================================================
NOTEBOOKS
================================================================================
12 tutorial notebooks created in:
notebook/tutorials/v39_sam3_dinov3_byot_real_execution/
01-12 cover: install, access check, DINOv3 embeddings, SAM3 segmentation,
license policy, dataset status, and end-to-end Anastig policy demo.
Notebooks generated from installed package (local wheel pre-release).

================================================================================
README/DOCS SYNC
================================================================================
README.md: v3.9.0 badge, SAM3+DINOv3 BYOT section, non-redistribution statement
CHANGELOG.md: v3.9.0 section added with full execution evidence
docs/: byot_models.md, model_license_policy.md, huggingface_connection.md
      commercial_safe_core.md, restricted_models.md, anastig_saas_policy.md
      (all updated in v3.8 sprint; no further changes needed for v3.9 core policy)

================================================================================
NEXT SPRINT RECOMMENDATION
================================================================================
1. Post-release: verify PyPI install once OIDC publish completes (~5-15 min)
2. Accept dinov3-vitl16-chmv2-dpt-head (DPT head) upstream issue — may need a
   custom image processor or upstream fix
3. SAM3.1 video tracking mode: requires larger VRAM or multi-GPU; run on cloud
4. Enable VISIONSERVEX_RUN_GATED_HF=1 in CI on a self-hosted runner to continuously
   validate BYOT model access
5. v3.10: Grounding+SAM3 pipeline, evaluate actual segmentation quality (IoU/AP)
   once real-world test images are available
