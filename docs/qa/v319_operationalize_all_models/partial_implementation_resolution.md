# Partial-implementation resolution (v3.19)

The three v3.18 `PARTIAL_IMPLEMENTATION_BLOCKED` rows, resolved.

| Model | v3.18 state | v3.19 outcome | Evidence / reason |
|---|---|---|---|
| `maxvit-tiny-tf-224` | partial | **OPERATIONALIZED → `INFERENCE_READY_LIVE`** | transformers 5.x loads the `timm/maxvit_tiny_tf_224.in1k` repo via `TimmWrapperForImageClassification`; registry flipped `partial → wired`, `auto_download: true`; live top-5 classify smoke verified (`v319_inference_matrix.json`). Apache-2.0. |
| `rtmdet-r2-s` | partial | **stays blocked — OpenMMLab Docker sidecar required** | host-native mmcv infeasible (py3.13/cu130/setuptools); see `openmmlab_engine_plan.md`. Hidden from Anastig (`PARTIAL_IMPLEMENTATION_BLOCKED` → `hide`). |
| `rtmpose-s` | partial | **stays blocked — OpenMMLab Docker sidecar required** | same as above; cached `rtmpose-*` checkpoints are ready for the sidecar. |

**Acceptance check:** no partial model is shown to Anastig users — `maxvit` is now
genuinely live; `rtmdet-r2-s` / `rtmpose-s` remain `hide`. The two remaining
partials are honestly an OpenMMLab-sidecar dependency, not in-process work.
